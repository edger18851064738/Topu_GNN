import math
import random
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass
from road_network import RoadNetwork


class VehicleState(Enum):
    IDLE = 'idle'
    PLANNING = 'planning'
    WAITING = 'waiting'
    CONFIRMED = 'confirmed'
    MOVING = 'moving'
    LOADING = 'loading'
    UNLOADING = 'unloading'
    BLOCKED = 'blocked'


class VehicleMode(Enum):
    PARKED = 'parked'
    EMPTY = 'empty'
    LOADED = 'loaded'
    RETURNING = 'returning'


@dataclass
class VehicleStats:
    """车辆统计信息"""
    completed_cycles: int = 0
    total_distance: float = 0.0
    wait_time: float = 0.0
    total_time: float = 0.0


class Vehicle:
    """车辆类，管理单个车辆的状态和行为"""
    
    def __init__(self, vehicle_id: int, start_parking_point: str, 
                 road_network: RoadNetwork, use_gnn: bool = True):
        self.id = vehicle_id
        self.road_network = road_network
        self.use_gnn = use_gnn
        
        # 车辆模式和目标
        self.mode = VehicleMode.PARKED
        self.current_parking_point = start_parking_point
        self.target_loading_point: Optional[str] = None
        self.target_unloading_point: Optional[str] = None
        self.target_parking_point: Optional[str] = None
        
        # 位置和移动
        start_node_id = self.road_network.get_point_node_id(start_parking_point)
        self.current_node = start_node_id or 'fallback_node'
        
        node_position = self.road_network.node_positions.get(self.current_node, [0, 0])
        self.position = node_position.copy()
        self.target_position = self.position.copy()
        self.previous_position = self.position.copy()
        
        # 路径规划
        self.path: List[str] = []
        self.path_times: List[float] = []
        self.path_index = 0
        self.path_confirmed = False
        self.path_start_time = 0.0
        
        # 移动参数
        self.progress = 0.0
        self.speed = 0.6 + random.random() * 0.4
        self.move_start_time = 0.0
        self.move_duration = 0.0
        self.heading = 0.0  # 车辆朝向（弧度）
        
        # 状态管理
        self.state = VehicleState.IDLE
        self.wait_until = 0.0
        self.retry_count = 0
        self.max_retries = 3
        self.operation_start_time = 0.0
        
        # 操作时间
        self.loading_time = 2.0
        self.unloading_time = 1.5
        
        # 统计信息
        self.stats = VehicleStats()
        
        # 可视化
        self.color = self._get_vehicle_color()
        
        # 在网络中注册
        self.road_network.add_vehicle_to_node(self.id, self.current_node)
        self.road_network.occupy_special_point(start_parking_point, self.id)
    
    def _get_vehicle_color(self) -> str:
        """获取车辆颜色"""
        colors = ['#ff4444', '#4444ff', '#44ff44', '#ffaa00', '#aa44ff', '#44aaff']
        return colors[self.id % len(colors)]
    
    @property
    def target_point_id(self) -> Optional[str]:
        """当前目标特殊点ID"""
        if self.mode in [VehicleMode.PARKED, VehicleMode.EMPTY]:
            return self.target_loading_point
        elif self.mode == VehicleMode.LOADED:
            return self.target_unloading_point
        elif self.mode == VehicleMode.RETURNING:
            return self.target_parking_point
        return None
    
    @property
    def target_node_id(self) -> Optional[str]:
        """当前目标节点ID"""
        target_point = self.target_point_id
        return self.road_network.get_point_node_id(target_point) if target_point else None
    
    def update(self, current_time: float, dt: float):
        """更新车辆状态"""
        # 更新统计
        self.stats.total_time += dt
        
        # 更新朝向
        self._update_heading()
        
        # 状态机
        if self.state == VehicleState.IDLE:
            self._handle_idle_state(current_time)
        elif self.state == VehicleState.PLANNING:
            self._handle_planning_state(current_time)
        elif self.state == VehicleState.WAITING:
            self._handle_waiting_state(current_time, dt)
        elif self.state == VehicleState.CONFIRMED:
            self._handle_confirmed_state(current_time)
        elif self.state == VehicleState.MOVING:
            self._handle_moving_state(current_time, dt)
        elif self.state == VehicleState.LOADING:
            self._handle_loading_state(current_time)
        elif self.state == VehicleState.UNLOADING:
            self._handle_unloading_state(current_time)
        elif self.state == VehicleState.BLOCKED:
            self._handle_blocked_state()
    
    def _update_heading(self):
        """更新车辆朝向"""
        if self.state != VehicleState.MOVING and self.target_node_id:
            target_pos = self.road_network.node_positions.get(self.target_node_id)
            current_pos = self.road_network.node_positions.get(self.current_node)
            if target_pos and current_pos:
                dx = target_pos[0] - current_pos[0]
                dy = target_pos[1] - current_pos[1]
                if dx != 0 or dy != 0:
                    self.heading = math.atan2(dy, dx)
    
    def _handle_idle_state(self, current_time: float):
        """处理空闲状态"""
        if self.mode == VehicleMode.PARKED:
            # 寻找可用的装载点
            available_loading = self.road_network.get_available_point('loading', self.id)
            if available_loading and self.road_network.reserve_special_point(available_loading, self.id):
                # 释放停车点
                if self.current_parking_point:
                    self.road_network.release_special_point(self.current_parking_point)
                    self.current_parking_point = None
                self.target_loading_point = available_loading
                self.mode = VehicleMode.EMPTY
                self._plan_path_to_target(current_time)
            else:
                self._wait_and_retry(current_time)
        
        elif self.mode == VehicleMode.EMPTY:
            if self.target_loading_point:
                self._plan_path_to_target(current_time)
            else:
                self.mode = VehicleMode.PARKED
                self.state = VehicleState.IDLE
        
        elif self.mode == VehicleMode.LOADED:
            # 寻找可用的卸载点
            available_unloading = self.road_network.get_available_point('unloading', self.id)
            if available_unloading and self.road_network.reserve_special_point(available_unloading, self.id):
                self.target_unloading_point = available_unloading
                self._plan_path_to_target(current_time)
            else:
                self._wait_and_retry(current_time)
        
        elif self.mode == VehicleMode.RETURNING:
            # 寻找可用的停车点
            available_parking = self.road_network.get_available_point('parking', self.id)
            if available_parking and self.road_network.reserve_special_point(available_parking, self.id):
                self.target_parking_point = available_parking
                self._plan_path_to_target(current_time)
            else:
                self._wait_and_retry(current_time)
    
    def _handle_planning_state(self, current_time: float):
        """处理路径规划状态"""
        if not self.path or len(self.path) < 2:
            self._wait_and_retry(current_time)
            return
        
        # 验证并预订路径
        success = False
        if self.use_gnn:
            success = self._validate_and_reserve_path(current_time)
        else:
            success = self._validate_simple_path(current_time)
        
        if success:
            self.path_confirmed = True
            self.path_index = 0
            
            # 设置路径开始时间
            self.path_start_time = max(
                self.path_times[0] if self.path_times else current_time + 0.5,
                current_time + 0.5
            )
            
            self.state = VehicleState.CONFIRMED
        else:
            if self.use_gnn:
                self.road_network.cancel_reservations(self.id)
            self._wait_and_retry(current_time)
    
    def _handle_waiting_state(self, current_time: float, dt: float):
        """处理等待状态"""
        if current_time >= self.wait_until:
            self.state = VehicleState.IDLE
            self.retry_count += 1
            if self.retry_count > self.max_retries:
                self._reset_current_task()
                self.retry_count = 0
        else:
            self.stats.wait_time += dt
    
    def _handle_confirmed_state(self, current_time: float):
        """处理已确认路径状态"""
        if current_time >= self.path_start_time:
            self._start_confirmed_path(current_time)
    
    def _handle_moving_state(self, current_time: float, dt: float):
        """处理移动状态"""
        if current_time < self.move_start_time:
            return
        
        elapsed = current_time - self.move_start_time
        self.progress = min(elapsed / self.move_duration, 1.0)
        
        if self.progress > 0:
            start_pos = self.road_network.node_positions.get(self.path[self.path_index])
            smooth_progress = self._smooth_step(self.progress)
            
            self.previous_position = self.position.copy()
            
            self.position[0] = start_pos[0] + (self.target_position[0] - start_pos[0]) * smooth_progress
            self.position[1] = start_pos[1] + (self.target_position[1] - start_pos[1]) * smooth_progress
            
            # 计算朝向
            dx = self.target_position[0] - start_pos[0]
            dy = self.target_position[1] - start_pos[1]
            if dx != 0 or dy != 0:
                self.heading = math.atan2(dy, dx)
            
            # 更新距离统计
            if dt > 0:
                distance = math.sqrt(dx * dx + dy * dy)
                self.stats.total_distance += distance * (self.progress / (elapsed / dt)) * dt * 0.01
        
        if self.progress >= 1.0:
            self._arrive_at_node(current_time)
    
    def _handle_loading_state(self, current_time: float):
        """处理装载状态"""
        if current_time - self.operation_start_time >= self.loading_time:
            self.road_network.release_special_point(self.target_loading_point)
            self.mode = VehicleMode.LOADED
            self.target_loading_point = None
            self.state = VehicleState.IDLE
            self.path_confirmed = False
    
    def _handle_unloading_state(self, current_time: float):
        """处理卸载状态"""
        if current_time - self.operation_start_time >= self.unloading_time:
            self.road_network.release_special_point(self.target_unloading_point)
            self.mode = VehicleMode.RETURNING
            self.target_unloading_point = None
            self.state = VehicleState.IDLE
            self.path_confirmed = False
    
    def _handle_blocked_state(self):
        """处理阻塞状态"""
        self.road_network.cancel_reservations(self.id)
        self.state = VehicleState.IDLE
        self.retry_count += 1
    
    def _plan_path_to_target(self, current_time: float):
        """规划到目标的路径"""
        target_node = self.target_node_id
        if not target_node:
            self._wait_and_retry(current_time)
            return
        
        self.state = VehicleState.PLANNING
        self.road_network.cancel_reservations(self.id)
        
        if self.use_gnn:
            # 使用GNN路径查找
            result = self.road_network.gnn_pathfinding_with_reservation(
                self.current_node, target_node, self.id, current_time
            )
            self.path = result['path']
            self.path_times = result['times']
        else:
            # 简单路径查找
            self.path = self.road_network.simple_pathfinding(self.current_node, target_node)
            if self.path:
                self._calculate_simple_path_times(current_time)
            else:
                self.path_times = []
    
    def _calculate_simple_path_times(self, current_time: float):
        """计算简单路径的时间"""
        self.path_times = []
        current_t = current_time + 0.5
        
        for i in range(len(self.path)):
            if i == 0:
                self.path_times.append(current_t)
            else:
                travel_time = self.road_network._compute_travel_time(
                    self.path[i-1], self.path[i], self.id
                )
                current_t += travel_time
                self.path_times.append(current_t)
    
    def _validate_and_reserve_path(self, current_time: float) -> bool:
        """验证并预订GNN路径"""
        if not self.path_times or len(self.path_times) != len(self.path):
            return False
        
        # 调整时间确保从当前时间开始
        base_time = max(current_time + 0.5, self.path_times[0])
        adjusted_times = []
        
        for i, path_time in enumerate(self.path_times):
            if i == 0:
                adjusted_times.append(base_time)
            else:
                interval = self.path_times[i] - self.path_times[i-1]
                adjusted_times.append(adjusted_times[i-1] + interval)
        
        self.path_times = adjusted_times
        
        node_duration = 0.4
        
        # 验证所有节点可用性
        for i, node in enumerate(self.path):
            node_start_time = self.path_times[i]
            duration = node_duration * 3 if i == len(self.path) - 1 else node_duration
            
            if not self.road_network._is_node_available(node, node_start_time, duration, self.id):
                return False
        
        # 验证所有边可用性
        for i in range(len(self.path) - 1):
            from_node = self.path[i]
            to_node = self.path[i + 1]
            edge_start_time = self.path_times[i] + node_duration
            edge_duration = self.path_times[i + 1] - edge_start_time
            
            if edge_duration <= 0:
                return False
            
            if not self.road_network._is_edge_available(from_node, to_node, 
                                                       edge_start_time, edge_duration, self.id):
                return False
        
        # 预订所有节点
        for i, node in enumerate(self.path):
            node_start_time = self.path_times[i]
            duration = node_duration * 3 if i == len(self.path) - 1 else node_duration
            
            if not self.road_network.reserve_node(node, self.id, node_start_time, duration):
                self.road_network.cancel_reservations(self.id)
                return False
        
        # 预订所有边
        for i in range(len(self.path) - 1):
            from_node = self.path[i]
            to_node = self.path[i + 1]
            edge_start_time = self.path_times[i] + node_duration
            edge_duration = self.path_times[i + 1] - edge_start_time
            
            if not self.road_network.reserve_edge(from_node, to_node, self.id, 
                                                  edge_start_time, edge_duration):
                self.road_network.cancel_reservations(self.id)
                return False
        
        return True
    
    def _validate_simple_path(self, current_time: float) -> bool:
        """验证简单路径"""
        if not self.path_times or len(self.path_times) != len(self.path):
            return False
        
        # 简单模式下只检查基本冲突
        base_time = current_time + 0.5
        
        # 重新计算时间
        adjusted_times = []
        current_t = base_time
        
        for i in range(len(self.path)):
            if i == 0:
                adjusted_times.append(current_t)
            else:
                travel_time = self.road_network._compute_travel_time(
                    self.path[i-1], self.path[i], self.id
                )
                current_t += travel_time
                adjusted_times.append(current_t)
        
        self.path_times = adjusted_times
        return True
    
    def _start_confirmed_path(self, current_time: float):
        """开始执行已确认的路径"""
        if self.path_index + 1 >= len(self.path):
            self.state = VehicleState.IDLE
            return
        
        self._start_next_move(current_time)
    
    def _start_next_move(self, current_time: float):
        """开始下一段移动"""
        if self.path_index + 1 >= len(self.path):
            self.state = VehicleState.IDLE
            return
        
        next_node = self.path[self.path_index + 1]
        self.target_position = self.road_network.node_positions.get(next_node).copy()
        
        if self.use_gnn and self.path_times:
            # GNN模式的严格时间控制
            self.move_start_time = self.path_times[self.path_index]
            self.move_duration = self.path_times[self.path_index + 1] - self.move_start_time
        else:
            # 简单模式的时间控制
            self.move_start_time = current_time
            self.move_duration = 1.0 / self.speed
        
        self.progress = 0
        self.state = VehicleState.MOVING
    
    def _arrive_at_node(self, current_time: float):
        """到达节点"""
        # 从旧节点移除
        self.road_network.remove_vehicle_from_node(self.id, self.current_node)
        
        # 移动到下一个节点
        self.path_index += 1
        self.current_node = self.path[self.path_index]
        self.position = self.target_position.copy()
        
        # 添加到新节点
        self.road_network.add_vehicle_to_node(self.id, self.current_node)
        
        # 重置重试计数
        self.retry_count = 0
        
        # 检查路径是否完成
        if self.path_index + 1 >= len(self.path):
            # 路径完成，检查是否到达目标
            target_point = self.target_point_id
            target_node = self.target_node_id
            
            if target_node and self.current_node == target_node:
                if self.mode == VehicleMode.EMPTY and target_point == self.target_loading_point:
                    self._start_loading(current_time)
                elif self.mode == VehicleMode.LOADED and target_point == self.target_unloading_point:
                    self._start_unloading(current_time)
                elif self.mode == VehicleMode.RETURNING and target_point == self.target_parking_point:
                    self._start_parking(current_time)
                else:
                    self.state = VehicleState.IDLE
                    self.path_confirmed = False
            else:
                self.state = VehicleState.IDLE
                self.path_confirmed = False
        else:
            # 继续路径
            self._start_next_move(current_time)
    
    def _start_loading(self, current_time: float):
        """开始装载"""
        self.road_network.occupy_special_point(self.target_loading_point, self.id)
        self.state = VehicleState.LOADING
        self.operation_start_time = current_time
    
    def _start_unloading(self, current_time: float):
        """开始卸载"""
        self.road_network.occupy_special_point(self.target_unloading_point, self.id)
        self.state = VehicleState.UNLOADING
        self.operation_start_time = current_time
    
    def _start_parking(self, current_time: float):
        """开始停车"""
        self.road_network.occupy_special_point(self.target_parking_point, self.id)
        self.current_parking_point = self.target_parking_point
        self.target_parking_point = None
        self.mode = VehicleMode.PARKED
        self.state = VehicleState.IDLE
        self.stats.completed_cycles += 1
        self.path_confirmed = False
    
    def _wait_and_retry(self, current_time: float):
        """等待并重试"""
        self.wait_until = current_time + 1.0 + random.random()
        self.state = VehicleState.WAITING
    
    def _reset_current_task(self):
        """重置当前任务"""
        self.target_loading_point = None
        self.target_unloading_point = None
        self.target_parking_point = None
        self.retry_count = 0
    
    def _smooth_step(self, t: float) -> float:
        """平滑插值函数"""
        return t * t * (3.0 - 2.0 * t)
    
    def get_state(self) -> Dict[str, Any]:
        """获取车辆状态"""
        return {
            'id': self.id,
            'position': self.position.copy(),
            'current_node': self.current_node,
            'heading': self.heading,
            'state': self.state.value,
            'mode': self.mode.value,
            'color': self.color,
            'target_point_id': self.target_point_id,
            'target_node_id': self.target_node_id,
            'path': self.path.copy(),
            'path_confirmed': self.path_confirmed,
            'progress': self.progress,
            'stats': {
                'completed_cycles': self.stats.completed_cycles,
                'total_distance': self.stats.total_distance,
                'wait_time': self.stats.wait_time,
                'total_time': self.stats.total_time
            },
            'loading_progress': (
                (self.road_network.global_time - self.operation_start_time) / self.loading_time
                if self.state == VehicleState.LOADING else 0
            ),
            'unloading_progress': (
                (self.road_network.global_time - self.operation_start_time) / self.unloading_time
                if self.state == VehicleState.UNLOADING else 0
            )
        }


class VehicleManager:
    """车辆管理器，管理所有车辆"""
    
    def __init__(self, road_network: RoadNetwork, use_gnn: bool = True):
        self.road_network = road_network
        self.use_gnn = use_gnn
        self.vehicles: List[Vehicle] = []
        self.max_vehicles = 6
        
        # 创建初始车辆
        self._create_initial_vehicles()
    
    def _create_initial_vehicles(self):
        """创建初始车辆"""
        parking_points = list(self.road_network.special_points['parking'].keys())
        num_vehicles = min(4, len(parking_points))
        
        for i in range(num_vehicles):
            start_parking = parking_points[i % len(parking_points)]
            vehicle = Vehicle(i, start_parking, self.road_network, self.use_gnn)
            self.vehicles.append(vehicle)
    
    def update_all(self, current_time: float, dt: float):
        """更新所有车辆"""
        for vehicle in self.vehicles:
            vehicle.update(current_time, dt)
    
    def add_vehicle(self) -> bool:
        """添加新车辆"""
        if len(self.vehicles) >= min(self.max_vehicles, 
                                   len(self.road_network.special_points['parking'])):
            return False
        
        available_parking = self.road_network.get_available_point('parking')
        if not available_parking:
            return False
        
        vehicle_id = len(self.vehicles)
        vehicle = Vehicle(vehicle_id, available_parking, self.road_network, self.use_gnn)
        self.vehicles.append(vehicle)
        return True
    
    def remove_vehicle(self) -> bool:
        """移除车辆"""
        if len(self.vehicles) <= 1:
            return False
        
        removed = self.vehicles.pop()
        
        # 清理车辆的预订和占用
        self.road_network.cancel_reservations(removed.id)
        self.road_network.remove_vehicle_from_node(removed.id, removed.current_node)
        
        # 释放特殊点
        if removed.current_parking_point:
            self.road_network.release_special_point(removed.current_parking_point)
        if removed.target_loading_point:
            self.road_network.release_special_point(removed.target_loading_point)
        if removed.target_unloading_point:
            self.road_network.release_special_point(removed.target_unloading_point)
        if removed.target_parking_point:
            self.road_network.release_special_point(removed.target_parking_point)
        
        return True
    
    def toggle_gnn_mode(self):
        """切换GNN模式"""
        self.use_gnn = not self.use_gnn
        
        # 更新所有车辆的模式
        for vehicle in self.vehicles:
            vehicle.use_gnn = self.use_gnn
            vehicle.state = VehicleState.IDLE
            vehicle.path_confirmed = False
        
        # 清理所有预订
        self.road_network.edge_reservations.clear()
        self.road_network.node_reservations.clear()
    
    def reset_all(self):
        """重置所有车辆"""
        # 清理现有车辆
        for vehicle in self.vehicles:
            self.road_network.cancel_reservations(vehicle.id)
            self.road_network.remove_vehicle_from_node(vehicle.id, vehicle.current_node)
            
            # 释放特殊点
            if vehicle.current_parking_point:
                self.road_network.release_special_point(vehicle.current_parking_point)
            if vehicle.target_loading_point:
                self.road_network.release_special_point(vehicle.target_loading_point)
            if vehicle.target_unloading_point:
                self.road_network.release_special_point(vehicle.target_unloading_point)
            if vehicle.target_parking_point:
                self.road_network.release_special_point(vehicle.target_parking_point)
        
        # 清空所有数据
        self.vehicles.clear()
        self.road_network.edge_reservations.clear()
        self.road_network.node_reservations.clear()
        
        # 清空节点占用
        for vehicles in self.road_network.node_occupancy.values():
            vehicles.clear()
        
        # 重置所有特殊点
        for points in self.road_network.special_points.values():
            for point in points.values():
                point.is_occupied = False
                point.reserved_by = None
        
        # 重新创建车辆
        self._create_initial_vehicles()
    
    def get_all_states(self) -> List[Dict[str, Any]]:
        """获取所有车辆状态"""
        return [vehicle.get_state() for vehicle in self.vehicles]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.vehicles:
            return {
                'total_vehicles': 0,
                'total_cycles': 0,
                'total_distance': 0.0,
                'total_wait_time': 0.0,
                'avg_cycle_time': 0.0,
                'state_counts': {},
                'mode_counts': {}
            }
        
        total_cycles = sum(v.stats.completed_cycles for v in self.vehicles)
        total_distance = sum(v.stats.total_distance for v in self.vehicles)
        total_wait_time = sum(v.stats.wait_time for v in self.vehicles)
        total_time = sum(v.stats.total_time for v in self.vehicles)
        
        # 状态统计
        state_counts = {}
        mode_counts = {}
        
        for state in VehicleState:
            state_counts[state.value] = 0
        for mode in VehicleMode:
            mode_counts[mode.value] = 0
        
        for vehicle in self.vehicles:
            state_counts[vehicle.state.value] += 1
            mode_counts[vehicle.mode.value] += 1
        
        return {
            'total_vehicles': len(self.vehicles),
            'total_cycles': total_cycles,
            'total_distance': total_distance,
            'total_wait_time': total_wait_time,
            'avg_cycle_time': total_time / total_cycles if total_cycles > 0 else 0.0,
            'state_counts': state_counts,
            'mode_counts': mode_counts
        }