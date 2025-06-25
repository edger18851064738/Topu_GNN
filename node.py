import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
import networkx as nx
from collections import defaultdict, deque
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set
import heapq
from enum import Enum

class VehicleState(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    WAITING = "waiting"
    MOVING = "moving"
    BLOCKED = "blocked"

@dataclass
class NodeFeature:
    """节点特征类"""
    occupancy: float = 0.0      # 占用度
    connectivity: float = 0.0   # 连通性
    congestion: float = 0.0     # 拥堵度
    centrality: float = 0.0     # 中心度

@dataclass
class EdgeReservation:
    """边预留信息"""
    vehicle_id: int
    start_time: float
    end_time: float
    direction: Tuple[str, str]  # (from_node, to_node)

class RoadNetwork:
    """道路网络类 - 带严格冲突避免的GNN算法"""
    
    def __init__(self, grid_size: int = 6):
        self.grid_size = grid_size
        self.graph = nx.Graph()
        self.node_positions = {}
        self.edge_reservations = defaultdict(list)  # 边预留列表
        self.node_occupancy = defaultdict(set)      # 节点占用
        self.node_features = {}                     # 节点特征
        self.global_time = 0.0
        
        self._create_grid_network()
        self._initialize_features()
    
    def _create_grid_network(self):
        """创建网格道路网络"""
        # 创建节点
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                node_id = f"{i}-{j}"
                self.graph.add_node(node_id)
                self.node_positions[node_id] = (j, i)
        
        # 创建边
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                current = f"{i}-{j}"
                
                # 水平连接
                if j < self.grid_size - 1:
                    right = f"{i}-{j+1}"
                    self.graph.add_edge(current, right, weight=1.0)
                
                # 垂直连接
                if i < self.grid_size - 1:
                    down = f"{i+1}-{j}"
                    self.graph.add_edge(current, down, weight=1.0)
    
    def _initialize_features(self):
        """初始化节点特征"""
        for node in self.graph.nodes():
            self.node_features[node] = NodeFeature()
            self._update_node_feature(node)
    
    def _update_node_feature(self, node: str):
        """更新单个节点特征"""
        # 连通性
        connectivity = len(list(self.graph.neighbors(node)))
        
        # 拥堵度 - 基于当前和预留的占用情况
        neighbors = list(self.graph.neighbors(node))
        congestion = 0.0
        current_time = self.global_time
        
        for neighbor in neighbors:
            edge_key = tuple(sorted([node, neighbor]))
            reservations = self.edge_reservations[edge_key]
            # 计算未来一段时间内的占用情况
            future_occupied = any(
                r.start_time <= current_time + 2.0 and r.end_time >= current_time
                for r in reservations
            )
            if future_occupied:
                congestion += 1
        
        congestion = congestion / max(len(neighbors), 1)
        
        # 中心度
        centrality = connectivity / 8.0  # 网格中最大连接数为4，但我们用8作为归一化
        
        # 占用度 - 当前节点上的车辆数
        occupancy = len(self.node_occupancy[node]) * 0.5
        
        self.node_features[node] = NodeFeature(
            occupancy=occupancy,
            connectivity=connectivity,
            congestion=congestion,
            centrality=centrality
        )
    
    def update_time(self, current_time: float):
        """更新全局时间并清理过期预留"""
        self.global_time = current_time
        
        # 清理过期的边预留
        for edge_key in list(self.edge_reservations.keys()):
            reservations = self.edge_reservations[edge_key]
            # 保留未过期的预留
            active_reservations = [
                r for r in reservations 
                if r.end_time > current_time
            ]
            if active_reservations:
                self.edge_reservations[edge_key] = active_reservations
            else:
                del self.edge_reservations[edge_key]
    
    def reserve_edge(self, from_node: str, to_node: str, vehicle_id: int, 
                    start_time: float, duration: float) -> bool:
        """预留边使用权"""
        edge_key = tuple(sorted([from_node, to_node]))
        end_time = start_time + duration
        
        # 检查是否与现有预留冲突
        existing_reservations = self.edge_reservations[edge_key]
        for reservation in existing_reservations:
            if not (end_time <= reservation.start_time or 
                   start_time >= reservation.end_time):
                return False  # 时间冲突
        
        # 添加新预留
        new_reservation = EdgeReservation(
            vehicle_id=vehicle_id,
            start_time=start_time,
            end_time=end_time,
            direction=(from_node, to_node)
        )
        self.edge_reservations[edge_key].append(new_reservation)
        return True
    
    def cancel_reservations(self, vehicle_id: int):
        """取消车辆的所有预留"""
        for edge_key in list(self.edge_reservations.keys()):
            reservations = self.edge_reservations[edge_key]
            remaining = [r for r in reservations if r.vehicle_id != vehicle_id]
            if remaining:
                self.edge_reservations[edge_key] = remaining
            else:
                del self.edge_reservations[edge_key]
    
    def is_edge_available(self, from_node: str, to_node: str, 
                         start_time: float, duration: float, 
                         exclude_vehicle: int = -1) -> bool:
        """检查边在指定时间段是否可用"""
        edge_key = tuple(sorted([from_node, to_node]))
        end_time = start_time + duration
        
        reservations = self.edge_reservations.get(edge_key, [])
        for reservation in reservations:
            if reservation.vehicle_id == exclude_vehicle:
                continue
            if not (end_time <= reservation.start_time or 
                   start_time >= reservation.end_time):
                return False
        return True
    
    def add_vehicle_to_node(self, vehicle_id: int, node: str):
        """将车辆添加到节点"""
        self.node_occupancy[node].add(vehicle_id)
    
    def remove_vehicle_from_node(self, vehicle_id: int, node: str):
        """从节点移除车辆"""
        self.node_occupancy[node].discard(vehicle_id)
    
    def gnn_pathfinding_with_reservation(self, start: str, end: str, 
                                       vehicle_id: int, current_time: float) -> Tuple[List[str], List[float]]:
        """GNN路径规划 + 时间预留"""
        if start == end:
            return [start], [current_time]
        
        # 更新所有节点特征
        for node in self.graph.nodes():
            self._update_node_feature(node)
        
        # A*算法结合时间预留
        # 状态: (cost, current_node, path, arrival_times)
        heap = [(0, start, [start], [current_time])]
        visited = {}  # node -> earliest_arrival_time
        
        while heap:
            cost, current, path, times = heapq.heappop(heap)
            current_arrival_time = times[-1]
            
            # 如果已访问且时间更早，跳过
            if current in visited and visited[current] <= current_arrival_time:
                continue
            visited[current] = current_arrival_time
            
            if current == end:
                return path, times
            
            # 遍历邻居节点
            for neighbor in self.graph.neighbors(current):
                if neighbor in path:  # 避免环路
                    continue
                
                # 计算移动时间
                travel_time = self._compute_travel_time(current, neighbor, vehicle_id)
                departure_time = current_arrival_time + 0.1  # 在节点停留时间
                arrival_time = departure_time + travel_time
                
                # 检查边是否可用
                if not self.is_edge_available(current, neighbor, departure_time, 
                                            travel_time, exclude_vehicle=vehicle_id):
                    # 如果不可用，等待一段时间后重试
                    wait_time = self._find_next_available_time(
                        current, neighbor, departure_time, travel_time, vehicle_id)
                    if wait_time > departure_time + 5.0:  # 最多等待5秒
                        continue
                    departure_time = wait_time
                    arrival_time = departure_time + travel_time
                
                # 计算GNN增强的成本
                edge_cost = self._compute_gnn_edge_cost(current, neighbor, departure_time)
                heuristic = self._manhattan_distance(neighbor, end)
                total_cost = arrival_time + edge_cost + heuristic
                
                new_path = path + [neighbor]
                new_times = times + [arrival_time]
                heapq.heappush(heap, (total_cost, neighbor, new_path, new_times))
        
        return [], []  # 无路径
    
    def _compute_travel_time(self, from_node: str, to_node: str, vehicle_id: int) -> float:
        """计算移动时间"""
        base_time = 1.0
        
        # 基于节点特征调整时间
        to_feature = self.node_features[to_node]
        time_factor = 1.0 + to_feature.congestion * 0.5 + to_feature.occupancy * 0.3
        
        return base_time * time_factor
    
    def _find_next_available_time(self, from_node: str, to_node: str, 
                                 earliest_start: float, duration: float, 
                                 vehicle_id: int) -> float:
        """找到边的下一个可用时间"""
        edge_key = tuple(sorted([from_node, to_node]))
        reservations = self.edge_reservations.get(edge_key, [])
        
        # 过滤掉当前车辆的预留
        other_reservations = [r for r in reservations if r.vehicle_id != vehicle_id]
        
        if not other_reservations:
            return earliest_start
        
        # 按开始时间排序
        other_reservations.sort(key=lambda x: x.start_time)
        
        # 尝试在预留间隙中安排
        current_time = earliest_start
        for reservation in other_reservations:
            if current_time + duration <= reservation.start_time:
                return current_time
            current_time = max(current_time, reservation.end_time)
        
        return current_time
    
    def _compute_gnn_edge_cost(self, from_node: str, to_node: str, time: float) -> float:
        """计算GNN增强的边权重"""
        base_weight = 1.0
        
        # 获取目标节点特征
        to_feature = self.node_features[to_node]
        
        # GNN特征聚合
        cost = base_weight
        cost += to_feature.occupancy * 3.0      # 占用度惩罚
        cost += to_feature.congestion * 8.0     # 拥堵重度惩罚
        cost -= to_feature.centrality * 1.0     # 中心度小幅奖励
        
        # 时间相关的动态权重
        edge_key = tuple(sorted([from_node, to_node]))
        future_congestion = len([
            r for r in self.edge_reservations.get(edge_key, [])
            if r.start_time <= time + 3.0 and r.end_time >= time
        ])
        cost += future_congestion * 2.0
        
        return max(cost, 0.1)
    
    def _manhattan_distance(self, node1: str, node2: str) -> float:
        """计算曼哈顿距离启发式"""
        pos1 = self.node_positions[node1]
        pos2 = self.node_positions[node2]
        return (abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])) * 0.5
    
    def simple_pathfinding(self, start: str, end: str) -> List[str]:
        """简单BFS路径规划"""
        try:
            return nx.shortest_path(self.graph, start, end)
        except nx.NetworkXNoPath:
            return []

class Vehicle:
    """改进的车辆智能体类 - 严格冲突避免"""
    
    def __init__(self, vehicle_id: int, start_node: str, target_node: str, 
                 road_network: RoadNetwork, use_gnn: bool = True):
        self.id = vehicle_id
        self.current_node = start_node
        self.target_node = target_node
        self.road_network = road_network
        self.use_gnn = use_gnn
        
        # 路径和时间规划
        self.path = []
        self.path_times = []
        self.path_index = 0
        
        # 位置和动画
        self.position = np.array(road_network.node_positions[start_node], dtype=float)
        self.target_position = self.position.copy()
        self.progress = 0.0
        self.speed = 0.6 + random.random() * 0.4
        
        # 状态管理
        self.state = VehicleState.IDLE
        self.move_start_time = 0.0
        self.move_duration = 0.0
        self.wait_until = 0.0
        self.retry_count = 0
        self.max_retries = 3
        
        # 统计
        self.total_distance = 0.0
        self.completed_tasks = 0
        self.wait_time = 0.0
        
        # 可视化
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
        self.color = colors[vehicle_id % len(colors)]
        
        # 将车辆添加到起始节点
        self.road_network.add_vehicle_to_node(self.id, self.current_node)
    
    def update(self, current_time: float, dt: float):
        """主更新函数"""
        if self.state == VehicleState.IDLE:
            self._plan_new_path(current_time)
            
        elif self.state == VehicleState.PLANNING:
            self._execute_planning(current_time)
            
        elif self.state == VehicleState.WAITING:
            if current_time >= self.wait_until:
                self.state = VehicleState.IDLE
                self.retry_count += 1
                if self.retry_count > self.max_retries:
                    self._select_new_target()
                    self.retry_count = 0
            else:
                self.wait_time += dt
                
        elif self.state == VehicleState.MOVING:
            self._update_movement(current_time, dt)
            
        elif self.state == VehicleState.BLOCKED:
            # 重新规划
            self.road_network.cancel_reservations(self.id)
            self.state = VehicleState.IDLE
            self.retry_count += 1
    
    def _plan_new_path(self, current_time: float):
        """规划新路径"""
        if self.current_node == self.target_node:
            self.completed_tasks += 1
            self._select_new_target()
        
        self.state = VehicleState.PLANNING
        
        # 取消之前的预留
        self.road_network.cancel_reservations(self.id)
        
        # 规划路径
        if self.use_gnn:
            self.path, self.path_times = self.road_network.gnn_pathfinding_with_reservation(
                self.current_node, self.target_node, self.id, current_time)
        else:
            simple_path = self.road_network.simple_pathfinding(
                self.current_node, self.target_node)
            self.path = simple_path
            # 为简单路径生成时间序列
            if simple_path:
                self.path_times = [current_time + i * 1.5 for i in range(len(simple_path))]
            else:
                self.path_times = []
    
    def _execute_planning(self, current_time: float):
        """执行路径规划结果"""
        if not self.path or len(self.path) < 2:
            # 无法找到路径，等待后重试
            self.wait_until = current_time + 1.0 + random.random()
            self.state = VehicleState.WAITING
            return
        
        # 尝试预留整条路径
        success = True
        if self.use_gnn:  # 只有GNN模式需要预留，因为已经计算了时间
            for i in range(len(self.path) - 1):
                from_node = self.path[i]
                to_node = self.path[i + 1]
                start_time = self.path_times[i]
                duration = self.path_times[i + 1] - start_time
                
                if not self.road_network.reserve_edge(from_node, to_node, self.id, 
                                                    start_time, duration):
                    success = False
                    break
        
        if success:
            self.path_index = 0
            self._start_next_move(current_time)
        else:
            # 预留失败，等待后重试
            self.road_network.cancel_reservations(self.id)
            self.wait_until = current_time + 0.5 + random.random() * 0.5
            self.state = VehicleState.WAITING
    
    def _start_next_move(self, current_time: float):
        """开始下一段移动"""
        if self.path_index + 1 >= len(self.path):
            # 到达目标
            self.state = VehicleState.IDLE
            return
        
        next_node = self.path[self.path_index + 1]
        
        # 设置移动参数
        self.target_position = np.array(
            self.road_network.node_positions[next_node], dtype=float)
        
        if self.use_gnn and self.path_times:
            self.move_start_time = self.path_times[self.path_index]
            self.move_duration = self.path_times[self.path_index + 1] - self.move_start_time
        else:
            self.move_start_time = current_time
            self.move_duration = 1.0 / self.speed
        
        self.progress = 0.0
        self.state = VehicleState.MOVING
    
    def _update_movement(self, current_time: float, dt: float):
        """更新移动状态"""
        if current_time < self.move_start_time:
            return  # 还没到移动时间
        
        # 计算移动进度
        elapsed = current_time - self.move_start_time
        self.progress = min(elapsed / self.move_duration, 1.0)
        
        # 插值位置
        start_pos = np.array(self.road_network.node_positions[self.path[self.path_index]])
        self.position = start_pos + (self.target_position - start_pos) * self.progress
        
        # 更新里程
        if self.progress > 0:
            distance = np.linalg.norm(self.target_position - start_pos) * dt / self.move_duration
            self.total_distance += distance
        
        # 检查是否到达
        if self.progress >= 1.0:
            self._arrive_at_node(current_time)
    
    def _arrive_at_node(self, current_time: float):
        """到达节点处理"""
        # 从当前节点移除
        self.road_network.remove_vehicle_from_node(self.id, self.current_node)
        
        # 移动到下一个节点
        self.path_index += 1
        self.current_node = self.path[self.path_index]
        self.position = self.target_position.copy()
        
        # 添加到新节点
        self.road_network.add_vehicle_to_node(self.id, self.current_node)
        
        # 重置重试计数
        self.retry_count = 0
        
        # 开始下一段移动
        self._start_next_move(current_time)
    
    def _select_new_target(self):
        """选择新目标"""
        nodes = list(self.road_network.graph.nodes())
        available_nodes = [n for n in nodes if n != self.current_node]
        if available_nodes:
            self.target_node = random.choice(available_nodes)
    
    def get_state_info(self) -> str:
        """获取状态信息"""
        return f"V{self.id}:{self.state.value}"

class GNNMultiAgentSimulation:
    """改进的GNN多智能体仿真类"""
    
    def __init__(self, grid_size: int = 6, num_vehicles: int = 6):
        self.grid_size = grid_size
        self.road_network = RoadNetwork(grid_size)
        self.vehicles = []
        self.use_gnn = True
        self.current_time = 0.0
        
        # 创建初始车辆
        self._create_initial_vehicles(num_vehicles)
        
        # 可视化设置
        self.fig, (self.ax_main, self.ax_stats) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 主仿真区域
        self.ax_main.set_xlim(-0.5, grid_size - 0.5)
        self.ax_main.set_ylim(-0.5, grid_size - 0.5)
        self.ax_main.set_aspect('equal')
        self.ax_main.set_title('GNN多智能体道路协同演示 (严格冲突避免)', fontsize=14, fontweight='bold')
        self.ax_main.grid(True, alpha=0.3)
        
        # 统计信息区域
        self.ax_stats.set_xlim(0, 1)
        self.ax_stats.set_ylim(0, 1)
        self.ax_stats.axis('off')
        
        # 动画
        self.animation = None
        self.is_running = False
    
    def _create_initial_vehicles(self, num_vehicles: int):
        """创建初始车辆"""
        nodes = list(self.road_network.graph.nodes())
        
        for i in range(num_vehicles):
            start_node = random.choice(nodes)
            target_node = random.choice([n for n in nodes if n != start_node])
            
            vehicle = Vehicle(i, start_node, target_node, 
                            self.road_network, self.use_gnn)
            self.vehicles.append(vehicle)
    
    def toggle_gnn_mode(self):
        """切换GNN模式"""
        self.use_gnn = not self.use_gnn
        for vehicle in self.vehicles:
            vehicle.use_gnn = self.use_gnn
            vehicle.road_network.cancel_reservations(vehicle.id)
            vehicle.state = VehicleState.IDLE
            vehicle.retry_count = 0
        print(f"GNN模式: {'启用' if self.use_gnn else '禁用'}")
    
    def add_vehicle(self):
        """添加新车辆"""
        if len(self.vehicles) >= 15:  # 限制最大车辆数
            print("车辆数已达上限!")
            return
            
        nodes = list(self.road_network.graph.nodes())
        start_node = random.choice(nodes)
        target_node = random.choice([n for n in nodes if n != start_node])
        
        vehicle_id = len(self.vehicles)
        vehicle = Vehicle(vehicle_id, start_node, target_node,
                        self.road_network, self.use_gnn)
        self.vehicles.append(vehicle)
        print(f"添加车辆 {vehicle_id}")
    
    def remove_vehicle(self):
        """移除车辆"""
        if self.vehicles:
            removed = self.vehicles.pop()
            removed.road_network.cancel_reservations(removed.id)
            removed.road_network.remove_vehicle_from_node(removed.id, removed.current_node)
            print(f"移除车辆 {removed.id}")
    
    def update(self, frame):
        """动画更新函数"""
        dt = 0.1
        self.current_time += dt
        
        # 更新网络时间
        self.road_network.update_time(self.current_time)
        
        # 更新所有车辆
        for vehicle in self.vehicles:
            vehicle.update(self.current_time, dt)
        
        # 清除绘图
        self.ax_main.clear()
        self.ax_stats.clear()
        
        # 重新设置主绘图区域
        self.ax_main.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax_main.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax_main.set_aspect('equal')
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.set_title(f'GNN多智能体道路协同演示 - {"GNN模式" if self.use_gnn else "简单模式"}', 
                             fontsize=14, fontweight='bold')
        
        # 绘制内容
        self._draw_road_network()
        self._draw_vehicles()
        self._draw_reservations()
        self._draw_statistics()
        
        return []
    
    def _draw_road_network(self):
        """绘制道路网络"""
        # 绘制边
        for edge in self.road_network.graph.edges():
            node1, node2 = edge
            pos1 = self.road_network.node_positions[node1]
            pos2 = self.road_network.node_positions[node2]
            
            # 检查当前是否有预留
            edge_key = tuple(sorted([node1, node2]))
            has_reservation = len(self.road_network.edge_reservations.get(edge_key, [])) > 0
            
            if has_reservation:
                color = 'orange'
                width = 4
                alpha = 0.8
            else:
                color = 'lightgray'
                width = 2
                alpha = 0.6
            
            self.ax_main.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                            color=color, linewidth=width, alpha=alpha)
        
        # 绘制节点
        for node, pos in self.road_network.node_positions.items():
            # 根据节点上的车辆数调整颜色
            vehicle_count = len(self.road_network.node_occupancy[node])
            if vehicle_count > 0:
                color = 'lightcoral'
                size = 0.2
            else:
                color = 'lightblue'
                size = 0.15
            
            circle = Circle(pos, size, color=color, 
                          edgecolor='navy', linewidth=2, alpha=0.7)
            self.ax_main.add_patch(circle)
            self.ax_main.text(pos[0], pos[1], f'{node}\n({vehicle_count})', 
                            ha='center', va='center', fontsize=7)
    
    def _draw_vehicles(self):
        """绘制车辆"""
        for vehicle in self.vehicles:
            x, y = vehicle.position
            
            # 根据状态调整车辆外观
            if vehicle.state == VehicleState.MOVING:
                alpha = 1.0
                size = 0.12
            elif vehicle.state == VehicleState.WAITING:
                alpha = 0.6
                size = 0.10
            else:
                alpha = 0.8
                size = 0.11
            
            # 绘制车辆矩形
            rect = Rectangle((x-size, y-size), size*2, size*2, 
                           color=vehicle.color, alpha=alpha, 
                           edgecolor='white', linewidth=1)
            self.ax_main.add_patch(rect)
            
            # 绘制车辆ID和状态
            state_short = vehicle.state.value[:1].upper()
            self.ax_main.text(x, y, f'{vehicle.id}\n{state_short}', 
                            ha='center', va='center',
                            color='white', fontweight='bold', fontsize=7)
            
            # 绘制目标连线
            target_pos = self.road_network.node_positions[vehicle.target_node]
            self.ax_main.plot([x, target_pos[0]], [y, target_pos[1]], 
                            color=vehicle.color, linestyle=':', alpha=0.4, linewidth=1)
            
            # 绘制目标节点标记
            target_circle = Circle(target_pos, 0.25, 
                                 facecolor='none', edgecolor=vehicle.color, 
                                 linewidth=2, alpha=0.6)
            self.ax_main.add_patch(target_circle)
    
    def _draw_reservations(self):
        """绘制边预留信息"""
        current_time = self.current_time
        
        for edge_key, reservations in self.road_network.edge_reservations.items():
            if not reservations:
                continue
                
            node1, node2 = edge_key
            pos1 = self.road_network.node_positions[node1]
            pos2 = self.road_network.node_positions[node2]
            
            # 绘制预留时间线
            for i, reservation in enumerate(reservations):
                if reservation.end_time < current_time:
                    continue
                    
                # 计算预留进度
                progress = max(0, (current_time - reservation.start_time) / 
                             (reservation.end_time - reservation.start_time))
                progress = min(1, progress)
                
                # 绘制预留线段
                vehicle = next((v for v in self.vehicles if v.id == reservation.vehicle_id), None)
                if vehicle:
                    offset = (i - len(reservations)/2) * 0.05
                    x1, y1 = pos1[0] + offset, pos1[1] + offset
                    x2, y2 = pos2[0] + offset, pos2[1] + offset
                    
                    self.ax_main.plot([x1, x2], [y1, y2], 
                                    color=vehicle.color, linewidth=3, alpha=0.3)
    
    def _draw_statistics(self):
        """绘制统计信息"""
        self.ax_stats.set_xlim(0, 1)
        self.ax_stats.set_ylim(0, 1)
        self.ax_stats.axis('off')
        
        # 计算统计数据
        total_distance = sum(v.total_distance for v in self.vehicles)
        total_tasks = sum(v.completed_tasks for v in self.vehicles)
        total_wait_time = sum(v.wait_time for v in self.vehicles)
        
        # 状态统计
        state_counts = defaultdict(int)
        for vehicle in self.vehicles:
            state_counts[vehicle.state] += 1
        
        # 路网利用率
        active_reservations = sum(len(reservations) for reservations in 
                                self.road_network.edge_reservations.values())
        total_edges = len(self.road_network.graph.edges())
        utilization = (active_reservations / total_edges * 100) if total_edges > 0 else 0
        
        # 绘制文本信息
        stats_text = f"""
        ╔══════ 仿真统计 ══════╗
        ║ 模式: {'GNN智能' if self.use_gnn else '简单BFS':>12} ║
        ║ 时间: {self.current_time:>12.1f}s ║
        ║ 车辆: {len(self.vehicles):>12d} ║
        ║ 总里程: {total_distance:>10.1f} ║
        ║ 完成任务: {total_tasks:>8d} ║
        ║ 等待时间: {total_wait_time:>7.1f}s ║
        ║ 路网利用: {utilization:>7.1f}% ║
        ╠══════ 车辆状态 ══════╣
        ║ 空闲: {state_counts[VehicleState.IDLE]:>13d} ║
        ║ 规划: {state_counts[VehicleState.PLANNING]:>13d} ║
        ║ 等待: {state_counts[VehicleState.WAITING]:>13d} ║
        ║ 移动: {state_counts[VehicleState.MOVING]:>13d} ║
        ║ 阻塞: {state_counts[VehicleState.BLOCKED]:>13d} ║
        ╚══════════════════════╝
        
        控制说明:
        空格键: 开始/暂停
        'g': 切换GNN模式  
        '+': 添加车辆
        '-': 移除车辆
        'q': 退出
        """.strip()
        
        self.ax_stats.text(0.05, 0.95, stats_text, 
                         transform=self.ax_stats.transAxes,
                         fontfamily='monospace', fontsize=9,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round,pad=0.5', 
                                 facecolor='lightgray', alpha=0.8))
    
    def start_animation(self):
        """开始动画"""
        if not self.is_running:
            self.animation = animation.FuncAnimation(
                self.fig, self.update, interval=100, blit=False)
            self.is_running = True
            print("仿真开始")
    
    def stop_animation(self):
        """停止动画"""
        if self.animation:
            self.animation.event_source.stop()
            self.is_running = False
            print("仿真停止")
    
    def reset_simulation(self):
        """重置仿真"""
        self.current_time = 0.0
        self.road_network = RoadNetwork(self.grid_size)
        
        # 重新创建车辆
        num_vehicles = len(self.vehicles)
        self.vehicles.clear()
        self._create_initial_vehicles(num_vehicles)
        print("仿真重置")
    
    def show(self):
        """显示仿真界面"""
        plt.tight_layout()
        plt.show()

def main():
    """主函数"""
    print("GNN多智能体道路协同演示 (严格冲突避免版本)")
    print("=" * 60)
    
    # 创建仿真
    sim = GNNMultiAgentSimulation(grid_size=6, num_vehicles=4)
    
    # 控制说明
    print("控制说明:")
    print("- 空格键: 开始/停止仿真")
    print("- 'g': 切换GNN模式")
    print("- '+': 添加车辆")
    print("- '-': 移除车辆")
    print("- 'r': 重置仿真")
    print("- 'q': 退出")
    print("=" * 60)
    
    # 键盘事件处理
    def on_key(event):
        if event.key == ' ':
            if sim.is_running:
                sim.stop_animation()
            else:
                sim.start_animation()
        elif event.key == 'g':
            sim.toggle_gnn_mode()
        elif event.key == '+':
            sim.add_vehicle()
        elif event.key == '-':
            sim.remove_vehicle()
        elif event.key == 'r':
            sim.reset_simulation()
        elif event.key == 'q':
            plt.close()
    
    sim.fig.canvas.mpl_connect('key_press_event', on_key)
    
    # 开始仿真
    sim.start_animation()
    sim.show()

if __name__ == "__main__":
    main()