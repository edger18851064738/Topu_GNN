"""
environment.py - 增强版环境管理系统
支持安全矩形冲突检测、车辆安全参数管理、效率跟踪
保持原有接口兼容性，新增安全和性能优化功能
"""

import numpy as np
import math
import random
import time
import threading
import json
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from PyQt5.QtGui import QColor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnvironmentState(Enum):
    """环境状态"""
    UNINITIALIZED = "uninitialized"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"

class VehicleStatus(Enum):
    """车辆状态"""
    IDLE = "idle"
    MOVING = "moving"
    LOADING = "loading"
    UNLOADING = "unloading"
    PLANNING = "planning"
    WAITING = "waiting"
    MAINTENANCE = "maintenance"

@dataclass
class SafetyRectangle:
    """安全矩形"""
    center_x: float
    center_y: float
    width: float
    height: float
    rotation: float = 0.0  # 车辆朝向角度
    
    def get_corners(self) -> List[Tuple[float, float]]:
        """获取矩形四个角点"""
        half_width = self.width / 2
        half_height = self.height / 2
        
        # 相对于中心的四个角点
        corners = [
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height)
        ]
        
        # 应用旋转
        cos_r = math.cos(self.rotation)
        sin_r = math.sin(self.rotation)
        
        rotated_corners = []
        for x, y in corners:
            new_x = self.center_x + x * cos_r - y * sin_r
            new_y = self.center_y + x * sin_r + y * cos_r
            rotated_corners.append((new_x, new_y))
        
        return rotated_corners
    
    def intersects(self, other: 'SafetyRectangle') -> bool:
        """检查两个矩形是否相交 - 使用分离轴定理"""
        corners1 = self.get_corners()
        corners2 = other.get_corners()
        
        # 获取两个矩形的轴
        axes = []
        
        # 第一个矩形的轴
        for i in range(4):
            j = (i + 1) % 4
            edge = (corners1[j][0] - corners1[i][0], corners1[j][1] - corners1[i][1])
            # 垂直向量作为分离轴
            axis = (-edge[1], edge[0])
            length = math.sqrt(axis[0]**2 + axis[1]**2)
            if length > 0:
                axes.append((axis[0]/length, axis[1]/length))
        
        # 第二个矩形的轴
        for i in range(4):
            j = (i + 1) % 4
            edge = (corners2[j][0] - corners2[i][0], corners2[j][1] - corners2[i][1])
            axis = (-edge[1], edge[0])
            length = math.sqrt(axis[0]**2 + axis[1]**2)
            if length > 0:
                axes.append((axis[0]/length, axis[1]/length))
        
        # 检查每个轴上的投影
        for axis in axes:
            # 投影第一个矩形
            proj1 = [corner[0] * axis[0] + corner[1] * axis[1] for corner in corners1]
            min1, max1 = min(proj1), max(proj1)
            
            # 投影第二个矩形
            proj2 = [corner[0] * axis[0] + corner[1] * axis[1] for corner in corners2]
            min2, max2 = min(proj2), max(proj2)
            
            # 检查是否分离
            if max1 < min2 or max2 < min1:
                return False
        
        return True

@dataclass
class VehicleSafetyParams:
    """车辆安全参数"""
    length: float = 6.0
    width: float = 3.0
    safety_margin: float = 1.5
    turning_radius: float = 8.0
    max_speed: float = 2.0
    braking_distance: float = 5.0
    
    def get_safe_dimensions(self) -> Tuple[float, float]:
        """获取安全尺寸（包含安全边距）"""
        return (self.length + self.safety_margin * 2, 
                self.width + self.safety_margin * 2)
    
    def create_safety_rectangle(self, position: Tuple[float, float, float]) -> SafetyRectangle:
        """为指定位置创建安全矩形"""
        x, y = position[0], position[1]
        theta = position[2] if len(position) > 2 else 0.0
        
        safe_length, safe_width = self.get_safe_dimensions()
        
        return SafetyRectangle(
            center_x=x,
            center_y=y,
            width=safe_width,
            height=safe_length,
            rotation=theta
        )
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'length': self.length,
            'width': self.width,
            'safety_margin': self.safety_margin,
            'turning_radius': self.turning_radius,
            'max_speed': self.max_speed,
            'braking_distance': self.braking_distance
        }

@dataclass
class VehiclePerformanceMetrics:
    """车辆性能指标"""
    total_distance: float = 0.0
    total_time: float = 0.0
    idle_time: float = 0.0
    productive_time: float = 0.0
    efficiency_score: float = 0.5
    backbone_usage_ratio: float = 0.0
    conflict_count: int = 0
    last_conflict_time: float = 0.0
    conflict_resolution_success_rate: float = 1.0
    path_switches: int = 0
    stability_score: float = 1.0
    
    def calculate_utilization_rate(self) -> float:
        """计算车辆利用率"""
        if self.total_time <= 0:
            return 0.0
        return self.productive_time / self.total_time
    
    def update_efficiency(self, new_efficiency: float):
        """更新效率分数（使用移动平均）"""
        alpha = 0.3  # 平滑因子
        self.efficiency_score = alpha * new_efficiency + (1 - alpha) * self.efficiency_score

@dataclass
class EnhancedVehicleInfo:
    """增强版车辆信息"""
    vehicle_id: str
    position: Tuple[float, float, float]
    initial_position: Tuple[float, float, float]
    goal: Optional[Tuple[float, float, float]] = None
    
    # 基本属性
    vehicle_type: str = "dump_truck"
    max_load: float = 100
    current_load: float = 0
    speed: float = 1.0
    
    # 状态信息
    status: str = 'idle'
    vehicle_status: VehicleStatus = VehicleStatus.IDLE
    path: Optional[List] = None
    path_index: int = 0
    progress: float = 0.0
    
    # 任务信息
    completed_cycles: int = 0
    current_task_id: Optional[str] = None
    task_queue: List[str] = field(default_factory=list)
    
    # 新增：安全参数
    safety_params: VehicleSafetyParams = field(default_factory=VehicleSafetyParams)
    
    # 新增：性能指标
    performance_metrics: VehiclePerformanceMetrics = field(default_factory=VehiclePerformanceMetrics)
    
    # 路径结构信息
    path_structure: Dict = field(default_factory=dict)
    backbone_path_id: Optional[str] = None
    last_backbone_switch_time: float = 0.0
    
    # 显示属性
    color: Optional[QColor] = None
    
    def __post_init__(self):
        if self.color is None:
            self.color = QColor(
                random.randint(100, 255), 
                random.randint(100, 255), 
                random.randint(100, 255)
            )
    
    def get_safety_rectangle(self) -> SafetyRectangle:
        """获取当前位置的安全矩形"""
        return self.safety_params.create_safety_rectangle(self.position)
    
    def update_position(self, new_position: Tuple[float, float, float], 
                       time_delta: float = 0.0):
        """更新位置并计算移动距离"""
        if self.position and time_delta > 0:
            # 计算移动距离
            distance = math.sqrt(
                (new_position[0] - self.position[0])**2 +
                (new_position[1] - self.position[1])**2
            )
            self.performance_metrics.total_distance += distance
            self.performance_metrics.total_time += time_delta
            
            # 更新状态时间
            if self.vehicle_status == VehicleStatus.IDLE:
                self.performance_metrics.idle_time += time_delta
            else:
                self.performance_metrics.productive_time += time_delta
        
        self.position = new_position
    
    def record_conflict(self, resolved: bool = False):
        """记录冲突事件"""
        self.performance_metrics.conflict_count += 1
        self.performance_metrics.last_conflict_time = time.time()
        
        # 更新冲突解决成功率
        if resolved:
            success_count = getattr(self, '_conflict_success_count', 0) + 1
            setattr(self, '_conflict_success_count', success_count)
            
            total_conflicts = self.performance_metrics.conflict_count
            self.performance_metrics.conflict_resolution_success_rate = success_count / total_conflicts
    
    def record_backbone_switch(self, new_backbone_id: str):
        """记录骨干路径切换"""
        if self.backbone_path_id and self.backbone_path_id != new_backbone_id:
            self.performance_metrics.path_switches += 1
            self.last_backbone_switch_time = time.time()
            
            # 更新稳定性分数
            switch_penalty = 0.1 if self.performance_metrics.path_switches <= 3 else 0.2
            self.performance_metrics.stability_score = max(0.1, 
                self.performance_metrics.stability_score - switch_penalty)
        
        self.backbone_path_id = new_backbone_id
    
    # 字典访问支持（保持向后兼容）
    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        
        # 向后兼容的别名映射 - 完全兼容原GUI访问方式
        alias_map = {
            'load': 'current_load',
            'type': 'vehicle_type', 
            'id': 'vehicle_id',
            'efficiency_score': lambda: self.performance_metrics.efficiency_score,
            'conflict_count': lambda: self.performance_metrics.conflict_count,
            'total_distance': lambda: self.performance_metrics.total_distance,
            'backbone_usage_ratio': lambda: self.performance_metrics.backbone_usage_ratio,
            'safety_params': lambda: self.safety_params.to_dict(),
            # 新增：GUI可能需要的所有属性
            'vehicle_length': lambda: self.safety_params.length,
            'vehicle_width': lambda: self.safety_params.width,
            'safety_margin': lambda: self.safety_params.safety_margin,
            'turning_radius': lambda: self.safety_params.turning_radius,
            'total_time': lambda: self.performance_metrics.total_time,
            'idle_time': lambda: self.performance_metrics.idle_time,
            'productive_time': lambda: self.performance_metrics.productive_time,
            'average_speed': lambda: (self.performance_metrics.total_distance / 
                                    max(0.1, self.performance_metrics.total_time))
        }
        
        if key in alias_map:
            value = alias_map[key]
            return value() if callable(value) else getattr(self, value)
        
        raise KeyError(f"'{key}' not found")
    
    def __setitem__(self, key, value):
        alias_map = {
            'load': 'current_load',
            'type': 'vehicle_type',
            'id': 'vehicle_id'
        }
        
        actual_key = alias_map.get(key, key)
        
        if hasattr(self, actual_key):
            setattr(self, actual_key, value)
        else:
            setattr(self, key, value)
    
    def __contains__(self, key):
        alias_map = {
            'load': 'current_load',
            'type': 'vehicle_type',
            'id': 'vehicle_id',
            'efficiency_score': 'performance_metrics',
            'safety_params': 'safety_params'
        }
        return hasattr(self, key) or key in alias_map
    
    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

class SafetyCollisionDetector:
    """安全矩形碰撞检测器"""
    
    def __init__(self, env):
        self.env = env
        self.collision_cache = {}
        self.cache_timeout = 1.0  # 缓存超时时间
    
    def check_safety_rectangle_collision(self, vehicle_id: str, 
                                       position: Tuple[float, float, float],
                                       safety_params: VehicleSafetyParams = None,
                                       exclude_vehicles: List[str] = None) -> bool:
        """检查安全矩形碰撞"""
        if vehicle_id not in self.env.vehicles:
            return True
        
        vehicle_info = self.env.vehicles[vehicle_id]
        if safety_params is None:
            safety_params = vehicle_info.safety_params
        
        # 创建当前位置的安全矩形
        current_rect = safety_params.create_safety_rectangle(position)
        
        # 检查与环境边界的碰撞
        if self._check_boundary_collision(current_rect):
            return True
        
        # 检查与障碍物的碰撞
        if self._check_obstacle_collision(current_rect):
            return True
        
        # 检查与其他车辆的碰撞
        exclude_vehicles = exclude_vehicles or []
        if vehicle_id not in exclude_vehicles:
            exclude_vehicles.append(vehicle_id)
        
        return self._check_vehicle_collision(current_rect, exclude_vehicles)
    
    def _check_boundary_collision(self, safety_rect: SafetyRectangle) -> bool:
        """检查与环境边界的碰撞"""
        corners = safety_rect.get_corners()
        
        for x, y in corners:
            if x < 0 or x >= self.env.width or y < 0 or y >= self.env.height:
                return True
        
        return False
    
    def _check_obstacle_collision(self, safety_rect: SafetyRectangle) -> bool:
        """检查与障碍物的碰撞"""
        if not hasattr(self.env, 'grid'):
            return False
        
        corners = safety_rect.get_corners()
        
        # 检查矩形覆盖的所有网格点
        min_x = int(min(corner[0] for corner in corners))
        max_x = int(max(corner[0] for corner in corners))
        min_y = int(min(corner[1] for corner in corners))
        max_y = int(max(corner[1] for corner in corners))
        
        for x in range(max(0, min_x), min(self.env.width, max_x + 1)):
            for y in range(max(0, min_y), min(self.env.height, max_y + 1)):
                if self.env.grid[x, y] == 1:  # 障碍物
                    # 检查点是否在安全矩形内
                    if self._point_in_rectangle((x, y), safety_rect):
                        return True
        
        return False
    
    def _check_vehicle_collision(self, safety_rect: SafetyRectangle, 
                               exclude_vehicles: List[str]) -> bool:
        """检查与其他车辆的碰撞"""
        for vehicle_id, vehicle_info in self.env.vehicles.items():
            if vehicle_id in exclude_vehicles:
                continue
            
            # 获取其他车辆的安全矩形
            other_rect = vehicle_info.get_safety_rectangle()
            
            # 检查矩形碰撞
            if safety_rect.intersects(other_rect):
                return True
        
        return False
    
    def _point_in_rectangle(self, point: Tuple[float, float], 
                          rect: SafetyRectangle) -> bool:
        """检查点是否在矩形内"""
        # 简化实现：使用包围盒检查
        corners = rect.get_corners()
        min_x = min(corner[0] for corner in corners)
        max_x = max(corner[0] for corner in corners)
        min_y = min(corner[1] for corner in corners)
        max_y = max(corner[1] for corner in corners)
        
        x, y = point
        return min_x <= x <= max_x and min_y <= y <= max_y
    
    def batch_collision_check(self, vehicle_positions: Dict[str, Tuple]) -> Dict[str, bool]:
        """批量碰撞检测"""
        results = {}
        
        for vehicle_id, position in vehicle_positions.items():
            results[vehicle_id] = self.check_safety_rectangle_collision(
                vehicle_id, position
            )
        
        return results

class OptimizedOpenPitMineEnv:
    """增强版露天矿环境管理器"""
    
    def __init__(self, config=None):
        # 基本配置
        self.width = 500
        self.height = 500
        self.grid_resolution = 1.0
        
        if config:
            self.width = getattr(config, 'width', 500)
            self.height = getattr(config, 'height', 500)
            self.grid_resolution = getattr(config, 'grid_resolution', 1.0)
        
        # 状态管理
        self.state = EnvironmentState.UNINITIALIZED
        self.state_lock = threading.RLock()
        
        # 地图数据
        self.grid = np.zeros((self.width, self.height), dtype=np.uint8)
        self.obstacle_points = []
        
        # 关键点
        self.loading_points = []
        self.unloading_points = []
        self.parking_areas = []
        
        # 增强车辆管理
        self.vehicles = OrderedDict()  # {vehicle_id: EnhancedVehicleInfo}
        
        # 新增：安全碰撞检测器
        self.collision_detector = SafetyCollisionDetector(self)
        
        # 组件引用
        self.components = {}
        
        # 仿真状态
        self.current_time = 0.0
        self.time_step = 0.5
        self.running = False
        self.paused = False
        
        # 增强统计
        self.stats = {
            'total_vehicles': 0,
            'active_vehicles': 0,
            'total_conflicts': 0,
            'resolved_conflicts': 0,
            'safety_violations': 0,
            'average_efficiency': 0.0,
            'total_distance_traveled': 0.0,
            'backbone_utilization': 0.0
        }
        
        # 性能监控
        self.performance_monitor = {
            'collision_checks_per_second': 0,
            'last_performance_update': 0,
            'safety_warnings': []
        }
        
        # 事件回调
        self.changed_callback = None
        self.collision_callback = None
        self.safety_warning_callback = None
        
        logger.info(f"增强环境初始化: {self.width}x{self.height} (安全矩形支持)")
        self.state = EnvironmentState.READY
    
    # ==================== 状态管理 ====================
    
    def set_state(self, new_state: EnvironmentState):
        """设置环境状态"""
        with self.state_lock:
            old_state = self.state
            self.state = new_state
            logger.info(f"环境状态变更: {old_state.value} -> {new_state.value}")
    
    def get_state(self) -> EnvironmentState:
        """获取环境状态"""
        return self.state
    
    def is_ready(self) -> bool:
        """检查是否就绪"""
        return self.state in [EnvironmentState.READY, EnvironmentState.RUNNING, EnvironmentState.PAUSED]
    
    def is_running(self) -> bool:
        """检查是否运行中"""
        return self.state == EnvironmentState.RUNNING and not self.paused
    
    # ==================== 组件管理 ====================
    
    def register_component(self, name: str, component: Any):
        """注册组件"""
        self.components[name] = component
        logger.info(f"注册组件: {name}")
        
        # 为安全检测器设置组件引用
        if name == 'traffic_manager':
            self.collision_detector.traffic_manager = component
    
    def get_component(self, name: str) -> Optional[Any]:
        """获取组件"""
        return self.components.get(name)
    
    def set_backbone_network(self, backbone_network):
        """设置骨干网络"""
        self.register_component('backbone_network', backbone_network)
    
    def set_vehicle_scheduler(self, scheduler):
        """设置调度器"""
        self.register_component('vehicle_scheduler', scheduler)
    
    def set_traffic_manager(self, traffic_manager):
        """设置交通管理器"""
        self.register_component('traffic_manager', traffic_manager)
    
    def set_path_planner(self, path_planner):
        """设置路径规划器"""
        self.register_component('path_planner', path_planner)
    
    # ==================== 地图管理 ====================
    
    def add_obstacle_point(self, x: int, y: int) -> bool:
        """添加障碍物点"""
        try:
            x = max(0, min(int(x), self.width - 1))
            y = max(0, min(int(y), self.height - 1))
            
            self.grid[x, y] = 1
            
            if (x, y) not in self.obstacle_points:
                self.obstacle_points.append((x, y))
            
            self._trigger_changed()
            return True
            
        except Exception as e:
            logger.error(f"添加障碍物失败: {e}")
            return False
    
    def add_obstacle(self, x: int, y: int, width: int = 1, height: int = 1) -> bool:
        """添加矩形障碍物"""
        try:
            success_count = 0
            total_count = width * height
            
            for i in range(x, min(x + width, self.width)):
                for j in range(y, min(y + height, self.height)):
                    if self.add_obstacle_point(i, j):
                        success_count += 1
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"添加障碍物区域失败: {e}")
            return False
    
    def remove_obstacle_point(self, x: int, y: int) -> bool:
        """移除障碍物点"""
        try:
            x = max(0, min(int(x), self.width - 1))
            y = max(0, min(int(y), self.height - 1))
            
            if self.grid[x, y] == 1:
                self.grid[x, y] = 0
                
                if (x, y) in self.obstacle_points:
                    self.obstacle_points.remove((x, y))
                
                self._trigger_changed()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"移除障碍物失败: {e}")
            return False
    
    def clear_obstacles(self):
        """清除所有障碍物"""
        self.grid.fill(0)
        self.obstacle_points.clear()
        self._trigger_changed()
        logger.info("已清除所有障碍物")
    
    # ==================== 关键点管理 ====================
    
    def add_loading_point(self, position: Tuple[float, float, float], capacity: int = 1) -> int:
        """添加装载点"""
        try:
            x, y = float(position[0]), float(position[1])
            theta = float(position[2]) if len(position) > 2 else 0.0
            
            if not (0 <= x < self.width and 0 <= y < self.height):
                raise ValueError(f"坐标超出范围: ({x}, {y})")
            
            point = (x, y, theta)
            self.loading_points.append(point)
            index = len(self.loading_points) - 1
            
            self._trigger_changed()
            logger.info(f"添加装载点 {index}: ({x:.1f}, {y:.1f})")
            return index
            
        except Exception as e:
            logger.error(f"添加装载点失败: {e}")
            return -1
    
    def add_unloading_point(self, position: Tuple[float, float, float], capacity: int = 1) -> int:
        """添加卸载点"""
        try:
            x, y = float(position[0]), float(position[1])
            theta = float(position[2]) if len(position) > 2 else 0.0
            
            if not (0 <= x < self.width and 0 <= y < self.height):
                raise ValueError(f"坐标超出范围: ({x}, {y})")
            
            point = (x, y, theta)
            self.unloading_points.append(point)
            index = len(self.unloading_points) - 1
            
            self._trigger_changed()
            logger.info(f"添加卸载点 {index}: ({x:.1f}, {y:.1f})")
            return index
            
        except Exception as e:
            logger.error(f"添加卸载点失败: {e}")
            return -1
    
    def add_parking_area(self, position: Tuple[float, float, float], capacity: int = 5) -> int:
        """添加停车区"""
        try:
            x, y = float(position[0]), float(position[1])
            theta = float(position[2]) if len(position) > 2 else 0.0
            
            if not (0 <= x < self.width and 0 <= y < self.height):
                raise ValueError(f"坐标超出范围: ({x}, {y})")
            
            point = (x, y, theta)
            self.parking_areas.append(point)
            index = len(self.parking_areas) - 1
            
            self._trigger_changed()
            logger.info(f"添加停车区 {index}: ({x:.1f}, {y:.1f})")
            return index
            
        except Exception as e:
            logger.error(f"添加停车区失败: {e}")
            return -1
    
    # ==================== 增强车辆管理 ====================
    
    def add_vehicle(self, vehicle_id: str, position: Tuple[float, float, float], 
                   goal: Tuple[float, float, float] = None, 
                   vehicle_type: str = "dump_truck", max_load: float = 100,
                   safety_params: Dict = None) -> bool:
        """添加车辆（增强版，完全向后兼容）"""
        try:
            if vehicle_id in self.vehicles:
                logger.warning(f"车辆ID已存在，更新车辆: {vehicle_id}")
                # 兼容模式：如果车辆已存在，更新而不是报错
                return self.update_vehicle_info(vehicle_id, position, goal, vehicle_type, max_load, safety_params)
            
            x, y = float(position[0]), float(position[1])
            theta = float(position[2]) if len(position) > 2 else 0.0
            
            if not (0 <= x < self.width and 0 <= y < self.height):
                logger.warning(f"车辆位置超出范围，自动调整: ({x}, {y}) -> 边界内")
                x = max(10, min(x, self.width - 10))
                y = max(10, min(y, self.height - 10))
            
            # 创建安全参数对象 - 兼容多种输入格式
            if safety_params:
                vehicle_safety_params = VehicleSafetyParams(
                    length=safety_params.get('length', 6.0),
                    width=safety_params.get('width', 3.0),
                    safety_margin=safety_params.get('safety_margin', 1.5),
                    turning_radius=safety_params.get('turning_radius', 8.0),
                    max_speed=safety_params.get('max_speed', 2.0),
                    braking_distance=safety_params.get('braking_distance', 5.0)
                )
            else:
                vehicle_safety_params = VehicleSafetyParams()
            
            # 创建增强车辆信息
            vehicle_info = EnhancedVehicleInfo(
                vehicle_id=vehicle_id,
                position=(x, y, theta),
                initial_position=(x, y, theta),
                goal=goal,
                vehicle_type=vehicle_type,
                max_load=max_load,
                safety_params=vehicle_safety_params
            )
            
            # 确保颜色被正确初始化
            if vehicle_info.color is None:
                vehicle_info.color = QColor(
                    random.randint(100, 255),
                    random.randint(100, 255), 
                    random.randint(100, 255)
                )
            
            # 安全检查（非阻塞）
            try:
                if self.collision_detector.check_safety_rectangle_collision(
                    vehicle_id, (x, y, theta), vehicle_safety_params, [vehicle_id]
                ):
                    logger.warning(f"车辆 {vehicle_id} 初始位置可能不安全，但仍然添加")
                    self.stats['safety_violations'] += 1
            except Exception as safety_check_error:
                logger.debug(f"安全检查失败，跳过: {safety_check_error}")
            
            self.vehicles[vehicle_id] = vehicle_info
            self.stats['total_vehicles'] = len(self.vehicles)
            
            self._trigger_changed()
            logger.info(f"成功添加兼容车辆 {vehicle_id}: ({x:.1f}, {y:.1f})")
            return True
            
        except Exception as e:
            logger.error(f"添加车辆失败 {vehicle_id}: {e}")
            return False
    
    def update_vehicle_info(self, vehicle_id: str, position: Tuple = None, 
                           goal: Tuple = None, vehicle_type: str = None,
                           max_load: float = None, safety_params: Dict = None) -> bool:
        """更新车辆信息（新增兼容方法）"""
        if vehicle_id not in self.vehicles:
            return False
        
        try:
            vehicle_info = self.vehicles[vehicle_id]
            
            if position is not None:
                vehicle_info.position = position
            if goal is not None:
                vehicle_info.goal = goal
            if vehicle_type is not None:
                vehicle_info.vehicle_type = vehicle_type
            if max_load is not None:
                vehicle_info.max_load = max_load
            if safety_params is not None:
                for key, value in safety_params.items():
                    if hasattr(vehicle_info.safety_params, key):
                        setattr(vehicle_info.safety_params, key, value)
            
            logger.debug(f"更新车辆信息: {vehicle_id}")
            return True
        
        except Exception as e:
            logger.error(f"更新车辆信息失败 {vehicle_id}: {e}")
            return False
    
    # 新增：GUI兼容性辅助方法
    def get_vehicle_display_info(self, vehicle_id: str) -> Dict:
        """获取车辆显示信息（GUI专用）"""
        if vehicle_id not in self.vehicles:
            return {}
        
        vehicle_info = self.vehicles[vehicle_id]
        
        # 返回GUI需要的所有信息
        return {
            'vehicle_id': vehicle_id,
            'position': vehicle_info.position,
            'status': vehicle_info.status,
            'load': vehicle_info.current_load,
            'max_load': vehicle_info.max_load,
            'color': vehicle_info.color,
            'path': vehicle_info.path,
            'progress': vehicle_info.progress,
            'completed_cycles': vehicle_info.completed_cycles,
            'vehicle_type': vehicle_info.vehicle_type,
            'goal': vehicle_info.goal,
            # 新增的显示信息
            'efficiency_score': vehicle_info.performance_metrics.efficiency_score,
            'conflict_count': vehicle_info.performance_metrics.conflict_count,
            'safety_params': vehicle_info.safety_params.to_dict(),
            'backbone_usage_ratio': vehicle_info.performance_metrics.backbone_usage_ratio
        }
    
    def get_all_vehicles_display_info(self) -> Dict[str, Dict]:
        """获取所有车辆显示信息（GUI专用）"""
        return {
            vehicle_id: self.get_vehicle_display_info(vehicle_id) 
            for vehicle_id in self.vehicles.keys()
        }
    
    def remove_vehicle(self, vehicle_id: str) -> bool:
        """移除车辆"""
        try:
            if vehicle_id not in self.vehicles:
                return False
            
            del self.vehicles[vehicle_id]
            self.stats['total_vehicles'] = len(self.vehicles)
            
            self._trigger_changed()
            logger.info(f"移除车辆 {vehicle_id}")
            return True
            
        except Exception as e:
            logger.error(f"移除车辆失败 {vehicle_id}: {e}")
            return False
    
    def update_vehicle_position(self, vehicle_id: str, 
                              position: Tuple[float, float, float],
                              time_delta: float = 0.0,
                              check_safety: bool = True) -> bool:
        """更新车辆位置（增强版）"""
        try:
            if vehicle_id not in self.vehicles:
                return False
            
            x, y, theta = float(position[0]), float(position[1]), float(position[2])
            
            # 安全检查
            if check_safety:
                if self.collision_detector.check_safety_rectangle_collision(
                    vehicle_id, (x, y, theta)
                ):
                    logger.warning(f"车辆 {vehicle_id} 位置更新被阻止：安全检查失败")
                    self.stats['safety_violations'] += 1
                    
                    # 触发安全警告回调
                    if self.safety_warning_callback:
                        self.safety_warning_callback(vehicle_id, position, "安全矩形碰撞")
                    
                    return False
            
            vehicle_info = self.vehicles[vehicle_id]
            
            # 更新位置并计算性能指标
            vehicle_info.update_position((x, y, theta), time_delta)
            
            return True
            
        except Exception as e:
            logger.error(f"更新车辆位置失败 {vehicle_id}: {e}")
            return False
    
    def get_vehicle_info(self, vehicle_id: str) -> Optional[EnhancedVehicleInfo]:
        """获取车辆信息"""
        return self.vehicles.get(vehicle_id)
    
    def update_vehicle_safety_params(self, vehicle_id: str, 
                                   safety_params: Dict) -> bool:
        """更新车辆安全参数"""
        if vehicle_id not in self.vehicles:
            return False
        
        try:
            vehicle_info = self.vehicles[vehicle_id]
            
            # 更新安全参数
            for key, value in safety_params.items():
                if hasattr(vehicle_info.safety_params, key):
                    setattr(vehicle_info.safety_params, key, value)
            
            logger.info(f"更新车辆 {vehicle_id} 安全参数")
            return True
            
        except Exception as e:
            logger.error(f"更新安全参数失败 {vehicle_id}: {e}")
            return False
    
    def record_vehicle_conflict(self, vehicle_id: str, resolved: bool = False):
        """记录车辆冲突事件"""
        if vehicle_id in self.vehicles:
            vehicle_info = self.vehicles[vehicle_id]
            vehicle_info.record_conflict(resolved)
            
            self.stats['total_conflicts'] += 1
            if resolved:
                self.stats['resolved_conflicts'] += 1
            
            logger.info(f"记录车辆 {vehicle_id} 冲突事件 (解决: {resolved})")
    
    def record_backbone_switch(self, vehicle_id: str, new_backbone_id: str):
        """记录骨干路径切换"""
        if vehicle_id in self.vehicles:
            vehicle_info = self.vehicles[vehicle_id]
            vehicle_info.record_backbone_switch(new_backbone_id)
            
            logger.info(f"记录车辆 {vehicle_id} 骨干路径切换: {new_backbone_id}")
    
    # ==================== 增强碰撞检测 ====================
    
    def check_collision(self, position: Tuple[float, float, float], 
                       vehicle_dim: Tuple[float, float] = (6, 3),
                       exclude_vehicle: str = None) -> bool:
        """简化碰撞检测（保持向后兼容）"""
        return self.check_collision_enhanced(position, None, exclude_vehicle)
    
    def check_collision_enhanced(self, position: Tuple[float, float, float], 
                               vehicle_params: Dict = None,
                               exclude_vehicle: str = None) -> bool:
        """增强的碰撞检测 - 考虑车辆真实尺寸"""
        try:
            x, y, theta = float(position[0]), float(position[1]), float(position[2])
            
            # 创建安全参数
            if vehicle_params:
                safety_params = VehicleSafetyParams(
                    length=vehicle_params.get('length', 6.0),
                    width=vehicle_params.get('width', 3.0),
                    safety_margin=vehicle_params.get('safety_margin', 1.5),
                    turning_radius=vehicle_params.get('turning_radius', 8.0)
                )
            else:
                safety_params = VehicleSafetyParams()
            
            # 使用安全矩形检测
            return self.collision_detector.check_safety_rectangle_collision(
                exclude_vehicle or "temp_check", 
                position, 
                safety_params,
                [exclude_vehicle] if exclude_vehicle else []
            )
            
        except Exception:
            return True
    
    def batch_collision_check(self, vehicle_positions: Dict[str, Tuple]) -> Dict[str, bool]:
        """批量碰撞检测"""
        return self.collision_detector.batch_collision_check(vehicle_positions)
    
    def get_nearby_vehicles(self, position: Tuple[float, float, float], 
                          radius: float = 20.0) -> List[str]:
        """获取附近的车辆"""
        nearby_vehicles = []
        x, y = position[0], position[1]
        
        for vehicle_id, vehicle_info in self.vehicles.items():
            vx, vy = vehicle_info.position[0], vehicle_info.position[1]
            distance = math.sqrt((x - vx)**2 + (y - vy)**2)
            
            if distance <= radius:
                nearby_vehicles.append(vehicle_id)
        
        return nearby_vehicles
    
    # ==================== 仿真控制 ====================
    
    def start(self):
        """开始仿真"""
        if self.state in [EnvironmentState.READY, EnvironmentState.PAUSED]:
            self.running = True
            self.paused = False
            self.set_state(EnvironmentState.RUNNING)
            logger.info("仿真已开始")
    
    def pause(self):
        """暂停仿真"""
        if self.state == EnvironmentState.RUNNING:
            self.paused = True
            self.set_state(EnvironmentState.PAUSED)
            logger.info("仿真已暂停")
    
    def resume(self):
        """恢复仿真"""
        if self.state == EnvironmentState.PAUSED:
            self.paused = False
            self.set_state(EnvironmentState.RUNNING)
            logger.info("仿真已恢复")
    
    def stop(self):
        """停止仿真"""
        self.running = False
        self.paused = False
        self.set_state(EnvironmentState.READY)
        logger.info("仿真已停止")
    
    def reset(self):
        """重置环境"""
        try:
            self.stop()
            
            # 重置增强车辆状态
            for vehicle_info in self.vehicles.values():
                vehicle_info.position = vehicle_info.initial_position
                vehicle_info.current_load = 0
                vehicle_info.status = 'idle'
                vehicle_info.vehicle_status = VehicleStatus.IDLE
                vehicle_info.path = None
                vehicle_info.path_index = 0
                vehicle_info.progress = 0.0
                vehicle_info.completed_cycles = 0
                vehicle_info.current_task_id = None
                vehicle_info.task_queue.clear()
                vehicle_info.path_structure.clear()
                vehicle_info.backbone_path_id = None
                
                # 重置性能指标
                vehicle_info.performance_metrics = VehiclePerformanceMetrics()
            
            self.current_time = 0.0
            
            # 重置统计
            self.stats.update({
                'total_conflicts': 0,
                'resolved_conflicts': 0,
                'safety_violations': 0,
                'average_efficiency': 0.0,
                'total_distance_traveled': 0.0,
                'backbone_utilization': 0.0
            })
            
            self._trigger_changed()
            logger.info("增强环境已重置")
            return True
            
        except Exception as e:
            logger.error(f"环境重置失败: {e}")
            self.set_state(EnvironmentState.ERROR)
            return False
    
    def update(self, time_delta: float = None):
        """更新环境（增强版）"""
        try:
            if not self.is_running():
                return False
            
            if time_delta is None:
                time_delta = self.time_step
            
            self.current_time += time_delta
            
            # 更新性能统计
            self._update_performance_stats(time_delta)
            
            # 更新性能监控
            self._update_performance_monitoring()
            
            return True
            
        except Exception as e:
            logger.error(f"环境更新失败: {e}")
            return False
    
    def _update_performance_stats(self, time_delta: float):
        """更新性能统计"""
        active_count = 0
        total_efficiency = 0.0
        total_distance = 0.0
        backbone_usage_sum = 0.0
        
        for vehicle_info in self.vehicles.values():
            if vehicle_info.vehicle_status != VehicleStatus.IDLE:
                active_count += 1
            
            # 更新车辆时间
            if vehicle_info.vehicle_status != VehicleStatus.IDLE:
                vehicle_info.performance_metrics.productive_time += time_delta
            else:
                vehicle_info.performance_metrics.idle_time += time_delta
            
            vehicle_info.performance_metrics.total_time += time_delta
            
            # 累积统计
            total_efficiency += vehicle_info.performance_metrics.efficiency_score
            total_distance += vehicle_info.performance_metrics.total_distance
            backbone_usage_sum += vehicle_info.performance_metrics.backbone_usage_ratio
        
        # 更新全局统计
        self.stats['active_vehicles'] = active_count
        
        if len(self.vehicles) > 0:
            self.stats['average_efficiency'] = total_efficiency / len(self.vehicles)
            self.stats['backbone_utilization'] = backbone_usage_sum / len(self.vehicles)
        
        self.stats['total_distance_traveled'] = total_distance
    
    def _update_performance_monitoring(self):
        """更新性能监控"""
        current_time = time.time()
        
        if current_time - self.performance_monitor['last_performance_update'] > 1.0:
            # 计算碰撞检测频率等性能指标
            self.performance_monitor['last_performance_update'] = current_time
            
            # 清理过期的安全警告
            self.performance_monitor['safety_warnings'] = [
                warning for warning in self.performance_monitor['safety_warnings']
                if current_time - warning['timestamp'] < 300  # 5分钟
            ]
    
    # ==================== 文件操作（增强版） ====================
    
    def save_to_file(self, filename: str) -> bool:
        """保存环境到文件（增强版）"""
        try:
            data = {
                "width": self.width,
                "height": self.height,
                "grid_resolution": self.grid_resolution,
                "current_time": self.current_time,
                "running": self.running,
                "paused": self.paused,
                
                # 障碍物
                "obstacles": [
                    {"x": x, "y": y} for x, y in self.obstacle_points
                ],
                
                # 关键点
                "loading_points": [
                    {"x": p[0], "y": p[1], "theta": p[2]} 
                    for p in self.loading_points
                ],
                "unloading_points": [
                    {"x": p[0], "y": p[1], "theta": p[2]} 
                    for p in self.unloading_points
                ],
                "parking_areas": [
                    {"x": p[0], "y": p[1], "theta": p[2]} 
                    for p in self.parking_areas
                ],
                
                # 增强车辆信息
                "vehicles": [
                    {
                        "id": vehicle_id,
                        "x": vehicle_info.position[0],
                        "y": vehicle_info.position[1],
                        "theta": vehicle_info.position[2],
                        "type": vehicle_info.vehicle_type,
                        "max_load": vehicle_info.max_load,
                        "load": vehicle_info.current_load,
                        "status": vehicle_info.status,
                        "completed_cycles": vehicle_info.completed_cycles,
                        "safety_params": vehicle_info.safety_params.to_dict(),
                        "performance_metrics": {
                            "total_distance": vehicle_info.performance_metrics.total_distance,
                            "total_time": vehicle_info.performance_metrics.total_time,
                            "efficiency_score": vehicle_info.performance_metrics.efficiency_score,
                            "conflict_count": vehicle_info.performance_metrics.conflict_count,
                            "backbone_usage_ratio": vehicle_info.performance_metrics.backbone_usage_ratio
                        }
                    }
                    for vehicle_id, vehicle_info in self.vehicles.items()
                ],
                
                # 统计信息
                "statistics": self.stats
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"增强环境已保存到: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"保存环境失败: {e}")
            return False
    
    def load_from_file(self, filename: str) -> bool:
        """从文件加载环境（增强版）"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 更新基本配置
            self.width = int(data.get("width", 500))
            self.height = int(data.get("height", 500))
            self.grid_resolution = float(data.get("grid_resolution", 1.0))
            
            # 重新初始化网格
            self.grid = np.zeros((self.width, self.height), dtype=np.uint8)
            
            # 加载障碍物
            self.obstacle_points = []
            for obstacle in data.get("obstacles", []):
                if isinstance(obstacle, dict):
                    x = obstacle.get("x", 0)
                    y = obstacle.get("y", 0)
                    if "width" in obstacle and "height" in obstacle:
                        self.add_obstacle(x, y, obstacle["width"], obstacle["height"])
                    else:
                        self.add_obstacle_point(x, y)
            
            # 加载关键点
            self._load_special_points(data)
            
            # 加载增强车辆信息
            self.vehicles = OrderedDict()
            for vehicle_data in data.get("vehicles", []):
                self._load_enhanced_vehicle(vehicle_data)
            
            # 加载统计信息
            if "statistics" in data:
                self.stats.update(data["statistics"])
            
            # 设置环境状态
            self.current_time = data.get("current_time", 0.0)
            self.running = data.get("running", False)
            self.paused = data.get("paused", False)
            
            logger.info(f"增强环境加载成功: {len(self.vehicles)} 个车辆")
            return True
            
        except Exception as e:
            logger.error(f"加载环境失败: {e}")
            return False
    
    def _load_special_points(self, data: Dict):
        """加载特殊点"""
        # 装载点
        self.loading_points = []
        for point in data.get("loading_points", []):
            if isinstance(point, dict):
                x, y = point.get("x", 0), point.get("y", 0)
                theta = point.get("theta", 0.0)
                self.add_loading_point((x, y, theta))
            elif isinstance(point, list) and len(point) >= 2:
                row, col = point[0], point[1]
                theta = 0.0 if len(point) <= 2 else point[2]
                self.add_loading_point((col, row, theta))
        
        # 卸载点
        self.unloading_points = []
        for point in data.get("unloading_points", []):
            if isinstance(point, dict):
                x, y = point.get("x", 0), point.get("y", 0)
                theta = point.get("theta", 0.0)
                self.add_unloading_point((x, y, theta))
            elif isinstance(point, list) and len(point) >= 2:
                row, col = point[0], point[1]
                theta = 0.0 if len(point) <= 2 else point[2]
                self.add_unloading_point((col, row, theta))
        
        # 停车区
        self.parking_areas = []
        for point in data.get("parking_areas", []):
            if isinstance(point, dict):
                x, y = point.get("x", 0), point.get("y", 0)
                theta = point.get("theta", 0.0)
                self.add_parking_area((x, y, theta))
            elif isinstance(point, list) and len(point) >= 2:
                row, col = point[0], point[1]
                theta = 0.0 if len(point) <= 2 else point[2]
                capacity = 5 if len(point) <= 3 else point[3]
                self.add_parking_area((col, row, theta), capacity)
    
    def _load_enhanced_vehicle(self, vehicle_data: Dict):
        """加载增强车辆信息 - 完全兼容旧格式"""
        try:
            vehicle_id = str(vehicle_data.get("id", f"v_{len(self.vehicles) + 1}"))
            x = float(vehicle_data.get("x", 0))
            y = float(vehicle_data.get("y", 0))
            theta = float(vehicle_data.get("theta", 0.0))
            v_type = vehicle_data.get("type", "dump_truck")
            max_load = float(vehicle_data.get("max_load", 100))
            
            # 兼容处理安全参数 - 支持旧格式
            safety_params_data = vehicle_data.get("safety_params", {})
            if not safety_params_data:
                # 如果没有安全参数，使用默认值
                safety_params_data = {
                    'length': 6.0,
                    'width': 3.0,
                    'safety_margin': 1.5,
                    'turning_radius': 8.0,
                    'max_speed': 2.0,
                    'braking_distance': 5.0
                }
            
            if self.add_vehicle(vehicle_id, (x, y, theta), None, v_type, max_load, safety_params_data):
                vehicle_info = self.vehicles[vehicle_id]
                
                # 更新基本状态 - 完全兼容旧格式
                vehicle_info.status = vehicle_data.get("status", "idle")
                vehicle_info.current_load = vehicle_data.get("load", 0)  # 注意：兼容'load'键名
                vehicle_info.completed_cycles = vehicle_data.get("completed_cycles", 0)
                
                # 兼容处理颜色信息
                if "color" in vehicle_data:
                    # 如果有颜色信息，保持原有颜色
                    color_data = vehicle_data["color"]
                    if isinstance(color_data, dict):
                        # 处理QColor序列化格式
                        vehicle_info.color = QColor(
                            color_data.get("r", random.randint(100, 255)),
                            color_data.get("g", random.randint(100, 255)),
                            color_data.get("b", random.randint(100, 255))
                        )
                    elif isinstance(color_data, (list, tuple)) and len(color_data) >= 3:
                        vehicle_info.color = QColor(color_data[0], color_data[1], color_data[2])
                
                # 兼容加载性能指标
                perf_data = vehicle_data.get("performance_metrics", {})
                if perf_data:
                    vehicle_info.performance_metrics.total_distance = perf_data.get("total_distance", 0.0)
                    vehicle_info.performance_metrics.total_time = perf_data.get("total_time", 0.0)
                    vehicle_info.performance_metrics.efficiency_score = perf_data.get("efficiency_score", 0.5)
                    vehicle_info.performance_metrics.conflict_count = perf_data.get("conflict_count", 0)
                    vehicle_info.performance_metrics.backbone_usage_ratio = perf_data.get("backbone_usage_ratio", 0.0)
                
                # 兼容处理旧格式的直接属性访问
                for old_key, new_attr in [
                    ("total_distance", "performance_metrics.total_distance"),
                    ("total_time", "performance_metrics.total_time"), 
                    ("efficiency_score", "performance_metrics.efficiency_score"),
                    ("vehicle_length", "safety_params.length"),
                    ("vehicle_width", "safety_params.width"),
                    ("safety_margin", "safety_params.safety_margin")
                ]:
                    if old_key in vehicle_data:
                        # 直接设置到对应的新属性
                        obj, attr = new_attr.split('.')
                        setattr(getattr(vehicle_info, obj), attr, vehicle_data[old_key])
                
                logger.debug(f"✓ 成功加载兼容车辆: {vehicle_id}")
                
        except Exception as e:
            logger.error(f"加载车辆失败 {vehicle_data}: {e}")
            # 提供降级处理
            try:
                # 最小化车辆信息，确保能显示
                vehicle_id = str(vehicle_data.get("id", f"fallback_{len(self.vehicles)}"))
                position = (
                    float(vehicle_data.get("x", 10)),
                    float(vehicle_data.get("y", 10)), 
                    float(vehicle_data.get("theta", 0))
                )
                if self.add_vehicle(vehicle_id, position):
                    logger.info(f"✓ 使用降级模式加载车辆: {vehicle_id}")
            except:
                logger.error(f"车辆降级加载也失败: {vehicle_id}")
    
    def load_from_file(self, filename: str) -> bool:
        """从文件加载环境（增强兼容版）"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 更新基本配置
            self.width = int(data.get("width", 500))
            self.height = int(data.get("height", 500))
            self.grid_resolution = float(data.get("grid_resolution", 1.0))
            
            # 重新初始化网格
            self.grid = np.zeros((self.width, self.height), dtype=np.uint8)
            
            # 加载障碍物
            self.obstacle_points = []
            for obstacle in data.get("obstacles", []):
                if isinstance(obstacle, dict):
                    x = obstacle.get("x", 0)
                    y = obstacle.get("y", 0)
                    if "width" in obstacle and "height" in obstacle:
                        self.add_obstacle(x, y, obstacle["width"], obstacle["height"])
                    else:
                        self.add_obstacle_point(x, y)
            
            # 加载关键点
            self._load_special_points(data)
            
            # 兼容加载车辆信息 - 支持多种格式
            self.vehicles = OrderedDict()
            
            # 优先处理 vehicles_info 字段（新格式）
            for vehicle_data in data.get("vehicles_info", []):
                if isinstance(vehicle_data, dict):
                    self._load_enhanced_vehicle(vehicle_data)
            
            # 然后处理 vehicles 字段（可能是旧格式）
            for vehicle_data in data.get("vehicles", []):
                if isinstance(vehicle_data, dict):
                    vehicle_id = str(vehicle_data.get("id"))
                    if vehicle_id not in self.vehicles:  # 避免重复加载
                        self._load_enhanced_vehicle(vehicle_data)
            
            # 加载统计信息
            if "statistics" in data:
                self.stats.update(data["statistics"])
            
            # 设置环境状态
            self.current_time = data.get("current_time", 0.0)
            self.running = data.get("running", False)
            self.paused = data.get("paused", False)
            
            logger.info(f"兼容环境加载成功: {len(self.vehicles)} 个车辆")
            
            # 验证车辆数据完整性
            self._validate_loaded_vehicles()
            
            return True
            
        except Exception as e:
            logger.error(f"加载环境失败: {e}")
            return False
    
    def _validate_loaded_vehicles(self):
        """验证载入的车辆数据完整性"""
        for vehicle_id, vehicle_info in self.vehicles.items():
            try:
                # 确保所有GUI需要的属性都可访问
                test_access = [
                    vehicle_info['position'],
                    vehicle_info['status'], 
                    vehicle_info['load'],
                    vehicle_info['max_load'],
                    vehicle_info.get('color'),
                    vehicle_info.get('path'),
                    vehicle_info.get('progress', 0.0)
                ]
                logger.debug(f"车辆 {vehicle_id} 数据验证通过")
                
            except Exception as e:
                logger.warning(f"车辆 {vehicle_id} 数据不完整，尝试修复: {e}")
                # 修复缺失的属性
                if not hasattr(vehicle_info, 'color') or vehicle_info.color is None:
                    vehicle_info.color = QColor(
                        random.randint(100, 255),
                        random.randint(100, 255), 
                        random.randint(100, 255)
                    )
                if not hasattr(vehicle_info, 'progress'):
                    vehicle_info.progress = 0.0
    
    # ==================== 工具方法 ====================
    
    def _trigger_changed(self):
        """触发变更回调"""
        if self.changed_callback:
            try:
                self.changed_callback(self)
            except Exception:
                pass
    
    def set_changed_callback(self, callback):
        """设置变更回调"""
        self.changed_callback = callback
    
    def set_collision_callback(self, callback):
        """设置碰撞回调"""
        self.collision_callback = callback
    
    def set_safety_warning_callback(self, callback):
        """设置安全警告回调"""
        self.safety_warning_callback = callback
    
    def get_environment_summary(self):
        """获取环境摘要（增强版）"""
        return {
            'dimensions': f"{self.width}x{self.height}",
            'vehicles': len(self.vehicles),
            'loading_points': len(self.loading_points),
            'unloading_points': len(self.unloading_points),
            'parking_areas': len(self.parking_areas),
            'obstacles': len(self.obstacle_points),
            'state': self.state.value,
            'components': len(self.components),
            'safety_features': {
                'rectangle_collision_detection': True,
                'vehicle_safety_params': True,
                'performance_monitoring': True
            },
            'statistics': self.stats,
            'active_vehicles': self.stats['active_vehicles'],
            'total_conflicts': self.stats['total_conflicts'],
            'safety_violations': self.stats['safety_violations']
        }
    
    def get_performance_report(self) -> Dict:
        """获取性能报告"""
        report = {
            'environment_stats': self.stats.copy(),
            'vehicle_performance': {},
            'safety_analysis': {
                'total_safety_violations': self.stats['safety_violations'],
                'recent_warnings': len(self.performance_monitor['safety_warnings']),
                'collision_detection_active': True
            }
        }
        
        # 车辆性能详情
        for vehicle_id, vehicle_info in self.vehicles.items():
            report['vehicle_performance'][vehicle_id] = {
                'efficiency_score': vehicle_info.performance_metrics.efficiency_score,
                'utilization_rate': vehicle_info.performance_metrics.calculate_utilization_rate(),
                'conflict_count': vehicle_info.performance_metrics.conflict_count,
                'total_distance': vehicle_info.performance_metrics.total_distance,
                'backbone_usage': vehicle_info.performance_metrics.backbone_usage_ratio,
                'stability_score': vehicle_info.performance_metrics.stability_score
            }
        
        return report
    
    def shutdown(self):
        """关闭环境"""
        try:
            self.stop()
            
            # 清理资源
            self.vehicles.clear()
            self.components.clear()
            
            logger.info("增强环境已关闭")
        except Exception as e:
            logger.error(f"环境关闭失败: {e}")

# 向后兼容性
OpenPitMineEnv = OptimizedOpenPitMineEnv