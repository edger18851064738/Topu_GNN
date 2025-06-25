import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
import networkx as nx
from collections import defaultdict, deque
import random
import time
import platform
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set
import heapq
from enum import Enum
import math
import json
import os
from pathlib import Path

# 设置字体 - 解决中文显示问题
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'Arial', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'Arial', 'DejaVu Sans']

class VehicleState(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    WAITING = "waiting"
    CONFIRMED = "confirmed"
    MOVING = "moving"
    BLOCKED = "blocked"
    LOADING = "loading"      # 装载中
    UNLOADING = "unloading"  # 卸载中

class VehicleMode(Enum):
    PARKED = "parked"        # 停车状态，准备前往装载点
    EMPTY = "empty"          # 空载，前往装载点
    LOADED = "loaded"        # 重载，前往卸载点
    RETURNING = "returning"   # 返回停车点

@dataclass
class SpecialPoint:
    """特殊点信息（装载点、卸载点、停车点）"""
    node_id: str
    point_type: str  # "loading", "unloading", "parking"
    is_occupied: bool = False
    reserved_by: Optional[int] = None  # 被哪个车辆预留
    position: Tuple[float, float, float] = None

@dataclass
class NodeFeature:
    """节点特征类"""
    occupancy: float = 0.0
    connectivity: float = 0.0
    congestion: float = 0.0
    centrality: float = 0.0

@dataclass
class NodeReservation:
    """节点预留信息"""
    vehicle_id: int
    start_time: float
    end_time: float
    action: str  # "arrive" or "depart"

@dataclass
class EdgeReservation:
    """边预留信息"""
    vehicle_id: int
    start_time: float
    end_time: float
    direction: Tuple[str, str]

class Stage2TopologyLoader:
    """第二阶段拓扑加载器 - 解析特殊点信息"""
    
    def __init__(self, topology_file_path: str):
        self.topology_file_path = topology_file_path
        self.topology_data = None
        self.graph = None
        self.node_positions = {}
        self.special_points = {}  # 存储所有特殊点
        
    def load_topology(self) -> bool:
        """加载拓扑数据"""
        try:
            print(f"🔄 正在加载第一阶段拓扑结构: {self.topology_file_path}")
            
            with open(self.topology_file_path, 'r', encoding='utf-8') as f:
                self.topology_data = json.load(f)
            
            # 验证数据完整性
            if not self._validate_topology_data():
                return False
            
            # 构建图结构
            if not self._build_graph_from_data():
                return False
            
            # 解析特殊点
            self._parse_special_points()
            
            print(f"✅ 拓扑结构加载成功:")
            print(f"   节点数: {len(self.graph.nodes())}")
            print(f"   边数: {len(self.graph.edges())}")
            print(f"   装载点: {len([p for p in self.special_points.values() if p.point_type == 'loading'])}")
            print(f"   卸载点: {len([p for p in self.special_points.values() if p.point_type == 'unloading'])}")
            print(f"   停车点: {len([p for p in self.special_points.values() if p.point_type == 'parking'])}")
            
            return True
            
        except Exception as e:
            print(f"❌ 拓扑结构加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _validate_topology_data(self) -> bool:
        """验证拓扑数据完整性"""
        if not self.topology_data.get('ready_for_stage2', False):
            print(f"❌ 第一阶段未完成，无法进入第二阶段")
            return False
        return True
    
    def _build_graph_from_data(self) -> bool:
        """从数据构建图结构"""
        self.graph = nx.Graph()
        
        # 策略1: 优先使用增强版数据
        if self._try_build_from_enhanced_data():
            print(f"✅ 使用增强版拓扑数据构建图结构")
            return True
        
        # 策略2: 回退到基本图结构
        if self._try_build_from_basic_graph():
            print(f"✅ 使用基本图结构数据构建图")
            return True
        
        print(f"❌ 无法从任何数据源构建图结构")
        return False
    
    def _try_build_from_enhanced_data(self) -> bool:
        """尝试从增强版数据构建图"""
        if not self.topology_data.get('enhanced_consolidation_applied', False):
            return False
        
        try:
            # 获取关键节点信息
            key_nodes_info = self.topology_data.get('key_nodes_info', {})
            if not key_nodes_info:
                return False
            
            # 添加关键节点
            for node_id, node_info in key_nodes_info.items():
                position = node_info['position']
                self.graph.add_node(node_id)
                self.node_positions[node_id] = (position[0], position[1])
            
            # 获取整合路径信息构建边
            consolidated_paths_info = self.topology_data.get('consolidated_paths_info', {})
            
            for path_id, path_info in consolidated_paths_info.items():
                key_nodes = path_info.get('key_nodes', [])
                
                # 连接相邻的关键节点
                for i in range(len(key_nodes) - 1):
                    node1, node2 = key_nodes[i], key_nodes[i + 1]
                    if node1 in self.graph.nodes() and node2 in self.graph.nodes():
                        # 计算边权重（基于距离）
                        pos1 = self.node_positions[node1]
                        pos2 = self.node_positions[node2]
                        weight = math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
                        weight = max(0.5, min(weight / 10.0, 3.0))  # 归一化权重
                        
                        self.graph.add_edge(node1, node2, weight=weight)
            
            # 检查连通性
            if not nx.is_connected(self.graph):
                print(f"⚠️ 增强版图不连通，尝试修复...")
                self._ensure_graph_connectivity()
            
            return len(self.graph.nodes()) > 0 and len(self.graph.edges()) > 0
            
        except Exception as e:
            print(f"⚠️ 增强版数据构建失败: {e}")
            return False
    
    def _try_build_from_basic_graph(self) -> bool:
        """尝试从基本图结构构建"""
        graph_nodes = self.topology_data.get('graph_nodes', [])
        graph_edges = self.topology_data.get('graph_edges', [])
        position_mapping = self.topology_data.get('position_mapping', {})
        
        if not graph_nodes:
            return False
        
        try:
            # 添加节点
            for node in graph_nodes:
                node_str = str(node)
                self.graph.add_node(node_str)
                
                # 查找位置信息
                if node_str in position_mapping:
                    pos = position_mapping[node_str]
                    self.node_positions[node_str] = (pos[0], pos[1])
                else:
                    # 生成随机位置
                    self.node_positions[node_str] = (
                        random.uniform(0, 100), 
                        random.uniform(0, 100)
                    )
            
            # 添加边
            for edge in graph_edges:
                node1, node2 = str(edge[0]), str(edge[1])
                if node1 in self.graph.nodes() and node2 in self.graph.nodes():
                    pos1 = self.node_positions[node1]
                    pos2 = self.node_positions[node2]
                    weight = math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
                    weight = max(0.5, min(weight / 10.0, 3.0))
                    
                    self.graph.add_edge(node1, node2, weight=weight)
            
            return len(self.graph.nodes()) > 0
            
        except Exception as e:
            print(f"⚠️ 基本图结构构建失败: {e}")
            return False
    
    def _ensure_graph_connectivity(self):
        """确保图的连通性"""
        if not self.graph.nodes():
            return
        
        # 检查连通分量
        components = list(nx.connected_components(self.graph))
        
        if len(components) <= 1:
            return  # 已经连通
        
        print(f"   修复图连通性: {len(components)} 个连通分量")
        
        # 连接各个分量
        main_component = max(components, key=len)
        
        for component in components:
            if component == main_component:
                continue
            
            # 找到最近的节点对
            min_distance = float('inf')
            best_pair = None
            
            for node1 in main_component:
                for node2 in component:
                    pos1 = self.node_positions[node1]
                    pos2 = self.node_positions[node2]
                    distance = math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_pair = (node1, node2)
            
            # 添加连接边
            if best_pair:
                weight = max(0.5, min(min_distance / 10.0, 3.0))
                self.graph.add_edge(best_pair[0], best_pair[1], weight=weight)
                main_component.update(component)
    
    def _parse_special_points(self):
        """解析特殊点信息"""
        self.special_points = {}
        
        # 从key_nodes_info中解析特殊点
        key_nodes_info = self.topology_data.get('key_nodes_info', {})
        
        for node_id, node_info in key_nodes_info.items():
            position = node_info.get('position', [0, 0, 0])
            
            # 根据节点ID前缀判断特殊点类型
            if 'L' in node_id and ('_to_' in node_id):
                # 解析装载点ID，例如：endpoint_start_L0_to_U0 -> L0
                point_type = 'loading'
                # 提取L后面的数字
                import re
                match = re.search(r'L(\d+)', node_id)
                if match:
                    point_id = f"L{match.group(1)}"
                else:
                    point_id = node_id
                    
            elif 'U' in node_id and ('_to_' in node_id):
                # 解析卸载点ID，例如：endpoint_end_L0_to_U0 -> U0
                point_type = 'unloading'
                import re
                match = re.search(r'U(\d+)', node_id)
                if match:
                    point_id = f"U{match.group(1)}"
                else:
                    point_id = node_id
                    
            elif 'P' in node_id:
                # 停车点
                point_type = 'parking'
                import re
                match = re.search(r'P(\d+)', node_id)
                if match:
                    point_id = f"P{match.group(1)}"
                else:
                    point_id = node_id
            else:
                continue  # 跳过非特殊点
            
            # 创建特殊点对象
            special_point = SpecialPoint(
                node_id=node_id,
                point_type=point_type,
                position=(position[0], position[1], position[2])
            )
            
            self.special_points[point_id] = special_point
        
        print(f"🎯 解析特殊点完成:")
        loading_points = [p for p in self.special_points.keys() if p.startswith('L')]
        unloading_points = [p for p in self.special_points.keys() if p.startswith('U')]
        parking_points = [p for p in self.special_points.keys() if p.startswith('P')]
        
        print(f"   装载点: {sorted(loading_points)}")
        print(f"   卸载点: {sorted(unloading_points)}")
        print(f"   停车点: {sorted(parking_points)}")
    
    def get_graph(self) -> nx.Graph:
        """获取构建的图"""
        return self.graph
    
    def get_node_positions(self) -> Dict[str, Tuple[float, float]]:
        """获取节点位置"""
        return self.node_positions.copy()
    
    def get_special_points(self) -> Dict[str, SpecialPoint]:
        """获取特殊点信息"""
        return self.special_points.copy()

class Stage2RoadNetwork:
    """第二阶段道路网络类 - 基于第一阶段导出的拓扑"""
    
    def __init__(self, topology_file_path: str = None, num_vehicles: int = 6):
        self.topology_file_path = topology_file_path
        self.num_vehicles = num_vehicles
        self.topology_loader = None
        self.graph = nx.Graph()
        self.node_positions = {}
        
        # 特殊点管理
        self.loading_points = {}    # point_id -> SpecialPoint
        self.unloading_points = {}  # point_id -> SpecialPoint  
        self.parking_points = {}    # point_id -> SpecialPoint
        
        # 预留和管理系统
        self.edge_reservations = defaultdict(list)
        self.node_reservations = defaultdict(list)
        self.node_occupancy = defaultdict(set)
        self.node_features = {}
        self.global_time = 0.0
        
        if topology_file_path:
            self._load_topology_from_file()
        else:
            self._create_fallback_topology()
        
        self._setup_special_points()
        self._initialize_features()
    
    def _load_topology_from_file(self):
        """从文件加载拓扑"""
        self.topology_loader = Stage2TopologyLoader(self.topology_file_path)
        
        if self.topology_loader.load_topology():
            self.graph = self.topology_loader.get_graph()
            self.node_positions = self.topology_loader.get_node_positions()
            
            print(f"🎯 第二阶段网络构建成功:")
            print(f"   节点: {len(self.graph.nodes())}")
            print(f"   边: {len(self.graph.edges())}")
        else:
            print(f"⚠️ 拓扑加载失败，使用回退拓扑")
            self._create_fallback_topology()
    
    def _create_fallback_topology(self):
        """创建回退拓扑"""
        print(f"🔄 创建回退网络拓扑...")
        
        # 创建一个基本的不规则网络
        self.graph = nx.Graph()
        self.node_positions = {}
        
        # 生成节点
        num_nodes = 20
        for i in range(num_nodes):
            node_id = f"fallback_node_{i}"
            x = random.uniform(0, 80) + random.gauss(0, 5)
            y = random.uniform(0, 60) + random.gauss(0, 5)
            x = max(0, min(80, x))
            y = max(0, min(60, y))
            
            self.graph.add_node(node_id)
            self.node_positions[node_id] = (x, y)
        
        # 生成边
        nodes = list(self.graph.nodes())
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i+1:], i+1):
                pos1 = self.node_positions[node1]
                pos2 = self.node_positions[node2]
                dist = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                
                # 连接较近的节点
                if dist < 20 and self.graph.degree(node1) < 4 and self.graph.degree(node2) < 4:
                    weight = max(0.5, min(2.0, dist/10.0))
                    self.graph.add_edge(node1, node2, weight=weight)
        
        # 确保连通性
        if not nx.is_connected(self.graph):
            components = list(nx.connected_components(self.graph))
            main_component = max(components, key=len)
            
            for component in components:
                if component == main_component:
                    continue
                
                # 连接到主分量
                min_dist = float('inf')
                best_pair = None
                
                for node1 in main_component:
                    for node2 in component:
                        pos1 = self.node_positions[node1]
                        pos2 = self.node_positions[node2]
                        dist = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                        if dist < min_dist:
                            min_dist = dist
                            best_pair = (node1, node2)
                
                if best_pair:
                    weight = max(0.5, min(2.0, min_dist/10.0))
                    self.graph.add_edge(best_pair[0], best_pair[1], weight=weight)
    
    def _setup_special_points(self):
        """设置特殊点"""
        if self.topology_loader and self.topology_loader.special_points:
            # 使用拓扑文件中的特殊点
            special_points = self.topology_loader.get_special_points()
            
            for point_id, special_point in special_points.items():
                if special_point.point_type == 'loading':
                    self.loading_points[point_id] = special_point
                elif special_point.point_type == 'unloading':
                    self.unloading_points[point_id] = special_point
                elif special_point.point_type == 'parking':
                    self.parking_points[point_id] = special_point
        else:
            # 创建回退特殊点
            self._create_fallback_special_points()
        
        print(f"🚛 特殊点设置完成:")
        print(f"   装载点: {list(self.loading_points.keys())}")
        print(f"   卸载点: {list(self.unloading_points.keys())}")
        print(f"   停车点: {list(self.parking_points.keys())}")
    
    def _create_fallback_special_points(self):
        """创建回退特殊点"""
        all_nodes = list(self.graph.nodes())
        
        # 确保有足够的节点
        if len(all_nodes) < self.num_vehicles * 3:
            print(f"⚠️ 节点数量不足，创建最小特殊点集合")
        
        selected_nodes = random.sample(all_nodes, min(len(all_nodes), self.num_vehicles * 3))
        
        # 平均分配特殊点
        points_per_type = len(selected_nodes) // 3
        
        # 创建装载点
        for i in range(min(points_per_type, self.num_vehicles)):
            node_id = selected_nodes[i]
            point_id = f"L{i}"
            position = self.node_positions.get(node_id, (0, 0))
            
            special_point = SpecialPoint(
                node_id=node_id,
                point_type='loading',
                position=(position[0], position[1], 0)
            )
            self.loading_points[point_id] = special_point
        
        # 创建卸载点
        start_idx = points_per_type
        for i in range(min(points_per_type, self.num_vehicles)):
            if start_idx + i >= len(selected_nodes):
                break
            node_id = selected_nodes[start_idx + i]
            point_id = f"U{i}"
            position = self.node_positions.get(node_id, (0, 0))
            
            special_point = SpecialPoint(
                node_id=node_id,
                point_type='unloading',
                position=(position[0], position[1], 0)
            )
            self.unloading_points[point_id] = special_point
        
        # 创建停车点
        start_idx = points_per_type * 2
        for i in range(min(points_per_type, self.num_vehicles)):
            if start_idx + i >= len(selected_nodes):
                break
            node_id = selected_nodes[start_idx + i]
            point_id = f"P{i}"
            position = self.node_positions.get(node_id, (0, 0))
            
            special_point = SpecialPoint(
                node_id=node_id,
                point_type='parking',
                position=(position[0], position[1], 0)
            )
            self.parking_points[point_id] = special_point
    
    # ============ 特殊点管理方法 ============
    
    def get_available_loading_point(self, exclude_vehicle: int = -1) -> Optional[str]:
        """获取可用的装载点"""
        for point_id, point in self.loading_points.items():
            if not point.is_occupied and (point.reserved_by is None or point.reserved_by == exclude_vehicle):
                return point_id
        return None
    
    def get_available_unloading_point(self, exclude_vehicle: int = -1) -> Optional[str]:
        """获取可用的卸载点"""
        for point_id, point in self.unloading_points.items():
            if not point.is_occupied and (point.reserved_by is None or point.reserved_by == exclude_vehicle):
                return point_id
        return None
    
    def get_available_parking_point(self, exclude_vehicle: int = -1) -> Optional[str]:
        """获取可用的停车点"""
        for point_id, point in self.parking_points.items():
            if not point.is_occupied and (point.reserved_by is None or point.reserved_by == exclude_vehicle):
                return point_id
        return None
    
    def reserve_special_point(self, point_id: str, vehicle_id: int) -> bool:
        """预留特殊点"""
        # 查找点的类型
        point = None
        if point_id in self.loading_points:
            point = self.loading_points[point_id]
        elif point_id in self.unloading_points:
            point = self.unloading_points[point_id]
        elif point_id in self.parking_points:
            point = self.parking_points[point_id]
        
        if point and not point.is_occupied and point.reserved_by is None:
            point.reserved_by = vehicle_id
            print(f"🎯 {point.point_type.title()} point {point_id} reserved by vehicle V{vehicle_id}")
            return True
        return False
    
    def occupy_special_point(self, point_id: str, vehicle_id: int):
        """占用特殊点"""
        point = None
        if point_id in self.loading_points:
            point = self.loading_points[point_id]
        elif point_id in self.unloading_points:
            point = self.unloading_points[point_id]
        elif point_id in self.parking_points:
            point = self.parking_points[point_id]
        
        if point:
            point.is_occupied = True
            point.reserved_by = vehicle_id
    
    def release_special_point(self, point_id: str):
        """释放特殊点"""
        point = None
        if point_id in self.loading_points:
            point = self.loading_points[point_id]
        elif point_id in self.unloading_points:
            point = self.unloading_points[point_id]
        elif point_id in self.parking_points:
            point = self.parking_points[point_id]
        
        if point:
            point.is_occupied = False
            point.reserved_by = None
            print(f"✅ {point.point_type.title()} point {point_id} released")
    
    def cancel_point_reservations(self, vehicle_id: int):
        """取消车辆的特殊点预留"""
        for point in self.loading_points.values():
            if point.reserved_by == vehicle_id and not point.is_occupied:
                point.reserved_by = None
        
        for point in self.unloading_points.values():
            if point.reserved_by == vehicle_id and not point.is_occupied:
                point.reserved_by = None
        
        for point in self.parking_points.values():
            if point.reserved_by == vehicle_id and not point.is_occupied:
                point.reserved_by = None
    
    def get_point_node_id(self, point_id: str) -> Optional[str]:
        """获取特殊点对应的图节点ID"""
        point = None
        if point_id in self.loading_points:
            point = self.loading_points[point_id]
        elif point_id in self.unloading_points:
            point = self.unloading_points[point_id]
        elif point_id in self.parking_points:
            point = self.parking_points[point_id]
        
        return point.node_id if point else None
    
    # ============ 其他方法保持不变 ============
    
    def _initialize_features(self):
        """初始化节点特征"""
        for node in self.graph.nodes():
            self.node_features[node] = NodeFeature()
            self._update_node_feature(node)
    
    def _update_node_feature(self, node: str):
        """更新单个节点特征"""
        neighbors = list(self.graph.neighbors(node))
        connectivity = len(neighbors)
        
        # 拥堵度计算
        congestion = 0.0
        current_time = self.global_time
        
        for neighbor in neighbors:
            edge_key = tuple(sorted([node, neighbor]))
            reservations = self.edge_reservations[edge_key]
            future_occupied = any(
                r.start_time <= current_time + 2.0 and r.end_time >= current_time
                for r in reservations
            )
            if future_occupied:
                congestion += 1
        
        congestion = congestion / max(len(neighbors), 1) if neighbors else 0
        
        # 中心度
        centrality = connectivity / 8.0
        
        # 占用度
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
            active_reservations = [
                r for r in reservations 
                if r.end_time > current_time
            ]
            if active_reservations:
                self.edge_reservations[edge_key] = active_reservations
            else:
                del self.edge_reservations[edge_key]
        
        # 清理过期的节点预留
        for node_id in list(self.node_reservations.keys()):
            reservations = self.node_reservations[node_id]
            active_reservations = [
                r for r in reservations 
                if r.end_time > current_time
            ]
            if active_reservations:
                self.node_reservations[node_id] = active_reservations
            else:
                del self.node_reservations[node_id]
    
    def reserve_edge(self, from_node: str, to_node: str, vehicle_id: int, 
                    start_time: float, duration: float) -> bool:
        """预留边使用权"""
        edge_key = tuple(sorted([from_node, to_node]))
        end_time = start_time + duration
        
        safety_buffer = 0.15
        precision_buffer = 0.01
        
        existing_reservations = self.edge_reservations[edge_key]
        for reservation in existing_reservations:
            if not (end_time + safety_buffer + precision_buffer <= reservation.start_time or 
                   start_time >= reservation.end_time + safety_buffer + precision_buffer):
                return False
        
        new_reservation = EdgeReservation(
            vehicle_id=vehicle_id,
            start_time=start_time,
            end_time=end_time,
            direction=(from_node, to_node)
        )
        self.edge_reservations[edge_key].append(new_reservation)
        return True
    
    def reserve_node(self, node: str, vehicle_id: int, 
                    start_time: float, duration: float) -> bool:
        """预留节点使用权"""
        end_time = start_time + duration
        
        safety_buffer = 0.3
        precision_buffer = 0.01
        
        existing_reservations = self.node_reservations[node]
        for reservation in existing_reservations:
            if not (end_time + safety_buffer + precision_buffer <= reservation.start_time or 
                   start_time >= reservation.end_time + safety_buffer + precision_buffer):
                return False
        
        new_reservation = NodeReservation(
            vehicle_id=vehicle_id,
            start_time=start_time,
            end_time=end_time,
            action="occupy"
        )
        self.node_reservations[node].append(new_reservation)
        return True
    
    def is_node_available(self, node: str, start_time: float, duration: float, 
                         exclude_vehicle: int = -1) -> bool:
        """检查节点在指定时间段是否可用"""
        end_time = start_time + duration
        
        safety_buffer = 0.3
        precision_buffer = 0.01
        
        reservations = self.node_reservations.get(node, [])
        for reservation in reservations:
            if reservation.vehicle_id == exclude_vehicle:
                continue
            
            if not (end_time + safety_buffer + precision_buffer <= reservation.start_time or 
                   start_time >= reservation.end_time + safety_buffer + precision_buffer):
                return False
        
        return True
    
    def is_edge_available(self, from_node: str, to_node: str, 
                         start_time: float, duration: float, 
                         exclude_vehicle: int = -1) -> bool:
        """检查边在指定时间段是否可用"""
        edge_key = tuple(sorted([from_node, to_node]))
        end_time = start_time + duration
        
        safety_buffer = 0.15
        precision_buffer = 0.01
        
        reservations = self.edge_reservations.get(edge_key, [])
        for reservation in reservations:
            if reservation.vehicle_id == exclude_vehicle:
                continue
            
            if not (end_time + safety_buffer + precision_buffer <= reservation.start_time or 
                   start_time >= reservation.end_time + safety_buffer + precision_buffer):
                return False
        
        return True
    
    def cancel_reservations(self, vehicle_id: int):
        """取消车辆的所有预留（边和节点）"""
        # 取消边预留
        for edge_key in list(self.edge_reservations.keys()):
            reservations = self.edge_reservations[edge_key]
            remaining = [r for r in reservations if r.vehicle_id != vehicle_id]
            if remaining:
                self.edge_reservations[edge_key] = remaining
            else:
                del self.edge_reservations[edge_key]
        
        # 取消节点预留
        for node_id in list(self.node_reservations.keys()):
            reservations = self.node_reservations[node_id]
            remaining = [r for r in reservations if r.vehicle_id != vehicle_id]
            if remaining:
                self.node_reservations[node_id] = remaining
            else:
                del self.node_reservations[node_id]
    
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
        heap = [(0, start, [start], [current_time])]
        visited = {}
        
        while heap:
            cost, current, path, times = heapq.heappop(heap)
            current_arrival_time = times[-1]
            
            if current in visited and visited[current] <= current_arrival_time:
                continue
            visited[current] = current_arrival_time
            
            if current == end:
                return path, times
            
            for neighbor in self.graph.neighbors(current):
                if neighbor in path:
                    continue
                
                travel_time = self._compute_travel_time(current, neighbor, vehicle_id)
                departure_time = current_arrival_time + 0.1
                arrival_time = departure_time + travel_time
                
                if not self.is_edge_available(current, neighbor, departure_time, 
                                            travel_time, exclude_vehicle=vehicle_id):
                    wait_time = self._find_next_available_time(
                        current, neighbor, departure_time, travel_time, vehicle_id)
                    if wait_time > departure_time + 5.0:
                        continue
                    departure_time = wait_time
                    arrival_time = departure_time + travel_time
                
                edge_cost = self._compute_gnn_edge_cost(current, neighbor, departure_time)
                heuristic = self._compute_heuristic(neighbor, end)
                total_cost = arrival_time + edge_cost + heuristic
                
                new_path = path + [neighbor]
                new_times = times + [arrival_time]
                heapq.heappush(heap, (total_cost, neighbor, new_path, new_times))
        
        return [], []
    
    def _compute_travel_time(self, from_node: str, to_node: str, vehicle_id: int) -> float:
        """计算移动时间"""
        base_time = self.graph[from_node][to_node].get('weight', 1.0)
        to_feature = self.node_features[to_node]
        time_factor = 1.0 + to_feature.congestion * 0.5 + to_feature.occupancy * 0.3
        return base_time * time_factor
    
    def _find_next_available_time(self, from_node: str, to_node: str, 
                                 earliest_start: float, duration: float, 
                                 vehicle_id: int) -> float:
        """找到边的下一个可用时间"""
        edge_key = tuple(sorted([from_node, to_node]))
        reservations = self.edge_reservations.get(edge_key, [])
        
        other_reservations = [r for r in reservations if r.vehicle_id != vehicle_id]
        
        if not other_reservations:
            return earliest_start
        
        other_reservations.sort(key=lambda x: x.start_time)
        
        current_time = earliest_start
        for reservation in other_reservations:
            if current_time + duration <= reservation.start_time:
                return current_time
            current_time = max(current_time, reservation.end_time)
        
        return current_time
    
    def _compute_gnn_edge_cost(self, from_node: str, to_node: str, time: float) -> float:
        """计算GNN增强的边权重"""
        base_weight = self.graph[from_node][to_node].get('weight', 1.0)
        to_feature = self.node_features[to_node]
        
        cost = base_weight
        cost += to_feature.occupancy * 3.0
        cost += to_feature.congestion * 8.0
        cost -= to_feature.centrality * 1.0
        
        # 时间相关的动态权重
        edge_key = tuple(sorted([from_node, to_node]))
        future_congestion = len([
            r for r in self.edge_reservations.get(edge_key, [])
            if r.start_time <= time + 3.0 and r.end_time >= time
        ])
        cost += future_congestion * 2.0
        
        return max(cost, 0.1)
    
    def _compute_heuristic(self, node1: str, node2: str) -> float:
        """计算启发式距离"""
        pos1 = self.node_positions[node1]
        pos2 = self.node_positions[node2]
        return np.linalg.norm(np.array(pos1) - np.array(pos2)) * 0.3
    
    def simple_pathfinding(self, start: str, end: str) -> List[str]:
        """简单最短路径规划"""
        try:
            return nx.shortest_path(self.graph, start, end, weight='weight')
        except nx.NetworkXNoPath:
            return []

class Vehicle:
    """车辆智能体类 - 支持完整的停车点循环作业"""
    
    def __init__(self, vehicle_id: int, start_parking_point: str, road_network: Stage2RoadNetwork, use_gnn: bool = True):
        self.id = vehicle_id
        self.road_network = road_network
        self.use_gnn = use_gnn
        
        # 作业模式和目标
        self.mode = VehicleMode.PARKED  # 开始时在停车点
        self.current_parking_point = start_parking_point
        self.target_loading_point = None
        self.target_unloading_point = None
        self.target_parking_point = None
        
        # 获取起始节点ID
        start_node_id = self.road_network.get_point_node_id(start_parking_point)
        if not start_node_id:
            # 回退到第一个可用节点
            start_node_id = list(road_network.graph.nodes())[0]
        
        self.current_node = start_node_id
        
        # 路径规划
        self.path = []
        self.path_times = []
        self.path_index = 0
        
        # 物理状态
        self.position = np.array(road_network.node_positions[self.current_node], dtype=float)
        self.target_position = self.position.copy()
        self.progress = 0.0
        self.speed = 0.6 + random.random() * 0.4
        
        # 车辆状态
        self.state = VehicleState.IDLE
        self.move_start_time = 0.0
        self.move_duration = 0.0
        self.wait_until = 0.0
        self.path_start_time = 0.0
        self.retry_count = 0
        self.max_retries = 3
        self.path_confirmed = False
        
        # 作业统计
        self.total_distance = 0.0
        self.completed_cycles = 0  # 完成的完整循环次数
        self.wait_time = 0.0
        self.loading_time = 2.0    # 装载耗时
        self.unloading_time = 1.5  # 卸载耗时
        self.parking_time = 0.5    # 停车耗时
        self.operation_start_time = 0.0
        
        # 颜色
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan', 'lime', 'magenta']
        self.color = colors[vehicle_id % len(colors)]
        
        # 占用起始停车点
        self.road_network.occupy_special_point(start_parking_point, self.id)
        self.road_network.add_vehicle_to_node(self.id, self.current_node)
    
    @property
    def target_point_id(self):
        """动态目标点ID - 根据模式返回相应的目标"""
        if self.mode == VehicleMode.PARKED:
            return self.target_loading_point
        elif self.mode == VehicleMode.EMPTY:
            return self.target_loading_point
        elif self.mode == VehicleMode.LOADED:
            return self.target_unloading_point
        elif self.mode == VehicleMode.RETURNING:
            return self.target_parking_point
        return None
    
    @property
    def target_node_id(self):
        """动态目标节点ID"""
        target_point = self.target_point_id
        if target_point:
            return self.road_network.get_point_node_id(target_point)
        return None
    
    def update(self, current_time: float, dt: float):
        """主更新函数 - 完整的停车点循环作业逻辑"""
        if self.state == VehicleState.IDLE:
            self._plan_next_task(current_time)
        elif self.state == VehicleState.PLANNING:
            self._execute_planning(current_time)
        elif self.state == VehicleState.WAITING:
            if current_time >= self.wait_until:
                self.state = VehicleState.IDLE
                self.retry_count += 1
                if self.retry_count > self.max_retries:
                    self._reset_current_task()
                    self.retry_count = 0
            else:
                self.wait_time += dt
        elif self.state == VehicleState.CONFIRMED:
            if current_time >= self.path_start_time:
                self._start_confirmed_path(current_time)
        elif self.state == VehicleState.MOVING:
            self._update_movement(current_time, dt)
        elif self.state == VehicleState.LOADING:
            self._update_loading(current_time)
        elif self.state == VehicleState.UNLOADING:
            self._update_unloading(current_time)
        elif self.state == VehicleState.BLOCKED:
            self.road_network.cancel_reservations(self.id)
            self.road_network.cancel_point_reservations(self.id)
            self.state = VehicleState.IDLE
            self.retry_count += 1
    
    def _plan_next_task(self, current_time: float):
        """规划下一个任务"""
        if self.mode == VehicleMode.PARKED:
            # 在停车点，寻找装载点
            available_loading = self.road_network.get_available_loading_point(exclude_vehicle=self.id)
            if available_loading:
                if self.road_network.reserve_special_point(available_loading, self.id):
                    # 释放当前停车点
                    if self.current_parking_point:
                        self.road_network.release_special_point(self.current_parking_point)
                        self.current_parking_point = None
                    
                    self.target_loading_point = available_loading
                    self.mode = VehicleMode.EMPTY
                    self._plan_path_to_target(current_time)
                else:
                    self._wait_and_retry(current_time)
            else:
                self._wait_and_retry(current_time)
        
        elif self.mode == VehicleMode.EMPTY:
            # 空载状态，已在前往装载点的路上，这里不应该被调用
            # 如果被调用，说明目标丢失，重新规划
            if not self.target_loading_point:
                self.mode = VehicleMode.PARKED
                self._plan_next_task(current_time)
        
        elif self.mode == VehicleMode.LOADED:
            # 重载状态，寻找卸载点
            available_unloading = self.road_network.get_available_unloading_point(exclude_vehicle=self.id)
            if available_unloading:
                if self.road_network.reserve_special_point(available_unloading, self.id):
                    self.target_unloading_point = available_unloading
                    self._plan_path_to_target(current_time)
                else:
                    self._wait_and_retry(current_time)
            else:
                self._wait_and_retry(current_time)
        
        elif self.mode == VehicleMode.RETURNING:
            # 返回模式，寻找停车点
            available_parking = self.road_network.get_available_parking_point(exclude_vehicle=self.id)
            if available_parking:
                if self.road_network.reserve_special_point(available_parking, self.id):
                    self.target_parking_point = available_parking
                    self._plan_path_to_target(current_time)
                else:
                    self._wait_and_retry(current_time)
            else:
                self._wait_and_retry(current_time)
    
    def _plan_path_to_target(self, current_time: float):
        """规划到目标点的路径"""
        target_node = self.target_node_id
        if not target_node:
            self._wait_and_retry(current_time)
            return
        
        self.state = VehicleState.PLANNING
        self.road_network.cancel_reservations(self.id)
        
        if self.use_gnn:
            self.path, self.path_times = self.road_network.gnn_pathfinding_with_reservation(
                self.current_node, target_node, self.id, current_time)
        else:
            simple_path = self.road_network.simple_pathfinding(self.current_node, target_node)
            self.path = simple_path
            if simple_path:
                self.path_times = []
                current_t = current_time + 0.5
                
                for i, node in enumerate(simple_path):
                    if i == 0:
                        self.path_times.append(current_t)
                    else:
                        prev_node = simple_path[i-1]
                        travel_time = self.road_network._compute_travel_time(prev_node, node, self.id)
                        current_t += travel_time
                        self.path_times.append(current_t)
            else:
                self.path_times = []
        
        mode_text = f"{self.mode.value.title()}->{self.target_point_id}"
        print(f"Vehicle {self.id} ({mode_text}): Planned path {self.path}")
    
    def _wait_and_retry(self, current_time: float):
        """等待并重试"""
        self.wait_until = current_time + 1.0 + random.random()
        self.state = VehicleState.WAITING
    
    def _reset_current_task(self):
        """重置当前任务"""
        self.road_network.cancel_point_reservations(self.id)
        self.target_loading_point = None
        self.target_unloading_point = None
        self.target_parking_point = None
        self.retry_count = 0
    
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
        
        print(f"Vehicle {self.id}: Arrived at {self.current_node}")
        
        # 检查是否到达目标特殊点
        if self.path_index + 1 >= len(self.path):
            # 路径完成，检查是否到达目标特殊点
            target_point = self.target_point_id
            target_node = self.target_node_id
            
            if target_node and self.current_node == target_node:
                if self.mode == VehicleMode.EMPTY and target_point == self.target_loading_point:
                    # 到达装载点，开始装载
                    self._start_loading(current_time)
                elif self.mode == VehicleMode.LOADED and target_point == self.target_unloading_point:
                    # 到达卸载点，开始卸载
                    self._start_unloading(current_time)
                elif self.mode == VehicleMode.RETURNING and target_point == self.target_parking_point:
                    # 到达停车点，开始停车
                    self._start_parking(current_time)
                else:
                    # 其他情况，回到idle状态
                    self.state = VehicleState.IDLE
                    self.path_confirmed = False
            else:
                # 没有到达正确的目标，回到idle状态
                self.state = VehicleState.IDLE
                self.path_confirmed = False
        else:
            # 继续路径
            self._start_next_move(current_time)
    
    def _start_loading(self, current_time: float):
        """开始装载作业"""
        self.road_network.occupy_special_point(self.target_loading_point, self.id)
        self.state = VehicleState.LOADING
        self.operation_start_time = current_time
        print(f"🔄 Vehicle {self.id}: Starting loading at {self.target_loading_point}")
    
    def _start_unloading(self, current_time: float):
        """开始卸载作业"""
        self.road_network.occupy_special_point(self.target_unloading_point, self.id)
        self.state = VehicleState.UNLOADING
        self.operation_start_time = current_time
        print(f"🔄 Vehicle {self.id}: Starting unloading at {self.target_unloading_point}")
    
    def _start_parking(self, current_time: float):
        """开始停车"""
        self.road_network.occupy_special_point(self.target_parking_point, self.id)
        self.current_parking_point = self.target_parking_point
        self.target_parking_point = None
        self.mode = VehicleMode.PARKED
        self.state = VehicleState.IDLE
        self.completed_cycles += 1
        print(f"🅿️ Vehicle {self.id}: Parked at {self.current_parking_point}, completed cycle {self.completed_cycles}")
    
    def _update_loading(self, current_time: float):
        """更新装载状态"""
        if current_time - self.operation_start_time >= self.loading_time:
            # 装载完成
            self.road_network.release_special_point(self.target_loading_point)
            self.mode = VehicleMode.LOADED
            self.target_loading_point = None
            self.state = VehicleState.IDLE
            print(f"✅ Vehicle {self.id}: Loading completed, switching to loaded mode")
    
    def _update_unloading(self, current_time: float):
        """更新卸载状态"""
        if current_time - self.operation_start_time >= self.unloading_time:
            # 卸载完成
            self.road_network.release_special_point(self.target_unloading_point)
            self.mode = VehicleMode.RETURNING
            self.target_unloading_point = None
            self.state = VehicleState.IDLE
            print(f"✅ Vehicle {self.id}: Unloading completed, returning to parking")
    
    # ============ 保留原有的移动相关方法 ============
    
    def _execute_planning(self, current_time: float):
        """执行路径规划结果"""
        if not self.path or len(self.path) < 2:
            self.wait_until = current_time + 1.0 + random.random()
            self.state = VehicleState.WAITING
            return
        
        if self.use_gnn:
            success = self._validate_and_reserve_path(current_time)
        else:
            success = self._validate_simple_path(current_time)
        
        if success:
            self.path_confirmed = True
            self.path_index = 0
            
            if self.path_times:
                self.path_start_time = max(self.path_times[0], current_time + 0.5)
            else:
                self.path_start_time = current_time + 0.5
            
            self.state = VehicleState.CONFIRMED
            print(f"Vehicle {self.id}: Path confirmed, will start at {self.path_start_time:.1f}s")
        else:
            if self.use_gnn:
                self.road_network.cancel_reservations(self.id)
            self.wait_until = current_time + 0.5 + random.random() * 1.0
            self.state = VehicleState.WAITING
            print(f"Vehicle {self.id}: Path validation failed, waiting to retry")
    
    def _validate_simple_path(self, current_time: float) -> bool:
        """简单模式路径验证"""
        if not self.path_times or len(self.path_times) != len(self.path):
            return False
        
        base_time = current_time + 0.5
        
        other_vehicles = getattr(self.road_network, 'vehicles', [])
        for other_vehicle in other_vehicles:
            if other_vehicle.id == self.id:
                continue
            
            if other_vehicle.state in [VehicleState.MOVING, VehicleState.CONFIRMED]:
                if hasattr(other_vehicle, 'path_start_time') and other_vehicle.path_start_time:
                    if other_vehicle.path_times:
                        other_end_time = other_vehicle.path_times[-1]
                    else:
                        other_end_time = other_vehicle.path_start_time + len(other_vehicle.path) * 1.0
                    
                    if base_time < other_end_time + 0.5:
                        base_time = other_end_time + 0.5 + random.random() * 0.5
        
        adjusted_times = []
        current_t = base_time
        
        for i, node in enumerate(self.path):
            if i == 0:
                adjusted_times.append(current_t)
            else:
                prev_node = self.path[i-1]
                travel_time = self.road_network._compute_travel_time(prev_node, node, self.id)
                current_t += travel_time
                adjusted_times.append(current_t)
        
        self.path_times = adjusted_times
        print(f"Vehicle {self.id}: Simple mode path scheduled from {base_time:.2f}")
        return True
    
    def _validate_and_reserve_path(self, current_time: float) -> bool:
        """验证并预留整条路径"""
        if not self.use_gnn:
            return True
        
        if not self.path_times or len(self.path_times) != len(self.path):
            print(f"Vehicle {self.id}: Invalid path times")
            return False
        
        adjusted_times = []
        base_time = max(current_time + 0.5, self.path_times[0])
        
        for i, original_time in enumerate(self.path_times):
            if i == 0:
                adjusted_times.append(base_time)
            else:
                interval = self.path_times[i] - self.path_times[i-1]
                adjusted_times.append(adjusted_times[-1] + interval)
        
        self.path_times = adjusted_times
        
        print(f"Vehicle {self.id}: Attempting to reserve path {self.path} with times {[f'{t:.2f}' for t in self.path_times]}")
        
        node_duration = 0.4
        
        # 验证所有节点占用
        for i, node in enumerate(self.path):
            node_start_time = self.path_times[i]
            if i == len(self.path) - 1:
                node_end_time = node_start_time + node_duration * 3
            else:
                node_end_time = node_start_time + node_duration
            
            if not self.road_network.is_node_available(node, node_start_time, 
                                                     node_end_time - node_start_time, 
                                                     exclude_vehicle=self.id):
                print(f"Vehicle {self.id}: Node validation failed at {node}")
                return False
        
        # 验证所有边占用
        for i in range(len(self.path) - 1):
            from_node = self.path[i]
            to_node = self.path[i + 1]
            edge_start_time = self.path_times[i] + node_duration
            edge_duration = self.path_times[i + 1] - edge_start_time
            
            if edge_duration <= 0:
                print(f"Vehicle {self.id}: Invalid edge duration {edge_duration} for edge {from_node}-{to_node}")
                return False
            
            if not self.road_network.is_edge_available(from_node, to_node, 
                                                     edge_start_time, edge_duration, 
                                                     exclude_vehicle=self.id):
                print(f"Vehicle {self.id}: Edge validation failed at edge {from_node}-{to_node}")
                return False
        
        # 预留所有节点
        for i, node in enumerate(self.path):
            node_start_time = self.path_times[i]
            if i == len(self.path) - 1:
                node_end_time = node_start_time + node_duration * 3
            else:
                node_end_time = node_start_time + node_duration
            
            success = self.road_network.reserve_node(node, self.id, 
                                                   node_start_time, 
                                                   node_end_time - node_start_time)
            if not success:
                print(f"Vehicle {self.id}: Failed to reserve node {node}, canceling all reservations")
                self.road_network.cancel_reservations(self.id)
                return False
        
        # 预留所有边
        for i in range(len(self.path) - 1):
            from_node = self.path[i]
            to_node = self.path[i + 1]
            edge_start_time = self.path_times[i] + node_duration
            edge_duration = self.path_times[i + 1] - edge_start_time
            
            success = self.road_network.reserve_edge(from_node, to_node, self.id, 
                                                   edge_start_time, edge_duration)
            if not success:
                print(f"Vehicle {self.id}: Failed to reserve edge {from_node}-{to_node}, canceling all reservations")
                self.road_network.cancel_reservations(self.id)
                return False
        
        print(f"Vehicle {self.id}: Successfully reserved entire path")
        return True
    
    def _start_confirmed_path(self, current_time: float):
        """开始已确认的路径"""
        if self.path_index + 1 >= len(self.path):
            self.state = VehicleState.IDLE
            return
        
        self._start_next_move(current_time)
        print(f"Vehicle {self.id}: Starting confirmed path execution")
        
    def _start_next_move(self, current_time: float):
        """开始下一段移动"""
        if self.path_index + 1 >= len(self.path):
            self.state = VehicleState.IDLE
            return
        
        next_node = self.path[self.path_index + 1]
        
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
        print(f"Vehicle {self.id}: Moving from {self.path[self.path_index]} to {next_node}")
    
    def _update_movement(self, current_time: float, dt: float):
        """更新移动状态"""
        if current_time < self.move_start_time:
            return
        
        elapsed = current_time - self.move_start_time
        self.progress = min(elapsed / self.move_duration, 1.0)
        
        if self.progress > 0:
            start_pos = np.array(self.road_network.node_positions[self.path[self.path_index]])
            smooth_progress = self._smooth_step(self.progress)
            self.position = start_pos + (self.target_position - start_pos) * smooth_progress
            
            if dt > 0:
                distance = np.linalg.norm(self.target_position - start_pos) * (self.progress / (elapsed / dt)) * dt
                self.total_distance += abs(distance) * 0.01
        
        if self.progress >= 1.0:
            self._arrive_at_node(current_time)
    
    def _smooth_step(self, t: float) -> float:
        return t * t * (3.0 - 2.0 * t)

# ============ 可视化模块分离 ============

class Stage2Visualization:
    """第二阶段可视化模块 - 分离的可视化系统"""
    
    def __init__(self, road_network: Stage2RoadNetwork, vehicles: List[Vehicle]):
        self.road_network = road_network
        self.vehicles = vehicles
        
        # 可视化设置
        self.fig, (self.ax_main, self.ax_stats) = plt.subplots(1, 2, figsize=(16, 8))
        self._setup_visualization()
    
    def _setup_visualization(self):
        """设置可视化"""
        positions = list(self.road_network.node_positions.values())
        if positions:
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            margin = 10.0
            self.ax_main.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
            self.ax_main.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
        else:
            self.ax_main.set_xlim(-10, 110)
            self.ax_main.set_ylim(-10, 110)
        
        self.ax_main.set_aspect('equal')
        self.ax_main.set_title('Stage 2: GNN Multi-Vehicle Coordination\nComplete Parking-Loading-Unloading Cycle', 
                              fontsize=12, fontweight='bold')
        self.ax_main.grid(True, alpha=0.3)
        
        self.ax_stats.set_xlim(0, 1)
        self.ax_stats.set_ylim(0, 1)
        self.ax_stats.axis('off')
    
    def update_visualization(self, current_time: float):
        """更新可视化"""
        self.ax_main.clear()
        self.ax_stats.clear()
        
        self._setup_visualization()
        
        self._draw_network()
        self._draw_special_points()
        self._draw_reservations()
        self._draw_vehicles()
        self._draw_statistics(current_time)
    
    def _draw_network(self):
        """绘制网络"""
        # 绘制边
        for edge in self.road_network.graph.edges():
            node1, node2 = edge
            pos1 = self.road_network.node_positions[node1]
            pos2 = self.road_network.node_positions[node2]
            
            weight = self.road_network.graph[node1][node2].get('weight', 1.0)
            linewidth = max(0.5, min(3.0, weight))
            
            self.ax_main.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                            color='lightgray', linewidth=linewidth, alpha=0.7)
        
        # 绘制普通节点
        current_time = self.road_network.global_time
        
        for node, pos in self.road_network.node_positions.items():
            # 跳过特殊点，它们会单独绘制
            is_special = any(
                point.node_id == node 
                for points in [self.road_network.loading_points, 
                              self.road_network.unloading_points, 
                              self.road_network.parking_points]
                for point in points.values()
            )
            
            if is_special:
                continue
            
            vehicle_count = len(self.road_network.node_occupancy[node])
            degree = self.road_network.graph.degree(node)
            
            # 检查预留状态
            node_reservations = self.road_network.node_reservations.get(node, [])
            has_active_reservation = any(
                r.start_time <= current_time <= r.end_time 
                for r in node_reservations
            )
            
            has_future_reservation = any(
                r.start_time > current_time 
                for r in node_reservations
            )
            
            # 选择颜色
            if has_active_reservation:
                color = 'red'
                alpha = 0.9
            elif has_future_reservation:
                color = 'orange'
                alpha = 0.8
            elif vehicle_count > 0:
                color = 'pink'
                alpha = 0.8
            else:
                if degree == 2:
                    color = 'lightblue'
                elif degree == 3:
                    color = 'lightgreen'
                elif degree == 4:
                    color = 'lightcoral'
                else:
                    color = 'lightyellow'
                alpha = 0.8
            
            size = 10 + degree * 3
            circle = Circle(pos, size*0.01, color=color, 
                          edgecolor='navy', linewidth=1, alpha=alpha)
            self.ax_main.add_patch(circle)
            
            # 节点标签
            label = f"{node[-4:]}\n({degree})"
            if has_active_reservation:
                active_vehicle = None
                for r in node_reservations:
                    if r.start_time <= current_time <= r.end_time:
                        active_vehicle = r.vehicle_id
                        break
                if active_vehicle is not None:
                    label += f"\nV{active_vehicle}"
            
            self.ax_main.text(pos[0], pos[1], label, ha='center', va='center', 
                            fontsize=6, fontweight='bold')
    
    def _draw_special_points(self):
        """绘制特殊点"""
        # 绘制装载点
        for point_id, point in self.road_network.loading_points.items():
            pos = self.road_network.node_positions.get(point.node_id)
            if not pos:
                continue
                
            x, y = pos[0], pos[1]
            
            if point.is_occupied:
                color = 'darkgreen'
                marker = '■'
                status = f"Loading V{point.reserved_by}"
            elif point.reserved_by is not None:
                color = 'orange'
                marker = '□'
                status = f"Reserved V{point.reserved_by}"
            else:
                color = 'green'
                marker = '□'
                status = "Available"
            
            size = 25
            circle = Circle(pos, size*0.01, color=color, 
                          edgecolor='darkgreen', linewidth=2, alpha=0.9)
            self.ax_main.add_patch(circle)
            
            label = f"{point_id}\n{marker}\n{status}"
            self.ax_main.text(x, y, label, ha='center', va='center', 
                            fontsize=6, fontweight='bold', color='white')
        
        # 绘制卸载点
        for point_id, point in self.road_network.unloading_points.items():
            pos = self.road_network.node_positions.get(point.node_id)
            if not pos:
                continue
                
            x, y = pos[0], pos[1]
            
            if point.is_occupied:
                color = 'darkblue'
                marker = '▼'
                status = f"Unloading V{point.reserved_by}"
            elif point.reserved_by is not None:
                color = 'orange'
                marker = '△'
                status = f"Reserved V{point.reserved_by}"
            else:
                color = 'blue'
                marker = '△'
                status = "Available"
            
            size = 25
            circle = Circle(pos, size*0.01, color=color, 
                          edgecolor='darkblue', linewidth=2, alpha=0.9)
            self.ax_main.add_patch(circle)
            
            label = f"{point_id}\n{marker}\n{status}"
            self.ax_main.text(x, y, label, ha='center', va='center', 
                            fontsize=6, fontweight='bold', color='white')
        
        # 绘制停车点
        for point_id, point in self.road_network.parking_points.items():
            pos = self.road_network.node_positions.get(point.node_id)
            if not pos:
                continue
                
            x, y = pos[0], pos[1]
            
            if point.is_occupied:
                color = 'darkgray'
                marker = '🅿️'
                status = f"Parked V{point.reserved_by}"
            elif point.reserved_by is not None:
                color = 'orange'
                marker = 'P'
                status = f"Reserved V{point.reserved_by}"
            else:
                color = 'gray'
                marker = 'P'
                status = "Available"
            
            size = 25
            circle = Circle(pos, size*0.01, color=color, 
                          edgecolor='black', linewidth=2, alpha=0.9)
            self.ax_main.add_patch(circle)
            
            label = f"{point_id}\n{marker}\n{status}"
            self.ax_main.text(x, y, label, ha='center', va='center', 
                            fontsize=6, fontweight='bold', color='white')
    
    def _draw_reservations(self):
        """绘制预留"""
        current_time = self.road_network.global_time
        
        for edge_key, reservations in self.road_network.edge_reservations.items():
            if not reservations:
                continue
                
            node1, node2 = edge_key
            pos1 = self.road_network.node_positions[node1]
            pos2 = self.road_network.node_positions[node2]
            
            for i, reservation in enumerate(reservations):
                if reservation.end_time < current_time:
                    continue
                
                vehicle = next((v for v in self.vehicles if v.id == reservation.vehicle_id), None)
                if vehicle:
                    offset_factor = (i - len(reservations)/2 + 0.5) * 0.1
                    offset_x = (pos2[1] - pos1[1]) * offset_factor
                    offset_y = (pos1[0] - pos2[0]) * offset_factor
                    
                    x1, y1 = pos1[0] + offset_x, pos1[1] + offset_y
                    x2, y2 = pos2[0] + offset_x, pos2[1] + offset_y
                    
                    self.ax_main.plot([x1, x2], [y1, y2], 
                                    color=vehicle.color, linewidth=4, alpha=0.7)
                    
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    self.ax_main.text(mid_x, mid_y, f'V{vehicle.id}', 
                                    ha='center', va='center', fontsize=6,
                                    bbox=dict(boxstyle='round,pad=0.2', 
                                            facecolor='white', alpha=0.8))
    
    def _draw_vehicles(self):
        """绘制车辆"""
        for vehicle in self.vehicles:
            x, y = vehicle.position
            
            # 根据状态调整视觉效果
            if vehicle.state == VehicleState.LOADING:
                alpha = 1.0
                edge_color = 'green'
                edge_width = 4
                size = 16
                symbol = 'L'
            elif vehicle.state == VehicleState.UNLOADING:
                alpha = 1.0
                edge_color = 'blue'
                edge_width = 4
                size = 16
                symbol = 'U'
            elif vehicle.state == VehicleState.CONFIRMED:
                alpha = 0.95
                edge_color = 'gold'
                edge_width = 4
                size = 14
                symbol = 'C'
            elif vehicle.state == VehicleState.MOVING:
                alpha = 1.0
                edge_color = 'white'
                edge_width = 2
                size = 14 if vehicle.mode == VehicleMode.LOADED else 12
                symbol = 'M'
            elif vehicle.state == VehicleState.WAITING:
                alpha = 0.5
                edge_color = 'red'
                edge_width = 2
                size = 10
                symbol = 'W'
            elif vehicle.state == VehicleState.PLANNING:
                alpha = 0.7
                edge_color = 'orange'
                edge_width = 2
                size = 11
                symbol = 'P'
            else:  # IDLE
                alpha = 0.8
                edge_color = 'black'
                edge_width = 1
                size = 11
                symbol = 'I'
            
            # 车辆形状 - 根据模式区分
            if vehicle.mode == VehicleMode.LOADED:
                # 重载 - 方形
                rect = Rectangle((x-size*0.01, y-size*0.01), size*0.02, size*0.02, 
                               color=vehicle.color, alpha=alpha, 
                               edgecolor=edge_color, linewidth=edge_width)
                self.ax_main.add_patch(rect)
                mode_symbol = '■'
            elif vehicle.mode == VehicleMode.PARKED:
                # 停车 - 六边形
                circle = Circle((x, y), size*0.01, color=vehicle.color, alpha=alpha,
                              edgecolor=edge_color, linewidth=edge_width)
                self.ax_main.add_patch(circle)
                mode_symbol = '🅿️'
            elif vehicle.mode == VehicleMode.RETURNING:
                # 返回 - 菱形
                circle = Circle((x, y), size*0.01, color=vehicle.color, alpha=alpha,
                              edgecolor=edge_color, linewidth=edge_width)
                self.ax_main.add_patch(circle)
                mode_symbol = '◆'
            else:
                # 空载 - 圆形
                circle = Circle((x, y), size*0.01, color=vehicle.color, alpha=alpha,
                              edgecolor=edge_color, linewidth=edge_width)
                self.ax_main.add_patch(circle)
                mode_symbol = '○'
            
            # 车辆ID、状态和模式标识
            display_text = f'{vehicle.id}\n{symbol}\n{mode_symbol}'
            
            self.ax_main.text(x, y, display_text, 
                            ha='center', va='center',
                            color='white', fontweight='bold', fontsize=7)
            
            # 目标连线
            target_point_id = vehicle.target_point_id
            if target_point_id:
                target_node_id = vehicle.target_node_id
                if target_node_id and target_node_id in self.road_network.node_positions:
                    target_pos = self.road_network.node_positions[target_node_id]
                    
                    # 根据模式选择线条颜色
                    if vehicle.mode == VehicleMode.PARKED or vehicle.mode == VehicleMode.EMPTY:
                        line_color = 'green'  # 前往装载点
                        line_style = '-.'
                    elif vehicle.mode == VehicleMode.LOADED:
                        line_color = 'blue'   # 前往卸载点
                        line_style = '--'
                    elif vehicle.mode == VehicleMode.RETURNING:
                        line_color = 'gray'   # 前往停车点
                        line_style = ':'
                    else:
                        line_color = 'black'
                        line_style = '-'
                    
                    line_alpha = 0.8 if vehicle.state == VehicleState.CONFIRMED else 0.5
                    self.ax_main.plot([x, target_pos[0]], [y, target_pos[1]], 
                                    color=line_color, linestyle=line_style, 
                                    alpha=line_alpha, linewidth=2)
            
            # 已确认路径显示
            if vehicle.state == VehicleState.CONFIRMED and vehicle.path:
                self._draw_confirmed_path(vehicle)
    
    def _draw_confirmed_path(self, vehicle):
        """绘制已确认的路径"""
        if len(vehicle.path) < 2:
            return
        
        path_positions = [self.road_network.node_positions[node] for node in vehicle.path]
        xs = [pos[0] for pos in path_positions]
        ys = [pos[1] for pos in path_positions]
        
        self.ax_main.plot(xs, ys, color=vehicle.color, linewidth=6, 
                         alpha=0.4, linestyle='-', zorder=1)
        
        for i, pos in enumerate(path_positions):
            self.ax_main.text(pos[0]-3, pos[1]+3, str(i), 
                            ha='center', va='center', fontsize=8,
                            bbox=dict(boxstyle='circle,pad=0.1', 
                                    facecolor=vehicle.color, alpha=0.7),
                            color='white', fontweight='bold')
    
    def _draw_statistics(self, current_time: float):
        """绘制统计信息"""
        self.ax_stats.set_xlim(0, 1)
        self.ax_stats.set_ylim(0, 1)
        self.ax_stats.axis('off')
        
        # 统计数据
        total_distance = sum(v.total_distance for v in self.vehicles)
        total_cycles = sum(v.completed_cycles for v in self.vehicles)
        total_wait_time = sum(v.wait_time for v in self.vehicles)
        
        # 车辆状态统计
        state_counts = defaultdict(int)
        mode_counts = defaultdict(int)
        for vehicle in self.vehicles:
            state_counts[vehicle.state] += 1
            mode_counts[vehicle.mode] += 1
        
        # 特殊点使用统计
        loading_occupied = sum(1 for p in self.road_network.loading_points.values() if p.is_occupied)
        loading_reserved = sum(1 for p in self.road_network.loading_points.values() if p.reserved_by is not None and not p.is_occupied)
        loading_available = len(self.road_network.loading_points) - loading_occupied - loading_reserved
        
        unloading_occupied = sum(1 for p in self.road_network.unloading_points.values() if p.is_occupied)
        unloading_reserved = sum(1 for p in self.road_network.unloading_points.values() if p.reserved_by is not None and not p.is_occupied)
        unloading_available = len(self.road_network.unloading_points) - unloading_occupied - unloading_reserved
        
        parking_occupied = sum(1 for p in self.road_network.parking_points.values() if p.is_occupied)
        parking_reserved = sum(1 for p in self.road_network.parking_points.values() if p.reserved_by is not None and not p.is_occupied)
        parking_available = len(self.road_network.parking_points) - parking_occupied - parking_reserved
        
        # 预留统计
        total_edge_reservations = sum(len(reservations) for reservations in 
                                    self.road_network.edge_reservations.values())
        total_node_reservations = sum(len(reservations) for reservations in 
                                    self.road_network.node_reservations.values())
        
        stats_text = f"""
        ╔═══ STAGE 2 GNN SYSTEM ═══╗
        ║ Complete Cycle Mode       ║
        ║ Time: {current_time:>17.1f}s ║
        ║ Vehicles: {len(self.vehicles):>13d} ║
        ╠═══ OPERATION STATS ══════╣
        ║ Completed Cycles: {total_cycles:>7d} ║
        ║ Total Distance: {total_distance:>9.1f} ║
        ║ Wait Time: {total_wait_time:>11.1f}s ║
        ╠═══ VEHICLE STATES ═══════╣
        ║ Idle: {state_counts[VehicleState.IDLE]:>17d} ║
        ║ Planning: {state_counts[VehicleState.PLANNING]:>13d} ║
        ║ Waiting: {state_counts[VehicleState.WAITING]:>14d} ║
        ║ Confirmed: {state_counts[VehicleState.CONFIRMED]:>12d} ║
        ║ Moving: {state_counts[VehicleState.MOVING]:>15d} ║
        ║ Loading: {state_counts[VehicleState.LOADING]:>14d} ║
        ║ Unloading: {state_counts[VehicleState.UNLOADING]:>12d} ║
        ║ Blocked: {state_counts[VehicleState.BLOCKED]:>14d} ║
        ╠═══ VEHICLE MODES ════════╣
        ║ Parked: {mode_counts[VehicleMode.PARKED]:>15d} ║
        ║ Empty: {mode_counts[VehicleMode.EMPTY]:>16d} ║
        ║ Loaded: {mode_counts[VehicleMode.LOADED]:>15d} ║
        ║ Returning: {mode_counts[VehicleMode.RETURNING]:>12d} ║
        ╠═══ LOADING POINTS ═══════╣
        ║ In Use: {loading_occupied:>15d} ║
        ║ Reserved: {loading_reserved:>13d} ║
        ║ Available: {loading_available:>12d} ║
        ╠═══ UNLOADING POINTS ═════╣
        ║ In Use: {unloading_occupied:>15d} ║
        ║ Reserved: {unloading_reserved:>13d} ║
        ║ Available: {unloading_available:>12d} ║
        ╠═══ PARKING POINTS ═══════╣
        ║ In Use: {parking_occupied:>15d} ║
        ║ Reserved: {parking_reserved:>13d} ║
        ║ Available: {parking_available:>12d} ║
        ╠═══ RESERVATION STATUS ═══╣
        ║ Edge Rsv: {total_edge_reservations:>13d} ║
        ║ Node Rsv: {total_node_reservations:>13d} ║
        ╚═══════════════════════════╝
        
        Complete Cycle Operations:
        🅿️ → 🟢 → 🔵 → 🅿️ (P→L→U→P)
        
        Vehicle Display:
        🅿️ Parked (at parking point)
        ○ Empty (to loading point)
        ■ Loaded (to unloading point)  
        ◆ Returning (to parking point)
        
        Special Points:
        🟢 L0,L1,.. Loading Points (Green)
        🔵 U0,U1,.. Unloading Points (Blue)
        🅿️ P0,P1,.. Parking Points (Gray)
        
        Controls:
        Space: Start/Pause
        'g': Toggle GNN/Simple
        '+': Add Vehicle
        '-': Remove Vehicle
        'r': Reset
        'q': Quit
        """.strip()
        
        self.ax_stats.text(0.02, 0.98, stats_text, 
                         transform=self.ax_stats.transAxes,
                         fontfamily='monospace', fontsize=7.5,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round,pad=0.5', 
                                 facecolor='lightblue', alpha=0.9))

class Stage2GNNSimulation:
    """第二阶段GNN仿真 - 核心逻辑类"""
    
    def __init__(self, topology_file_path: str = None, num_vehicles: int = 4):
        self.topology_file_path = topology_file_path
        self.num_vehicles = num_vehicles
        self.road_network = Stage2RoadNetwork(topology_file_path, num_vehicles)
        self.vehicles = []
        self.use_gnn = True
        self.current_time = 0.0
        
        self._create_initial_vehicles()
        
        # 可视化系统（可选）
        self.visualization = None
        self.animation = None
        self.is_running = False
    
    def enable_visualization(self):
        """启用可视化"""
        self.visualization = Stage2Visualization(self.road_network, self.vehicles)
    
    def _create_initial_vehicles(self):
        """创建初始车辆"""
        # 为每个车辆分配停车点
        parking_points = list(self.road_network.parking_points.keys())
        
        if len(parking_points) < self.num_vehicles:
            print(f"⚠️ 停车点数量({len(parking_points)})少于车辆数量({self.num_vehicles})")
            # 创建额外的停车点
            self.road_network._create_fallback_special_points()
            parking_points = list(self.road_network.parking_points.keys())
        
        for i in range(self.num_vehicles):
            # 循环分配停车点
            start_parking = parking_points[i % len(parking_points)]
            
            vehicle = Vehicle(i, start_parking, self.road_network, self.use_gnn)
            self.vehicles.append(vehicle)
        
        # 将车辆列表传递给道路网络
        self.road_network.vehicles = self.vehicles
        
        print(f"🚛 创建了 {len(self.vehicles)} 辆车辆")
        for i, vehicle in enumerate(self.vehicles):
            print(f"   V{i}: 停车在 {vehicle.current_parking_point}")
    
    def update(self, dt: float = 0.1):
        """更新仿真状态"""
        self.current_time += dt
        
        self.road_network.update_time(self.current_time)
        
        for vehicle in self.vehicles:
            vehicle.update(self.current_time, dt)
    
    def update_visualization_frame(self, frame):
        """可视化更新回调"""
        self.update()
        
        if self.visualization:
            self.visualization.update_visualization(self.current_time)
        
        return []
    
    def toggle_gnn_mode(self):
        """切换GNN模式"""
        self.use_gnn = not self.use_gnn
        for vehicle in self.vehicles:
            vehicle.use_gnn = self.use_gnn
            vehicle.road_network.cancel_reservations(vehicle.id)
            vehicle.state = VehicleState.IDLE
            vehicle.retry_count = 0
            vehicle.path_confirmed = False
        print(f"GNN Mode: {'Enabled' if self.use_gnn else 'Disabled'}")
    
    def add_vehicle(self):
        """添加车辆"""
        max_vehicles = min(12, len(self.road_network.parking_points))
        if len(self.vehicles) >= max_vehicles:
            print(f"Maximum vehicles reached! (Limited by parking points: {max_vehicles})")
            return
        
        # 寻找空闲的停车点
        available_parking = self.road_network.get_available_parking_point()
        if not available_parking:
            print(f"No available parking points for new vehicle!")
            return
        
        vehicle_id = len(self.vehicles)
        vehicle = Vehicle(vehicle_id, available_parking, self.road_network, self.use_gnn)
        self.vehicles.append(vehicle)
        
        # 更新车辆数量
        self.num_vehicles = len(self.vehicles)
        self.road_network.num_vehicles = self.num_vehicles
        
        # 更新道路网络中的车辆引用
        self.road_network.vehicles = self.vehicles
        print(f"Added vehicle {vehicle_id}, total: {len(self.vehicles)} vehicles")
    
    def remove_vehicle(self):
        """移除车辆"""
        if len(self.vehicles) <= 1:
            print("Must keep at least 1 vehicle!")
            return
        
        if self.vehicles:
            removed = self.vehicles.pop()
            removed.road_network.cancel_reservations(removed.id)
            removed.road_network.cancel_point_reservations(removed.id)
            removed.road_network.remove_vehicle_from_node(removed.id, removed.current_node)
            
            # 释放停车点
            if removed.current_parking_point:
                removed.road_network.release_special_point(removed.current_parking_point)
            
            # 更新车辆数量
            self.num_vehicles = len(self.vehicles)
            self.road_network.num_vehicles = self.num_vehicles
            
            # 更新道路网络中的车辆引用
            self.road_network.vehicles = self.vehicles
            print(f"Removed vehicle {removed.id}, total: {len(self.vehicles)} vehicles")
    
    def start_animation(self):
        """开始动画"""
        if not self.visualization:
            print("Visualization not enabled. Call enable_visualization() first.")
            return
        
        if not self.is_running:
            self.animation = animation.FuncAnimation(
                self.visualization.fig, self.update_visualization_frame, interval=100, blit=False)
            self.is_running = True
            print("Stage 2 simulation started")
    
    def stop_animation(self):
        """停止动画"""
        if self.animation:
            self.animation.event_source.stop()
            self.is_running = False
            print("Stage 2 simulation stopped")
    
    def reset_simulation(self):
        """重置仿真"""
        self.current_time = 0.0
        
        num_vehicles = len(self.vehicles)
        self.vehicles.clear()
        self._create_initial_vehicles()
        print(f"Reset Stage 2 simulation - {num_vehicles} vehicles")
        
        for vehicle in self.vehicles:
            vehicle.use_gnn = self.use_gnn
        
        # 确保道路网络有车辆引用
        self.road_network.vehicles = self.vehicles
    
    def show(self):
        """显示仿真"""
        if self.visualization:
            plt.tight_layout()
            plt.show()

def select_topology_file():
    """选择拓扑文件"""
    import tkinter as tk
    from tkinter import filedialog
    
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    file_path = filedialog.askopenfilename(
        title="选择第一阶段导出的拓扑文件",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        initialdir="."
    )
    
    root.destroy()
    return file_path

def main():
    """主函数"""
    print("第二阶段：基于第一阶段拓扑的多车协同GNN演示")
    print("=" * 80)
    print("🎯 完整循环作业模式:")
    print("🅿️ 停车点 → 🟢 装载点 → 🔵 卸载点 → 🅿️ 停车点")
    print("⚡ 完整的P→L→U→P循环作业流程")
    print()
    
    # 选择拓扑文件
    print("请选择第一阶段导出的拓扑文件...")
    topology_file = select_topology_file()
    
    if not topology_file:
        print("❌ 未选择文件，将使用默认回退拓扑")
        topology_file = None
    else:
        print(f"✅ 已选择拓扑文件: {Path(topology_file).name}")
    
    print()
    print("🌍 第二阶段网络特性:")
    print("✅ 智能特殊点解析 - 自动识别L、U、P点")
    print("✅ 完整循环作业 - 停车→装载→卸载→停车")
    print("✅ GNN感知路径规划 - 利用拓扑结构优化")
    print("✅ 多车协同调度 - 避免冲突的时空预留")
    print("✅ 可视化系统分离 - 便于后续GUI集成")
    print()
    print("🎨 可视化增强:")
    print("🟢 L0,L1,.. 装载点显示")
    print("🔵 U0,U1,.. 卸载点显示")
    print("🅿️ P0,P1,.. 停车点显示")
    print("🚛 完整车辆状态 - 停车/空载/重载/返回")
    print()
    print("控制说明:")
    print("- Space: 开始/暂停仿真")
    print("- 'g': 切换 GNN/传统 调度模式")
    print("- '+'/'-': 增加/减少车辆数量")
    print("- 'r': 重置仿真")
    print("- 'q': 退出程序")
    print("=" * 80)
    
    # 创建第二阶段仿真
    try:
        sim = Stage2GNNSimulation(topology_file_path=topology_file, num_vehicles=4)
        sim.enable_visualization()  # 启用可视化
    except Exception as e:
        print(f"❌ 创建第二阶段仿真失败: {e}")
        print("🔄 使用回退拓扑...")
        sim = Stage2GNNSimulation(topology_file_path=None, num_vehicles=4)
        sim.enable_visualization()
    
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
    
    sim.visualization.fig.canvas.mpl_connect('key_press_event', on_key)
    sim.start_animation()
    sim.show()

if __name__ == "__main__":
    main()