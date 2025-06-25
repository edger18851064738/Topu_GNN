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
    EMPTY = "empty"    # 空载，前往装载点
    LOADED = "loaded"  # 重载，前往卸载点

@dataclass
class LoadingPoint:
    """装载点信息"""
    node_id: str
    is_occupied: bool = False
    reserved_by: Optional[int] = None  # 被哪个车辆预留
    
@dataclass
class UnloadingPoint:
    """卸载点信息"""
    node_id: str
    is_occupied: bool = False
    reserved_by: Optional[int] = None  # 被哪个车辆预留

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
    """第二阶段拓扑加载器 - 读取第一阶段导出的数据"""
    
    def __init__(self, topology_file_path: str):
        self.topology_file_path = topology_file_path
        self.topology_data = None
        self.graph = None
        self.node_positions = {}
        self.loading_candidates = []
        self.unloading_candidates = []
        
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
            
            # 识别装载卸载候选点
            self._identify_loading_unloading_candidates()
            
            print(f"✅ 拓扑结构加载成功:")
            print(f"   节点数: {len(self.graph.nodes())}")
            print(f"   边数: {len(self.graph.edges())}")
            print(f"   装载候选点: {len(self.loading_candidates)}")
            print(f"   卸载候选点: {len(self.unloading_candidates)}")
            
            return True
            
        except Exception as e:
            print(f"❌ 拓扑结构加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _validate_topology_data(self) -> bool:
        """验证拓扑数据完整性"""
        required_fields = ['stage1_progress', 'ready_for_stage2']
        
        for field in required_fields:
            if field not in self.topology_data:
                print(f"❌ 缺少必需字段: {field}")
                return False
        
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
        
        # 策略2: 回退到原始骨干路径数据
        if self._try_build_from_raw_paths():
            print(f"✅ 使用原始骨干路径数据构建图结构")
            return True
        
        # 策略3: 最后回退到基本图结构
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
    
    def _try_build_from_raw_paths(self) -> bool:
        """尝试从原始骨干路径构建图"""
        raw_paths = self.topology_data.get('raw_backbone_paths', {})
        if not raw_paths or 'paths_info' not in raw_paths:
            return False
        
        try:
            paths_info = raw_paths['paths_info']
            node_counter = 0
            
            # 从路径中提取关键节点
            for path_id, path_data in paths_info.items():
                forward_path = path_data.get('forward_path', [])
                if len(forward_path) < 2:
                    continue
                
                # 添加路径端点和一些中间点
                key_indices = [0]  # 起点
                
                # 添加一些中间关键点
                path_length = len(forward_path)
                if path_length > 10:
                    step = path_length // 5
                    for i in range(step, path_length - step, step):
                        key_indices.append(i)
                
                key_indices.append(path_length - 1)  # 终点
                
                prev_node_id = None
                for idx in key_indices:
                    point = forward_path[idx]
                    node_id = f"node_{node_counter}"
                    
                    self.graph.add_node(node_id)
                    self.node_positions[node_id] = (point[0], point[1])
                    
                    # 连接到前一个节点
                    if prev_node_id:
                        pos1 = self.node_positions[prev_node_id]
                        pos2 = self.node_positions[node_id]
                        weight = math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
                        weight = max(0.5, min(weight / 10.0, 3.0))
                        
                        self.graph.add_edge(prev_node_id, node_id, weight=weight)
                    
                    prev_node_id = node_id
                    node_counter += 1
            
            return len(self.graph.nodes()) > 0
            
        except Exception as e:
            print(f"⚠️ 原始路径数据构建失败: {e}")
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
    
    def _identify_loading_unloading_candidates(self):
        """识别装载和卸载候选点"""
        if not self.graph.nodes():
            return
        
        # 根据节点度数和位置特征识别边缘节点
        node_degrees = dict(self.graph.degree())
        
        # 计算坐标范围
        x_coords = [pos[0] for pos in self.node_positions.values()]
        y_coords = [pos[1] for pos in self.node_positions.values()]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # 边界阈值
        x_margin = (x_max - x_min) * 0.2
        y_margin = (y_max - y_min) * 0.2
        
        edge_nodes = []
        low_degree_nodes = []
        
        for node_id, pos in self.node_positions.items():
            x, y = pos
            degree = node_degrees.get(node_id, 0)
            
            # 边缘节点
            is_edge = (x <= x_min + x_margin or x >= x_max - x_margin or 
                      y <= y_min + y_margin or y >= y_max - y_margin)
            
            # 低度数节点
            is_low_degree = degree <= 3
            
            if is_edge:
                edge_nodes.append(node_id)
            if is_low_degree:
                low_degree_nodes.append(node_id)
        
        # 组合边缘节点和低度数节点作为候选
        candidates = list(set(edge_nodes + low_degree_nodes))
        
        if len(candidates) < 4:
            # 如果候选不够，选择所有节点
            candidates = list(self.graph.nodes())
        
        # 随机打乱并分配
        random.shuffle(candidates)
        
        # 分配装载点和卸载点（交替分配）
        for i, node_id in enumerate(candidates):
            if i % 2 == 0 and len(self.loading_candidates) < len(candidates) // 2:
                self.loading_candidates.append(node_id)
            elif len(self.unloading_candidates) < len(candidates) // 2:
                self.unloading_candidates.append(node_id)
    
    def get_graph(self) -> nx.Graph:
        """获取构建的图"""
        return self.graph
    
    def get_node_positions(self) -> Dict[str, Tuple[float, float]]:
        """获取节点位置"""
        return self.node_positions.copy()
    
    def get_topology_info(self) -> Dict:
        """获取拓扑信息"""
        return {
            'topology_source': self.topology_data.get('system', 'Unknown'),
            'stage1_progress': self.topology_data.get('stage1_progress', {}),
            'enhanced_consolidation': self.topology_data.get('enhanced_consolidation_applied', False),
            'construction_stats': self.topology_data.get('construction_stats', {}),
            'export_time': self.topology_data.get('export_time', 'Unknown'),
            'gnn_input_ready': self.topology_data.get('gnn_input_ready', False),
        }

class Stage2RoadNetwork:
    """第二阶段道路网络类 - 基于第一阶段导出的拓扑"""
    
    def __init__(self, topology_file_path: str = None, num_vehicles: int = 6):
        self.topology_file_path = topology_file_path
        self.num_vehicles = num_vehicles
        self.topology_loader = None
        self.graph = nx.Graph()
        self.node_positions = {}
        self.topology_info = {}
        
        # 继承原有的预留和管理系统
        self.edge_reservations = defaultdict(list)
        self.node_reservations = defaultdict(list)
        self.node_occupancy = defaultdict(set)
        self.node_features = {}
        self.global_time = 0.0
        
        # 露天矿场景相关
        self.loading_points = {}
        self.unloading_points = {}
        
        if topology_file_path:
            self._load_topology_from_file()
        else:
            self._create_fallback_topology()
        
        self._setup_mining_points()
        self._initialize_features()
    
    def _load_topology_from_file(self):
        """从文件加载拓扑"""
        self.topology_loader = Stage2TopologyLoader(self.topology_file_path)
        
        if self.topology_loader.load_topology():
            self.graph = self.topology_loader.get_graph()
            self.node_positions = self.topology_loader.get_node_positions()
            self.topology_info = self.topology_loader.get_topology_info()
            
            print(f"🎯 第二阶段网络构建成功:")
            print(f"   来源: {self.topology_info.get('topology_source', 'Unknown')}")
            print(f"   增强版: {'✅' if self.topology_info.get('enhanced_consolidation', False) else '❌'}")
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
        
        self.topology_info = {
            'topology_source': 'Fallback Network',
            'enhanced_consolidation': False,
            'gnn_input_ready': True,
        }
    
    def _setup_mining_points(self):
        """设置露天矿的装载点和卸载点"""
        if not self.graph.nodes():
            return
        
        # 使用加载器识别的候选点，或自行识别
        if self.topology_loader:
            loading_candidates = self.topology_loader.loading_candidates
            unloading_candidates = self.topology_loader.unloading_candidates
        else:
            # 自行识别边缘节点
            edge_nodes = self._find_edge_nodes()
            selected_nodes = random.sample(edge_nodes, min(self.num_vehicles * 2, len(edge_nodes)))
            loading_candidates = selected_nodes[:self.num_vehicles]
            unloading_candidates = selected_nodes[self.num_vehicles:self.num_vehicles * 2]
        
        # 确保有足够的候选点
        all_nodes = list(self.graph.nodes())
        while len(loading_candidates) < self.num_vehicles:
            candidate = random.choice(all_nodes)
            if candidate not in loading_candidates:
                loading_candidates.append(candidate)
        
        while len(unloading_candidates) < self.num_vehicles:
            candidate = random.choice(all_nodes)
            if candidate not in unloading_candidates and candidate not in loading_candidates:
                unloading_candidates.append(candidate)
        
        # 创建装载点
        for i, node in enumerate(loading_candidates[:self.num_vehicles]):
            self.loading_points[node] = LoadingPoint(node_id=node)
        
        # 创建卸载点
        for i, node in enumerate(unloading_candidates[:self.num_vehicles]):
            self.unloading_points[node] = UnloadingPoint(node_id=node)
        
        print(f"Stage 2 mining setup complete:")
        print(f"Loading points: {list(self.loading_points.keys())}")
        print(f"Unloading points: {list(self.unloading_points.keys())}")
    
    def _find_edge_nodes(self):
        """找到网络边缘的节点"""
        if not self.node_positions:
            return list(self.graph.nodes())
        
        x_coords = [pos[0] for pos in self.node_positions.values()]
        y_coords = [pos[1] for pos in self.node_positions.values()]
        
        if not x_coords or not y_coords:
            return list(self.graph.nodes())
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        x_margin = (x_max - x_min) * 0.2
        y_margin = (y_max - y_min) * 0.2
        
        edge_nodes = []
        for node, pos in self.node_positions.items():
            x, y = pos
            is_edge = (x <= x_min + x_margin or x >= x_max - x_margin or 
                      y <= y_min + y_margin or y >= y_max - y_margin)
            is_low_degree = self.graph.degree(node) <= 3
            
            if is_edge or is_low_degree:
                edge_nodes.append(node)
        
        return edge_nodes if edge_nodes else list(self.graph.nodes())
    
    def get_topology_description(self) -> str:
        """获取拓扑描述"""
        source = self.topology_info.get('topology_source', 'Unknown')
        enhanced = self.topology_info.get('enhanced_consolidation', False)
        ready = self.topology_info.get('gnn_input_ready', False)
        
        desc = f"{source}"
        if enhanced:
            desc += " (Enhanced)"
        if ready:
            desc += " - GNN Ready"
        
        return desc
    
    # ============ 以下方法完全继承自demo_GNN.py ============
    
    def get_available_loading_point(self, exclude_vehicle: int = -1) -> Optional[str]:
        """获取可用的装载点"""
        for node_id, point in self.loading_points.items():
            if not point.is_occupied and (point.reserved_by is None or point.reserved_by == exclude_vehicle):
                return node_id
        return None
    
    def get_available_unloading_point(self, exclude_vehicle: int = -1) -> Optional[str]:
        """获取可用的卸载点"""
        for node_id, point in self.unloading_points.items():
            if not point.is_occupied and (point.reserved_by is None or point.reserved_by == exclude_vehicle):
                return node_id
        return None
    
    def reserve_loading_point(self, node_id: str, vehicle_id: int) -> bool:
        """预留装载点"""
        if node_id in self.loading_points:
            point = self.loading_points[node_id]
            if not point.is_occupied and point.reserved_by is None:
                point.reserved_by = vehicle_id
                print(f"Loading point {node_id} reserved by vehicle V{vehicle_id}")
                return True
        return False
    
    def reserve_unloading_point(self, node_id: str, vehicle_id: int) -> bool:
        """预留卸载点"""
        if node_id in self.unloading_points:
            point = self.unloading_points[node_id]
            if not point.is_occupied and point.reserved_by is None:
                point.reserved_by = vehicle_id
                print(f"Unloading point {node_id} reserved by vehicle V{vehicle_id}")
                return True
        return False
    
    def occupy_loading_point(self, node_id: str, vehicle_id: int):
        """占用装载点"""
        if node_id in self.loading_points:
            point = self.loading_points[node_id]
            point.is_occupied = True
            point.reserved_by = vehicle_id
    
    def occupy_unloading_point(self, node_id: str, vehicle_id: int):
        """占用卸载点"""
        if node_id in self.unloading_points:
            point = self.unloading_points[node_id]
            point.is_occupied = True
            point.reserved_by = vehicle_id
    
    def release_loading_point(self, node_id: str):
        """释放装载点"""
        if node_id in self.loading_points:
            point = self.loading_points[node_id]
            point.is_occupied = False
            point.reserved_by = None
            print(f"Loading point {node_id} released")
    
    def release_unloading_point(self, node_id: str):
        """释放卸载点"""
        if node_id in self.unloading_points:
            point = self.unloading_points[node_id]
            point.is_occupied = False
            point.reserved_by = None
            print(f"Unloading point {node_id} released")
    
    def cancel_point_reservations(self, vehicle_id: int):
        """取消车辆的装卸点预留"""
        for point in self.loading_points.values():
            if point.reserved_by == vehicle_id and not point.is_occupied:
                point.reserved_by = None
        
        for point in self.unloading_points.values():
            if point.reserved_by == vehicle_id and not point.is_occupied:
                point.reserved_by = None
    
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
        """预留边使用权 - 增加安全缓冲时间"""
        edge_key = tuple(sorted([from_node, to_node]))
        end_time = start_time + duration
        
        safety_buffer = 0.15
        precision_buffer = 0.01
        
        existing_reservations = self.edge_reservations[edge_key]
        for reservation in existing_reservations:
            if not (end_time + safety_buffer + precision_buffer <= reservation.start_time or 
                   start_time >= reservation.end_time + safety_buffer + precision_buffer):
                print(f"EDGE CONFLICT: Edge {edge_key} conflict between V{vehicle_id}({start_time:.2f}-{end_time:.2f}) and V{reservation.vehicle_id}({reservation.start_time:.2f}-{reservation.end_time:.2f}) [safety buffer: {safety_buffer}s]")
                return False
        
        new_reservation = EdgeReservation(
            vehicle_id=vehicle_id,
            start_time=start_time,
            end_time=end_time,
            direction=(from_node, to_node)
        )
        self.edge_reservations[edge_key].append(new_reservation)
        print(f"EDGE RESERVED: Edge {edge_key} by V{vehicle_id} from {start_time:.2f} to {end_time:.2f} [with safety buffer]")
        return True
    
    def reserve_node(self, node: str, vehicle_id: int, 
                    start_time: float, duration: float) -> bool:
        """预留节点使用权 - 增加安全缓冲时间"""
        end_time = start_time + duration
        
        safety_buffer = 0.3
        precision_buffer = 0.01
        
        existing_reservations = self.node_reservations[node]
        for reservation in existing_reservations:
            if not (end_time + safety_buffer + precision_buffer <= reservation.start_time or 
                   start_time >= reservation.end_time + safety_buffer + precision_buffer):
                print(f"NODE CONFLICT: Node {node} conflict between V{vehicle_id}({start_time:.2f}-{end_time:.2f}) and V{reservation.vehicle_id}({reservation.start_time:.2f}-{reservation.end_time:.2f}) [safety buffer: {safety_buffer}s]")
                return False
        
        new_reservation = NodeReservation(
            vehicle_id=vehicle_id,
            start_time=start_time,
            end_time=end_time,
            action="occupy"
        )
        self.node_reservations[node].append(new_reservation)
        print(f"NODE RESERVED: Node {node} by V{vehicle_id} from {start_time:.2f} to {end_time:.2f} [with safety buffer]")
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
                print(f"NODE UNAVAILABLE: Node {node} at {start_time:.2f}-{end_time:.2f} conflicts with V{reservation.vehicle_id}({reservation.start_time:.2f}-{reservation.end_time:.2f}) [safety buffer: {safety_buffer}s]")
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
                print(f"UNAVAILABLE: Edge {edge_key} at {start_time:.2f}-{end_time:.2f} conflicts with V{reservation.vehicle_id}({reservation.start_time:.2f}-{reservation.end_time:.2f}) [safety buffer: {safety_buffer}s]")
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

# ============ 完全继承Vehicle类 ============
class Vehicle:
    """车辆智能体类 - 支持露天矿作业模式（完全继承自demo_GNN.py）"""
    
    def __init__(self, vehicle_id: int, start_node: str, road_network: Stage2RoadNetwork, use_gnn: bool = True):
        self.id = vehicle_id
        self.current_node = start_node
        self.road_network = road_network
        self.use_gnn = use_gnn
        
        # 露天矿作业模式
        self.mode = VehicleMode.EMPTY  # 开始时空载
        self.target_loading_point = None
        self.target_unloading_point = None
        
        self.path = []
        self.path_times = []
        self.path_index = 0
        
        self.position = np.array(road_network.node_positions[start_node], dtype=float)
        self.target_position = self.position.copy()
        self.progress = 0.0
        self.speed = 0.6 + random.random() * 0.4
        
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
        self.completed_cycles = 0  # 完成的装载-卸载循环次数
        self.wait_time = 0.0
        self.loading_time = 2.0    # 装载耗时
        self.unloading_time = 1.5  # 卸载耗时
        self.operation_start_time = 0.0
        
        # 简化颜色方案
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan', 'lime', 'magenta']
        self.color = colors[vehicle_id % len(colors)]
        
        self.road_network.add_vehicle_to_node(self.id, self.current_node)
    
    @property
    def target_node(self):
        """动态目标节点 - 根据模式返回相应的目标"""
        if self.mode == VehicleMode.EMPTY:
            return self.target_loading_point
        else:
            return self.target_unloading_point
    
    def update(self, current_time: float, dt: float):
        """主更新函数 - 露天矿作业逻辑"""
        if self.state == VehicleState.IDLE:
            self._plan_mining_task(current_time)
        elif self.state == VehicleState.PLANNING:
            self._execute_planning(current_time)
        elif self.state == VehicleState.WAITING:
            if current_time >= self.wait_until:
                self.state = VehicleState.IDLE
                self.retry_count += 1
                if self.retry_count > self.max_retries:
                    self._reset_mining_task()
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
    
    def _plan_mining_task(self, current_time: float):
        """规划挖矿任务 - 选择装载点或卸载点"""
        if self.mode == VehicleMode.EMPTY:
            # 空载状态，寻找装载点
            available_loading = self.road_network.get_available_loading_point(exclude_vehicle=self.id)
            if available_loading:
                if self.road_network.reserve_loading_point(available_loading, self.id):
                    self.target_loading_point = available_loading
                    self._plan_path_to_target(current_time)
                else:
                    self._wait_and_retry(current_time)
            else:
                self._wait_and_retry(current_time)
        else:
            # 重载状态，寻找卸载点
            available_unloading = self.road_network.get_available_unloading_point(exclude_vehicle=self.id)
            if available_unloading:
                if self.road_network.reserve_unloading_point(available_unloading, self.id):
                    self.target_unloading_point = available_unloading
                    self._plan_path_to_target(current_time)
                else:
                    self._wait_and_retry(current_time)
            else:
                self._wait_and_retry(current_time)
    
    def _plan_path_to_target(self, current_time: float):
        """规划到目标点的路径"""
        target = self.target_node
        if not target:
            self._wait_and_retry(current_time)
            return
        
        self.state = VehicleState.PLANNING
        self.road_network.cancel_reservations(self.id)
        
        if self.use_gnn:
            self.path, self.path_times = self.road_network.gnn_pathfinding_with_reservation(
                self.current_node, target, self.id, current_time)
        else:
            simple_path = self.road_network.simple_pathfinding(self.current_node, target)
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
        
        mode_text = "Empty->LoadPt" if self.mode == VehicleMode.EMPTY else "Loaded->UnloadPt"
        print(f"Vehicle {self.id} ({mode_text}): Planned path {self.path}")
    
    def _wait_and_retry(self, current_time: float):
        """等待并重试"""
        self.wait_until = current_time + 1.0 + random.random()
        self.state = VehicleState.WAITING
    
    def _reset_mining_task(self):
        """重置挖矿任务"""
        self.road_network.cancel_point_reservations(self.id)
        self.target_loading_point = None
        self.target_unloading_point = None
        self.retry_count = 0
    
    def _arrive_at_node(self, current_time: float):
        """到达节点处理 - 露天矿作业逻辑"""
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
        
        # 检查是否到达目标装卸点
        if self.path_index + 1 >= len(self.path):
            # 路径完成，检查是否到达装卸点
            if self.mode == VehicleMode.EMPTY and self.current_node == self.target_loading_point:
                # 到达装载点，开始装载
                self._start_loading(current_time)
            elif self.mode == VehicleMode.LOADED and self.current_node == self.target_unloading_point:
                # 到达卸载点，开始卸载
                self._start_unloading(current_time)
            else:
                # 其他情况，回到idle状态
                self.state = VehicleState.IDLE
                self.path_confirmed = False
        else:
            # 继续路径
            self._start_next_move(current_time)
    
    def _start_loading(self, current_time: float):
        """开始装载作业"""
        self.road_network.occupy_loading_point(self.current_node, self.id)
        self.state = VehicleState.LOADING
        self.operation_start_time = current_time
        print(f"Vehicle {self.id}: Starting loading at {self.current_node}")
    
    def _start_unloading(self, current_time: float):
        """开始卸载作业"""
        self.road_network.occupy_unloading_point(self.current_node, self.id)
        self.state = VehicleState.UNLOADING
        self.operation_start_time = current_time
        print(f"Vehicle {self.id}: Starting unloading at {self.current_node}")
    
    def _update_loading(self, current_time: float):
        """更新装载状态"""
        if current_time - self.operation_start_time >= self.loading_time:
            # 装载完成
            self.road_network.release_loading_point(self.current_node)
            self.mode = VehicleMode.LOADED
            self.target_loading_point = None
            self.state = VehicleState.IDLE
            print(f"Vehicle {self.id}: Loading completed, switching to loaded mode")
    
    def _update_unloading(self, current_time: float):
        """更新卸载状态"""
        if current_time - self.operation_start_time >= self.unloading_time:
            # 卸载完成
            self.road_network.release_unloading_point(self.current_node)
            self.mode = VehicleMode.EMPTY
            self.target_unloading_point = None
            self.completed_cycles += 1
            self.state = VehicleState.IDLE
            print(f"Vehicle {self.id}: Unloading completed, switching to empty mode, cycle {self.completed_cycles}")
    
    # 保留原有的移动相关方法（完全继承）
    def _execute_planning(self, current_time: float):
        """执行路径规划结果 - 严格冲突避免版本"""
        if not self.path or len(self.path) < 2:
            # 无法找到路径，等待后重试
            self.wait_until = current_time + 1.0 + random.random()
            self.state = VehicleState.WAITING
            return
        
        # 验证并预留整条路径（GNN模式）或检查可行性（简单模式）
        if self.use_gnn:
            success = self._validate_and_reserve_path(current_time)
        else:
            # 简单模式：检查路径是否与当前移动的车辆冲突
            success = self._validate_simple_path(current_time)
        
        if success:
            # 路径验证成功，进入确认状态
            self.path_confirmed = True
            self.path_index = 0
            
            # 设置路径开始时间（比当前时间稍晚一点，确保同步）
            if self.path_times:
                self.path_start_time = max(self.path_times[0], current_time + 0.5)
            else:
                self.path_start_time = current_time + 0.5
            
            self.state = VehicleState.CONFIRMED
            print(f"Vehicle {self.id}: Path confirmed, will start at {self.path_start_time:.1f}s")
        else:
            # 验证失败，等待后重试
            if self.use_gnn:
                self.road_network.cancel_reservations(self.id)
            self.wait_until = current_time + 0.5 + random.random() * 1.0
            self.state = VehicleState.WAITING
            print(f"Vehicle {self.id}: Path validation failed, waiting to retry")
    
    def _validate_simple_path(self, current_time: float) -> bool:
        """简单模式路径验证 - 基本冲突检查"""
        if not self.path_times or len(self.path_times) != len(self.path):
            return False
        
        # 重新调整时间，确保不与正在移动的车辆冲突
        base_time = current_time + 0.5
        
        # 检查是否与其他车辆的移动时间冲突
        other_vehicles = getattr(self.road_network, 'vehicles', [])
        for other_vehicle in other_vehicles:
            if other_vehicle.id == self.id:
                continue
            
            # 如果其他车辆正在移动或已确认路径，避开它们的时间窗口
            if other_vehicle.state in [VehicleState.MOVING, VehicleState.CONFIRMED]:
                if hasattr(other_vehicle, 'path_start_time') and other_vehicle.path_start_time:
                    # 计算其他车辆的预计结束时间
                    if other_vehicle.path_times:
                        other_end_time = other_vehicle.path_times[-1]
                    else:
                        other_end_time = other_vehicle.path_start_time + len(other_vehicle.path) * 1.0
                    
                    # 如果时间窗口重叠，延后开始时间
                    if base_time < other_end_time + 0.5:
                        base_time = other_end_time + 0.5 + random.random() * 0.5
        
        # 重新计算时间序列
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
        """验证并预留整条路径 - 严格边和节点冲突避免"""
        if not self.use_gnn:
            # 简单模式不需要预留，直接返回成功
            return True
        
        if not self.path_times or len(self.path_times) != len(self.path):
            print(f"Vehicle {self.id}: Invalid path times")
            return False
        
        # 重新计算路径时间，确保从当前时间开始
        adjusted_times = []
        base_time = max(current_time + 0.5, self.path_times[0])  # 确保有足够的准备时间
        
        for i, original_time in enumerate(self.path_times):
            if i == 0:
                adjusted_times.append(base_time)
            else:
                # 保持相对时间间隔
                interval = self.path_times[i] - self.path_times[i-1]
                adjusted_times.append(adjusted_times[-1] + interval)
        
        self.path_times = adjusted_times
        
        print(f"Vehicle {self.id}: Attempting to reserve path {self.path} with times {[f'{t:.2f}' for t in self.path_times]}")
        
        # 阶段1: 严格验证整条路径（边和节点）是否可用
        node_duration = 0.4  # 增加节点停留时间，让车辆行为更自然，减少"抢夺"感觉
        
        # 验证所有节点占用
        for i, node in enumerate(self.path):
            node_start_time = self.path_times[i]
            if i == len(self.path) - 1:
                # 最后一个节点，停留时间更长
                node_end_time = node_start_time + node_duration * 3
            else:
                # 中间节点，标准停留时间
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
            edge_start_time = self.path_times[i] + node_duration  # 离开起始节点后开始使用边
            edge_duration = self.path_times[i + 1] - edge_start_time
            
            if edge_duration <= 0:
                print(f"Vehicle {self.id}: Invalid edge duration {edge_duration} for edge {from_node}-{to_node}")
                return False
            
            if not self.road_network.is_edge_available(from_node, to_node, 
                                                     edge_start_time, edge_duration, 
                                                     exclude_vehicle=self.id):
                print(f"Vehicle {self.id}: Edge validation failed at edge {from_node}-{to_node}")
                return False
        
        # 阶段2: 如果验证通过，进行实际预留
        reserved_items = []
        
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
            else:
                reserved_items.append(f"Node({node})")
        
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
            else:
                reserved_items.append(f"Edge({from_node}-{to_node})")
        
        print(f"Vehicle {self.id}: Successfully reserved entire path: {reserved_items}")
        return True
    
    def _start_confirmed_path(self, current_time: float):
        """开始已确认的路径 - 延迟显示版本"""
        if self.path_index + 1 >= len(self.path):
            # 路径已完成
            self.state = VehicleState.IDLE
            return
        
        # 开始第一段移动
        self._start_next_move(current_time)
        print(f"Vehicle {self.id}: Starting confirmed path execution")
        
    def _start_next_move(self, current_time: float):
        """开始下一段移动 - 严格时间控制"""
        if self.path_index + 1 >= len(self.path):
            # 到达目标
            self.state = VehicleState.IDLE
            return
        
        next_node = self.path[self.path_index + 1]
        
        # 设置移动参数
        self.target_position = np.array(
            self.road_network.node_positions[next_node], dtype=float)
        
        if self.use_gnn and self.path_times:
            # 严格按照预留的时间执行
            self.move_start_time = self.path_times[self.path_index]
            self.move_duration = self.path_times[self.path_index + 1] - self.move_start_time
        else:
            self.move_start_time = current_time
            self.move_duration = 1.0 / self.speed
        
        self.progress = 0.0
        self.state = VehicleState.MOVING
        print(f"Vehicle {self.id}: Moving from {self.path[self.path_index]} to {next_node}")
    
    def _update_movement(self, current_time: float, dt: float):
        """更新移动状态 - 延迟显示版本"""
        if current_time < self.move_start_time:
            # 还没到移动时间，保持当前位置不变
            return
        
        # 计算移动进度
        elapsed = current_time - self.move_start_time
        self.progress = min(elapsed / self.move_duration, 1.0)
        
        # 平滑插值位置 - 只有在实际移动时间内才更新位置
        if self.progress > 0:
            start_pos = np.array(self.road_network.node_positions[self.path[self.path_index]])
            # 使用smooth step函数使移动更平滑
            smooth_progress = self._smooth_step(self.progress)
            self.position = start_pos + (self.target_position - start_pos) * smooth_progress
            
            # 更新里程
            if dt > 0:
                distance = np.linalg.norm(self.target_position - start_pos) * (self.progress / (elapsed / dt)) * dt
                self.total_distance += abs(distance) * 0.01  # 缩放因子
        
        # 检查是否到达
        if self.progress >= 1.0:
            self._arrive_at_node(current_time)
    
    def _smooth_step(self, t: float) -> float:
        return t * t * (3.0 - 2.0 * t)

class Stage2GNNSimulation:
    """第二阶段GNN仿真 - 基于第一阶段拓扑结构"""
    
    def __init__(self, topology_file_path: str = None, num_vehicles: int = 4):
        self.topology_file_path = topology_file_path
        self.num_vehicles = num_vehicles
        self.road_network = Stage2RoadNetwork(topology_file_path, num_vehicles)
        self.vehicles = []
        self.use_gnn = True
        self.current_time = 0.0
        
        self._create_initial_vehicles()
        
        # 简洁的布局
        self.fig, (self.ax_main, self.ax_stats) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 主仿真区域
        self._setup_visualization()
        
        self.animation = None
        self.is_running = False
    
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
        
        # 获取拓扑描述
        topo_desc = self.road_network.get_topology_description()
        self.ax_main.set_title(f'Stage 2: GNN Multi-Vehicle Coordination\n{topo_desc}', 
                              fontsize=12, fontweight='bold')
        self.ax_main.grid(True, alpha=0.3)
        
        self.ax_stats.set_xlim(0, 1)
        self.ax_stats.set_ylim(0, 1)
        self.ax_stats.axis('off')
    
    def _create_initial_vehicles(self):
        """创建初始车辆"""
        nodes = list(self.road_network.graph.nodes())
        
        for i in range(self.num_vehicles):
            # 车辆从随机位置开始
            start_node = random.choice(nodes)
            
            vehicle = Vehicle(i, start_node, self.road_network, self.use_gnn)
            self.vehicles.append(vehicle)
        
        # 将车辆列表传递给道路网络
        self.road_network.vehicles = self.vehicles
    
    def toggle_gnn_mode(self):
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
        max_vehicles = min(12, len(self.road_network.loading_points))
        if len(self.vehicles) >= max_vehicles:
            print(f"Maximum vehicles reached! (Limited by loading points: {max_vehicles})")
            return
            
        nodes = list(self.road_network.graph.nodes())
        start_node = random.choice(nodes)
        
        vehicle_id = len(self.vehicles)
        vehicle = Vehicle(vehicle_id, start_node, self.road_network, self.use_gnn)
        self.vehicles.append(vehicle)
        
        # 更新车辆数量并重新配置装卸点
        self.num_vehicles = len(self.vehicles)
        self.road_network.num_vehicles = self.num_vehicles
        self.road_network._setup_mining_points()
        
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
            
            # 更新车辆数量并重新配置装卸点
            self.num_vehicles = len(self.vehicles)
            self.road_network.num_vehicles = self.num_vehicles
            self.road_network._setup_mining_points()
            
            # 更新道路网络中的车辆引用
            self.road_network.vehicles = self.vehicles
            print(f"Removed vehicle {removed.id}, total: {len(self.vehicles)} vehicles")
    
    def update(self, frame):
        dt = 0.1
        self.current_time += dt
        
        self.road_network.update_time(self.current_time)
        
        for vehicle in self.vehicles:
            vehicle.update(self.current_time, dt)
        
        self.ax_main.clear()
        self.ax_stats.clear()
        
        # 重新设置坐标轴
        self._setup_visualization()
        
        self._draw_network()
        self._draw_reservations()
        self._draw_vehicles()
        self._draw_statistics()
        
        return []
    
    def _draw_network(self):
        """绘制网络 - 显示装载点、卸载点和节点预留状态"""
        # 绘制边 - 统一灰色
        for edge in self.road_network.graph.edges():
            node1, node2 = edge
            pos1 = self.road_network.node_positions[node1]
            pos2 = self.road_network.node_positions[node2]
            
            # 根据边权重调整线条粗细
            weight = self.road_network.graph[node1][node2].get('weight', 1.0)
            linewidth = max(0.5, min(3.0, weight))
            
            self.ax_main.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                            color='lightgray', linewidth=linewidth, alpha=0.7)
        
        # 绘制节点 - 区分普通节点、装载点、卸载点
        current_time = self.current_time
        
        for node, pos in self.road_network.node_positions.items():
            vehicle_count = len(self.road_network.node_occupancy[node])
            degree = self.road_network.graph.degree(node)
            
            # 检查节点类型
            is_loading_point = node in self.road_network.loading_points
            is_unloading_point = node in self.road_network.unloading_points
            
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
            
            # 检查冷却期
            cooling_time = 0.3
            recently_vacated = any(
                current_time - r.end_time <= cooling_time and current_time > r.end_time
                for r in node_reservations
            )
            
            # 选择颜色和标记
            if is_loading_point:
                loading_point = self.road_network.loading_points[node]
                if loading_point.is_occupied:
                    color = 'darkred'
                    marker = '■'  # Loading
                    status = f"Loading V{loading_point.reserved_by}"
                elif loading_point.reserved_by is not None:
                    color = 'orange'
                    marker = '□'  # Reserved loading point
                    status = f"Reserved V{loading_point.reserved_by}"
                else:
                    color = 'green'
                    marker = '□'  # Available loading point
                    status = "LoadPt"
                size = 25  # 显示大小，修复可视化问题
                alpha = 0.9
            elif is_unloading_point:
                unloading_point = self.road_network.unloading_points[node]
                if unloading_point.is_occupied:
                    color = 'darkblue'
                    marker = '▼'  # Unloading
                    status = f"Unload V{unloading_point.reserved_by}"
                elif unloading_point.reserved_by is not None:
                    color = 'orange'
                    marker = '△'  # Reserved unloading point
                    status = f"Reserved V{unloading_point.reserved_by}"
                else:
                    color = 'blue'
                    marker = '△'  # Available unloading point
                    status = "UnloadPt"
                size = 25
                alpha = 0.9
            else:
                # 普通节点
                if degree == 2:
                    base_color = 'lightblue'
                elif degree == 3:
                    base_color = 'lightgreen'
                elif degree == 4:
                    base_color = 'lightcoral'
                elif degree == 5:
                    base_color = 'plum'
                else:
                    base_color = 'lightyellow'
                
                if has_active_reservation:
                    color = 'red'
                    alpha = 0.9
                    status = "OCCUPIED"
                elif recently_vacated:
                    color = 'yellow'
                    alpha = 0.8
                    status = "COOLING"
                elif has_future_reservation:
                    color = 'orange'
                    alpha = 0.8
                    status = "RESERVED"
                elif vehicle_count > 0:
                    color = 'pink'
                    alpha = 0.8
                    status = "PRESENT"
                else:
                    color = base_color
                    alpha = 0.8
                    status = "FREE"
                
                marker = '●'
                size = 10 + degree * 3
            
            # 绘制节点
            circle = Circle(pos, size*0.01, color=color, 
                          edgecolor='navy', linewidth=1, alpha=alpha)
            self.ax_main.add_patch(circle)
            
            # 显示节点信息
            if is_loading_point or is_unloading_point:
                # 装卸点显示特殊信息
                label = f"{node[-6:]}\n{marker}\n{status}"
            else:
                # 普通节点显示度数
                label = f"{node[-4:]}\n({degree})"
                if has_active_reservation:
                    active_vehicle = None
                    for r in node_reservations:
                        if r.start_time <= current_time <= r.end_time:
                            active_vehicle = r.vehicle_id
                            break
                    if active_vehicle is not None:
                        label += f"\nV{active_vehicle}"
                elif recently_vacated:
                    last_end_time = max(r.end_time for r in node_reservations if current_time > r.end_time)
                    remaining_cooling = cooling_time - (current_time - last_end_time)
                    label += f"\ncooling{remaining_cooling:.1f}s"
            
            self.ax_main.text(pos[0], pos[1], label, ha='center', va='center', 
                            fontsize=6, fontweight='bold')
    
    def _draw_reservations(self):
        """突出显示边预留"""
        current_time = self.current_time
        
        for edge_key, reservations in self.road_network.edge_reservations.items():
            if not reservations:
                continue
                
            node1, node2 = edge_key
            pos1 = self.road_network.node_positions[node1]
            pos2 = self.road_network.node_positions[node2]
            
            # 为每个预留绘制不同颜色的线
            for i, reservation in enumerate(reservations):
                if reservation.end_time < current_time:
                    continue
                
                # 获取对应车辆颜色
                vehicle = next((v for v in self.vehicles if v.id == reservation.vehicle_id), None)
                if vehicle:
                    # 计算偏移以显示多个预留
                    offset_factor = (i - len(reservations)/2 + 0.5) * 0.1
                    offset_x = (pos2[1] - pos1[1]) * offset_factor
                    offset_y = (pos1[0] - pos2[0]) * offset_factor
                    
                    x1, y1 = pos1[0] + offset_x, pos1[1] + offset_y
                    x2, y2 = pos2[0] + offset_x, pos2[1] + offset_y
                    
                    # 预留线条 - 粗线突出显示
                    self.ax_main.plot([x1, x2], [y1, y2], 
                                    color=vehicle.color, linewidth=4, alpha=0.7,
                                    label=f'Reserved by V{vehicle.id}' if i == 0 else "")
                    
                    # 在线条中间标注车辆ID
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    self.ax_main.text(mid_x, mid_y, f'V{vehicle.id}', 
                                    ha='center', va='center', fontsize=6,
                                    bbox=dict(boxstyle='round,pad=0.2', 
                                            facecolor='white', alpha=0.8))
    
    def _draw_vehicles(self):
        """绘制车辆 - 区分空载/重载和作业状态"""
        for vehicle in self.vehicles:
            x, y = vehicle.position
            
            # 根据状态调整视觉效果
            if vehicle.state == VehicleState.LOADING:
                # 装载中 - 绿色边框，较大尺寸
                alpha = 1.0
                edge_color = 'green'
                edge_width = 4
                size = 16
                symbol = 'L'
            elif vehicle.state == VehicleState.UNLOADING:
                # 卸载中 - 蓝色边框，较大尺寸
                alpha = 1.0
                edge_color = 'blue'
                edge_width = 4
                size = 16
                symbol = 'U'
            elif vehicle.state == VehicleState.CONFIRMED:
                # 路径已确认，等待开始时间 - 金色边框
                alpha = 0.95
                edge_color = 'gold'
                edge_width = 4
                size = 14
                symbol = 'C'
            elif vehicle.state == VehicleState.MOVING:
                # 正在移动 - 根据载重状态调整
                alpha = 1.0
                edge_color = 'white'
                edge_width = 2
                size = 14 if vehicle.mode == VehicleMode.LOADED else 12
                symbol = 'M'
            elif vehicle.state == VehicleState.WAITING:
                # 等待中 - 暗淡显示
                alpha = 0.5
                edge_color = 'red'
                edge_width = 2
                size = 10
                symbol = 'W'
            elif vehicle.state == VehicleState.PLANNING:
                # 规划中 - 橙色边框
                alpha = 0.7
                edge_color = 'orange'
                edge_width = 2
                size = 11
                symbol = 'P'
            else:  # IDLE, BLOCKED
                alpha = 0.8
                edge_color = 'black'
                edge_width = 1
                size = 11
                symbol = 'I'
            
            # 车辆形状 - 空载用圆形，重载用方形
            if vehicle.mode == VehicleMode.LOADED:
                # 重载 - 方形，更大
                rect = Rectangle((x-size*0.01, y-size*0.01), size*0.02, size*0.02, 
                               color=vehicle.color, alpha=alpha, 
                               edgecolor=edge_color, linewidth=edge_width)
                self.ax_main.add_patch(rect)
                mode_symbol = '■'
            else:
                # 空载 - 圆形
                circle = Circle((x, y), size*0.01, color=vehicle.color, alpha=alpha,
                              edgecolor=edge_color, linewidth=edge_width)
                self.ax_main.add_patch(circle)
                mode_symbol = '○'
            
            # 车辆ID、状态和载重标识
            display_text = f'{vehicle.id}\n{symbol}\n{mode_symbol}'
            
            self.ax_main.text(x, y, display_text, 
                            ha='center', va='center',
                            color='white', fontweight='bold', fontsize=7)
            
            # 目标连线 - 虚线，根据任务类型调整颜色
            target_node = vehicle.target_node
            if target_node and target_node in self.road_network.node_positions:
                target_pos = self.road_network.node_positions[target_node]
                
                # 根据任务类型选择线条颜色
                if vehicle.mode == VehicleMode.EMPTY:
                    line_color = 'green'  # 前往装载点
                    line_style = '-.'
                else:
                    line_color = 'blue'   # 前往卸载点
                    line_style = '--'
                
                line_alpha = 0.8 if vehicle.state == VehicleState.CONFIRMED else 0.5
                self.ax_main.plot([x, target_pos[0]], [y, target_pos[1]], 
                                color=line_color, linestyle=line_style, 
                                alpha=line_alpha, linewidth=2)
                
                # 目标标记 - 装载点用方形，卸载点用三角形
                if vehicle.mode == VehicleMode.EMPTY:
                    marker = 's'  # 方形代表装载点
                    marker_color = 'green'
                else:
                    marker = '^'  # 三角形代表卸载点
                    marker_color = 'blue'
                
                self.ax_main.scatter(target_pos[0], target_pos[1], 
                                   s=150, color=marker_color, marker=marker, 
                                   alpha=0.8, edgecolors='black', linewidths=1)
            
            # 已确认路径 - 重点突出显示
            if vehicle.state == VehicleState.CONFIRMED and vehicle.path:
                self._draw_confirmed_path(vehicle)
            
            # 为确认状态的车辆添加特殊标识
            if vehicle.state == VehicleState.CONFIRMED:
                # 在车辆周围绘制脉冲圆环表示等待开始
                pulse_radius = 2.5 + 1.0 * np.sin(self.current_time * 4)
                pulse_circle = Circle((x, y), pulse_radius, 
                                    facecolor='none', edgecolor='gold', 
                                    linewidth=2, alpha=0.6)
                self.ax_main.add_patch(pulse_circle)
            
            # 装载/卸载进度显示
            if vehicle.state in [VehicleState.LOADING, VehicleState.UNLOADING]:
                # 显示作业进度
                if vehicle.state == VehicleState.LOADING:
                    total_time = vehicle.loading_time
                    progress_color = 'green'
                else:
                    total_time = vehicle.unloading_time
                    progress_color = 'blue'
                
                elapsed = self.current_time - vehicle.operation_start_time
                progress = min(elapsed / total_time, 1.0)
                
                # 进度条
                bar_width = 3.0
                bar_height = 0.5
                bar_x = x - bar_width/2
                bar_y = y + size*0.01 + 1.0
                
                # 背景
                bg_rect = Rectangle((bar_x, bar_y), bar_width, bar_height,
                                  color='white', alpha=0.8, edgecolor='black')
                self.ax_main.add_patch(bg_rect)
                
                # 进度
                progress_rect = Rectangle((bar_x, bar_y), bar_width * progress, bar_height,
                                        color=progress_color, alpha=0.9)
                self.ax_main.add_patch(progress_rect)
                
                # 进度文字
                self.ax_main.text(x, bar_y + bar_height/2, f'{progress*100:.0f}%',
                                ha='center', va='center', fontsize=6, fontweight='bold')
    
    def _draw_confirmed_path(self, vehicle):
        """突出显示已确认的路径"""
        if len(vehicle.path) < 2:
            return
        
        path_positions = [self.road_network.node_positions[node] for node in vehicle.path]
        xs = [pos[0] for pos in path_positions]
        ys = [pos[1] for pos in path_positions]
        
        # 粗线显示确认路径
        self.ax_main.plot(xs, ys, color=vehicle.color, linewidth=6, 
                         alpha=0.4, linestyle='-', zorder=1)
        
        # 路径节点序号
        for i, pos in enumerate(path_positions):
            self.ax_main.text(pos[0]-3, pos[1]+3, str(i), 
                            ha='center', va='center', fontsize=8,
                            bbox=dict(boxstyle='circle,pad=0.1', 
                                    facecolor=vehicle.color, alpha=0.7),
                            color='white', fontweight='bold')
    
    def _draw_statistics(self):
        """绘制露天矿作业统计信息"""
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
        
        # 装卸点使用统计
        loading_occupied = sum(1 for p in self.road_network.loading_points.values() if p.is_occupied)
        loading_reserved = sum(1 for p in self.road_network.loading_points.values() if p.reserved_by is not None and not p.is_occupied)
        loading_available = len(self.road_network.loading_points) - loading_occupied - loading_reserved
        
        unloading_occupied = sum(1 for p in self.road_network.unloading_points.values() if p.is_occupied)
        unloading_reserved = sum(1 for p in self.road_network.unloading_points.values() if p.reserved_by is not None and not p.is_occupied)
        unloading_available = len(self.road_network.unloading_points) - unloading_occupied - unloading_reserved
        
        # 预留统计
        total_edge_reservations = sum(len(reservations) for reservations in 
                                    self.road_network.edge_reservations.values())
        total_node_reservations = sum(len(reservations) for reservations in 
                                    self.road_network.node_reservations.values())
        
        # 拓扑统计
        total_edges = len(self.road_network.graph.edges())
        total_nodes = len(self.road_network.graph.nodes())
        topo_info = self.road_network.topology_info
        
        stats_text = f"""
        ╔═══ STAGE 2 GNN SYSTEM ═══╗
        ║ Source: {topo_info.get('topology_source', 'Unknown')[:13]} ║
        ║ Mode: {'GNN' if self.use_gnn else 'Simple':>17} ║
        ║ Time: {self.current_time:>17.1f}s ║
        ║ Vehicles: {len(self.vehicles):>13d} ║
        ║ Nodes: {total_nodes:>16d} ║
        ║ Edges: {total_edges:>16d} ║
        ╠═══ OPERATION STATS ══════╣
        ║ Cycles: {total_cycles:>15d} ║
        ║ Distance: {total_distance:>13.1f} ║
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
        ╠═══ LOAD STATUS ══════════╣
        ║ Empty: {mode_counts[VehicleMode.EMPTY]:>16d} ║
        ║ Loaded: {mode_counts[VehicleMode.LOADED]:>15d} ║
        ╠═══ LOADING POINTS ═══════╣
        ║ In Use: {loading_occupied:>15d} ║
        ║ Reserved: {loading_reserved:>13d} ║
        ║ Available: {loading_available:>12d} ║
        ╠═══ UNLOADING POINTS ═════╣
        ║ In Use: {unloading_occupied:>15d} ║
        ║ Reserved: {unloading_reserved:>13d} ║
        ║ Available: {unloading_available:>12d} ║
        ╠═══ RESERVATION STATUS ═══╣
        ║ Edge Rsv: {total_edge_reservations:>13d} ║
        ║ Node Rsv: {total_node_reservations:>13d} ║
        ╠═══ TOPOLOGY INFO ════════╣
        ║ Enhanced: {'Yes' if topo_info.get('enhanced_consolidation', False) else 'No':>15s} ║
        ║ GNN Ready: {'Yes' if topo_info.get('gnn_input_ready', False) else 'No':>14s} ║
        ╚═══════════════════════════╝
        
        Stage 2 Operations:
        🚛 Multi-vehicle coordination based on 
           Stage 1 topology optimization
        📍 LoadPt: Green squares □ 
        📍 UnloadPt: Blue triangles △ 
        
        Vehicle Display:
        ○ Empty vehicles (circles)
        ■ Loaded vehicles (squares)  
        L Loading, U Unloading
        Green→LoadPt, Blue→UnloadPt
        
        Status Colors:
        🔴 Red: Currently occupied
        🟡 Yellow: Cooling (0.3s)
        🟠 Orange: Reserved
        🟢 Green: Available LoadPt
        🔵 Blue: Available UnloadPt
        
        Topology Features:
        ✅ Stage 1 optimized network
        ✅ GNN-aware pathfinding
        ✅ Multi-vehicle coordination
        ✅ Conflict-free scheduling
        
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
    
    def start_animation(self):
        if not self.is_running:
            self.animation = animation.FuncAnimation(
                self.fig, self.update, interval=100, blit=False)
            self.is_running = True
            print("Stage 2 simulation started")
    
    def stop_animation(self):
        if self.animation:
            self.animation.event_source.stop()
            self.is_running = False
            print("Stage 2 simulation stopped")
    
    def reset_simulation(self):
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
    print("🎯 第二阶段特点:")
    print("📁 读取第一阶段导出的拓扑结构")
    print("🧠 基于优化拓扑的GNN多车协调")
    print("🚛 继承完整的露天矿作业逻辑")
    print("⚡ 保持所有冲突避免和安全机制")
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
    print("✅ 第一阶段优化拓扑 - 关键节点和曲线拟合路径")
    print("✅ 智能装卸点分配 - 基于网络拓扑特征")
    print("✅ GNN感知路径规划 - 利用拓扑结构优化")
    print("✅ 多车协同调度 - 避免冲突的时空预留")
    print("✅ 完整作业循环 - 装载→卸载→装载")
    print()
    print("🎨 可视化增强:")
    print("🟢 优化拓扑显示 - 基于第一阶段结果")
    print("🔵 智能节点标识 - 度数和功能分类")
    print("🚛 完整车辆状态 - 空载/重载/作业进度")
    print("📊 拓扑来源信息 - 增强版/原始/回退")
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
    except Exception as e:
        print(f"❌ 创建第二阶段仿真失败: {e}")
        print("🔄 使用回退拓扑...")
        sim = Stage2GNNSimulation(topology_file_path=None, num_vehicles=4)
    
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
    sim.start_animation()
    sim.show()

if __name__ == "__main__":
    main()