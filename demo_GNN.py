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

class NetworkTopology:
    """网络拓扑生成器"""
    
    @staticmethod
    def create_grid(size: int) -> Tuple[nx.Graph, Dict[str, Tuple[float, float]]]:
        """创建网格拓扑"""
        G = nx.Graph()
        positions = {}
        
        for i in range(size):
            for j in range(size):
                node_id = f"{i}-{j}"
                G.add_node(node_id)
                positions[node_id] = (j, i)
        
        for i in range(size):
            for j in range(size):
                current = f"{i}-{j}"
                if j < size - 1:
                    right = f"{i}-{j+1}"
                    G.add_edge(current, right, weight=1.0)
                if i < size - 1:
                    down = f"{i+1}-{j}"
                    G.add_edge(current, down, weight=1.0)
        
        return G, positions
    
    @staticmethod
    def create_star(center_node: str, num_spokes: int) -> Tuple[nx.Graph, Dict[str, Tuple[float, float]]]:
        """创建星形拓扑"""
        G = nx.Graph()
        positions = {}
        
        # 中心节点
        G.add_node(center_node)
        positions[center_node] = (0, 0)
        
        # 辐射节点
        for i in range(num_spokes):
            angle = 2 * np.pi * i / num_spokes
            node_id = f"spoke-{i}"
            G.add_node(node_id)
            positions[node_id] = (2 * np.cos(angle), 2 * np.sin(angle))
            G.add_edge(center_node, node_id, weight=1.0)
            
            # 每个辐射添加一些分支
            for j in range(2):
                branch_id = f"branch-{i}-{j}"
                G.add_node(branch_id)
                branch_angle = angle + (j - 0.5) * 0.5
                positions[branch_id] = (3.5 * np.cos(branch_angle), 3.5 * np.sin(branch_angle))
                G.add_edge(node_id, branch_id, weight=1.0)
        
        return G, positions
    
    @staticmethod
    def create_ring_with_chords(num_nodes: int) -> Tuple[nx.Graph, Dict[str, Tuple[float, float]]]:
        """创建带弦的环形拓扑"""
        G = nx.Graph()
        positions = {}
        
        # 创建环形
        for i in range(num_nodes):
            angle = 2 * np.pi * i / num_nodes
            node_id = f"ring-{i}"
            G.add_node(node_id)
            positions[node_id] = (3 * np.cos(angle), 3 * np.sin(angle))
        
        # 连接环形
        for i in range(num_nodes):
            current = f"ring-{i}"
            next_node = f"ring-{(i + 1) % num_nodes}"
            G.add_edge(current, next_node, weight=1.0)
        
        # 添加一些弦（跨越连接）
        for i in range(0, num_nodes, 3):
            chord_target = (i + num_nodes // 2) % num_nodes
            G.add_edge(f"ring-{i}", f"ring-{chord_target}", weight=0.8)
        
        return G, positions
    
    @staticmethod
    def create_tree(depth: int, branching: int) -> Tuple[nx.Graph, Dict[str, Tuple[float, float]]]:
        """创建树形拓扑"""
        G = nx.Graph()
        positions = {}
        
        def add_tree_nodes(node_id: str, level: int, x: float, y: float, width: float):
            G.add_node(node_id)
            positions[node_id] = (x, y)
            
            if level < depth:
                child_width = width / branching
                start_x = x - width / 2 + child_width / 2
                
                for i in range(branching):
                    child_id = f"{node_id}-{i}"
                    child_x = start_x + i * child_width
                    child_y = y - 1.5
                    
                    add_tree_nodes(child_id, level + 1, child_x, child_y, child_width)
                    G.add_edge(node_id, child_id, weight=1.0)
        
        add_tree_nodes("root", 0, 0, 0, 8)
        return G, positions
    
    @staticmethod
    def create_small_world(n: int, k: int, p: float) -> Tuple[nx.Graph, Dict[str, Tuple[float, float]]]:
        """创建小世界网络"""
        G = nx.watts_strogatz_graph(n, k, p)
        
        # 创建环形布局
        positions = {}
        for i, node in enumerate(G.nodes()):
            angle = 2 * np.pi * i / n
            positions[str(node)] = (3 * np.cos(angle), 3 * np.sin(angle))
        
        # 重命名节点为字符串
        mapping = {node: str(node) for node in G.nodes()}
        G = nx.relabel_nodes(G, mapping)
        
        # 设置边权重
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1.0
        
        return G, positions

    @staticmethod
    def create_irregular_network(num_nodes: int = 25, width: float = 10.0, height: float = 8.0) -> Tuple[nx.Graph, Dict[str, Tuple[float, float]]]:
        """创建不规则网络 - 节点随机分布，度数2-5"""
        G = nx.Graph()
        positions = {}
        
        # 随机生成节点位置
        random.seed(42)  # 固定种子确保可重现
        for i in range(num_nodes):
            node_id = f"node-{i}"
            # 使用稍微聚集的分布而不是完全随机
            x = random.uniform(0, width) + random.gauss(0, 0.5)
            y = random.uniform(0, height) + random.gauss(0, 0.5)
            x = max(0, min(width, x))  # 确保在边界内
            y = max(0, min(height, y))
            
            G.add_node(node_id)
            positions[node_id] = (x, y)
        
        # 基于距离和度数限制连接节点
        nodes = list(G.nodes())
        
        # 计算所有节点间距离
        distances = {}
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i+1:], i+1):
                pos1 = positions[node1]
                pos2 = positions[node2]
                dist = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                distances[(node1, node2)] = dist
        
        # 按距离排序连接
        sorted_edges = sorted(distances.items(), key=lambda x: x[1])
        
        # 贪婪地添加边，确保度数在2-5之间且图连通
        for (node1, node2), dist in sorted_edges:
            degree1 = G.degree(node1)
            degree2 = G.degree(node2)
            
            # 如果两个节点度数都小于5，且至少有一个度数小于2，则连接
            if degree1 < 5 and degree2 < 5:
                if degree1 < 2 or degree2 < 2 or (not nx.is_connected(G) and degree1 < 4 and degree2 < 4):
                    G.add_edge(node1, node2, weight=max(0.5, min(2.0, dist/2.0)))
        
        # 确保图连通，如果不连通则强制连接最近的组件
        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            for i in range(len(components) - 1):
                # 找到两个组件间最近的节点对
                min_dist = float('inf')
                best_pair = None
                
                for node1 in components[i]:
                    for node2 in components[i+1]:
                        pos1 = positions[node1]
                        pos2 = positions[node2]
                        dist = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                        if dist < min_dist:
                            min_dist = dist
                            best_pair = (node1, node2)
                
                if best_pair and G.degree(best_pair[0]) < 5 and G.degree(best_pair[1]) < 5:
                    G.add_edge(best_pair[0], best_pair[1], weight=max(0.5, min(2.0, min_dist/2.0)))
        
        return G, positions
    
    @staticmethod
    def create_city_roads(grid_size: int = 8) -> Tuple[nx.Graph, Dict[str, Tuple[float, float]]]:
        """创建城市道路网络 - 主干道 + 支路"""
        G = nx.Graph()
        positions = {}
        
        # 创建基础网格作为主干道
        main_roads = []
        
        # 垂直主干道
        for i in range(0, grid_size + 1, 2):
            for j in range(grid_size):
                node_id = f"main-v-{i}-{j}"
                G.add_node(node_id)
                positions[node_id] = (i, j)
                main_roads.append(node_id)
                
                if j > 0:
                    prev_node = f"main-v-{i}-{j-1}"
                    G.add_edge(prev_node, node_id, weight=0.8)  # 主干道权重较小
        
        # 水平主干道
        for j in range(0, grid_size + 1, 2):
            for i in range(grid_size):
                node_id = f"main-h-{i}-{j}"
                if node_id not in G:
                    G.add_node(node_id)
                    positions[node_id] = (i, j)
                    main_roads.append(node_id)
                
                if i > 0:
                    prev_node = f"main-h-{i-1}-{j}"
                    if prev_node in G:
                        G.add_edge(prev_node, node_id, weight=0.8)
        
        # 连接主干道交叉点
        for i in range(0, grid_size + 1, 2):
            for j in range(0, grid_size + 1, 2):
                v_node = f"main-v-{i}-{j}"
                h_node = f"main-h-{i}-{j}"
                if v_node in G and h_node in G and v_node != h_node:
                    # 合并为一个交叉口节点
                    intersection = f"intersection-{i}-{j}"
                    if intersection not in G:
                        G.add_node(intersection)
                        positions[intersection] = (i, j)
                    
                    # 连接到相邻的主干道节点
                    for neighbor in list(G.neighbors(v_node)) + list(G.neighbors(h_node)):
                        if not G.has_edge(intersection, neighbor):
                            G.add_edge(intersection, neighbor, weight=0.8)
                    
                    # 移除原来的节点
                    if v_node in G:
                        G.remove_node(v_node)
                    if h_node in G:
                        G.remove_node(h_node)
        
        # 添加支路和小巷
        support_nodes = []
        for i in range(grid_size):
            for j in range(grid_size):
                if (i % 2 == 1 or j % 2 == 1) and f"intersection-{i}-{j}" not in G:
                    # 添加一些随机扰动让布局更自然
                    offset_x = random.uniform(-0.3, 0.3)
                    offset_y = random.uniform(-0.3, 0.3)
                    
                    node_id = f"support-{i}-{j}"
                    G.add_node(node_id)
                    positions[node_id] = (i + offset_x, j + offset_y)
                    support_nodes.append(node_id)
        
        # 连接支路到主干道
        for support_node in support_nodes:
            pos = positions[support_node]
            # 找到最近的主干道节点
            min_dist = float('inf')
            nearest_main = None
            
            for main_node in G.nodes():
                if main_node.startswith('intersection-') or main_node.startswith('main-'):
                    main_pos = positions[main_node]
                    dist = math.sqrt((pos[0] - main_pos[0])**2 + (pos[1] - main_pos[1])**2)
                    if dist < min_dist and dist < 2.0:  # 只连接较近的
                        min_dist = dist
                        nearest_main = main_node
            
            if nearest_main and G.degree(support_node) < 3:
                G.add_edge(support_node, nearest_main, weight=1.2)  # 支路权重稍大
        
        # 在支路间添加一些连接
        for i, node1 in enumerate(support_nodes):
            for node2 in support_nodes[i+1:]:
                if G.degree(node1) < 4 and G.degree(node2) < 4:
                    pos1 = positions[node1]
                    pos2 = positions[node2]
                    dist = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    
                    # 只连接很近的支路节点，形成小巷
                    if dist < 1.5 and random.random() < 0.3:
                        G.add_edge(node1, node2, weight=1.5)
        
        return G, positions
    
    @staticmethod
    def create_suburban_network(num_districts: int = 4, district_size: int = 6) -> Tuple[nx.Graph, Dict[str, Tuple[float, float]]]:
        """创建郊区网络 - 多个小区域通过主干道连接"""
        G = nx.Graph()
        positions = {}
        
        # 每个区域的中心位置
        district_centers = []
        for i in range(num_districts):
            angle = 2 * math.pi * i / num_districts
            radius = 6
            center_x = radius * math.cos(angle)
            center_y = radius * math.sin(angle)
            district_centers.append((center_x, center_y))
        
        # 为每个区域创建小型网络
        main_nodes = []  # 每个区域的主要节点
        
        for district_id, (center_x, center_y) in enumerate(district_centers):
            district_nodes = []
            
            # 在每个区域创建随机分布的节点
            for i in range(district_size):
                # 使用极坐标生成更自然的分布
                r = random.uniform(0.5, 2.5)
                theta = random.uniform(0, 2 * math.pi)
                x = center_x + r * math.cos(theta)
                y = center_y + r * math.sin(theta)
                
                node_id = f"d{district_id}-{i}"
                G.add_node(node_id)
                positions[node_id] = (x, y)
                district_nodes.append(node_id)
            
            # 区域内连接 - 确保度数2-4
            for i, node1 in enumerate(district_nodes):
                for j, node2 in enumerate(district_nodes[i+1:], i+1):
                    if G.degree(node1) < 4 and G.degree(node2) < 4:
                        pos1 = positions[node1]
                        pos2 = positions[node2]
                        dist = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                        
                        # 连接较近的节点
                        if dist < 2.0 and (G.degree(node1) < 2 or G.degree(node2) < 2 or random.random() < 0.4):
                            G.add_edge(node1, node2, weight=1.0)
            
            # 选择一个主要节点作为区域中心
            if district_nodes:
                main_node = district_nodes[0]  # 或选择度数最高的
                main_nodes.append(main_node)
        
        # 在区域间添加主干道连接
        for i, main1 in enumerate(main_nodes):
            for j, main2 in enumerate(main_nodes[i+1:], i+1):
                # 添加区域间的主干道
                if random.random() < 0.7:  # 不是所有区域都直接相连
                    G.add_edge(main1, main2, weight=2.0)  # 主干道权重较大
        
        # 添加一些区域间的次要连接
        all_nodes = list(G.nodes())
        for i, node1 in enumerate(all_nodes):
            district1 = int(node1.split('-')[0][1:])
            for node2 in all_nodes[i+1:]:
                district2 = int(node2.split('-')[0][1:])
                
                # 不同区域间的连接
                if district1 != district2 and G.degree(node1) < 5 and G.degree(node2) < 5:
                    pos1 = positions[node1]
                    pos2 = positions[node2]
                    dist = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    
                    # 很小概率添加跨区域连接
                    if dist < 4.0 and random.random() < 0.1:
                        G.add_edge(node1, node2, weight=1.8)
        
        return G, positions
    
    @staticmethod
    def create_highway_network(num_cities: int = 6) -> Tuple[nx.Graph, Dict[str, Tuple[float, float]]]:
        """创建高速公路网络 - 城市 + 高速公路连接"""
        G = nx.Graph()
        positions = {}
        
        # 创建主要城市节点
        cities = []
        for i in range(num_cities):
            # 城市分布在一个大的区域内
            x = random.uniform(-8, 8)
            y = random.uniform(-6, 6)
            city_id = f"city-{i}"
            G.add_node(city_id)
            positions[city_id] = (x, y)
            cities.append(city_id)
        
        # 为每个城市添加周边节点（郊区/卫星城）
        all_nodes = cities.copy()
        
        for city in cities:
            city_pos = positions[city]
            # 每个城市周围2-4个卫星节点
            num_satellites = random.randint(2, 4)
            
            for j in range(num_satellites):
                angle = 2 * math.pi * j / num_satellites + random.uniform(-0.5, 0.5)
                radius = random.uniform(1.5, 3.0)
                sat_x = city_pos[0] + radius * math.cos(angle)
                sat_y = city_pos[1] + radius * math.sin(angle)
                
                sat_id = f"sat-{city.split('-')[1]}-{j}"
                G.add_node(sat_id)
                positions[sat_id] = (sat_x, sat_y)
                all_nodes.append(sat_id)
                
                # 连接到主城市
                G.add_edge(city, sat_id, weight=1.0)
        
        # 创建高速公路主干网 - 连接主要城市
        for i, city1 in enumerate(cities):
            for city2 in cities[i+1:]:
                pos1 = positions[city1]
                pos2 = positions[city2]
                dist = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                
                # 距离较近的城市用高速公路连接
                if dist < 8.0:
                    G.add_edge(city1, city2, weight=0.6)  # 高速公路很快
        
        # 添加区域间的次要连接
        for node1 in all_nodes:
            for node2 in all_nodes:
                if node1 != node2 and not G.has_edge(node1, node2):
                    if G.degree(node1) < 5 and G.degree(node2) < 5:
                        pos1 = positions[node1]
                        pos2 = positions[node2]
                        dist = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                        
                        # 短距离连接
                        if dist < 3.0 and random.random() < 0.3:
                            G.add_edge(node1, node2, weight=1.2)
        
        return G, positions

class SimpleRoadNetwork:
    """简化的道路网络类 - 专注于算法验证，支持露天矿场景"""
    
    def __init__(self, topology_type: str = "grid", topology_size: int = 6, num_vehicles: int = 6):
        self.topology_type = topology_type
        self.topology_size = topology_size
        self.num_vehicles = num_vehicles
        self.graph = nx.Graph()
        self.node_positions = {}
        self.edge_reservations = defaultdict(list)
        self.node_reservations = defaultdict(list)  # 节点预留
        self.node_occupancy = defaultdict(set)
        self.node_features = {}
        self.global_time = 0.0
        
        # 露天矿场景相关
        self.loading_points = {}    # 装载点字典 {node_id: LoadingPoint}
        self.unloading_points = {}  # 卸载点字典 {node_id: UnloadingPoint}
        
        self._create_topology()
        self._setup_mining_points()
        self._initialize_features()
    
    def _setup_mining_points(self):
        """设置露天矿的装载点和卸载点 - 在网络边缘"""
        if not self.graph.nodes():
            return
            
        # 找到边缘节点（度数较小的节点，或者在网络边界的节点）
        edge_nodes = self._find_edge_nodes()
        
        if len(edge_nodes) < self.num_vehicles * 2:
            # 如果边缘节点不够，选择所有度数最小的节点
            degrees = [(node, self.graph.degree(node)) for node in self.graph.nodes()]
            degrees.sort(key=lambda x: x[1])
            edge_nodes = [node for node, _ in degrees[:self.num_vehicles * 2]]
        
        # 分配装载点和卸载点
        selected_nodes = random.sample(edge_nodes, min(self.num_vehicles * 2, len(edge_nodes)))
        
        # 前一半作为装载点，后一半作为卸载点
        loading_nodes = selected_nodes[:self.num_vehicles]
        unloading_nodes = selected_nodes[self.num_vehicles:self.num_vehicles * 2]
        
        # 如果节点不够，重复使用
        while len(loading_nodes) < self.num_vehicles:
            loading_nodes.extend(selected_nodes[:self.num_vehicles - len(loading_nodes)])
        while len(unloading_nodes) < self.num_vehicles:
            unloading_nodes.extend(selected_nodes[:self.num_vehicles - len(unloading_nodes)])
        
        # 创建装载点
        for i, node in enumerate(loading_nodes[:self.num_vehicles]):
            self.loading_points[node] = LoadingPoint(node_id=node)
        
        # 创建卸载点
        for i, node in enumerate(unloading_nodes[:self.num_vehicles]):
            self.unloading_points[node] = UnloadingPoint(node_id=node)
        
        print(f"Open-pit mine setup complete:")
        print(f"Loading points: {list(self.loading_points.keys())}")
        print(f"Unloading points: {list(self.unloading_points.keys())}")
    
    def _find_edge_nodes(self):
        """找到网络边缘的节点"""
        if not self.node_positions:
            return list(self.graph.nodes())
        
        # 计算坐标范围
        x_coords = [pos[0] for pos in self.node_positions.values()]
        y_coords = [pos[1] for pos in self.node_positions.values()]
        
        if not x_coords or not y_coords:
            return list(self.graph.nodes())
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # 定义边界阈值
        x_margin = (x_max - x_min) * 0.2
        y_margin = (y_max - y_min) * 0.2
        
        edge_nodes = []
        for node, pos in self.node_positions.items():
            x, y = pos
            # 检查是否在边界附近
            is_edge = (x <= x_min + x_margin or x >= x_max - x_margin or 
                      y <= y_min + y_margin or y >= y_max - y_margin)
            
            # 或者度数较小（连接少）
            is_low_degree = self.graph.degree(node) <= 3
            
            if is_edge or is_low_degree:
                edge_nodes.append(node)
        
        return edge_nodes if edge_nodes else list(self.graph.nodes())
    
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
    
    def _create_topology(self):
        """根据类型创建网络拓扑"""
        if self.topology_type == "grid":
            self.graph, self.node_positions = NetworkTopology.create_grid(self.topology_size)
        elif self.topology_type == "star":
            self.graph, self.node_positions = NetworkTopology.create_star("center", 8)
        elif self.topology_type == "ring":
            self.graph, self.node_positions = NetworkTopology.create_ring_with_chords(12)
        elif self.topology_type == "tree":
            self.graph, self.node_positions = NetworkTopology.create_tree(3, 3)
        elif self.topology_type == "small_world":
            self.graph, self.node_positions = NetworkTopology.create_small_world(16, 4, 0.3)
        elif self.topology_type == "irregular":
            self.graph, self.node_positions = NetworkTopology.create_irregular_network(25)
        elif self.topology_type == "city":
            self.graph, self.node_positions = NetworkTopology.create_city_roads(8)
        elif self.topology_type == "suburban":
            self.graph, self.node_positions = NetworkTopology.create_suburban_network(4, 6)
        elif self.topology_type == "highway":
            self.graph, self.node_positions = NetworkTopology.create_highway_network(6)
        else:
            # 默认网格
            self.graph, self.node_positions = NetworkTopology.create_grid(6)
    
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
        
        # 边的安全缓冲时间可以稍短一些，因为边冲突不如节点冲突那么明显
        safety_buffer = 0.15  # 边的安全缓冲时间
        precision_buffer = 0.01  # 浮点精度缓冲
        
        existing_reservations = self.edge_reservations[edge_key]
        for reservation in existing_reservations:
            # 严格的时间重叠检测 + 安全缓冲
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
        
        # 增加安全缓冲时间，避免视觉上的"危险驾驶"
        safety_buffer = 0.3  # 安全缓冲时间，车辆离开后需要等待0.3秒其他车辆才能进入
        precision_buffer = 0.01  # 浮点精度缓冲
        
        existing_reservations = self.node_reservations[node]
        for reservation in existing_reservations:
            # 严格的时间重叠检测 + 安全缓冲：
            # 新预留必须在现有预留结束后至少safety_buffer时间才能开始
            # 或者现有预留必须在新预留结束后至少safety_buffer时间才能开始
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
        """检查节点在指定时间段是否可用 - 包含安全缓冲时间"""
        end_time = start_time + duration
        
        # 增加安全缓冲时间，避免视觉上的"危险驾驶"
        safety_buffer = 0.3  # 安全缓冲时间
        precision_buffer = 0.01  # 浮点精度缓冲
        
        reservations = self.node_reservations.get(node, [])
        for reservation in reservations:
            if reservation.vehicle_id == exclude_vehicle:
                continue
            
            # 严格的时间重叠检测 + 安全缓冲：任何在安全缓冲区内的重叠都视为不可用
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
        """检查边在指定时间段是否可用 - 包含安全缓冲时间"""
        edge_key = tuple(sorted([from_node, to_node]))
        end_time = start_time + duration
        
        # 边的安全缓冲时间
        safety_buffer = 0.15  # 边的安全缓冲时间
        precision_buffer = 0.01  # 浮点精度缓冲
        
        reservations = self.edge_reservations.get(edge_key, [])
        for reservation in reservations:
            if reservation.vehicle_id == exclude_vehicle:
                continue
            
            # 严格的时间重叠检测 + 安全缓冲：任何在安全缓冲区内的重叠都视为不可用
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

class Vehicle:
    """车辆智能体类 - 支持露天矿作业模式"""
    
    def __init__(self, vehicle_id: int, start_node: str, road_network: SimpleRoadNetwork, use_gnn: bool = True):
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
    
    # 保留原有的移动相关方法
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
        
        print(f"Vehicle {self.id}: 到达 {self.current_node}")
        
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

class CleanGNNSimulation:
    """清晰的GNN算法验证仿真 - 露天矿场景"""
    
    def __init__(self, topology_type: str = "grid", num_vehicles: int = 4):
        self.topology_type = topology_type
        self.num_vehicles = num_vehicles
        self.road_network = SimpleRoadNetwork(topology_type, 6, num_vehicles)
        self.vehicles = []
        self.use_gnn = True
        self.current_time = 0.0
        
        self._create_initial_vehicles()
        
        # 简洁的布局
        self.fig, (self.ax_main, self.ax_stats) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 主仿真区域
        positions = list(self.road_network.node_positions.values())
        if positions:
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            margin = 1.0
            self.ax_main.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
            self.ax_main.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
        else:
            self.ax_main.set_xlim(-1, 7)
            self.ax_main.set_ylim(-1, 7)
        
        self.ax_main.set_aspect('equal')
        self.ax_main.set_title(f'Open-pit Mine GNN Scheduling - {topology_type.upper()} Topology', 
                              fontsize=14, fontweight='bold')
        self.ax_main.grid(True, alpha=0.3)
        
        self.ax_stats.set_xlim(0, 1)
        self.ax_stats.set_ylim(0, 1)
        self.ax_stats.axis('off')
        
        self.animation = None
        self.is_running = False
    
    def _create_initial_vehicles(self):
        """创建初始车辆 - 露天矿模式"""
        nodes = list(self.road_network.graph.nodes())
        
        for i in range(self.num_vehicles):
            # 车辆从随机位置开始，但会立即寻找装载点
            start_node = random.choice(nodes)
            
            vehicle = Vehicle(i, start_node, self.road_network, self.use_gnn)
            self.vehicles.append(vehicle)
        
        # 将车辆列表传递给道路网络
        self.road_network.vehicles = self.vehicles
    
    def switch_topology(self):
        """切换网络拓扑类型"""
        topologies = ["grid", "star", "ring", "tree", "small_world", "irregular", "city", "suburban", "highway"]
        current_index = topologies.index(self.topology_type)
        self.topology_type = topologies[(current_index + 1) % len(topologies)]
        self.reset_simulation()
        print(f"Switched to {self.topology_type} topology")
    
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
        """添加车辆 - 露天矿模式限制"""
        max_vehicles = min(12, len(self.road_network.loading_points))  # 受装载点数量限制
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
        """移除车辆 - 露天矿模式"""
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
        positions = list(self.road_network.node_positions.values())
        if positions:
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            margin = 1.0
            self.ax_main.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
            self.ax_main.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
        
        self.ax_main.set_aspect('equal')
        self.ax_main.grid(True, alpha=0.3)
        
        # 显示拓扑信息和节点度数统计
        degree_stats = self._get_degree_statistics()
        title = f'GNN Demo - {self.topology_type.upper()} - {"GNN" if self.use_gnn else "Simple"} Mode\n{degree_stats}'
        self.ax_main.set_title(title, fontsize=12, fontweight='bold')
        
        self._draw_network()
        self._draw_reservations()
        self._draw_vehicles()
        self._draw_statistics()
        
        return []
    
    def _get_degree_statistics(self):
        """获取节点度数统计"""
        degrees = [self.road_network.graph.degree(node) for node in self.road_network.graph.nodes()]
        if degrees:
            from collections import Counter
            degree_counts = Counter(degrees)
            stats_str = " | ".join([f"度数{d}: {count}节点" for d, count in sorted(degree_counts.items())])
            return stats_str
        return "No nodes"
    
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
                size = 0.25
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
                size = 0.25
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
                size = 0.1 + degree * 0.03
            
            # 绘制节点
            circle = Circle(pos, size, color=color, 
                          edgecolor='navy', linewidth=1, alpha=alpha)
            self.ax_main.add_patch(circle)
            
            # 显示节点信息
            if is_loading_point or is_unloading_point:
                # 装卸点显示特殊信息
                label = f"{node}\n{marker}\n{status}"
            else:
                # 普通节点显示度数
                label = f"{node}\n({degree})"
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
        """突出显示边预留 - 这是重点"""
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
                size = 0.16
                symbol = 'L'
            elif vehicle.state == VehicleState.UNLOADING:
                # 卸载中 - 蓝色边框，较大尺寸
                alpha = 1.0
                edge_color = 'blue'
                edge_width = 4
                size = 0.16
                symbol = 'U'
            elif vehicle.state == VehicleState.CONFIRMED:
                # 路径已确认，等待开始时间 - 金色边框
                alpha = 0.95
                edge_color = 'gold'
                edge_width = 4
                size = 0.14
                symbol = 'C'
            elif vehicle.state == VehicleState.MOVING:
                # 正在移动 - 根据载重状态调整
                alpha = 1.0
                edge_color = 'white'
                edge_width = 2
                size = 0.14 if vehicle.mode == VehicleMode.LOADED else 0.12
                symbol = 'M'
            elif vehicle.state == VehicleState.WAITING:
                # 等待中 - 暗淡显示
                alpha = 0.5
                edge_color = 'red'
                edge_width = 2
                size = 0.10
                symbol = 'W'
            elif vehicle.state == VehicleState.PLANNING:
                # 规划中 - 橙色边框
                alpha = 0.7
                edge_color = 'orange'
                edge_width = 2
                size = 0.11
                symbol = 'P'
            else:  # IDLE, BLOCKED
                alpha = 0.8
                edge_color = 'black'
                edge_width = 1
                size = 0.11
                symbol = 'I'
            
            # 车辆形状 - 空载用圆形，重载用方形
            if vehicle.mode == VehicleMode.LOADED:
                # 重载 - 方形，更大
                rect = Rectangle((x-size, y-size), size*2, size*2, 
                               color=vehicle.color, alpha=alpha, 
                               edgecolor=edge_color, linewidth=edge_width)
                self.ax_main.add_patch(rect)
                mode_symbol = '■'
            else:
                # 空载 - 圆形
                circle = Circle((x, y), size, color=vehicle.color, alpha=alpha,
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
                pulse_radius = 0.25 + 0.1 * np.sin(self.current_time * 4)
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
                bar_width = 0.3
                bar_height = 0.05
                bar_x = x - bar_width/2
                bar_y = y + size + 0.1
                
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
            self.ax_main.text(pos[0]-0.3, pos[1]+0.3, str(i), 
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
        degrees = [self.road_network.graph.degree(node) for node in self.road_network.graph.nodes()]
        from collections import Counter
        degree_counts = Counter(degrees)
        
        stats_text = f"""
        ╔═══ OPEN-PIT MINE GNN ════╗
        ║ Topology: {self.topology_type.upper():>13} ║
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
        ╠═══ NODE DEGREE DIST. ════╣"""
        
        for degree in sorted(degree_counts.keys()):
            count = degree_counts[degree]
            stats_text += f"\n        ║ 度数{degree}: {count:>12d} ║"
        
        stats_text += f"""
        ╠═══ RESERVATION STATUS ═══╣
        ║ Edge Rsv: {total_edge_reservations:>13d} ║
        ║ Node Rsv: {total_node_reservations:>13d} ║
        ╚═══════════════════════════╝
        
        Open-pit Mine Operations:
        🚛 Cycle: LoadPt → UnloadPt → LoadPt
        📍 LoadPt: Green squares □ (edge pos)
        📍 UnloadPt: Blue triangles △ (edge pos)
        
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
        
        Safety Features:
        🛡️ Node safety buffer: 0.3s
        🛡️ Edge safety buffer: 0.15s  
        ⏱️ Conflict avoidance: Guaranteed
        🚦 Smart scheduling: Deadlock-free
        
        Controls:
        Space: Start/Pause
        'g': Toggle GNN/Simple
        't': Switch Topology
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
                                 facecolor='lightgray', alpha=0.9))
        
        self.ax_stats.text(0.02, 0.98, stats_text, 
                         transform=self.ax_stats.transAxes,
                         fontfamily='monospace', fontsize=8,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round,pad=0.5', 
                                 facecolor='lightgray', alpha=0.9))
    
    def _get_topology_description(self):
        """获取拓扑描述"""
        descriptions = {
            "grid": "Regular lattice structure",
            "star": "Hub-and-spoke network",
            "ring": "Ring with shortcuts",
            "tree": "Hierarchical tree",
            "small_world": "Small-world network",
            "irregular": "Random irregular network",
            "city": "Urban road system",
            "suburban": "Multi-district suburbs",
            "highway": "Highway network"
        }
        return descriptions.get(self.topology_type, "Unknown topology")
    
    def start_animation(self):
        if not self.is_running:
            self.animation = animation.FuncAnimation(
                self.fig, self.update, interval=100, blit=False)
            self.is_running = True
            print("Simulation started")
    
    def stop_animation(self):
        if self.animation:
            self.animation.event_source.stop()
            self.is_running = False
            print("Simulation stopped")
    
    def reset_simulation(self):
        self.current_time = 0.0
        self.road_network = SimpleRoadNetwork(self.topology_type, 6, self.num_vehicles)
        
        num_vehicles = len(self.vehicles)
        self.vehicles.clear()
        self._create_initial_vehicles()
        print(f"Reset open-pit mine simulation - {self.topology_type} topology, {num_vehicles} vehicles")
        
        for vehicle in self.vehicles:
            vehicle.use_gnn = self.use_gnn
        
        # 确保道路网络有车辆引用
        self.road_network.vehicles = self.vehicles
    
    def show(self):
        plt.tight_layout()
        plt.show()

def main():
    """主函数"""
    print("露天矿GNN智能调度系统 - 装卸点循环作业")
    print("=" * 80)
    print("🎯 露天矿场景特点:")
    print("🚛 车辆循环作业: 装载点 → 卸载点 → 装载点 (无限循环)")
    print("📍 智能装卸点分配: 每台车对应专用装载点和卸载点")
    print("🚫 冲突避免: 车辆不会前往已被其他车辆预留的装卸点")
    print("🌍 边缘部署: 装载点和卸载点位于网络边缘 (模拟矿场边界)")
    print("⚡ 实时调度: GNN算法实现最优路径规划和冲突避免")
    print()
    print("🛡️ 安全保障系统:")
    print("   • 节点安全缓冲: 0.3s (避免视觉冲突)")
    print("   • 边缘安全缓冲: 0.15s (平滑路径切换)")
    print("   • 装卸作业时间: 装载2.0s, 卸载1.5s")
    print("   • 冷却期显示: 实时倒计时，确保安全间隔")
    print()
    print("🎨 可视化特色:")
    print("🟢 装载点: 绿色方块 □ (空闲) → 深红色 ■ (使用中)")
    print("🔵 卸载点: 蓝色三角 △ (空闲) → 深蓝色 ▼ (使用中)")  
    print("🚛 车辆状态: ○空载 / ■重载 + 作业进度条")
    print("🛣️ 路径指示: 绿线→装载点, 蓝线→卸载点")
    print("⏱️ 作业进度: 实时显示装载/卸载进度百分比")
    print()
    print("📊 作业指标:")
    print("- 完成循环数: 衡量整体作业效率") 
    print("- 装卸点利用率: 资源使用优化")
    print("- 空载/重载车辆比: 负载均衡分析")
    print("- 等待时间: 调度算法性能")
    print()
    print("🗺️ 可用网络拓扑:")
    print("1. IRREGULAR - 不规则网络 (推荐)")
    print("2. CITY - 城市道路 (复杂路网)")
    print("3. SUBURBAN - 郊区网络 (多区域)")
    print("4. HIGHWAY - 高速网络 (长距离)")
    print("5. GRID - 规则网格 (基准测试)")
    print("6. 其他拓扑...")
    print()
    print("🔧 智能特性:")
    print("✅ 动态装卸点分配 - 根据车辆数量自动配置")
    print("✅ 预留冲突检测 - 避免多车竞争同一资源")
    print("✅ 自适应路径规划 - GNN优化 vs 传统最短路径")
    print("✅ 实时状态监控 - 全链路可视化跟踪")
    print("✅ 作业循环统计 - 性能指标实时更新")
    print()
    print("控制说明:")
    print("- Space: 开始/暂停仿真")
    print("- 'g': 切换 GNN/传统 调度模式")
    print("- 't': 循环切换网络拓扑")
    print("- '+'/'-': 增加/减少车辆数量")
    print("- 'r': 重置仿真")
    print("- 'q': 退出程序")
    print("=" * 80)
    
    # 创建露天矿仿真，默认从不规则网络开始
    sim = CleanGNNSimulation(topology_type="irregular", num_vehicles=4)
    
    def on_key(event):
        if event.key == ' ':
            if sim.is_running:
                sim.stop_animation()
            else:
                sim.start_animation()
        elif event.key == 'g':
            sim.toggle_gnn_mode()
        elif event.key == 't':
            sim.switch_topology()
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