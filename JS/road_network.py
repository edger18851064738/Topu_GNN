import json
import math
import heapq
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import time


@dataclass
class NodeFeature:
    """节点的GNN特征"""
    occupancy: float = 0.0
    connectivity: float = 0.0
    congestion: float = 0.0
    centrality: float = 0.0


@dataclass
class SpecialPoint:
    """特殊点：装载点、卸载点、停车点"""
    node_id: str
    point_type: str  # 'loading', 'unloading', 'parking'
    position: List[float] = field(default_factory=lambda: [0.0, 0.0])
    is_occupied: bool = False
    reserved_by: Optional[int] = None


@dataclass
class EdgeReservation:
    """边预订"""
    vehicle_id: int
    start_time: float
    end_time: float
    direction: List[str]


@dataclass
class NodeReservation:
    """节点预订"""
    vehicle_id: int
    start_time: float
    end_time: float
    action: str


class RoadNetwork:
    """道路网络管理类，包含GNN功能和预订系统"""
    
    def __init__(self):
        # 网络结构
        self.graph: Dict[str, Set[str]] = {}  # 邻接表
        self.node_positions: Dict[str, List[float]] = {}
        self.node_features: Dict[str, NodeFeature] = {}
        
        # 特殊点
        self.special_points = {
            'loading': {},
            'unloading': {},
            'parking': {}
        }
        
        # 预订系统
        self.edge_reservations: Dict[str, List[EdgeReservation]] = {}
        self.node_reservations: Dict[str, List[NodeReservation]] = {}
        self.node_occupancy: Dict[str, Set[int]] = {}
        self.global_time: float = 0.0
        
        # 安全参数
        self.EDGE_SAFETY_BUFFER = 0.15
        self.NODE_SAFETY_BUFFER = 0.3
        self.PRECISION_BUFFER = 0.01
        self.NODE_COOLING_TIME = 0.3
        
        # 初始化默认拓扑
        self._create_fallback_topology()
        self._initialize_node_features()
    
    def _create_fallback_topology(self):
        """创建默认的网格拓扑"""
        grid_size = 5
        node_spacing = 80
        
        # 创建网格节点
        for i in range(grid_size):
            for j in range(grid_size):
                node_id = f"N{i}_{j}"
                x = 50 + j * node_spacing + (hash(node_id) % 40 - 20)
                y = 50 + i * node_spacing + (hash(node_id) % 40 - 20)
                
                self.node_positions[node_id] = [x, y]
                self.graph[node_id] = set()
                self.node_occupancy[node_id] = set()
        
        # 创建边连接
        for i in range(grid_size):
            for j in range(grid_size):
                current_node = f"N{i}_{j}"
                
                # 右连接
                if j < grid_size - 1:
                    right_node = f"N{i}_{j + 1}"
                    self.graph[current_node].add(right_node)
                    self.graph[right_node].add(current_node)
                
                # 下连接
                if i < grid_size - 1:
                    down_node = f"N{i + 1}_{j}"
                    self.graph[current_node].add(down_node)
                    self.graph[down_node].add(current_node)
                
                # 对角线连接（部分）
                if i < grid_size - 1 and j < grid_size - 1 and hash(current_node) % 2 == 0:
                    diag_node = f"N{i + 1}_{j + 1}"
                    self.graph[current_node].add(diag_node)
                    self.graph[diag_node].add(current_node)
        
        # 创建特殊点
        self._create_default_special_points()
    
    def _create_default_special_points(self):
        """创建默认特殊点"""
        special_indices = {
            'loading': [(0, 0), (0, 2), (0, 4), (2, 0), (4, 0), (4, 2)],
            'unloading': [(4, 4), (2, 4), (0, 4), (4, 2), (4, 0), (2, 4)],
            'parking': [(2, 2), (1, 1), (3, 3), (1, 3), (3, 1), (2, 0)]
        }
        
        for point_type, indices in special_indices.items():
            for i, (row, col) in enumerate(indices):
                if i >= 6:  # 限制每种类型6个点
                    break
                node_id = f"N{row}_{col}"
                if node_id in self.node_positions:
                    point_id = f"{point_type[0].upper()}{i}"
                    pos = self.node_positions[node_id].copy()
                    self.special_points[point_type][point_id] = SpecialPoint(
                        node_id=node_id,
                        point_type=point_type,
                        position=pos
                    )
    
    def _initialize_node_features(self):
        """初始化节点GNN特征"""
        for node_id in self.graph.keys():
            self.node_features[node_id] = NodeFeature()
            self._update_node_feature(node_id)
    
    def _update_node_feature(self, node_id: str):
        """更新单个节点的GNN特征"""
        neighbors = list(self.graph.get(node_id, set()))
        connectivity = len(neighbors)
        
        # 计算拥堵度
        congestion = 0.0
        current_time = self.global_time
        
        for neighbor in neighbors:
            edge_key = self._get_edge_key(node_id, neighbor)
            reservations = self.edge_reservations.get(edge_key, [])
            future_occupied = any(
                r.start_time <= current_time + 2.0 and r.end_time >= current_time
                for r in reservations
            )
            if future_occupied:
                congestion += 1
        
        congestion = congestion / len(neighbors) if neighbors else 0
        
        # 中心性（归一化）
        centrality = connectivity / 8.0
        
        # 占用度
        occupied_vehicles = self.node_occupancy.get(node_id, set())
        occupancy = len(occupied_vehicles) * 0.5
        
        feature = self.node_features[node_id]
        feature.occupancy = occupancy
        feature.connectivity = connectivity
        feature.congestion = congestion
        feature.centrality = centrality
    
    def load_topology_from_json(self, json_data: Dict[str, Any]) -> bool:
        """从JSON数据加载拓扑结构"""
        try:
            print(f"Loading topology from JSON data...")
            
            # 清除现有数据
            self.graph.clear()
            self.node_positions.clear()
            self.node_features.clear()
            self.node_occupancy.clear()
            self.special_points = {'loading': {}, 'unloading': {}, 'parking': {}}
            
            # 根据数据格式加载
            if json_data.get('enhanced_consolidation_applied') and json_data.get('key_nodes_info'):
                print("Loading from enhanced Stage 1 data...")
                self._load_from_enhanced_data(json_data)
            elif json_data.get('graph_nodes') and json_data.get('graph_edges'):
                print("Loading from basic Stage 1 data...")
                self._load_from_basic_data(json_data)
            else:
                raise ValueError("Invalid topology format")
            
            self._parse_special_points(json_data)
            self._ensure_sufficient_special_points()
            self._initialize_node_features()
            
            print(f"Topology loaded successfully: {len(self.graph)} nodes, "
                  f"{self._count_edges()} edges")
            return True
            
        except Exception as e:
            print(f"Failed to load topology: {e}")
            self._create_fallback_topology()
            self._initialize_node_features()
            return False
    
    def _load_from_enhanced_data(self, json_data: Dict[str, Any]):
        """从增强数据格式加载"""
        key_nodes = json_data['key_nodes_info']
        
        # 添加节点
        for node_id, node_info in key_nodes.items():
            pos = node_info['position']
            self.node_positions[node_id] = [pos[0], pos[1]]
            self.graph[node_id] = set()
            self.node_occupancy[node_id] = set()
        
        # 从合并路径添加边
        consolidated_paths = json_data.get('consolidated_paths_info', {})
        
        for path_id, path_info in consolidated_paths.items():
            key_nodes_list = path_info.get('key_nodes', [])
            for i in range(len(key_nodes_list) - 1):
                node1 = key_nodes_list[i]
                node2 = key_nodes_list[i + 1]
                if node1 in self.graph and node2 in self.graph:
                    self.graph[node1].add(node2)
                    self.graph[node2].add(node1)
    
    def _load_from_basic_data(self, json_data: Dict[str, Any]):
        """从基本数据格式加载"""
        # 添加节点
        for node in json_data['graph_nodes']:
            node_str = str(node)
            self.graph[node_str] = set()
            self.node_occupancy[node_str] = set()
            
            pos = json_data.get('position_mapping', {}).get(node_str, 
                [hash(node_str) % 400, hash(node_str) % 300])
            self.node_positions[node_str] = pos
        
        # 添加边
        for edge in json_data['graph_edges']:
            node1 = str(edge[0])
            node2 = str(edge[1])
            if node1 in self.graph and node2 in self.graph:
                self.graph[node1].add(node2)
                self.graph[node2].add(node1)
    
    def _parse_special_points(self, json_data: Dict[str, Any]):
        """从JSON数据解析特殊点"""
        if not json_data.get('key_nodes_info'):
            return
        
        for node_id, node_info in json_data['key_nodes_info'].items():
            position = node_info.get('position', [0, 0])
            path_memberships = node_info.get('path_memberships', [])
            is_endpoint = node_info.get('is_endpoint', False)
            
            if not is_endpoint or not path_memberships:
                continue
            
            # 分析路径模式确定特殊点类型
            start_points = set()
            end_points = set()
            
            for path_id in path_memberships:
                parts = path_id.split('_to_')
                if len(parts) == 2:
                    start_points.add(parts[0])
                    end_points.add(parts[1])
            
            primary_special_point = None
            
            # 根据路径模式确定主要类型
            if len(start_points) == 1:
                start_point = list(start_points)[0]
                if start_point.startswith('L') and start_point[1:].isdigit():
                    primary_special_point = {'id': start_point, 'type': 'loading'}
            elif len(end_points) == 1:
                end_point = list(end_points)[0]
                if end_point.startswith('U') and end_point[1:].isdigit():
                    primary_special_point = {'id': end_point, 'type': 'unloading'}
                elif end_point.startswith('P') and end_point[1:].isdigit():
                    primary_special_point = {'id': end_point, 'type': 'parking'}
            
            # 创建特殊点
            if primary_special_point:
                special_point = SpecialPoint(
                    node_id=node_id,
                    point_type=primary_special_point['type'],
                    position=position.copy()
                )
                point_id = primary_special_point['id']
                self.special_points[primary_special_point['type']][point_id] = special_point
    
    def _ensure_sufficient_special_points(self):
        """确保有足够的特殊点用于演示"""
        min_required = 4
        
        for point_type in ['loading', 'unloading', 'parking']:
            current_count = len(self.special_points[point_type])
            if current_count < min_required:
                self._create_additional_special_points(point_type, min_required)
    
    def _create_additional_special_points(self, point_type: str, target_count: int):
        """创建额外的特殊点"""
        current_points = self.special_points[point_type]
        
        # 收集已使用的节点
        used_nodes = set()
        for type_points in self.special_points.values():
            for point in type_points.values():
                used_nodes.add(point.node_id)
        
        # 找到未使用的节点
        available_nodes = [node_id for node_id in self.node_positions.keys()
                          if node_id not in used_nodes]
        
        prefix = point_type[0].upper()
        current_count = len(current_points)
        max_possible = min(target_count, current_count + len(available_nodes))
        
        for i in range(current_count, max_possible):
            if i - current_count >= len(available_nodes):
                break
            
            node_id = available_nodes[i - current_count]
            point_id = f"{prefix}{i}"
            position = self.node_positions[node_id].copy()
            
            special_point = SpecialPoint(
                node_id=node_id,
                point_type=point_type,
                position=position
            )
            current_points[point_id] = special_point
    
    def _get_edge_key(self, node1: str, node2: str) -> str:
        """获取边的唯一键"""
        return '-'.join(sorted([node1, node2]))
    
    def _count_edges(self) -> int:
        """计算边的总数"""
        return sum(len(neighbors) for neighbors in self.graph.values()) // 2
    
    # GNN增强的路径查找
    def gnn_pathfinding_with_reservation(self, start: str, end: str, 
                                       vehicle_id: int, current_time: float) -> Dict[str, Any]:
        """使用GNN增强的带预订的路径查找"""
        if start == end:
            return {'path': [start], 'times': [current_time]}
        
        # 更新所有节点特征
        for node_id in self.graph.keys():
            self._update_node_feature(node_id)
        
        # A* 算法与时空预订
        heap = [(0, start, [start], [current_time])]
        visited = {}
        
        while heap:
            cost, current, path, times = heapq.heappop(heap)
            current_arrival_time = times[-1]
            
            # 跳过已访问的节点
            if current in visited and visited[current] <= current_arrival_time:
                continue
            visited[current] = current_arrival_time
            
            if current == end:
                return {'path': path, 'times': times}
            
            neighbors = list(self.graph.get(current, set()))
            for neighbor in neighbors:
                if neighbor in path:
                    continue
                
                travel_time = self._compute_travel_time(current, neighbor, vehicle_id)
                departure_time = current_arrival_time + 0.1  # 节点处理时间
                arrival_time = departure_time + travel_time
                
                # 检查边可用性
                if not self._is_edge_available(current, neighbor, departure_time, 
                                             travel_time, vehicle_id):
                    next_available = self._find_next_available_time(
                        current, neighbor, departure_time, travel_time, vehicle_id
                    )
                    if next_available > departure_time + 5.0:
                        continue  # 等待时间太长，跳过
                    arrival_time = next_available + travel_time
                
                edge_cost = self._compute_gnn_edge_cost(current, neighbor, departure_time)
                heuristic = self._compute_heuristic(neighbor, end)
                total_cost = arrival_time + edge_cost + heuristic
                
                heapq.heappush(heap, (
                    total_cost,
                    neighbor,
                    path + [neighbor],
                    times + [arrival_time]
                ))
        
        return {'path': [], 'times': []}
    
    def _compute_travel_time(self, from_node: str, to_node: str, vehicle_id: int) -> float:
        """计算旅行时间"""
        base_time = 1.0
        to_feature = self.node_features.get(to_node, NodeFeature())
        time_factor = 1.0 + to_feature.congestion * 0.5 + to_feature.occupancy * 0.3
        return base_time * time_factor
    
    def _compute_gnn_edge_cost(self, from_node: str, to_node: str, time: float) -> float:
        """计算GNN边成本"""
        base_weight = 1.0
        to_feature = self.node_features.get(to_node, NodeFeature())
        
        cost = base_weight
        cost += to_feature.occupancy * 3.0
        cost += to_feature.congestion * 8.0
        cost -= to_feature.centrality * 1.0
        
        # 时间相关的拥堵
        edge_key = self._get_edge_key(from_node, to_node)
        reservations = self.edge_reservations.get(edge_key, [])
        future_congestion = sum(
            1 for r in reservations
            if r.start_time <= time + 3.0 and r.end_time >= time
        )
        cost += future_congestion * 2.0
        
        return max(cost, 0.1)
    
    def _compute_heuristic(self, node1: str, node2: str) -> float:
        """计算启发式距离"""
        pos1 = self.node_positions.get(node1)
        pos2 = self.node_positions.get(node2)
        if not pos1 or not pos2:
            return 0
        
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return math.sqrt(dx * dx + dy * dy) * 0.3
    
    # 预订系统方法
    def _is_node_available(self, node_id: str, start_time: float, 
                          duration: float, exclude_vehicle: int = -1) -> bool:
        """检查节点是否可用"""
        end_time = start_time + duration
        reservations = self.node_reservations.get(node_id, [])
        
        for reservation in reservations:
            if reservation.vehicle_id == exclude_vehicle:
                continue
            
            if not (end_time + self.NODE_SAFETY_BUFFER + self.PRECISION_BUFFER <= reservation.start_time or
                    start_time >= reservation.end_time + self.NODE_SAFETY_BUFFER + self.PRECISION_BUFFER):
                return False
        
        return True
    
    def _is_edge_available(self, from_node: str, to_node: str, start_time: float,
                          duration: float, exclude_vehicle: int = -1) -> bool:
        """检查边是否可用"""
        edge_key = self._get_edge_key(from_node, to_node)
        end_time = start_time + duration
        reservations = self.edge_reservations.get(edge_key, [])
        
        for reservation in reservations:
            if reservation.vehicle_id == exclude_vehicle:
                continue
            
            if not (end_time + self.EDGE_SAFETY_BUFFER + self.PRECISION_BUFFER <= reservation.start_time or
                    start_time >= reservation.end_time + self.EDGE_SAFETY_BUFFER + self.PRECISION_BUFFER):
                return False
        
        return True
    
    def _find_next_available_time(self, from_node: str, to_node: str, 
                                 earliest_start: float, duration: float, 
                                 vehicle_id: int) -> float:
        """找到下一个可用时间"""
        edge_key = self._get_edge_key(from_node, to_node)
        reservations = self.edge_reservations.get(edge_key, [])
        
        other_reservations = [r for r in reservations if r.vehicle_id != vehicle_id]
        other_reservations.sort(key=lambda r: r.start_time)
        
        current_time = earliest_start
        for reservation in other_reservations:
            if current_time + duration <= reservation.start_time:
                return current_time
            current_time = max(current_time, reservation.end_time + self.EDGE_SAFETY_BUFFER)
        
        return current_time
    
    def reserve_node(self, node_id: str, vehicle_id: int, 
                    start_time: float, duration: float) -> bool:
        """预订节点"""
        if not self._is_node_available(node_id, start_time, duration, vehicle_id):
            return False
        
        if node_id not in self.node_reservations:
            self.node_reservations[node_id] = []
        
        end_time = start_time + duration
        reservation = NodeReservation(
            vehicle_id=vehicle_id,
            start_time=start_time,
            end_time=end_time,
            action='occupy'
        )
        self.node_reservations[node_id].append(reservation)
        return True
    
    def reserve_edge(self, from_node: str, to_node: str, vehicle_id: int,
                    start_time: float, duration: float) -> bool:
        """预订边"""
        if not self._is_edge_available(from_node, to_node, start_time, duration, vehicle_id):
            return False
        
        edge_key = self._get_edge_key(from_node, to_node)
        if edge_key not in self.edge_reservations:
            self.edge_reservations[edge_key] = []
        
        end_time = start_time + duration
        reservation = EdgeReservation(
            vehicle_id=vehicle_id,
            start_time=start_time,
            end_time=end_time,
            direction=[from_node, to_node]
        )
        self.edge_reservations[edge_key].append(reservation)
        return True
    
    def cancel_reservations(self, vehicle_id: int):
        """取消车辆的所有预订"""
        # 取消边预订
        for edge_key in list(self.edge_reservations.keys()):
            remaining = [r for r in self.edge_reservations[edge_key] 
                        if r.vehicle_id != vehicle_id]
            if remaining:
                self.edge_reservations[edge_key] = remaining
            else:
                del self.edge_reservations[edge_key]
        
        # 取消节点预订
        for node_id in list(self.node_reservations.keys()):
            remaining = [r for r in self.node_reservations[node_id]
                        if r.vehicle_id != vehicle_id]
            if remaining:
                self.node_reservations[node_id] = remaining
            else:
                del self.node_reservations[node_id]
    
    # 特殊点管理
    def get_available_point(self, point_type: str, exclude_vehicle: int = -1) -> Optional[str]:
        """获取可用的特殊点"""
        points = self.special_points[point_type]
        for point_id, point in points.items():
            if not point.is_occupied and (point.reserved_by is None or 
                                        point.reserved_by == exclude_vehicle):
                return point_id
        return None
    
    def reserve_special_point(self, point_id: str, vehicle_id: int) -> bool:
        """预订特殊点"""
        for point_type, points in self.special_points.items():
            if point_id in points:
                point = points[point_id]
                if not point.is_occupied and point.reserved_by is None:
                    point.reserved_by = vehicle_id
                    return True
        return False
    
    def occupy_special_point(self, point_id: str, vehicle_id: int):
        """占用特殊点"""
        for point_type, points in self.special_points.items():
            if point_id in points:
                point = points[point_id]
                point.is_occupied = True
                point.reserved_by = vehicle_id
                return
    
    def release_special_point(self, point_id: str):
        """释放特殊点"""
        for point_type, points in self.special_points.items():
            if point_id in points:
                point = points[point_id]
                point.is_occupied = False
                point.reserved_by = None
                return
    
    def get_point_node_id(self, point_id: str) -> Optional[str]:
        """获取特殊点对应的节点ID"""
        for point_type, points in self.special_points.items():
            if point_id in points:
                return points[point_id].node_id
        return None
    
    def update_time(self, current_time: float):
        """更新全局时间并清理过期预订"""
        self.global_time = current_time
        
        # 清理过期的边预订
        for edge_key in list(self.edge_reservations.keys()):
            active = [r for r in self.edge_reservations[edge_key] 
                     if r.end_time > current_time]
            if active:
                self.edge_reservations[edge_key] = active
            else:
                del self.edge_reservations[edge_key]
        
        # 清理过期的节点预订
        for node_id in list(self.node_reservations.keys()):
            active = [r for r in self.node_reservations[node_id]
                     if r.end_time > current_time]
            if active:
                self.node_reservations[node_id] = active
            else:
                del self.node_reservations[node_id]
    
    def add_vehicle_to_node(self, vehicle_id: int, node_id: str):
        """将车辆添加到节点"""
        if node_id not in self.node_occupancy:
            self.node_occupancy[node_id] = set()
        self.node_occupancy[node_id].add(vehicle_id)
    
    def remove_vehicle_from_node(self, vehicle_id: int, node_id: str):
        """从节点移除车辆"""
        if node_id in self.node_occupancy:
            self.node_occupancy[node_id].discard(vehicle_id)
    
    def simple_pathfinding(self, start: str, end: str) -> List[str]:
        """简单的BFS路径查找"""
        if start == end:
            return [start]
        
        queue = [(start, [start])]
        visited = {start}
        
        while queue:
            current, path = queue.pop(0)
            
            for neighbor in self.graph.get(current, set()):
                if neighbor == end:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return []
    
    def get_network_state(self) -> Dict[str, Any]:
        """获取网络状态用于前端显示"""
        return {
            'nodes': {
                node_id: {
                    'position': pos,
                    'neighbors': list(neighbors),
                    'occupancy': len(self.node_occupancy.get(node_id, set())),
                    'features': {
                        'occupancy': self.node_features[node_id].occupancy,
                        'connectivity': self.node_features[node_id].connectivity,
                        'congestion': self.node_features[node_id].congestion,
                        'centrality': self.node_features[node_id].centrality
                    } if node_id in self.node_features else {},
                    'reservations': [
                        {
                            'vehicle_id': r.vehicle_id,
                            'start_time': r.start_time,
                            'end_time': r.end_time
                        }
                        for r in self.node_reservations.get(node_id, [])
                    ]
                }
                for node_id, pos in self.node_positions.items()
                for neighbors in [self.graph.get(node_id, set())]
            },
            'special_points': {
                point_type: {
                    point_id: {
                        'node_id': point.node_id,
                        'position': point.position,
                        'is_occupied': point.is_occupied,
                        'reserved_by': point.reserved_by
                    }
                    for point_id, point in points.items()
                }
                for point_type, points in self.special_points.items()
            },
            'edge_reservations': {
                edge_key: [
                    {
                        'vehicle_id': r.vehicle_id,
                        'start_time': r.start_time,
                        'end_time': r.end_time,
                        'direction': r.direction
                    }
                    for r in reservations
                ]
                for edge_key, reservations in self.edge_reservations.items()
            }
        }