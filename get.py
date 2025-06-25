"""
第二阶段GNN拓扑提取器
从第一阶段导出的拓扑结构中提取GNN建模所需的关键信息
支持多种GNN框架：PyTorch Geometric, DGL, NetworkX
"""

import json
import numpy as np
import torch
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import pickle
import warnings

try:
    import torch_geometric
    from torch_geometric.data import Data, HeteroData
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    warnings.warn("PyTorch Geometric not available. Some features will be disabled.")

try:
    import dgl
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False
    warnings.warn("DGL not available. Some features will be disabled.")

@dataclass
class GNNTopologyConfig:
    """GNN拓扑提取配置"""
    include_spatial_features: bool = True
    include_road_features: bool = True
    include_traffic_features: bool = True
    normalize_features: bool = True
    add_self_loops: bool = True
    directed_graph: bool = True
    max_edge_distance: float = 200.0  # 最大边连接距离
    use_hierarchical_structure: bool = True  # 是否使用层次结构
    
@dataclass
class NodeFeatures:
    """节点特征数据结构"""
    node_id: str
    position: np.ndarray
    node_type: str  # 'endpoint', 'key_node', 'intersection'
    importance: float
    traffic_capacity: int
    road_class: str
    is_endpoint: bool
    path_memberships: List[str]
    spatial_features: np.ndarray = field(default_factory=lambda: np.array([]))
    
@dataclass
class EdgeFeatures:
    """边特征数据结构"""
    source: str
    target: str
    path_length: float
    road_class: str
    curvature: float
    grade: float
    capacity: float
    bidirectional: bool = True

@dataclass
class GraphTopology:
    """图拓扑结构"""
    nodes: Dict[str, NodeFeatures]
    edges: List[EdgeFeatures]
    adjacency_matrix: np.ndarray
    node_feature_matrix: np.ndarray
    edge_feature_matrix: np.ndarray
    node_to_index: Dict[str, int]
    index_to_node: Dict[int, str]

class Stage1TopologyExtractor:
    """第一阶段拓扑提取器"""
    
    def __init__(self, config: GNNTopologyConfig = None):
        self.config = config or GNNTopologyConfig()
        self.topology = None
        self.raw_data = None
        
    def load_stage1_data(self, json_file_path: str) -> Dict:
        """加载第一阶段导出数据"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                self.raw_data = json.load(f)
            
            print(f"✅ 成功加载第一阶段数据: {json_file_path}")
            print(f"📊 数据概览:")
            print(f"  - 关键节点数: {len(self.raw_data.get('key_nodes_info', {}))}")
            print(f"  - 整合路径数: {len(self.raw_data.get('consolidated_paths_info', {}))}")
            print(f"  - 节点减少率: {self.raw_data.get('enhanced_consolidation_stats', {}).get('node_reduction_ratio', 0):.1%}")
            
            return self.raw_data
            
        except Exception as e:
            raise RuntimeError(f"❌ 加载第一阶段数据失败: {e}")
    
    def extract_topology(self) -> GraphTopology:
        """提取图拓扑结构"""
        if not self.raw_data:
            raise RuntimeError("请先加载第一阶段数据")
        
        print("🔄 开始提取图拓扑结构...")
        
        # 提取节点
        nodes = self._extract_nodes()
        print(f"📍 提取节点完成: {len(nodes)}个节点")
        
        # 提取边
        edges = self._extract_edges()
        print(f"🔗 提取边完成: {len(edges)}条边")
        
        # 构建邻接矩阵和特征矩阵
        node_to_index = {node_id: idx for idx, node_id in enumerate(nodes.keys())}
        index_to_node = {idx: node_id for node_id, idx in node_to_index.items()}
        
        adjacency_matrix = self._build_adjacency_matrix(nodes, edges, node_to_index)
        node_feature_matrix = self._build_node_feature_matrix(nodes, node_to_index)
        edge_feature_matrix = self._build_edge_feature_matrix(edges)
        
        self.topology = GraphTopology(
            nodes=nodes,
            edges=edges,
            adjacency_matrix=adjacency_matrix,
            node_feature_matrix=node_feature_matrix,
            edge_feature_matrix=edge_feature_matrix,
            node_to_index=node_to_index,
            index_to_node=index_to_node
        )
        
        print("✅ 图拓扑结构提取完成")
        return self.topology
    
    def _extract_nodes(self) -> Dict[str, NodeFeatures]:
        """提取节点信息"""
        nodes = {}
        key_nodes_info = self.raw_data.get('key_nodes_info', {})
        
        for node_id, node_data in key_nodes_info.items():
            # 基础特征
            position = np.array(node_data['position'][:2])  # 只取x,y坐标
            
            # 空间特征
            spatial_features = []
            if self.config.include_spatial_features:
                # 添加空间特征：坐标、角度等
                spatial_features.extend([
                    position[0], position[1],  # x, y坐标
                    node_data['position'][2] if len(node_data['position']) > 2 else 0.0,  # 角度或高程
                ])
            
            # 道路特征
            if self.config.include_road_features:
                road_class_encoding = self._encode_road_class(node_data.get('road_class', 'secondary'))
                spatial_features.extend(road_class_encoding)
            
            # 交通特征
            if self.config.include_traffic_features:
                spatial_features.extend([
                    node_data.get('importance', 1.0) / 10.0,  # 归一化重要性
                    node_data.get('traffic_capacity', 100) / 200.0,  # 归一化容量
                    len(node_data.get('path_memberships', [])) / 50.0,  # 归一化路径数
                ])
            
            nodes[node_id] = NodeFeatures(
                node_id=node_id,
                position=position,
                node_type=node_data.get('node_type', 'key_node'),
                importance=node_data.get('importance', 1.0),
                traffic_capacity=node_data.get('traffic_capacity', 100),
                road_class=node_data.get('road_class', 'secondary'),
                is_endpoint=node_data.get('is_endpoint', False),
                path_memberships=node_data.get('path_memberships', []),
                spatial_features=np.array(spatial_features)
            )
        
        return nodes
    
    def _extract_edges(self) -> List[EdgeFeatures]:
        """提取边信息"""
        edges = []
        consolidated_paths = self.raw_data.get('consolidated_paths_info', {})
        key_nodes_info = self.raw_data.get('key_nodes_info', {})
        
        # 从整合路径中提取边
        for path_id, path_info in consolidated_paths.items():
            key_nodes = path_info.get('key_nodes', [])
            
            for i in range(len(key_nodes) - 1):
                source_node = key_nodes[i]
                target_node = key_nodes[i + 1]
                
                # 检查节点是否存在
                if source_node not in key_nodes_info or target_node not in key_nodes_info:
                    continue
                
                # 计算边特征
                source_pos = np.array(key_nodes_info[source_node]['position'][:2])
                target_pos = np.array(key_nodes_info[target_node]['position'][:2])
                distance = np.linalg.norm(target_pos - source_pos)
                
                # 跳过过长的边（可能是错误连接）
                if distance > self.config.max_edge_distance:
                    continue
                
                edge = EdgeFeatures(
                    source=source_node,
                    target=target_node,
                    path_length=distance,
                    road_class=path_info.get('road_class', 'secondary'),
                    curvature=path_info.get('avg_curvature', 0.0),
                    grade=0.0,  # 可以从path_info中提取
                    capacity=self._get_road_capacity(path_info.get('road_class', 'secondary')),
                    bidirectional=True
                )
                edges.append(edge)
        
        # 添加空间邻近边（用于增强连通性）
        edges.extend(self._add_spatial_proximity_edges(key_nodes_info))
        
        return edges
    
    def _add_spatial_proximity_edges(self, key_nodes_info: Dict) -> List[EdgeFeatures]:
        """添加空间邻近边"""
        proximity_edges = []
        nodes_list = list(key_nodes_info.items())
        
        for i, (node1_id, node1_data) in enumerate(nodes_list):
            pos1 = np.array(node1_data['position'][:2])
            
            for j, (node2_id, node2_data) in enumerate(nodes_list[i+1:], i+1):
                pos2 = np.array(node2_data['position'][:2])
                distance = np.linalg.norm(pos2 - pos1)
                
                # 添加邻近节点间的边
                if distance < 50.0:  # 50米内的节点
                    edge = EdgeFeatures(
                        source=node1_id,
                        target=node2_id,
                        path_length=distance,
                        road_class='auxiliary',
                        curvature=0.0,
                        grade=0.0,
                        capacity=50.0,
                        bidirectional=True
                    )
                    proximity_edges.append(edge)
        
        return proximity_edges
    
    def _build_adjacency_matrix(self, nodes: Dict, edges: List[EdgeFeatures], 
                               node_to_index: Dict[str, int]) -> np.ndarray:
        """构建邻接矩阵"""
        n_nodes = len(nodes)
        adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        
        for edge in edges:
            if edge.source in node_to_index and edge.target in node_to_index:
                i = node_to_index[edge.source]
                j = node_to_index[edge.target]
                
                # 使用距离的倒数作为权重
                weight = 1.0 / (edge.path_length + 1e-6)
                adj_matrix[i, j] = weight
                
                if edge.bidirectional:
                    adj_matrix[j, i] = weight
        
        # 添加自环
        if self.config.add_self_loops:
            np.fill_diagonal(adj_matrix, 1.0)
        
        return adj_matrix
    
    def _build_node_feature_matrix(self, nodes: Dict, node_to_index: Dict[str, int]) -> np.ndarray:
        """构建节点特征矩阵"""
        n_nodes = len(nodes)
        
        # 确定特征维度
        sample_features = next(iter(nodes.values())).spatial_features
        feature_dim = len(sample_features)
        
        feature_matrix = np.zeros((n_nodes, feature_dim), dtype=np.float32)
        
        for node_id, node in nodes.items():
            idx = node_to_index[node_id]
            feature_matrix[idx] = node.spatial_features
        
        # 特征归一化
        if self.config.normalize_features:
            feature_matrix = self._normalize_features(feature_matrix)
        
        return feature_matrix
    
    def _build_edge_feature_matrix(self, edges: List[EdgeFeatures]) -> np.ndarray:
        """构建边特征矩阵"""
        n_edges = len(edges)
        
        # 边特征：[路径长度, 道路等级编码, 曲率, 坡度, 容量]
        edge_features = []
        
        for edge in edges:
            features = [
                edge.path_length / 100.0,  # 归一化距离
                *self._encode_road_class(edge.road_class),  # 道路等级编码
                edge.curvature,
                edge.grade,
                edge.capacity / 200.0,  # 归一化容量
            ]
            edge_features.append(features)
        
        edge_matrix = np.array(edge_features, dtype=np.float32)
        
        if self.config.normalize_features:
            edge_matrix = self._normalize_features(edge_matrix)
        
        return edge_matrix
    
    def _encode_road_class(self, road_class: str) -> List[float]:
        """道路等级编码"""
        encoding_map = {
            'primary': [1.0, 0.0, 0.0],
            'secondary': [0.0, 1.0, 0.0],
            'auxiliary': [0.0, 0.0, 1.0],
        }
        return encoding_map.get(road_class, [0.0, 1.0, 0.0])
    
    def _get_road_capacity(self, road_class: str) -> float:
        """获取道路容量"""
        capacity_map = {
            'primary': 200.0,
            'secondary': 100.0,
            'auxiliary': 50.0,
        }
        return capacity_map.get(road_class, 100.0)
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """特征归一化"""
        # 使用标准化：(x - mean) / std
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std[std == 0] = 1.0  # 避免除零
        
        normalized = (features - mean) / std
        return normalized

class GNNDataConverter:
    """GNN数据格式转换器"""
    
    def __init__(self, topology: GraphTopology):
        self.topology = topology
    
    def to_pytorch_geometric(self) -> 'Data':
        """转换为PyTorch Geometric格式"""
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise RuntimeError("PyTorch Geometric not available")
        
        # 构建边索引
        edge_index = []
        edge_attr = []
        
        for i, edge in enumerate(self.topology.edges):
            source_idx = self.topology.node_to_index[edge.source]
            target_idx = self.topology.node_to_index[edge.target]
            
            edge_index.append([source_idx, target_idx])
            edge_attr.append(self.topology.edge_feature_matrix[i])
            
            if edge.bidirectional:
                edge_index.append([target_idx, source_idx])
                edge_attr.append(self.topology.edge_feature_matrix[i])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # 节点特征
        x = torch.tensor(self.topology.node_feature_matrix, dtype=torch.float)
        
        # 节点位置
        pos = torch.tensor([
            node.position for node in self.topology.nodes.values()
        ], dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
        
        # 添加节点类型
        node_types = []
        for node in self.topology.nodes.values():
            if node.is_endpoint:
                node_types.append(0)  # 端点
            elif node.node_type == 'key_node':
                node_types.append(1)  # 关键节点
            else:
                node_types.append(2)  # 其他
        
        data.node_type = torch.tensor(node_types, dtype=torch.long)
        
        return data
    
    def to_dgl(self) -> 'dgl.DGLGraph':
        """转换为DGL格式"""
        if not DGL_AVAILABLE:
            raise RuntimeError("DGL not available")
        
        # 构建边列表
        src_nodes = []
        dst_nodes = []
        
        for edge in self.topology.edges:
            source_idx = self.topology.node_to_index[edge.source]
            target_idx = self.topology.node_to_index[edge.target]
            
            src_nodes.append(source_idx)
            dst_nodes.append(target_idx)
            
            if edge.bidirectional:
                src_nodes.append(target_idx)
                dst_nodes.append(source_idx)
        
        # 创建图
        g = dgl.graph((src_nodes, dst_nodes))
        
        # 添加节点特征
        g.ndata['feat'] = torch.tensor(self.topology.node_feature_matrix, dtype=torch.float)
        g.ndata['pos'] = torch.tensor([
            node.position for node in self.topology.nodes.values()
        ], dtype=torch.float)
        
        # 添加边特征
        edge_features = []
        for i, edge in enumerate(self.topology.edges):
            edge_features.append(self.topology.edge_feature_matrix[i])
            if edge.bidirectional:
                edge_features.append(self.topology.edge_feature_matrix[i])
        
        g.edata['feat'] = torch.tensor(edge_features, dtype=torch.float)
        
        return g
    
    def to_networkx(self) -> nx.Graph:
        """转换为NetworkX格式"""
        if self.topology.config.directed_graph:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        
        # 添加节点
        for node_id, node in self.topology.nodes.items():
            G.add_node(node_id, 
                      pos=node.position,
                      node_type=node.node_type,
                      importance=node.importance,
                      capacity=node.traffic_capacity,
                      is_endpoint=node.is_endpoint)
        
        # 添加边
        for edge in self.topology.edges:
            G.add_edge(edge.source, edge.target,
                      weight=edge.path_length,
                      road_class=edge.road_class,
                      capacity=edge.capacity)
        
        return G

class MultiAgentEnvironmentExtractor:
    """多智能体环境提取器"""
    
    def __init__(self, topology: GraphTopology):
        self.topology = topology
    
    def extract_agent_config(self) -> Dict:
        """提取智能体配置"""
        # 识别装卸点和停车场
        loading_points = []
        unloading_points = []
        parking_points = []
        
        for node_id, node in self.topology.nodes.items():
            if node.is_endpoint:
                if 'L' in node_id and 'to' in node_id:
                    loading_points.append(node_id)
                elif 'U' in node_id:
                    unloading_points.append(node_id)
                elif 'P' in node_id:
                    parking_points.append(node_id)
        
        return {
            'loading_points': loading_points,
            'unloading_points': unloading_points,
            'parking_points': parking_points,
            'total_nodes': len(self.topology.nodes),
            'total_edges': len(self.topology.edges),
            'max_agents': min(len(loading_points) * 2, 50),  # 限制智能体数量
        }
    
    def extract_constraints(self) -> Dict:
        """提取约束条件"""
        # 从第一阶段数据中提取动力学约束
        constraints = {
            'max_speed': 30.0,  # km/h
            'max_acceleration': 2.0,  # m/s²
            'turning_radius': 15.0,  # m
            'max_grade': 0.15,  # 15%
            'vehicle_length': 12.0,  # m
            'safety_distance': 5.0,  # m
        }
        
        # 节点容量约束
        node_constraints = {}
        for node_id, node in self.topology.nodes.items():
            node_constraints[node_id] = {
                'max_occupancy': node.traffic_capacity // 10,  # 同时容纳车辆数
                'service_time': 30.0 if node.is_endpoint else 5.0,  # 服务时间（秒）
            }
        
        constraints['node_constraints'] = node_constraints
        return constraints

def main():
    """主函数示例"""
    # 配置提取器
    config = GNNTopologyConfig(
        include_spatial_features=True,
        include_road_features=True,
        include_traffic_features=True,
        normalize_features=True,
        add_self_loops=True,
        directed_graph=True
    )
    
    # 创建提取器
    extractor = Stage1TopologyExtractor(config)
    
    # 加载第一阶段数据
    try:
        extractor.load_stage1_data("Topu_Nanjing.json")
        
        # 提取拓扑结构
        topology = extractor.extract_topology()
        
        print(f"\n📊 拓扑结构统计:")
        print(f"  - 节点数量: {len(topology.nodes)}")
        print(f"  - 边数量: {len(topology.edges)}")
        print(f"  - 节点特征维度: {topology.node_feature_matrix.shape[1]}")
        print(f"  - 边特征维度: {topology.edge_feature_matrix.shape[1]}")
        
        # 转换为不同格式
        converter = GNNDataConverter(topology)
        
        # PyTorch Geometric格式
        if TORCH_GEOMETRIC_AVAILABLE:
            pyg_data = converter.to_pytorch_geometric()
            print(f"\n🔥 PyTorch Geometric数据:")
            print(f"  - 节点特征: {pyg_data.x.shape}")
            print(f"  - 边索引: {pyg_data.edge_index.shape}")
            print(f"  - 边特征: {pyg_data.edge_attr.shape}")
            
            # 保存PyG数据
            torch.save(pyg_data, "gnn_topology_pyg.pt")
            print("💾 PyTorch Geometric数据已保存: gnn_topology_pyg.pt")
        
        # DGL格式
        if DGL_AVAILABLE:
            dgl_graph = converter.to_dgl()
            print(f"\n🔥 DGL数据:")
            print(f"  - 节点数: {dgl_graph.num_nodes()}")
            print(f"  - 边数: {dgl_graph.num_edges()}")
            
            # 保存DGL数据
            dgl.save_graphs("gnn_topology_dgl.bin", [dgl_graph])
            print("💾 DGL数据已保存: gnn_topology_dgl.bin")
        
        # NetworkX格式
        nx_graph = converter.to_networkx()
        print(f"\n🔥 NetworkX数据:")
        print(f"  - 节点数: {nx_graph.number_of_nodes()}")
        print(f"  - 边数: {nx_graph.number_of_edges()}")
        
        # 多智能体环境配置
        ma_extractor = MultiAgentEnvironmentExtractor(topology)
        agent_config = ma_extractor.extract_agent_config()
        constraints = ma_extractor.extract_constraints()
        
        print(f"\n🤖 多智能体环境:")
        print(f"  - 装载点: {len(agent_config['loading_points'])}")
        print(f"  - 卸载点: {len(agent_config['unloading_points'])}")
        print(f"  - 停车场: {len(agent_config['parking_points'])}")
        print(f"  - 最大智能体数: {agent_config['max_agents']}")
        
        # 保存完整配置
        full_config = {
            'topology_config': config.__dict__,
            'agent_config': agent_config,
            'constraints': constraints,
            'node_mapping': topology.node_to_index,
        }
        
        with open("stage2_gnn_config.json", 'w', encoding='utf-8') as f:
            json.dump(full_config, f, indent=2, ensure_ascii=False, default=str)
        
        print("\n✅ 第二阶段GNN配置已保存: stage2_gnn_config.json")
        print("🚀 准备就绪，可以开始第二阶段GNN建模！")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")

if __name__ == "__main__":
    main()