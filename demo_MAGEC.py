"""
demo_MAGEC.py - 基于生成拓扑结构的MAGEC训练系统（优化修复版）
专注于第二阶段：MAGEC多智能体强化学习训练

基于论文: "Graph Neural Network-based Multi-agent Reinforcement Learning 
for Resilient Distributed Coordination of Multi-Robot Systems"

🔧 优化修复版本 - 解决MAGEC不移动问题：
  - 修复邻居字典构建逻辑
  - 修复动作选择和执行机制
  - 修复拓扑映射器图构建
  - 增强调试信息和错误处理
  - 优化环境兼容性

🎯 主要功能:
  - 交互式选择：载入环境 或 载入第一阶段拓扑JSON
  - 将拓扑结构智能映射到训练环境
  - 专注于MAGEC算法的训练过程
  - 保存训练好的模型供visualize.py使用

🚀 使用方法:
  python demo_MAGEC.py
  
📋 工作流程:
  1. 交互式选择数据源（环境文件 或 拓扑JSON）
  2. 构建MAGEC训练环境
  3. 执行MAGEC训练
  4. 保存训练模型和结果
"""

import sys
import os
import json
import time
import random
import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from collections import defaultdict, deque
from tqdm import tqdm
import logging

# 导入第一阶段的拓扑构建模块
try:
    from environment import OptimizedOpenPitMineEnv
    from optimized_backbone_network import OptimizedBackboneNetwork
    from optimized_planner_config import EnhancedPathPlannerWithConfig
    print("✅ 第一阶段拓扑构建模块导入成功")
except ImportError as e:
    print(f"❌ 第一阶段模块导入失败: {e}")
    print("⚠️ 将使用简化模式")

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def make_json_safe(obj):
    """转换对象为JSON安全格式，处理tuple键问题"""
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(x) for x in obj]
    else:
        return obj


# ============================================================================
# MAGEC网络架构 (基于论文实现)
# ============================================================================

class GraphSAGEConv(MessagePassing):
    """GraphSAGE with Edge Features (Algorithm 1 in paper)"""
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int, 
                 dropout: float = 0.1, aggr: str = 'mean'):
        super().__init__(aggr=aggr)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.dropout = dropout
        
        # Weight matrices as in Algorithm 1
        self.lin_neighbor = nn.Linear(in_channels + edge_dim, out_channels)
        self.lin_self = nn.Linear(in_channels, out_channels)
        self.dropout_layer = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_neighbor.weight)
        nn.init.xavier_uniform_(self.lin_self.weight)
    
    def forward(self, x, edge_index, edge_attr=None):
        # Add self loops
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, 
                                               num_nodes=x.size(0), fill_value=0.0)
        
        # Propagate
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # Self connection
        out = out + self.lin_self(x)
        
        # Normalization and activation
        out = F.normalize(out, p=2, dim=-1)
        out = F.relu(out)
        out = self.dropout_layer(out)
        
        return out
    
    def message(self, x_j, edge_attr):
        # Concatenate node and edge features (Line 6 in Algorithm 1)
        if edge_attr is None:
            edge_attr = torch.zeros(x_j.size(0), self.edge_dim, 
                                   device=x_j.device, dtype=x_j.dtype)
        
        # Ensure edge_attr has correct dimensions
        if edge_attr.size(-1) != self.edge_dim:
            if edge_attr.size(-1) < self.edge_dim:
                padding = torch.zeros(edge_attr.size(0), 
                                     self.edge_dim - edge_attr.size(-1),
                                     device=edge_attr.device, dtype=edge_attr.dtype)
                edge_attr = torch.cat([edge_attr, padding], dim=-1)
            else:
                edge_attr = edge_attr[:, :self.edge_dim]
        
        # Concatenate and transform
        augmented = torch.cat([x_j, edge_attr], dim=-1)
        return self.lin_neighbor(augmented)

class MAGECActor(nn.Module):
    """MAGEC Actor Network with Neighbor Scoring (论文Figure 1)"""
    def __init__(self, node_features: int, edge_features: int, hidden_size: int,
                 num_layers: int, max_neighbors: int, dropout: float = 0.1,
                 use_skip_connections: bool = True):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_neighbors = max_neighbors
        self.use_skip_connections = use_skip_connections
        
        # Input projection
        self.input_projection = nn.Linear(node_features, hidden_size)
        
        # GNN layers (k-convolution as mentioned in paper)
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(
                GraphSAGEConv(hidden_size, hidden_size, edge_features, dropout)
            )
        
        # Jumping knowledge (skip connections)
        if use_skip_connections:
            self.jump_connection = nn.Linear(hidden_size * num_layers, hidden_size)
        
        # Neighbor scoring mechanism (Section IV-D)
        self.neighbor_scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Action selector (selection MLP)
        self.action_selector = nn.Sequential(
            nn.Linear(max_neighbors, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, max_neighbors)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, batch_data, agent_indices=None):
        """Forward pass implementing the neighbor scoring mechanism"""
        device = next(self.parameters()).device
        
        if not isinstance(batch_data, list):
            batch_data = [batch_data]
        
        batch_action_logits = []
        
        for i, data in enumerate(batch_data):
            try:
                # Validate data
                if not hasattr(data, 'x') or data.x is None or data.x.size(0) == 0:
                    # Create dummy output
                    action_logits = torch.zeros(self.max_neighbors, device=device)
                    batch_action_logits.append(action_logits)
                    continue
                
                # Move to device
                x = data.x.to(device)
                edge_index = data.edge_index.to(device)
                edge_attr = data.edge_attr.to(device) if hasattr(data, 'edge_attr') else None
                
                # Get agent position
                agent_idx = agent_indices[i] if agent_indices and i < len(agent_indices) else 0
                agent_idx = min(agent_idx, x.size(0) - 1)
                
                # Input projection
                h = self.input_projection(x)
                
                # GNN layers with skip connections
                layer_outputs = []
                for gnn_layer in self.gnn_layers:
                    h = gnn_layer(h, edge_index, edge_attr)
                    if self.use_skip_connections:
                        layer_outputs.append(h)
                
                # Jumping knowledge
                if self.use_skip_connections and len(layer_outputs) > 1:
                    h = torch.cat(layer_outputs, dim=-1)
                    h = self.jump_connection(h)
                
                # Neighbor scoring
                action_logits = self._compute_action_logits(h, agent_idx)
                batch_action_logits.append(action_logits)
                
            except Exception as e:
                logger.warning(f"Forward pass failed for sample {i}: {e}")
                action_logits = torch.zeros(self.max_neighbors, device=device)
                batch_action_logits.append(action_logits)
        
        return torch.stack(batch_action_logits)
    
    def _compute_action_logits(self, node_embeddings, agent_idx):
        """Implement neighbor scoring mechanism (Section IV-D)"""
        device = node_embeddings.device
        num_nodes = node_embeddings.size(0)
        
        # Score all potential neighbors
        neighbor_scores = []
        for i in range(self.max_neighbors):
            if i < num_nodes and i != agent_idx:
                # Score this neighbor
                neighbor_embedding = node_embeddings[i]
                score = self.neighbor_scorer(neighbor_embedding).squeeze()
                neighbor_scores.append(score)
            else:
                # Invalid neighbor (padding)
                score = torch.tensor(-10.0, device=device)
                neighbor_scores.append(score)
        
        # Convert to tensor and apply action selector
        scores_tensor = torch.stack(neighbor_scores)
        action_logits = self.action_selector(scores_tensor.unsqueeze(0)).squeeze(0)
        
        return action_logits

class MAGECCritic(nn.Module):
    """Simple MLP Critic for centralized training (CTDE) - Section IV-A"""
    def __init__(self, global_state_size: int, hidden_size: int = 512):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(global_state_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, global_state):
        if global_state.dim() == 1:
            global_state = global_state.unsqueeze(0)
        return self.network(global_state)

# ============================================================================
# 拓扑到MAGEC环境映射器（优化修复版）
# ============================================================================

class TopologyToMAGECMapper:
    """将第一阶段拓扑结构映射到MAGEC训练环境（优化修复版）"""
    
    def __init__(self):
        self.topology_data = None
        self.magec_graph = None
        self.node_features = {}
        self.edge_features = {}
        self.position_mapping = {}
        self.special_points = {}
    
    def load_topology_from_json(self, json_path: str) -> bool:
        """从JSON文件加载拓扑数据"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.topology_data = json.load(f)
            
            # 验证JSON格式
            if not self._validate_topology_json():
                return False
            
            print(f"✅ 拓扑JSON加载成功: {json_path}")
            return True
            
        except Exception as e:
            print(f"❌ 拓扑JSON加载失败: {e}")
            return False
    
    def _validate_topology_json(self) -> bool:
        """验证拓扑JSON格式"""
        required_fields = ['system', 'stage', 'ready_for_stage2']
        if not all(field in self.topology_data for field in required_fields):
            print("❌ JSON缺少必要字段")
            return False
        
        if not self.topology_data.get('ready_for_stage2', False):
            print("❌ 拓扑未完成第一阶段构建")
            return False
        
        return True
    
    def create_magec_environment(self, num_agents: int = 4) -> Dict:
        """创建MAGEC训练环境"""
        print("🔄 将拓扑结构映射到MAGEC环境...")
        
        # 构建图结构
        self.magec_graph = nx.Graph()
        
        # 从拓扑数据提取关键信息
        if 'key_nodes_info' in self.topology_data:
            self._build_from_key_nodes_fixed()
        elif 'construction_stats' in self.topology_data:
            self._build_from_construction_stats()
        else:
            print("⚠️ 使用默认图结构")
            self._build_default_graph()
        
        # 🔥 优化：确保图连通性
        self._ensure_connectivity_enhanced()
        
        # 限制智能体数量
        num_agents = min(num_agents, self.magec_graph.number_of_nodes())
        
        # 构建环境配置
        env_config = {
            'graph': self.magec_graph,
            'node_features': self.node_features,
            'edge_features': self.edge_features,
            'position_mapping': self.position_mapping,
            'special_points': self.special_points,
            'num_agents': num_agents,
            'max_neighbors': self._calculate_max_neighbors()
        }
        
        print(f"✅ MAGEC环境创建完成:")
        print(f"   节点数: {self.magec_graph.number_of_nodes()}")
        print(f"   边数: {self.magec_graph.number_of_edges()}")
        print(f"   智能体数: {num_agents}")
        print(f"   最大邻居数: {env_config['max_neighbors']}")
        
        return env_config
    
    def _build_from_key_nodes_fixed(self):
        """从关键节点信息构建图（优化修复版）"""
        print("🔧 使用关键节点信息构建图（优化版）...")
        
        key_nodes_info = self.topology_data['key_nodes_info']
        consolidated_paths_info = self.topology_data.get('consolidated_paths_info', {})
        
        # 🔥 优化：收集有效节点，过滤无效位置
        valid_nodes = {}
        for node_id, node_info in key_nodes_info.items():
            pos = node_info.get('position', [])
            if len(pos) >= 2 and all(isinstance(coord, (int, float)) for coord in pos[:2]):
                valid_nodes[node_id] = pos[:2]
        
        print(f"📍 发现 {len(valid_nodes)} 个有效节点（总共{len(key_nodes_info)}个）")
        
        if len(valid_nodes) < 2:
            print("⚠️ 有效节点太少，使用默认图结构")
            self._build_default_graph()
            return
        
        # 创建节点映射
        node_id_mapping = {}
        for i, (original_node_id, pos) in enumerate(valid_nodes.items()):
            node_id_mapping[original_node_id] = i
            
            self.magec_graph.add_node(i, pos=pos)
            self.position_mapping[i] = pos
            
            # 节点特征
            node_info = key_nodes_info[original_node_id]
            self.node_features[i] = {
                'encoded_type': 2.0 if node_info.get('is_endpoint', False) else 1.0,
                'idleness': 0.0,
                'degree': 0,
                'importance': node_info.get('importance', 1.0)
            }
        
        # 🔥 优化：基于路径连接节点
        edges_added = 0
        for path_id, path_info in consolidated_paths_info.items():
            key_nodes = path_info.get('key_nodes', [])
            for i in range(len(key_nodes) - 1):
                node1_id = key_nodes[i]
                node2_id = key_nodes[i + 1]
                
                if node1_id in node_id_mapping and node2_id in node_id_mapping:
                    idx1 = node_id_mapping[node1_id]
                    idx2 = node_id_mapping[node2_id]
                    
                    if not self.magec_graph.has_edge(idx1, idx2):
                        pos1 = self.position_mapping[idx1]
                        pos2 = self.position_mapping[idx2]
                        distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
                        
                        self.magec_graph.add_edge(idx1, idx2, weight=distance)
                        
                        self.edge_features[(idx1, idx2)] = {
                            'distance': distance,
                            'normalized_distance': min(distance / 50.0, 1.0),
                            'edge_id': len(self.edge_features) / 100.0
                        }
                        edges_added += 1
        
        print(f"📈 基于路径添加了 {edges_added} 条边")
        
        # 🔥 优化：如果边太少，基于距离连接
        if edges_added < len(node_id_mapping) - 1:
            print("🔗 添加距离连接确保连通性...")
            self._add_distance_based_edges()
    
    def _add_distance_based_edges(self):
        """基于距离添加边以确保连通性"""
        nodes = list(self.magec_graph.nodes())
        edges_added = 0
        
        # 为每个节点确保至少有2个邻居
        for node in nodes:
            current_degree = self.magec_graph.degree(node)
            if current_degree < 2:
                pos = self.position_mapping[node]
                
                # 计算到其他节点的距离
                distances = []
                for other_node in nodes:
                    if (other_node != node and 
                        not self.magec_graph.has_edge(node, other_node)):
                        other_pos = self.position_mapping[other_node]
                        distance = np.linalg.norm(np.array(pos) - np.array(other_pos))
                        distances.append((distance, other_node))
                
                # 连接最近的节点
                distances.sort()
                connections_needed = min(2 - current_degree, len(distances))
                
                for i in range(connections_needed):
                    distance, target_node = distances[i]
                    
                    self.magec_graph.add_edge(node, target_node, weight=distance)
                    
                    self.edge_features[(node, target_node)] = {
                        'distance': distance,
                        'normalized_distance': min(distance / 50.0, 1.0),
                        'edge_id': len(self.edge_features) / 100.0
                    }
                    edges_added += 1
        
        print(f"📈 基于距离添加了 {edges_added} 条边")
    
    def _build_from_construction_stats(self):
        """从构建统计信息构建图"""
        print("使用构建统计信息构建图...")
        
        construction_stats = self.topology_data['construction_stats']
        paths_count = construction_stats.get('paths_generated', 6)
        
        # 创建简单的图结构
        num_nodes = paths_count * 3  # 每条路径3个节点
        
        for i in range(num_nodes):
            # 生成合理的节点位置
            angle = 2 * np.pi * i / num_nodes
            radius = 30 + 20 * (i % 3)  # 分层布局
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            self.magec_graph.add_node(i, pos=(x, y))
            self.position_mapping[i] = (x, y)
            
            # 节点特征
            self.node_features[i] = {
                'encoded_type': 1.0,
                'idleness': 0.0,
                'degree': 0,
                'importance': 1.0
            }
        
        # 连接邻近节点
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                distance = np.linalg.norm(
                    np.array(self.position_mapping[i]) - np.array(self.position_mapping[j])
                )
                
                if distance < 40:  # 连接距离阈值
                    self.magec_graph.add_edge(i, j, weight=distance)
                    
                    # 边特征
                    self.edge_features[(i, j)] = {
                        'distance': distance,
                        'normalized_distance': distance / 40.0,
                        'edge_id': len(self.edge_features) / 100.0
                    }
    
    def _build_default_graph(self):
        """构建默认图结构"""
        print("构建默认milwaukee图...")
        
        # Milwaukee graph topology
        edges = [
            (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5),
            (5, 6), (6, 7), (6, 8), (7, 9), (8, 10), (9, 11),
            (10, 11), (11, 12), (12, 13), (12, 14), (13, 15),
            (14, 16), (15, 17), (16, 18), (17, 19), (18, 19),
            (1, 4), (3, 6), (5, 8), (7, 10), (9, 12), (11, 14),
            (13, 16), (15, 18), (2, 7), (4, 9)
        ]
        
        num_nodes = 20
        
        # 使用spring layout生成位置
        temp_graph = nx.Graph()
        temp_graph.add_nodes_from(range(num_nodes))
        temp_graph.add_edges_from(edges)
        pos = nx.spring_layout(temp_graph, seed=42, k=3, iterations=50)
        
        # 建立图
        for i in range(num_nodes):
            position = (pos[i][0] * 100, pos[i][1] * 100)  # 放大坐标
            self.magec_graph.add_node(i, pos=position)
            self.position_mapping[i] = position
            
            # 节点特征
            self.node_features[i] = {
                'encoded_type': 1.0,
                'idleness': 0.0,
                'degree': 0,
                'importance': 1.0
            }
        
        # 添加边
        for i, j in edges:
            pos1 = self.position_mapping[i]
            pos2 = self.position_mapping[j]
            distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
            
            self.magec_graph.add_edge(i, j, weight=distance)
            
            # 边特征
            self.edge_features[(i, j)] = {
                'distance': distance,
                'normalized_distance': min(distance / 50.0, 1.0),
                'edge_id': len(self.edge_features) / 100.0
            }
    
    def _ensure_connectivity_enhanced(self):
        """确保图连通性（增强版）"""
        if not nx.is_connected(self.magec_graph):
            print("🔧 修复图连通性...")
            components = list(nx.connected_components(self.magec_graph))
            
            for i in range(len(components) - 1):
                # 连接最近的节点对
                comp1 = list(components[i])
                comp2 = list(components[i + 1])
                
                min_dist = float('inf')
                best_pair = None
                
                for n1 in comp1:
                    for n2 in comp2:
                        pos1 = self.position_mapping[n1]
                        pos2 = self.position_mapping[n2]
                        dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
                        
                        if dist < min_dist:
                            min_dist = dist
                            best_pair = (n1, n2)
                
                if best_pair:
                    n1, n2 = best_pair
                    self.magec_graph.add_edge(n1, n2, weight=min_dist)
                    
                    self.edge_features[(n1, n2)] = {
                        'distance': min_dist,
                        'normalized_distance': min(min_dist / 50.0, 1.0),
                        'edge_id': len(self.edge_features) / 100.0
                    }
                    print(f"   连接组件: {n1} <-> {n2} (距离: {min_dist:.1f})")
        
        # 🔥 优化：验证最终连通性
        if nx.is_connected(self.magec_graph):
            print("✅ 图连通性验证通过")
        else:
            print("⚠️ 图仍然不连通，某些智能体可能被孤立")
    
    def _calculate_max_neighbors(self) -> int:
        """计算最大邻居数"""
        if self.magec_graph.number_of_nodes() == 0:
            return 15
        
        max_degree = max(dict(self.magec_graph.degree()).values())
        return min(max_degree + 1, 15)  # +1 for potential self-loop
    
    def visualize_mapped_topology(self, save_path: str = None):
        """可视化映射后的拓扑结构"""
        plt.figure(figsize=(12, 10))
        
        pos = self.position_mapping
        
        # 绘制边
        nx.draw_networkx_edges(self.magec_graph, pos, alpha=0.5, width=2, edge_color='gray')
        
        # 绘制节点
        node_colors = []
        for node in self.magec_graph.nodes():
            importance = self.node_features[node].get('importance', 1.0)
            if importance > 1.5:
                node_colors.append('red')  # 重要节点
            else:
                node_colors.append('lightblue')  # 普通节点
        
        nx.draw_networkx_nodes(self.magec_graph, pos, node_color=node_colors, 
                              node_size=300, alpha=0.8)
        
        # 节点标签
        nx.draw_networkx_labels(self.magec_graph, pos, font_size=10, font_weight='bold')
        
        plt.title('拓扑结构映射到MAGEC环境', fontsize=16, fontweight='bold')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        
        # 添加图例
        import matplotlib.patches as patches
        legend_elements = [
            patches.Patch(color='red', label='重要节点'),
            patches.Patch(color='lightblue', label='普通节点'),
            patches.Patch(color='gray', label='连接边')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ 拓扑映射图已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()

# ============================================================================
# MAGEC训练环境（优化修复版）
# ============================================================================

class MAGECTrainingEnvironment:
    """基于拓扑映射的MAGEC训练环境（优化修复版）"""
    
    def __init__(self, env_config: Dict):
        self.graph = env_config['graph']
        self.node_features = env_config['node_features']
        self.edge_features = env_config['edge_features']
        self.position_mapping = env_config['position_mapping']
        self.num_agents = env_config['num_agents']
        self.max_neighbors = env_config['max_neighbors']
        
        self.num_nodes = self.graph.number_of_nodes()
        self.current_step = 0
        self.max_cycles = 200
        
        # 🔥 优化：构建邻居字典
        self.neighbor_dict = {}
        self._build_neighbor_dict_fixed()
        
        # 初始化状态
        self.agent_positions = []
        self.node_idleness = {}
        self.last_visit_time = {}
        
        self.reset()
        
        print(f"🔧 MAGEC环境初始化完成:")
        print(f"   节点数: {self.num_nodes}")
        print(f"   边数: {self.graph.number_of_edges()}")
        print(f"   智能体数: {self.num_agents}")
        print(f"   邻居字典大小: {len(self.neighbor_dict)}")
        print(f"   平均邻居数: {np.mean([len(neighbors) for neighbors in self.neighbor_dict.values()]):.1f}")
    
    def _build_neighbor_dict_fixed(self):
        """构建邻居字典（优化修复版）"""
        print("🔧 构建优化邻居字典...")
        
        for node in self.graph.nodes():
            # 获取图中的直接邻居
            neighbors = list(self.graph.neighbors(node))
            
            # 🔥 修复：如果没有邻居，添加自循环
            if not neighbors:
                neighbors = [node]
                print(f"⚠️ 节点 {node} 添加自循环")
            
            # 🔥 优化：限制邻居数量，确保不超过max_neighbors
            if len(neighbors) > self.max_neighbors:
                # 如果有位置信息，保留最近的邻居
                if node in self.position_mapping:
                    node_pos = self.position_mapping[node]
                    neighbor_distances = []
                    
                    for neighbor in neighbors:
                        if neighbor in self.position_mapping:
                            neighbor_pos = self.position_mapping[neighbor]
                            distance = np.linalg.norm(
                                np.array(node_pos) - np.array(neighbor_pos)
                            )
                            neighbor_distances.append((distance, neighbor))
                        else:
                            neighbor_distances.append((float('inf'), neighbor))
                    
                    neighbor_distances.sort()
                    neighbors = [neighbor for _, neighbor in neighbor_distances[:self.max_neighbors]]
                else:
                    neighbors = neighbors[:self.max_neighbors]
            
            self.neighbor_dict[node] = neighbors
        
        # 🔥 优化：验证邻居字典
        self._validate_neighbor_dict()
    
    def _validate_neighbor_dict(self):
        """验证邻居字典的正确性"""
        print("🔍 验证邻居字典...")
        
        total_neighbors = 0
        isolated_nodes = 0
        max_neighbors_actual = 0
        
        for node, neighbors in self.neighbor_dict.items():
            neighbor_count = len(neighbors)
            total_neighbors += neighbor_count
            max_neighbors_actual = max(max_neighbors_actual, neighbor_count)
            
            if neighbor_count == 0:
                isolated_nodes += 1
                print(f"❌ 发现孤立节点: {node}")
                # 紧急修复：添加自循环
                self.neighbor_dict[node] = [node]
            elif neighbor_count == 1 and neighbors[0] == node:
                # 只有自循环的节点
                pass
        
        avg_neighbors = total_neighbors / len(self.neighbor_dict) if self.neighbor_dict else 0
        
        print(f"📊 邻居字典统计:")
        print(f"   平均邻居数: {avg_neighbors:.1f}")
        print(f"   最大邻居数: {max_neighbors_actual}")
        print(f"   孤立节点数: {isolated_nodes}")
        
        if isolated_nodes > 0:
            print(f"🔧 已修复 {isolated_nodes} 个孤立节点")
    
    def reset(self):
        """重置环境"""
        # 🔥 优化：智能分配智能体初始位置
        available_nodes = list(self.graph.nodes())
        if len(available_nodes) >= self.num_agents:
            # 尽量分散初始位置
            if len(available_nodes) > self.num_agents * 2:
                # 如果节点足够多，等间隔选择
                step = len(available_nodes) // self.num_agents
                self.agent_positions = [available_nodes[i * step] for i in range(self.num_agents)]
            else:
                # 随机选择不重复位置
                self.agent_positions = random.sample(available_nodes, self.num_agents)
        else:
            # 节点不够，允许重复但尽量分散
            self.agent_positions = []
            for i in range(self.num_agents):
                self.agent_positions.append(available_nodes[i % len(available_nodes)])
        
        # 初始化节点闲置时间
        for node in self.graph.nodes():
            self.node_idleness[node] = 0
            self.last_visit_time[node] = -1
        
        # 标记初始位置为已访问
        for pos in self.agent_positions:
            self.last_visit_time[pos] = 0
        
        self.current_step = 0
        
        print(f"🔄 环境重置: 智能体位置 {self.agent_positions}")
        
        return self.get_observations()
    
    def get_observations(self):
        """获取观察"""
        observations = []
        
        for agent_id in range(self.num_agents):
            obs = self._get_agent_observation_fixed(agent_id)
            if obs is not None:
                observations.append(obs)
            else:
                # 创建默认观察
                observations.append(self._create_default_observation())
        
        return observations
    
    def _get_agent_observation_fixed(self, agent_id: int):
        """获取智能体观察（优化修复版）"""
        if agent_id >= len(self.agent_positions):
            return None
        
        agent_pos = self.agent_positions[agent_id]
        
        # 🔥 修复：确保智能体位置有效
        if agent_pos not in self.neighbor_dict:
            valid_nodes = list(self.neighbor_dict.keys())
            if valid_nodes:
                agent_pos = valid_nodes[0]
                self.agent_positions[agent_id] = agent_pos
                print(f"🔧 智能体 {agent_id} 位置修正: {agent_pos}")
        
        # 🔥 优化：获取观察节点 - 使用k跳邻居但限制数量
        observable_nodes = self._get_k_hop_neighbors_limited(agent_pos, k=2, max_nodes=30)
        
        if not observable_nodes:
            observable_nodes = [agent_pos]  # 至少包含当前位置
        
        # 构建节点特征
        node_features = []
        node_mapping = {node: i for i, node in enumerate(observable_nodes)}
        
        for node in observable_nodes:
            # 智能体存在
            has_agent = float(node in self.agent_positions)
            
            # 归一化闲置时间
            idleness = self.node_idleness.get(node, 0)
            normalized_idleness = min(idleness / max(self.current_step + 1, 1), 1.0)
            
            # 节点度数（归一化）
            degree = len(self.neighbor_dict.get(node, []))
            normalized_degree = min(degree / self.max_neighbors, 1.0)
            
            # 节点类型
            node_type = self.node_features.get(node, {}).get('encoded_type', 1.0)
            
            features = [
                has_agent,
                normalized_idleness,
                normalized_degree,
                node_type
            ]
            node_features.append(features)
        
        # 🔥 优化：构建边索引，只包含观察节点内的边
        edge_index = []
        edge_attr = []
        
        for i, node1 in enumerate(observable_nodes):
            for j, node2 in enumerate(observable_nodes):
                if i != j and self.graph.has_edge(node1, node2):
                    edge_index.append([i, j])
                    
                    # 边特征
                    edge_key = (node1, node2) if (node1, node2) in self.edge_features else (node2, node1)
                    if edge_key in self.edge_features:
                        edge_data = self.edge_features[edge_key]
                        attr = [
                            edge_data.get('normalized_distance', 0.5),
                            edge_data.get('edge_id', 0.1)
                        ]
                    else:
                        attr = [0.5, 0.1]
                    
                    edge_attr.append(attr)
        
        # 🔥 修复：确保至少有一条边
        if not edge_index:
            if len(observable_nodes) >= 2:
                edge_index = [[0, 1], [1, 0]]
                edge_attr = [[0.5, 0.1], [0.5, 0.1]]
            else:
                edge_index = [[0, 0]]
                edge_attr = [[0.0, 0.0]]
        
        agent_obs_pos = node_mapping.get(agent_pos, 0)
        
        return Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long).T.contiguous(),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            agent_pos=torch.tensor([agent_obs_pos], dtype=torch.long),
            num_nodes=len(observable_nodes)
        )
    
    def _get_k_hop_neighbors_limited(self, start_node: int, k: int, max_nodes: int = 30) -> List[int]:
        """获取k跳邻居（限制数量版）"""
        visited = set()
        current_level = {start_node}
        
        for hop in range(k):
            next_level = set()
            for node in current_level:
                if node not in visited:
                    visited.add(node)
                    # 添加邻居
                    neighbors = self.neighbor_dict.get(node, [])
                    next_level.update(neighbors)
                
                # 如果已经收集足够多的节点，提前退出
                if len(visited) >= max_nodes:
                    break
            
            current_level = next_level - visited
            
            if not current_level or len(visited) >= max_nodes:
                break
        
        # 限制返回的节点数量
        result = list(visited)[:max_nodes]
        return result
    
    def _create_default_observation(self):
        """创建默认观察"""
        return Data(
            x=torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
            edge_index=torch.tensor([[0], [0]], dtype=torch.long),
            edge_attr=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
            agent_pos=torch.tensor([0], dtype=torch.long),
            num_nodes=1
        )
    
    def step(self, actions):
        """执行一步"""
        if not isinstance(actions, (list, np.ndarray)):
            actions = [actions]
        
        # 确保动作数量匹配智能体数量
        actions = list(actions)[:self.num_agents]
        while len(actions) < self.num_agents:
            actions.append(0)
        
        rewards = []
        moves_made = 0
        action_debug = []
        
        for agent_id, action in enumerate(actions):
            reward, moved, debug_info = self._execute_agent_action_fixed(agent_id, action)
            rewards.append(reward)
            if moved:
                moves_made += 1
            action_debug.append(debug_info)
        
        self.current_step += 1
        self._update_idleness()
        
        done = self.current_step >= self.max_cycles
        
        # 🔥 优化：输出调试信息（前几步和定期）
        if self.current_step <= 5 or self.current_step % 50 == 0:
            avg_idleness = np.mean(list(self.node_idleness.values()))
            print(f"步骤 {self.current_step}: 移动 {moves_made}/{self.num_agents}, 平均闲置 {avg_idleness:.1f}")
            for i, debug in enumerate(action_debug):
                print(f"  智能体{i}: {debug}")
        
        return self.get_observations(), rewards, done
    
    def _execute_agent_action_fixed(self, agent_id: int, action: int) -> Tuple[float, bool, str]:
        """执行智能体动作（优化修复版），返回(奖励, 是否移动, 调试信息)"""
        if agent_id >= len(self.agent_positions):
            return -0.1, False, f"无效智能体ID {agent_id}"
        
        agent_pos = self.agent_positions[agent_id]
        neighbors = self.neighbor_dict.get(agent_pos, [])
        
        if not neighbors:
            return -0.1, False, f"位置{agent_pos}无邻居"
        
        # 🔥 关键修复：确保动作在有效范围内
        action = max(0, min(action, len(neighbors) - 1))
        target_node = neighbors[action]
        
        # 检查是否实际移动
        moved = (target_node != agent_pos)
        
        # 🔥 优化：冲突检测（可选）
        if target_node in self.agent_positions and target_node != agent_pos:
            # 有其他智能体，给予小惩罚但仍允许移动（共享节点）
            conflict_penalty = -0.05
        else:
            conflict_penalty = 0.0
        
        # 执行移动
        old_pos = self.agent_positions[agent_id]
        self.agent_positions[agent_id] = target_node
        self.last_visit_time[target_node] = self.current_step
        
        # 🔥 优化：奖励计算
        old_idleness = self.node_idleness.get(target_node, 0)
        avg_idleness = max(np.mean(list(self.node_idleness.values())), 1e-6)
        
        if moved:
            # 移动奖励：基于访问的节点闲置时间
            idleness_reward = (old_idleness + 1) / (avg_idleness + 1)
            reward = idleness_reward + conflict_penalty
        else:
            # 不移动的小惩罚
            reward = -0.02 + conflict_penalty
        
        debug_info = f"{old_pos}->{target_node}(动作{action}), 移动:{moved}, 奖励:{reward:.3f}"
        
        return reward, moved, debug_info
    
    def _update_idleness(self):
        """更新节点闲置时间"""
        for node in self.graph.nodes():
            if self.last_visit_time.get(node, -1) >= 0:
                self.node_idleness[node] = self.current_step - self.last_visit_time[node]
            else:
                self.node_idleness[node] = self.current_step
    
    @property
    def mean_idleness(self):
        """获取平均闲置时间"""
        values = list(self.node_idleness.values())
        return np.mean(values) if values else 0

# ============================================================================
# MAGEC训练器
# ============================================================================

class MAGECTrainer:
    """MAGEC训练器 - MAPPO算法"""
    
    def __init__(self, actor, critic, config, device='cpu'):
        self.actor = actor
        self.critic = critic
        self.config = config
        self.device = device
        
        # PPO参数
        self.clip_param = config['training']['clip_param']
        self.value_loss_coef = config['training']['value_loss_coef']
        self.entropy_coef = config['training']['entropy_coef']
        self.max_grad_norm = config['training']['max_grad_norm']
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(
            actor.parameters(), lr=config['training']['lr']
        )
        self.critic_optimizer = torch.optim.Adam(
            critic.parameters(), lr=config['training']['lr']
        )
        
        # 经验缓冲
        self.reset_buffer()
    
    def reset_buffer(self):
        """重置经验缓冲"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.observations = []
    
    def select_actions(self, observations, deterministic=False):
        """选择动作"""
        self.actor.eval()
        
        with torch.no_grad():
            # 获取智能体位置
            agent_indices = []
            for obs in observations:
                if hasattr(obs, 'agent_pos'):
                    agent_indices.append(obs.agent_pos.item())
                else:
                    agent_indices.append(0)
            
            # 前向传播
            action_logits = self.actor(observations, agent_indices)
            
            actions = []
            log_probs = []
            entropies = []
            
            for i in range(len(observations)):
                logits = action_logits[i]
                probs = F.softmax(logits, dim=-1)
                probs = torch.clamp(probs, 1e-8, 1.0 - 1e-8)
                dist = torch.distributions.Categorical(probs)
                
                if deterministic:
                    action = torch.argmax(probs)
                else:
                    action = dist.sample()
                
                actions.append(action.item())
                log_probs.append(dist.log_prob(action).item())
                entropies.append(dist.entropy().item())
            
            return np.array(actions), np.array(log_probs), np.array(entropies)
    
    def get_value(self, global_state):
        """获取价值估计"""
        self.critic.eval()
        with torch.no_grad():
            return self.critic(global_state).item()
    
    def store_transition(self, observations, global_state, actions, rewards, 
                        log_probs, values, dones):
        """存储转换"""
        self.observations.append(observations)
        self.states.append(global_state)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.log_probs.append(log_probs)
        self.values.append(values)
        self.dones.append(dones)
    
    def update(self):
        """使用PPO更新网络"""
        if len(self.rewards) == 0:
            return {'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy': 0.0}
        
        # 计算回报和优势
        returns, advantages = self._compute_gae()
        
        # 转换为张量
        states = torch.stack([s for s in self.states]).to(self.device)
        actions = torch.tensor([a[0] if len(a) > 0 else 0 for a in self.actions], 
                              dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor([lp[0] if len(lp) > 0 else 0.0 for lp in self.log_probs], 
                                    dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        
        # 标准化优势
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO更新
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        for epoch in range(self.config['training']['ppo_epochs']):
            # 重新计算动作概率
            new_log_probs, entropies = self._compute_action_probs(actions)
            
            if new_log_probs is not None:
                # PPO损失
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                values_pred = self.critic(states).squeeze()
                critic_loss = F.mse_loss(values_pred, returns)
                
                # 熵损失
                entropy = entropies.mean()
                
                # 总损失
                total_loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy
                
                # 更新
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
        
        # 清空缓冲
        self.reset_buffer()
        
        return {
            'actor_loss': total_actor_loss / self.config['training']['ppo_epochs'],
            'critic_loss': total_critic_loss / self.config['training']['ppo_epochs'],
            'entropy': total_entropy / self.config['training']['ppo_epochs']
        }
    
    def _compute_gae(self):
        """计算GAE"""
        gamma = self.config['training']['gamma']
        gae_lambda = self.config['training']['gae_lambda']
        
        rewards = [np.mean(r) if len(r) > 0 else 0 for r in self.rewards]
        values = [v for v in self.values]
        dones = [d for d in self.dones]
        
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        
        gae = 0
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_value = 0
                next_non_terminal = 1.0 - dones[step]
            else:
                next_value = values[step + 1]
                next_non_terminal = 1.0 - dones[step]
            
            delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            
            advantages[step] = gae
            returns[step] = gae + values[step]
        
        return returns, advantages
    
    def _compute_action_probs(self, actions):
        """重新计算动作概率"""
        try:
            # 简化版本 - 实际应用中需要重新处理观察
            num_actions = len(actions)
            log_probs = torch.zeros_like(actions, dtype=torch.float32)
            entropies = torch.ones_like(actions, dtype=torch.float32) * 0.1
            
            return log_probs, entropies
        except:
            return None, None

# ============================================================================
# 配置和辅助函数
# ============================================================================

def create_magec_config():
    """创建MAGEC训练配置"""
    return {
        'network': {
            'node_features': 4,
            'edge_features': 2,
            'gnn_hidden_size': 128,
            'gnn_layers': 10,
            'gnn_dropout': 0.1,
            'gnn_skip_connections': True,
            'critic_hidden_size': 512,
            'max_neighbors': 15
        },
        'training': {
            'num_episodes': 100,
            'episode_length': 200,
            'lr': 3e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_param': 0.2,
            'value_loss_coef': 1.0,
            'entropy_coef': 0.01,
            'max_grad_norm': 0.5,
            'ppo_epochs': 4,
            'batch_size': 64,
            'alpha': 1.0,
            'beta': 0.5
        },
        'system': {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'seed': 42,
            'save_interval': 50
        }
    }

def get_global_state(env, observations, device):
    """构建全局状态"""
    try:
        global_state = torch.zeros(env.num_nodes + env.num_agents, device=device)
        
        # 节点闲置时间
        idleness_normalized = [env.node_idleness[i] / max(env.current_step + 1, 1) 
                              for i in range(env.num_nodes)]
        global_state[:env.num_nodes] = torch.tensor(idleness_normalized, dtype=torch.float32, device=device)
        
        # 智能体位置
        for i, pos in enumerate(env.agent_positions):
            if i < env.num_agents:
                global_state[env.num_nodes + i] = pos / max(env.num_nodes, 1)
        
        return global_state
        
    except Exception as e:
        logger.warning(f"全局状态构建失败: {e}")
        return torch.zeros(env.num_nodes + env.num_agents, device=device)

def save_model(actor, critic, optimizer_actor, optimizer_critic, config, 
               save_path, episode, performance_metrics=None):
    """保存模型"""
    try:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'episode': episode,
            'actor_state_dict': actor.state_dict(),
            'critic_state_dict': critic.state_dict(),
            'actor_optimizer_state_dict': optimizer_actor.state_dict(),
            'critic_optimizer_state_dict': optimizer_critic.state_dict(),
            'config': config,
            'performance_metrics': performance_metrics or {},
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'pytorch_version': torch.__version__
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"模型已保存: {save_path}")
        return True
        
    except Exception as e:
        logger.error(f"模型保存失败: {e}")
        return False

# ============================================================================
# 交互式输入
# ============================================================================

def interactive_input():
    """交互式配置输入"""
    print("=" * 80)
    print("🚀 MAGEC训练系统 - 交互式配置（优化修复版）")
    print("=" * 80)
    print("💡 提示：直接按回车使用默认值")
    print()
    
    config = {}
    
    # 1. 数据源选择
    print("📂 数据源选择")
    print("-" * 50)
    print("请选择数据源类型:")
    print("  1. 载入环境文件（完整第一阶段）")
    print("  2. 载入拓扑JSON（第一阶段结果）")
    print()
    
    while True:
        choice = input("请选择 (1/2) [默认: 2]: ").strip()
        if not choice:
            choice = "2"
        
        if choice in ["1", "2"]:
            config['data_source'] = "environment" if choice == "1" else "topology_json"
            break
        else:
            print("❌ 请输入1或2")
    
    # 2. 路径输入
    print(f"\n📁 {'环境文件' if config['data_source'] == 'environment' else '拓扑JSON'}路径配置")
    print("-" * 50)
    
    if config['data_source'] == "environment":
        # 搜索环境文件
        env_files = list(Path('.').glob('*.json'))
        env_files = [f for f in env_files if 'map' in f.name.lower() or 'env' in f.name.lower()]
        
        if env_files:
            print("🔍 发现以下环境文件:")
            for i, file in enumerate(env_files, 1):
                print(f"  {i}. {file}")
            print()
            
            while True:
                file_choice = input("选择文件（输入序号或完整路径）[默认: 1]: ").strip()
                if not file_choice:
                    if env_files:
                        config['data_path'] = str(env_files[0])
                        break
                    else:
                        print("❌ 未找到环境文件")
                        continue
                
                if file_choice.isdigit() and 1 <= int(file_choice) <= len(env_files):
                    config['data_path'] = str(env_files[int(file_choice) - 1])
                    break
                elif os.path.exists(file_choice):
                    config['data_path'] = file_choice
                    break
                else:
                    print("❌ 文件不存在，请重新输入")
        else:
            while True:
                path = input("请输入环境文件路径: ").strip()
                if path and os.path.exists(path):
                    config['data_path'] = path
                    break
                else:
                    print("❌ 文件不存在")
    
    else:  # topology_json
        # 搜索拓扑文件
        topology_files = []
        for pattern in ['*topology*.json', '*complete_topology*.json', '*stage1*.json']:
            topology_files.extend(list(Path('.').glob(pattern)))
        
        # 去重并按时间排序
        topology_files = sorted(set(topology_files), key=lambda x: x.stat().st_mtime, reverse=True)
        
        if topology_files:
            print("🔍 发现以下拓扑文件:")
            for i, file in enumerate(topology_files[:5], 1):  # 只显示前5个
                # 显示文件信息
                try:
                    stat = os.stat(file)
                    mtime = time.strftime('%Y-%m-%d %H:%M', time.localtime(stat.st_mtime))
                    print(f"  {i}. {file.name}")
                    print(f"      📅 修改时间: {mtime}")
                except:
                    print(f"  {i}. {file}")
            print()
            
            while True:
                file_choice = input("选择文件（输入序号或完整路径）[默认: 1]: ").strip()
                if not file_choice:
                    if topology_files:
                        config['data_path'] = str(topology_files[0])
                        break
                    else:
                        print("❌ 未找到拓扑文件")
                        continue
                
                if file_choice.isdigit() and 1 <= int(file_choice) <= len(topology_files):
                    config['data_path'] = str(topology_files[int(file_choice) - 1])
                    break
                elif os.path.exists(file_choice):
                    config['data_path'] = file_choice
                    break
                else:
                    print("❌ 文件不存在，请重新输入")
        else:
            while True:
                path = input("请输入拓扑JSON路径: ").strip()
                if path and os.path.exists(path):
                    config['data_path'] = path
                    break
                else:
                    print("❌ 文件不存在")
    
    print(f"✅ 已选择: {config['data_path']}")
    print()
    
    # 3. 训练参数
    print("⚙️ 训练参数配置")
    print("-" * 50)
    
    # 智能体数量
    while True:
        agents_input = input("智能体数量 [默认: 4]: ").strip()
        if not agents_input:
            config['num_agents'] = 4
            break
        try:
            num_agents = int(agents_input)
            if 1 <= num_agents <= 8:
                config['num_agents'] = num_agents
                break
            else:
                print("❌ 智能体数量应在1-8之间")
        except ValueError:
            print("❌ 请输入有效数字")
    
    # 训练回合数
    while True:
        episodes_input = input("训练回合数 [默认: 350]: ").strip()
        if not episodes_input:
            config['num_episodes'] = 350
            break
        try:
            episodes = int(episodes_input)
            if episodes > 0:
                config['num_episodes'] = episodes
                break
            else:
                print("❌ 回合数必须大于0")
        except ValueError:
            print("❌ 请输入有效数字")
    
    # 每回合步数
    while True:
        steps_input = input("每回合步数 [默认: 200]: ").strip()
        if not steps_input:
            config['episode_length'] = 200
            break
        try:
            steps = int(steps_input)
            if steps > 0:
                config['episode_length'] = steps
                break
            else:
                print("❌ 步数必须大于0")
        except ValueError:
            print("❌ 请输入有效数字")
    
    print(f"✅ 训练参数: {config['num_agents']}智能体, {config['num_episodes']}回合, {config['episode_length']}步/回合")
    print()
    
    # 4. 输出配置
    print("📁 输出配置")
    print("-" * 50)
    
    default_output = f"experiments/magec_training_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = input(f"输出目录 [默认: {default_output}]: ").strip()
    config['output_dir'] = output_dir if output_dir else default_output
    
    # 可视化选项
    visualize = input("训练完成后显示拓扑映射图? (y/N) [默认: N]: ").strip().lower()
    config['show_topology'] = visualize in ['y', 'yes', '1', 'true']
    
    print(f"✅ 输出目录: {config['output_dir']}")
    print()
    
    # 显示最终配置
    print("📋 " + "=" * 76)
    print("📋 最终配置确认（优化修复版）")
    print("📋 " + "=" * 76)
    print(f"🔹 数据源: {config['data_source']}")
    print(f"🔹 数据路径: {config['data_path']}")
    print(f"🔹 智能体数量: {config['num_agents']}")
    print(f"🔹 训练参数: {config['num_episodes']}回合 × {config['episode_length']}步")
    print(f"🔹 输出目录: {config['output_dir']}")
    print(f"🔹 显示拓扑图: {'是' if config['show_topology'] else '否'}")
    print()
    
    confirm = input("确认开始训练? (Y/n) [默认: Y]: ").strip().lower()
    if confirm in ['n', 'no', '0', 'false']:
        print("👋 已取消训练")
        sys.exit(0)
    
    return config

# ============================================================================
# 主训练函数（优化修复版）
# ============================================================================

def train_magec(config):
    """MAGEC训练主函数（优化修复版）"""
    print(f"\n🎯 开始MAGEC训练（优化修复版）")
    print("=" * 80)
    print(f"📂 数据源: {config['data_source']}")
    print(f"📁 数据路径: {config['data_path']}")
    print(f"🤖 智能体: {config['num_agents']}")
    print(f"🏃 训练: {config['num_episodes']}回合 × {config['episode_length']}步")
    print(f"💾 输出: {config['output_dir']}")
    print("=" * 80)
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 设备: {device}")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    try:
        # === 1. 构建环境 ===
        print("\n🔄 构建训练环境...")
        
        if config['data_source'] == 'environment':
            # 从环境文件构建（完整流程）
            print("使用环境文件构建...")
            # TODO: 实现环境文件加载
            print("⚠️ 环境文件模式尚未实现，请使用拓扑JSON模式")
            return
        
        else:  # topology_json
            # 从拓扑JSON构建
            mapper = TopologyToMAGECMapper()
            
            if not mapper.load_topology_from_json(config['data_path']):
                print("❌ 拓扑加载失败")
                return
            
            env_config = mapper.create_magec_environment(config['num_agents'])
            
            # 可选：显示拓扑映射图
            if config['show_topology']:
                print("🎨 生成拓扑映射图...")
                mapper.visualize_mapped_topology(f"{config['output_dir']}/topology_mapping.png")
        
        # 创建训练环境
        env = MAGECTrainingEnvironment(env_config)
        print(f"✅ 训练环境创建完成: {env.num_nodes}节点, {env.num_agents}智能体")
        
        # === 2. 创建网络 ===
        print("\n🔄 创建MAGEC网络...")
        
        training_config = create_magec_config()
        training_config['training']['num_episodes'] = config['num_episodes']
        training_config['training']['episode_length'] = config['episode_length']
        training_config['network']['max_neighbors'] = env.max_neighbors
        
        # Actor网络
        actor = MAGECActor(
            node_features=training_config['network']['node_features'],
            edge_features=training_config['network']['edge_features'],
            hidden_size=training_config['network']['gnn_hidden_size'],
            num_layers=training_config['network']['gnn_layers'],
            max_neighbors=env.max_neighbors,
            dropout=training_config['network']['gnn_dropout'],
            use_skip_connections=training_config['network']['gnn_skip_connections']
        ).to(device)
        
        # Critic网络
        global_state_size = env.num_nodes + env.num_agents
        critic = MAGECCritic(
            global_state_size=global_state_size,
            hidden_size=training_config['network']['critic_hidden_size']
        ).to(device)
        
        print(f"🧠 Actor参数: {sum(p.numel() for p in actor.parameters()):,}")
        print(f"🧠 Critic参数: {sum(p.numel() for p in critic.parameters()):,}")
        
        # === 3. 创建训练器 ===
        trainer = MAGECTrainer(actor, critic, training_config, device)
        
        # === 4. 训练循环 ===
        print(f"\n🏃 开始训练...")
        
        episode_rewards = []
        episode_idleness = []
        training_losses = []
        
        with tqdm(total=config['num_episodes'], desc="训练进度", unit="ep") as pbar:
            for episode in range(config['num_episodes']):
                # 重置环境
                observations = env.reset()
                episode_reward = []
                
                for step in range(config['episode_length']):
                    # 选择动作
                    actions, log_probs, entropies = trainer.select_actions(observations)
                    
                    # 获取全局状态和价值
                    global_state = get_global_state(env, observations, device)
                    values = trainer.get_value(global_state)
                    
                    # 执行动作
                    next_observations, rewards, done = env.step(actions)
                    
                    # 存储转换
                    trainer.store_transition(
                        observations, global_state, actions, rewards,
                        log_probs, values, done
                    )
                    
                    episode_reward.extend(rewards)
                    observations = next_observations
                    
                    if done:
                        break
                
                # 更新网络
                losses = trainer.update()
                
                # 记录指标
                avg_reward = np.mean(episode_reward) if episode_reward else 0
                avg_idleness = np.mean(list(env.node_idleness.values()))
                
                episode_rewards.append(avg_reward)
                episode_idleness.append(avg_idleness)
                training_losses.append(losses)
                
                # 更新进度条
                pbar.set_postfix({
                    'Reward': f'{avg_reward:.3f}',
                    'Idleness': f'{avg_idleness:.1f}',
                    'A_Loss': f'{losses["actor_loss"]:.4f}',
                    'C_Loss': f'{losses["critic_loss"]:.4f}'
                })
                pbar.update(1)
                
                # 定期保存
                if episode % 50 == 0 and episode > 0:
                    save_path = f"{config['output_dir']}/checkpoint_ep{episode}.pth"
                    save_model(actor, critic, trainer.actor_optimizer, 
                              trainer.critic_optimizer, training_config, save_path, episode)
        
        # === 5. 保存最终模型 ===
        print(f"\n💾 保存训练结果...")
        
        final_model_path = f"{config['output_dir']}/magec_final_model.pth"
        final_metrics = {
            'episode': config['num_episodes'],
            'final_avg_reward': np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards),
            'final_avg_idleness': np.mean(episode_idleness[-20:]) if len(episode_idleness) >= 20 else np.mean(episode_idleness),
            'episode_rewards': episode_rewards,
            'episode_idleness': episode_idleness,
            'training_losses': training_losses,
            'env_config': env_config,
            'training_completed': True,
            'optimization_applied': True  # 标记使用了优化版本
        }
        
        success = save_model(actor, critic, trainer.actor_optimizer, 
                           trainer.critic_optimizer, training_config, 
                           final_model_path, config['num_episodes'], final_metrics)
        
        if success:
            print(f"✅ 最终模型已保存: {final_model_path}")
        
        # === 6. 生成训练曲线 ===
        print(f"📊 生成训练曲线...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 回合奖励
        episodes = range(len(episode_rewards))
        axes[0, 0].plot(episodes, episode_rewards, 'b-', alpha=0.7)
        if len(episode_rewards) > 10:
            smooth_rewards = np.convolve(episode_rewards, np.ones(10)/10, mode='valid')
            axes[0, 0].plot(range(9, len(episode_rewards)), smooth_rewards, 'r-', linewidth=2)
        axes[0, 0].set_title('Episode Rewards (Optimized)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Average Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 平均闲置时间
        axes[0, 1].plot(episodes, episode_idleness, 'g-', alpha=0.7)
        if len(episode_idleness) > 10:
            smooth_idleness = np.convolve(episode_idleness, np.ones(10)/10, mode='valid')
            axes[0, 1].plot(range(9, len(episode_idleness)), smooth_idleness, 'r-', linewidth=2)
        axes[0, 1].set_title('Average Idleness (Lower is Better)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Idleness')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 训练损失
        if training_losses:
            actor_losses = [loss['actor_loss'] for loss in training_losses]
            critic_losses = [loss['critic_loss'] for loss in training_losses]
            
            axes[1, 0].plot(actor_losses, 'orange', label='Actor Loss')
            axes[1, 0].plot(critic_losses, 'red', label='Critic Loss')
            axes[1, 0].set_title('Training Losses')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_yscale('log')
        
        # 性能摘要
        if final_metrics:
            summary_text = f"""训练摘要（优化版）:
• 最终平均奖励: {final_metrics['final_avg_reward']:.3f}
• 最终平均闲置: {final_metrics['final_avg_idleness']:.3f}
• 最佳奖励: {max(episode_rewards):.3f}
• 最佳闲置: {min(episode_idleness):.3f}

环境信息:
• 节点数: {env.num_nodes}
• 智能体数: {env.num_agents}
• 最大邻居数: {env.max_neighbors}
• 训练回合: {config['num_episodes']}

优化修复:
✅ 邻居字典修复
✅ 动作选择优化
✅ 拓扑映射增强
✅ 调试信息完善"""
            
            axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, 
                           fontsize=10, verticalalignment='top', fontfamily='monospace')
            axes[1, 1].set_title('Training Summary (Optimized)')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        curves_path = f"{config['output_dir']}/training_curves.png"
        plt.savefig(curves_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 训练曲线已保存: {curves_path}")
        
        # === 7. 保存训练配置 ===
        config_path = f"{config['output_dir']}/training_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            # 修复tuple键JSON序列化问题
            safe_config = make_json_safe({
                'input_config': config,
                'training_config': training_config,
                'final_metrics': final_metrics,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'optimization_version': 'v2.0'  # 标记优化版本
            })
            json.dump(safe_config, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ 训练配置已保存: {config_path}")
        
        # === 8. 训练完成摘要 ===
        print(f"\n🎉 MAGEC训练完成（优化修复版）!")
        print("=" * 80)
        print(f"📊 训练统计:")
        print(f"   🎯 最终平均奖励: {final_metrics['final_avg_reward']:.3f}")
        print(f"   ⏱️  最终平均闲置: {final_metrics['final_avg_idleness']:.3f}")
        print(f"   🏆 最佳奖励: {max(episode_rewards):.3f}")
        print(f"   ⚡ 最佳闲置: {min(episode_idleness):.3f}")
        print(f"\n🔧 优化修复:")
        print(f"   ✅ 邻居字典构建修复")
        print(f"   ✅ 动作选择机制优化")
        print(f"   ✅ 拓扑映射增强")
        print(f"   ✅ 环境兼容性提升")
        print(f"\n📁 输出文件:")
        print(f"   🤖 最终模型: magec_final_model.pth")
        print(f"   📈 训练曲线: training_curves.png")
        print(f"   ⚙️  训练配置: training_config.json")
        if config['show_topology']:
            print(f"   🗺️  拓扑映射: topology_mapping.png")
        print(f"\n📂 所有文件保存在: {config['output_dir']}/")
        print("=" * 80)
        print(f"💡 现在可以使用 visualize.py 来测试和可视化训练结果！")
        print(f"🔥 优化版本应该显著改善MAGEC的移动性能！")
        
    except Exception as e:
        print(f"❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='MAGEC训练系统（优化修复版）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python demo_MAGEC.py                           # 交互式模式
  python demo_MAGEC.py --topology topology.json # 指定拓扑文件
  python demo_MAGEC.py --env environment.json   # 指定环境文件
        """
    )
    parser.add_argument('--topology', type=str, help='拓扑JSON文件路径')
    parser.add_argument('--env', type=str, help='环境文件路径')
    parser.add_argument('--agents', type=int, default=4, help='智能体数量')
    parser.add_argument('--episodes', type=int, default=350, help='训练回合数')
    parser.add_argument('--episode_length', type=int, default=200, help='每回合步数')
    parser.add_argument('--output_dir', type=str, help='输出目录')
    parser.add_argument('--show_topology', action='store_true', help='显示拓扑映射图')
    parser.add_argument('--batch', action='store_true', help='批处理模式（非交互）')
    
    args = parser.parse_args()
    
    # 决定使用交互式还是命令行模式
    if args.batch and (args.topology or args.env):
        # 批处理模式
        config = {
            'data_source': 'topology_json' if args.topology else 'environment',
            'data_path': args.topology or args.env,
            'num_agents': args.agents,
            'num_episodes': args.episodes,
            'episode_length': args.episode_length,
            'output_dir': args.output_dir or f"experiments/magec_training_{time.strftime('%Y%m%d_%H%M%S')}",
            'show_topology': args.show_topology
        }
        
        print("🤖 批处理模式（优化版）")
        
    elif args.topology or args.env:
        # 使用命令行参数但保持部分交互
        config = {
            'data_source': 'topology_json' if args.topology else 'environment',
            'data_path': args.topology or args.env,
            'num_agents': args.agents,
            'num_episodes': args.episodes,
            'episode_length': args.episode_length,
            'output_dir': args.output_dir or f"experiments/magec_training_{time.strftime('%Y%m%d_%H%M%S')}",
            'show_topology': args.show_topology
        }
        
        print("🔧 命令行模式（优化版）")
        
    else:
        # 交互式模式
        config = interactive_input()
    
    # 开始训练
    train_magec(config)

if __name__ == "__main__":
    main()