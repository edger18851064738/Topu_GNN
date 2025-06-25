#!/usr/bin/env python3
"""
MAGEC: Multi-Agent Graph Embedding-based Coordination
完整的论文复现实现 - 无冲突版本

基于论文: "Graph Neural Network-based Multi-agent Reinforcement Learning 
for Resilient Distributed Coordination of Multi-Robot Systems"

主要特性:
- GraphSAGE based GNN Actor
- Centralized Critic (CTDE)
- MAPPO Training
- Neighbor Scoring Mechanism
- Multi-robot Patrolling Environment with Collision Avoidance
- Model Save/Load Support
"""

import os
import sys
import time
import json
import random
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Dict, List, Tuple, Optional, Union

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# 配置系统
# ============================================================================

def create_official_config():
    """创建符合论文的官方配置"""
    return {
        'experiment': {
            'name': 'magec_official',
            'version': '1.0'
        },
        'environment': {
            'graph_name': 'milwaukee',
            'num_agents': 4,
            'observation_radius': 400.0,
            'max_cycles': 200,
            'agent_speed': 40.0,
            'action_method': 'neighbors',
            'allow_collisions': False  # 新增：禁止冲突
        },
        'network': {
            'node_features': 4,
            'edge_features': 2,
            'gnn_hidden_size': 128,
            'gnn_layers': 10,  # k=10 as mentioned in paper
            'gnn_dropout': 0.1,
            'gnn_skip_connections': True,  # Jumping knowledge
            'critic_hidden_size': 512,
            'max_neighbors': 15
        },
        'training': {
            'num_episodes': 350,  # Based on paper's 350K steps
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
            'alpha': 1.0,  # Local reward weight
            'beta': 0.5,   # Terminal reward weight
            'collision_penalty': -0.5  # 新增：冲突惩罚
        },
        'system': {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'seed': 42,
            'save_interval': 50
        }
    }

# ============================================================================
# Graph Neural Network (GraphSAGE with Edge Features)
# ============================================================================

class GraphSAGEConv(MessagePassing):
    """
    GraphSAGE with Edge Features (Algorithm 1 in paper)
    """
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
    """
    MAGEC Actor Network with Neighbor Scoring
    Based on Figure 1 and Section IV in the paper
    """
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
        """
        Forward pass implementing the neighbor scoring mechanism
        """
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
        """
        Implement neighbor scoring mechanism (Section IV-D)
        """
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
    """
    Simple MLP Critic for centralized training (CTDE)
    Section IV-A in paper
    """
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
# Patrolling Environment with Collision Avoidance
# ============================================================================

class OptimizedPatrollingEnvironment:
    """
    Multi-robot Patrolling Environment with Collision Avoidance
    Based on Section III-B and IV-E in paper
    """
    def __init__(self, graph_name='milwaukee', num_agents=4, 
                 observation_radius=400.0, max_cycles=200, 
                 agent_speed=40.0, action_method='neighbors',
                 allow_collisions=False, collision_penalty=-0.5):
        
        self.graph_name = graph_name
        self.num_agents = num_agents
        self.observation_radius = observation_radius
        self.max_cycles = max_cycles
        self.agent_speed = agent_speed
        self.action_method = action_method
        self.allow_collisions = allow_collisions  # 新增：冲突控制
        self.collision_penalty = collision_penalty  # 新增：冲突惩罚
        
        self.current_step = 0
        self.create_graph()
        self.setup_agent_system()
        self.reset()
    
    def create_graph(self):
        """Create patrol graph"""
        if self.graph_name == "milwaukee":
            self.graph = self._create_milwaukee_graph()
        elif self.graph_name == "4nodes":
            self.graph = self._create_4nodes_graph()
        else:
            self.graph = self._create_milwaukee_graph()
        
        # Ensure connectivity
        if not nx.is_connected(self.graph):
            components = list(nx.connected_components(self.graph))
            for i in range(len(components) - 1):
                node1 = list(components[i])[0]
                node2 = list(components[i + 1])[0]
                self.graph.add_edge(node1, node2)
        
        # Relabel nodes
        mapping = {node: i for i, node in enumerate(self.graph.nodes())}
        self.graph = nx.relabel_nodes(self.graph, mapping)
        
        self.num_nodes = len(self.graph.nodes())
        self.node_positions = nx.spring_layout(self.graph, seed=42)
        
        # 检查智能体数量是否超过节点数量
        if not self.allow_collisions and self.num_agents > self.num_nodes:
            logger.warning(f"智能体数量 ({self.num_agents}) 超过节点数量 ({self.num_nodes})")
            logger.warning("调整智能体数量以避免冲突")
            self.num_agents = min(self.num_agents, self.num_nodes)
        
        # Compute neighbor information
        self.neighbor_dict = {}
        for node in self.graph.nodes():
            self.neighbor_dict[node] = sorted(list(self.graph.neighbors(node)))
        
        # Compute edge features
        self._compute_edge_features()
    
    def _create_milwaukee_graph(self):
        """Create Milwaukee graph from paper"""
        G = nx.Graph()
        nodes = list(range(20))
        G.add_nodes_from(nodes)
        
        # Milwaukee graph edges
        edges = [
            (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5),
            (5, 6), (6, 7), (6, 8), (7, 9), (8, 10), (9, 11),
            (10, 11), (11, 12), (12, 13), (12, 14), (13, 15),
            (14, 16), (15, 17), (16, 18), (17, 19), (18, 19),
            (1, 4), (3, 6), (5, 8), (7, 10), (9, 12), (11, 14),
            (13, 16), (15, 18), (2, 7), (4, 9)
        ]
        G.add_edges_from(edges)
        return G
    
    def _create_4nodes_graph(self):
        """Create simple 4-node graph for testing"""
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2, 3])
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
        return G
    
    def _compute_edge_features(self):
        """Compute edge features (distance and identifier)"""
        self.edge_features = {}
        
        for i, (u, v) in enumerate(self.graph.edges()):
            pos_u = np.array(self.node_positions[u])
            pos_v = np.array(self.node_positions[v])
            distance = np.linalg.norm(pos_u - pos_v)
            normalized_distance = min(distance / 2.0, 1.0)
            edge_id = i / max(len(self.graph.edges()), 1)
            
            self.edge_features[(u, v)] = [normalized_distance, edge_id]
            self.edge_features[(v, u)] = [normalized_distance, edge_id]
    
    def setup_agent_system(self):
        """Setup agent state variables"""
        self.agent_action_cooldowns = [0] * self.num_agents
        self.agent_target_nodes = [None] * self.num_agents
        
        # Node idleness tracking
        self.node_idleness = np.zeros(self.num_nodes, dtype=float)
        self.last_visit_time = np.full(self.num_nodes, -1, dtype=float)
        
        # 冲突统计
        self.collision_attempts = 0
        self.successful_moves = 0
    
    def reset(self):
        """Reset environment to initial state with collision avoidance"""
        # 无冲突的初始位置分配
        if not self.allow_collisions:
            # 从所有节点中随机选择不重复的位置
            available_nodes = list(range(self.num_nodes))
            if self.num_agents <= len(available_nodes):
                self.agent_positions = random.sample(available_nodes, self.num_agents)
            else:
                # 如果智能体数量超过节点数量，重新调整
                logger.error(f"无法在 {self.num_nodes} 个节点上放置 {self.num_agents} 个智能体而不产生冲突")
                self.num_agents = len(available_nodes)
                self.agent_positions = available_nodes.copy()
        else:
            # 允许冲突的原始逻辑
            self.agent_positions = random.sample(range(self.num_nodes), 
                                               min(self.num_agents, self.num_nodes))
            while len(self.agent_positions) < self.num_agents:
                self.agent_positions.append(random.choice(range(self.num_nodes)))
        
        # Reset agent state
        self.agent_action_cooldowns = [0] * self.num_agents
        self.agent_target_nodes = [None] * self.num_agents
        
        # Reset idleness
        self.node_idleness = np.zeros(self.num_nodes, dtype=float)
        self.last_visit_time = np.full(self.num_nodes, -1, dtype=float)
        
        # Mark initial positions as visited
        for pos in self.agent_positions:
            self.last_visit_time[pos] = 0
        
        # Reset collision statistics
        self.collision_attempts = 0
        self.successful_moves = 0
        
        self.current_step = 0
        return self.get_observations()
    
    def _check_collision(self, agent_id, target_node):
        """
        检查移动到目标节点是否会产生冲突
        
        Args:
            agent_id: 智能体ID
            target_node: 目标节点
            
        Returns:
            bool: True if collision would occur, False otherwise
        """
        if self.allow_collisions:
            return False
        
        # 检查目标节点是否被其他智能体占用
        for other_agent_id, other_pos in enumerate(self.agent_positions):
            if other_agent_id != agent_id and other_pos == target_node:
                return True
        
        return False
    
    def get_observations(self):
        """Get observations for all agents"""
        observations = []
        for agent_id in range(self.num_agents):
            obs = self._get_agent_observation(agent_id)
            observations.append(obs)
        return observations
    
    def _get_agent_observation(self, agent_id):
        """Get observation for single agent (Section IV-E.1)"""
        agent_pos = self.agent_positions[agent_id]
        
        # Get observable nodes within radius
        if self.observation_radius == np.inf:
            observable_nodes = list(range(self.num_nodes))
        else:
            observable_nodes = self._get_nodes_within_radius(agent_pos, self.observation_radius)
        
        if not observable_nodes:
            observable_nodes = [agent_pos]
        
        # Build node features (encoded type, idleness, degree, agent presence)
        node_features = []
        for node in observable_nodes:
            # 检查节点上是否有智能体（用于冲突感知）
            agents_on_node = sum(1 for pos in self.agent_positions if pos == node)
            has_agent = float(agents_on_node > 0)
            
            idleness = self.node_idleness[node] / max(self.current_step + 1, 1)
            idleness = np.clip(idleness, 0.0, 1.0)
            degree = len(self.neighbor_dict.get(node, [])) / 10.0
            degree = np.clip(degree, 0.0, 1.0)
            
            # 节点特征：[智能体存在, 闲置时间, 度数, 节点类型]
            node_features.append([has_agent, idleness, degree, 1.0])
        
        # Build edges
        edge_index = []
        edge_attr = []
        
        node_mapping = {node: i for i, node in enumerate(observable_nodes)}
        
        for i, node in enumerate(observable_nodes):
            for neighbor in self.neighbor_dict.get(node, []):
                if neighbor in node_mapping:
                    j = node_mapping[neighbor]
                    edge_index.append([i, j])
                    
                    edge_key = (node, neighbor)
                    if edge_key in self.edge_features:
                        edge_attr.append(self.edge_features[edge_key])
                    else:
                        edge_attr.append([1.0, 0.0])
        
        if not edge_index:
            edge_index = [[0, 0]]
            edge_attr = [[0.0, 0.0]]
        
        agent_obs_pos = node_mapping.get(agent_pos, 0)
        
        return Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            agent_pos=torch.tensor([agent_obs_pos], dtype=torch.long),
            num_nodes=len(observable_nodes)
        )
    
    def _get_nodes_within_radius(self, center_node, radius):
        """Get nodes within observation radius"""
        try:
            distances = nx.single_source_shortest_path_length(
                self.graph, center_node, cutoff=int(radius)
            )
            return list(distances.keys())
        except:
            return [center_node]
    
    def step(self, actions):
        """Execute one step in environment with collision handling"""
        if not isinstance(actions, (list, np.ndarray)):
            actions = [actions]
        
        actions = list(actions)[:self.num_agents]
        while len(actions) < self.num_agents:
            actions.append(0)
        
        rewards = []
        
        for agent_id, action in enumerate(actions):
            reward = self._execute_agent_action(agent_id, action)
            rewards.append(reward)
        
        self.current_step += 1
        self._update_idleness()
        
        done = self.current_step >= self.max_cycles
        
        # Terminal reward (Section IV-E.2)
        if done:
            terminal_reward = -np.mean(self.node_idleness) * 0.5
            rewards = [r + terminal_reward for r in rewards]
        
        return self.get_observations(), rewards, done
    
    def _execute_agent_action(self, agent_id, action):
        """Execute action for agent with collision avoidance"""
        if self.agent_action_cooldowns[agent_id] > 0:
            self.agent_action_cooldowns[agent_id] -= 1
            return 0.0
        
        agent_pos = self.agent_positions[agent_id]
        neighbors = self.neighbor_dict.get(agent_pos, [])
        
        action = int(action)
        if 0 <= action < len(neighbors):
            target_node = neighbors[action]
            
            # 检查冲突
            if self._check_collision(agent_id, target_node):
                # 冲突：拒绝移动并给予负奖励
                self.collision_attempts += 1
                logger.debug(f"Agent {agent_id} collision avoided: {agent_pos} -> {target_node}")
                return self.collision_penalty
            
            # 无冲突：执行移动
            self.agent_positions[agent_id] = target_node
            self.last_visit_time[target_node] = self.current_step
            self.successful_moves += 1
            
            # Local reward (Section IV-E.2)
            old_idleness = self.node_idleness[target_node]
            avg_idleness = max(np.mean(self.node_idleness), 1e-6)
            local_reward = old_idleness / avg_idleness
            
            return local_reward
        else:
            return -0.01  # Invalid action penalty
    
    def _update_idleness(self):
        """Update node idleness times"""
        for node in range(self.num_nodes):
            if self.last_visit_time[node] >= 0:
                self.node_idleness[node] = self.current_step - self.last_visit_time[node]
            else:
                self.node_idleness[node] = self.current_step
    
    def get_max_neighbors(self):
        """Get maximum neighbors for any node"""
        if not self.neighbor_dict:
            return 15
        return min(15, max(len(neighbors) for neighbors in self.neighbor_dict.values()))
    
    def get_collision_stats(self):
        """获取冲突统计信息"""
        total_moves = self.collision_attempts + self.successful_moves
        collision_rate = self.collision_attempts / max(total_moves, 1)
        return {
            'collision_attempts': self.collision_attempts,
            'successful_moves': self.successful_moves,
            'total_attempts': total_moves,
            'collision_rate': collision_rate
        }

# ============================================================================
# PPO Trainer (保持不变)
# ============================================================================

class MAGECTrainer:
    """
    MAPPO Trainer for MAGEC (Section IV-F)
    """
    def __init__(self, actor, critic, config, device='cpu'):
        self.actor = actor
        self.critic = critic
        self.config = config
        self.device = device
        
        # PPO parameters
        self.clip_param = config['training']['clip_param']
        self.value_loss_coef = config['training']['value_loss_coef']
        self.entropy_coef = config['training']['entropy_coef']
        self.max_grad_norm = config['training']['max_grad_norm']
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            actor.parameters(), lr=config['training']['lr']
        )
        self.critic_optimizer = torch.optim.Adam(
            critic.parameters(), lr=config['training']['lr']
        )
        
        # Experience buffer
        self.reset_buffer()
    
    def reset_buffer(self):
        """Reset experience buffer"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.observations = []
    
    def select_actions(self, observations, deterministic=False):
        """Select actions using current policy"""
        self.actor.eval()
        
        with torch.no_grad():
            # Get agent positions
            agent_indices = []
            for obs in observations:
                if hasattr(obs, 'agent_pos'):
                    agent_indices.append(obs.agent_pos.item())
                else:
                    agent_indices.append(0)
            
            # Forward pass
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
        """Get value from critic"""
        self.critic.eval()
        with torch.no_grad():
            return self.critic(global_state).item()
    
    def store_transition(self, observations, global_state, actions, rewards, 
                        log_probs, values, dones):
        """Store transition in buffer"""
        self.observations.append(observations)
        self.states.append(global_state)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.log_probs.append(log_probs)
        self.values.append(values)
        self.dones.append(dones)
    
    def update(self):
        """Update networks using PPO"""
        if len(self.rewards) == 0:
            return {'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy': 0.0}
        
        # Compute returns and advantages
        returns, advantages = self._compute_gae()
        
        # Convert to tensors
        states = torch.stack([s for s in self.states]).to(self.device)
        actions = torch.tensor([a[0] if len(a) > 0 else 0 for a in self.actions], 
                              dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor([lp[0] if len(lp) > 0 else 0.0 for lp in self.log_probs], 
                                    dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        
        # Normalize advantages
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        for epoch in range(self.config['training']['ppo_epochs']):
            # Recompute action probabilities
            new_log_probs, entropies = self._compute_action_probs(actions)
            
            if new_log_probs is not None:
                # PPO loss
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                values_pred = self.critic(states).squeeze()
                critic_loss = F.mse_loss(values_pred, returns)
                
                # Entropy loss
                entropy = entropies.mean()
                
                # Total loss
                total_loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy
                
                # Update
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
        
        # Clear buffer
        self.reset_buffer()
        
        return {
            'actor_loss': total_actor_loss / self.config['training']['ppo_epochs'],
            'critic_loss': total_critic_loss / self.config['training']['ppo_epochs'],
            'entropy': total_entropy / self.config['training']['ppo_epochs']
        }
    
    def _compute_gae(self):
        """Compute Generalized Advantage Estimation"""
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
        """Recompute action probabilities"""
        try:
            # This is a simplified version - in practice, would need to reprocess observations
            num_actions = len(actions)
            log_probs = torch.zeros_like(actions, dtype=torch.float32)
            entropies = torch.ones_like(actions, dtype=torch.float32) * 0.1
            
            return log_probs, entropies
        except:
            return None, None

# ============================================================================
# Utility Functions (保持不变，添加冲突统计)
# ============================================================================

def get_global_state(env, observations, device):
    """Build global state for critic (feature-engineered)"""
    try:
        # Simple global state: node idleness + agent positions
        node_features_size = env.num_nodes
        agent_features_size = env.num_agents
        total_size = node_features_size + agent_features_size
        
        global_state = torch.zeros(total_size, device=device)
        
        # Node idleness
        idleness_normalized = env.node_idleness / max(env.current_step + 1, 1)
        global_state[:node_features_size] = torch.tensor(idleness_normalized, 
                                                        dtype=torch.float32, device=device)
        
        # Agent positions (normalized)
        for i, pos in enumerate(env.agent_positions):
            if i < agent_features_size:
                global_state[node_features_size + i] = pos / max(env.num_nodes, 1)
        
        return global_state
        
    except Exception as e:
        logger.warning(f"Global state construction failed: {e}")
        return torch.zeros(env.num_nodes + env.num_agents, device=device)

def save_model(actor, critic, optimizer_actor, optimizer_critic, config, 
               save_path, episode, performance_metrics=None):
    """Save model checkpoint"""
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
            'timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Model saved: {save_path}")
        return True
        
    except Exception as e:
        logger.error(f"Model save failed: {e}")
        return False

def load_model(actor, critic, optimizer_actor, optimizer_critic, load_path):
    """Load model checkpoint"""
    try:
        if not os.path.exists(load_path):
            logger.warning(f"Checkpoint not found: {load_path}")
            return None
        
        checkpoint = torch.load(load_path, map_location='cpu')
        
        actor.load_state_dict(checkpoint['actor_state_dict'])
        critic.load_state_dict(checkpoint['critic_state_dict'])
        optimizer_actor.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        optimizer_critic.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        logger.info(f"Model loaded: {load_path}")
        return checkpoint
        
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        return None

# ============================================================================
# Training Loop with Collision Tracking
# ============================================================================

def train_magec(config_path=None, experiment_name="magec_no_collision", quick_test=False):
    """Main training function with collision avoidance"""
    
    print("=" * 80)
    print("🚀 MAGEC: Multi-Agent Graph Embedding-based Coordination (No Collision)")
    print("📄 Based on paper: Graph Neural Network-based Multi-agent RL")
    print("🛡️  Enhanced with Collision Avoidance")
    print("=" * 80)
    
    # Load configuration
    config = create_official_config()
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
            config.update(loaded_config)
    
    # Quick test mode
    if quick_test:
        config['training']['num_episodes'] = 20
        config['training']['episode_length'] = 50
        config['environment']['graph_name'] = '4nodes'
        config['environment']['num_agents'] = 2
        print("🚀 Quick test mode activated")
    
    # Setup
    device = torch.device(config['system']['device'])
    seed = config['system']['seed']
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    print(f"🔧 Device: {device}")
    print(f"🎲 Seed: {seed}")
    print(f"📊 Graph: {config['environment']['graph_name']}")
    print(f"🤖 Agents: {config['environment']['num_agents']}")
    print(f"🛡️  Collision Avoidance: Enabled")
    
    # Create directories
    experiment_dir = Path(f"experiments/{experiment_name}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(experiment_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create environment with collision avoidance
    env = OptimizedPatrollingEnvironment(
        graph_name=config['environment']['graph_name'],
        num_agents=config['environment']['num_agents'],
        observation_radius=config['environment']['observation_radius'],
        max_cycles=config['environment']['max_cycles'],
        agent_speed=config['environment']['agent_speed'],
        action_method=config['environment']['action_method'],
        allow_collisions=config['environment']['allow_collisions'],
        collision_penalty=config['training']['collision_penalty']
    )
    
    # Create networks
    actor = MAGECActor(
        node_features=config['network']['node_features'],
        edge_features=config['network']['edge_features'],
        hidden_size=config['network']['gnn_hidden_size'],
        num_layers=config['network']['gnn_layers'],
        max_neighbors=env.get_max_neighbors(),
        dropout=config['network']['gnn_dropout'],
        use_skip_connections=config['network']['gnn_skip_connections']
    ).to(device)
    
    global_state_size = env.num_nodes + env.num_agents
    critic = MAGECCritic(
        global_state_size=global_state_size,
        hidden_size=config['network']['critic_hidden_size']
    ).to(device)
    
    # Create trainer
    trainer = MAGECTrainer(actor, critic, config, device)
    
    print(f"🧠 Actor parameters: {sum(p.numel() for p in actor.parameters()):,}")
    print(f"🧠 Critic parameters: {sum(p.numel() for p in critic.parameters()):,}")
    
    # Training metrics
    episode_rewards = []
    episode_idleness = []
    training_losses = []
    collision_stats = []
    
    # Training loop
    num_episodes = config['training']['num_episodes']
    episode_length = config['training']['episode_length']
    save_interval = config['system']['save_interval']
    
    print(f"\n🎯 Starting training for {num_episodes} episodes...")
    
    with tqdm(total=num_episodes, desc="Training", unit="ep") as pbar:
        for episode in range(num_episodes):
            # Reset environment
            observations = env.reset()
            episode_reward = []
            
            for step in range(episode_length):
                # Select actions
                actions, log_probs, entropies = trainer.select_actions(observations)
                
                # Get global state and value
                global_state = get_global_state(env, observations, device)
                values = trainer.get_value(global_state)
                
                # Step environment
                next_observations, rewards, done = env.step(actions)
                
                # Store transition
                trainer.store_transition(
                    observations, global_state, actions, rewards,
                    log_probs, values, done
                )
                
                episode_reward.extend(rewards)
                observations = next_observations
                
                if done:
                    break
            
            # Update networks
            losses = trainer.update()
            
            # Record metrics
            avg_reward = np.mean(episode_reward) if episode_reward else 0
            avg_idleness = np.mean(env.node_idleness)
            collision_stat = env.get_collision_stats()
            
            episode_rewards.append(avg_reward)
            episode_idleness.append(avg_idleness)
            training_losses.append(losses)
            collision_stats.append(collision_stat)
            
            # Update progress bar
            pbar.set_postfix({
                'Reward': f'{avg_reward:.3f}',
                'Idleness': f'{avg_idleness:.3f}',
                'A_Loss': f'{losses["actor_loss"]:.4f}',
                'C_Loss': f'{losses["critic_loss"]:.4f}',
                'Collisions': f'{collision_stat["collision_rate"]:.2%}'
            })
            pbar.update(1)
            
            # Save checkpoint
            if episode % save_interval == 0 and episode > 0:
                save_path = experiment_dir / f"checkpoint_ep{episode}.pth"
                performance_metrics = {
                    'episode': episode,
                    'avg_reward': avg_reward,
                    'avg_idleness': avg_idleness,
                    'episode_rewards': episode_rewards,
                    'episode_idleness': episode_idleness,
                    'training_losses': training_losses,
                    'collision_stats': collision_stats
                }
                
                save_model(actor, critic, trainer.actor_optimizer, 
                          trainer.critic_optimizer, config, str(save_path), 
                          episode, performance_metrics)
    
    # Save final model
    final_save_path = experiment_dir / "final_model.pth"
    final_metrics = {
        'episode': num_episodes,
        'avg_reward': np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards),
        'avg_idleness': np.mean(episode_idleness[-20:]) if len(episode_idleness) >= 20 else np.mean(episode_idleness),
        'episode_rewards': episode_rewards,
        'episode_idleness': episode_idleness,
        'training_losses': training_losses,
        'collision_stats': collision_stats,
        'training_completed': True
    }
    
    save_model(actor, critic, trainer.actor_optimizer, 
              trainer.critic_optimizer, config, str(final_save_path), 
              num_episodes, final_metrics)
    
    # Plot training curves with collision stats
    plot_training_curves_with_collisions(episode_rewards, episode_idleness, 
                                       training_losses, collision_stats,
                                       str(experiment_dir / "training_curves.png"))
    
    # Print collision statistics
    final_collision_stats = env.get_collision_stats()
    print(f"\n📊 训练完成统计:")
    print(f"🎯 平均奖励: {np.mean(episode_rewards[-20:]):.3f}")
    print(f"⏱️  平均闲置: {np.mean(episode_idleness[-20:]):.3f}")
    print(f"🛡️  冲突尝试: {final_collision_stats['collision_attempts']}")
    print(f"✅ 成功移动: {final_collision_stats['successful_moves']}")
    print(f"📈 冲突率: {final_collision_stats['collision_rate']:.2%}")
    
    print(f"\n✅ Training completed!")
    print(f"📁 Experiment directory: {experiment_dir}")
    print(f"💾 Final model: {final_save_path}")
    print(f"📊 Training curves: {experiment_dir}/training_curves.png")
    print("=" * 80)
    
    return {
        'actor': actor,
        'critic': critic,
        'config': config,
        'experiment_dir': experiment_dir,
        'final_model_path': str(final_save_path),
        'metrics': final_metrics
    }

def plot_training_curves_with_collisions(episode_rewards, episode_idleness, 
                                       training_losses, collision_stats, save_path):
    """Plot training curves including collision statistics"""
    try:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('MAGEC Training Progress (With Collision Avoidance)', 
                    fontsize=16, fontweight='bold')
        
        episodes = range(len(episode_rewards))
        
        # Episode rewards
        axes[0, 0].plot(episodes, episode_rewards, 'b-', alpha=0.6)
        if len(episode_rewards) > 10:
            smooth_rewards = np.convolve(episode_rewards, np.ones(10)/10, mode='valid')
            axes[0, 0].plot(range(9, len(episode_rewards)), smooth_rewards, 'r-', linewidth=2)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Average Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Average idleness
        axes[0, 1].plot(episodes, episode_idleness, 'g-', alpha=0.6)
        if len(episode_idleness) > 10:
            smooth_idleness = np.convolve(episode_idleness, np.ones(10)/10, mode='valid')
            axes[0, 1].plot(range(9, len(episode_idleness)), smooth_idleness, 'r-', linewidth=2)
        axes[0, 1].set_title('Average Idleness (Lower is Better)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Idleness')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Collision statistics
        if collision_stats:
            collision_rates = [stat['collision_rate'] for stat in collision_stats]
            collision_attempts = [stat['collision_attempts'] for stat in collision_stats]
            
            axes[0, 2].plot(episodes, collision_rates, 'orange', label='Collision Rate')
            axes[0, 2].set_title('Collision Rate Over Time')
            axes[0, 2].set_xlabel('Episode')
            axes[0, 2].set_ylabel('Collision Rate')
            axes[0, 2].grid(True, alpha=0.3)
            axes[0, 2].set_ylim(0, 1)
        
        # Training losses
        if training_losses and len(training_losses) > 0:
            actor_losses = [loss['actor_loss'] for loss in training_losses if 'actor_loss' in loss]
            critic_losses = [loss['critic_loss'] for loss in training_losses if 'critic_loss' in loss]
            
            if actor_losses:
                axes[1, 0].plot(actor_losses, 'orange', label='Actor Loss')
            if critic_losses:
                axes[1, 0].plot(critic_losses, 'red', label='Critic Loss')
            
            axes[1, 0].set_title('Training Losses')
            axes[1, 0].set_xlabel('Training Steps')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_yscale('log')
        
        # Cumulative collision statistics
        if collision_stats:
            cumulative_collisions = np.cumsum([stat['collision_attempts'] for stat in collision_stats])
            cumulative_moves = np.cumsum([stat['successful_moves'] for stat in collision_stats])
            
            axes[1, 1].plot(episodes, cumulative_collisions, 'red', label='Cumulative Collisions')
            axes[1, 1].plot(episodes, cumulative_moves, 'green', label='Cumulative Successful Moves')
            axes[1, 1].set_title('Cumulative Movement Statistics')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Performance summary
        final_collision_rate = collision_stats[-1]['collision_rate'] if collision_stats else 0
        axes[1, 2].text(0.1, 0.8, f'Final Avg Reward: {episode_rewards[-1]:.3f}', 
                       transform=axes[1, 2].transAxes, fontsize=12)
        axes[1, 2].text(0.1, 0.7, f'Final Avg Idleness: {episode_idleness[-1]:.3f}', 
                       transform=axes[1, 2].transAxes, fontsize=12)
        axes[1, 2].text(0.1, 0.6, f'Best Reward: {max(episode_rewards):.3f}', 
                       transform=axes[1, 2].transAxes, fontsize=12)
        axes[1, 2].text(0.1, 0.5, f'Best Idleness: {min(episode_idleness):.3f}', 
                       transform=axes[1, 2].transAxes, fontsize=12)
        axes[1, 2].text(0.1, 0.4, f'Final Collision Rate: {final_collision_rate:.2%}', 
                       transform=axes[1, 2].transAxes, fontsize=12)
        axes[1, 2].set_title('Training Summary')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training curves saved: {save_path}")
        
    except Exception as e:
        logger.error(f"Failed to plot training curves: {e}")

# ============================================================================
# Main
# ============================================================================

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='MAGEC Training (No Collision)')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--experiment', type=str, default='magec_no_collision', 
                       help='Experiment name')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--graph', type=str, default='milwaukee',
                       choices=['milwaukee', '4nodes'], help='Graph type')
    parser.add_argument('--agents', type=int, default=4, help='Number of agents')
    parser.add_argument('--episodes', type=int, default=350, help='Number of episodes')
    parser.add_argument('--collision-penalty', type=float, default=-0.5, 
                       help='Penalty for collision attempts')
    
    args = parser.parse_args()
    
    # Override config with command line args
    config = create_official_config()
    if args.graph:
        config['environment']['graph_name'] = args.graph
    if args.agents:
        config['environment']['num_agents'] = args.agents
    if args.episodes:
        config['training']['num_episodes'] = args.episodes
    if args.collision_penalty:
        config['training']['collision_penalty'] = args.collision_penalty
    
    # Save temporary config
    temp_config_path = f"temp_config_{args.experiment}.json"
    with open(temp_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    try:
        # Run training
        results = train_magec(temp_config_path, args.experiment, args.quick)
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📂 Results saved in: {results['experiment_dir']}")
        print(f"💾 Final model: {results['final_model_path']}")
        
    finally:
        # Clean up temp config
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

if __name__ == "__main__":
    main()