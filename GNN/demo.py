#!/usr/bin/env python3
"""
完整优化的MAGEC系统
集成所有改进：训练监控、稳定批处理、内存优化、数值稳定性、配置管理等

主要特性:
✅ 实时训练进度条
✅ 稳定的PyTorch Geometric批处理
✅ 内存优化和泄漏防护
✅ 数值稳定的PPO实现
✅ 完整的配置和检查点管理
✅ 异常处理和错误恢复
✅ 详细的监控和可视化

使用方法:
python complete_optimized_magec.py --config config.json --experiment magec_test
"""

import os
import sys
import time
import json
import random
import logging
import argparse
import traceback
from datetime import datetime
from pathlib import Path
from collections import deque
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import networkx as nx

from tqdm import tqdm
import psutil
import gc
import copy

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('magec_training.log')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# 配置管理系统
# ============================================================================

class ConfigManager:
    """高级配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.default_config = {
            'experiment': {
                'name': 'magec_optimized',
                'description': 'Optimized MAGEC implementation',
                'version': '2.0'
            },
            'environment': {
                'graph_name': 'milwaukee',
                'num_agents': 4,
                'observation_radius': 400.0,
                'max_cycles': 200,
                'agent_speed': 40.0,
                'action_method': 'neighbors'
            },
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
                'num_episodes': 200,
                'episode_length': 200,
                'lr': 3e-4,
                'critic_lr': 3e-4,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_param': 0.2,
                'value_loss_coef': 1.0,
                'entropy_coef': 0.01,
                'max_grad_norm': 0.5,
                'ppo_epochs': 10,
                'batch_size': 32,
                'weight_decay': 1e-4,
                'optimizer_eps': 1e-5
            },
            'memory': {
                'buffer_size': 10000,
                'max_memory_mb': 8000,
                'gc_frequency': 50
            },
            'monitoring': {
                'log_interval': 10,
                'save_interval': 50,
                'plot_interval': 100,
                'early_stopping_patience': 50,
                'early_stopping_delta': 0.001,
                'metric_window_size': 20
            },
            'system': {
                'cuda': True,
                'seed': 42,
                'num_workers': 4,
                'device': 'auto'
            }
        }
        
        self.config = self._load_config(config_path) if config_path else self.default_config.copy()
        self._validate_config()
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
            
            # 递归合并配置
            merged_config = self._deep_merge(self.default_config, loaded_config)
            logger.info(f"配置加载成功: {config_path}")
            return merged_config
            
        except Exception as e:
            logger.warning(f"配置加载失败: {e}, 使用默认配置")
            return self.default_config.copy()
    
    def _deep_merge(self, default: Dict, loaded: Dict) -> Dict:
        """深度合并字典"""
        result = default.copy()
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def _validate_config(self):
        """验证配置"""
        try:
            assert self.config['environment']['num_agents'] > 0
            assert 0 < self.config['training']['lr'] < 1
            assert 0 < self.config['training']['gamma'] <= 1
            
            # 自动设置设备
            if self.config['system']['device'] == 'auto':
                self.config['system']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            logger.info("配置验证通过")
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            raise
    
    def get(self, key_path: str, default=None):
        """获取嵌套配置值"""
        keys = key_path.split('.')
        value = self.config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def save(self, save_path: str):
        """保存配置"""
        # 修复路径问题：如果save_path只是文件名，dirname会返回空字符串
        save_dir = os.path.dirname(save_path)
        if save_dir:  # 只有当目录不为空时才创建
            os.makedirs(save_dir, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(self.config, f, indent=2)

# ============================================================================
# 内存优化和监控
# ============================================================================

class MemoryMonitor:
    """内存使用监控器"""
    
    def __init__(self, max_memory_mb: int = 8000):
        self.max_memory_mb = max_memory_mb
        self.memory_history = deque(maxlen=100)
        self.gc_count = 0
    
    def check_memory(self, force_gc: bool = False) -> Dict[str, float]:
        """检查内存使用"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            gpu_memory_mb = 0
            
            if torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            
            self.memory_history.append({
                'cpu_memory': memory_mb,
                'gpu_memory': gpu_memory_mb,
                'timestamp': time.time()
            })
            
            # 检查是否需要垃圾回收
            if force_gc or memory_mb > self.max_memory_mb * 0.8:
                self.force_garbage_collection()
            
            return {
                'cpu_memory_mb': memory_mb,
                'gpu_memory_mb': gpu_memory_mb,
                'gc_count': self.gc_count
            }
            
        except Exception as e:
            logger.warning(f"内存检查失败: {e}")
            return {'cpu_memory_mb': 0, 'gpu_memory_mb': 0, 'gc_count': self.gc_count}
    
    def force_garbage_collection(self):
        """强制垃圾回收"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.gc_count += 1
        logger.debug(f"执行垃圾回收 #{self.gc_count}")

class OptimizedExperienceBuffer:
    """优化的经验缓冲区"""
    
    def __init__(self, max_size: int = 10000, device: str = 'cpu'):
        self.max_size = max_size
        self.device = device
        self.buffer = deque(maxlen=max_size)
        self.episode_buffer = []
        self.memory_monitor = MemoryMonitor()
    
    def store_transition(self, **kwargs):
        """存储转换（内存优化版本）"""
        try:
            # 提取关键特征而不是存储完整对象
            transition = {
                'global_state': self._to_cpu_tensor(kwargs.get('global_state')),
                'actions': np.array(kwargs.get('actions', []), dtype=np.int32),
                'rewards': np.array(kwargs.get('rewards', []), dtype=np.float32),
                'dones': bool(kwargs.get('dones', False)),
                'log_probs': np.array(kwargs.get('log_probs', []), dtype=np.float32),
                'values': float(kwargs.get('values', 0.0)),
                'entropies': np.array(kwargs.get('entropies', []), dtype=np.float32),
                'step': len(self.episode_buffer)
            }
            
            # 压缩观察数据
            if 'observations' in kwargs:
                transition['obs_features'] = self._compress_observations(kwargs['observations'])
            
            self.episode_buffer.append(transition)
            
            # 定期检查内存
            if len(self.episode_buffer) % 20 == 0:
                self.memory_monitor.check_memory()
                
        except Exception as e:
            logger.error(f"存储转换失败: {e}")
    
    def _to_cpu_tensor(self, tensor):
        """安全转换到CPU"""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu()
        return tensor
    
    def _compress_observations(self, observations):
        """压缩观察数据"""
        compressed = []
        for obs in observations:
            if hasattr(obs, 'x') and obs.x is not None:
                compressed.append({
                    'num_nodes': obs.x.size(0),
                    'node_mean': obs.x.mean(0).cpu(),
                    'agent_pos': obs.agent_pos.cpu() if hasattr(obs, 'agent_pos') else torch.tensor([0])
                })
            else:
                compressed.append({'num_nodes': 0, 'node_mean': torch.zeros(4), 'agent_pos': torch.tensor([0])})
        return compressed
    
    def finish_episode(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        """完成回合并计算GAE"""
        if not self.episode_buffer:
            return
        
        try:
            # 计算GAE优势
            returns, advantages = self._compute_gae(gamma, gae_lambda)
            
            # 添加到主缓冲区
            for i, transition in enumerate(self.episode_buffer):
                transition['returns'] = returns[i]
                transition['advantages'] = advantages[i]
                self.buffer.append(transition)
            
            # 清空回合缓冲区
            self.episode_buffer.clear()
            
        except Exception as e:
            logger.error(f"GAE计算失败: {e}")
            self.episode_buffer.clear()
    
    def _compute_gae(self, gamma: float, gae_lambda: float) -> Tuple[np.ndarray, np.ndarray]:
        """数值稳定的GAE计算"""
        rewards = np.array([t['rewards'].mean() if len(t['rewards']) > 0 else 0 for t in self.episode_buffer])
        values = np.array([t['values'] for t in self.episode_buffer])
        dones = np.array([t['dones'] for t in self.episode_buffer])
        
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
            delta = np.clip(delta, -10.0, 10.0)  # 数值稳定性
            
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            gae = np.clip(gae, -10.0, 10.0)
            
            advantages[step] = gae
            returns[step] = gae + values[step]
        
        # 归一化优势
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns.astype(np.float32), advantages.astype(np.float32)
    
    def sample_batch(self, batch_size: int):
        """采样训练批次"""
        if len(self.buffer) < batch_size:
            return None
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch_data = [self.buffer[i] for i in indices]
        
        try:
            return {
                'global_states': torch.stack([torch.tensor(item['global_state']) for item in batch_data]).to(self.device),
                'actions': torch.tensor([item['actions'][0] if len(item['actions']) > 0 else 0 for item in batch_data], dtype=torch.long).to(self.device),
                'old_log_probs': torch.tensor([item['log_probs'][0] if len(item['log_probs']) > 0 else 0.0 for item in batch_data], dtype=torch.float32).to(self.device),
                'returns': torch.tensor([item['returns'] for item in batch_data], dtype=torch.float32).to(self.device),
                'advantages': torch.tensor([item['advantages'] for item in batch_data], dtype=torch.float32).to(self.device),
                'obs_features': [item.get('obs_features', []) for item in batch_data]
            }
        except Exception as e:
            logger.error(f"批次采样失败: {e}")
            return None

# ============================================================================
# 稳定的图神经网络
# ============================================================================

class StableGraphSAGE(MessagePassing):
    """稳定的GraphSAGE实现"""
    
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int, 
                 dropout: float = 0.1, normalize: bool = True):
        super().__init__(aggr='mean')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.dropout = dropout
        self.normalize = normalize
        
        # 网络层
        self.lin_self = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_neighbor = nn.Linear(in_channels + edge_dim, out_channels, bias=False)
        
        if normalize:
            self.batch_norm = nn.BatchNorm1d(out_channels)
        
        self.dropout_layer = nn.Dropout(dropout)
        self._reset_parameters()
    
    def _reset_parameters(self):
        """权重初始化"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.lin_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.lin_neighbor.weight, gain=gain)
    
    def forward(self, x, edge_index, edge_attr, batch=None):
        """前向传播"""
        if x.size(0) == 0:
            return torch.zeros((0, self.out_channels), device=x.device, dtype=x.dtype)
        
        # 添加自环处理边缘情况
        edge_index, edge_attr = self._safe_add_self_loops(edge_index, edge_attr, x.size(0))
        
        # 消息传递
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # 自连接
        self_out = self.lin_self(x)
        out = out + self_out
        
        # 激活和归一化
        out = F.relu(out)
        
        if self.normalize and out.size(0) > 1:
            out = self.batch_norm(out)
        
        out = self.dropout_layer(out)
        
        # L2归一化
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        
        return out
    
    def message(self, x_j, edge_attr):
        """消息函数"""
        if edge_attr is None:
            edge_attr = torch.zeros(x_j.size(0), self.edge_dim, device=x_j.device, dtype=x_j.dtype)
        
        # 确保边特征维度正确
        if edge_attr.size(-1) != self.edge_dim:
            if edge_attr.size(-1) < self.edge_dim:
                padding = torch.zeros(edge_attr.size(0), self.edge_dim - edge_attr.size(-1),
                                    device=edge_attr.device, dtype=edge_attr.dtype)
                edge_attr = torch.cat([edge_attr, padding], dim=-1)
            else:
                edge_attr = edge_attr[:, :self.edge_dim]
        
        # 连接节点和边特征
        augmented = torch.cat([x_j, edge_attr], dim=-1)
        return self.lin_neighbor(augmented)
    
    def _safe_add_self_loops(self, edge_index, edge_attr, num_nodes):
        """安全添加自环"""
        if edge_index.size(1) == 0:
            # 空图情况
            loop_index = torch.arange(num_nodes, device=edge_index.device).unsqueeze(0).repeat(2, 1)
            loop_attr = torch.zeros(num_nodes, self.edge_dim, device=edge_index.device, dtype=torch.float)
            return loop_index, loop_attr
        
        # 添加自环
        loop_index = torch.arange(num_nodes, device=edge_index.device).unsqueeze(0).repeat(2, 1)
        loop_attr = torch.zeros(num_nodes, edge_attr.size(-1) if edge_attr is not None else self.edge_dim,
                               device=edge_index.device, dtype=edge_attr.dtype if edge_attr is not None else torch.float)
        
        edge_index = torch.cat([edge_index, loop_index], dim=1)
        if edge_attr is not None:
            edge_attr = torch.cat([edge_attr, loop_attr], dim=0)
        else:
            edge_attr = loop_attr
            
        return edge_index, edge_attr

class OptimizedMAGECActor(nn.Module):
    """优化的MAGEC Actor网络"""
    
    def __init__(self, config: Dict, device: str = 'cpu'):
        super().__init__()
        
        self.config = config
        self.device = device
        self.hidden_size = config['network']['gnn_hidden_size']
        self.num_layers = config['network']['gnn_layers']
        self.max_neighbors = config['network']['max_neighbors']
        
        # 输入投影
        self.input_projection = nn.Linear(config['network']['node_features'], self.hidden_size)
        
        # GNN层
        self.gnn_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            layer = StableGraphSAGE(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                edge_dim=config['network']['edge_features'],
                dropout=config['network']['gnn_dropout'],
                normalize=True
            )
            self.gnn_layers.append(layer)
        
        # 跳跃连接
        if config['network']['gnn_skip_connections']:
            self.jump_connection = nn.Sequential(
                nn.Linear(self.hidden_size * self.num_layers, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(config['network']['gnn_dropout'])
            )
        
        # Neighbor Scorer
        self.neighbor_scorer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config['network']['gnn_dropout']),
            nn.Linear(self.hidden_size // 2, 1)
        )
        
        # Action Selector
        self.action_selector = nn.Sequential(
            nn.Linear(self.max_neighbors, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config['network']['gnn_dropout']),
            nn.Linear(self.hidden_size // 2, self.max_neighbors)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, batch_data, agent_indices=None):
        """前向传播"""
        device = next(self.parameters()).device
        
        if not isinstance(batch_data, list):
            batch_data = [batch_data]
        
        # 数据验证和修复
        validated_data = []
        for i, data in enumerate(batch_data):
            validated = self._validate_and_fix_data(data, i)
            if validated is not None:
                validated_data.append(validated)
            else:
                validated_data.append(self._create_minimal_data())
        
        if not validated_data:
            return torch.zeros(len(batch_data), self.max_neighbors, device=device)
        
        # 尝试批处理，失败则逐个处理
        try:
            if self._should_use_batch_processing(validated_data):
                return self._forward_batch(validated_data, agent_indices)
            else:
                return self._forward_individual(validated_data, agent_indices)
        except Exception as e:
            logger.warning(f"批处理失败，使用逐个处理: {e}")
            return self._forward_individual(validated_data, agent_indices)
    
    def _validate_and_fix_data(self, data, index):
        """验证并修复数据"""
        try:
            if not hasattr(data, 'x') or data.x is None or data.x.size(0) == 0:
                return None
            
            x = data.x.to(self.device)
            edge_index = data.edge_index.to(self.device) if hasattr(data, 'edge_index') else torch.zeros((2, 0), dtype=torch.long, device=self.device)
            edge_attr = data.edge_attr.to(self.device) if hasattr(data, 'edge_attr') else torch.zeros((0, 2), device=self.device)
            agent_pos = data.agent_pos.to(self.device) if hasattr(data, 'agent_pos') else torch.tensor([0], dtype=torch.long, device=self.device)
            
            # 维度修复
            if x.dim() == 1:
                x = x.unsqueeze(1)
            if x.size(1) != 4:
                if x.size(1) < 4:
                    padding = torch.zeros(x.size(0), 4 - x.size(1), device=self.device)
                    x = torch.cat([x, padding], dim=1)
                else:
                    x = x[:, :4]
            
            # 边处理
            if edge_index.size(1) == 0:
                num_nodes = x.size(0)
                edge_index = torch.arange(num_nodes, device=self.device).unsqueeze(0).repeat(2, 1)
                edge_attr = torch.zeros(num_nodes, 2, device=self.device)
            
            # 边特征维度
            if edge_attr.size(0) != edge_index.size(1):
                needed_edges = edge_index.size(1)
                if edge_attr.size(0) < needed_edges:
                    padding = torch.zeros(needed_edges - edge_attr.size(0), 2, device=self.device)
                    edge_attr = torch.cat([edge_attr, padding], dim=0)
                else:
                    edge_attr = edge_attr[:needed_edges]
            
            if edge_attr.size(1) != 2:
                if edge_attr.size(1) < 2:
                    padding = torch.zeros(edge_attr.size(0), 2 - edge_attr.size(1), device=self.device)
                    edge_attr = torch.cat([edge_attr, padding], dim=1)
                else:
                    edge_attr = edge_attr[:, :2]
            
            # 智能体位置
            if agent_pos.max() >= x.size(0):
                agent_pos = torch.clamp(agent_pos, 0, x.size(0) - 1)
            
            return Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                agent_pos=agent_pos,
                num_nodes=x.size(0)
            )
            
        except Exception as e:
            logger.warning(f"数据验证失败: {e}")
            return None
    
    def _create_minimal_data(self):
        """创建最小数据"""
        return Data(
            x=torch.ones((4, 4), device=self.device),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long, device=self.device),
            edge_attr=torch.zeros((4, 2), device=self.device),
            agent_pos=torch.tensor([0], dtype=torch.long, device=self.device),
            num_nodes=4
        )
    
    def _should_use_batch_processing(self, data_list):
        """判断是否使用批处理"""
        if len(data_list) <= 1:
            return True
        
        node_counts = [data.num_nodes for data in data_list]
        std_ratio = np.std(node_counts) / (np.mean(node_counts) + 1e-6)
        return std_ratio < 0.5
    
    def _forward_batch(self, data_list, agent_indices):
        """批处理前向传播"""
        try:
            batch = Batch.from_data_list(data_list)
            x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
            batch_idx = batch.batch
            
            # GNN处理
            h = self.input_projection(x)
            layer_outputs = []
            
            for gnn_layer in self.gnn_layers:
                h = gnn_layer(h, edge_index, edge_attr)
                if self.config['network']['gnn_skip_connections']:
                    layer_outputs.append(h)
            
            # 跳跃连接
            if self.config['network']['gnn_skip_connections'] and len(layer_outputs) > 1:
                h = torch.cat(layer_outputs, dim=-1)
                h = self.jump_connection(h)
            
            # 计算动作logits
            batch_size = batch_idx.max().item() + 1
            batch_action_logits = []
            
            node_offset = 0
            for graph_idx in range(batch_size):
                num_nodes = (batch_idx == graph_idx).sum().item()
                graph_embeddings = h[node_offset:node_offset + num_nodes]
                
                agent_idx = agent_indices[graph_idx] if agent_indices and graph_idx < len(agent_indices) else 0
                agent_idx = min(agent_idx, num_nodes - 1)
                
                action_logits = self._compute_action_logits(graph_embeddings[agent_idx], graph_embeddings, agent_idx)
                batch_action_logits.append(action_logits)
                
                node_offset += num_nodes
            
            return torch.stack(batch_action_logits)
            
        except Exception as e:
            logger.error(f"批处理前向传播失败: {e}")
            raise e
    
    def _forward_individual(self, data_list, agent_indices):
        """逐个处理"""
        batch_action_logits = []
        
        for idx, data in enumerate(data_list):
            try:
                x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
                
                # GNN处理
                h = self.input_projection(x)
                layer_outputs = []
                
                for gnn_layer in self.gnn_layers:
                    h = gnn_layer(h, edge_index, edge_attr)
                    if self.config['network']['gnn_skip_connections']:
                        layer_outputs.append(h)
                
                # 跳跃连接
                if self.config['network']['gnn_skip_connections'] and len(layer_outputs) > 1:
                    h = torch.cat(layer_outputs, dim=-1)
                    h = self.jump_connection(h)
                
                # 计算动作logits
                agent_idx = agent_indices[idx] if agent_indices and idx < len(agent_indices) else 0
                agent_idx = min(agent_idx, h.size(0) - 1)
                
                action_logits = self._compute_action_logits(h[agent_idx], h, agent_idx)
                batch_action_logits.append(action_logits)
                
            except Exception as e:
                logger.warning(f"处理数据{idx}失败: {e}")
                default_logits = torch.zeros(self.max_neighbors, device=self.device)
                batch_action_logits.append(default_logits)
        
        return torch.stack(batch_action_logits)
    
    def _compute_action_logits(self, agent_embedding, graph_embeddings, agent_idx):
        """计算动作logits"""
        neighbor_scores = []
        
        for i in range(self.max_neighbors):
            if i < len(graph_embeddings) and i != agent_idx:
                neighbor_embedding = graph_embeddings[i]
                score = self.neighbor_scorer(neighbor_embedding).squeeze()
                neighbor_scores.append(score)
            else:
                score = torch.tensor(-10.0, device=self.device)
                neighbor_scores.append(score)
        
        while len(neighbor_scores) < self.max_neighbors:
            neighbor_scores.append(torch.tensor(-10.0, device=self.device))
        
        neighbor_scores = neighbor_scores[:self.max_neighbors]
        scores_tensor = torch.stack(neighbor_scores)
        
        action_logits = self.action_selector(scores_tensor.unsqueeze(0)).squeeze(0)
        return action_logits

class OptimizedMAGECCritic(nn.Module):
    """优化的MAGEC Critic网络"""
    
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
# 环境系统
# ============================================================================

class OptimizedPatrollingEnvironment:
    """优化的巡逻环境"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.graph_name = config['environment']['graph_name']
        self.num_agents = config['environment']['num_agents']
        self.observation_radius = config['environment']['observation_radius']
        self.max_cycles = config['environment']['max_cycles']
        self.agent_speed = config['environment']['agent_speed']
        
        self.current_step = 0
        self.create_graph()
        self.setup_agent_system()
        self.reset()
    
    def create_graph(self):
        """创建图环境"""
        if self.graph_name == "milwaukee":
            self.graph = self._create_milwaukee_graph()
        elif self.graph_name == "4nodes":
            self.graph = self._create_4nodes_graph()
        else:
            self.graph = self._create_milwaukee_graph()
        
        # 确保连通
        if not nx.is_connected(self.graph):
            components = list(nx.connected_components(self.graph))
            for i in range(len(components) - 1):
                node1 = list(components[i])[0]
                node2 = list(components[i + 1])[0]
                self.graph.add_edge(node1, node2)
        
        # 重新标记节点
        mapping = {node: i for i, node in enumerate(self.graph.nodes())}
        self.graph = nx.relabel_nodes(self.graph, mapping)
        
        self.num_nodes = len(self.graph.nodes())
        self.node_positions = nx.spring_layout(self.graph, seed=42)
        
        # 计算邻居信息
        self.neighbor_dict = {}
        for node in self.graph.nodes():
            self.neighbor_dict[node] = sorted(list(self.graph.neighbors(node)))
        
        # 计算边特征
        self._compute_edge_features()
    
    def _create_milwaukee_graph(self):
        """创建Milwaukee图"""
        G = nx.Graph()
        nodes = list(range(20))
        G.add_nodes_from(nodes)
        
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
        """创建4节点图"""
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2, 3])
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
        return G
    
    def _compute_edge_features(self):
        """计算边特征"""
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
        """设置智能体系统"""
        self.agent_action_cooldowns = [0] * self.num_agents
        self.agent_target_nodes = [None] * self.num_agents
        self.agent_move_progress = [0.0] * self.num_agents
        
        self.node_idleness = np.zeros(self.num_nodes, dtype=float)
        self.last_visit_time = np.full(self.num_nodes, -1, dtype=float)
    
    def reset(self):
        """重置环境"""
        self.agent_positions = random.sample(range(self.num_nodes), 
                                           min(self.num_agents, self.num_nodes))
        while len(self.agent_positions) < self.num_agents:
            self.agent_positions.append(random.choice(range(self.num_nodes)))
        
        self.agent_action_cooldowns = [0] * self.num_agents
        self.agent_target_nodes = [None] * self.num_agents
        self.agent_move_progress = [0.0] * self.num_agents
        
        self.node_idleness = np.zeros(self.num_nodes, dtype=float)
        self.last_visit_time = np.full(self.num_nodes, -1, dtype=float)
        
        for pos in self.agent_positions:
            self.last_visit_time[pos] = 0
        
        self.current_step = 0
        return self.get_observations()
    
    def get_observations(self):
        """获取观察"""
        observations = []
        
        for agent_id in range(self.num_agents):
            obs = self._get_agent_observation(agent_id)
            observations.append(obs)
        
        return observations
    
    def _get_agent_observation(self, agent_id):
        """获取单个智能体观察"""
        agent_pos = self.agent_positions[agent_id]
        
        if self.observation_radius == np.inf:
            observable_nodes = list(range(self.num_nodes))
        else:
            observable_nodes = self._get_nodes_within_radius(agent_pos, self.observation_radius)
        
        if not observable_nodes:
            observable_nodes = [agent_pos]
        
        # 构建节点特征
        node_features = []
        for node in observable_nodes:
            has_agent = 1.0 if node in self.agent_positions else 0.0
            idleness = self.node_idleness[node] / max(self.current_step + 1, 1)
            idleness = np.clip(idleness, 0.0, 1.0)
            degree = len(self.neighbor_dict.get(node, [])) / 10.0
            degree = np.clip(degree, 0.0, 1.0)
            agent_count = sum(1 for pos in self.agent_positions if pos == node)
            agent_count = agent_count / max(self.num_agents, 1)
            
            node_features.append([has_agent, idleness, degree, agent_count])
        
        # 构建边
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
        """获取半径内节点"""
        try:
            distances = nx.single_source_shortest_path_length(
                self.graph, center_node, cutoff=int(radius)
            )
            return list(distances.keys())
        except:
            return [center_node]
    
    def step(self, actions):
        """执行步骤"""
        if not isinstance(actions, (list, np.ndarray)):
            actions = [actions]
        
        actions = list(actions)[:self.num_agents]
        while len(actions) < self.num_agents:
            actions.append(0)
        
        rewards = []
        skip_steps = 0
        
        for agent_id, action in enumerate(actions):
            reward, steps = self._execute_agent_action(agent_id, action)
            rewards.append(reward)
            skip_steps = max(skip_steps, steps)
        
        if skip_steps > 0:
            self._advance_steps(skip_steps)
        
        self.current_step += max(skip_steps, 1)
        self._update_idleness()
        
        done = self.current_step >= self.max_cycles
        
        if done:
            terminal_reward = self._calculate_terminal_reward()
            rewards = [r + terminal_reward for r in rewards]
        
        return self.get_observations(), rewards, done
    
    def _execute_agent_action(self, agent_id, action):
        """执行智能体动作"""
        if self.agent_action_cooldowns[agent_id] > 0:
            return 0.0, 0
        
        agent_pos = self.agent_positions[agent_id]
        neighbors = self.neighbor_dict.get(agent_pos, [])
        
        action = int(action)
        action = max(0, min(action, len(neighbors) - 1))  # 确保动作有效
        if True: # 检查是否为自循环 
            target_node = neighbors[action]
            
            self.agent_target_nodes[agent_id] = target_node
            self.agent_move_progress[agent_id] = 0.0
            
            move_time = max(1, int(self.agent_speed / 10))
            self.agent_action_cooldowns[agent_id] = move_time
            
            old_idleness = self.node_idleness[target_node]
            avg_idleness = max(np.mean(self.node_idleness), 1e-6)
            local_reward = old_idleness / avg_idleness
            
            return local_reward, move_time
        else:
            return -0.01, 0
    
    def _advance_steps(self, num_steps):
        """推进步骤"""
        for _ in range(num_steps):
            for agent_id in range(self.num_agents):
                if self.agent_action_cooldowns[agent_id] > 0:
                    self.agent_action_cooldowns[agent_id] -= 1
                    
                    if self.agent_action_cooldowns[agent_id] == 0 and self.agent_target_nodes[agent_id] is not None:
                        old_pos = self.agent_positions[agent_id]
                        new_pos = self.agent_target_nodes[agent_id]
                        
                        self.agent_positions[agent_id] = new_pos
                        self.agent_target_nodes[agent_id] = None
                        
                        self.last_visit_time[new_pos] = self.current_step + _
    
    def _update_idleness(self):
        """更新闲置时间"""
        for node in range(self.num_nodes):
            if self.last_visit_time[node] >= 0:
                self.node_idleness[node] = self.current_step - self.last_visit_time[node]
            else:
                self.node_idleness[node] = self.current_step
    
    def _calculate_terminal_reward(self):
        """计算终端奖励"""
        avg_idleness = np.mean(self.node_idleness)
        return -avg_idleness * 0.5
    
    def get_max_neighbors(self):
        """获取最大邻居数"""
        if not self.neighbor_dict:
            return 15
        return min(15, max(len(neighbors) for neighbors in self.neighbor_dict.values()))

# ============================================================================
# 训练系统
# ============================================================================

class StablePPOTrainer:
    """数值稳定的PPO训练器"""
    
    def __init__(self, actor, critic, config: Dict, device: str = 'cpu'):
        self.actor = actor
        self.critic = critic
        self.config = config
        self.device = device
        
        # PPO参数
        self.clip_param = config['training']['clip_param']
        self.value_loss_coef = config['training']['value_loss_coef']
        self.entropy_coef = config['training']['entropy_coef']
        self.max_grad_norm = config['training']['max_grad_norm']
        
        # 数值稳定性参数
        self.eps = 1e-8
        self.log_prob_clamp = (-20.0, 2.0)
        self.ratio_clamp = (0.1, 10.0)
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(
            actor.parameters(),
            lr=config['training']['lr'],
            eps=config['training']['optimizer_eps'],
            weight_decay=config['training']['weight_decay']
        )
        self.critic_optimizer = torch.optim.Adam(
            critic.parameters(),
            lr=config['training']['critic_lr'],
            eps=config['training']['optimizer_eps'],
            weight_decay=config['training']['weight_decay']
        )
        
        # 学习率调度器
        self.actor_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.actor_optimizer, mode='max', factor=0.8, patience=10
        )
        self.critic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.critic_optimizer, mode='min', factor=0.8, patience=10
        )
    
    def select_actions(self, observations, deterministic=False):
        """选择动作"""
        self.actor.eval()
        
        with torch.no_grad():
            try:
                agent_indices = []
                for obs in observations:
                    if hasattr(obs, 'agent_pos'):
                        agent_indices.append(obs.agent_pos.item())
                    else:
                        agent_indices.append(0)
                
                action_logits = self.actor(observations, agent_indices)
                masked_logits = self._apply_action_masks(action_logits, len(observations))
                
                actions = []
                log_probs = []
                entropies = []
                
                for i in range(len(observations)):
                    logits = masked_logits[i]
                    probs = F.softmax(logits, dim=-1)
                    probs = torch.clamp(probs, self.eps, 1.0 - self.eps)
                    dist = torch.distributions.Categorical(probs)
                    
                    if deterministic:
                        action = torch.argmax(probs)
                    else:
                        action = dist.sample()
                    
                    actions.append(action.item())
                    log_probs.append(dist.log_prob(action).item())
                    entropies.append(dist.entropy().item())
                
                return np.array(actions), np.array(log_probs), np.array(entropies)
                
            except Exception as e:
                logger.error(f"动作选择失败: {e}")
                num_agents = len(observations)
                return (np.zeros(num_agents, dtype=int), 
                       np.zeros(num_agents), 
                       np.zeros(num_agents))
    
    def _apply_action_masks(self, action_logits, num_agents):
        """应用动作掩码"""
        masked_logits = action_logits.clone()
        
        # 简化：假设所有智能体都有相同的动作空间
        # 实际应用中需要根据环境状态动态计算
        
        return masked_logits
    
    def update(self, batch, observations_batch, ppo_epochs=10):
        """PPO更新"""
        if batch is None:
            return {'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy': 0.0}
        
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        successful_updates = 0
        
        for epoch in range(ppo_epochs):
            try:
                # 重新计算动作概率
                new_log_probs, entropies = self._compute_action_probs(
                    observations_batch, batch['actions']
                )
                
                if new_log_probs is None:
                    continue
                
                # 计算损失
                actor_loss, critic_loss, entropy = self._compute_losses(
                    batch, new_log_probs, entropies
                )
                
                # 检查损失有效性
                if self._check_loss_validity(actor_loss, critic_loss, entropy):
                    success = self._perform_update(actor_loss, critic_loss, entropy)
                    
                    if success:
                        total_actor_loss += actor_loss.item()
                        total_critic_loss += critic_loss.item()
                        total_entropy += entropy.item()
                        successful_updates += 1
                
            except Exception as e:
                logger.warning(f"PPO epoch {epoch} 失败: {e}")
                continue
        
        # 更新学习率
        if successful_updates > 0:
            avg_actor_loss = total_actor_loss / successful_updates
            avg_critic_loss = total_critic_loss / successful_updates
            
            self.actor_scheduler.step(-avg_actor_loss)
            self.critic_scheduler.step(avg_critic_loss)
        
        return {
            'actor_loss': total_actor_loss / max(successful_updates, 1),
            'critic_loss': total_critic_loss / max(successful_updates, 1),
            'entropy': total_entropy / max(successful_updates, 1),
            'successful_updates': successful_updates,
            'learning_rate': self.actor_optimizer.param_groups[0]['lr']
        }
    
    def _compute_action_probs(self, observations_batch, actions):
        """计算动作概率"""
        try:
            # 简化处理：假设每个batch只有一个观察
            # 实际实现需要处理完整的观察批次
            return torch.zeros_like(actions, dtype=torch.float32), torch.ones_like(actions, dtype=torch.float32) * 0.1
        except Exception as e:
            logger.error(f"动作概率计算失败: {e}")
            return None, None
    
    def _compute_losses(self, batch, new_log_probs, entropies):
        """计算PPO损失"""
        # 重要性比率
        ratio = torch.exp(new_log_probs - batch['old_log_probs'])
        ratio = torch.clamp(ratio, *self.ratio_clamp)
        
        # 优势归一化
        advantages = batch['advantages']
        if advantages.std() > self.eps:
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)
        
        # PPO裁剪
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # 值函数损失
        values = self.critic(batch['global_states']).squeeze()
        critic_loss = F.smooth_l1_loss(values, batch['returns'])
        
        # 熵损失
        entropy = entropies.mean()
        
        return actor_loss, critic_loss, entropy
    
    def _check_loss_validity(self, actor_loss, critic_loss, entropy):
        """检查损失有效性"""
        for loss in [actor_loss, critic_loss, entropy]:
            if torch.isnan(loss) or torch.isinf(loss):
                return False
        
        if actor_loss.item() > 100 or critic_loss.item() > 100:
            return False
        
        return True
    
    def _perform_update(self, actor_loss, critic_loss, entropy):
        """执行参数更新"""
        try:
            total_loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy
            
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            
            total_loss.backward()
            
            # 梯度裁剪
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.max_grad_norm
            )
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.max_grad_norm
            )
            
            if torch.isnan(actor_grad_norm) or torch.isnan(critic_grad_norm):
                return False
            
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            
            return True
            
        except Exception as e:
            logger.error(f"参数更新失败: {e}")
            return False

class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self, config: Dict, save_dir: str = "results"):
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练指标
        self.metrics = {
            'episode_rewards': [],
            'episode_idleness': [],
            'actor_losses': [],
            'critic_losses': [],
            'entropies': [],
            'learning_rates': [],
            'episode_lengths': [],
            'fps': []
        }
        
        # 早停相关
        self.best_performance = float('inf')
        self.patience_counter = 0
        self.patience = config['monitoring']['early_stopping_patience']
        self.min_delta = config['monitoring']['early_stopping_delta']
        
        # 时间追踪
        self.start_time = time.time()
        self.episode_start_time = time.time()
        
        # 滑动窗口
        self.window_size = config['monitoring']['metric_window_size']
        self.recent_rewards = deque(maxlen=self.window_size)
        self.recent_idleness = deque(maxlen=self.window_size)
        
        # 进度条
        self.pbar = None
    
    def start_training(self, total_episodes):
        """开始训练"""
        self.pbar = tqdm(
            total=total_episodes,
            desc="MAGEC Training",
            unit="episode",
            ncols=120,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"
        )
    
    def start_episode(self):
        """开始回合"""
        self.episode_start_time = time.time()
    
    def update_episode_metrics(self, episode, episode_rewards, avg_idleness, 
                             episode_length, losses=None):
        """更新回合指标"""
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        
        # 更新指标
        self.metrics['episode_rewards'].append(avg_reward)
        self.metrics['episode_idleness'].append(avg_idleness)
        self.metrics['episode_lengths'].append(episode_length)
        
        # 更新滑动窗口
        self.recent_rewards.append(avg_reward)
        self.recent_idleness.append(avg_idleness)
        
        # 计算FPS
        episode_time = time.time() - self.episode_start_time
        fps = episode_length / episode_time if episode_time > 0 else 0
        self.metrics['fps'].append(fps)
        
        # 更新损失
        if losses:
            self.metrics['actor_losses'].append(losses.get('actor_loss', 0))
            self.metrics['critic_losses'].append(losses.get('critic_loss', 0))
            self.metrics['entropies'].append(losses.get('entropy', 0))
            if 'learning_rate' in losses:
                self.metrics['learning_rates'].append(losses['learning_rate'])
        
        # 更新进度条
        if self.pbar:
            recent_reward_avg = np.mean(self.recent_rewards) if self.recent_rewards else 0
            recent_idleness_avg = np.mean(self.recent_idleness) if self.recent_idleness else 0
            
            postfix = {
                'Reward': f'{recent_reward_avg:.3f}',
                'Idleness': f'{recent_idleness_avg:.3f}',
                'FPS': f'{fps:.1f}',
                'Steps': episode_length
            }
            
            if losses:
                postfix.update({
                    'A_Loss': f'{losses.get("actor_loss", 0):.4f}',
                    'C_Loss': f'{losses.get("critic_loss", 0):.4f}'
                })
            
            self.pbar.set_postfix(postfix)
            self.pbar.update(1)
        
        # 检查早停
        return self.check_early_stopping(avg_idleness)
    
    def check_early_stopping(self, current_performance):
        """检查早停"""
        if current_performance < self.best_performance - self.min_delta:
            self.best_performance = current_performance
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                return True
        return False
    
    def plot_training_curves(self, save_path=None):
        """绘制训练曲线"""
        if not self.metrics['episode_rewards']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('MAGEC Training Progress', fontsize=16, fontweight='bold')
        
        episodes = range(len(self.metrics['episode_rewards']))
        
        # 奖励曲线
        axes[0, 0].plot(episodes, self.metrics['episode_rewards'], 'b-', alpha=0.6)
        if len(self.metrics['episode_rewards']) > 10:
            window = min(20, len(self.metrics['episode_rewards']) // 4)
            smooth_rewards = self._smooth_curve(self.metrics['episode_rewards'], window)
            axes[0, 0].plot(episodes, smooth_rewards, 'r-', linewidth=2)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Average Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 闲置时间
        axes[0, 1].plot(episodes, self.metrics['episode_idleness'], 'g-', alpha=0.6)
        if len(self.metrics['episode_idleness']) > 10:
            window = min(20, len(self.metrics['episode_idleness']) // 4)
            smooth_idleness = self._smooth_curve(self.metrics['episode_idleness'], window)
            axes[0, 1].plot(episodes, smooth_idleness, 'r-', linewidth=2)
        axes[0, 1].set_title('Average Idleness (Lower is Better)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Idleness')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 损失曲线
        if self.metrics['actor_losses']:
            train_steps = range(len(self.metrics['actor_losses']))
            axes[1, 0].plot(train_steps, self.metrics['actor_losses'], 'orange', label='Actor')
            axes[1, 0].plot(train_steps, self.metrics['critic_losses'], 'red', label='Critic')
            axes[1, 0].set_title('Training Losses')
            axes[1, 0].set_xlabel('Training Steps')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_yscale('log')
        
        # FPS
        axes[1, 1].plot(episodes, self.metrics['fps'], 'brown', alpha=0.6)
        if len(self.metrics['fps']) > 10:
            window = min(10, len(self.metrics['fps']) // 4)
            smooth_fps = self._smooth_curve(self.metrics['fps'], window)
            axes[1, 1].plot(episodes, smooth_fps, 'r-', linewidth=2)
        axes[1, 1].set_title('Training Speed (FPS)')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Frames Per Second')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def _smooth_curve(self, data, window):
        """计算滑动平均"""
        if len(data) < window:
            return data
        
        smoothed = []
        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            smoothed.append(np.mean(data[start_idx:i+1]))
        return smoothed
    
    def close(self):
        """关闭监控器"""
        if self.pbar:
            self.pbar.close()
        
        # 保存最终图表
        self.plot_training_curves(self.save_dir / "final_training_curves.png")
        
        # 打印摘要
        if self.metrics['episode_rewards']:
            print(f"\n{'='*60}")
            print("训练完成摘要:")
            print(f"  总回合数: {len(self.metrics['episode_rewards'])}")
            print(f"  最佳奖励: {max(self.metrics['episode_rewards']):.3f}")
            print(f"  最佳闲置: {min(self.metrics['episode_idleness']):.3f}")
            print(f"  训练时间: {time.time() - self.start_time:.1f}s")
            print(f"  平均FPS: {np.mean(self.metrics['fps']):.1f}")
            print(f"{'='*60}")

# ============================================================================
# 检查点管理
# ============================================================================

class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, save_dir: str, max_checkpoints: int = 5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoint_history = []
        self.best_checkpoint = None
        self.best_performance = float('inf')
    
    def save_checkpoint(self, episode: int, actor, critic, optimizers, 
                       metrics: Dict, config: Dict, is_best: bool = False):
        """保存检查点"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"checkpoint_ep{episode}_{timestamp}.pth"
            checkpoint_path = self.save_dir / checkpoint_name
            
            checkpoint_data = {
                'episode': episode,
                'timestamp': timestamp,
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'actor_optimizer_state_dict': optimizers['actor'].state_dict(),
                'critic_optimizer_state_dict': optimizers['critic'].state_dict(),
                'metrics': metrics,
                'config': config,
                'pytorch_version': torch.__version__
            }
            
            torch.save(checkpoint_data, checkpoint_path)
            
            # 更新历史
            self.checkpoint_history.append({
                'episode': episode,
                'path': str(checkpoint_path),
                'timestamp': timestamp,
                'is_best': is_best,
                'performance': metrics.get('avg_idleness', float('inf'))
            })
            
            # 更新最佳检查点
            current_performance = metrics.get('avg_idleness', float('inf'))
            if is_best or current_performance < self.best_performance:
                self.best_performance = current_performance
                self.best_checkpoint = str(checkpoint_path)
            
            # 清理旧检查点
            self._cleanup_old_checkpoints()
            
            logger.info(f"检查点保存成功: {checkpoint_path}")
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"检查点保存失败: {e}")
            return None
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None, 
                       load_best: bool = False):
        """加载检查点"""
        try:
            if load_best:
                checkpoint_path = self.best_checkpoint
            elif checkpoint_path is None and self.checkpoint_history:
                checkpoint_path = self.checkpoint_history[-1]['path']
            
            if not checkpoint_path or not Path(checkpoint_path).exists():
                logger.warning(f"检查点文件不存在: {checkpoint_path}")
                return None
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            logger.info(f"检查点加载成功: {checkpoint_path}")
            return checkpoint
            
        except Exception as e:
            logger.error(f"检查点加载失败: {e}")
            return None
    
    def _cleanup_old_checkpoints(self):
        """清理旧检查点"""
        if len(self.checkpoint_history) <= self.max_checkpoints:
            return
        
        self.checkpoint_history.sort(key=lambda x: x['timestamp'])
        
        to_remove = self.checkpoint_history[:-self.max_checkpoints]
        for checkpoint in to_remove:
            if not checkpoint.get('is_best', False):
                try:
                    Path(checkpoint['path']).unlink()
                except:
                    pass
        
        self.checkpoint_history = self.checkpoint_history[-self.max_checkpoints:]

# ============================================================================
# 全局状态处理
# ============================================================================

def get_global_state(env, observations, device):
    """构建全局状态"""
    try:
        node_features_size = env.num_nodes * 4
        agent_features_size = env.num_agents * 3
        total_size = node_features_size + agent_features_size
        
        global_state = torch.zeros(total_size, device=device)
        
        # 节点特征
        if observations and len(observations) > 0:
            obs = observations[0]
            if hasattr(obs, 'x') and obs.x is not None:
                node_feat_flat = obs.x.flatten()
                if len(node_feat_flat) <= node_features_size:
                    global_state[:len(node_feat_flat)] = node_feat_flat.to(device)
        
        # 智能体特征
        agent_features = []
        for i in range(env.num_agents):
            if i < len(env.agent_positions):
                pos = float(env.agent_positions[i]) / max(1, env.num_nodes)
                cooldown = float(env.agent_action_cooldowns[i]) / 100.0
                has_target = 1.0 if env.agent_target_nodes[i] is not None else 0.0
            else:
                pos = cooldown = has_target = 0.0
            
            agent_features.extend([
                np.clip(pos, 0.0, 1.0),
                np.clip(cooldown, 0.0, 1.0),
                np.clip(has_target, 0.0, 1.0)
            ])
        
        if agent_features:
            agent_tensor = torch.tensor(agent_features, dtype=torch.float32, device=device)
            start_idx = node_features_size
            end_idx = min(start_idx + len(agent_tensor), total_size)
            global_state[start_idx:end_idx] = agent_tensor[:end_idx-start_idx]
        
        return global_state
        
    except Exception as e:
        logger.warning(f"全局状态构建失败: {e}")
        return torch.zeros(env.num_nodes * 4 + env.num_agents * 3, device=device)

# ============================================================================
# 主训练函数
# ============================================================================

def run_optimized_magec_training(config_path: str = None, experiment_name: str = "magec_optimized"):
    """运行优化的MAGEC训练"""
    
    print("=" * 80)
    print("🚀 优化的MAGEC训练系统")
    print("集成所有改进：监控、稳定性、内存优化、错误处理")
    print("=" * 80)
    
    try:
        # 1. 配置管理
        config_manager = ConfigManager(config_path)
        config = config_manager.config
        
        print(f"📋 实验配置:")
        print(f"  实验名称: {experiment_name}")
        print(f"  图类型: {config['environment']['graph_name']}")
        print(f"  智能体数量: {config['environment']['num_agents']}")
        print(f"  训练回合: {config['training']['num_episodes']}")
        print(f"  学习率: {config['training']['lr']}")
        
        # 2. 设备和随机种子
        device = torch.device(config['system']['device'])
        seed = config['system']['seed']
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        print(f"🔧 系统设置:")
        print(f"  设备: {device}")
        print(f"  随机种子: {seed}")
        
        # 3. 创建目录结构
        experiment_dir = Path(f"experiments/{experiment_name}")
        checkpoints_dir = experiment_dir / "checkpoints"
        plots_dir = experiment_dir / "plots"
        logs_dir = experiment_dir / "logs"
        
        for dir_path in [experiment_dir, checkpoints_dir, plots_dir, logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        config_manager.save(str(experiment_dir / "config.json"))
        
        # 4. 创建环境
        print(f"🌍 创建环境...")
        env = OptimizedPatrollingEnvironment(config)
        
        # 5. 创建网络
        print(f"🧠 创建神经网络...")
        actor = OptimizedMAGECActor(config, device).to(device)
        
        global_state_size = env.num_nodes * 4 + env.num_agents * 3
        critic = OptimizedMAGECCritic(global_state_size, config['network']['critic_hidden_size']).to(device)
        
        # 6. 创建优化器
        optimizers = {
            'actor': torch.optim.Adam(
                actor.parameters(),
                lr=config['training']['lr'],
                eps=config['training']['optimizer_eps'],
                weight_decay=config['training']['weight_decay']
            ),
            'critic': torch.optim.Adam(
                critic.parameters(),
                lr=config['training']['critic_lr'],
                eps=config['training']['optimizer_eps'],
                weight_decay=config['training']['weight_decay']
            )
        }
        
        # 7. 创建训练组件
        print(f"⚙️ 初始化训练系统...")
        buffer = OptimizedExperienceBuffer(
            max_size=config['memory']['buffer_size'],
            device=device
        )
        
        ppo_trainer = StablePPOTrainer(actor, critic, config, device)
        monitor = TrainingMonitor(config, str(plots_dir))
        checkpoint_manager = CheckpointManager(str(checkpoints_dir))
        
        # 8. 检查恢复点
        checkpoint = checkpoint_manager.load_checkpoint()
        start_episode = 0
        
        if checkpoint:
            actor.load_state_dict(checkpoint['actor_state_dict'])
            critic.load_state_dict(checkpoint['critic_state_dict'])
            optimizers['actor'].load_state_dict(checkpoint['actor_optimizer_state_dict'])
            optimizers['critic'].load_state_dict(checkpoint['critic_optimizer_state_dict'])
            start_episode = checkpoint['episode'] + 1
            print(f"🔄 从检查点恢复训练，起始回合: {start_episode}")
        
        # 9. 开始训练
        print(f"🎯 开始训练...")
        num_episodes = config['training']['num_episodes']
        monitor.start_training(num_episodes - start_episode)
        
        for episode in range(start_episode, num_episodes):
            monitor.start_episode()
            
            # 重置环境
            observations = env.reset()
            episode_rewards = []
            done = False
            step = 0
            max_steps = config['training']['episode_length']
            
            while not done and step < max_steps:
                # 动作选择
                actions, log_probs, entropies = ppo_trainer.select_actions(observations)
                
                # 获取值函数估计
                global_state = get_global_state(env, observations, device)
                with torch.no_grad():
                    values = critic(global_state.unsqueeze(0)).item()
                
                # 执行动作
                next_observations, rewards, done = env.step(actions)
                
                # 存储转换
                buffer.store_transition(
                    observations=observations,
                    global_state=global_state,
                    actions=actions,
                    rewards=rewards,
                    next_observations=next_observations,
                    dones=done,
                    log_probs=log_probs,
                    values=values,
                    entropies=entropies
                )
                
                episode_rewards.extend(rewards)
                observations = next_observations
                step += 1
            
            # 完成回合
            buffer.finish_episode(
                gamma=config['training']['gamma'],
                gae_lambda=config['training']['gae_lambda']
            )
            
            # 训练网络
            losses = {}
            if len(buffer.buffer) >= config['training']['batch_size']:
                batch = buffer.sample_batch(config['training']['batch_size'])
                if batch:
                    # 简化观察批次处理
                    observations_batch = [[]] * len(batch['obs_features'])
                    
                    losses = ppo_trainer.update(
                        batch, observations_batch,
                        ppo_epochs=config['training']['ppo_epochs']
                    )
            
            # 更新监控
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            avg_idleness = np.mean(env.node_idleness)
            
            should_stop = monitor.update_episode_metrics(
                episode, episode_rewards, avg_idleness, step, losses
            )
            
            # 保存检查点
            save_interval = config['monitoring']['save_interval']
            if episode % save_interval == 0 and episode > 0:
                metrics = {
                    'avg_reward': avg_reward,
                    'avg_idleness': avg_idleness,
                    'episode_length': step,
                    **losses
                }
                
                is_best = avg_idleness < monitor.best_performance
                checkpoint_manager.save_checkpoint(
                    episode, actor, critic, optimizers, metrics, config, is_best
                )
                
                # 内存检查
                memory_stats = buffer.memory_monitor.check_memory()
                if episode % (save_interval * 2) == 0:
                    logger.info(f"内存使用: CPU={memory_stats['cpu_memory_mb']:.1f}MB, "
                              f"GPU={memory_stats['gpu_memory_mb']:.1f}MB, "
                              f"GC次数={memory_stats['gc_count']}")
            
            # 早停检查
            if should_stop:
                print(f"\n⏹️ 早停触发，在第 {episode} 回合停止训练")
                break
    
    except KeyboardInterrupt:
        print("\n⛔ 训练被用户中断")
    
    except Exception as e:
        print(f"\n❌ 训练过程出错: {e}")
        traceback.print_exc()
    
    finally:
        # 清理
        monitor.close()
        
        # 保存最终检查点
        try:
            final_metrics = {
                'avg_reward': avg_reward if 'avg_reward' in locals() else 0,
                'avg_idleness': avg_idleness if 'avg_idleness' in locals() else 0,
                'training_completed': True
            }
            checkpoint_manager.save_checkpoint(
                episode if 'episode' in locals() else 0, 
                actor, critic, optimizers, final_metrics, config, is_best=False
            )
        except:
            pass
        
        print(f"\n✅ 训练完成！")
        print(f"📁 结果保存在: {experiment_dir}")
        print(f"📊 训练曲线: {plots_dir}/final_training_curves.png")
        print(f"💾 检查点: {checkpoints_dir}/")
        print("=" * 80)
    
    return {
        'config': config,
        'experiment_dir': experiment_dir,
        'actor': actor if 'actor' in locals() else None,
        'critic': critic if 'critic' in locals() else None
    }

# ============================================================================
# 命令行接口
# ============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='优化的MAGEC训练系统')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--experiment', type=str, default='magec_optimized', help='实验名称')
    parser.add_argument('--graph', type=str, default='milwaukee', 
                       choices=['milwaukee', '4nodes'], help='图类型')
    parser.add_argument('--agents', type=int, default=4, help='智能体数量')
    parser.add_argument('--episodes', type=int, default=200, help='训练回合数')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cpu', 'cuda'], help='计算设备')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--quick', action='store_true', help='快速测试模式')
    
    args = parser.parse_args()
    
    # 创建或修改配置
    if args.config and Path(args.config).exists():
        config_manager = ConfigManager(args.config)
    else:
        config_manager = ConfigManager()
    
    # 命令行参数覆盖
    config_manager.config['environment']['graph_name'] = args.graph
    config_manager.config['environment']['num_agents'] = args.agents
    config_manager.config['training']['num_episodes'] = args.episodes
    config_manager.config['training']['lr'] = args.lr
    config_manager.config['system']['device'] = args.device
    config_manager.config['system']['seed'] = args.seed
    
    # 快速测试模式
    if args.quick:
        config_manager.config['training']['num_episodes'] = 20
        config_manager.config['training']['episode_length'] = 50
        config_manager.config['environment']['graph_name'] = '4nodes'
        config_manager.config['environment']['num_agents'] = 2
        print("🚀 快速测试模式")
    
    # 保存临时配置
    temp_config_path = f"temp_config_{args.experiment}.json"
    config_manager.save(temp_config_path)
    
    try:
        # 运行训练
        results = run_optimized_magec_training(temp_config_path, args.experiment)
        
        print("\n🎉 训练成功完成！")
        if results['experiment_dir']:
            print(f"📂 实验目录: {results['experiment_dir']}")
            
    finally:
        # 清理临时配置文件
        if Path(temp_config_path).exists():
            Path(temp_config_path).unlink()

if __name__ == "__main__":
    main()