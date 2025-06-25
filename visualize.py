#!/usr/bin/env python3
"""
MAGEC评估和可视化工具 - 增强版：支持第一阶段拓扑测试
实现与论文基准算法的对比：AHPA, SEBS, CBLS
提供完整的性能评估和可视化，支持第一阶段生成的拓扑

基于论文: "Graph Neural Network-based Multi-agent Reinforcement Learning 
for Resilient Distributed Coordination of Multi-Robot Systems"

使用方法:
    1. 交互式模式 (推荐新手用户):
       python visualize.py
    
    2. 使用第一阶段拓扑:
       python visualize.py --topology path/to/topology.json --magec_model path/to/model.pth
    
    3. 命令行模式:
       python visualize.py --magec_model path/to/model.pth --output_dir results/
    
    4. 批处理模式:
       python visualize.py --batch --magec_model path/to/model.pth
    
    5. 快速测试:
       python visualize.py --quick_test --animate

特性:
    ✅ 自动发现训练好的模型
    ✅ 自动发现第一阶段拓扑文件
    ✅ 交互式配置向导
    ✅ 多种基准算法对比
    ✅ 改进的动画可视化巡逻过程
    ✅ 详细的性能报告
    ✅ 多种干扰场景测试
    ✅ 支持第一阶段生成的自定义拓扑
"""

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyBboxPatch
import networkx as nx
import torch
import argparse
import logging
import json
import random
import time
from collections import deque
from typing import Dict, List, Tuple, Optional
import copy
import inspect 
# 导入我们的MAGEC实现
try:
    from demo_MAGEC import (
        MAGECActor, MAGECCritic, MAGECTrainingEnvironment,
        MAGECTrainer, create_magec_config, TopologyToMAGECMapper
    )
    MAGEC_AVAILABLE = True
    print("✅ MAGEC模块导入成功")
except ImportError as e:
    print(f"❌ MAGEC模块导入失败: {e}")
    print("请确保demo_MAGEC.py在同一目录下")
    MAGEC_AVAILABLE = False

# 尝试导入原版demo作为备选
if not MAGEC_AVAILABLE:
    try:
        from demo import (
            MAGECActor, MAGECCritic, OptimizedPatrollingEnvironment,
            MAGECTrainer, create_official_config
        )
        DEMO_AVAILABLE = True
        print("✅ 原版demo模块导入成功（备选）")
    except ImportError as e:
        print(f"❌ 原版demo模块也不可用: {e}")
        DEMO_AVAILABLE = False
        print("请确保demo.py或demo_MAGEC.py在同一目录下")
        sys.exit(1)

logger = logging.getLogger(__name__)

# ============================================================================
# 可视化配置
# ============================================================================

VISUALIZATION_CONFIG = {
    'node_size': 0.04,           # 节点大小
    'agent_size': 250,           # 智能体标记大小
    'animation_interval': 400,    # 动画间隔(ms)
    'animation_fps': 4,          # 动画帧率
    'show_idleness_values': True, # 显示闲置时间数值
    'show_legend': True,         # 显示图例
    'show_grid': True,           # 显示网格
    'margin_ratio': 0.15,        # 边距比例
    'colors': {
        'low_idleness': 'lightgreen',
        'medium_idleness': 'gold', 
        'high_idleness': 'lightcoral',
        'agents': ['blue', 'red', 'purple', 'brown', 'orange'],
        'edges': 'gray',
        'edge_borders': ['darkgreen', 'orange', 'darkred']
    },
    'markers': ['o', 's', '^', 'D', 'v']  # 智能体形状
}

# ============================================================================
# 环境兼容性包装器
# ============================================================================

class EnvironmentCompatibilityWrapper:
    """环境兼容性包装器 - 统一不同环境的接口"""
    
    def __init__(self, env):
        self.env = env
    
    def __getattr__(self, name):
        """代理所有属性访问"""
        return getattr(self.env, name)
    
    @property
    def node_idleness_values(self):
        """获取节点闲置时间值列表"""
        if isinstance(self.env.node_idleness, dict):
            return list(self.env.node_idleness.values())
        else:
            return self.env.node_idleness
    
    @property 
    def mean_idleness(self):
        """获取平均闲置时间"""
        values = self.node_idleness_values
        return np.mean(values) if values else 0
    
    def get_node_idleness(self, node):
        """获取指定节点的闲置时间"""
        if isinstance(self.env.node_idleness, dict):
            return self.env.node_idleness.get(node, 0)
        else:
            return self.env.node_idleness[node]

def wrap_environment(env):
    """包装环境以提供兼容性"""
    return EnvironmentCompatibilityWrapper(env)

# ============================================================================
# 拓扑感知环境适配器
# ============================================================================

class TopologyAwareEnvironmentAdapter:
    """第一阶段拓扑感知环境适配器"""
    
    def __init__(self, topology_file=None, env_config=None):
        self.topology_file = topology_file
        self.env_config = env_config or {}
        self.mapper = None
        self.magec_env_config = None
        self.is_custom_topology = bool(topology_file)
        self.topology_data = None
        
        if self.is_custom_topology:
            self._load_topology()
    
    def _load_topology(self):
        """加载第一阶段拓扑"""
        if not MAGEC_AVAILABLE:
            raise ImportError("MAGEC模块不可用，无法加载自定义拓扑")
        
        try:
            print(f"📍 正在加载第一阶段拓扑: {self.topology_file}")
            self.mapper = TopologyToMAGECMapper()
            
            if not self.mapper.load_topology_from_json(self.topology_file):
                raise Exception("拓扑JSON加载失败")
            
            # 读取拓扑数据用于信息显示
            with open(self.topology_file, 'r', encoding='utf-8') as f:
                self.topology_data = json.load(f)
            
            # 创建MAGEC环境配置
            num_agents = self.env_config.get('num_agents', 4)
            self.magec_env_config = self.mapper.create_magec_environment(num_agents)
            
            print(f"✅ 拓扑加载成功:")
            print(f"   节点数: {self.magec_env_config['graph'].number_of_nodes()}")
            print(f"   边数: {self.magec_env_config['graph'].number_of_edges()}")
            print(f"   智能体数: {self.magec_env_config['num_agents']}")
            print(f"   最大邻居数: {self.magec_env_config['max_neighbors']}")
            
        except Exception as e:
            print(f"❌ 加载拓扑失败: {e}")
            raise
    
    def create_environment(self):
        """创建环境实例"""
        if self.is_custom_topology and MAGEC_AVAILABLE:
            # 使用第一阶段拓扑创建环境
            env = MAGECTrainingEnvironment(self.magec_env_config)
        else:
            # 使用标准环境
            if MAGEC_AVAILABLE:
                # 尝试使用MAGEC环境（但没有自定义拓扑）
                env = MAGECTrainingEnvironment({
                    'graph': self._create_standard_graph(),
                    'node_features': {},
                    'edge_features': {},
                    'position_mapping': {},
                    'special_points': {},
                    'num_agents': self.env_config.get('num_agents', 4),
                    'max_neighbors': 15
                })
            else:
                # 回退到原版demo环境
                env = OptimizedPatrollingEnvironment(**self.env_config)
        
        # 包装环境以提供兼容性
        return wrap_environment(env)
    
    def _create_standard_graph(self):
        """创建标准图结构（当没有自定义拓扑时）"""
        graph_name = self.env_config.get('graph_name', 'milwaukee')
        
        if graph_name == 'milwaukee':
            # Milwaukee图拓扑
            G = nx.Graph()
            edges = [
                (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5),
                (5, 6), (6, 7), (6, 8), (7, 9), (8, 10), (9, 11),
                (10, 11), (11, 12), (12, 13), (12, 14), (13, 15),
                (14, 16), (15, 17), (16, 18), (17, 19), (18, 19),
                (1, 4), (3, 6), (5, 8), (7, 10), (9, 12), (11, 14),
                (13, 16), (15, 18), (2, 7), (4, 9)
            ]
            G.add_edges_from(edges)
            
            # 添加节点位置
            pos = nx.spring_layout(G, seed=42, k=3, iterations=50)
            for node, position in pos.items():
                G.nodes[node]['pos'] = (position[0] * 100, position[1] * 100)
            
        elif graph_name == '4nodes':
            # 简单4节点图
            G = nx.cycle_graph(4)
            pos = {0: (0, 0), 1: (1, 0), 2: (1, 1), 3: (0, 1)}
            for node, position in pos.items():
                G.nodes[node]['pos'] = (position[0] * 50, position[1] * 50)
        
        else:
            # 默认小图
            G = nx.cycle_graph(6)
            pos = nx.circular_layout(G)
            for node, position in pos.items():
                G.nodes[node]['pos'] = (position[0] * 50, position[1] * 50)
        
        return G
    
    def get_topology_info(self):
        """获取拓扑信息"""
        if self.is_custom_topology and self.topology_data:
            return {
                'type': 'custom_stage1',
                'source_file': self.topology_file,
                'system': self.topology_data.get('system', 'Unknown'),
                'stage': self.topology_data.get('stage', 'Unknown'),
                'ready_for_stage2': self.topology_data.get('ready_for_stage2', False),
                'construction_stats': self.topology_data.get('construction_stats', {}),
                'nodes_count': self.magec_env_config['graph'].number_of_nodes() if self.magec_env_config else 0,
                'edges_count': self.magec_env_config['graph'].number_of_edges() if self.magec_env_config else 0
            }
        else:
            return {
                'type': 'standard',
                'graph_name': self.env_config.get('graph_name', 'milwaukee'),
                'nodes_count': 0,
                'edges_count': 0
            }

# ============================================================================
# 基准算法实现（保持原有）
# ============================================================================

class BaselinePatrollingAlgorithm:
    """基准巡逻算法基类"""
    
    def __init__(self, env, name="Baseline"):
        self.env = env
        self.name = name
        self.reset()
    
    def reset(self):
        """重置算法状态"""
        pass
    
    def select_actions(self, observations):
        """选择动作 - 子类实现"""
        raise NotImplementedError
    
    def update(self, observations, actions, rewards, next_observations, dones):
        """更新算法状态 - 子类可选实现"""
        pass

class AHPAAlgorithm(BaselinePatrollingAlgorithm):
    """AHPA (Adaptive Heuristic-based Patrolling Algorithm)"""
    
    def __init__(self, env):
        super().__init__(env, "AHPA")
        self.agent_plans = [deque() for _ in range(env.num_agents)]
        self.plan_horizons = [5] * env.num_agents
    
    def reset(self):
        """重置AHPA状态"""
        self.agent_plans = [deque() for _ in range(self.env.num_agents)]
    
    def select_actions(self, observations):
        """AHPA动作选择"""
        actions = []
        
        for agent_id in range(self.env.num_agents):
            # 检查是否需要重新规划
            if len(self.agent_plans[agent_id]) == 0:
                self._plan_for_agent(agent_id)
            
            # 执行计划中的下一个动作
            if len(self.agent_plans[agent_id]) > 0:
                target_node = self.agent_plans[agent_id].popleft()
                action = self._get_action_to_node(agent_id, target_node)
            else:
                action = 0
            
            actions.append(action)
        
        return np.array(actions)
    
    def _plan_for_agent(self, agent_id):
        """为智能体生成巡逻计划"""
        current_pos = self.env.agent_positions[agent_id]
        
        # 选择最高闲置时间的未访问节点
        unvisited_nodes = []
        for node in range(self.env.num_nodes):
            if node not in self.env.agent_positions:
                idleness = self.env.get_node_idleness(node)
                unvisited_nodes.append((idleness, node))
        
        # 按闲置时间排序
        unvisited_nodes.sort(reverse=True)
        
        # 选择前几个节点作为目标
        plan = []
        for _, node in unvisited_nodes[:self.plan_horizons[agent_id]]:
            plan.append(node)
        
        self.agent_plans[agent_id] = deque(plan)
    
    def _get_action_to_node(self, agent_id, target_node):
        """计算到达目标节点的动作"""
        current_pos = self.env.agent_positions[agent_id]
        neighbors = getattr(self.env, 'neighbor_dict', {}).get(current_pos, [])
        
        if target_node in neighbors:
            return neighbors.index(target_node)
        elif neighbors:
            # 使用最短路径导航
            try:
                if hasattr(self.env, 'graph'):
                    path = nx.shortest_path(self.env.graph, current_pos, target_node)
                    if len(path) > 1:
                        next_node = path[1]
                        if next_node in neighbors:
                            return neighbors.index(next_node)
            except:
                pass
            
            # 随机选择邻居
            return random.randint(0, len(neighbors) - 1)
        
        return 0

class SEBSAlgorithm(BaselinePatrollingAlgorithm):
    """SEBS (State Exchange Bayesian Strategy)"""
    
    def __init__(self, env):
        super().__init__(env, "SEBS")
        self.agent_beliefs = [{} for _ in range(env.num_agents)]
        self.communication_range = 2
    
    def reset(self):
        """重置SEBS状态"""
        self.agent_beliefs = [{} for _ in range(self.env.num_agents)]
    
    def select_actions(self, observations):
        """SEBS动作选择"""
        # 更新信念
        self._update_beliefs()
        
        # 智能体间信息交换
        self._exchange_information()
        
        actions = []
        for agent_id in range(self.env.num_agents):
            action = self._select_action_for_agent(agent_id)
            actions.append(action)
        
        return np.array(actions)
    
    def _update_beliefs(self):
        """更新智能体信念"""
        for agent_id in range(self.env.num_agents):
            agent_pos = self.env.agent_positions[agent_id]
            
            # 更新可观察节点的信念
            for node in range(self.env.num_nodes):
                if node == agent_pos:
                    self.agent_beliefs[agent_id][node] = {
                        'idleness': self.env.get_node_idleness(node),
                        'last_update': self.env.current_step,
                        'confidence': 1.0
                    }
    
    def _exchange_information(self):
        """智能体间信息交换"""
        for i in range(self.env.num_agents):
            for j in range(i + 1, self.env.num_agents):
                # 检查是否在通信范围内
                pos_i = self.env.agent_positions[i]
                pos_j = self.env.agent_positions[j]
                
                try:
                    if hasattr(self.env, 'graph'):
                        distance = nx.shortest_path_length(self.env.graph, pos_i, pos_j)
                        if distance <= self.communication_range:
                            # 交换信念
                            self._merge_beliefs(i, j)
                            self._merge_beliefs(j, i)
                except:
                    continue
    
    def _merge_beliefs(self, receiver_id, sender_id):
        """合并信念"""
        receiver_beliefs = self.agent_beliefs[receiver_id]
        sender_beliefs = self.agent_beliefs[sender_id]
        
        for node, belief in sender_beliefs.items():
            if node not in receiver_beliefs:
                receiver_beliefs[node] = belief.copy()
            else:
                if belief['last_update'] > receiver_beliefs[node]['last_update']:
                    receiver_beliefs[node] = belief.copy()
    
    def _select_action_for_agent(self, agent_id):
        """为智能体选择动作"""
        current_pos = self.env.agent_positions[agent_id]
        neighbors = getattr(self.env, 'neighbor_dict', {}).get(current_pos, [])
        
        if not neighbors:
            return 0
        
        # 基于信念选择最佳邻居
        best_action = 0
        best_utility = -float('inf')
        
        for i, neighbor in enumerate(neighbors):
            utility = self._calculate_utility(agent_id, neighbor)
            if utility > best_utility:
                best_utility = utility
                best_action = i
        
        return best_action
    
    def _calculate_utility(self, agent_id, node):
        """计算节点效用"""
        beliefs = self.agent_beliefs[agent_id]
        
        if node in beliefs:
            believed_idleness = beliefs[node]['idleness']
            confidence = beliefs[node]['confidence']
            return believed_idleness * confidence
        else:
            return self.env.get_node_idleness(node)

class CBLSAlgorithm(BaselinePatrollingAlgorithm):
    """CBLS (Concurrent Bayesian Learning Strategy)"""
    
    def __init__(self, env):
        super().__init__(env, "CBLS")
        self.q_values = np.zeros((env.num_agents, env.num_nodes, env.num_nodes))
        self.learning_rate = 0.1
        self.epsilon = 0.1
        self.decay_rate = 0.99
    
    def reset(self):
        """重置CBLS状态"""
        self.q_values = np.zeros((self.env.num_agents, self.env.num_nodes, self.env.num_nodes))
        self.epsilon = 0.1
    
    def select_actions(self, observations):
        """CBLS动作选择"""
        actions = []
        
        for agent_id in range(self.env.num_agents):
            action = self._select_action_for_agent(agent_id)
            actions.append(action)
        
        return np.array(actions)
    
    def _select_action_for_agent(self, agent_id):
        """epsilon-greedy动作选择"""
        current_pos = self.env.agent_positions[agent_id]
        neighbors = getattr(self.env, 'neighbor_dict', {}).get(current_pos, [])
        
        if not neighbors:
            return 0
        
        if random.random() < self.epsilon:
            return random.randint(0, len(neighbors) - 1)
        else:
            best_action = 0
            best_value = -float('inf')
            
            for i, neighbor in enumerate(neighbors):
                q_value = self.q_values[agent_id, current_pos, neighbor]
                if q_value > best_value:
                    best_value = q_value
                    best_action = i
            
            return best_action
    
    def update(self, observations, actions, rewards, next_observations, dones):
        """更新Q值"""
        for agent_id in range(self.env.num_agents):
            if agent_id < len(actions):
                current_pos = self.env.agent_positions[agent_id]
                neighbors = getattr(self.env, 'neighbor_dict', {}).get(current_pos, [])
                
                if actions[agent_id] < len(neighbors):
                    next_pos = neighbors[actions[agent_id]]
                    reward = rewards[agent_id] if agent_id < len(rewards) else 0
                    
                    old_q = self.q_values[agent_id, current_pos, next_pos]
                    
                    future_neighbors = getattr(self.env, 'neighbor_dict', {}).get(next_pos, [])
                    max_future_q = 0
                    if future_neighbors:
                        max_future_q = max(self.q_values[agent_id, next_pos, n] 
                                         for n in future_neighbors)
                    
                    new_q = old_q + self.learning_rate * (reward + 0.9 * max_future_q - old_q)
                    self.q_values[agent_id, current_pos, next_pos] = new_q
        
        self.epsilon *= self.decay_rate

class RandomAlgorithm(BaselinePatrollingAlgorithm):
    """随机算法作为基准"""
    
    def __init__(self, env):
        super().__init__(env, "Random")
    
    def select_actions(self, observations):
        """随机选择动作"""
        actions = []
        
        for agent_id in range(self.env.num_agents):
            current_pos = self.env.agent_positions[agent_id]
            neighbors = getattr(self.env, 'neighbor_dict', {}).get(current_pos, [])
            
            if neighbors:
                action = random.randint(0, len(neighbors) - 1)
            else:
                action = 0
            
            actions.append(action)
        
        return np.array(actions)

# ============================================================================
# 增强版MAGEC评估器
# ============================================================================

class EnhancedMAGECEvaluator:
    """增强版MAGEC评估器 - 支持第一阶段拓扑"""
    
    def __init__(self, env_config, topology_file=None):
        self.env_config = env_config
        self.topology_file = topology_file
        self.algorithms = {}
        self.results = {}
        
        # 创建拓扑适配器
        self.topology_adapter = TopologyAwareEnvironmentAdapter(topology_file, env_config)
        self.topology_info = self.topology_adapter.get_topology_info()
        
        print(f"🌍 环境类型: {self.topology_info['type']}")
        if self.topology_info['type'] == 'custom_stage1':
            print(f"   拓扑文件: {self.topology_info['source_file']}")
            print(f"   系统: {self.topology_info['system']}")
            print(f"   阶段: {self.topology_info['stage']}")
            print(f"   节点数: {self.topology_info['nodes_count']}")
            print(f"   边数: {self.topology_info['edges_count']}")
    
    def register_algorithm(self, name, algorithm):
        """注册算法"""
        self.algorithms[name] = algorithm
        logger.info(f"注册算法: {name}")
    
    def load_magec_model(self, model_path):
            """加载训练好的MAGEC模型 - 修复PyTorch 2.6兼容性问题"""
            try:
                # 确保所有必要的模块都被正确导入
                import torch
                import torch.nn as nn
                import numpy as np
                
                print(f"🔍 检测PyTorch版本: {torch.__version__}")
                
                # 多重回退策略加载模型
                checkpoint = None
                
                # 策略1：使用安全的全局对象列表（PyTorch 2.6+推荐方式）
                try:
                    # 检查是否有torch.serialization模块
                    if hasattr(torch, 'serialization'):
                        safe_globals = [
                            'numpy.core.multiarray.scalar',
                            'numpy.core.multiarray._reconstruct', 
                            'numpy.ndarray',
                            'numpy.dtype',
                            'collections.OrderedDict',
                            'torch._utils._rebuild_tensor_v2'
                        ]
                        
                        print("🔒 尝试使用安全全局对象加载...")
                        with torch.serialization.safe_globals(safe_globals):
                            checkpoint = torch.load(model_path, map_location='cpu')
                        print("✅ 安全加载成功")
                    else:
                        raise AttributeError("torch.serialization不可用")
                    
                except (AttributeError, ImportError, Exception) as e:
                    print(f"⚠️ 安全加载失败: {e}")
                    
                    # 策略2：添加安全全局对象（适用于部分PyTorch 2.6版本）
                    try:
                        if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
                            print("🔧 尝试添加安全全局对象...")
                            torch.serialization.add_safe_globals([
                                'numpy.core.multiarray.scalar',
                                'numpy.core.multiarray._reconstruct',
                                'numpy.ndarray',
                                'numpy.dtype'
                            ])
                            checkpoint = torch.load(model_path, map_location='cpu')
                            print("✅ 添加全局对象后加载成功")
                        else:
                            raise AttributeError("add_safe_globals不可用")
                            
                    except (AttributeError, ImportError, Exception) as e2:
                        print(f"⚠️ 添加全局对象失败: {e2}")
                        
                        # 策略3：使用weights_only=False（适用于可信模型）
                        try:
                            print("🛡️ 尝试使用非安全模式加载可信模型...")
                            # 检查torch.load是否支持weights_only参数
                            import inspect
                            load_signature = inspect.signature(torch.load)
                            if 'weights_only' in load_signature.parameters:
                                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                            else:
                                checkpoint = torch.load(model_path, map_location='cpu')
                            print("✅ 非安全模式加载成功")
                            
                        except Exception as e3:
                            print(f"⚠️ 非安全模式失败: {e3}")
                            
                            # 策略4：最终回退（旧版PyTorch标准加载）
                            try:
                                print("🔄 尝试标准加载...")
                                checkpoint = torch.load(model_path, map_location='cpu')
                                print("✅ 标准加载成功")
                            except Exception as e4:
                                print(f"❌ 标准加载失败: {e4}")
                                raise e4
                
                if checkpoint is None:
                    raise Exception("所有加载策略都失败，无法加载模型")
                
                # 验证checkpoint内容
                required_keys = ['actor_state_dict']
                missing_keys = [key for key in required_keys if key not in checkpoint]
                if missing_keys:
                    raise Exception(f"模型文件缺少必要的键: {missing_keys}")
                
                print("📋 验证模型文件内容...")
                print(f"   包含的键: {list(checkpoint.keys())}")
                
                # 尝试获取配置
                if MAGEC_AVAILABLE:
                    config = checkpoint.get('config', create_magec_config())
                else:
                    config = checkpoint.get('config', create_official_config())
                
                print("🌍 创建训练环境...")
                # 创建环境
                env = self.topology_adapter.create_environment()
                
                # 获取最大邻居数
                if hasattr(env, 'max_neighbors'):
                    max_neighbors = env.max_neighbors
                elif hasattr(env, 'get_max_neighbors'):
                    max_neighbors = env.get_max_neighbors()
                else:
                    max_neighbors = 15  # 默认值
                
                print(f"🧠 创建Actor网络 (max_neighbors={max_neighbors})...")
                # 创建网络
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                try:
                    actor = MAGECActor(
                        node_features=config['network']['node_features'],
                        edge_features=config['network']['edge_features'],
                        hidden_size=config['network']['gnn_hidden_size'],
                        num_layers=config['network']['gnn_layers'],
                        max_neighbors=max_neighbors,
                        dropout=config['network']['gnn_dropout'],
                        use_skip_connections=config['network']['gnn_skip_connections']
                    ).to(device)
                    print("✅ Actor网络创建成功")
                except Exception as e:
                    print(f"❌ Actor网络创建失败: {e}")
                    print(f"配置信息: {config['network']}")
                    raise e
                
                print("⚖️ 加载模型权重...")
                # 加载权重
                try:
                    actor.load_state_dict(checkpoint['actor_state_dict'])
                    actor.eval()
                    print("✅ 模型权重加载成功")
                except Exception as e:
                    print(f"⚠️ 严格权重加载失败: {e}")
                    # 尝试部分加载
                    try:
                        missing_keys, unexpected_keys = actor.load_state_dict(checkpoint['actor_state_dict'], strict=False)
                        if missing_keys:
                            print(f"   缺少的键: {missing_keys}")
                        if unexpected_keys:
                            print(f"   多余的键: {unexpected_keys}")
                        actor.eval()
                        print("⚠️ 部分权重加载成功（非严格模式）")
                    except Exception as e2:
                        print(f"❌ 非严格模式权重加载也失败: {e2}")
                        raise Exception(f"权重加载完全失败: {e2}")
                
                print("🤖 创建MAGEC算法包装器...")
                # 创建MAGEC算法包装器
                class MAGECAlgorithmWrapper(BaselinePatrollingAlgorithm):
                    def __init__(self, actor, device, max_neighbors):
                        self.actor = actor
                        self.device = device
                        self.max_neighbors = max_neighbors
                        self.name = "MAGEC"
                    
                    def select_actions(self, observations):
                        try:
                            with torch.no_grad():
                                # 获取智能体位置
                                agent_indices = []
                                for obs in observations:
                                    if hasattr(obs, 'agent_pos'):
                                        agent_indices.append(obs.agent_pos.item())
                                    else:
                                        agent_indices.append(0)
                                
                                # 移动到正确设备
                                observations_device = []
                                for obs in observations:
                                    obs_device = obs.clone()
                                    obs_device.x = obs_device.x.to(self.device)
                                    obs_device.edge_index = obs_device.edge_index.to(self.device)
                                    if hasattr(obs_device, 'edge_attr') and obs_device.edge_attr is not None:
                                        obs_device.edge_attr = obs_device.edge_attr.to(self.device)
                                    if hasattr(obs_device, 'agent_pos'):
                                        obs_device.agent_pos = obs_device.agent_pos.to(self.device)
                                    observations_device.append(obs_device)
                                
                                # 前向传播
                                action_logits = self.actor(observations_device, agent_indices)
                                
                                # 确定性动作选择
                                actions = []
                                for i in range(len(observations)):
                                    # 获取实际可用的邻居数量
                                    if hasattr(self, 'env') and hasattr(self.env, 'agent_positions'):
                                        agent_pos = self.env.agent_positions[i] if i < len(self.env.agent_positions) else 0
                                        max_neighbors = len(self.env.neighbor_dict.get(agent_pos, [1]))
                                    else:
                                        max_neighbors = self.max_neighbors
                                    
                                    # 限制动作范围
                                    if action_logits.dim() == 1:
                                        logits = action_logits
                                    else:
                                        logits = action_logits[i] if i < action_logits.size(0) else action_logits[0]
                                    
                                    # 只取前max_neighbors个logits
                                    valid_logits = logits[:max_neighbors] if logits.size(0) > max_neighbors else logits
                                    action = torch.argmax(valid_logits).item()
                                    actions.append(min(action, max_neighbors - 1))

                                return np.array(actions)
                                
                                
                                
                        except Exception as e:
                            print(f"⚠️ MAGEC动作选择失败，使用随机动作: {e}")
                            # 回退到随机动作
                            return np.random.randint(0, self.max_neighbors, size=len(observations))
                
                magec_wrapper = MAGECAlgorithmWrapper(actor, device, max_neighbors)
                magec_wrapper.env = env 
                self.register_algorithm("MAGEC", magec_wrapper)
                
                logger.info(f"MAGEC模型加载成功: {model_path}")
                print(f"✅ MAGEC模型加载成功: {model_path}")
                print(f"📊 模型配置: {config['network']['gnn_hidden_size']}隐藏层, {config['network']['gnn_layers']}层GNN")
                print(f"🎯 设备: {device}")
                return env
                
            except Exception as e:
                import traceback
                logger.error(f"加载MAGEC模型失败: {e}")
                print(f"❌ 加载MAGEC模型失败: {e}")
                
                # 打印详细的错误信息
                print("\n🔍 详细错误信息:")
                traceback.print_exc()
                
                # 提供详细的故障排除建议
                print("\n🔧 故障排除建议:")
                print("1. 检查模型文件是否存在且完整")
                print(f"   模型路径: {model_path}")
                print(f"   文件存在: {os.path.exists(model_path)}")
                if os.path.exists(model_path):
                    print(f"   文件大小: {os.path.getsize(model_path) / (1024*1024):.1f}MB")
                print("2. 确认PyTorch版本兼容性")
                print(f"   当前PyTorch版本: {torch.__version__}")
                print("3. 尝试使用更简单的模型文件")
                print("4. 考虑重新训练模型")
                
                return None
    
    def evaluate_algorithms(self, num_episodes=50, episode_length=100, 
                          test_scenarios=None):
        """评估所有算法"""
        if test_scenarios is None:
            test_scenarios = [
                {'name': 'normal', 'attrition': False, 'comm_loss': 0.0},
                {'name': 'attrition', 'attrition': True, 'comm_loss': 0.0},
                {'name': 'comm_loss', 'attrition': False, 'comm_loss': 0.5},
                {'name': 'both', 'attrition': True, 'comm_loss': 0.5}
            ]
        
        logger.info(f"开始评估 {len(self.algorithms)} 个算法")
        logger.info(f"测试场景: {[s['name'] for s in test_scenarios]}")
        
        # 显示拓扑信息
        if self.topology_info['type'] == 'custom_stage1':
            print(f"📍 在第一阶段拓扑上测试: {self.topology_info['nodes_count']}节点, {self.topology_info['edges_count']}边")
        
        for scenario in test_scenarios:
            logger.info(f"测试场景: {scenario['name']}")
            print(f"🎭 正在测试场景: {scenario['name']}")
            scenario_results = {}
            
            for alg_idx, (alg_name, algorithm) in enumerate(self.algorithms.items(), 1):
                logger.info(f"  评估算法: {alg_name}")
                print(f"  🤖 ({alg_idx}/{len(self.algorithms)}) 评估算法: {alg_name}")
                
                # 运行多个回合
                episode_rewards = []
                episode_idleness = []
                episode_steps = []
                
                # 添加进度条
                from tqdm import tqdm
                progress_bar = tqdm(range(num_episodes), 
                                  desc=f"    {alg_name}", 
                                  unit="ep", 
                                  leave=False,
                                  ncols=80)
                
                for episode in progress_bar:
                    # 创建环境
                    env = self.topology_adapter.create_environment()
                    
                    # 模拟智能体损失
                    if scenario['attrition'] and episode > num_episodes // 3:
                        if env.num_agents > 1:
                            env.num_agents -= 1
                            env.agent_positions = env.agent_positions[:-1]
                    
                    observations = env.reset()
                    algorithm.reset()
                    
                    episode_reward = []
                    done = False
                    step = 0
                    
                    while not done and step < episode_length:
                        # 选择动作
                        actions = algorithm.select_actions(observations)
                        
                        # 模拟通信损失
                        if scenario['comm_loss'] > 0:
                            for i in range(len(actions)):
                                if random.random() < scenario['comm_loss']:
                                    actions[i] = 0
                        
                        # 执行动作
                        next_observations, rewards, done = env.step(actions)
                        
                        # 更新算法
                        if hasattr(algorithm, 'update'):
                            algorithm.update(observations, actions, rewards, 
                                           next_observations, done)
                        
                        episode_reward.extend(rewards)
                        observations = next_observations
                        step += 1
                    
                    # 记录结果
                    episode_rewards.append(np.mean(episode_reward))
                    episode_idleness.append(env.mean_idleness)
                    episode_steps.append(step)
                    
                    # 更新进度条
                    progress_bar.set_postfix({
                        'Reward': f'{np.mean(episode_reward):.2f}',
                        'Idleness': f'{env.mean_idleness:.1f}'
                    })
                
                progress_bar.close()
                
                # 计算统计
                scenario_results[alg_name] = {
                    'avg_reward': np.mean(episode_rewards),
                    'std_reward': np.std(episode_rewards),
                    'avg_idleness': np.mean(episode_idleness),
                    'std_idleness': np.std(episode_idleness),
                    'avg_steps': np.mean(episode_steps),
                    'episode_rewards': episode_rewards,
                    'episode_idleness': episode_idleness
                }
                
                logger.info(f"    平均奖励: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
                logger.info(f"    平均闲置: {np.mean(episode_idleness):.3f} ± {np.std(episode_idleness):.3f}")
                print(f"    ✅ {alg_name}: 奖励={np.mean(episode_rewards):.3f}±{np.std(episode_rewards):.3f}, "
                      f"闲置={np.mean(episode_idleness):.3f}±{np.std(episode_idleness):.3f}")
            
            self.results[scenario['name']] = scenario_results
            print(f"✅ 场景 '{scenario['name']}' 评估完成\n")
        
        logger.info("评估完成！")
        return self.results
    
    def plot_comparison_results(self, save_path=None):
        """绘制对比结果"""
        if not self.results:
            logger.warning("没有结果可绘制")
            return
        
        scenarios = list(self.results.keys())
        algorithms = list(self.algorithms.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 添加拓扑信息到标题
        if self.topology_info['type'] == 'custom_stage1':
            title_suffix = f" (第一阶段拓扑: {self.topology_info['nodes_count']}节点)"
        else:
            title_suffix = f" (标准拓扑: {self.topology_info.get('graph_name', 'unknown')})"
        
        fig.suptitle(f'MAGEC vs Baseline Algorithms Comparison{title_suffix}', 
                    fontsize=16, fontweight='bold')
        
        # 平均奖励对比
        ax = axes[0, 0]
        x = np.arange(len(scenarios))
        width = 0.15
        
        for i, alg in enumerate(algorithms):
            rewards = [self.results[s][alg]['avg_reward'] for s in scenarios]
            errors = [self.results[s][alg]['std_reward'] for s in scenarios]
            ax.bar(x + i * width, rewards, width, label=alg, yerr=errors, capsize=3)
        
        ax.set_xlabel('Test Scenarios')
        ax.set_ylabel('Average Reward')
        ax.set_title('Average Reward Comparison')
        ax.set_xticks(x + width * (len(algorithms) - 1) / 2)
        ax.set_xticklabels(scenarios, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 平均闲置时间对比
        ax = axes[0, 1]
        for i, alg in enumerate(algorithms):
            idleness = [self.results[s][alg]['avg_idleness'] for s in scenarios]
            errors = [self.results[s][alg]['std_idleness'] for s in scenarios]
            ax.bar(x + i * width, idleness, width, label=alg, yerr=errors, capsize=3)
        
        ax.set_xlabel('Test Scenarios')
        ax.set_ylabel('Average Idleness')
        ax.set_title('Average Idleness Comparison (Lower is Better)')
        ax.set_xticks(x + width * (len(algorithms) - 1) / 2)
        ax.set_xticklabels(scenarios, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 奖励分布箱线图（normal场景）
        ax = axes[1, 0]
        if 'normal' in self.results:
            reward_data = [self.results['normal'][alg]['episode_rewards'] for alg in algorithms]
            box_plot = ax.boxplot(reward_data, labels=algorithms, patch_artist=True)
            
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
            for patch, color in zip(box_plot['boxes'], colors[:len(algorithms)]):
                patch.set_facecolor(color)
        
        ax.set_xlabel('Algorithms')
        ax.set_ylabel('Episode Rewards')
        ax.set_title('Reward Distribution (Normal Scenario)')
        ax.grid(True, alpha=0.3)
        
        # 性能热力图
        ax = axes[1, 1]
        
        perf_matrix = np.zeros((len(algorithms), len(scenarios)))
        for i, alg in enumerate(algorithms):
            for j, scenario in enumerate(scenarios):
                perf_matrix[i, j] = -self.results[scenario][alg]['avg_idleness']
        
        perf_matrix = (perf_matrix - perf_matrix.min()) / (perf_matrix.max() - perf_matrix.min())
        
        im = ax.imshow(perf_matrix, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(len(scenarios)))
        ax.set_yticks(range(len(algorithms)))
        ax.set_xticklabels(scenarios, rotation=45)
        ax.set_yticklabels(algorithms)
        ax.set_title('Performance Heatmap (Green=Better)')
        
        for i in range(len(algorithms)):
            for j in range(len(scenarios)):
                text = ax.text(j, i, f'{perf_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"对比结果已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_patrolling_animation(self, algorithm_name, save_path=None, 
                                     episode_length=100):
        """创建巡逻过程的动画可视化（增强版，支持第一阶段拓扑）"""
        if algorithm_name not in self.algorithms:
            logger.error(f"算法 {algorithm_name} 未注册")
            return
        
        # 创建环境和算法
        env = self.topology_adapter.create_environment()
        algorithm = self.algorithms[algorithm_name]
        
        observations = env.reset()
        algorithm.reset()
        
        # 记录巡逻轨迹
        trajectory = []
        idleness_history = []
        
        print(f"📹 正在记录 {algorithm_name} 的巡逻轨迹...")
        
        for step in range(episode_length):
            # 记录当前状态
            current_state = {
                'agent_positions': env.agent_positions.copy(),
                'node_idleness': env.node_idleness.copy() if isinstance(env.node_idleness, dict) else env.node_idleness,
                'step': step
            }
            trajectory.append(current_state)
            idleness_history.append(env.mean_idleness)
            
            # 选择并执行动作
            actions = algorithm.select_actions(observations)
            next_observations, rewards, done = env.step(actions)
            
            if hasattr(algorithm, 'update'):
                algorithm.update(observations, actions, rewards, next_observations, done)
            
            observations = next_observations
            
            if done:
                break
        
        print(f"✅ 轨迹记录完成，共 {len(trajectory)} 步")
        
        # 创建动画
        self._create_enhanced_patrolling_animation(env, trajectory, idleness_history, 
                                                 algorithm_name, save_path)
    
    def _create_enhanced_patrolling_animation(self, env, trajectory, idleness_history, 
                                            algorithm_name, save_path):
        """创建增强版巡逻动画（支持第一阶段拓扑）"""
        try:
            print(f"🎬 正在创建 {algorithm_name} 的动画...")
            
            if not trajectory:
                print("❌ 轨迹数据为空，无法创建动画")
                return
            
            # 验证轨迹数据
            for i, state in enumerate(trajectory):
                if not isinstance(state, dict) or 'agent_positions' not in state:
                    print(f"❌ 轨迹数据格式错误，帧 {i}")
                    return
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
            
            # 获取节点位置和图信息
            if hasattr(env, 'env'):
                # 如果是包装的环境，获取内部环境
                inner_env = env.env
            else:
                inner_env = env
            
            if hasattr(inner_env, 'graph') and hasattr(inner_env, 'position_mapping'):
                # 第一阶段拓扑
                pos = inner_env.position_mapping
                graph = inner_env.graph
            elif hasattr(inner_env, 'node_positions'):
                # 标准环境
                pos = inner_env.node_positions
                graph = inner_env.graph
            else:
                # 回退方案：创建简单布局
                print("⚠️ 使用回退节点布局")
                pos = {i: (i*10, 0) for i in range(env.num_nodes)}
                graph = nx.path_graph(env.num_nodes)
            
            # 验证位置数据
            if not pos:
                print("❌ 节点位置数据为空")
                return
            
            # 计算合适的显示范围
            try:
                x_coords = [pos[node][0] for node in pos.keys() if node in pos]
                y_coords = [pos[node][1] for node in pos.keys() if node in pos]
                
                if not x_coords or not y_coords:
                    print("❌ 无有效的节点坐标")
                    return
                
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # 添加边距
                x_range = x_max - x_min if x_max != x_min else 1.0
                y_range = y_max - y_min if y_max != y_min else 1.0
                margin_ratio = VISUALIZATION_CONFIG['margin_ratio']
                x_margin = x_range * margin_ratio
                y_margin = y_range * margin_ratio
            except Exception as e:
                print(f"⚠️ 计算显示范围失败: {e}")
                x_min = y_min = 0
                x_max = y_max = 100
                x_margin = y_margin = 10
            
            # 存储绘图对象以避免重复创建
            plot_objects = {}
            
            def animate(frame):
                try:
                    if frame >= len(trajectory):
                        return []
                    
                    state = trajectory[frame]
                    
                    # 清空并重新绘制
                    ax1.clear()
                    ax2.clear()
                    
                    # 左图：拓扑可视化
                    topology_type = "第一阶段拓扑" if self.topology_info['type'] == 'custom_stage1' else "标准拓扑"
                    ax1.set_title(f'{algorithm_name} Multi-Agent Patrolling\n{topology_type} - Step {state["step"]} / {len(trajectory)-1}', 
                                 fontsize=14, fontweight='bold')
                    
                    # 绘制所有边
                    edge_color = VISUALIZATION_CONFIG['colors']['edges']
                    if hasattr(graph, 'edges'):
                        for edge in graph.edges():
                            if edge[0] in pos and edge[1] in pos:
                                x_coords_edge = [pos[edge[0]][0], pos[edge[1]][0]]
                                y_coords_edge = [pos[edge[0]][1], pos[edge[1]][1]]
                                ax1.plot(x_coords_edge, y_coords_edge, color=edge_color, 
                                        alpha=0.6, linewidth=2, zorder=1)
                    
                    # 绘制每个节点
                    # 处理node_idleness可能是字典的情况
                    if isinstance(state['node_idleness'], dict):
                        idleness_values = list(state['node_idleness'].values())
                    else:
                        idleness_values = state['node_idleness']
                    
                    # 确保max_idleness不为0，避免除零错误
                    max_idleness = max(idleness_values) if idleness_values else 1
                    max_idleness = max(max_idleness, 1)  # 至少为1
                    
                    node_radius = VISUALIZATION_CONFIG['node_size']
                    
                    # 获取颜色配置
                    colors = VISUALIZATION_CONFIG['colors']
                    edge_colors = colors['edge_borders']
                    
                    # 限制绘制的节点数量以提高性能
                    max_nodes_to_draw = min(env.num_nodes, 100)
                    
                    for node in range(max_nodes_to_draw):
                        if node not in pos:
                            continue
                            
                        # 计算节点颜色 - 处理node_idleness可能是字典的情况
                        if isinstance(state['node_idleness'], dict):
                            idleness = state['node_idleness'].get(node, 0)
                        else:
                            idleness = state['node_idleness'][node] if node < len(state['node_idleness']) else 0
                        
                        normalized_idleness = idleness / max_idleness
                        
                        # 颜色区分
                        if normalized_idleness < 0.33:
                            face_color = colors['low_idleness']
                            edge_color = edge_colors[0]  # 深绿色
                        elif normalized_idleness < 0.67:
                            face_color = colors['medium_idleness']
                            edge_color = edge_colors[1]  # 橙色
                        else:
                            face_color = colors['high_idleness']
                            edge_color = edge_colors[2]  # 深红色
                        
                        # 检查节点是否有智能体
                        has_agent = node in state['agent_positions']
                        
                        # 绘制节点圆圈
                        circle = plt.Circle(pos[node], node_radius, 
                                          facecolor=face_color, 
                                          edgecolor=edge_color,
                                          linewidth=3 if has_agent else 2, 
                                          alpha=0.9, zorder=2)
                        ax1.add_patch(circle)
                        
                        # 节点编号（只在小图中显示）
                        if env.num_nodes <= 20:
                            ax1.text(pos[node][0], pos[node][1], str(node), 
                                    ha='center', va='center', 
                                    fontsize=8, fontweight='bold', 
                                    color='black', zorder=4)
                        
                        # 显示闲置时间数值（只在小图中显示）
                        if VISUALIZATION_CONFIG['show_idleness_values'] and env.num_nodes <= 20:
                            ax1.text(pos[node][0], pos[node][1] - node_radius * 2, 
                                    f'{idleness:.0f}', 
                                    ha='center', va='center', 
                                    fontsize=7, color='darkblue', 
                                    fontweight='bold', zorder=4)
                    
                    # 绘制智能体
                    agent_colors = VISUALIZATION_CONFIG['colors']['agents']
                    agent_markers = VISUALIZATION_CONFIG['markers']
                    agent_size = VISUALIZATION_CONFIG['agent_size']
                    
                    for i, agent_pos in enumerate(state['agent_positions']):
                        if i < len(agent_colors) and agent_pos in pos:
                            color = agent_colors[i % len(agent_colors)]
                            marker = agent_markers[i % len(agent_markers)]
                            
                            # 绘制智能体
                            ax1.scatter(pos[agent_pos][0], pos[agent_pos][1], 
                                       s=agent_size, c=color, marker=marker, 
                                       edgecolors='white', linewidth=3, 
                                       alpha=0.95, zorder=5)
                            
                            # 智能体标签
                            ax1.text(pos[agent_pos][0], pos[agent_pos][1] + node_radius * 3, 
                                    f'A{i}', ha='center', va='center', 
                                    fontsize=10, fontweight='bold', 
                                    color=color, zorder=6,
                                    bbox=dict(boxstyle="round,pad=0.3", 
                                            facecolor='white', alpha=0.9,
                                            edgecolor=color, linewidth=2))
                    
                    # 设置坐标轴范围
                    ax1.set_xlim(x_min - x_margin, x_max + x_margin)
                    ax1.set_ylim(y_min - y_margin, y_max + y_margin)
                    ax1.set_aspect('equal')
                    
                    if VISUALIZATION_CONFIG['show_grid']:
                        ax1.grid(True, alpha=0.3)
                    
                    ax1.set_xlabel('X Position', fontsize=12)
                    ax1.set_ylabel('Y Position', fontsize=12)
                    
                    # 右图：性能指标图表
                    steps = list(range(len(idleness_history[:frame+1])))
                    if steps and idleness_history:
                        valid_history = idleness_history[:frame+1]
                        if valid_history:
                            ax2.plot(steps, valid_history, 'b-', 
                                    linewidth=3, label='Average Idleness', alpha=0.8)
                            ax2.fill_between(steps, valid_history, 
                                           alpha=0.3, color='lightblue')
                            
                            # 当前点标记
                            if frame < len(idleness_history):
                                ax2.scatter(frame, idleness_history[frame], 
                                           s=100, c='red', zorder=5)
                    
                    ax2.set_xlabel('Time Steps', fontsize=12)
                    ax2.set_ylabel('Average Idleness', fontsize=12)
                    ax2.set_title('Performance Over Time', fontsize=13, fontweight='bold')
                    ax2.grid(True, alpha=0.4)
                    ax2.legend(fontsize=11)
                    
                    if idleness_history:
                        max_idleness_hist = max(idleness_history)
                        ax2.set_ylim(0, max_idleness_hist * 1.1 if max_idleness_hist > 0 else 1)
                        ax2.set_xlim(0, len(idleness_history))
                    
                    # 添加当前统计信息
                    if frame < len(trajectory):
                        # 处理node_idleness可能是字典的情况
                        if isinstance(state['node_idleness'], dict):
                            current_idleness = np.mean(list(state['node_idleness'].values()))
                        else:
                            current_idleness = np.mean(state['node_idleness'])
                        
                        min_idleness = min(idleness_history[:frame+1]) if idleness_history[:frame+1] else 0
                        
                        stats_text = f'Current: {current_idleness:.2f}\nBest: {min_idleness:.2f}\nStep: {frame}'
                        if self.topology_info['type'] == 'custom_stage1':
                            stats_text += f'\nTopology: {self.topology_info["nodes_count"]}N'
                        
                        ax2.text(0.02, 0.98, stats_text, 
                                transform=ax2.transAxes, fontsize=10, 
                                verticalalignment='top',
                                bbox=dict(boxstyle="round,pad=0.4", 
                                        facecolor='lightyellow', alpha=0.8,
                                        edgecolor='orange'))
                    
                    plt.tight_layout()
                    return []
                
                except Exception as e:
                    print(f"⚠️ 动画帧 {frame} 渲染失败: {e}")
                    return []
            
            # 创建动画，限制帧数以避免内存问题
            max_frames = min(len(trajectory), 200)  # 最多200帧
            print(f"🎞️ 正在生成动画文件... (共{max_frames}帧)")
            
            ani = animation.FuncAnimation(
                fig, animate, frames=max_frames,
                interval=VISUALIZATION_CONFIG['animation_interval'], 
                repeat=False, blit=False  # 设置blit=False避免兼容性问题
            )
            
            if save_path:
                # 确保目录存在
                save_dir = os.path.dirname(save_path)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                
                # 保存动画，使用更保守的参数
                print(f"💾 保存动画到: {save_path}")
                try:
                    ani.save(save_path, writer='pillow', 
                            fps=max(1, VISUALIZATION_CONFIG['animation_fps']), 
                            dpi=80,  # 降低DPI以减少文件大小
                            savefig_kwargs={'bbox_inches': 'tight', 'pad_inches': 0.1})
                    print(f"✅ 动画已保存: {save_path}")
                    logger.info(f"动画已保存: {save_path}")
                except Exception as save_error:
                    print(f"❌ 保存动画失败: {save_error}")
                    # 尝试备用保存方法
                    try:
                        backup_path = save_path.replace('.gif', '_backup.gif')
                        ani.save(backup_path, writer='pillow', fps=2, dpi=60)
                        print(f"✅ 备用动画已保存: {backup_path}")
                    except:
                        print("❌ 备用保存也失败")
            else:
                plt.show()
            
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"创建动画失败: {e}")
            print(f"❌ 创建动画失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 清理资源
            try:
                plt.close('all')
            except:
                pass
    
    def save_results_report(self, save_path="results/evaluation_report.json"):
        """保存评估报告"""
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            report = {
                'algorithms': list(self.algorithms.keys()),
                'topology_info': self.topology_info,
                'env_config': self.env_config,
                'results': self.results,
                'summary': self._generate_summary(),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=lambda x: float(x) if isinstance(x, np.float64) else x)
            
            logger.info(f"评估报告已保存: {save_path}")
            
        except Exception as e:
            logger.error(f"保存报告失败: {e}")
    
    def _generate_summary(self):
        """生成评估摘要"""
        if not self.results:
            return {}
        
        summary = {}
        
        for scenario, scenario_results in self.results.items():
            best_reward_alg = max(scenario_results.keys(), 
                                key=lambda x: scenario_results[x]['avg_reward'])
            best_idleness_alg = min(scenario_results.keys(), 
                                  key=lambda x: scenario_results[x]['avg_idleness'])
            
            summary[scenario] = {
                'best_reward_algorithm': best_reward_alg,
                'best_reward_value': scenario_results[best_reward_alg]['avg_reward'],
                'best_idleness_algorithm': best_idleness_alg,
                'best_idleness_value': scenario_results[best_idleness_alg]['avg_idleness'],
                'magec_performance': scenario_results.get('MAGEC', {})
            }
        
        return summary

# ============================================================================
# 辅助函数
# ============================================================================

def get_node_idleness(env, node=None):
    """统一获取节点闲置时间的辅助函数"""
    if isinstance(env.node_idleness, dict):
        if node is not None:
            return env.node_idleness.get(node, 0)
        else:
            return list(env.node_idleness.values())
    else:
        if node is not None:
            return env.node_idleness[node]
        else:
            return env.node_idleness

def get_mean_idleness(env):
    """获取平均闲置时间"""
    if isinstance(env.node_idleness, dict):
        values = list(env.node_idleness.values())
        return np.mean(values) if values else 0
    else:
        return np.mean(env.node_idleness)

def discover_topology_files():
    """发现第一阶段拓扑文件"""
    topology_files = []
    
    # 搜索模式
    search_patterns = [
        '*topology*.json',
        '*complete_topology*.json', 
        '*stage1*.json',
        'experiments/*topology*.json',
        'results/*topology*.json'
    ]
    
    import glob
    for pattern in search_patterns:
        files = glob.glob(pattern, recursive=True)
        for file in files:
            if os.path.isfile(file):
                # 验证是否是有效的拓扑文件
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 检查是否包含第一阶段标识
                    if ('stage' in data and 'topology' in data.get('stage', '').lower()) or \
                       ('system' in data and 'topology' in data.get('system', '').lower()) or \
                       'ready_for_stage2' in data:
                        topology_files.append(file)
                except:
                    continue
    
    # 去重并按时间排序
    topology_files = sorted(set(topology_files), key=lambda x: os.path.getmtime(x), reverse=True)
    
    return topology_files

def test_visualization_with_topology():
    """使用第一阶段拓扑测试可视化功能"""
    print("🧪 开始第一阶段拓扑可视化测试...")
    
    # 搜索拓扑文件
    topology_files = discover_topology_files()
    
    if not topology_files:
        print("❌ 未找到第一阶段拓扑文件，使用标准环境测试")
        topology_file = None
    else:
        print(f"✅ 发现拓扑文件: {topology_files[0]}")
        topology_file = topology_files[0]
    
    # 创建环境配置
    env_config = {
        'graph_name': 'milwaukee',
        'num_agents': 2,
        'observation_radius': 400.0,
        'max_cycles': 30,
        'agent_speed': 40.0,
        'action_method': 'neighbors'
    }
    
    # 创建评估器
    evaluator = EnhancedMAGECEvaluator(env_config, topology_file)
    
    # 创建环境测试
    env = evaluator.topology_adapter.create_environment()
    
    # 注册测试算法
    evaluator.register_algorithm('Random', RandomAlgorithm(env))
    
    # 创建输出目录
    os.makedirs('test_output', exist_ok=True)
    
    # 生成测试动画
    print("🎬 生成测试动画...")
    evaluator.visualize_patrolling_animation(
        'Random', 
        'test_output/topology_test_animation.gif', 
        episode_length=20
    )
    
    print("✅ 第一阶段拓扑测试完成！查看 test_output/topology_test_animation.gif")

# ============================================================================
# 增强的交互式输入
# ============================================================================

def enhanced_interactive_input():
    """增强的交互式输入配置（支持第一阶段拓扑）"""
    print("🚀 " + "=" * 76)
    print("🚀 MAGEC 算法评估和可视化工具 - 增强版交互式配置")
    print("🚀 支持第一阶段生成的拓扑测试")
    print("🚀 " + "=" * 76)
    print("💡 提示：直接按回车使用默认值，输入 'q' 退出")
    print()
    
    config = {}
    
    # 1. 拓扑选择
    print("🗺️ 拓扑选择")
    print("-" * 50)
    
    # 搜索第一阶段拓扑文件
    topology_files = discover_topology_files()
    
    print("请选择测试拓扑:")
    print("  0. 使用标准拓扑（milwaukee, 4nodes等）")
    
    if topology_files:
        print("  第一阶段生成的拓扑文件:")
        for i, topo_file in enumerate(topology_files[:5], 1):  # 最多显示5个
            try:
                with open(topo_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                system = data.get('system', 'Unknown')
                stage = data.get('stage', 'Unknown')
                ready = data.get('ready_for_stage2', False)
                mtime = time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(topo_file)))
                
                print(f"  {i}. {os.path.basename(topo_file)}")
                print(f"      📊 系统: {system}")
                print(f"      🏗️ 阶段: {stage}")
                print(f"      ✅ 就绪: {'是' if ready else '否'}")
                print(f"      📅 时间: {mtime}")
            except:
                print(f"  {i}. {os.path.basename(topo_file)}")
        print()
    else:
        print("  ⚠️ 未发现第一阶段拓扑文件")
        print("  💡 请先运行 gui.py 完成第一阶段拓扑构建")
        print()
    
    while True:
        topo_choice = input("请选择拓扑 [默认: 0-标准拓扑]: ").strip()
        if not topo_choice:
            config['topology_file'] = None
            config['topology_type'] = 'standard'
            break
        elif topo_choice == '0':
            config['topology_file'] = None
            config['topology_type'] = 'standard'
            break
        elif topo_choice.isdigit() and 1 <= int(topo_choice) <= len(topology_files):
            config['topology_file'] = topology_files[int(topo_choice) - 1]
            config['topology_type'] = 'custom_stage1'
            break
        elif os.path.exists(topo_choice):
            config['topology_file'] = topo_choice
            config['topology_type'] = 'custom_stage1'
            break
        else:
            print("❌ 无效选择，请重新输入")
    
    if config['topology_type'] == 'custom_stage1':
        print(f"✅ 已选择第一阶段拓扑: {config['topology_file']}")
    else:
        print("✅ 已选择标准拓扑")
    print()
    
    # 2. 模型路径配置（如果使用标准拓扑或需要MAGEC）
    print("📂 模型路径配置")
    print("-" * 50)
    
    # 搜索可用的模型文件
    possible_models = []
    if os.path.exists("experiments"):
        for root, dirs, files in os.walk("experiments"):
            for file in files:
                if file.endswith(('.pth', '.pt')):
                    possible_models.append(os.path.join(root, file))
    
    if possible_models:
        print("🔍 发现以下训练好的模型:")
        for i, model in enumerate(possible_models, 1):
            try:
                stat = os.stat(model)
                size_mb = stat.st_size / (1024 * 1024)
                mtime = time.strftime('%Y-%m-%d %H:%M', time.localtime(stat.st_mtime))
                print(f"  {i}. {model}")
                print(f"      📊 大小: {size_mb:.1f}MB, 📅 修改: {mtime}")
            except:
                print(f"  {i}. {model}")
        print()
        
        while True:
            choice = input("请选择模型 (输入序号或完整路径，跳过请按回车): ").strip()
            if not choice:
                config['magec_model'] = None
                print("⚠️ 未选择MAGEC模型，将只测试基准算法")
                break
            elif choice.lower() == 'q':
                print("👋 退出程序")
                sys.exit(0)
            elif choice.isdigit() and 1 <= int(choice) <= len(possible_models):
                config['magec_model'] = possible_models[int(choice) - 1]
                break
            elif os.path.exists(choice):
                config['magec_model'] = choice
                break
            else:
                print("❌ 无效的选择或文件不存在，请重新输入")
    else:
        print("⚠️ 未发现训练好的模型文件")
        config['magec_model'] = None
        print("将只测试基准算法")
    
    if config['magec_model']:
        print(f"✅ 已选择模型: {config['magec_model']}")
    print()
    
    # 3. 如果使用标准拓扑，配置环境参数
    if config['topology_type'] == 'standard':
        print("🌍 标准环境配置")
        print("-" * 50)
        
        # 图类型
        graphs = ['milwaukee', '4nodes']
        print("可选图类型:")
        for i, graph in enumerate(graphs, 1):
            print(f"  {i}. {graph}")
        
        while True:
            graph_choice = input("选择图类型 [默认: 1-milwaukee]: ").strip()
            if not graph_choice:
                config['graph_name'] = 'milwaukee'
                break
            elif graph_choice.isdigit() and 1 <= int(graph_choice) <= len(graphs):
                config['graph_name'] = graphs[int(graph_choice) - 1]
                break
            else:
                print("❌ 无效选择，请输入1-2")
        
        print(f"✅ 图类型: {config['graph_name']}")
    else:
        config['graph_name'] = 'custom'
    
    # 4. 智能体数量
    print("\n🤖 智能体配置")
    print("-" * 50)
    while True:
        agents_input = input("智能体数量 [默认: 4]: ").strip()
        if not agents_input:
            config['num_agents'] = 4
            break
        try:
            num_agents = int(agents_input)
            if 1 <= num_agents <= 10:
                config['num_agents'] = num_agents
                break
            else:
                print("❌ 智能体数量应在1-10之间")
        except ValueError:
            print("❌ 请输入有效数字")
    
    print(f"✅ 智能体数量: {config['num_agents']}")
    print()
    
    # 5. 输出目录配置
    print("📁 输出目录配置")
    print("-" * 50)
    default_output = f"results/enhanced_evaluation_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = input(f"输出目录 [默认: {default_output}]: ").strip()
    config['output_dir'] = output_dir if output_dir else default_output
    print(f"✅ 输出目录: {config['output_dir']}")
    print()
    
    # 6. 测试参数配置
    print("⚙️ 测试参数配置")
    print("-" * 50)
    
    # 快速测试模式
    quick_test = input("启用快速测试模式? (y/N) [默认: N]: ").strip().lower()
    config['quick_test'] = quick_test in ['y', 'yes', '1', 'true']
    
    if config['quick_test']:
        config['num_episodes'] = 10
        config['episode_length'] = 50
        print("✅ 快速测试模式: 10回合 × 50步")
    else:
        # 回合数
        while True:
            episodes_input = input("测试回合数 [默认: 50]: ").strip()
            if not episodes_input:
                config['num_episodes'] = 50
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
            steps_input = input("每回合步数 [默认: 100]: ").strip()
            if not steps_input:
                config['episode_length'] = 100
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
        
        print(f"✅ 测试参数: {config['num_episodes']}回合 × {config['episode_length']}步")
    
    print()
    
    # 7. 算法选择
    print("🤖 算法选择")
    print("-" * 50)
    available_algorithms = []
    if config['magec_model']:
        available_algorithms.append('MAGEC')
    available_algorithms.extend(['AHPA', 'SEBS', 'CBLS', 'Random'])
    
    print("可选算法:")
    for i, alg in enumerate(available_algorithms, 1):
        print(f"  {i}. {alg}")
    
    print("请选择要测试的算法 (用空格分隔多个选择，如: 1 2 3)")
    while True:
        alg_input = input(f"[默认: 1-{len(available_algorithms)} (全部)]: ").strip()
        if not alg_input:
            config['algorithms'] = available_algorithms.copy()
            break
        
        try:
            choices = [int(x) for x in alg_input.split()]
            if all(1 <= choice <= len(available_algorithms) for choice in choices):
                config['algorithms'] = [available_algorithms[i-1] for i in choices]
                break
            else:
                print(f"❌ 选择超出范围，请输入1-{len(available_algorithms)}")
        except ValueError:
            print("❌ 请输入有效的数字序列")
    
    print(f"✅ 选择算法: {', '.join(config['algorithms'])}")
    print()
    
    # 8. 测试场景
    print("🎭 测试场景")
    print("-" * 50)
    available_scenarios = [
        ('normal', '正常场景'),
        ('attrition', '智能体损失'),
        ('comm_loss', '通信干扰'),
        ('both', '复合干扰')
    ]
    
    print("可选测试场景:")
    for i, (name, desc) in enumerate(available_scenarios, 1):
        print(f"  {i}. {desc} ({name})")
    
    print("请选择测试场景 (用空格分隔多个选择)")
    while True:
        scenario_input = input("[默认: 1 2 3 4 (全部)]: ").strip()
        if not scenario_input:
            config['scenarios'] = [name for name, _ in available_scenarios]
            break
        
        try:
            choices = [int(x) for x in scenario_input.split()]
            if all(1 <= choice <= len(available_scenarios) for choice in choices):
                config['scenarios'] = [available_scenarios[i-1][0] for i in choices]
                break
            else:
                print("❌ 选择超出范围，请输入1-4")
        except ValueError:
            print("❌ 请输入有效的数字序列")
    
    print(f"✅ 测试场景: {', '.join(config['scenarios'])}")
    print()
    
    # 9. 可视化选项
    print("🎬 可视化选项")
    print("-" * 50)
    
    animate = input("生成动画可视化? (y/N) [默认: N]: ").strip().lower()
    config['animate'] = animate in ['y', 'yes', '1', 'true']
    
    if config['animate']:
        print("选择要生成动画的算法:")
        for i, alg in enumerate(config['algorithms'], 1):
            print(f"  {i}. {alg}")
        
        while True:
            anim_choice = input("[默认: 1-第一个算法]: ").strip()
            if not anim_choice:
                config['animate_algorithm'] = config['algorithms'][0]
                break
            try:
                choice = int(anim_choice)
                if 1 <= choice <= len(config['algorithms']):
                    config['animate_algorithm'] = config['algorithms'][choice-1]
                    break
                else:
                    print("❌ 选择超出范围")
            except ValueError:
                print("❌ 请输入有效数字")
        
        print(f"✅ 动画算法: {config['animate_algorithm']}")
    else:
        config['animate_algorithm'] = config['algorithms'][0] if config['algorithms'] else 'Random'
    
    print()
    
    # 10. 其他设置
    config['seed'] = 42
    
    # 显示最终配置
    print("📋 " + "=" * 76)
    print("📋 增强版最终配置确认")
    print("📋 " + "=" * 76)
    print(f"🔹 拓扑类型: {config['topology_type']}")
    if config['topology_type'] == 'custom_stage1':
        print(f"🔹 拓扑文件: {config['topology_file']}")
    else:
        print(f"🔹 标准图型: {config['graph_name']}")
    print(f"🔹 模型路径: {config['magec_model'] or '未选择（仅基准算法）'}")
    print(f"🔹 输出目录: {config['output_dir']}")
    print(f"🔹 智能体数: {config['num_agents']}")
    print(f"🔹 测试参数: {config['num_episodes']}回合 × {config['episode_length']}步")
    print(f"🔹 测试算法: {', '.join(config['algorithms'])}")
    print(f"🔹 测试场景: {', '.join(config['scenarios'])}")
    print(f"🔹 生成动画: {'是' if config['animate'] else '否'}")
    if config['animate']:
        print(f"🔹 动画算法: {config['animate_algorithm']}")
    print()
    
    confirm = input("确认开始评估? (Y/n) [默认: Y]: ").strip().lower()
    if confirm in ['n', 'no', '0', 'false']:
        print("👋 已取消评估")
        sys.exit(0)
    
    return config

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='MAGEC算法评估和可视化 - 增强版（支持第一阶段拓扑）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python visualize.py                                      # 交互式模式
  python visualize.py --quick_test --animate               # 快速测试
  python visualize.py --topology topology.json            # 使用第一阶段拓扑
  python visualize.py --magec_model model.pth             # 指定模型
  python visualize.py --batch --magec_model model.pth     # 批处理模式
  python visualize.py --test_viz                          # 测试可视化
  python visualize.py --test_topology                     # 测试第一阶段拓扑
        """
    )
    parser.add_argument('--topology', type=str, help='第一阶段拓扑JSON文件路径')
    parser.add_argument('--magec_model', type=str, help='训练好的MAGEC模型路径')
    parser.add_argument('--graph_name', type=str, default='milwaukee',
                       choices=['milwaukee', '4nodes'], help='标准图类型（当不使用自定义拓扑时）')
    parser.add_argument('--num_agents', type=int, default=4, help='智能体数量')
    parser.add_argument('--num_episodes', type=int, default=50, help='每个算法的测试回合数')
    parser.add_argument('--episode_length', type=int, default=100, help='每回合步数')
    parser.add_argument('--output_dir', type=str, help='输出目录')
    parser.add_argument('--algorithms', nargs='+', 
                       default=['AHPA', 'SEBS', 'CBLS', 'Random'],
                       help='要测试的算法')
    parser.add_argument('--scenarios', nargs='+',
                       default=['normal', 'attrition', 'comm_loss', 'both'],
                       help='测试场景')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--quick_test', action='store_true', help='快速测试模式')
    parser.add_argument('--animate', action='store_true', help='生成动画可视化')
    parser.add_argument('--animate_algorithm', type=str, default='Random',
                       help='要生成动画的算法')
    parser.add_argument('--interactive', action='store_true', help='交互式配置模式')
    parser.add_argument('--batch', action='store_true', help='批处理模式（使用命令行参数）')
    parser.add_argument('--test_viz', action='store_true', help='测试标准可视化功能')
    parser.add_argument('--test_topology', action='store_true', help='测试第一阶段拓扑可视化')
    
    args = parser.parse_args()
    
    # 测试功能
    if args.test_viz:
        test_visualization_with_topology()
        return
    
    if args.test_topology:
        test_visualization_with_topology()
        return
    
    # 决定使用交互式还是命令行模式
    if args.batch:
        # 批处理模式
        config = {
            'topology_file': args.topology,
            'topology_type': 'custom_stage1' if args.topology else 'standard',
            'magec_model': args.magec_model,
            'graph_name': args.graph_name,
            'num_agents': args.num_agents,
            'num_episodes': args.num_episodes,
            'episode_length': args.episode_length,
            'output_dir': args.output_dir or 'results/evaluation',
            'algorithms': args.algorithms,
            'scenarios': args.scenarios,
            'seed': args.seed,
            'quick_test': args.quick_test,
            'animate': args.animate,
            'animate_algorithm': args.animate_algorithm
        }
        
        print("🤖 批处理模式")
        
    elif args.interactive or (not args.magec_model and not args.topology):
        # 交互式模式
        config = enhanced_interactive_input()
    else:
        # 使用命令行参数
        config = {
            'topology_file': args.topology,
            'topology_type': 'custom_stage1' if args.topology else 'standard',
            'magec_model': args.magec_model,
            'graph_name': args.graph_name,
            'num_agents': args.num_agents,
            'num_episodes': args.num_episodes,
            'episode_length': args.episode_length,
            'output_dir': args.output_dir or 'results/evaluation',
            'algorithms': args.algorithms,
            'scenarios': args.scenarios,
            'seed': args.seed,
            'quick_test': args.quick_test,
            'animate': args.animate,
            'animate_algorithm': args.animate_algorithm
        }
    
    # 如果选择了MAGEC但没有提供模型
    if 'MAGEC' in config['algorithms'] and not config['magec_model']:
        print("⚠️ 选择了MAGEC算法但未提供模型，将从算法列表中移除MAGEC")
        config['algorithms'] = [alg for alg in config['algorithms'] if alg != 'MAGEC']
        if not config['algorithms']:
            config['algorithms'] = ['Random']
    
    # 设置随机种子
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    
    # 快速测试模式调整
    if config['quick_test']:
        config['num_episodes'] = 10
        config['episode_length'] = 50
        if config['topology_type'] == 'standard':
            config['graph_name'] = '4nodes'
        config['num_agents'] = 2
        logger.info("快速测试模式")
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 环境配置
    env_config = {
        'graph_name': config.get('graph_name', 'milwaukee'),
        'num_agents': config['num_agents'],
        'observation_radius': 400.0,
        'max_cycles': config['episode_length'],
        'agent_speed': 40.0,
        'action_method': 'neighbors'
    }
    
    print("\n" + "🎯 " + "=" * 76)
    print("🎯 开始增强版MAGEC算法评估")
    print("🎯 " + "=" * 76)
    if config['topology_type'] == 'custom_stage1':
        print(f"🗺️ 第一阶段拓扑: {config['topology_file']}")
    else:
        print(f"🗺️ 标准拓扑: {config['graph_name']}")
    print(f"📁 MAGEC模型: {config['magec_model'] or '未使用'}")
    print(f"🌍 智能体数量: {config['num_agents']}")
    print(f"🤖 测试算法: {', '.join(config['algorithms'])}")
    print(f"🎭 测试场景: {', '.join(config['scenarios'])}")
    print(f"⚙️ 测试设置: {config['num_episodes']} episodes × {config['episode_length']} steps")
    print(f"📂 输出目录: {config['output_dir']}")
    print("🎯 " + "=" * 76)
    
    # 创建增强版评估器
    evaluator = EnhancedMAGECEvaluator(env_config, config['topology_file'])
    
    # 加载MAGEC模型（如果需要）
    if 'MAGEC' in config['algorithms'] and config['magec_model']:
        if not os.path.exists(config['magec_model']):
            logger.error(f"MAGEC模型文件不存在: {config['magec_model']}")
            return
        
        print("📥 正在加载MAGEC模型...")
        env = evaluator.load_magec_model(config['magec_model'])
        if env is None:
            logger.error("MAGEC模型加载失败")
            return
        print("✅ MAGEC模型加载成功")
    else:
        env = evaluator.topology_adapter.create_environment()
    
    # 注册基准算法
    print("🔧 注册基准算法...")
    if 'AHPA' in config['algorithms']:
        evaluator.register_algorithm('AHPA', AHPAAlgorithm(env))
    
    if 'SEBS' in config['algorithms']:
        evaluator.register_algorithm('SEBS', SEBSAlgorithm(env))
    
    if 'CBLS' in config['algorithms']:
        evaluator.register_algorithm('CBLS', CBLSAlgorithm(env))
    
    if 'Random' in config['algorithms']:
        evaluator.register_algorithm('Random', RandomAlgorithm(env))
    
    print(f"✅ 已注册 {len(evaluator.algorithms)} 个算法")
    
    # 定义测试场景
    test_scenarios = []
    scenario_configs = {
        'normal': {'name': 'Normal', 'attrition': False, 'comm_loss': 0.0},
        'attrition': {'name': 'Agent Attrition', 'attrition': True, 'comm_loss': 0.0},
        'comm_loss': {'name': 'Communication Loss', 'attrition': False, 'comm_loss': 0.5},
        'both': {'name': 'Both Disturbances', 'attrition': True, 'comm_loss': 0.5}
    }
    
    for scenario_name in config['scenarios']:
        if scenario_name in scenario_configs:
            test_scenarios.append(scenario_configs[scenario_name])
    
    try:
        # 运行评估
        print(f"\n🏁 开始算法评估 ({len(test_scenarios)}个场景)...")
        results = evaluator.evaluate_algorithms(
            num_episodes=config['num_episodes'],
            episode_length=config['episode_length'],
            test_scenarios=test_scenarios
        )
        
        print("💾 保存评估结果...")
        # 保存结果
        evaluator.save_results_report(f"{config['output_dir']}/evaluation_report.json")
        print(f"✅ 评估报告已保存: {config['output_dir']}/evaluation_report.json")
        
        # 绘制对比图
        evaluator.plot_comparison_results(f"{config['output_dir']}/comparison_results.png")
        print(f"✅ 对比图表已保存: {config['output_dir']}/comparison_results.png")
        
        # 生成动画（如果请求）
        if config['animate'] and config['animate_algorithm'] in evaluator.algorithms:
            print(f"🎬 正在生成 {config['animate_algorithm']} 算法的动画...")
            animation_path = f"{config['output_dir']}/{config['animate_algorithm']}_animation.gif"
            evaluator.visualize_patrolling_animation(
                config['animate_algorithm'], animation_path, config['episode_length']
            )
            print(f"✅ 动画已保存: {animation_path}")
        
        # 打印摘要
        print("\n" + "📊 " + "=" * 76)
        print("📊 增强版评估结果摘要")
        print("📊 " + "=" * 76)
        
        # 显示拓扑信息
        topology_info = evaluator.topology_info
        if topology_info['type'] == 'custom_stage1':
            print(f"🗺️ 第一阶段拓扑测试结果:")
            print(f"   拓扑文件: {topology_info['source_file']}")
            print(f"   节点数: {topology_info['nodes_count']}")
            print(f"   边数: {topology_info['edges_count']}")
        else:
            print(f"🗺️ 标准拓扑测试结果: {topology_info.get('graph_name', 'unknown')}")
        
        summary = evaluator._generate_summary()
        for scenario, data in summary.items():
            print(f"\n🎭 {scenario.upper()} 场景:")
            print(f"  🏆 最佳奖励算法: {data['best_reward_algorithm']} ({data['best_reward_value']:.3f})")
            print(f"  ⚡ 最佳闲置算法: {data['best_idleness_algorithm']} ({data['best_idleness_value']:.3f})")
            
            if 'MAGEC' in data['magec_performance']:
                magec_perf = data['magec_performance']
                print(f"  🤖 MAGEC性能: 奖励={magec_perf['avg_reward']:.3f}, 闲置={magec_perf['avg_idleness']:.3f}")
        
        print("\n" + "🎉 " + "=" * 76)
        print("🎉 增强版评估完成！")
        print("🎉 " + "=" * 76)
        print(f"📁 详细结果保存在: {config['output_dir']}/")
        print("📋 主要文件:")
        print(f"  📄 评估报告: evaluation_report.json")
        print(f"  📈 对比图表: comparison_results.png")
        if config['animate']:
            print(f"  🎬 动画可视化: {config['animate_algorithm']}_animation.gif")
        print("🎉 " + "=" * 76)
        
        # 询问是否打开结果目录
        if not config.get('batch', False):
            open_dir = input("\n📂 是否打开结果目录? (y/N): ").strip().lower()
            if open_dir in ['y', 'yes', '1', 'true']:
                try:
                    import subprocess
                    import platform
                    
                    system = platform.system()
                    if system == 'Windows':
                        subprocess.run(['explorer', config['output_dir']])
                    elif system == 'Darwin':  # macOS
                        subprocess.run(['open', config['output_dir']])
                    else:  # Linux
                        subprocess.run(['xdg-open', config['output_dir']])
                except Exception as e:
                    print(f"⚠️ 无法自动打开目录: {e}")
                    print(f"📁 请手动访问: {config['output_dir']}")
        
    except KeyboardInterrupt:
        print("\n⛔ 评估被用户中断")
    except Exception as e:
        logger.error(f"评估过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 程序被用户中断，再见！")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        print("💡 请检查输入参数或查看帮助信息: python visualize.py --help")