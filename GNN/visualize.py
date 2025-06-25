#!/usr/bin/env python3
"""
MAGEC评估和可视化工具
实现与论文基准算法的对比：AHPA, SEBS, CBLS
提供完整的性能评估和可视化

基于论文: "Graph Neural Network-based Multi-agent Reinforcement Learning 
for Resilient Distributed Coordination of Multi-Robot Systems"

使用方法:
    1. 交互式模式 (推荐新手用户):
       python visualize.py
    
    2. 命令行模式:
       python visualize.py --magec_model path/to/model.pth --output_dir results/
    
    3. 批处理模式:
       python visualize.py --batch --magec_model path/to/model.pth
    
    4. 快速测试:
       python visualize.py --quick_test --animate

特性:
    ✅ 自动发现训练好的模型
    ✅ 交互式配置向导
    ✅ 多种基准算法对比
    ✅ 动画可视化巡逻过程
    ✅ 详细的性能报告
    ✅ 多种干扰场景测试
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

# 导入我们的MAGEC实现
try:
    from demo import (
        MAGECActor, MAGECCritic, OptimizedPatrollingEnvironment,
        MAGECTrainer, create_official_config
    )
except ImportError:
    print("请确保demo.py在同一目录下")
    sys.exit(1)

logger = logging.getLogger(__name__)

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
    """
    AHPA (Adaptive Heuristic-based Patrolling Algorithm)
    论文基准算法之一
    """
    
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
                idleness = self.env.node_idleness[node]
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
        neighbors = self.env.neighbor_dict.get(current_pos, [])
        
        if target_node in neighbors:
            return neighbors.index(target_node)
        elif neighbors:
            # 使用最短路径导航
            try:
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
    """
    SEBS (State Exchange Bayesian Strategy)
    论文基准算法之一
    """
    
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
                        'idleness': self.env.node_idleness[node],
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
        neighbors = self.env.neighbor_dict.get(current_pos, [])
        
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
            return self.env.node_idleness[node]

class CBLSAlgorithm(BaselinePatrollingAlgorithm):
    """
    CBLS (Concurrent Bayesian Learning Strategy)
    论文基准算法之一
    """
    
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
        neighbors = self.env.neighbor_dict.get(current_pos, [])
        
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
                neighbors = self.env.neighbor_dict.get(current_pos, [])
                
                if actions[agent_id] < len(neighbors):
                    next_pos = neighbors[actions[agent_id]]
                    reward = rewards[agent_id] if agent_id < len(rewards) else 0
                    
                    old_q = self.q_values[agent_id, current_pos, next_pos]
                    
                    future_neighbors = self.env.neighbor_dict.get(next_pos, [])
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
            neighbors = self.env.neighbor_dict.get(current_pos, [])
            
            if neighbors:
                action = random.randint(0, len(neighbors) - 1)
            else:
                action = 0
            
            actions.append(action)
        
        return np.array(actions)

class MAGECEvaluator:
    """MAGEC评估器 - 对比不同算法的性能"""
    
    def __init__(self, env_config):
        self.env_config = env_config
        self.algorithms = {}
        self.results = {}
    
    def register_algorithm(self, name, algorithm):
        """注册算法"""
        self.algorithms[name] = algorithm
        logger.info(f"注册算法: {name}")
    
    def load_magec_model(self, model_path):
        """加载训练好的MAGEC模型"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            config = checkpoint.get('config', create_official_config())
            
            # 创建环境
            env = OptimizedPatrollingEnvironment(
                graph_name=config['environment']['graph_name'],
                num_agents=config['environment']['num_agents'],
                observation_radius=config['environment']['observation_radius'],
                max_cycles=config['environment']['max_cycles'],
                agent_speed=config['environment']['agent_speed'],
                action_method=config['environment']['action_method']
            )
            
            # 创建网络
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            actor = MAGECActor(
                node_features=config['network']['node_features'],
                edge_features=config['network']['edge_features'],
                hidden_size=config['network']['gnn_hidden_size'],
                num_layers=config['network']['gnn_layers'],
                max_neighbors=env.get_max_neighbors(),
                dropout=config['network']['gnn_dropout'],
                use_skip_connections=config['network']['gnn_skip_connections']
            ).to(device)
            
            # 加载权重
            actor.load_state_dict(checkpoint['actor_state_dict'])
            actor.eval()
            
            # 创建MAGEC算法包装器
            class MAGECAlgorithmWrapper(BaselinePatrollingAlgorithm):
                def __init__(self, actor, device):
                    self.actor = actor
                    self.device = device
                    self.name = "MAGEC"
                    self.step_count = 0
                    self.env = None  # 环境引用，稍后设置                
                def select_actions(self, observations):
                    """选择动作 - 修复版"""
                    try:
                        self.step_count += 1
                        debug = (self.step_count <= 3)  # 前3次显示调试信息
                        
                        if debug:
                            print(f"🤖 MAGEC动作选择 #{self.step_count}")
                        
                        with torch.no_grad():
                            # 获取智能体位置
                            agent_indices = []
                            for i, obs in enumerate(observations):
                                if hasattr(obs, 'agent_pos'):
                                    agent_pos = obs.agent_pos.item()
                                    agent_indices.append(agent_pos)
                                else:
                                    agent_indices.append(0)
                            
                            if debug:
                                print(f"   智能体位置: {agent_indices}")
                            
                            # 移动到设备
                            observations_device = []
                            for obs in observations:
                                obs_device = obs.clone() if hasattr(obs, 'clone') else obs
                                obs_device.x = obs_device.x.to(self.device)
                                obs_device.edge_index = obs_device.edge_index.to(self.device)
                                if hasattr(obs_device, 'edge_attr') and obs_device.edge_attr is not None:
                                    obs_device.edge_attr = obs_device.edge_attr.to(self.device)
                                if hasattr(obs_device, 'agent_pos'):
                                    obs_device.agent_pos = obs_device.agent_pos.to(self.device)
                                observations_device.append(obs_device)
                            
                            # 前向传播
                            action_logits = self.actor(observations_device, agent_indices)
                            
                            if debug:
                                print(f"   Logits形状: {action_logits.shape}")
                                print(f"   Logits范围: [{action_logits.min().item():.3f}, {action_logits.max().item():.3f}]")
                            
                            # 🔥 关键修复：确保动作有效性
                            actions = []
                            for i in range(len(observations)):
                                # 🔥 修复：获取实际可用的邻居数量
                                if hasattr(self, 'env') and self.env and hasattr(self.env, 'agent_positions'):
                                    # 从环境获取实际邻居数
                                    if i < len(self.env.agent_positions):
                                        agent_pos = self.env.agent_positions[i]
                                        available_neighbors = len(getattr(self.env, 'neighbor_dict', {}).get(agent_pos, [1]))
                                    else:
                                        available_neighbors = 1
                                else:
                                    # 回退：从观察中估计
                                    if hasattr(observations[i], 'num_nodes'):
                                        available_neighbors = min(self.max_neighbors, observations[i].num_nodes.item())
                                    else:
                                        available_neighbors = min(self.max_neighbors, 5)
                                
                                # 确保至少有1个动作
                                available_neighbors = max(1, available_neighbors)
                                
                                # 获取该智能体的logits
                                if action_logits.dim() == 1:
                                    logits = action_logits
                                else:
                                    logits = action_logits[i] if i < action_logits.size(0) else action_logits[0]
                                
                                # 🔥 关键：限制到可用邻居数量
                                if logits.size(0) > available_neighbors:
                                    valid_logits = logits[:available_neighbors]
                                else:
                                    valid_logits = logits
                                
                                # 选择动作
                                if valid_logits.size(0) > 0:
                                    action = torch.argmax(valid_logits).item()
                                    action = min(action, available_neighbors - 1)  # 双重保险
                                else:
                                    action = 0
                                
                                actions.append(action)
                                
                                if debug:
                                    print(f"   智能体{i}: 可用邻居{available_neighbors}, 选择动作{action}")
                            
                            return np.array(actions)
                            
                    except Exception as e:
                        print(f"⚠️ MAGEC动作选择失败: {e}")
                        # 回退到安全的随机动作
                        actions = []
                        for i in range(len(observations)):
                            # 确保动作在安全范围内
                            if hasattr(observations[i], 'num_nodes'):
                                max_action = min(self.max_neighbors, observations[i].num_nodes.item()) - 1
                            else:
                                max_action = min(self.max_neighbors, 3) - 1
                            
                            action = np.random.randint(0, max(1, max_action + 1))
                            actions.append(action)
                        
                        return np.array(actions)
            
            magec_wrapper = MAGECAlgorithmWrapper(actor, device)
            self.register_algorithm("MAGEC", magec_wrapper)
            
            logger.info(f"MAGEC模型加载成功: {model_path}")
            return env
            
        except Exception as e:
            logger.error(f"加载MAGEC模型失败: {e}")
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
                    env = OptimizedPatrollingEnvironment(**self.env_config)
                    
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
                    episode_idleness.append(np.mean(env.node_idleness))
                    episode_steps.append(step)
                    
                    # 更新进度条
                    progress_bar.set_postfix({
                        'Reward': f'{np.mean(episode_reward):.2f}',
                        'Idleness': f'{np.mean(env.node_idleness):.1f}'
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
        fig.suptitle('MAGEC vs Baseline Algorithms Comparison', fontsize=16, fontweight='bold')
        
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
        """创建巡逻过程的动画可视化"""
        if algorithm_name not in self.algorithms:
            logger.error(f"算法 {algorithm_name} 未注册")
            return
        
        # 创建环境和算法
        env = OptimizedPatrollingEnvironment(**self.env_config)
        algorithm = self.algorithms[algorithm_name]
        
        observations = env.reset()
        algorithm.reset()
        
        # 记录巡逻轨迹
        trajectory = []
        idleness_history = []
        
        for step in range(episode_length):
            # 记录当前状态
            current_state = {
                'agent_positions': env.agent_positions.copy(),
                'node_idleness': env.node_idleness.copy(),
                'step': step
            }
            trajectory.append(current_state)
            idleness_history.append(np.mean(env.node_idleness))
            
            # 选择并执行动作
            actions = algorithm.select_actions(observations)
            next_observations, rewards, done = env.step(actions)
            
            if hasattr(algorithm, 'update'):
                algorithm.update(observations, actions, rewards, next_observations, done)
            
            observations = next_observations
            
            if done:
                break
        
        # 创建动画
        self._create_patrolling_animation(env, trajectory, idleness_history, 
                                        algorithm_name, save_path)
    
    def _create_patrolling_animation(self, env, trajectory, idleness_history, 
                                   algorithm_name, save_path):
        """创建巡逻动画"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 绘制图结构
            pos = env.node_positions
            
            def animate(frame):
                ax1.clear()
                ax2.clear()
                
                if frame >= len(trajectory):
                    return
                
                state = trajectory[frame]
                
                # 左图：图可视化
                ax1.set_title(f'{algorithm_name} Patrolling - Step {state["step"]}')
                
                # 绘制边
                for edge in env.graph.edges():
                    x_coords = [pos[edge[0]][0], pos[edge[1]][0]]
                    y_coords = [pos[edge[0]][1], pos[edge[1]][1]]
                    ax1.plot(x_coords, y_coords, 'k-', alpha=0.3, linewidth=1)
                
                # 绘制节点（颜色表示闲置时间）
                node_colors = []
                for node in range(env.num_nodes):
                    idleness = state['node_idleness'][node]
                    max_idleness = max(state['node_idleness']) if max(state['node_idleness']) > 0 else 1
                    normalized_idleness = idleness / max_idleness
                    node_colors.append(plt.cm.Reds(normalized_idleness))
                
                for node in range(env.num_nodes):
                    circle = Circle(pos[node], 0.05, color=node_colors[node], alpha=0.8)
                    ax1.add_patch(circle)
                    ax1.text(pos[node][0], pos[node][1], str(node), 
                            ha='center', va='center', fontsize=8, fontweight='bold')
                
                # 绘制智能体
                colors = ['blue', 'red', 'green', 'orange', 'purple']
                for i, agent_pos in enumerate(state['agent_positions']):
                    if i < len(colors) and agent_pos < len(pos):
                        circle = Circle(pos[agent_pos], 0.08, 
                                      color=colors[i], alpha=0.9, linewidth=2)
                        ax1.add_patch(circle)
                        ax1.text(pos[agent_pos][0], pos[agent_pos][1] + 0.12, 
                               f'A{i}', ha='center', va='center', 
                               fontsize=10, fontweight='bold', color=colors[i])
                
                ax1.set_xlim(-0.2, 1.2)
                ax1.set_ylim(-0.2, 1.2)
                ax1.set_aspect('equal')
                ax1.axis('off')
                
                # 添加颜色条说明
                ax1.text(1.1, 1.1, 'Idleness:', fontsize=10, fontweight='bold')
                ax1.text(1.1, 1.05, 'Low', fontsize=8, color='white')
                ax1.text(1.1, 0.95, 'High', fontsize=8, color='red')
                
                # 右图：性能指标
                steps = list(range(len(idleness_history[:frame+1])))
                ax2.plot(steps, idleness_history[:frame+1], 'b-', linewidth=2)
                ax2.set_xlabel('Step')
                ax2.set_ylabel('Average Idleness')
                ax2.set_title('Average Idleness Over Time')
                ax2.grid(True, alpha=0.3)
                
                if len(idleness_history) > 0:
                    ax2.set_ylim(0, max(idleness_history) * 1.1)
                    ax2.set_xlim(0, len(idleness_history))
                
                plt.tight_layout()
            
            # 创建动画
            ani = animation.FuncAnimation(fig, animate, frames=len(trajectory),
                                        interval=200, repeat=True, blit=False)
            
            if save_path:
                ani.save(save_path, writer='pillow', fps=5)
                logger.info(f"动画已保存: {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            logger.error(f"创建动画失败: {e}")
    
    def save_results_report(self, save_path="results/evaluation_report.json"):
        """保存评估报告"""
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            report = {
                'algorithms': list(self.algorithms.keys()),
                'env_config': self.env_config,
                'results': self.results,
                'summary': self._generate_summary(),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(save_path, 'w') as f:
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

def interactive_input():
    """交互式输入配置"""
    print("🚀 " + "=" * 76)
    print("🚀 MAGEC 算法评估和可视化工具 - 交互式配置")
    print("🚀 " + "=" * 76)
    print("💡 提示：直接按回车使用默认值，输入 'q' 退出")
    print()
    
    config = {}
    
    # 1. 模型路径配置
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
            # 显示文件大小和修改时间
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
            choice = input("请选择模型 (输入序号或完整路径): ").strip()
            if choice.lower() == 'q':
                print("👋 退出程序")
                sys.exit(0)
            
            if choice.isdigit() and 1 <= int(choice) <= len(possible_models):
                config['magec_model'] = possible_models[int(choice) - 1]
                break
            elif os.path.exists(choice):
                config['magec_model'] = choice
                break
            else:
                print("❌ 无效的选择或文件不存在，请重新输入")
                if choice and not choice.isdigit():
                    print("💡 提示: 可以输入完整的文件路径，或者输入序号选择上面列出的模型")
    else:
        print("⚠️ 未发现训练好的模型文件")
        print("💡 提示: 请先运行 demo.py 训练模型，或者手动输入模型路径")
        print()
        
        while True:
            model_path = input("请输入MAGEC模型路径: ").strip()
            if model_path.lower() == 'q':
                print("👋 退出程序")
                sys.exit(0)
            if model_path and os.path.exists(model_path):
                config['magec_model'] = model_path
                break
            else:
                print("❌ 文件不存在，请检查路径")
                print("💡 提示: 模型文件通常以 .pth 或 .pt 结尾")
    
    print(f"✅ 已选择模型: {config['magec_model']}")
    print()
    
    # 2. 输出目录配置
    print("📁 输出目录配置")
    print("-" * 50)
    default_output = f"results/evaluation_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = input(f"输出目录 [默认: {default_output}]: ").strip()
    config['output_dir'] = output_dir if output_dir else default_output
    print(f"✅ 输出目录: {config['output_dir']}")
    print()
    
    # 3. 测试环境配置
    print("🌍 测试环境配置")
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
    
    # 智能体数量
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
    
    print(f"✅ 测试环境: {config['graph_name']}, {config['num_agents']} agents")
    print()
    
    # 4. 测试参数配置
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
    
    # 5. 算法选择
    print("🤖 算法选择")
    print("-" * 50)
    available_algorithms = ['MAGEC', 'AHPA', 'SEBS', 'CBLS', 'Random']
    print("可选算法:")
    for i, alg in enumerate(available_algorithms, 1):
        print(f"  {i}. {alg}")
    
    print("请选择要测试的算法 (用空格分隔多个选择，如: 1 2 3)")
    while True:
        alg_input = input("[默认: 1 2 3 4 5 (全部)]: ").strip()
        if not alg_input:
            config['algorithms'] = available_algorithms.copy()
            break
        
        try:
            choices = [int(x) for x in alg_input.split()]
            if all(1 <= choice <= len(available_algorithms) for choice in choices):
                config['algorithms'] = [available_algorithms[i-1] for i in choices]
                break
            else:
                print("❌ 选择超出范围，请输入1-5")
        except ValueError:
            print("❌ 请输入有效的数字序列")
    
    print(f"✅ 选择算法: {', '.join(config['algorithms'])}")
    print()
    
    # 6. 测试场景
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
    
    # 7. 可视化选项
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
        config['animate_algorithm'] = 'MAGEC'
    
    print()
    
    # 8. 其他设置
    config['seed'] = 42
    
    # 显示最终配置
    print("📋 " + "=" * 76)
    print("📋 最终配置确认")
    print("📋 " + "=" * 76)
    print(f"🔹 模型路径: {config['magec_model']}")
    print(f"🔹 输出目录: {config['output_dir']}")
    print(f"🔹 测试环境: {config['graph_name']} ({config['num_agents']} agents)")
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
    # 检查是否需要显示帮助

    
    parser = argparse.ArgumentParser(
        description='MAGEC算法评估和可视化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python visualize.py                                    # 交互式模式
  python visualize.py --quick_test --animate             # 快速测试
  python visualize.py --magec_model model.pth           # 指定模型
  python visualize.py --batch --magec_model model.pth   # 批处理模式
        """
    )
    parser.add_argument('--magec_model', type=str, help='训练好的MAGEC模型路径')
    parser.add_argument('--graph_name', type=str, default='milwaukee',
                       choices=['milwaukee', '4nodes'], help='测试图类型')
    parser.add_argument('--num_agents', type=int, default=4, help='智能体数量')
    parser.add_argument('--num_episodes', type=int, default=50, help='每个算法的测试回合数')
    parser.add_argument('--episode_length', type=int, default=100, help='每回合步数')
    parser.add_argument('--output_dir', type=str, help='输出目录')
    parser.add_argument('--algorithms', nargs='+', 
                       default=['MAGEC', 'AHPA', 'SEBS', 'CBLS', 'Random'],
                       help='要测试的算法')
    parser.add_argument('--scenarios', nargs='+',
                       default=['normal', 'attrition', 'comm_loss', 'both'],
                       help='测试场景')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--quick_test', action='store_true', help='快速测试模式')
    parser.add_argument('--animate', action='store_true', help='生成动画可视化')
    parser.add_argument('--animate_algorithm', type=str, default='MAGEC',
                       help='要生成动画的算法')
    parser.add_argument('--interactive', action='store_true', help='交互式配置模式')
    parser.add_argument('--batch', action='store_true', help='批处理模式（使用命令行参数）')
    
    args = parser.parse_args()
    
    # 决定使用交互式还是命令行模式
    if args.batch:
        # 批处理模式：必须提供模型路径
        if not args.magec_model:
            print("❌ 批处理模式下必须指定 --magec_model 参数")
            sys.exit(1)
        
        config = {
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
        
    elif args.interactive or not args.magec_model:
        # 交互式模式：如果没有提供模型路径或者明确指定交互式
        config = interactive_input()
    else:
        # 使用命令行参数
        config = {
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
    
    # 设置随机种子
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    
    # 快速测试模式调整
    if config['quick_test']:
        config['num_episodes'] = 10
        config['episode_length'] = 50
        config['graph_name'] = '4nodes'
        config['num_agents'] = 2
        logger.info("快速测试模式")
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 环境配置
    env_config = {
        'graph_name': config['graph_name'],
        'num_agents': config['num_agents'],
        'observation_radius': 400.0,
        'max_cycles': config['episode_length'],
        'agent_speed': 40.0,
        'action_method': 'neighbors'
    }
    
    print("\n" + "🎯 " + "=" * 76)
    print("🎯 开始MAGEC算法评估")
    print("🎯 " + "=" * 76)
    print(f"📁 MAGEC模型: {config['magec_model']}")
    print(f"🌍 测试环境: {config['graph_name']} ({config['num_agents']} agents)")
    print(f"🤖 测试算法: {', '.join(config['algorithms'])}")
    print(f"🎭 测试场景: {', '.join(config['scenarios'])}")
    print(f"⚙️ 测试设置: {config['num_episodes']} episodes × {config['episode_length']} steps")
    print(f"📂 输出目录: {config['output_dir']}")
    print("🎯 " + "=" * 76)
    
    # 创建评估器
    evaluator = MAGECEvaluator(env_config)
    
    # 加载MAGEC模型
    if 'MAGEC' in config['algorithms']:
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
        env = OptimizedPatrollingEnvironment(**env_config)
    
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
        print("📊 评估结果摘要")
        print("📊 " + "=" * 76)
        
        summary = evaluator._generate_summary()
        for scenario, data in summary.items():
            print(f"\n🎭 {scenario.upper()} 场景:")
            print(f"  🏆 最佳奖励算法: {data['best_reward_algorithm']} ({data['best_reward_value']:.3f})")
            print(f"  ⚡ 最佳闲置算法: {data['best_idleness_algorithm']} ({data['best_idleness_value']:.3f})")
            
            if 'MAGEC' in data['magec_performance']:
                magec_perf = data['magec_performance']
                print(f"  🤖 MAGEC性能: 奖励={magec_perf['avg_reward']:.3f}, 闲置={magec_perf['avg_idleness']:.3f}")
        
        print("\n" + "🎉 " + "=" * 76)
        print("🎉 评估完成！")
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
        
# 快速开始示例
"""
🚀 快速开始示例:

1. 最简单的使用方式 (推荐新手):
   python visualize.py

2. 如果你已经有训练好的模型:
   python visualize.py --magec_model experiments/magec_official/final_model.pth

3. 快速测试 (只需几分钟):
   python visualize.py --quick_test --animate

4. 批量处理多个模型:
   for model in experiments/*/final_model.pth; do
       python visualize.py --batch --magec_model "$model" --output_dir "results/$(basename $(dirname $model))"
   done

5. 只测试特定算法和场景:
   python visualize.py --algorithms MAGEC AHPA --scenarios normal attrition --animate

💡 提示: 第一次使用建议选择交互式模式，程序会引导你完成所有配置！
"""