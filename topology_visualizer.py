"""
交互式拓扑结构可视化器
直接运行即可，通过问答选择文件和版本

使用方法：
python topology_visualizer.py

然后按提示选择文件和可视化类型
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional
import os
import glob
from pathlib import Path
import warnings
from matplotlib.patches import Circle, FancyBboxPatch, ConnectionPatch
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
warnings.filterwarnings('ignore')

# 设置现代化样式 - 修复matplotlib兼容性
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
        
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class InteractiveTopologyVisualizer:
    """交互式拓扑结构可视化器"""
    
    def __init__(self, json_file_path: str):
        self.json_file_path = json_file_path
        self.topology_data = None
        self.key_nodes = {}
        self.consolidated_paths = {}
        self.adjacency_matrix = None
        self.node_mapping = {}
        self.reverse_mapping = {}
        
        # 现代化颜色方案 - 基于Material Design
        self.color_scheme = {
            'primary': '#1976D2',      # 蓝色主色调
            'secondary': '#388E3C',    # 绿色次色调
            'accent': '#F57C00',       # 橙色强调色
            'error': '#D32F2F',        # 红色错误色
            'warning': '#F9A825',      # 黄色警告色
            'background': '#FAFAFA',   # 浅灰背景
            'surface': '#FFFFFF',      # 白色表面
            'on_primary': '#FFFFFF',   # 主色调上的文字
            'text_primary': '#212121', # 主要文字
            'text_secondary': '#757575' # 次要文字
        }
        
        # 节点类型颜色映射
        self.node_colors = {
            'start_endpoint': '#4CAF50',      # 绿色 - 起始点
            'end_endpoint': '#F44336',        # 红色 - 终点
            'other_endpoint': '#FF9800',      # 橙色 - 其他端点
            'primary_key': '#2196F3',         # 蓝色 - 主要关键节点
            'secondary_key': '#00BCD4',       # 青色 - 次要关键节点
            'service_key': '#9C27B0',         # 紫色 - 服务关键节点
            'intersection': '#E91E63'         # 粉色 - 交叉节点
        }
        
        # 加载和解析数据
        self.load_topology_data()
        self.parse_topology_structure()
        self.build_enhanced_adjacency_matrix()
    
    def load_topology_data(self):
        """加载拓扑数据"""
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                self.topology_data = json.load(f)
            print(f"✅ 成功加载拓扑数据: {self.json_file_path}")
        except Exception as e:
            raise Exception(f"加载数据失败: {e}")
    
    def parse_topology_structure(self):
        """解析拓扑结构"""
        self.key_nodes = self.topology_data.get('key_nodes_info', {})
        self.consolidated_paths = self.topology_data.get('consolidated_paths_info', {})
        
        # 创建增强的节点映射
        node_ids = list(self.key_nodes.keys())
        self.node_mapping = {node_id: idx for idx, node_id in enumerate(node_ids)}
        self.reverse_mapping = {idx: node_id for node_id, idx in self.node_mapping.items()}
        
        print(f"📍 解析完成: {len(self.key_nodes)} 个节点, {len(self.consolidated_paths)} 条路径")
    
    def build_enhanced_adjacency_matrix(self):
        """构建增强的邻接矩阵"""
        n_nodes = len(self.key_nodes)
        self.adjacency_matrix = np.zeros((n_nodes, n_nodes), dtype=float)
        
        # 构建加权邻接矩阵
        for path_id, path_info in self.consolidated_paths.items():
            key_nodes = path_info.get('key_nodes', [])
            quality = path_info.get('curve_quality_score', 0.7)
            
            for i in range(len(key_nodes) - 1):
                node1, node2 = key_nodes[i], key_nodes[i + 1]
                if node1 in self.node_mapping and node2 in self.node_mapping:
                    idx1, idx2 = self.node_mapping[node1], self.node_mapping[node2]
                    weight = quality * (1 + 0.5 * (len(key_nodes) - 2))  # 路径长度加权
                    self.adjacency_matrix[idx1, idx2] += weight
                    self.adjacency_matrix[idx2, idx1] += weight
    
    def create_interactive_plotly_visualization(self, save_path: Optional[str] = None):
        """创建交互式Plotly可视化 - 修复版"""
        print("🎨 创建现代交互式网络可视化...")
        
        # 使用NetworkX计算高质量布局
        G = self._build_networkx_graph()
        
        # 使用多种布局算法并选择最佳的
        layouts = self._compute_multiple_layouts(G)
        best_layout = self._select_best_layout(G, layouts)
        
        # 创建节点和边的traces
        node_trace, edge_trace, hover_info = self._create_plotly_traces(G, best_layout)
        
        # 创建图形 - 修复废弃属性
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=dict(
                               text='<b>智能拓扑网络结构图</b><br><sub>第一阶段：拓扑感知GNN架构构建结果</sub>',
                               x=0.5,
                               font=dict(size=20, color=self.color_scheme['text_primary'])
                           ),
                           # 移除废弃的titlefont_size属性
                           showlegend=True,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=80),
                           annotations=[ dict(
                               text="🖱️ 可拖拽节点 | 🔍 可缩放 | 📍 悬停查看详情",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color=self.color_scheme['text_secondary'], size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor=self.color_scheme['background'],
                           paper_bgcolor='white'
                       ))
        
        # 添加图例
        self._add_modern_legend(fig)
        
        # 添加统计信息面板
        self._add_stats_panel(fig, G)
        
        if save_path:
            pyo.plot(fig, filename=save_path, auto_open=True)
            print(f"💾 交互式可视化已保存并打开: {save_path}")
        else:
            fig.show()
        
        return fig
    
    def _build_networkx_graph(self):
        """构建NetworkX图"""
        G = nx.Graph()
        
        # 添加节点（带属性）
        for idx, (node_id, node_info) in enumerate(self.key_nodes.items()):
            G.add_node(idx,
                      node_id=node_id,
                      label=self._get_node_label(node_id, node_info),
                      color=self._get_node_color(node_info),
                      size=self._get_node_size(node_info),
                      importance=node_info.get('importance', 1.0),
                      is_endpoint=node_info.get('is_endpoint', False),
                      road_class=node_info.get('road_class', 'secondary'),
                      position=node_info.get('position', [0, 0, 0]))
        
        # 添加边（带权重）
        for i in range(len(self.adjacency_matrix)):
            for j in range(i + 1, len(self.adjacency_matrix)):
                weight = self.adjacency_matrix[i, j]
                if weight > 0:
                    G.add_edge(i, j, weight=weight)
        
        return G
    
    def _compute_multiple_layouts(self, G):
        """计算多种布局算法"""
        layouts = {}
        
        try:
            # 1. Spring布局（力导向）- 默认
            layouts['spring'] = nx.spring_layout(G, k=3, iterations=100, seed=42)
            
            # 2. Kamada-Kawai布局 - 适合中等大小的图
            if len(G) <= 100:
                layouts['kamada_kawai'] = nx.kamada_kawai_layout(G)
            
            # 3. 谱布局 - 基于特征向量
            if len(G) >= 3:
                try:
                    layouts['spectral'] = nx.spectral_layout(G)
                except:
                    print("⚠️ 谱布局计算失败，跳过")
            
            # 4. 圆形布局 - 规整但可能重叠较多
            layouts['circular'] = nx.circular_layout(G)
            
            # 5. Shell布局 - 层次结构
            shells = self._create_node_shells(G)
            if len(shells) > 1:
                layouts['shell'] = nx.shell_layout(G, shells)
            
        except Exception as e:
            print(f"⚠️ 布局计算警告: {e}")
            layouts['spring'] = nx.spring_layout(G, seed=42)  # 回退到基础布局
        
        return layouts
    
    def _create_node_shells(self, G):
        """创建节点壳层（用于shell布局）"""
        shells = []
        
        # 按重要性和类型分层
        endpoints = []
        high_importance = []
        medium_importance = []
        low_importance = []
        
        for node in G.nodes():
            node_id = self.reverse_mapping[node]
            node_info = self.key_nodes[node_id]
            importance = node_info.get('importance', 1.0)
            
            if node_info.get('is_endpoint', False):
                endpoints.append(node)
            elif importance >= 5.0:
                high_importance.append(node)
            elif importance >= 2.0:
                medium_importance.append(node)
            else:
                low_importance.append(node)
        
        # 构建壳层
        if endpoints:
            shells.append(endpoints)
        if high_importance:
            shells.append(high_importance)
        if medium_importance:
            shells.append(medium_importance)
        if low_importance:
            shells.append(low_importance)
        
        return shells if len(shells) > 1 else [list(G.nodes())]
    
    def _select_best_layout(self, G, layouts):
        """选择最佳布局算法"""
        if not layouts:
            return nx.spring_layout(G, seed=42)
        
        # 评估标准：节点分散度、边交叉最小化、美观度
        best_layout = None
        best_score = -float('inf')
        
        for layout_name, layout in layouts.items():
            score = self._evaluate_layout_quality(G, layout)
            if score > best_score:
                best_score = score
                best_layout = layout
        
        return best_layout if best_layout is not None else list(layouts.values())[0]
    
    def _evaluate_layout_quality(self, G, layout):
        """评估布局质量"""
        if not layout:
            return -float('inf')
        
        try:
            positions = np.array(list(layout.values()))
            
            # 1. 节点分散度（避免重叠）
            min_distance = float('inf')
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    min_distance = min(min_distance, dist)
            
            # 2. 边长度均匀性
            edge_lengths = []
            for edge in G.edges():
                pos1, pos2 = layout[edge[0]], layout[edge[1]]
                length = np.linalg.norm(np.array(pos1) - np.array(pos2))
                edge_lengths.append(length)
            
            edge_variance = np.var(edge_lengths) if edge_lengths else 0
            
            # 综合评分
            score = min_distance - 0.1 * edge_variance
            
            return score
        except Exception as e:
            print(f"⚠️ 布局质量评估错误: {e}")
            return 0
    
    def _create_plotly_traces(self, G, layout):
        """创建Plotly图形轨迹"""
        # 创建边轨迹
        edge_x, edge_y = [], []
        edge_weights = []
        
        for edge in G.edges():
            x0, y0 = layout[edge[0]]
            x1, y1 = layout[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(G.edges[edge].get('weight', 1.0))
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.5, color=self.color_scheme['text_secondary']),
            hoverinfo='none',
            mode='lines',
            opacity=0.6,
            name='连接'
        )
        
        # 创建节点轨迹
        node_x, node_y = [], []
        node_colors, node_sizes = [], []
        hover_texts = []
        node_labels = []
        
        for node in G.nodes():
            x, y = layout[node]
            node_x.append(x)
            node_y.append(y)
            
            node_data = G.nodes[node]
            node_colors.append(node_data['color'])
            node_sizes.append(node_data['size'])
            node_labels.append(node_data['label'])
            
            # 创建详细的悬停信息
            hover_text = self._create_hover_text(node, node_data)
            hover_texts.append(hover_text)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            hovertext=hover_texts,
            text=node_labels,
            textposition="middle center",
            textfont=dict(size=10, color='white'),
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white'),
                sizemode='diameter'
            ),
            name='节点'
        )
        
        return node_trace, edge_trace, hover_texts
    
    def _get_node_label(self, node_id: str, node_info: Dict) -> str:
        """获取节点标签 - 增强版：支持详细节点分类"""
        if node_info.get('is_endpoint', False):
            # 端点分类
            if 'start' in node_id.lower():
                return 'S'  # 起始点
            elif 'end' in node_id.lower():
                return 'E'  # 终点
            elif 'loading' in node_id.lower():
                return 'L'  # 装载点
            elif 'unloading' in node_id.lower():
                return 'U'  # 卸载点
            elif 'parking' in node_id.lower():
                return 'P'  # 停车点
            else:
                return 'EP' # 其他端点
        else:
            # 关键节点分类
            importance = node_info.get('importance', 1.0)
            road_class = node_info.get('road_class', 'secondary')
            node_type = node_info.get('node_type', 'key_node')
            
            # 根据重要性和道路等级确定标签
            if importance >= 5.0:
                return f'I{int(importance)}'  # 交叉节点 (Intersection)
            elif road_class == 'primary':
                return f'P{int(importance)}'  # 主干道关键节点 (Primary)
            elif road_class == 'secondary':
                return f'S{int(importance)}'  # 次干道关键节点 (Secondary)
            elif road_class == 'service':
                return f'V{int(importance)}'  # 服务道关键节点 (serVice)
            else:
                return f'K{int(importance)}'  # 通用关键节点 (Key)
    
    def _get_node_color(self, node_info: Dict) -> str:
        """获取节点颜色 - 增强版：支持详细节点分类"""
        if node_info.get('is_endpoint', False):
            # 端点颜色分类 - 更细致的区分
            node_id = str(node_info.get('node_id', ''))
            endpoint_type = node_info.get('endpoint_info', {}).get('type', '')
            
            if 'start' in node_id.lower() or endpoint_type == 'start':
                return '#4CAF50'  # 绿色 - 起始端点
            elif 'end' in node_id.lower() or endpoint_type == 'end':
                return '#F44336'  # 红色 - 终止端点
            elif 'loading' in node_id.lower() or endpoint_type == 'loading':
                return '#2196F3'  # 蓝色 - 装载端点
            elif 'unloading' in node_id.lower() or endpoint_type == 'unloading':
                return '#FF9800'  # 橙色 - 卸载端点
            elif 'parking' in node_id.lower() or endpoint_type == 'parking':
                return '#9C27B0'  # 紫色 - 停车端点
            else:
                return '#607D8B'  # 蓝灰色 - 其他端点
        else:
            # 关键节点颜色分类
            road_class = node_info.get('road_class', 'secondary')
            importance = node_info.get('importance', 1.0)
            path_count = len(node_info.get('path_memberships', []))
            is_intersection = path_count > 2
            
            # 交叉节点（高优先级）
            if is_intersection and importance >= 4.0:
                return '#E91E63'  # 粉红色 - 重要交叉节点
            elif is_intersection:
                return '#8E24AA'  # 紫色 - 普通交叉节点
            
            # 按道路等级分类
            elif road_class == 'primary':
                if importance >= 5.0:
                    return '#1976D2'  # 深蓝色 - 高重要性主干道
                else:
                    return '#2196F3'  # 蓝色 - 普通主干道
            elif road_class == 'secondary':
                if importance >= 3.0:
                    return '#388E3C'  # 深绿色 - 高重要性次干道
                else:
                    return '#4CAF50'  # 绿色 - 普通次干道
            elif road_class == 'service':
                if importance >= 2.0:
                    return '#F57C00'  # 深橙色 - 重要服务道
                else:
                    return '#FF9800'  # 橙色 - 普通服务道
            else:
                # 默认关键节点
                return '#757575'  # 灰色 - 未分类节点
    
    def _get_node_size(self, node_info: Dict) -> float:
        """获取节点大小 - 增强版：根据类型和重要性动态调整"""
        importance = node_info.get('importance', 1.0)
        road_class = node_info.get('road_class', 'secondary')
        path_count = len(node_info.get('path_memberships', []))
        is_intersection = path_count > 2
        
        if node_info.get('is_endpoint', False):
            # 端点基础大小
            base_size = 18
            
            # 根据端点类型调整
            endpoint_type = node_info.get('endpoint_info', {}).get('type', '')
            if endpoint_type in ['loading', 'unloading']:
                base_size = 20  # 装卸点稍大
            elif endpoint_type == 'parking':
                base_size = 16  # 停车点稍小
            
            return base_size
        else:
            # 关键节点大小计算
            base_size = 10
            
            # 根据道路等级调整基础大小
            if road_class == 'primary':
                base_size = 14
            elif road_class == 'secondary':
                base_size = 12
            elif road_class == 'service':
                base_size = 10
            
            # 根据重要性调整
            importance_bonus = min(importance * 2, 8)
            
            # 交叉节点奖励
            intersection_bonus = 4 if is_intersection else 0
            
            # 路径数量奖励
            path_bonus = min(path_count * 1.5, 6)
            
            final_size = base_size + importance_bonus + intersection_bonus + path_bonus
            
            # 限制最大最小值
            return max(8, min(final_size, 28))
    
    def _create_hover_text(self, node: int, node_data: Dict) -> str:
        """创建悬停文本 - 增强版：显示详细节点信息"""
        node_id = self.reverse_mapping[node]
        node_info = self.key_nodes[node_id]
        
        # 基本信息
        text_lines = [
            f"<b>节点 {node_data['label']}</b>",
            f"ID: {node_id}",
        ]
        
        # 节点类型信息
        if node_info.get('is_endpoint', False):
            endpoint_info = node_info.get('endpoint_info', {})
            endpoint_type = endpoint_info.get('type', 'unknown')
            text_lines.append(f"类型: 端点 ({endpoint_type.title()})")
            
            if 'paths' in endpoint_info:
                text_lines.append(f"连接路径: {len(endpoint_info['paths'])} 条")
        else:
            node_type = node_info.get('node_type', 'key_node')
            text_lines.append(f"类型: 关键节点 ({node_type.replace('_', ' ').title()})")
        
        # 重要性和等级
        importance = node_info.get('importance', 1.0)
        road_class = node_info.get('road_class', 'unknown')
        text_lines.append(f"重要性: {importance:.1f}")
        text_lines.append(f"道路等级: {road_class.title()}")
        
        # 路径成员信息
        path_memberships = node_info.get('path_memberships', [])
        if path_memberships:
            text_lines.append(f"参与路径: {len(path_memberships)} 条")
            if len(path_memberships) > 1:
                text_lines.append("🔀 交叉节点")
        
        # 位置信息
        position = node_info.get('position', [0, 0, 0])
        text_lines.append(f"位置: ({position[0]:.1f}, {position[1]:.1f})")
        
        # 容量和安全信息
        traffic_capacity = node_info.get('traffic_capacity', 0)
        if traffic_capacity > 0:
            text_lines.append(f"通行能力: {traffic_capacity} 车/小时")
        
        safety_rating = node_info.get('safety_rating', 0)
        if safety_rating > 0:
            text_lines.append(f"安全等级: {safety_rating:.1f}")
        
        # 聚类信息（如果是聚类生成的关键节点）
        cluster_info = node_info.get('cluster_info', {})
        if cluster_info:
            original_count = cluster_info.get('original_node_count', 0)
            if original_count > 0:
                text_lines.append(f"原始节点: {original_count} 个")
            
            if cluster_info.get('is_intersection', False):
                text_lines.append("🚦 重要交叉口")
        
        # 曲线拟合质量（如果有）
        curve_quality = node_info.get('curve_fitting_quality', 0)
        if curve_quality > 0:
            text_lines.append(f"拟合质量: {curve_quality:.2f}")
        
        dynamics_compliance = node_info.get('dynamics_compliance', True)
        if not dynamics_compliance:
            text_lines.append("⚠️ 动力学不合规")
        
        return "<br>".join(text_lines)
    
    def _add_modern_legend(self, fig):
        """添加现代化图例 - 增强版：详细节点分类"""
        # 端点图例
        endpoint_legend = [
            ("起始端点 (S)", '#4CAF50'),
            ("终止端点 (E)", '#F44336'),
            ("装载端点 (L)", '#2196F3'),
            ("卸载端点 (U)", '#FF9800'),
            ("停车端点 (P)", '#9C27B0'),
            ("其他端点", '#607D8B'),
        ]
        
        # 关键节点图例
        key_node_legend = [
            ("重要交叉节点 (I)", '#E91E63'),
            ("普通交叉节点", '#8E24AA'),
            ("主干道节点 (P)", '#2196F3'),
            ("次干道节点 (S)", '#4CAF50'),
            ("服务道节点 (V)", '#FF9800'),
            ("未分类节点", '#757575'),
        ]
        
        # 添加端点图例
        for i, (name, color) in enumerate(endpoint_legend):
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=12, color=color, line=dict(width=2, color='white')),
                name=name,
                showlegend=True,
                legendgroup="endpoints",
                legendgrouptitle_text="端点类型"
            ))
        
        # 添加关键节点图例
        for i, (name, color) in enumerate(key_node_legend):
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=12, color=color, line=dict(width=2, color='white')),
                name=name,
                showlegend=True,
                legendgroup="key_nodes",
                legendgrouptitle_text="关键节点类型"
            ))
    
    def _add_stats_panel(self, fig, G):
        """添加统计信息面板 - 增强版：详细节点分类统计"""
        # 计算基本网络统计
        try:
            basic_stats = {
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'density': nx.density(G),
                'components': nx.number_connected_components(G),
                'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
            }
            
            # 计算详细节点分类统计
            node_type_stats = {
                'endpoints': {'start': 0, 'end': 0, 'loading': 0, 'unloading': 0, 'parking': 0, 'other': 0},
                'key_nodes': {'primary': 0, 'secondary': 0, 'service': 0, 'intersection': 0, 'unclassified': 0},
                'road_class': {'primary': 0, 'secondary': 0, 'service': 0},
                'importance_levels': {'high': 0, 'medium': 0, 'low': 0}
            }
            
            for node in G.nodes():
                node_id = self.reverse_mapping[node]
                node_info = self.key_nodes[node_id]
                
                if node_info.get('is_endpoint', False):
                    # 端点分类
                    endpoint_type = node_info.get('endpoint_info', {}).get('type', 'other')
                    if 'start' in node_id.lower():
                        node_type_stats['endpoints']['start'] += 1
                    elif 'end' in node_id.lower():
                        node_type_stats['endpoints']['end'] += 1
                    elif endpoint_type in ['loading', 'unloading', 'parking']:
                        node_type_stats['endpoints'][endpoint_type] += 1
                    else:
                        node_type_stats['endpoints']['other'] += 1
                else:
                    # 关键节点分类
                    road_class = node_info.get('road_class', 'secondary')
                    importance = node_info.get('importance', 1.0)
                    path_count = len(node_info.get('path_memberships', []))
                    
                    # 道路等级统计
                    if road_class in node_type_stats['road_class']:
                        node_type_stats['road_class'][road_class] += 1
                    
                    # 交叉节点vs普通节点
                    if path_count > 2:
                        node_type_stats['key_nodes']['intersection'] += 1
                    elif road_class == 'primary':
                        node_type_stats['key_nodes']['primary'] += 1
                    elif road_class == 'secondary':
                        node_type_stats['key_nodes']['secondary'] += 1
                    elif road_class == 'service':
                        node_type_stats['key_nodes']['service'] += 1
                    else:
                        node_type_stats['key_nodes']['unclassified'] += 1
                    
                    # 重要性等级统计
                    if importance >= 5.0:
                        node_type_stats['importance_levels']['high'] += 1
                    elif importance >= 2.0:
                        node_type_stats['importance_levels']['medium'] += 1
                    else:
                        node_type_stats['importance_levels']['low'] += 1
            
            # 生成统计文本
            stats_text = f"<b>网络统计</b><br>"
            stats_text += f"节点数: {basic_stats['nodes']}<br>"
            stats_text += f"边数: {basic_stats['edges']}<br>"
            stats_text += f"网络密度: {basic_stats['density']:.3f}<br>"
            stats_text += f"连通分量: {basic_stats['components']}<br>"
            stats_text += f"平均度: {basic_stats['avg_degree']:.2f}<br><br>"
            
            # 端点统计
            endpoints = node_type_stats['endpoints']
            total_endpoints = sum(endpoints.values())
            if total_endpoints > 0:
                stats_text += f"<b>端点分布</b><br>"
                if endpoints['start'] > 0:
                    stats_text += f"起始点: {endpoints['start']}<br>"
                if endpoints['end'] > 0:
                    stats_text += f"终止点: {endpoints['end']}<br>"
                if endpoints['loading'] > 0:
                    stats_text += f"装载点: {endpoints['loading']}<br>"
                if endpoints['unloading'] > 0:
                    stats_text += f"卸载点: {endpoints['unloading']}<br>"
                if endpoints['parking'] > 0:
                    stats_text += f"停车点: {endpoints['parking']}<br>"
                if endpoints['other'] > 0:
                    stats_text += f"其他端点: {endpoints['other']}<br>"
                stats_text += "<br>"
            
            # 关键节点统计
            key_nodes = node_type_stats['key_nodes']
            total_key_nodes = sum(key_nodes.values())
            if total_key_nodes > 0:
                stats_text += f"<b>关键节点分布</b><br>"
                if key_nodes['intersection'] > 0:
                    stats_text += f"交叉节点: {key_nodes['intersection']}<br>"
                if key_nodes['primary'] > 0:
                    stats_text += f"主干道: {key_nodes['primary']}<br>"
                if key_nodes['secondary'] > 0:
                    stats_text += f"次干道: {key_nodes['secondary']}<br>"
                if key_nodes['service'] > 0:
                    stats_text += f"服务道: {key_nodes['service']}<br>"
                stats_text += "<br>"
            
            # 重要性分布
            importance = node_type_stats['importance_levels']
            total_importance = sum(importance.values())
            if total_importance > 0:
                stats_text += f"<b>重要性分布</b><br>"
                stats_text += f"高 (≥5.0): {importance['high']}<br>"
                stats_text += f"中 (2.0-4.9): {importance['medium']}<br>"
                stats_text += f"低 (<2.0): {importance['low']}<br>"
            
            fig.add_annotation(
                text=stats_text,
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                xanchor='left', yanchor='top',
                align='left',
                bordercolor=self.color_scheme['text_secondary'],
                borderwidth=1,
                bgcolor='rgba(255,255,255,0.9)',
                font=dict(size=10, color=self.color_scheme['text_primary'])
            )
        except Exception as e:
            print(f"⚠️ 统计面板添加失败: {e}")
            # 简化统计作为后备
            simple_stats = f"<b>网络统计</b><br>节点数: {G.number_of_nodes()}<br>边数: {G.number_of_edges()}"
            fig.add_annotation(
                text=simple_stats,
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                xanchor='left', yanchor='top',
                align='left',
                font=dict(size=12, color=self.color_scheme['text_primary'])
            )
    
    def create_layered_matplotlib_visualization(self, save_path: Optional[str] = None,
                                              figsize: Tuple[int, int] = (16, 12)):
        """创建分层的Matplotlib可视化"""
        print("🎨 创建分层网络结构图...")
        
        G = self._build_networkx_graph()
        
        # 创建图形和子图
        fig = plt.figure(figsize=figsize, facecolor='white')
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1])
        
        # 主网络图
        ax_main = fig.add_subplot(gs[0, 0])
        self._draw_layered_network(ax_main, G)
        
        # 统计面板
        ax_stats = fig.add_subplot(gs[0, 1])
        self._draw_statistics_panel(ax_stats, G)
        
        # 节点重要性分布
        ax_importance = fig.add_subplot(gs[1, 0])
        self._draw_importance_distribution(ax_importance)
        
        # 连接度分布
        ax_degree = fig.add_subplot(gs[1, 1])
        self._draw_degree_distribution(ax_degree, G)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"💾 分层可视化已保存: {save_path}")
        
        plt.show()
        return fig
    
    def _draw_layered_network(self, ax, G):
        """绘制分层网络"""
        # 使用增强的shell布局
        shells = self._create_enhanced_shells(G)
        pos = nx.shell_layout(G, shells, scale=2)
        
        # 绘制边（根据权重调整透明度和粗细）
        edges = G.edges()
        weights = [G[u][v].get('weight', 1.0) for u, v in edges]
        max_weight = max(weights) if weights else 1
        
        for (u, v), weight in zip(edges, weights):
            alpha = 0.3 + 0.7 * (weight / max_weight)
            width = 0.5 + 2.0 * (weight / max_weight)
            
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
                   color=self.color_scheme['text_secondary'], 
                   alpha=alpha, linewidth=width, zorder=1)
        
        # 绘制节点（分层着色）
        for shell_idx, shell in enumerate(shells):
            for node in shell:
                node_data = G.nodes[node]
                x, y = pos[node]
                
                # 绘制节点
                circle = Circle((x, y), node_data['size']/500, 
                              color=node_data['color'], 
                              ec='white', linewidth=2, zorder=3)
                ax.add_patch(circle)
                
                # 添加标签
                ax.text(x, y, node_data['label'], 
                       ha='center', va='center', 
                       fontsize=8, fontweight='bold', 
                       color='white', zorder=4)
        
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('智能拓扑网络结构 - 分层布局', 
                    fontsize=16, fontweight='bold', pad=20)
    
    def _create_enhanced_shells(self, G):
        """创建增强的壳层结构"""
        try:
            # 按连接度和重要性分层
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            
            shells = [[], [], [], []]  # 4层
            
            for node in G.nodes():
                node_id = self.reverse_mapping[node]
                node_info = self.key_nodes[node_id]
                
                # 计算综合重要性分数
                importance = node_info.get('importance', 1.0)
                degree_score = degree_centrality[node]
                betweenness_score = betweenness_centrality[node]
                
                combined_score = importance + degree_score * 10 + betweenness_score * 10
                
                # 分配到不同层
                if node_info.get('is_endpoint', False):
                    shells[0].append(node)  # 端点在最外层
                elif combined_score >= 8:
                    shells[1].append(node)  # 高重要性节点
                elif combined_score >= 4:
                    shells[2].append(node)  # 中等重要性节点
                else:
                    shells[3].append(node)  # 低重要性节点在中心
            
            # 移除空壳层
            return [shell for shell in shells if shell]
        except Exception as e:
            print(f"⚠️ 壳层创建失败: {e}")
            return [list(G.nodes())]
    
    def _draw_statistics_panel(self, ax, G):
        """绘制统计面板"""
        ax.axis('off')
        
        try:
            # 计算各种网络指标
            stats = {
                '节点总数': G.number_of_nodes(),
                '边总数': G.number_of_edges(),
                '网络密度': f"{nx.density(G):.3f}",
                '连通分量数': nx.number_connected_components(G),
                '平均聚类系数': f"{nx.average_clustering(G):.3f}",
                '网络直径': nx.diameter(G) if nx.is_connected(G) else 'N/A'
            }
            
            # 节点类型统计
            endpoint_count = sum(1 for node in G.nodes() 
                               if self.key_nodes[self.reverse_mapping[node]].get('is_endpoint', False))
            
            type_stats = {
                '端点数量': endpoint_count,
                '关键节点数': len(G) - endpoint_count,
                '平均度': f"{sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}" if G.number_of_nodes() > 0 else "0"
            }
            
            # 绘制统计信息
            y_pos = 0.95
            ax.text(0.05, y_pos, '📊 网络统计', fontsize=14, fontweight='bold', 
                   transform=ax.transAxes, color=self.color_scheme['text_primary'])
            
            y_pos -= 0.08
            for key, value in stats.items():
                ax.text(0.05, y_pos, f"{key}: {value}", fontsize=11,
                       transform=ax.transAxes, color=self.color_scheme['text_secondary'])
                y_pos -= 0.06
            
            y_pos -= 0.04
            ax.text(0.05, y_pos, '🔍 节点类型', fontsize=14, fontweight='bold',
                   transform=ax.transAxes, color=self.color_scheme['text_primary'])
            
            y_pos -= 0.08
            for key, value in type_stats.items():
                ax.text(0.05, y_pos, f"{key}: {value}", fontsize=11,
                       transform=ax.transAxes, color=self.color_scheme['text_secondary'])
                y_pos -= 0.06
        except Exception as e:
            ax.text(0.05, 0.5, f"统计信息计算错误: {e}", fontsize=12,
                   transform=ax.transAxes, color='red')
    
    def _draw_importance_distribution(self, ax):
        """绘制重要性分布"""
        try:
            importance_scores = [info.get('importance', 1.0) 
                               for info in self.key_nodes.values()]
            
            ax.hist(importance_scores, bins=15, alpha=0.7, 
                   color=self.color_scheme['primary'], edgecolor='white')
            ax.set_xlabel('重要性分数', fontsize=12)
            ax.set_ylabel('节点数量', fontsize=12)
            ax.set_title('节点重要性分布', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f"分布图生成错误: {e}", ha='center', va='center',
                   transform=ax.transAxes, color='red')
    
    def _draw_degree_distribution(self, ax, G):
        """绘制连接度分布"""
        try:
            degrees = [degree for node, degree in G.degree()]
            
            if degrees:
                ax.hist(degrees, bins=max(10, max(degrees)), alpha=0.7,
                       color=self.color_scheme['secondary'], edgecolor='white')
            ax.set_xlabel('节点度数', fontsize=12)
            ax.set_ylabel('节点数量', fontsize=12)
            ax.set_title('连接度分布', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f"度分布图生成错误: {e}", ha='center', va='center',
                   transform=ax.transAxes, color='red')
    
    def create_simple_test_visualization(self, save_path: Optional[str] = None):
        """创建简化测试版可视化 - 用于调试"""
        print("🧪 创建简化测试可视化...")
        
        try:
            G = self._build_networkx_graph()
            pos = nx.spring_layout(G, seed=42)
            
            # 简单的节点和边trace
            edge_x, edge_y = [], []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            node_x = [pos[node][0] for node in G.nodes()]
            node_y = [pos[node][1] for node in G.nodes()]
            
            fig = go.Figure()
            
            # 添加边
            fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', 
                                   line=dict(width=1, color='gray'),
                                   hoverinfo='none', showlegend=False))
            
            # 添加节点
            fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers',
                                   marker=dict(size=10, color='blue'),
                                   hoverinfo='skip', showlegend=False))
            
            fig.update_layout(
                title="拓扑网络测试图",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            if save_path:
                pyo.plot(fig, filename=save_path, auto_open=True)
                print(f"💾 测试可视化已保存并打开: {save_path}")
            else:
                fig.show()
                
            return fig
            
        except Exception as e:
            print(f"❌ 测试可视化失败: {e}")
            import traceback
            traceback.print_exc()
            return None


def find_json_files(directory="."):
    """查找目录中的JSON文件"""
    json_files = []
    patterns = ["*.json", "**/*.json"]
    
    for pattern in patterns:
        json_files.extend(glob.glob(os.path.join(directory, pattern), recursive=True))
    
    return sorted(set(json_files))


def interactive_file_selection():
    """交互式文件选择"""
    print("🔍 正在搜索JSON文件...")
    
    # 搜索当前目录和子目录中的JSON文件
    json_files = find_json_files(".")
    
    if not json_files:
        print("❌ 当前目录下未找到JSON文件")
        
        # 允许用户手动输入路径
        while True:
            file_path = input("请输入JSON文件的完整路径（或输入'q'退出）: ").strip()
            if file_path.lower() == 'q':
                print("👋 再见！")
                return None
            
            if os.path.exists(file_path) and file_path.endswith('.json'):
                return file_path
            else:
                print("❌ 文件不存在或不是JSON文件，请重新输入")
    
    print(f"\n📁 找到 {len(json_files)} 个JSON文件:")
    for i, file_path in enumerate(json_files, 1):
        file_size = os.path.getsize(file_path) / 1024  # KB
        print(f"  {i}. {file_path} ({file_size:.1f} KB)")
    
    while True:
        try:
            choice = input(f"\n请选择文件编号 (1-{len(json_files)}) 或输入完整路径: ").strip()
            
            # 如果是数字，选择对应文件
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(json_files):
                    return json_files[idx]
                else:
                    print(f"❌ 请输入 1 到 {len(json_files)} 之间的数字")
            
            # 如果是路径，验证文件
            elif os.path.exists(choice) and choice.endswith('.json'):
                return choice
            
            else:
                print("❌ 无效输入，请输入文件编号或有效的JSON文件路径")
                
        except KeyboardInterrupt:
            print("\n👋 再见！")
            return None


def interactive_visualization_selection():
    """交互式可视化类型选择 - 增强版：支持节点筛选"""
    print("\n🎨 请选择可视化类型:")
    options = {
        '1': ('简化测试版', '快速生成基础网络图，适合调试'),
        '2': ('完整交互版', '功能完整的交互式网络图，支持拖拽和缩放'),
        '3': ('分层静态版', '多面板分层分析图，适合学术报告'),
        '4': ('节点类型分析版', '专门展示详细节点分类的交互图'),
        '5': ('全部生成', '生成所有类型的可视化图')
    }
    
    for key, (name, desc) in options.items():
        print(f"  {key}. {name} - {desc}")
    
    while True:
        try:
            choice = input("\n请选择 (1-5): ").strip()
            if choice in options:
                return choice
            else:
                print("❌ 请输入 1, 2, 3, 4 或 5")
        except KeyboardInterrupt:
            print("\n👋 再见！")
            return None


def interactive_node_filter_selection():
    """交互式节点筛选选择"""
    print("\n🔍 节点显示筛选（可多选，用逗号分隔）:")
    filter_options = {
        '1': '显示所有节点',
        '2': '仅显示端点',
        '3': '仅显示关键节点',
        '4': '仅显示交叉节点',
        '5': '仅显示主干道节点',
        '6': '仅显示高重要性节点 (≥5.0)',
        '7': '自定义筛选'
    }
    
    for key, desc in filter_options.items():
        print(f"  {key}. {desc}")
    
    choice = input("\n请选择 (默认1-显示全部): ").strip()
    return choice if choice else '1'


def interactive_output_selection():
    """交互式输出目录选择"""
    default_dir = "./visualization_output"
    
    print(f"\n📁 输出目录设置:")
    print(f"  默认目录: {default_dir}")
    
    choice = input("使用默认目录？(y/n) 或输入自定义路径: ").strip()
    
    if choice.lower() in ['y', 'yes', '']:
        return default_dir
    elif choice.lower() in ['n', 'no']:
        custom_dir = input("请输入自定义输出目录: ").strip()
        return custom_dir if custom_dir else default_dir
    else:
        return choice  # 直接作为路径使用


def main():
    """交互式主函数"""
    print("🎯 智能拓扑网络可视化器")
    print("=" * 50)
    
    try:
        # 1. 文件选择
        json_file = interactive_file_selection()
        if not json_file:
            return
        
        print(f"✅ 选择的文件: {json_file}")
        
        # 2. 可视化类型选择
        viz_type = interactive_visualization_selection()
        if not viz_type:
            return
        
        # 3. 输出目录选择
        output_dir = interactive_output_selection()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 输出目录: {output_dir}")
        
        # 4. 创建可视化器
        print("\n🚀 正在初始化可视化器...")
        try:
            visualizer = InteractiveTopologyVisualizer(json_file)
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            print("💡 请检查JSON文件格式是否正确")
            return
        
        # 5. 生成可视化
        input_name = Path(json_file).stem
        
        print("\n🎨 开始生成可视化...")
        
        if viz_type == '1':
            # 简化测试版
            test_path = output_path / f"{input_name}_test.html"
            visualizer.create_simple_test_visualization(str(test_path))
            
        elif viz_type == '2':
            # 完整交互版
            interactive_path = output_path / f"{input_name}_interactive.html"
            visualizer.create_interactive_plotly_visualization(str(interactive_path))
            
        elif viz_type == '3':
            # 分层静态版
            layered_path = output_path / f"{input_name}_layered.png"
            visualizer.create_layered_matplotlib_visualization(str(layered_path))
            
        elif viz_type == '4':
            # 节点类型分析版
            print("\n📊 创建节点类型分析版可视化...")
            
            # 节点筛选选择
            filter_choice = interactive_node_filter_selection()
            
            # 应用筛选逻辑 (这里可以扩展)
            analysis_path = output_path / f"{input_name}_node_analysis.html"
            
            # 创建带有详细统计的交互版
            fig = visualizer.create_interactive_plotly_visualization(str(analysis_path))
            
            print(f"✅ 节点类型分析版已生成: {analysis_path}")
            
        elif viz_type == '5':
            # 全部生成
            print("📊 生成简化测试版...")
            test_path = output_path / f"{input_name}_test.html"
            visualizer.create_simple_test_visualization(str(test_path))
            
            print("📊 生成完整交互版...")
            interactive_path = output_path / f"{input_name}_interactive.html"
            visualizer.create_interactive_plotly_visualization(str(interactive_path))
            
            print("📊 生成分层静态版...")
            layered_path = output_path / f"{input_name}_layered.png"
            visualizer.create_layered_matplotlib_visualization(str(layered_path))
            
            print("📊 生成节点类型分析版...")
            analysis_path = output_path / f"{input_name}_node_analysis.html"
            visualizer.create_interactive_plotly_visualization(str(analysis_path))
        
        print(f"\n🎉 可视化生成完成！")
        print(f"📁 输出目录: {output_dir}")
        print(f"📄 生成的文件可以在浏览器中打开查看")
        
    except KeyboardInterrupt:
        print("\n👋 程序被用户中断，再见！")
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()