"""
物流系统可视化器 - logistics_visualizer.py
配合demo_GNN.py使用的专用可视化系统

功能特性：
1. ✅ 平滑车辆移动动画
2. ✅ 装载点/卸载点清晰标识
3. ✅ 车辆状态实时显示
4. ✅ 任务阶段可视化
5. ✅ 冲突高亮显示
6. ✅ 实时统计面板
7. ✅ 可保存GIF动画

使用方法：
1. 确保demo_GNN.py在同一目录
2. 运行 python logistics_visualizer.py
3. 选择是否保存GIF
"""

import numpy as np
import networkx as nx
import torch
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import random
import time
from typing import Dict, List, Tuple, Optional
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
import math

# 导入您保存的demo_GNN.py中的组件
try:
    from demo_GNN import (
        StableLogisticsEnvironment, 
        StableLogisticsGNN, 
        StableLogisticsTrainer,
        create_stable_logistics_graph,
        TaskPhase
    )
    print("✅ 成功导入demo_GNN.py中的组件")
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保demo_GNN.py在当前目录下")
    exit(1)

# 设置中文字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 专用可视化器
# ============================================================================

class LogisticsAnimationVisualizer:
    """物流系统动画可视化器"""
    
    def __init__(self, env, model, figsize=(20, 14)):
        self.env = env
        self.model = model
        self.figsize = figsize
        
        # 颜色方案
        self.vehicle_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#FF8C69', '#98FB98']
        self.node_color = '#E8F6F3'
        self.edge_color = '#BDC3C7'
        self.pickup_color = '#E74C3C'
        self.delivery_color = '#27AE60'
        self.conflict_color = '#F39C12'
        self.road_color = '#34495E'
        
        # 图布局优化
        self.pos = self._create_optimized_layout()
        
        # 动画数据
        self.animation_frames = []
        
        # 移动参数
        self.move_frames_per_step = 6  # 每步分解为6帧
        self.vehicle_size = (0.3, 0.2)  # 车辆矩形大小
    
    def _create_optimized_layout(self):
        """创建优化的图布局"""
        # 使用网格布局作为基础
        pos = {}
        
        # 如果是20节点的4x5网格
        if self.env.graph.number_of_nodes() == 20:
            rows, cols = 4, 5
            for i in range(rows):
                for j in range(cols):
                    node = i * cols + j
                    if node < self.env.graph.number_of_nodes():
                        pos[node] = (j * 2.5, (rows - 1 - i) * 2.0)  # 翻转Y轴，让(0,0)在左下角
        else:
            # 对于其他大小的图，使用spring布局
            pos = nx.spring_layout(self.env.graph, seed=42, k=2.0, iterations=100)
            # 扩展布局
            for node in pos:
                pos[node] = (pos[node][0] * 5, pos[node][1] * 5)
        
        return pos
    
    def visualize_logistics_process(self, max_steps=50, save_gif=False, gif_filename="logistics_animation.gif"):
        """可视化物流过程"""
        print(f"🎬 开始物流动画可视化")
        print(f"📊 图结构: {self.env.graph.number_of_nodes()}节点, {self.env.graph.number_of_edges()}边")
        
        # 收集基础数据
        obs = self.env.reset()
        base_steps = []
        
        print("📊 收集动画数据...")
        
        for step in range(max_steps):
            # 记录当前状态
            step_data = {
                'step': step,
                'vehicles': [
                    {
                        'id': v.vehicle_id,
                        'current': v.current_node,
                        'pickup': v.pickup_node,
                        'delivery': v.delivery_node,
                        'target': v.get_current_target(),
                        'phase': v.task_phase.value,
                        'cycles': v.cycles_completed,
                        'stuck': v.is_stuck(step),
                        'progress': v.steps_without_progress
                    }
                    for v in obs['vehicles']
                ],
                'conflicts': [],
                'stats': self.env.get_stats().copy()
            }
            
            # 模型推理
            with torch.no_grad():
                neighbor_scores = self.model(
                    obs['node_features'],
                    obs['edge_index'],
                    obs['vehicles'],
                    self.env.graph
                )
            
            # 动作选择
            actions = {}
            for i, vehicle in enumerate(obs['vehicles']):
                neighbors = list(self.env.graph.neighbors(vehicle.current_node))
                if neighbors:
                    if vehicle.is_stuck(step):
                        target = vehicle.get_current_target()
                        best_neighbor = self.env._get_shortest_path_next(vehicle, neighbors, target)
                        action = neighbors.index(best_neighbor) if best_neighbor in neighbors else 0
                    else:
                        valid_scores = neighbor_scores[i][:len(neighbors)]
                        action = torch.argmax(valid_scores).item()
                else:
                    action = 0
                actions[vehicle.vehicle_id] = action
            
            # 计算移动计划
            move_plans = self.env._compute_moves(actions)
            conflicts = self.env._detect_conflicts(move_plans)
            step_data['conflicts'] = conflicts
            step_data['move_plans'] = move_plans
            
            base_steps.append(step_data)
            
            # 执行步骤
            obs, rewards, done, all_done, info = self.env.step(actions)
            
            if all_done:
                break
        
        print(f"✅ 收集完成，共 {len(base_steps)} 步")
        
        # 生成平滑移动帧
        print("🎨 生成平滑移动帧...")
        self._generate_smooth_frames(base_steps)
        
        print(f"✨ 动画帧生成完成，共 {len(self.animation_frames)} 帧")
        
        # 创建动画
        self._create_animation(save_gif, gif_filename)
    
    def _generate_smooth_frames(self, base_steps):
        """生成平滑移动帧"""
        self.animation_frames = []
        
        for step_idx, step_data in enumerate(base_steps):
            vehicles = step_data['vehicles']
            move_plans = step_data.get('move_plans', {})
            
            # 为每个移动步骤生成多个中间帧
            for frame_idx in range(self.move_frames_per_step):
                frame_data = {
                    'step': step_data['step'],
                    'frame': frame_idx,
                    'total_frames': self.move_frames_per_step,
                    'progress': frame_idx / (self.move_frames_per_step - 1) if self.move_frames_per_step > 1 else 1.0,
                    'vehicles': [],
                    'conflicts': step_data['conflicts'],
                    'stats': step_data['stats']
                }
                
                # 为每个车辆计算当前位置
                for vehicle in vehicles:
                    vehicle_data = vehicle.copy()
                    
                    # 计算车辆在移动过程中的位置
                    if vehicle['id'] in move_plans:
                        start_node = vehicle['current']
                        end_node = move_plans[vehicle['id']]
                        
                        if start_node != end_node:
                            # 车辆正在移动
                            start_pos = self.pos[start_node]
                            end_pos = self.pos[end_node]
                            
                            # 线性插值计算中间位置
                            t = frame_data['progress']
                            current_x = start_pos[0] + t * (end_pos[0] - start_pos[0])
                            current_y = start_pos[1] + t * (end_pos[1] - start_pos[1])
                            
                            vehicle_data['display_pos'] = (current_x, current_y)
                            vehicle_data['is_moving'] = True
                            vehicle_data['move_direction'] = self._calculate_direction(start_pos, end_pos)
                            
                            # 最后一帧时更新目标节点
                            if frame_idx == self.move_frames_per_step - 1:
                                vehicle_data['current'] = end_node
                        else:
                            # 车辆未移动
                            vehicle_data['display_pos'] = self.pos[start_node]
                            vehicle_data['is_moving'] = False
                            vehicle_data['move_direction'] = 0
                    else:
                        # 没有移动计划
                        vehicle_data['display_pos'] = self.pos[vehicle['current']]
                        vehicle_data['is_moving'] = False
                        vehicle_data['move_direction'] = 0
                    
                    frame_data['vehicles'].append(vehicle_data)
                
                self.animation_frames.append(frame_data)
    
    def _calculate_direction(self, start_pos, end_pos):
        """计算移动方向（角度）"""
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        if dx == 0 and dy == 0:
            return 0
        angle = math.atan2(dy, dx)
        return math.degrees(angle)
    
    def _create_animation(self, save_gif=False, gif_filename="logistics_animation.gif"):
        """创建动画"""
        fig, (ax_main, ax_info) = plt.subplots(1, 2, figsize=self.figsize,
                                               gridspec_kw={'width_ratios': [3, 1]})
        
        def animate(frame_idx):
            ax_main.clear()
            ax_info.clear()
            
            if frame_idx >= len(self.animation_frames):
                return
            
            frame_data = self.animation_frames[frame_idx]
            self._draw_logistics_scene(ax_main, frame_data)
            self._draw_info_panel(ax_info, frame_data)
        
        # 动画参数
        total_frames = len(self.animation_frames)
        interval = 150  # 毫秒/帧
        
        print(f"🎬 开始播放动画，共{total_frames}帧...")
        
        anim = animation.FuncAnimation(fig, animate, frames=total_frames,
                                     interval=interval, repeat=True, blit=False)
        
        if save_gif:
            print(f"💾 保存GIF动画: {gif_filename}")
            anim.save(gif_filename, writer='pillow', fps=6)
            print("✅ GIF保存完成！")
        
        plt.tight_layout()
        plt.show()
        return anim
    
    def _draw_logistics_scene(self, ax, frame_data):
        """绘制物流场景"""
        step = frame_data['step']
        frame = frame_data['frame']
        vehicles = frame_data['vehicles']
        conflicts = frame_data['conflicts']
        stats = frame_data['stats']
        
        # 绘制道路网络
        for edge in self.env.graph.edges():
            x_vals = [self.pos[edge[0]][0], self.pos[edge[1]][0]]
            y_vals = [self.pos[edge[0]][1], self.pos[edge[1]][1]]
            # 道路效果
            ax.plot(x_vals, y_vals, color=self.road_color, linewidth=5, alpha=0.3, zorder=1)
            ax.plot(x_vals, y_vals, color=self.edge_color, linewidth=2.5, alpha=0.8, zorder=2)
        
        # 绘制普通节点
        for node in self.env.graph.nodes():
            if node not in self.env.pickup_nodes and node not in self.env.delivery_nodes:
                x, y = self.pos[node]
                circle = Circle((x, y), 0.15, color=self.node_color, ec='black', linewidth=1.5, zorder=3)
                ax.add_patch(circle)
                ax.text(x, y, str(node), ha='center', va='center', fontsize=8, fontweight='bold', zorder=4)
        
        # 绘制装载点（红色方块）
        for i, node in enumerate(self.env.pickup_nodes):
            x, y = self.pos[node]
            square = Rectangle((x-0.25, y-0.25), 0.5, 0.5, color=self.pickup_color, 
                             ec='black', linewidth=2.5, zorder=3)
            ax.add_patch(square)
            ax.text(x, y, str(node), ha='center', va='center', fontsize=9, 
                   color='white', fontweight='bold', zorder=4)
            ax.text(x, y-0.5, f'P{i}', ha='center', va='center', fontsize=10, 
                   color=self.pickup_color, fontweight='bold', zorder=4)
        
        # 绘制卸载点（绿色圆圈）
        for i, node in enumerate(self.env.delivery_nodes):
            x, y = self.pos[node]
            circle = Circle((x, y), 0.25, color=self.delivery_color, ec='black', linewidth=2.5, zorder=3)
            ax.add_patch(circle)
            ax.text(x, y, str(node), ha='center', va='center', fontsize=9, 
                   color='white', fontweight='bold', zorder=4)
            ax.text(x, y-0.5, f'D{i}', ha='center', va='center', fontsize=10, 
                   color=self.delivery_color, fontweight='bold', zorder=4)
        
        # 冲突车辆集合
        conflict_vehicles = set()
        for conflict in conflicts:
            conflict_vehicles.update(conflict)
        
        # 绘制车辆
        for vehicle in vehicles:
            vid = vehicle['id']
            display_pos = vehicle['display_pos']
            color = self.vehicle_colors[vid % len(self.vehicle_colors)]
            
            # 绘制到目标的连线（如果不在移动中）
            if not vehicle.get('is_moving', False):
                target_pos = self.pos[vehicle['target']]
                ax.plot([display_pos[0], target_pos[0]], [display_pos[1], target_pos[1]], 
                       color=color, linewidth=2, alpha=0.4, linestyle=':', zorder=2)
            
            # 车辆状态判断
            if vehicle['cycles'] >= 1:
                edge_color = 'green'
                edge_width = 4
                status_text = 'OK'
            elif vehicle['stuck']:
                edge_color = 'red'
                edge_width = 4
                status_text = '!'
            elif vid in conflict_vehicles:
                edge_color = 'orange'
                edge_width = 4
                status_text = 'X'
            else:
                edge_color = 'black'
                edge_width = 2
                status_text = str(vid)
            
            # 绘制车辆矩形
            angle = vehicle.get('move_direction', 0)
            self._draw_rotated_vehicle(ax, display_pos, angle, color, edge_color, edge_width)
            
            # 车辆内文字
            ax.text(display_pos[0], display_pos[1], status_text, ha='center', va='center', 
                   fontsize=9, fontweight='bold', color='white', zorder=6)
            
            # 车辆标签
            phase_text = "→P" if vehicle['phase'] == 'to_pickup' else "→D"
            label_color = color if not vehicle.get('is_moving', False) else 'navy'
            
            ax.text(display_pos[0], display_pos[1]+0.4, f'V{vid}{phase_text}', 
                   ha='center', va='center', fontsize=9, fontweight='bold', 
                   color=label_color, zorder=6)
            
            # 循环计数
            if vehicle['cycles'] > 0:
                ax.text(display_pos[0], display_pos[1]-0.4, f'×{vehicle["cycles"]}', 
                       ha='center', va='center', fontsize=8, color='green', 
                       fontweight='bold', zorder=6)
            
            # 移动轨迹效果
            if vehicle.get('is_moving', False) and frame_data['progress'] > 0.2:
                # 添加运动轨迹
                prev_progress = max(0, frame_data['progress'] - 0.4)
                if 'target' in vehicle:
                    try:
                        start_pos = self.pos[vehicle['current']]
                        end_pos = self.pos[vehicle['target']]
                        
                        prev_x = start_pos[0] + prev_progress * (end_pos[0] - start_pos[0])
                        prev_y = start_pos[1] + prev_progress * (end_pos[1] - start_pos[1])
                        
                        ax.plot([prev_x, display_pos[0]], [prev_y, display_pos[1]], 
                               color=color, linewidth=3, alpha=0.4, zorder=1)
                    except:
                        pass
        
        # 设置标题
        completion_rate = stats['completion_rate'] * 100
        title = f'稳定物流GNN系统 - 步骤 {step + 1}'
        if frame > 0:
            title += f' (移动 {frame}/{frame_data["total_frames"]})'
        title += f'\n完成率: {completion_rate:.1f}% | 总循环: {stats["total_cycles"]} | 冲突: {len(conflicts)}'
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # 设置坐标范围
        x_coords = [pos[0] for pos in self.pos.values()]
        y_coords = [pos[1] for pos in self.pos.values()]
        margin = 0.8
        ax.set_xlim(min(x_coords)-margin, max(x_coords)+margin)
        ax.set_ylim(min(y_coords)-margin, max(y_coords)+margin)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 图例
        legend_text = "红方块=装载点 绿圆=卸载点 彩色矩形=车辆\n绿框=完成 红框=卡住 橙框=冲突 虚线=目标连线"
        ax.text(0.02, 0.02, legend_text, transform=ax.transAxes, fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.9))
    
    def _draw_rotated_vehicle(self, ax, center, angle_deg, fill_color, edge_color, edge_width):
        """绘制旋转的车辆矩形"""
        x, y = center
        width, height = self.vehicle_size
        
        # 创建矩形点
        rect_points = np.array([
            [-width/2, -height/2],
            [width/2, -height/2],
            [width/2, height/2],
            [-width/2, height/2],
            [-width/2, -height/2]
        ])
        
        # 旋转
        angle_rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        rotated_points = rect_points @ rotation_matrix.T
        rotated_points[:, 0] += x
        rotated_points[:, 1] += y
        
        # 绘制
        vehicle_poly = plt.Polygon(rotated_points[:-1], facecolor=fill_color, 
                                 edgecolor=edge_color, linewidth=edge_width, zorder=5)
        ax.add_patch(vehicle_poly)
        
        return vehicle_poly
    
    def _draw_info_panel(self, ax, frame_data):
        """绘制信息面板"""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        step = frame_data['step']
        frame = frame_data['frame']
        vehicles = frame_data['vehicles']
        conflicts = frame_data['conflicts']
        stats = frame_data['stats']
        
        info_lines = []
        info_lines.append("📊 物流系统状态")
        info_lines.append("="*18)
        info_lines.append(f"步骤: {step + 1}")
        if frame > 0:
            info_lines.append(f"移动帧: {frame}/{frame_data['total_frames']}")
        info_lines.append(f"完成车辆: {stats['completed_vehicles']}/{len(vehicles)}")
        info_lines.append(f"完成率: {stats['completion_rate']*100:.1f}%")
        info_lines.append(f"总循环数: {stats['total_cycles']}")
        info_lines.append(f"平均循环: {stats['avg_cycles_per_vehicle']:.1f}")
        info_lines.append(f"当前冲突: {len(conflicts)}")
        info_lines.append(f"总冲突: {stats['total_conflicts']}")
        info_lines.append(f"卡住车辆: {stats['vehicles_stuck']}")
        info_lines.append("")
        
        info_lines.append("🚛 车辆详情")
        info_lines.append("="*18)
        
        moving_count = sum(1 for v in vehicles if v.get('is_moving', False))
        if moving_count > 0:
            info_lines.append(f"移动中: {moving_count}辆")
            info_lines.append("")
        
        for vehicle in vehicles:
            phase = "装载中" if vehicle['phase'] == 'to_pickup' else "配送中"
            status = ""
            if vehicle['cycles'] >= 1:
                status = " ✓"
            elif vehicle['stuck']:
                status = " (!)"
            
            moving_status = " [行驶中]" if vehicle.get('is_moving', False) else ""
            
            info_lines.append(f"车辆{vehicle['id']}: {phase}{status}{moving_status}")
            info_lines.append(f"  位置: {vehicle['current']}")
            info_lines.append(f"  目标: P{vehicle['pickup']} → D{vehicle['delivery']}")
            info_lines.append(f"  循环: {vehicle['cycles']}次")
            if vehicle['progress'] > 0:
                info_lines.append(f"  停滞: {vehicle['progress']}步")
            info_lines.append("")
        
        if conflicts:
            info_lines.append("⚠️ 冲突详情")
            info_lines.append("="*18)
            for conflict in conflicts:
                info_lines.append(f"车辆{conflict[0]} <-> 车辆{conflict[1]}")
        
        # 显示信息
        y_pos = 0.98
        for line in info_lines:
            if line.startswith("="):
                ax.text(0.05, y_pos, line, fontsize=8, color='gray', 
                       transform=ax.transAxes, family='monospace')
            elif line.startswith(("📊", "🚛", "⚠️")):
                ax.text(0.05, y_pos, line, fontsize=11, fontweight='bold',
                       transform=ax.transAxes)
            else:
                ax.text(0.05, y_pos, line, fontsize=9, transform=ax.transAxes)
            y_pos -= 0.04

# ============================================================================
# 主可视化程序
# ============================================================================

def main_visualization():
    """主可视化程序"""
    print("🎬" * 30)
    print("         物流系统动画可视化")
    print("🎬" * 30)
    print("🌟 可视化特性:")
    print("  ✅ 平滑车辆移动动画")
    print("  ✅ 装载点/卸载点标识")
    print("  ✅ 车辆状态实时显示")
    print("  ✅ 冲突高亮效果")
    print("  ✅ 任务阶段可视化")
    print("  ✅ 实时统计面板")
    print()
    
    # 创建系统
    graph = create_stable_logistics_graph()
    env = StableLogisticsEnvironment(graph, num_vehicles=4, max_steps=60)
    model = StableLogisticsGNN(node_features=8, hidden_dim=64, max_degree=10)
    
    print(f"📊 系统配置:")
    print(f"  图结构: {graph.number_of_nodes()}节点, {graph.number_of_edges()}边")
    print(f"  车辆数: {env.num_vehicles}")
    print(f"  装载点: {env.pickup_nodes}")
    print(f"  卸载点: {env.delivery_nodes}")
    
    # 快速训练模型
    print("\n🚀 快速训练模型以获得智能行为...")
    env.debug_mode = False  # 训练时关闭详细输出
    trainer = StableLogisticsTrainer(model, env)
    trainer.train(num_episodes=15)
    
    print(f"✅ 训练完成!")
    
    # 创建可视化器
    visualizer = LogisticsAnimationVisualizer(env, model)
    
    # 用户选择
    print("\n🎥 可视化选项:")
    user_choice = input("是否观看物流动画? (y/n): ").lower()
    
    if user_choice == 'y':
        save_gif = input("是否保存为GIF动画? (y/n): ").lower() == 'y'
        gif_name = "logistics_demo.gif"
        
        if save_gif:
            custom_name = input(f"GIF文件名 (默认: {gif_name}): ").strip()
            if custom_name:
                gif_name = custom_name if custom_name.endswith('.gif') else custom_name + '.gif'
        
        print(f"\n🎬 开始生成动画...")
        
        # 创建新的环境用于可视化
        vis_env = StableLogisticsEnvironment(graph, num_vehicles=4, max_steps=40)
        vis_env.debug_mode = False
        vis_visualizer = LogisticsAnimationVisualizer(vis_env, model)
        
        # 开始可视化
        vis_visualizer.visualize_logistics_process(
            max_steps=35, 
            save_gif=save_gif, 
            gif_filename=gif_name
        )
        
        print("\n🎉 可视化完成!")
        print("🎭 动画说明:")
        print("  🟥 P = 装载点")
        print("  🟢 D = 卸载点") 
        print("  🔲 彩色矩形 = 车辆")
        print("  →P = 前往装载点")
        print("  →D = 前往卸载点")
        print("  绿框 = 已完成循环")
        print("  红框 = 卡住车辆")
        print("  橙框 = 冲突车辆")
        print("  虚线 = 到目标连线")
        print("  ×数字 = 完成循环次数")
        
        if save_gif:
            print(f"💾 GIF已保存为: {gif_name}")
    
    else:
        print("👋 可视化已跳过")
    
    print(f"\n✨ 程序结束!")

if __name__ == "__main__":
    main_visualization()