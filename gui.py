"""
完整集成的智能拓扑构建GUI - 修复版本
集成增强版ClothoidCubic和EnhancedNodeClusteringConsolidator
支持聚类前后的完整可视化流程
"""

import sys
import os
import math
import time
import json
import threading
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QProgressBar, QTextEdit, QFileDialog, QMessageBox, QSplitter,
    QGroupBox, QGridLayout, QTableWidget, QTableWidgetItem,
    QGraphicsScene, QGraphicsView, QGraphicsEllipseItem,
    QGraphicsRectItem, QGraphicsPathItem, QTabWidget, QFrame,
    QCheckBox, QTreeWidget, QTreeWidgetItem, QListWidget, 
    QGraphicsLineItem, QGraphicsTextItem, QListWidgetItem
)
from PyQt5.QtGui import (
    QPen, QBrush, QColor, QPainter, QPainterPath, QFont, QPixmap, QWheelEvent
)

# 导入系统组件
try:
    from environment import OptimizedOpenPitMineEnv
    from optimized_backbone_network import OptimizedBackboneNetwork
    from optimized_planner_config import EnhancedPathPlannerWithConfig
    
    COMPONENTS_AVAILABLE = True
    print("✅ 系统组件加载成功")
except ImportError as e:
    print(f"❌ 系统组件加载失败: {e}")
    COMPONENTS_AVAILABLE = False
    sys.exit(1)

class Stage1ProgressWidget(QWidget):
    """第一阶段进度控制组件 - 集成增强版"""
    
    step_executed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.backbone_network = None
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        
        # 标题 - 正常字体
        title_label = QLabel("第一阶段：智能拓扑构建")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("", 12, QFont.Bold))
        layout.addWidget(title_label)
        
        # 总体进度
        self.overall_progress = QProgressBar()
        self.overall_progress.setMaximum(5)
        self.overall_progress.setValue(0)
        self.overall_progress.setTextVisible(True)
        self.overall_progress.setFormat("进度: %v/5 步骤")
        layout.addWidget(self.overall_progress)
        
        # 五个核心步骤
        self.step_widgets = {}
        self.create_step_widgets(layout)
        
        # 控制按钮
        control_layout = QHBoxLayout()
        
        self.start_all_btn = QPushButton("执行完整构建")
        self.start_all_btn.clicked.connect(self.start_full_construction)
        
        self.reset_btn = QPushButton("重置")
        self.reset_btn.clicked.connect(self.reset_progress)
        
        control_layout.addWidget(self.start_all_btn)
        control_layout.addWidget(self.reset_btn)
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # 状态显示
        self.status_label = QLabel("就绪")
        layout.addWidget(self.status_label)
    
    def create_step_widgets(self, layout):
        """创建步骤控件"""
        steps_data = [
            ("step1", "双向路径智能规划", "混合A*算法生成双向路径", True),
            ("step2", "动态节点密度控制", "基于曲率的自适应节点生成", True),
            ("step3", "关键节点聚类提取", "多轮聚类识别拓扑关键节点", False),
            ("step4", "车辆动力学约束拟合", "Clothoid-Cubic曲线拟合", False),
            ("step5", "图拓扑标准化输出", "生成GNN输入格式", False)
        ]
        
        for step_id, name, desc, enabled in steps_data:
            step_widget = self.create_single_step_widget(step_id, name, desc, enabled)
            self.step_widgets[step_id] = step_widget
            layout.addWidget(step_widget)
    
    def create_single_step_widget(self, step_id, name, desc, enabled):
        """创建单个步骤控件"""
        group = QGroupBox()
        layout = QHBoxLayout()
        
        # 步骤指示器
        indicator = QLabel("●")
        indicator.setFont(QFont("", 12))
        
        # 步骤信息
        info_layout = QVBoxLayout()
        name_label = QLabel(name)
        name_label.setFont(QFont("", 10, QFont.Bold))
        desc_label = QLabel(desc)
        desc_label.setFont(QFont("", 9))
        
        info_layout.addWidget(name_label)
        info_layout.addWidget(desc_label)
        
        # 执行按钮
        execute_btn = QPushButton("执行")
        execute_btn.setMaximumWidth(50)
        execute_btn.setEnabled(enabled)
        execute_btn.clicked.connect(lambda: self.execute_single_step(step_id))
        
        # 状态标签
        status_label = QLabel("等待")
        status_label.setMinimumWidth(40)
        
        layout.addWidget(indicator)
        layout.addLayout(info_layout, 1)
        layout.addWidget(execute_btn)
        layout.addWidget(status_label)
        
        group.setLayout(layout)
        
        # 存储控件引用
        group.indicator = indicator
        group.execute_btn = execute_btn
        group.status_label = status_label
        group.step_id = step_id
        
        return group
    
    def set_backbone_network(self, backbone_network):
        """设置骨干网络"""
        self.backbone_network = backbone_network
    
    def execute_single_step(self, step_id):
        """执行单个步骤 - 集成增强版"""
        if not self.backbone_network:
            QMessageBox.warning(self, "警告", "请先加载环境和初始化系统")
            return
        
        step_widget = self.step_widgets[step_id]
        self.update_step_status(step_id, "执行中")
        
        try:
            if step_id in ["step1", "step2"]:
                # 步骤1和2：生成原始骨干网络
                success = self.backbone_network.generate_backbone_network(
                    quality_threshold=0.6,
                    enable_consolidation=False  # 先不启用整合
                )
                
                if success:
                    self.update_step_status("step1", "完成")
                    self.update_step_status("step2", "完成")
                    self.overall_progress.setValue(2)
                    
                    # 启用后续步骤
                    self.step_widgets["step3"].execute_btn.setEnabled(True)
                    
                    self.step_executed.emit("raw_paths_generated")
                    QMessageBox.information(self, "成功", "原始骨干路径生成完成！\n现在可以在可视化中查看聚类前的路径网络")
                else:
                    self.update_step_status(step_id, "失败")
                    QMessageBox.critical(self, "失败", "原始路径生成失败")
            
            elif step_id == "step3":
                # 步骤3：关键节点聚类提取
                success = self._execute_enhanced_clustering_step()
                
                if success:
                    self.update_step_status("step3", "完成")
                    self.step_widgets["step4"].execute_btn.setEnabled(True)
                    self.overall_progress.setValue(3)
                    
                    self.step_executed.emit("clustering_completed")
                    QMessageBox.information(self, "成功", "关键节点聚类提取完成！")
                else:
                    self.update_step_status("step3", "失败")
                    QMessageBox.critical(self, "失败", "聚类执行失败")
            
            elif step_id == "step4":
                # 步骤4：增强曲线拟合
                success = self._execute_enhanced_curve_fitting()
                
                if success:
                    self.update_step_status("step4", "完成")
                    self.step_widgets["step5"].execute_btn.setEnabled(True)
                    self.overall_progress.setValue(4)
                    
                    self.step_executed.emit("curve_fitting_completed")
                    QMessageBox.information(self, "成功", "增强曲线拟合完成！")
                else:
                    self.update_step_status("step4", "失败")
                    QMessageBox.critical(self, "失败", "曲线拟合失败")
            
            elif step_id == "step5":
                # 步骤5：图拓扑标准化
                success = self._execute_topology_standardization()
                
                if success:
                    self.update_step_status("step5", "完成")
                    self.overall_progress.setValue(5)
                    
                    self.step_executed.emit("topology_standardized")
                    QMessageBox.information(self, "成功", "第一阶段全部完成！\n可以查看最终的重建道路网络")
                else:
                    self.update_step_status("step5", "失败")
                    QMessageBox.critical(self, "失败", "拓扑标准化失败")
        
        except Exception as e:
            self.update_step_status(step_id, "失败")
            QMessageBox.critical(self, "错误", f"执行失败: {str(e)}")
    
    def _execute_enhanced_clustering_step(self) -> bool:
        """执行增强版聚类步骤"""
        try:
            # 通过backbone_network执行聚类，而不是直接创建consolidator
            # backbone_network内部会创建和管理professional_consolidator
            if not hasattr(self.backbone_network, 'professional_consolidator') or not self.backbone_network.professional_consolidator:
                # 让backbone_network创建professional_consolidator
                # 这将在backbone_network的generate_backbone_network方法中处理
                print("通过backbone_network创建professional_consolidator...")
                return True
            
            # 如果已经有了consolidator，执行聚类步骤
            consolidator = self.backbone_network.professional_consolidator
            
            # 仅执行聚类，暂不重建
            success = consolidator._extract_original_paths(self.backbone_network)
            if success:
                consolidator._identify_and_protect_endpoints()
                success = consolidator._perform_multi_round_clustering()
                if success:
                    success = consolidator._generate_key_nodes()
            
            return success
        
        except Exception as e:
            print(f"增强版聚类步骤失败: {e}")
            return False
    
    def _execute_enhanced_curve_fitting(self) -> bool:
        """执行增强版曲线拟合"""
        try:
            if not self.backbone_network.professional_consolidator:
                return False
            
            # 执行路径重建
            return self.backbone_network.professional_consolidator._enhanced_reconstruct_backbone_paths_with_collision_repair()
    
        
        except Exception as e:
            print(f"增强版曲线拟合失败: {e}")
            return False
    
    def _execute_topology_standardization(self) -> bool:
        """执行拓扑标准化"""
        try:
            if not self.backbone_network.professional_consolidator:
                return False
            
            # 应用整合结果
            return self.backbone_network.professional_consolidator._apply_consolidation_to_backbone(self.backbone_network)
        
        except Exception as e:
            print(f"拓扑标准化失败: {e}")
            return False
    
    def start_full_construction(self):
        """开始完整构建 - 集成增强版"""
        if not self.backbone_network:
            QMessageBox.warning(self, "警告", "请先加载环境和初始化系统")
            return
        
        self.start_all_btn.setEnabled(False)
        self.status_label.setText("正在执行完整构建...")
        
        try:
            # 执行完整的第一阶段构建
            success = self.backbone_network.generate_backbone_network(
                quality_threshold=0.6,
                enable_consolidation=True  # 启用完整整合
            )
            
            if success:
                # 更新所有步骤状态
                for i, step_id in enumerate(["step1", "step2", "step3", "step4", "step5"], 1):
                    self.update_step_status(step_id, "完成")
                
                self.overall_progress.setValue(5)
                self.status_label.setText("第一阶段构建完成！")
                self.step_executed.emit("full_construction_completed")
                QMessageBox.information(self, "成功", "智能拓扑构建全部完成！\n可以查看完整的重建道路网络")
            else:
                self.status_label.setText("构建失败")
                QMessageBox.critical(self, "失败", "智能拓扑构建失败")
        
        except Exception as e:
            self.status_label.setText("构建异常")
            QMessageBox.critical(self, "错误", f"构建异常: {str(e)}")
        
        finally:
            self.start_all_btn.setEnabled(True)
    
    def update_step_status(self, step_id, status):
        """更新步骤状态"""
        step_widget = self.step_widgets[step_id]
        step_widget.status_label.setText(status)
    
    def reset_progress(self):
        """重置进度"""
        self.overall_progress.setValue(0)
        self.status_label.setText("就绪")
        
        for step_id, step_widget in self.step_widgets.items():
            self.update_step_status(step_id, "等待")
            step_widget.execute_btn.setEnabled(step_id in ["step1", "step2"])

class EnhancedGraphicsView(QGraphicsView):
    """增强的图形视图 - 支持缩放和平移"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 启用平移和缩放
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        
        # 缩放参数
        self.zoom_factor = 1.25
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        self.current_zoom = 1.0
        
        # 平移模式标志
        self.panning = False
        self.pan_start = None
    
    def wheelEvent(self, event: QWheelEvent):
        """鼠标滚轮缩放"""
        if event.angleDelta().y() > 0:
            factor = self.zoom_factor
        else:
            factor = 1.0 / self.zoom_factor
        
        new_zoom = self.current_zoom * factor
        if self.min_zoom <= new_zoom <= self.max_zoom:
            self.scale(factor, factor)
            self.current_zoom = new_zoom
    
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if event.button() == Qt.MiddleButton:
            self.panning = True
            self.pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.RightButton:
            if self.dragMode() == QGraphicsView.RubberBandDrag:
                self.setDragMode(QGraphicsView.ScrollHandDrag)
            else:
                self.setDragMode(QGraphicsView.RubberBandDrag)
        else:
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if self.panning and self.pan_start:
            delta = event.pos() - self.pan_start
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            self.pan_start = event.pos()
        else:
            super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if event.button() == Qt.MiddleButton:
            self.panning = False
            self.pan_start = None
            self.setCursor(Qt.ArrowCursor)
        else:
            super().mouseReleaseEvent(event)
    
    def reset_view(self):
        """重置视图"""
        self.resetTransform()
        self.current_zoom = 1.0
        self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)

class IntegratedTopologyVisualizationWidget(QWidget):
    """集成拓扑可视化组件 - 支持聚类前后完整显示"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.backbone_network = None
        self.current_display_mode = "raw_paths"  # "raw_paths", "clustered_nodes", "reconstructed_roads"
        self.auto_fit_enabled = True
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        
        # 可视化控制
        control_layout = QVBoxLayout()
        
        # 第一行：显示模式
        mode_layout = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["原始骨干路径", "聚类关键节点", "重建道路网络"])
        self.mode_combo.currentTextChanged.connect(self.change_display_mode)
        
        mode_layout.addWidget(QLabel("显示模式:"))
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()
        control_layout.addLayout(mode_layout)
        
        # 第二行：操作按钮
        button_layout = QHBoxLayout()
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.clicked.connect(self.refresh_visualization)
        
        self.fit_view_btn = QPushButton("适应视图")
        self.fit_view_btn.clicked.connect(self.fit_view)
        
        self.reset_view_btn = QPushButton("重置视图")
        self.reset_view_btn.clicked.connect(self.reset_view)
        
        self.debug_btn = QPushButton("调试信息")
        self.debug_btn.clicked.connect(self.show_debug_info)
        
        button_layout.addWidget(self.refresh_btn)
        button_layout.addWidget(self.fit_view_btn)
        button_layout.addWidget(self.reset_view_btn)
        button_layout.addWidget(self.debug_btn)
        button_layout.addStretch()
        control_layout.addLayout(button_layout)
        
        # 第三行：显示选项
        options_layout = QHBoxLayout()
        
        self.auto_fit_checkbox = QCheckBox("自动适配")
        self.auto_fit_checkbox.setChecked(True)
        self.auto_fit_checkbox.toggled.connect(self.toggle_auto_fit)
        
        self.show_obstacles_checkbox = QCheckBox("显示障碍物")
        self.show_obstacles_checkbox.setChecked(True)
        self.show_obstacles_checkbox.toggled.connect(self.refresh_visualization)
        
        self.show_interfaces_checkbox = QCheckBox("显示接口")
        self.show_interfaces_checkbox.setChecked(True)
        self.show_interfaces_checkbox.toggled.connect(self.refresh_visualization)
        
        self.show_path_nodes_checkbox = QCheckBox("显示路径节点")
        self.show_path_nodes_checkbox.setChecked(False)
        self.show_path_nodes_checkbox.toggled.connect(self.refresh_visualization)
        
        options_layout.addWidget(self.auto_fit_checkbox)
        options_layout.addWidget(self.show_obstacles_checkbox)
        options_layout.addWidget(self.show_interfaces_checkbox)
        options_layout.addWidget(self.show_path_nodes_checkbox)
        options_layout.addStretch()
        control_layout.addLayout(options_layout)
        
        layout.addLayout(control_layout)
        
        # 使用增强的图形视图
        self.graphics_view = EnhancedGraphicsView()
        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)
        
        layout.addWidget(self.graphics_view, 1)
        
        # 统计信息
        self.stats_label = QLabel("等待数据...")
        layout.addWidget(self.stats_label)
        
        # 操作提示
        help_label = QLabel("操作: 滚轮缩放 | 中键拖拽 | 右键切换模式")
        help_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(help_label)
    
    def set_backbone_network(self, backbone_network):
        """设置骨干网络"""
        self.backbone_network = backbone_network
    
    def toggle_auto_fit(self, enabled):
        """切换自动适配"""
        self.auto_fit_enabled = enabled
        if enabled:
            self.fit_view()
    
    def change_display_mode(self, mode_text):
        """改变显示模式"""
        if "原始" in mode_text:
            self.current_display_mode = "raw_paths"
        elif "聚类" in mode_text:
            self.current_display_mode = "clustered_nodes"
        elif "重建" in mode_text:
            self.current_display_mode = "reconstructed_roads"
        
        self.refresh_visualization()
    
    def refresh_visualization(self):
        """刷新可视化"""
        if not self.backbone_network:
            return
        
        self.scene.clear()
        
        if self.current_display_mode == "raw_paths":
            self.draw_raw_paths_network()
        elif self.current_display_mode == "clustered_nodes":
            self.draw_clustered_nodes_network()
        elif self.current_display_mode == "reconstructed_roads":
            self.draw_reconstructed_roads_network()
        
        self.update_statistics()
        
        if self.auto_fit_enabled:
            self.fit_view()
    
    def draw_raw_paths_network(self):
        """绘制原始骨干路径网络（聚类前）"""
        print("🎨 绘制原始骨干路径网络")
        
        raw_info = self.backbone_network.get_raw_backbone_paths_info()
        
        if raw_info['status'] != 'generated':
            self.draw_placeholder_text("原始骨干路径尚未生成\n请先执行步骤1-2")
            return
        
        # 自动调整场景大小
        self.auto_adjust_scene_rect(raw_info)
        
        # 绘制环境背景
        self.draw_environment_background()
        
        # 绘制特殊点
        self.draw_special_points(raw_info['special_points'])
        
        # 绘制原始双向路径
        paths_info = raw_info['paths_info']
        for i, (path_id, path_data) in enumerate(paths_info.items()):
            path_color = self.get_path_color(i)
            
            # 绘制前向路径
            self.draw_single_path(
                path_data['forward_path'], 
                path_color, 
                f"原始路径: {path_id}\\n长度: {len(path_data['forward_path'])}点\\n质量: {path_data.get('quality', 0):.2f}"
            )
            
            # 可选：绘制路径节点
            if self.show_path_nodes_checkbox.isChecked():
                self.draw_path_nodes(path_data['forward_path'], path_color.darker(150))
        
        # 绘制接口点
        if self.show_interfaces_checkbox.isChecked():
            interfaces_info = raw_info['interfaces_info']
            for interface_id, interface_data in interfaces_info.items():
                if hash(interface_id) % 3 == 0:  # 稀疏显示
                    self.draw_interface_point(interface_data['position'], QColor(255, 193, 7))
        
        print(f"✅ 原始路径网络绘制完成: {len(paths_info)}条路径")
    
    def draw_clustered_nodes_network(self):
        """绘制聚类关键节点网络（聚类后，重建前）"""
        print("🎨 绘制聚类关键节点网络")
        
        if not (self.backbone_network.professional_consolidator and 
                hasattr(self.backbone_network.professional_consolidator, 'key_nodes')):
            self.draw_placeholder_text("关键节点聚类尚未完成\\n请先执行步骤3")
            return
        
        consolidator = self.backbone_network.professional_consolidator
        key_nodes_info = consolidator.get_key_nodes_info()
        
        if not key_nodes_info:
            self.draw_placeholder_text("关键节点信息不可用")
            return
        
        # 自动调整场景大小
        self.auto_adjust_scene_rect_for_key_nodes(key_nodes_info)
        
        # 绘制环境背景
        self.draw_environment_background()
        
        # 绘制原始路径（半透明）作为背景参考
        raw_info = self.backbone_network.get_raw_backbone_paths_info()
        if raw_info['status'] == 'generated':
            paths_info = raw_info['paths_info']
            for i, (path_id, path_data) in enumerate(paths_info.items()):
                path_color = self.get_path_color(i)
                path_color.setAlpha(100)  # 半透明
                self.draw_single_path(path_data['forward_path'], path_color, f"原始: {path_id}")
        
        # 绘制关键节点
        for node_id, node_info in key_nodes_info.items():
            position = node_info['position']
            x, y = position[0], position[1]
            
            # 节点大小限制在1.5以内
            if node_info.get('is_endpoint', False):
                radius = 1.5  # 端点稍大
                color = QColor(255, 0, 0)  # 红色
            else:
                radius = 1.0  # 普通关键节点
                color = QColor(255, 193, 7)  # 黄色
            
            circle = QGraphicsEllipseItem(x-radius, y-radius, radius*2, radius*2)
            circle.setBrush(QBrush(color))
            circle.setPen(QPen(color.darker(150), 0.5))
            circle.setZValue(15)
            
            # 工具提示
            importance = node_info.get('importance', 1.0)
            original_count = node_info.get('original_nodes_count', 0)
            tooltip = f"关键节点: {node_id}\\n重要性: {importance:.1f}\\n原始节点: {original_count}个"
            if node_info.get('is_endpoint', False):
                tooltip += "\\n(端点)"
            circle.setToolTip(tooltip)
            
            self.scene.addItem(circle)
            
            # 节点标签
            if node_info.get('is_endpoint', False):
                label = QGraphicsTextItem(f"E{node_id.split('_')[-1]}")
            else:
                label = QGraphicsTextItem(f"K{node_id.split('_')[-1]}")
            label.setPos(x-5, y-10)
            label.setDefaultTextColor(color.darker(200))
            label.setFont(QFont("Arial", 2))  # 地图上的小字体
            label.setZValue(16)
            self.scene.addItem(label)
        
        # 绘制聚类连接（原始路径的关键节点连接）
        if hasattr(consolidator, 'original_paths'):
            for path_id, path_info in consolidator.original_paths.items():
                # 找到该路径对应的关键节点序列
                path_key_nodes = []
                for node_id, node_info in key_nodes_info.items():
                    if path_id in node_info.get('path_memberships', []):
                        path_key_nodes.append((node_id, node_info['position']))
                
                # 按原路径顺序排序并连接
                if len(path_key_nodes) >= 2:
                    for i in range(len(path_key_nodes) - 1):
                        pos1 = path_key_nodes[i][1]
                        pos2 = path_key_nodes[i + 1][1]
                        
                        line = QGraphicsLineItem(pos1[0], pos1[1], pos2[0], pos2[1])
                        pen = QPen(QColor(158, 158, 158), 1.0)
                        pen.setStyle(Qt.DashLine)
                        line.setPen(pen)
                        line.setZValue(3)
                        line.setToolTip(f"聚类连接: {path_id}")
                        self.scene.addItem(line)
        
        print(f"✅ 聚类关键节点网络绘制完成: {len(key_nodes_info)}个关键节点")
    
    def draw_reconstructed_roads_network(self):
        """绘制重建道路网络（最终结果）"""
        print("🎨 绘制重建道路网络")
        
        if not (self.backbone_network.professional_consolidator and 
                hasattr(self.backbone_network.professional_consolidator, 'consolidated_paths')):
            self.draw_placeholder_text("道路重建尚未完成\\n请先执行步骤4-5")
            return
        
        consolidator = self.backbone_network.professional_consolidator
        key_nodes_info = consolidator.get_key_nodes_info()
        consolidated_paths_info = consolidator.get_consolidated_paths_info()
        
        if not key_nodes_info or not consolidated_paths_info:
            self.draw_placeholder_text("重建道路数据不可用")
            return
        
        # 自动调整场景大小
        self.auto_adjust_scene_rect_for_reconstructed_roads()
        
        # 绘制环境背景
        self.draw_environment_background()
        
        # 绘制重建的道路
        for path_id, path_info in consolidated_paths_info.items():
            reconstructed_path = self.get_reconstructed_path_from_consolidator(path_id)
            
            if reconstructed_path and len(reconstructed_path) >= 2:
                self.draw_reconstructed_road(reconstructed_path, path_info)
            else:
                # 回退到关键节点直线连接
                self.draw_key_nodes_fallback_connection(path_info, key_nodes_info)
        
        # 绘制关键节点（在道路之上）
        for node_id, node_info in key_nodes_info.items():
            position = node_info['position']
            x, y = position[0], position[1]
            
            # 节点大小限制
            if node_info.get('is_endpoint', False):
                radius = 1.2
                color = QColor(255, 0, 0)
            else:
                radius = 0.8
                color = QColor(255, 193, 7)
            
            circle = QGraphicsEllipseItem(x-radius, y-radius, radius*2, radius*2)
            circle.setBrush(QBrush(color))
            circle.setPen(QPen(color.darker(150), 0.5))
            circle.setZValue(20)  # 确保在道路之上
            
            # 工具提示
            importance = node_info.get('importance', 1.0)
            curve_quality = node_info.get('curve_fitting_quality', 0.0)
            tooltip = f"关键节点: {node_id}\\n重要性: {importance:.1f}\\n拟合质量: {curve_quality:.2f}"
            circle.setToolTip(tooltip)
            
            self.scene.addItem(circle)
        
        print(f"✅ 重建道路网络绘制完成: {len(consolidated_paths_info)}条道路")
    
    def get_reconstructed_path_from_consolidator(self, path_id: str) -> List[Tuple]:
        """从整合器获取重建路径"""
        try:
            consolidator = self.backbone_network.professional_consolidator
            
            # 从整合路径中获取
            if hasattr(consolidator, 'consolidated_paths'):
                for consolidated_path_id, consolidated_path in consolidator.consolidated_paths.items():
                    if (path_id in consolidated_path_id or 
                        consolidated_path.original_path_id in path_id or
                        path_id in getattr(consolidated_path, 'original_path_id', '')):
                        
                        if (hasattr(consolidated_path, 'reconstructed_path') and 
                            consolidated_path.reconstructed_path and
                            getattr(consolidated_path, 'reconstruction_success', False)):
                            return consolidated_path.reconstructed_path
            
            # 从骨干网络获取（已整合后的）
            if hasattr(self.backbone_network, 'bidirectional_paths'):
                for backbone_path_id, backbone_path in self.backbone_network.bidirectional_paths.items():
                    if path_id in backbone_path_id or backbone_path_id in path_id:
                        if hasattr(backbone_path, 'forward_path') and backbone_path.forward_path:
                            return backbone_path.forward_path
            
            return None
        
        except Exception as e:
            print(f"获取重建路径失败 {path_id}: {e}")
            return None
    
    def draw_reconstructed_road(self, path: List[Tuple], path_info: Dict):
        """绘制重建的道路"""
        if not path or len(path) < 2:
            return
        
        # 确定道路样式
        road_class = path_info.get('road_class', 'secondary')
        curve_fitting_method = path_info.get('curve_fitting_method', 'unknown')
        quality_score = path_info.get('curve_quality_score', 0.0)
        reconstruction_success = path_info.get('reconstruction_success', False)
        
        # 道路等级颜色
        if road_class == 'primary':
            base_color = QColor(46, 125, 50)  # 深绿色
            line_width = 2.0
        elif road_class == 'secondary':
            base_color = QColor(76, 175, 80)  # 绿色
            line_width = 1.5
        else:
            base_color = QColor(139, 195, 74)  # 浅绿色
            line_width = 1.0
        
        # 根据拟合方法调整
        if 'enhanced' in curve_fitting_method:
            base_color = base_color.lighter(110)
            line_style = Qt.SolidLine
        elif 'traditional' in curve_fitting_method:
            line_style = Qt.SolidLine
        elif 'fallback' in curve_fitting_method:
            base_color = base_color.darker(120)
            line_style = Qt.DashLine
        else:
            line_style = Qt.DotLine
        
        # 根据质量调整透明度
        alpha = max(150, min(255, int(150 + quality_score * 105)))
        base_color.setAlpha(alpha)
        
        # 绘制路径
        painter_path = QPainterPath()
        painter_path.moveTo(path[0][0], path[0][1])
        
        for point in path[1:]:
            painter_path.lineTo(point[0], point[1])
        
        path_item = QGraphicsPathItem(painter_path)
        
        pen = QPen(base_color, line_width)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        pen.setStyle(line_style)
        
        path_item.setPen(pen)
        path_item.setZValue(5)
        
        # 详细工具提示
        dynamics_compliance = path_info.get('dynamics_compliance_rate', 0.0)
        tooltip = f"重建道路\\n道路等级: {road_class.title()}\\n拟合方法: {curve_fitting_method}\\n质量分数: {quality_score:.2f}\\n重建成功: {'✅' if reconstruction_success else '❌'}\\n动力学合规: {dynamics_compliance:.1%}\\n路径点数: {len(path)}"
        
        path_item.setToolTip(tooltip)
        self.scene.addItem(path_item)
        
        # 可选：绘制路径节点
        if self.show_path_nodes_checkbox.isChecked():
            self.draw_reconstructed_path_nodes(path, base_color)
    
    def draw_key_nodes_fallback_connection(self, path_info: Dict, key_nodes_info: Dict):
        """绘制关键节点回退连接（当重建失败时）"""
        key_nodes = path_info.get('key_nodes', [])
        if len(key_nodes) < 2:
            return
        
        for i in range(len(key_nodes) - 1):
            node1_info = key_nodes_info.get(key_nodes[i])
            node2_info = key_nodes_info.get(key_nodes[i + 1])
            
            if node1_info and node2_info:
                pos1 = node1_info['position']
                pos2 = node2_info['position']
                
                line = QGraphicsLineItem(pos1[0], pos1[1], pos2[0], pos2[1])
                pen = QPen(QColor(158, 158, 158), 1.0)
                pen.setStyle(Qt.DashDotLine)
                line.setPen(pen)
                line.setZValue(3)
                line.setToolTip("回退连接：直线连接关键节点")
                self.scene.addItem(line)
    
    def draw_reconstructed_path_nodes(self, path: List[Tuple], color: QColor):
        """绘制重建路径的节点"""
        if not path:
            return
        
        step = max(1, len(path) // 10)  # 最多显示10个节点
        node_color = color.darker(150)
        node_color.setAlpha(180)
        
        for i in range(0, len(path), step):
            point = path[i]
            x, y = point[0], point[1]
            
            # 节点大小限制
            node = QGraphicsEllipseItem(x-0.3, y-0.3, 0.6, 0.6)
            node.setBrush(QBrush(node_color))
            node.setPen(QPen(Qt.NoPen))
            node.setZValue(8)
            node.setToolTip(f"路径节点 {i}: ({x:.1f}, {y:.1f})")
            self.scene.addItem(node)
    
    # ==================== 通用辅助方法 ====================
    
    def get_path_color(self, index):
        """根据索引获取路径颜色"""
        colors = [
            QColor(0, 100, 200),    # 蓝色
            QColor(200, 100, 0),    # 橙色  
            QColor(0, 150, 100),    # 绿色
            QColor(150, 0, 150),    # 紫色
            QColor(200, 0, 100),    # 红紫色
            QColor(100, 150, 0),    # 黄绿色
            QColor(0, 150, 200),    # 青色
            QColor(150, 100, 0),    # 棕色
        ]
        return colors[index % len(colors)]
    
    def draw_environment_background(self):
        """绘制环境背景"""
        scene_rect = self.scene.sceneRect()
        bg_rect = QGraphicsRectItem(scene_rect)
        bg_rect.setBrush(QBrush(QColor(240, 240, 240)))
        bg_rect.setPen(QPen(Qt.NoPen))
        bg_rect.setZValue(-100)
        self.scene.addItem(bg_rect)
        
        if self.show_obstacles_checkbox.isChecked():
            self.draw_obstacles()
    
    def draw_obstacles(self):
        """绘制障碍物"""
        if not (self.backbone_network and self.backbone_network.env):
            return
        
        env = self.backbone_network.env
        
        if hasattr(env, 'obstacle_points') and env.obstacle_points:
            for x, y in env.obstacle_points:
                obstacle = QGraphicsRectItem(x-0.5, y-0.5, 1, 1)
                obstacle.setBrush(QBrush(QColor(80, 80, 80)))
                obstacle.setPen(QPen(Qt.NoPen))
                obstacle.setZValue(1)
                obstacle.setToolTip(f"障碍物 ({x}, {y})")
                self.scene.addItem(obstacle)
    
    def draw_special_points(self, special_points):
        """绘制特殊点"""
        for point_type, points in special_points.items():
            for point in points:
                pos = point['position']
                x, y = pos[0], pos[1]
                
                # 特殊点大小限制
                if point_type == 'loading':
                    item = QGraphicsEllipseItem(x-1.5, y-1.5, 3, 3)
                    item.setBrush(QBrush(QColor(0, 150, 0)))
                    item.setPen(QPen(QColor(0, 100, 0), 1))
                    
                    label = QGraphicsTextItem(f"L{point['id']}")
                    label.setPos(x-5, y-15)
                    label.setDefaultTextColor(QColor(0, 150, 0))
                    label.setFont(QFont("Arial", 2))  # 地图上的小字体
                    self.scene.addItem(label)
                    
                elif point_type == 'unloading':
                    item = QGraphicsRectItem(x-1.5, y-1.5, 3, 3)
                    item.setBrush(QBrush(QColor(200, 100, 0)))
                    item.setPen(QPen(QColor(150, 75, 0), 1))
                    
                    label = QGraphicsTextItem(f"U{point['id']}")
                    label.setPos(x-5, y-15)
                    label.setDefaultTextColor(QColor(200, 100, 0))
                    label.setFont(QFont("Arial", 2))  # 地图上的小字体
                    self.scene.addItem(label)
                
                elif point_type == 'parking':
                    item = QGraphicsRectItem(x-1.2, y-1.2, 2.4, 2.4)
                    item.setBrush(QBrush(QColor(100, 100, 200)))
                    item.setPen(QPen(QColor(70, 70, 150), 1))
                    
                    label = QGraphicsTextItem(f"P{point['id']}")
                    label.setPos(x-5, y-15)
                    label.setDefaultTextColor(QColor(100, 100, 200))
                    label.setFont(QFont("Arial", 2))  # 地图上的小字体
                    self.scene.addItem(label)
                
                item.setZValue(10)
                item.setToolTip(f"{point_type.title()} Point {point['id']}")
                self.scene.addItem(item)
    
    def draw_single_path(self, path, color, tooltip=""):
        """绘制单条路径"""
        if not path or len(path) < 2:
            return
        
        painter_path = QPainterPath()
        painter_path.moveTo(path[0][0], path[0][1])
        
        for point in path[1:]:
            painter_path.lineTo(point[0], point[1])
        
        path_item = QGraphicsPathItem(painter_path)
        pen = QPen(color, 1.5)
        pen.setCapStyle(Qt.RoundCap)
        path_item.setPen(pen)
        path_item.setZValue(5)
        
        if tooltip:
            path_item.setToolTip(tooltip)
        
        self.scene.addItem(path_item)
    
    def draw_path_nodes(self, path, color):
        """绘制路径节点（稀疏显示）"""
        if not path:
            return
        
        step = max(1, len(path) // 15)  # 最多显示15个节点
        
        for i in range(0, len(path), step):
            point = path[i]
            x, y = point[0], point[1]
            node = QGraphicsEllipseItem(x-0.5, y-0.5, 1, 1)
            node.setBrush(QBrush(color))
            node.setPen(QPen(Qt.NoPen))
            node.setZValue(8)
            self.scene.addItem(node)
    
    def draw_interface_point(self, position, color):
        """绘制接口点"""
        x, y = position[0], position[1]
        interface = QGraphicsEllipseItem(x-0.8, y-0.8, 1.6, 1.6)
        interface.setBrush(QBrush(color))
        interface.setPen(QPen(color.darker(150), 0.5))
        interface.setZValue(12)
        self.scene.addItem(interface)
    
    def draw_placeholder_text(self, text):
        """绘制占位符文本"""
        self.scene.setSceneRect(0, 0, 400, 300)
        
        text_item = QGraphicsTextItem(text)
        text_item.setPos(50, 100)
        text_item.setFont(QFont("Arial", 12))  # 占位符文本稍大一些
        self.scene.addItem(text_item)
    
    def auto_adjust_scene_rect(self, raw_info):
        """自动调整场景大小（原始路径）"""
        if not raw_info or 'paths_info' not in raw_info:
            if self.backbone_network and self.backbone_network.env:
                self.scene.setSceneRect(0, 0, 
                    self.backbone_network.env.width, 
                    self.backbone_network.env.height)
            else:
                self.scene.setSceneRect(0, 0, 500, 500)
            return
        
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        
        for path_id, path_data in raw_info['paths_info'].items():
            if 'forward_path' in path_data and path_data['forward_path']:
                for point in path_data['forward_path']:
                    x, y = point[0], point[1]
                    min_x, max_x = min(min_x, x), max(max_x, x)
                    min_y, max_y = min(min_y, y), max(max_y, y)
        
        margin = 50
        if min_x != float('inf'):
            self.scene.setSceneRect(
                min_x - margin, min_y - margin,
                max_x - min_x + 2 * margin, 
                max_y - min_y + 2 * margin
            )
    
    def auto_adjust_scene_rect_for_key_nodes(self, key_nodes_info):
        """自动调整场景大小（关键节点）"""
        if not key_nodes_info:
            return
        
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        
        for node_id, node_info in key_nodes_info.items():
            position = node_info['position']
            x, y = position[0], position[1]
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
        
        margin = 100
        if min_x != float('inf'):
            self.scene.setSceneRect(
                min_x - margin, min_y - margin,
                max_x - min_x + 2 * margin, 
                max_y - min_y + 2 * margin
            )
    
    def auto_adjust_scene_rect_for_reconstructed_roads(self):
        """自动调整场景大小（重建道路）"""
        if not (self.backbone_network.professional_consolidator and 
                hasattr(self.backbone_network.professional_consolidator, 'consolidated_paths')):
            return
        
        try:
            consolidated_paths_info = self.backbone_network.professional_consolidator.get_consolidated_paths_info()
            
            min_x, min_y = float('inf'), float('inf')
            max_x, max_y = float('-inf'), float('-inf')
            
            for path_id, path_info in consolidated_paths_info.items():
                reconstructed_path = self.get_reconstructed_path_from_consolidator(path_id)
                if reconstructed_path:
                    for point in reconstructed_path:
                        x, y = point[0], point[1]
                        min_x, max_x = min(min_x, x), max(max_x, x)
                        min_y, max_y = min(min_y, y), max(max_y, y)
            
            margin = 100
            if min_x != float('inf'):
                self.scene.setSceneRect(
                    min_x - margin, min_y - margin,
                    max_x - min_x + 2 * margin, 
                    max_y - min_y + 2 * margin
                )
        
        except Exception as e:
            print(f"自动调整重建道路场景大小失败: {e}")
    
    def update_statistics(self):
        """更新统计信息"""
        if not self.backbone_network:
            return
        
        if self.current_display_mode == "raw_paths":
            raw_info = self.backbone_network.get_raw_backbone_paths_info()
            if raw_info['status'] == 'generated':
                stats = raw_info['generation_stats']
                paths_count = len(raw_info['paths_info'])
                total_nodes = sum(len(p['forward_path']) for p in raw_info['paths_info'].values())
                avg_quality = sum(p['quality'] for p in raw_info['paths_info'].values()) / max(1, paths_count)
                
                base_text = f"原始骨干路径 | 路径: {paths_count}条 | 节点: {total_nodes}个 | 平均质量: {avg_quality:.2f}"
            else:
                base_text = "原始骨干路径尚未生成"
        
        elif self.current_display_mode == "clustered_nodes":
            if (self.backbone_network.professional_consolidator and 
                hasattr(self.backbone_network.professional_consolidator, 'key_nodes')):
                
                key_nodes_info = self.backbone_network.professional_consolidator.get_key_nodes_info()
                consolidation_stats = self.backbone_network.professional_consolidator.get_consolidation_stats()
                
                key_nodes_count = len(key_nodes_info)
                endpoint_count = sum(1 for n in key_nodes_info.values() if n.get('is_endpoint', False))
                reduction_ratio = consolidation_stats.get('node_reduction_ratio', 0.0)
                
                base_text = f"聚类关键节点 | 关键节点: {key_nodes_count}个 | 端点: {endpoint_count}个 | 节点减少: {reduction_ratio:.1%}"
            else:
                base_text = "关键节点聚类尚未完成"
        
        elif self.current_display_mode == "reconstructed_roads":
            if (self.backbone_network.professional_consolidator and 
                hasattr(self.backbone_network.professional_consolidator, 'consolidated_paths')):
                
                consolidated_paths_info = self.backbone_network.professional_consolidator.get_consolidated_paths_info()
                consolidation_stats = self.backbone_network.professional_consolidator.get_consolidation_stats()
                
                roads_count = len(consolidated_paths_info)
                reconstructed_count = sum(
                    1 for p in consolidated_paths_info.values() 
                    if p.get('reconstruction_success', False)
                )
                avg_quality = sum(
                    p.get('curve_quality_score', 0.0) for p in consolidated_paths_info.values()
                ) / max(1, roads_count)
                
                success_rate = consolidation_stats.get('reconstruction_success_rate', 0.0)
                
                base_text = f"重建道路网络 | 道路: {roads_count}条 | 重建成功: {reconstructed_count}条 | 成功率: {success_rate:.1%} | 平均质量: {avg_quality:.2f}"
            else:
                base_text = "道路重建尚未完成"
        
        self.stats_label.setText(base_text)
    
    def show_debug_info(self):
        """显示调试信息"""
        debug_info = []
        
        if self.backbone_network:
            debug_info.append("=== 骨干网络状态 ===")
            debug_info.append(f"双向路径数量: {len(getattr(self.backbone_network, 'bidirectional_paths', {}))}")
            
            raw_info = self.backbone_network.get_raw_backbone_paths_info()
            debug_info.append(f"原始路径状态: {raw_info.get('status', 'unknown')}")
            
            if self.backbone_network.professional_consolidator:
                debug_info.append("\\n=== 专业整合器状态 ===")
                
                consolidation_stats = self.backbone_network.professional_consolidator.get_consolidation_stats()
                debug_info.append(f"原始节点数: {consolidation_stats.get('original_nodes_count', 0)}")
                debug_info.append(f"关键节点数: {consolidation_stats.get('key_nodes_count', 0)}")
                debug_info.append(f"节点减少率: {consolidation_stats.get('node_reduction_ratio', 0.0):.1%}")
                debug_info.append(f"重建成功率: {consolidation_stats.get('reconstruction_success_rate', 0.0):.1%}")
                
                if hasattr(self.backbone_network.professional_consolidator, 'consolidated_paths'):
                    consolidated_paths = self.backbone_network.professional_consolidator.consolidated_paths
                    debug_info.append(f"\\n=== 整合路径详情 ===")
                    for path_id, path in consolidated_paths.items():
                        debug_info.append(f"路径: {path_id}")
                        debug_info.append(f"  重建成功: {getattr(path, 'reconstruction_success', False)}")
                        debug_info.append(f"  重建点数: {len(getattr(path, 'reconstructed_path', []))}")
                        debug_info.append(f"  拟合方法: {getattr(path, 'curve_fitting_method', 'N/A')}")
                        debug_info.append(f"  质量分数: {getattr(path, 'curve_quality_score', 0.0):.2f}")
            else:
                debug_info.append("专业整合器: 未初始化")
        
        debug_text = "\\n".join(debug_info)
        QMessageBox.information(self, "调试信息", debug_text)
    
    def fit_view(self):
        """适应视图"""
        if self.scene.items():
            item_rect = self.scene.itemsBoundingRect()
            if not item_rect.isEmpty():
                self.graphics_view.fitInView(item_rect, Qt.KeepAspectRatio)
            else:
                self.graphics_view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        else:
            self.graphics_view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
    
    def reset_view(self):
        """重置视图"""
        self.graphics_view.reset_view()

class CompleteIntegratedTopologyGUI(QMainWindow):
    """完整集成的智能拓扑构建主界面"""
    
    def __init__(self):
        super().__init__()
        
        # 系统组件
        self.env = None
        self.backbone_network = None
        self.path_planner = None
        
        # 状态
        self.map_file_path = None
        
        self.init_ui()
        print("🚀 完整集成的智能拓扑构建GUI启动成功")
        print("注意：增强版模块通过backbone_network间接使用")
    
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("完整集成：基于拓扑感知GNN架构 - 第一阶段智能拓扑构建")
        self.setGeometry(100, 100, 1400, 900)
        
        # 中央组件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(6)
        main_layout.setContentsMargins(6, 6, 6, 6)
        
        # 左侧控制面板
        left_panel = self.create_left_panel()
        left_panel.setMaximumWidth(300)
        main_layout.addWidget(left_panel)
        
        # 中央可视化区域
        self.visualization_widget = IntegratedTopologyVisualizationWidget()
        main_layout.addWidget(self.visualization_widget, 1)
        
        # 右侧进度控制
        self.progress_widget = Stage1ProgressWidget()
        self.progress_widget.step_executed.connect(self.on_step_executed)
        right_panel = self.progress_widget
        right_panel.setMaximumWidth(280)
        main_layout.addWidget(right_panel)
        
        # 创建状态栏
        self.create_status_bar()
    
    def create_left_panel(self):
        """创建左侧控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)
        
        # 系统标题
        title_label = QLabel("智能拓扑构建系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("", 12, QFont.Bold))
        layout.addWidget(title_label)
        
        # 环境管理
        env_group = self.create_environment_section()
        layout.addWidget(env_group)
        
        # 系统初始化
        init_group = self.create_initialization_section()
        layout.addWidget(init_group)
        
        # 构建状态
        status_group = self.create_status_section()
        layout.addWidget(status_group)
        
        layout.addStretch()
        
        return panel
    
    def create_environment_section(self):
        """创建环境管理区域"""
        group = QGroupBox("环境管理")
        layout = QVBoxLayout()
        
        # 文件选择
        file_layout = QHBoxLayout()
        self.file_label = QLabel("未选择文件")
        
        self.browse_btn = QPushButton("浏览")
        self.browse_btn.clicked.connect(self.browse_file)
        
        file_layout.addWidget(self.file_label, 1)
        file_layout.addWidget(self.browse_btn)
        layout.addLayout(file_layout)
        
        # 环境操作
        self.load_btn = QPushButton("加载环境")
        self.load_btn.clicked.connect(self.load_environment)
        layout.addWidget(self.load_btn)
        
        # 环境信息
        self.env_info_label = QLabel("环境: 未加载")
        layout.addWidget(self.env_info_label)
        
        group.setLayout(layout)
        return group
    
    def create_initialization_section(self):
        """创建初始化区域"""
        group = QGroupBox("系统初始化")
        layout = QVBoxLayout()
        
        self.init_backbone_btn = QPushButton("初始化骨干网络")
        self.init_backbone_btn.clicked.connect(self.initialize_backbone_network)
        
        self.init_planner_btn = QPushButton("初始化路径规划器")
        self.init_planner_btn.clicked.connect(self.initialize_path_planner)
        
        layout.addWidget(self.init_backbone_btn)
        layout.addWidget(self.init_planner_btn)
        
        # 初始化状态
        self.init_status_label = QLabel("系统: 未初始化")
        layout.addWidget(self.init_status_label)
        
        group.setLayout(layout)
        return group
    
    def create_status_section(self):
        """创建状态区域"""
        group = QGroupBox("构建状态")
        layout = QVBoxLayout()
        
        # 状态信息
        self.construction_status = QLabel("等待开始...")
        layout.addWidget(self.construction_status)
        
        # 导出按钮
        self.export_btn = QPushButton("导出拓扑结构")
        self.export_btn.clicked.connect(self.export_topology)
        self.export_btn.setEnabled(False)
        layout.addWidget(self.export_btn)
        
        group.setLayout(layout)
        return group
    
    def create_status_bar(self):
        """创建状态栏"""
        self.status_bar = self.statusBar()
        self.status_label = QLabel("系统就绪")
        self.status_bar.addWidget(self.status_label)
    
    # ==================== 功能实现方法 ====================
    
    def browse_file(self):
        """浏览文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开地图文件", "", "JSON文件 (*.json);;所有文件 (*)"
        )
        
        if file_path:
            self.map_file_path = file_path
            filename = os.path.basename(file_path)
            self.file_label.setText(filename)
    
    def load_environment(self):
        """加载环境"""
        if not self.map_file_path:
            QMessageBox.warning(self, "警告", "请先选择地图文件")
            return
        
        try:
            self.status_label.setText("正在加载环境...")
            
            self.env = OptimizedOpenPitMineEnv()
            if not self.env.load_from_file(self.map_file_path):
                raise Exception("环境加载失败")
            
            vehicle_count = len(self.env.vehicles) if hasattr(self.env, 'vehicles') else 0
            self.env_info_label.setText(f"环境: 已加载 ({vehicle_count} 车辆)")
            
            # 重置可视化
            self.visualization_widget.set_backbone_network(None)
            
            self.status_label.setText("环境加载成功")
            
        except Exception as e:
            self.status_label.setText("加载失败")
            QMessageBox.critical(self, "错误", f"加载环境失败:\\n{str(e)}")
    
    def initialize_backbone_network(self):
        """初始化骨干网络"""
        if not self.env:
            QMessageBox.warning(self, "警告", "请先加载环境")
            return
        
        try:
            self.status_label.setText("正在初始化骨干网络...")
            
            self.backbone_network = OptimizedBackboneNetwork(self.env)
            
            # 设置到组件
            self.progress_widget.set_backbone_network(self.backbone_network)
            self.visualization_widget.set_backbone_network(self.backbone_network)
            
            self.init_status_label.setText("骨干网络: 已初始化")
            self.status_label.setText("骨干网络初始化成功")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"初始化骨干网络失败:\\n{str(e)}")
    
    def initialize_path_planner(self):
        """初始化路径规划器"""
        if not self.env:
            QMessageBox.warning(self, "警告", "请先加载环境")
            return
        
        try:
            self.status_label.setText("正在初始化路径规划器...")
            
            self.path_planner = EnhancedPathPlannerWithConfig(self.env)
            
            if self.backbone_network:
                self.backbone_network.set_path_planner(self.path_planner)
            
            self.init_status_label.setText("系统: 已初始化")
            self.status_label.setText("路径规划器初始化成功")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"初始化路径规划器失败:\\n{str(e)}")
    
    def on_step_executed(self, step_type):
        """步骤执行回调"""
        if step_type == "raw_paths_generated":
            self.construction_status.setText("✅ 原始骨干路径生成完成\\n✅ 动态节点密度控制完成")
            # 自动切换到原始路径显示
            self.visualization_widget.mode_combo.setCurrentText("原始骨干路径")
            self.visualization_widget.refresh_visualization()
            
        elif step_type == "clustering_completed":
            self.construction_status.setText("✅ 关键节点聚类提取完成")
            # 自动切换到聚类节点显示
            self.visualization_widget.mode_combo.setCurrentText("聚类关键节点")
            self.visualization_widget.refresh_visualization()
            
        elif step_type == "curve_fitting_completed":
            self.construction_status.setText("✅ 增强曲线拟合完成")
            
        elif step_type == "topology_standardized":
            self.construction_status.setText("✅ 图拓扑标准化完成")
            # 自动切换到重建道路显示
            self.visualization_widget.mode_combo.setCurrentText("重建道路网络")
            self.visualization_widget.refresh_visualization()
            self.export_btn.setEnabled(True)
            
        elif step_type == "full_construction_completed":
            self.construction_status.setText("🎉 第一阶段全部完成！\\n准备进入第二阶段")
            # 自动切换到重建道路显示
            self.visualization_widget.mode_combo.setCurrentText("重建道路网络")
            self.visualization_widget.refresh_visualization()
            self.export_btn.setEnabled(True)
   
    def export_topology(self):
        def make_json_safe(obj):
            """转换对象为JSON安全格式，处理tuple键问题"""
            if isinstance(obj, dict):
                return {str(k): make_json_safe(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_safe(x) for x in obj]
            else:
                return obj     
        """导出拓扑结构 - 修复版"""
        if not self.backbone_network:
            QMessageBox.warning(self, "警告", "没有可导出的拓扑结构")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出拓扑结构", 
            f"complete_topology_{time.strftime('%Y%m%d_%H%M%S')}.json",
            "JSON文件 (*.json);;所有文件 (*)"
        )
        
        if file_path:
            try:
                summary = self.backbone_network.get_topology_construction_summary()
                
                export_data = {
                    "system": "完整集成：基于拓扑感知GNN架构的露天矿智能调度系统",
                    "stage": "第一阶段：智能拓扑构建",
                    "export_time": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "ready_for_stage2": summary['ready_for_stage2'],
                    "gnn_input_ready": summary['gnn_input_ready'],
                    "stage1_progress": summary['stage1_progress'],
                    "construction_stats": summary['construction_stats']
                }
                
                if summary['ready_for_stage2']:
                    export_data["consolidation_stats"] = summary['consolidation_stats']
                
                # ✅ 增强版数据导出 - 包含完整位置信息
                if self.backbone_network.professional_consolidator:
                    consolidation_stats = self.backbone_network.professional_consolidator.get_consolidation_stats()
                    key_nodes_info = self.backbone_network.professional_consolidator.get_key_nodes_info()
                    consolidated_paths_info = self.backbone_network.professional_consolidator.get_consolidated_paths_info()
                    
                    export_data.update({
                        "enhanced_consolidation_applied": True,
                        "enhanced_consolidation_stats": consolidation_stats,
                        "key_nodes_info": key_nodes_info,
                        "consolidated_paths_info": consolidated_paths_info
                    })
                    
                    # ✅ 新增：导出完整的图结构和位置映射
                    if hasattr(self.backbone_network.professional_consolidator, 'position_mapping'):
                        export_data["position_mapping"] = make_json_safe(
                            self.backbone_network.professional_consolidator.position_mapping
                        )
                    
                    # ✅ 导出图的边信息
                    if hasattr(self.backbone_network.professional_consolidator, 'graph'):
                        graph = self.backbone_network.professional_consolidator.graph
                        export_data["graph_edges"] = list(graph.edges())
                        export_data["graph_nodes"] = list(graph.nodes())
                
                # ✅ 如果没有professional_consolidator，导出原始骨干网络信息
                else:
                    raw_info = self.backbone_network.get_raw_backbone_paths_info()
                    if raw_info['status'] == 'generated':
                        export_data.update({
                            "raw_backbone_paths": raw_info,
                            "special_points": self.backbone_network.special_points,
                            "backbone_interfaces": make_json_safe(self.backbone_network.backbone_interfaces)
                        })
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
                
                QMessageBox.information(self, "成功", f"完整拓扑结构已导出到:\n{file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败:\n{str(e)}")

    def closeEvent(self, event):
        """关闭事件"""
        reply = QMessageBox.question(
            self, '确认退出',
            '确定要退出完整集成构建系统吗？',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("完整集成：基于拓扑感知GNN架构的露天矿智能调度系统")
    app.setApplicationVersion("Stage 1 - Complete Integrated Topology Construction")
    
    try:
        main_window = CompleteIntegratedTopologyGUI()
        main_window.show()
        
        print("🎯 完整集成：第一阶段智能拓扑构建系统启动成功")
        print("📋 核心功能（完整集成）:")
        print("  1. 双向路径智能规划 - 原始骨干路径生成")
        print("  2. 动态节点密度控制 - 自适应节点生成")  
        print("  3. 关键节点聚类提取 - 多轮DBSCAN聚类")
        print("  4. 车辆动力学约束拟合 - Clothoid-Cubic曲线拟合")
        print("  5. 图拓扑标准化输出 - 为第二阶段GNN准备")
        print("🚀 完全集成优化的骨干网络和路径规划器，支持完整可视化流程")
        print("\\n🖼️  三种可视化模式:")
        print("  📍 原始骨干路径：聚类前的双向路径网络")
        print("  🟡 聚类关键节点：聚类后的关键节点及其连接")
        print("  🟢 重建道路网络：最终的曲线拟合道路")
        print("\\n🖱️  可视化操作指南:")
        print("  • 鼠标滚轮: 缩放视图")
        print("  • 中键拖拽: 平移视图")
        print("  • 右键: 切换选择/平移模式")
        print("  • 适应视图: 自动调整到最佳显示大小")
        print("  • 重置视图: 恢复到原始缩放和位置")
        print("  • 调试信息: 查看详细的系统状态")
        
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"❌ 应用程序启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)