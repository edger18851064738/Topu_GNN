"""
优化后的第二阶段GUI可视化组件
重点优化：
1. 统一节点大小，用颜色区分级别
2. 保持字体大小较小
3. 清晰展示节点道路预留
4. 展示车辆预测路径
"""

import sys
import os
import math
import time
import json
import threading
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject, QPointF, QRectF
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
    QPen, QBrush, QColor, QPainter, QPainterPath, QFont, QPixmap, QWheelEvent,
    QTransform, QPolygonF
)

# 导入第二阶段demo组件
try:
    from demo_stage2 import (
        Stage2TopologyLoader, Stage2RoadNetwork, Vehicle, 
        Stage2GNNSimulation, VehicleState, VehicleMode
    )
    STAGE2_AVAILABLE = True
    print("✅ 第二阶段Demo组件加载成功")
except ImportError as e:
    print(f"❌ 第二阶段Demo组件加载失败: {e}")
    try:
        # 尝试备用导入路径
        from demo2 import (
            Stage2TopologyLoader, Stage2RoadNetwork, Vehicle, 
            Stage2GNNSimulation, VehicleState, VehicleMode
        )
        STAGE2_AVAILABLE = True
        print("✅ 第二阶段Demo组件加载成功 (备用路径)")
    except ImportError:
        STAGE2_AVAILABLE = False
        print("❌ 无法找到第二阶段Demo组件，请确保文件存在")
        sys.exit(1)

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
        self.min_zoom = 0.05
        self.max_zoom = 20.0
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

class Stage2ControlWidget(QWidget):
    """第二阶段控制组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.simulation = None
        self.topology_file_path = None
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # 标题
        title_label = QLabel("GNN多车协同演示")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("", 14, QFont.Bold))
        title_label.setStyleSheet("color: #2e7d32; padding: 10px;")
        layout.addWidget(title_label)
        
        # 拓扑文件加载
        topo_group = QGroupBox("拓扑文件管理")
        topo_layout = QVBoxLayout()
        
        file_layout = QHBoxLayout()
        self.topo_file_label = QLabel("未选择拓扑文件")
        self.topo_file_label.setStyleSheet("padding: 5px; background: #f5f5f5; border: 1px solid #ddd;")
        self.browse_topo_btn = QPushButton("浏览...")
        self.browse_topo_btn.clicked.connect(self.browse_topology_file)
        
        file_layout.addWidget(self.topo_file_label, 1)
        file_layout.addWidget(self.browse_topo_btn)
        topo_layout.addLayout(file_layout)
        
        self.load_topo_btn = QPushButton("🔄 加载拓扑结构")
        self.load_topo_btn.clicked.connect(self.load_topology)
        self.load_topo_btn.setStyleSheet("QPushButton { padding: 8px; font-weight: bold; }")
        topo_layout.addWidget(self.load_topo_btn)
        
        topo_group.setLayout(topo_layout)
        layout.addWidget(topo_group)
        
        # 仿真参数配置
        config_group = QGroupBox("仿真参数配置")
        config_layout = QVBoxLayout()
        
        # 车辆数量
        vehicle_layout = QHBoxLayout()
        vehicle_layout.addWidget(QLabel("车辆数量:"))
        self.vehicle_count_spin = QSpinBox()
        self.vehicle_count_spin.setRange(1, 12)
        self.vehicle_count_spin.setValue(4)
        vehicle_layout.addWidget(self.vehicle_count_spin)
        vehicle_layout.addStretch()
        config_layout.addLayout(vehicle_layout)
        
        # 调度模式选择
        mode_layout = QHBoxLayout()
        self.gnn_mode_checkbox = QCheckBox("启用GNN智能调度")
        self.gnn_mode_checkbox.setChecked(True)
        self.gnn_mode_checkbox.setStyleSheet("QCheckBox { font-weight: bold; }")
        mode_layout.addWidget(self.gnn_mode_checkbox)
        mode_layout.addStretch()
        config_layout.addLayout(mode_layout)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # 仿真控制
        sim_group = QGroupBox("仿真控制")
        sim_layout = QVBoxLayout()
        
        # 主控制按钮
        main_button_layout = QHBoxLayout()
        
        self.init_sim_btn = QPushButton("🚀 初始化仿真")
        self.init_sim_btn.clicked.connect(self.initialize_simulation)
        self.init_sim_btn.setStyleSheet("QPushButton { padding: 10px; font-weight: bold; background: #4caf50; color: white; }")
        
        self.start_btn = QPushButton("▶️ 开始")
        self.start_btn.clicked.connect(self.start_simulation)
        self.start_btn.setEnabled(False)
        self.start_btn.setStyleSheet("QPushButton { padding: 8px; font-weight: bold; }")
        
        self.pause_btn = QPushButton("⏸️ 暂停")
        self.pause_btn.clicked.connect(self.pause_simulation)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setStyleSheet("QPushButton { padding: 8px; font-weight: bold; }")
        
        main_button_layout.addWidget(self.init_sim_btn)
        main_button_layout.addWidget(self.start_btn)
        main_button_layout.addWidget(self.pause_btn)
        sim_layout.addLayout(main_button_layout)
        
        # 次要控制按钮
        secondary_button_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("🔄 重置")
        self.reset_btn.clicked.connect(self.reset_simulation)
        self.reset_btn.setEnabled(False)
        
        self.add_vehicle_btn = QPushButton("➕ 添加车辆")
        self.add_vehicle_btn.clicked.connect(self.add_vehicle)
        self.add_vehicle_btn.setEnabled(False)
        
        self.remove_vehicle_btn = QPushButton("➖ 移除车辆")
        self.remove_vehicle_btn.clicked.connect(self.remove_vehicle)
        self.remove_vehicle_btn.setEnabled(False)
        
        secondary_button_layout.addWidget(self.reset_btn)
        secondary_button_layout.addWidget(self.add_vehicle_btn)
        secondary_button_layout.addWidget(self.remove_vehicle_btn)
        sim_layout.addLayout(secondary_button_layout)
        
        # 高级控制
        advanced_layout = QHBoxLayout()
        
        self.toggle_gnn_btn = QPushButton("🔄 切换调度模式")
        self.toggle_gnn_btn.clicked.connect(self.toggle_gnn_mode)
        self.toggle_gnn_btn.setEnabled(False)
        
        advanced_layout.addWidget(self.toggle_gnn_btn)
        advanced_layout.addStretch()
        sim_layout.addLayout(advanced_layout)
        
        sim_group.setLayout(sim_layout)
        layout.addWidget(sim_group)
        
        # 仿真状态监控
        status_group = QGroupBox("仿真状态监控")
        status_layout = QVBoxLayout()
        
        self.sim_status_label = QLabel("📊 状态: 未初始化")
        self.sim_status_label.setStyleSheet("padding: 5px; background: #fff3e0; border-left: 4px solid #ff9800;")
        
        self.sim_time_label = QLabel("⏱️ 时间: 0.0s")
        self.sim_time_label.setStyleSheet("padding: 5px;")
        
        self.vehicle_info_label = QLabel("🚛 车辆: 0 | 模式: 未设置")
        self.vehicle_info_label.setStyleSheet("padding: 5px;")
        
        self.performance_label = QLabel("📈 性能: 等待数据...")
        self.performance_label.setStyleSheet("padding: 5px;")
        
        status_layout.addWidget(self.sim_status_label)
        status_layout.addWidget(self.sim_time_label)
        status_layout.addWidget(self.vehicle_info_label)
        status_layout.addWidget(self.performance_label)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # 帮助信息
        help_group = QGroupBox("操作提示")
        help_layout = QVBoxLayout()
        
        help_text = QLabel(
            "💡 <b>工作流程:</b><br/>"
            "1. 选择并加载第一阶段导出的拓扑文件<br/>"
            "2. 配置车辆数量和调度模式<br/>"
            "3. 初始化仿真系统<br/>"
            "4. 开始演示并观察协同效果<br/><br/>"
            "🖱️ <b>视图操作:</b><br/>"
            "• 滚轮: 缩放视图<br/>"
            "• 中键拖拽: 平移视图<br/>"
            "• 右键: 切换选择/平移模式<br/>"
            "• 悬停: 查看详细信息"
        )
        help_text.setWordWrap(True)
        help_text.setStyleSheet("padding: 10px; background: #e3f2fd; border: 1px solid #2196f3; font-size: 11px;")
        help_layout.addWidget(help_text)
        
        help_group.setLayout(help_layout)
        layout.addWidget(help_group)
        
        layout.addStretch()
    
    def browse_topology_file(self):
        """浏览拓扑文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择第一阶段导出的拓扑文件", "", "JSON文件 (*.json);;所有文件 (*)"
        )
        
        if file_path:
            self.topology_file_path = file_path
            filename = os.path.basename(file_path)
            self.topo_file_label.setText(filename)
            self.topo_file_label.setStyleSheet("padding: 5px; background: #e8f5e8; border: 1px solid #4caf50;")
    
    def load_topology(self):
        """加载拓扑"""
        if not self.topology_file_path:
            QMessageBox.warning(self, "警告", "请先选择拓扑文件")
            return
        
        try:
            # 验证文件格式
            with open(self.topology_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data.get('ready_for_stage2', False):
                QMessageBox.warning(self, "警告", 
                    "所选文件不是有效的第二阶段拓扑文件！\n\n"
                    "请确保选择的是第一阶段导出的完整拓扑文件，\n"
                    "该文件应包含 'ready_for_stage2': true 标记。")
                return
            
            # 显示拓扑信息
            info_text = "✅ 拓扑文件验证成功！\n\n"
            info_text += f"📁 文件: {os.path.basename(self.topology_file_path)}\n"
            info_text += f"⏰ 导出时间: {data.get('export_time', '未知')}\n"
            
            if 'enhanced_consolidation_applied' in data:
                info_text += f"🔧 增强版整合: {'是' if data['enhanced_consolidation_applied'] else '否'}\n"
            
            if 'key_nodes_info' in data:
                info_text += f"🎯 关键节点: {len(data['key_nodes_info'])}个\n"
            
            if 'consolidated_paths_info' in data:
                info_text += f"🛤️ 整合路径: {len(data['consolidated_paths_info'])}条\n"
            
            info_text += "\n💡 现在可以进行仿真初始化！"
            
            QMessageBox.information(self, "拓扑加载成功", info_text)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载拓扑失败:\n{str(e)}")
    
    def initialize_simulation(self):
        """初始化仿真"""
        if not self.topology_file_path:
            QMessageBox.warning(self, "警告", "请先选择并加载拓扑文件")
            return
        
        try:
            num_vehicles = self.vehicle_count_spin.value()
            
            # 创建仿真
            self.simulation = Stage2GNNSimulation(
                topology_file_path=self.topology_file_path,
                num_vehicles=num_vehicles
            )
            
            # 设置GNN模式
            use_gnn = self.gnn_mode_checkbox.isChecked()
            self.simulation.use_gnn = use_gnn
            for vehicle in self.simulation.vehicles:
                vehicle.use_gnn = use_gnn
            
            self.sim_status_label.setText("📊 状态: 仿真已初始化")
            self.sim_status_label.setStyleSheet("padding: 5px; background: #e8f5e8; border-left: 4px solid #4caf50;")
            
            # 启用控制按钮
            self.start_btn.setEnabled(True)
            self.reset_btn.setEnabled(True)
            self.add_vehicle_btn.setEnabled(True)
            self.remove_vehicle_btn.setEnabled(True)
            self.toggle_gnn_btn.setEnabled(True)
            
            self.update_vehicle_info()
            
            # 发送信号给可视化组件
            if hasattr(self.parent(), 'visualization_widget'):
                self.parent().visualization_widget.set_simulation(self.simulation)
            
            success_msg = f"🎉 第二阶段仿真初始化成功！\n\n"
            success_msg += f"🚛 车辆数量: {num_vehicles}\n"
            success_msg += f"🧠 调度模式: {'GNN智能调度' if use_gnn else '传统调度'}\n"
            success_msg += f"🎯 特殊点配置:\n"
            success_msg += f"  • 装载点: {len(self.simulation.road_network.loading_points)}个\n"
            success_msg += f"  • 卸载点: {len(self.simulation.road_network.unloading_points)}个\n\n"
            success_msg += f"💡 点击'开始'按钮启动演示！"
            
            QMessageBox.information(self, "初始化成功", success_msg)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"初始化仿真失败:\n{str(e)}")
    
    def start_simulation(self):
        """开始仿真"""
        if not self.simulation:
            return
        
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.sim_status_label.setText("📊 状态: 仿真运行中...")
        self.sim_status_label.setStyleSheet("padding: 5px; background: #e3f2fd; border-left: 4px solid #2196f3;")
        
        # 发送信号给可视化组件
        if hasattr(self.parent(), 'visualization_widget'):
            self.parent().visualization_widget.start_animation()
    
    def pause_simulation(self):
        """暂停仿真"""
        if not self.simulation:
            return
        
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.sim_status_label.setText("📊 状态: 仿真已暂停")
        self.sim_status_label.setStyleSheet("padding: 5px; background: #fff3e0; border-left: 4px solid #ff9800;")
        
        # 发送信号给可视化组件
        if hasattr(self.parent(), 'visualization_widget'):
            self.parent().visualization_widget.stop_animation()
    
    def reset_simulation(self):
        """重置仿真"""
        if not self.simulation:
            return
        
        try:
            self.simulation.reset_simulation()
            self.simulation.current_time = 0.0
            
            # 重置按钮状态
            self.start_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.sim_status_label.setText("📊 状态: 仿真已重置")
            self.sim_status_label.setStyleSheet("padding: 5px; background: #f3e5f5; border-left: 4px solid #9c27b0;")
            
            self.update_vehicle_info()
            self.update_sim_time()
            self.update_performance_info()
            
            # 发送信号给可视化组件
            if hasattr(self.parent(), 'visualization_widget'):
                self.parent().visualization_widget.reset_visualization()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"重置仿真失败:\n{str(e)}")
    
    def add_vehicle(self):
        """添加车辆"""
        if not self.simulation:
            return
        
        try:
            original_count = len(self.simulation.vehicles)
            self.simulation.add_vehicle()
            new_count = len(self.simulation.vehicles)
            
            if new_count > original_count:
                self.update_vehicle_info()
                QMessageBox.information(self, "成功", f"成功添加车辆！当前车辆数: {new_count}")
            
        except Exception as e:
            QMessageBox.warning(self, "警告", f"添加车辆失败:\n{str(e)}")
    
    def remove_vehicle(self):
        """移除车辆"""
        if not self.simulation:
            return
        
        try:
            original_count = len(self.simulation.vehicles)
            self.simulation.remove_vehicle()
            new_count = len(self.simulation.vehicles)
            
            if new_count < original_count:
                self.update_vehicle_info()
                QMessageBox.information(self, "成功", f"成功移除车辆！当前车辆数: {new_count}")
            
        except Exception as e:
            QMessageBox.warning(self, "警告", f"移除车辆失败:\n{str(e)}")
    
    def toggle_gnn_mode(self):
        """切换GNN模式"""
        if not self.simulation:
            return
        
        old_mode = "GNN智能调度" if self.simulation.use_gnn else "传统调度"
        self.simulation.toggle_gnn_mode()
        new_mode = "GNN智能调度" if self.simulation.use_gnn else "传统调度"
        
        self.gnn_mode_checkbox.setChecked(self.simulation.use_gnn)
        self.update_vehicle_info()
        
        QMessageBox.information(self, "调度模式切换", 
            f"调度模式已从'{old_mode}'切换到'{new_mode}'\n\n"
            f"所有车辆的路径规划将重新计算。")
    
    def update_vehicle_info(self):
        """更新车辆信息"""
        if self.simulation:
            count = len(self.simulation.vehicles)
            gnn_status = "GNN智能调度" if self.simulation.use_gnn else "传统调度"
            self.vehicle_info_label.setText(f"🚛 车辆: {count} | 模式: {gnn_status}")
    
    def update_sim_time(self):
        """更新仿真时间"""
        if self.simulation:
            self.sim_time_label.setText(f"⏱️ 时间: {self.simulation.current_time:.1f}s")
    
    def update_performance_info(self):
        """更新性能信息"""
        if self.simulation:
            total_cycles = sum(v.completed_cycles for v in self.simulation.vehicles)
            total_distance = sum(v.total_distance for v in self.simulation.vehicles)
            total_wait_time = sum(v.wait_time for v in self.simulation.vehicles)
            
            self.performance_label.setText(
                f"📈 性能: 完成循环 {total_cycles} | 总距离 {total_distance:.1f} | 等待时间 {total_wait_time:.1f}s"
            )

class OptimizedVisualizationWidget(QWidget):
    """优化的第二阶段PyQt可视化组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.simulation = None
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_visualization)
        self.animation_running = False
        
        # 可视化状态
        self.vehicle_items = {}  # vehicle_id -> QGraphicsItem
        self.special_point_items = {}  # point_id -> QGraphicsItem
        self.network_items = []  # 网络元素
        self.vehicle_labels = {}  # 车辆标签
        self.target_lines = {}  # 目标连线
        self.path_items = {}  # 路径显示
        self.reservation_items = []  # 预留信息显示
        self.predicted_path_items = {}  # 预测路径显示
        
        # 可视化配置
        self.node_colors = self._get_node_color_scheme()
        self.uniform_node_size = 2.0  # 统一节点大小
        self.small_font = QFont("Arial", 3, QFont.Bold)  # 小字体
        self.tiny_font = QFont("Arial", 2, QFont.Bold)  # 更小字体
        
        self.init_ui()
    
    def _get_node_color_scheme(self):
        """获取节点颜色方案 - 根据度数分级"""
        return {
            1: QColor(255, 183, 183),  # 浅红色 - 端点
            2: QColor(183, 223, 255),  # 浅蓝色 - 路径节点
            3: QColor(183, 255, 183),  # 浅绿色 - 分支节点
            4: QColor(255, 223, 183),  # 浅橙色 - 枢纽节点
            5: QColor(223, 183, 255),  # 浅紫色 - 重要节点
            'high': QColor(255, 153, 153)  # 深红色 - 高度数节点
        }
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(4, 4, 4, 4)
        
        # 控制面板
        control_layout = QHBoxLayout()
        
        self.refresh_btn = QPushButton("🔄 刷新")
        self.refresh_btn.clicked.connect(self.refresh_visualization)
        self.refresh_btn.setFixedWidth(60)
        
        self.fit_view_btn = QPushButton("📐 适应")
        self.fit_view_btn.clicked.connect(self.fit_view)
        self.fit_view_btn.setFixedWidth(60)
        
        self.reset_view_btn = QPushButton("🏠 重置")
        self.reset_view_btn.clicked.connect(self.reset_view)
        self.reset_view_btn.setFixedWidth(60)
        
        # 显示选项
        self.show_network_checkbox = QCheckBox("网络")
        self.show_network_checkbox.setChecked(True)
        self.show_network_checkbox.toggled.connect(self.refresh_visualization)
        
        self.show_paths_checkbox = QCheckBox("路径")
        self.show_paths_checkbox.setChecked(True)
        self.show_paths_checkbox.toggled.connect(self.refresh_visualization)
        
        self.show_reservations_checkbox = QCheckBox("预留")
        self.show_reservations_checkbox.setChecked(True)
        self.show_reservations_checkbox.toggled.connect(self.refresh_visualization)
        
        self.show_predicted_paths_checkbox = QCheckBox("预测路径")
        self.show_predicted_paths_checkbox.setChecked(True)
        self.show_predicted_paths_checkbox.toggled.connect(self.refresh_visualization)
        
        self.show_labels_checkbox = QCheckBox("标签")
        self.show_labels_checkbox.setChecked(True)
        self.show_labels_checkbox.toggled.connect(self.refresh_visualization)
        
        control_layout.addWidget(self.refresh_btn)
        control_layout.addWidget(self.fit_view_btn)
        control_layout.addWidget(self.reset_view_btn)
        control_layout.addWidget(QLabel("|"))
        control_layout.addWidget(self.show_network_checkbox)
        control_layout.addWidget(self.show_paths_checkbox)
        control_layout.addWidget(self.show_predicted_paths_checkbox)
        control_layout.addWidget(self.show_reservations_checkbox)
        control_layout.addWidget(self.show_labels_checkbox)
        control_layout.addStretch()
        
        layout.addLayout(control_layout)
        
        # 图形视图
        self.graphics_view = EnhancedGraphicsView()
        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)
        
        layout.addWidget(self.graphics_view, 1)
        
        # 状态信息
        self.status_label = QLabel("等待仿真初始化...")
        self.status_label.setStyleSheet("padding: 4px; background: #f5f5f5; border: 1px solid #ddd; font-size: 12px;")
        layout.addWidget(self.status_label)
        
        # 帮助信息
        help_label = QLabel("🖱️ 滚轮缩放 | 中键拖拽 | 右键切换模式 | 悬停查看详情")
        help_label.setStyleSheet("color: gray; font-size: 10px; padding: 2px;")
        layout.addWidget(help_label)
    
    def set_simulation(self, simulation):
        """设置仿真对象"""
        self.simulation = simulation
        self.reset_visualization()
        self.refresh_visualization()
    
    def start_animation(self):
        """开始动画"""
        if not self.simulation:
            return
        
        self.animation_timer.start(100)  # 10fps
        self.animation_running = True
    
    def stop_animation(self):
        """停止动画"""
        self.animation_timer.stop()
        self.animation_running = False
    
    def reset_visualization(self):
        """重置可视化"""
        self.stop_animation()
        self.scene.clear()
        self.vehicle_items.clear()
        self.special_point_items.clear()
        self.network_items.clear()
        self.vehicle_labels.clear()
        self.target_lines.clear()
        self.path_items.clear()
        self.reservation_items.clear()
        self.predicted_path_items.clear()
    
    def refresh_visualization(self):
        """刷新可视化"""
        if not self.simulation:
            return
        
        self.scene.clear()
        self.vehicle_items.clear()
        self.special_point_items.clear()
        self.network_items.clear()
        self.vehicle_labels.clear()
        self.target_lines.clear()
        self.path_items.clear()
        self.reservation_items.clear()
        self.predicted_path_items.clear()
        
        # 设置场景大小
        self.auto_adjust_scene_size()
        
        # 绘制背景
        self.draw_background()
        
        # 绘制网络
        if self.show_network_checkbox.isChecked():
            self.draw_network()
        
        # 绘制预留信息
        if self.show_reservations_checkbox.isChecked():
            self.draw_reservations()
        
        # 绘制特殊点
        self.draw_special_points()
        
        # 绘制车辆路径
        if self.show_paths_checkbox.isChecked():
            self.draw_vehicle_paths()
        
        # 绘制预测路径
        if self.show_predicted_paths_checkbox.isChecked():
            self.draw_predicted_paths()
        
        # 绘制车辆
        self.draw_vehicles()
        
        # 更新状态
        self.update_status()
        
        # 适应视图
        if not self.animation_running:
            self.fit_view()
    
    def auto_adjust_scene_size(self):
        """自动调整场景大小"""
        if not self.simulation or not self.simulation.road_network:
            self.scene.setSceneRect(0, 0, 500, 500)
            return
        
        positions = list(self.simulation.road_network.node_positions.values())
        if not positions:
            self.scene.setSceneRect(0, 0, 500, 500)
            return
        
        min_x = min(pos[0] for pos in positions)
        max_x = max(pos[0] for pos in positions)
        min_y = min(pos[1] for pos in positions)
        max_y = max(pos[1] for pos in positions)
        
        margin = 15
        self.scene.setSceneRect(
            min_x - margin, min_y - margin,
            max_x - min_x + 2 * margin,
            max_y - min_y + 2 * margin
        )
    
    def draw_background(self):
        """绘制背景"""
        scene_rect = self.scene.sceneRect()
        bg_rect = QGraphicsRectItem(scene_rect)
        bg_rect.setBrush(QBrush(QColor(248, 248, 248)))
        bg_rect.setPen(QPen(Qt.NoPen))
        bg_rect.setZValue(-100)
        self.scene.addItem(bg_rect)
        self.network_items.append(bg_rect)
    
    def draw_network(self):
        """绘制道路网络 - 统一节点大小，颜色区分级别"""
        if not self.simulation or not self.simulation.road_network:
            return
        
        road_network = self.simulation.road_network
        current_time = self.simulation.current_time
        
        # 绘制边
        for edge in road_network.graph.edges():
            node1, node2 = edge
            pos1 = road_network.node_positions[node1]
            pos2 = road_network.node_positions[node2]
            
            # 检查边是否被预留
            edge_key = tuple(sorted([node1, node2]))
            is_reserved = self._is_edge_reserved(edge_key, current_time)
            
            if is_reserved:
                # 被预留的边用更粗的线和不同颜色
                line = QGraphicsLineItem(pos1[0], pos1[1], pos2[0], pos2[1])
                pen = QPen(QColor(255, 87, 34), 1.0)  # 橙色粗线
                pen.setStyle(Qt.DashLine)
                line.setPen(pen)
                line.setZValue(2)
            else:
                # 普通边
                line = QGraphicsLineItem(pos1[0], pos1[1], pos2[0], pos2[1])
                pen = QPen(QColor(200, 200, 200), 0.3)
                line.setPen(pen)
                line.setZValue(1)
            
            self.scene.addItem(line)
            self.network_items.append(line)
        
        # 绘制普通节点 - 统一大小，颜色区分级别
        for node, pos in road_network.node_positions.items():
            # 跳过特殊点
            if self._is_special_node(node):
                continue
            
            degree = road_network.graph.degree(node)
            
            # 根据度数选择颜色
            if degree >= 6:
                color = self.node_colors['high']
            else:
                color = self.node_colors.get(degree, self.node_colors[2])
            
            # 检查节点预留状态
            node_status = self._get_node_reservation_status(node, current_time)
            
            # 根据预留状态调整颜色
            if node_status == 'occupied':
                # 被占用 - 红色边框
                edge_color = QColor(244, 67, 54)
                edge_width = 1.0
            elif node_status == 'reserved':
                # 被预留 - 橙色边框
                edge_color = QColor(255, 152, 0)
                edge_width = 0.8
            elif node_status == 'cooling':
                # 冷却期 - 黄色边框
                edge_color = QColor(255, 235, 59)
                edge_width = 0.6
            else:
                # 空闲 - 灰色边框
                edge_color = QColor(120, 120, 120)
                edge_width = 0.3
            
            # 统一节点大小
            radius = self.uniform_node_size
            circle = QGraphicsEllipseItem(pos[0]-radius, pos[1]-radius, radius*2, radius*2)
            circle.setBrush(QBrush(color))
            circle.setPen(QPen(edge_color, edge_width))
            circle.setZValue(5)
            
            # 工具提示
            tooltip = f"节点: {node}\n度数: {degree}\n状态: {node_status}"
            if node_status != 'free':
                reservations = road_network.node_reservations.get(node, [])
                for r in reservations:
                    if r.start_time <= current_time <= r.end_time:
                        tooltip += f"\n占用车辆: V{r.vehicle_id}"
                        break
            circle.setToolTip(tooltip)
            
            self.scene.addItem(circle)
            self.network_items.append(circle)
            
            # 节点标签 - 小字体
            if self.show_labels_checkbox.isChecked():
                label_text = node[-3:]  # 只显示后3位
                label = QGraphicsTextItem(label_text)
                label.setPos(pos[0]-1.5, pos[1]-3.5)
                label.setDefaultTextColor(QColor(60, 60, 60))
                label.setFont(self.tiny_font)
                label.setZValue(6)
                self.scene.addItem(label)
                self.network_items.append(label)
    
    def draw_reservations(self):
        """绘制预留信息 - 优化显示"""
        if not self.simulation or not self.simulation.road_network:
            return
        
        road_network = self.simulation.road_network
        current_time = self.simulation.current_time
        
        # 绘制边预留
        for edge_key, reservations in road_network.edge_reservations.items():
            if not reservations:
                continue
            
            node1, node2 = edge_key
            pos1 = road_network.node_positions.get(node1)
            pos2 = road_network.node_positions.get(node2)
            
            if not pos1 or not pos2:
                continue
            
            active_reservations = [
                r for r in reservations 
                if r.end_time >= current_time
            ]
            
            for i, reservation in enumerate(active_reservations):
                vehicle = next((v for v in self.simulation.vehicles if v.id == reservation.vehicle_id), None)
                if not vehicle:
                    continue
                
                # 计算偏移以显示多个预留
                offset_factor = (i - len(active_reservations)/2 + 0.5) * 0.5
                offset_x = (pos2[1] - pos1[1]) * offset_factor * 0.02
                offset_y = (pos1[0] - pos2[0]) * offset_factor * 0.02
                
                x1, y1 = pos1[0] + offset_x, pos1[1] + offset_y
                x2, y2 = pos2[0] + offset_x, pos2[1] + offset_y
                
                # 预留线条 - 根据时间状态调整样式
                if reservation.start_time <= current_time <= reservation.end_time:
                    # 当前活动的预留 - 实线
                    line_style = Qt.SolidLine
                    line_width = 2.0
                    alpha = 0.9
                else:
                    # 未来的预留 - 虚线
                    line_style = Qt.DashLine
                    line_width = 1.5
                    alpha = 0.6
                
                line = QGraphicsLineItem(x1, y1, x2, y2)
                vehicle_color = self.get_vehicle_color(vehicle.id)
                vehicle_color.setAlpha(int(255 * alpha))
                pen = QPen(vehicle_color, line_width)
                pen.setStyle(line_style)
                line.setPen(pen)
                line.setZValue(3)
                
                # 工具提示
                status = "活动中" if reservation.start_time <= current_time <= reservation.end_time else "未来"
                line.setToolTip(f"边预留: V{vehicle.id}\n状态: {status}\n时间: {reservation.start_time:.1f}-{reservation.end_time:.1f}s")
                
                self.scene.addItem(line)
                self.reservation_items.append(line)
                
                # 在线条中间标注车辆ID - 小字体
                if reservation.start_time <= current_time <= reservation.end_time:
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    label = QGraphicsTextItem(f'V{vehicle.id}')
                    label.setPos(mid_x-2, mid_y-2)
                    label.setDefaultTextColor(vehicle_color)
                    label.setFont(self.tiny_font)
                    label.setZValue(4)
                    self.scene.addItem(label)
                    self.reservation_items.append(label)
        
        # 绘制节点预留指示器
        for node_id, reservations in road_network.node_reservations.items():
            if not reservations:
                continue
            
            pos = road_network.node_positions.get(node_id)
            if not pos or self._is_special_node(node_id):
                continue
            
            active_reservations = [
                r for r in reservations 
                if r.end_time >= current_time
            ]
            
            for i, reservation in enumerate(active_reservations):
                vehicle = next((v for v in self.simulation.vehicles if v.id == reservation.vehicle_id), None)
                if not vehicle:
                    continue
                
                # 节点预留环形指示器
                if reservation.start_time <= current_time <= reservation.end_time:
                    # 当前活动的预留 - 实心环
                    radius = self.uniform_node_size + 1.0 + i * 0.5
                    ring = QGraphicsEllipseItem(pos[0]-radius, pos[1]-radius, radius*2, radius*2)
                    vehicle_color = self.get_vehicle_color(vehicle.id)
                    ring.setBrush(QBrush(Qt.NoBrush))
                    ring.setPen(QPen(vehicle_color, 1.0))
                    ring.setZValue(4)
                    
                    ring.setToolTip(f"节点预留: V{vehicle.id}\n活动中: {reservation.start_time:.1f}-{reservation.end_time:.1f}s")
                    
                    self.scene.addItem(ring)
                    self.reservation_items.append(ring)
    
    def draw_special_points(self):
        """绘制特殊点 - 保持小字体"""
        if not self.simulation or not self.simulation.road_network:
            return
        
        road_network = self.simulation.road_network
        
        # 绘制装载点 - 绿色方形
        for point_id, point in road_network.loading_points.items():
            pos = road_network.node_positions.get(point.node_id)
            if not pos:
                continue
            
            x, y = pos[0], pos[1]
            
            # 根据状态选择颜色
            if point.is_occupied:
                color = QColor(46, 125, 50)
                border_color = QColor(27, 94, 32)
                status = f"Loading V{point.reserved_by}"
            elif point.reserved_by is not None:
                color = QColor(255, 193, 7)
                border_color = QColor(255, 143, 0)
                status = f"Reserved V{point.reserved_by}"
            else:
                color = QColor(129, 199, 132)
                border_color = QColor(46, 125, 50)
                status = "Available"
            
            # 装载点 - 统一大小
            size = 2.5
            rect = QGraphicsRectItem(x-size, y-size, size*2, size*2)
            rect.setBrush(QBrush(color))
            rect.setPen(QPen(border_color, 0.8))
            rect.setZValue(10)
            rect.setToolTip(f"装载点 {point_id}\n状态: {status}")
            
            self.scene.addItem(rect)
            self.special_point_items[point_id] = rect
            
            # 标签 - 小字体
            if self.show_labels_checkbox.isChecked():
                label_text = f"L{point_id.split('_')[-1]}" if '_' in point_id else point_id[-2:]
                label = QGraphicsTextItem(label_text)
                label.setPos(x-3, y-6)
                label.setDefaultTextColor(border_color)
                label.setFont(self.tiny_font)
                label.setZValue(11)
                self.scene.addItem(label)
                self.network_items.append(label)
        
        # 绘制卸载点 - 蓝色三角形
        for point_id, point in road_network.unloading_points.items():
            pos = road_network.node_positions.get(point.node_id)
            if not pos:
                continue
            
            x, y = pos[0], pos[1]
            
            # 根据状态选择颜色
            if point.is_occupied:
                color = QColor(25, 118, 210)
                border_color = QColor(13, 71, 161)
                status = f"Unloading V{point.reserved_by}"
            elif point.reserved_by is not None:
                color = QColor(255, 193, 7)
                border_color = QColor(255, 143, 0)
                status = f"Reserved V{point.reserved_by}"
            else:
                color = QColor(100, 181, 246)
                border_color = QColor(25, 118, 210)
                status = "Available"
            
            # 卸载点（三角形）- 统一大小
            size = 2.5
            points = [
                QPointF(x, y-size),
                QPointF(x-size, y+size*0.7),
                QPointF(x+size, y+size*0.7)
            ]
            
            polygon = QGraphicsPathItem()
            path = QPainterPath()
            path.addPolygon(QPolygonF(points))
            polygon.setPath(path)
            polygon.setBrush(QBrush(color))
            polygon.setPen(QPen(border_color, 0.8))
            polygon.setZValue(10)
            polygon.setToolTip(f"卸载点 {point_id}\n状态: {status}")
            
            self.scene.addItem(polygon)
            self.special_point_items[point_id] = polygon
            
            # 标签 - 小字体
            if self.show_labels_checkbox.isChecked():
                label_text = f"U{point_id.split('_')[-1]}" if '_' in point_id else point_id[-2:]
                label = QGraphicsTextItem(label_text)
                label.setPos(x-3, y-8)
                label.setDefaultTextColor(border_color)
                label.setFont(self.tiny_font)
                label.setZValue(11)
                self.scene.addItem(label)
                self.network_items.append(label)
    
    def draw_vehicles(self):
        """绘制车辆"""
        if not self.simulation:
            return
        
        for vehicle in self.simulation.vehicles:
            self.create_vehicle_item(vehicle)
    
    def create_vehicle_item(self, vehicle):
        """创建车辆图形项"""
        x, y = vehicle.position[0], vehicle.position[1]
        
        # 获取车辆颜色
        vehicle_color = self.get_vehicle_color(vehicle.id)
        
        # 根据模式确定车辆大小和形状
        if vehicle.mode == VehicleMode.LOADED:
            radius = 2.5
            shape_type = "square"  # 重载用方形
        else:
            radius = 2.0
            shape_type = "circle"  # 空载用圆形
        
        # 根据状态设置边框
        edge_color, edge_width = self._get_vehicle_edge_style(vehicle.state)
        
        # 创建车辆形状
        if shape_type == "square":
            vehicle_item = QGraphicsRectItem(x-radius, y-radius, radius*2, radius*2)
        else:
            vehicle_item = QGraphicsEllipseItem(x-radius, y-radius, radius*2, radius*2)
        
        vehicle_item.setBrush(QBrush(vehicle_color))
        vehicle_item.setPen(QPen(edge_color, edge_width))
        vehicle_item.setZValue(20)
        
        # 获取目标信息 - 适配不同的属性名
        target_info = ""
        target_node = None
        
        # 尝试不同的目标属性名
        if hasattr(vehicle, 'target_node') and vehicle.target_node:
            target_node = vehicle.target_node
            target_info = f" -> {target_node}"
        elif hasattr(vehicle, 'target_loading_point') and vehicle.target_loading_point:
            target_node = vehicle.target_loading_point
            target_info = f" -> {target_node} (装载)"
        elif hasattr(vehicle, 'target_unloading_point') and vehicle.target_unloading_point:
            target_node = vehicle.target_unloading_point
            target_info = f" -> {target_node} (卸载)"
        
        # 工具提示
        tooltip = (f"车辆 V{vehicle.id}\n"
                  f"状态: {vehicle.state.value}\n"
                  f"模式: {vehicle.mode.value}{target_info}\n"
                  f"位置: ({x:.1f}, {y:.1f})\n"
                  f"完成循环: {vehicle.completed_cycles}")
        vehicle_item.setToolTip(tooltip)
        
        self.scene.addItem(vehicle_item)
        self.vehicle_items[vehicle.id] = vehicle_item
        
        # 车辆标签 - 小字体
        if self.show_labels_checkbox.isChecked():
            label = QGraphicsTextItem(f"V{vehicle.id}")
            label.setPos(x - 3, y - radius - 4)
            label.setDefaultTextColor(QColor(0, 0, 0))
            label.setFont(self.small_font)
            label.setZValue(21)
            self.scene.addItem(label)
            self.vehicle_labels[vehicle.id] = label
        
        # 存储目标节点用于绘制连线
        if target_node:
            setattr(vehicle, '_gui_target_node', target_node)
    
    def draw_vehicle_paths(self):
        """绘制车辆路径"""
        if not self.simulation:
            return
        
        for vehicle in self.simulation.vehicles:
            # 绘制目标连线
            target_node = getattr(vehicle, '_gui_target_node', None)
            if target_node:
                self.draw_vehicle_target_line(vehicle, target_node)
            
            # 绘制已确认路径
            if (vehicle.state == VehicleState.CONFIRMED and 
                hasattr(vehicle, 'path') and vehicle.path):
                self.draw_confirmed_path(vehicle)
    
    def draw_vehicle_target_line(self, vehicle, target_node):
        """绘制车辆到目标的连线"""
        road_network = self.simulation.road_network
        target_pos = road_network.node_positions.get(target_node)
        if not target_pos:
            return
        
        x1, y1 = vehicle.position[0], vehicle.position[1]
        x2, y2 = target_pos[0], target_pos[1]
        
        # 根据模式选择线条颜色和样式
        if vehicle.mode == VehicleMode.EMPTY:
            line_color = QColor(46, 125, 50)  # 绿色 - 前往装载点
            line_style = Qt.DashLine
        else:
            line_color = QColor(25, 118, 210)  # 蓝色 - 前往卸载点
            line_style = Qt.DashLine
        
        line = QGraphicsLineItem(x1, y1, x2, y2)
        pen = QPen(line_color, 1.0)
        pen.setStyle(line_style)
        line.setPen(pen)
        line.setZValue(15)
        
        self.scene.addItem(line)
        self.target_lines[vehicle.id] = line
    
    def draw_predicted_paths(self):
        """绘制车辆预测路径"""
        if not self.simulation:
            return
        
        for vehicle in self.simulation.vehicles:
            # 为规划中的车辆绘制预测路径
            if vehicle.state == VehicleState.PLANNING:
                target_node = getattr(vehicle, '_gui_target_node', None)
                if target_node:
                    self.draw_vehicle_predicted_path(vehicle, target_node)
    
    def draw_vehicle_predicted_path(self, vehicle, target_node):
        """绘制车辆预测路径"""
        road_network = self.simulation.road_network
        target_pos = road_network.node_positions.get(target_node)
        if not target_pos:
            return
        
        # 尝试获取简单路径
        try:
            simple_path = road_network.simple_pathfinding(vehicle.current_node, target_node)
            if simple_path and len(simple_path) > 1:
                path_positions = []
                for node in simple_path:
                    if node in road_network.node_positions:
                        pos = road_network.node_positions[node]
                        path_positions.append(QPointF(pos[0], pos[1]))
                
                if len(path_positions) >= 2:
                    # 创建预测路径
                    path = QPainterPath()
                    path.moveTo(path_positions[0])
                    for pos in path_positions[1:]:
                        path.lineTo(pos)
                    
                    path_item = QGraphicsPathItem(path)
                    vehicle_color = self.get_vehicle_color(vehicle.id)
                    vehicle_color.setAlpha(128)  # 半透明
                    pen = QPen(vehicle_color, 1.0)
                    pen.setStyle(Qt.DotLine)  # 点线表示预测
                    path_item.setPen(pen)
                    path_item.setZValue(7)
                    
                    self.scene.addItem(path_item)
                    self.predicted_path_items[vehicle.id] = path_item
                    
        except Exception:
            pass  # 如果无法获取路径，就不显示预测路径
    
    def draw_confirmed_path(self, vehicle):
        """绘制已确认的路径"""
        if not hasattr(vehicle, 'path') or not vehicle.path or len(vehicle.path) < 2:
            return
        
        road_network = self.simulation.road_network
        path_positions = []
        
        for node in vehicle.path:
            if node in road_network.node_positions:
                pos = road_network.node_positions[node]
                path_positions.append(QPointF(pos[0], pos[1]))
        
        if len(path_positions) < 2:
            return
        
        # 创建路径
        path = QPainterPath()
        path.moveTo(path_positions[0])
        for pos in path_positions[1:]:
            path.lineTo(pos)
        
        path_item = QGraphicsPathItem(path)
        vehicle_color = self.get_vehicle_color(vehicle.id)
        pen = QPen(vehicle_color, 2.0)
        pen.setStyle(Qt.SolidLine)
        path_item.setPen(pen)
        path_item.setZValue(8)
        
        self.scene.addItem(path_item)
        self.path_items[vehicle.id] = path_item
        
        # 路径节点标记
        for i, pos in enumerate(path_positions):
            if i % 2 == 0:  # 只显示部分节点
                node_mark = QGraphicsEllipseItem(pos.x()-0.8, pos.y()-0.8, 1.6, 1.6)
                node_mark.setBrush(QBrush(vehicle_color))
                node_mark.setPen(QPen(Qt.NoPen))
                node_mark.setZValue(9)
                self.scene.addItem(node_mark)
                self.network_items.append(node_mark)
    
    def update_visualization(self):
        """更新可视化（动画回调）"""
        if not self.simulation:
            return
        
        # 更新仿真状态
        dt = 0.1
        self.simulation.current_time += dt
        self.simulation.road_network.update_time(self.simulation.current_time)
        
        for vehicle in self.simulation.vehicles:
            vehicle.update(self.simulation.current_time, dt)
        
        # 更新车辆位置和状态
        self.update_vehicles()
        
        # 更新特殊点状态
        self.update_special_points()
        
        # 更新预留信息
        if self.show_reservations_checkbox.isChecked():
            self.update_reservations()
        
        # 更新状态信息
        self.update_status()
        
        # 通知控制组件更新时间和性能
        if hasattr(self.parent(), 'control_widget'):
            self.parent().control_widget.update_sim_time()
            self.parent().control_widget.update_performance_info()
    
    def update_vehicles(self):
        """更新车辆位置和状态"""
        if not self.simulation:
            return
        
        # 清除旧的目标线和路径
        for line in self.target_lines.values():
            self.scene.removeItem(line)
        self.target_lines.clear()
        
        for path_item in self.path_items.values():
            self.scene.removeItem(path_item)
        self.path_items.clear()
        
        for pred_path in self.predicted_path_items.values():
            self.scene.removeItem(pred_path)
        self.predicted_path_items.clear()
        
        for vehicle in self.simulation.vehicles:
            if vehicle.id in self.vehicle_items:
                # 移除旧的车辆项
                self.scene.removeItem(self.vehicle_items[vehicle.id])
                if vehicle.id in self.vehicle_labels:
                    self.scene.removeItem(self.vehicle_labels[vehicle.id])
                
                # 重新创建车辆项
                self.create_vehicle_item(vehicle)
        
        # 重新绘制路径
        if self.show_paths_checkbox.isChecked():
            self.draw_vehicle_paths()
        
        if self.show_predicted_paths_checkbox.isChecked():
            self.draw_predicted_paths()
    
    def update_special_points(self):
        """更新特殊点状态"""
        if not self.simulation or not self.simulation.road_network:
            return
        
        road_network = self.simulation.road_network
        
        # 更新装载点
        for point_id, point in road_network.loading_points.items():
            if point_id in self.special_point_items:
                item = self.special_point_items[point_id]
                
                if point.is_occupied:
                    color = QColor(46, 125, 50)
                    status = f"Loading V{point.reserved_by}"
                elif point.reserved_by is not None:
                    color = QColor(255, 193, 7)
                    status = f"Reserved V{point.reserved_by}"
                else:
                    color = QColor(129, 199, 132)
                    status = "Available"
                
                item.setBrush(QBrush(color))
                item.setToolTip(f"装载点 {point_id}\n状态: {status}")
        
        # 更新卸载点
        for point_id, point in road_network.unloading_points.items():
            if point_id in self.special_point_items:
                item = self.special_point_items[point_id]
                
                if point.is_occupied:
                    color = QColor(25, 118, 210)
                    status = f"Unloading V{point.reserved_by}"
                elif point.reserved_by is not None:
                    color = QColor(255, 193, 7)
                    status = f"Reserved V{point.reserved_by}"
                else:
                    color = QColor(100, 181, 246)
                    status = "Available"
                
                item.setBrush(QBrush(color))
                item.setToolTip(f"卸载点 {point_id}\n状态: {status}")
    
    def update_reservations(self):
        """更新预留信息显示"""
        # 清除旧的预留显示
        for item in self.reservation_items:
            self.scene.removeItem(item)
        self.reservation_items.clear()
        
        # 重新绘制预留信息
        self.draw_reservations()
    
    def _is_special_node(self, node_id):
        """检查是否为特殊节点"""
        if not self.simulation or not self.simulation.road_network:
            return False
        
        road_network = self.simulation.road_network
        return any(
            point.node_id == node_id 
            for points in [road_network.loading_points, road_network.unloading_points]
            for point in points.values()
        )
    
    def _is_edge_reserved(self, edge_key, current_time):
        """检查边是否被预留"""
        if not self.simulation or not self.simulation.road_network:
            return False
        
        reservations = self.simulation.road_network.edge_reservations.get(edge_key, [])
        return any(r.end_time >= current_time for r in reservations)
    
    def _get_node_reservation_status(self, node_id, current_time):
        """获取节点预留状态"""
        if not self.simulation or not self.simulation.road_network:
            return 'free'
        
        reservations = self.simulation.road_network.node_reservations.get(node_id, [])
        
        for r in reservations:
            if r.start_time <= current_time <= r.end_time:
                return 'occupied'
            elif current_time < r.start_time:
                return 'reserved'
            elif current_time - r.end_time <= 0.3:  # 冷却期
                return 'cooling'
        
        return 'free'
    
    def _get_vehicle_edge_style(self, state):
        """根据车辆状态获取边框样式"""
        if state == VehicleState.LOADING:
            return QColor(46, 125, 50), 1.5  # 绿色
        elif state == VehicleState.UNLOADING:
            return QColor(25, 118, 210), 1.5  # 蓝色
        elif state == VehicleState.MOVING:
            return QColor(255, 255, 255), 1.0  # 白色
        elif state == VehicleState.WAITING:
            return QColor(244, 67, 54), 1.0  # 红色
        elif state == VehicleState.CONFIRMED:
            return QColor(255, 193, 7), 1.5  # 金色
        else:
            return QColor(0, 0, 0), 0.5  # 黑色
    
    def get_vehicle_color(self, vehicle_id):
        """获取车辆颜色"""
        colors = [
            QColor(244, 67, 54),    # 红色
            QColor(33, 150, 243),   # 蓝色
            QColor(76, 175, 80),    # 绿色
            QColor(255, 152, 0),    # 橙色
            QColor(156, 39, 176),   # 紫色
            QColor(121, 85, 72),    # 棕色
            QColor(233, 30, 99),    # 粉色
            QColor(0, 188, 212),    # 青色
            QColor(139, 195, 74),   # 浅绿色
            QColor(255, 87, 34),    # 深橙色
        ]
        return colors[vehicle_id % len(colors)]
    
    def update_status(self):
        """更新状态信息"""
        if not self.simulation:
            self.status_label.setText("等待仿真初始化...")
            return
        
        # 统计信息
        total_vehicles = len(self.simulation.vehicles)
        total_cycles = sum(v.completed_cycles for v in self.simulation.vehicles)
        gnn_mode = "GNN智能调度" if self.simulation.use_gnn else "传统调度"
        
        # 车辆状态统计
        state_counts = {}
        mode_counts = {}
        for vehicle in self.simulation.vehicles:
            state = vehicle.state.value
            mode = vehicle.mode.value
            state_counts[state] = state_counts.get(state, 0) + 1
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        # 特殊点使用情况
        road_network = self.simulation.road_network
        loading_busy = sum(1 for p in road_network.loading_points.values() if p.is_occupied)
        unloading_busy = sum(1 for p in road_network.unloading_points.values() if p.is_occupied)
        
        # 预留统计
        total_edge_reservations = sum(len(reservations) for reservations in 
                                    road_network.edge_reservations.values())
        total_node_reservations = sum(len(reservations) for reservations in 
                                    road_network.node_reservations.values())
        
        status_text = (f"⏱️ {self.simulation.current_time:.1f}s | "
                      f"🚛 {total_vehicles}辆 | "
                      f"🧠 {gnn_mode} | "
                      f"🔄 完成{total_cycles}循环 | "
                      f"🟢 {loading_busy}装载 🔵 {unloading_busy}卸载 | "
                      f"📊 预留: 边{total_edge_reservations} 节点{total_node_reservations}")
        
        self.status_label.setText(status_text)
    
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
    
    def update_visualization(self):
        """更新可视化（动画回调）"""
        if not self.simulation:
            return
        
        # 更新仿真状态
        dt = 0.1
        self.simulation.current_time += dt
        self.simulation.road_network.update_time(self.simulation.current_time)
        
        for vehicle in self.simulation.vehicles:
            vehicle.update(self.simulation.current_time, dt)
        
        # 更新车辆位置和状态
        self.update_vehicles()
        
        # 更新特殊点状态
        self.update_special_points()
        
        # 更新预留信息
        if self.show_reservations_checkbox.isChecked():
            self.update_reservations()
        
        # 更新状态信息
        self.update_status()
        
        # 通知控制组件更新时间和性能
        if hasattr(self.parent(), 'control_widget'):
            self.parent().control_widget.update_sim_time()
            self.parent().control_widget.update_performance_info()
    
    def update_vehicles(self):
        """更新车辆位置和状态"""
        if not self.simulation:
            return
        
        # 清除旧的目标线和路径
        for line in self.target_lines.values():
            self.scene.removeItem(line)
        self.target_lines.clear()
        
        for path_item in self.path_items.values():
            self.scene.removeItem(path_item)
        self.path_items.clear()
        
        for pred_path in self.predicted_path_items.values():
            self.scene.removeItem(pred_path)
        self.predicted_path_items.clear()
        
        for vehicle in self.simulation.vehicles:
            if vehicle.id in self.vehicle_items:
                # 移除旧的车辆项
                self.scene.removeItem(self.vehicle_items[vehicle.id])
                if vehicle.id in self.vehicle_labels:
                    self.scene.removeItem(self.vehicle_labels[vehicle.id])
                
                # 重新创建车辆项
                self.create_vehicle_item(vehicle)
        
        # 重新绘制路径
        if self.show_paths_checkbox.isChecked():
            self.draw_vehicle_paths()
        
        if self.show_predicted_paths_checkbox.isChecked():
            self.draw_predicted_paths()
    
    def update_special_points(self):
        """更新特殊点状态"""
        if not self.simulation or not self.simulation.road_network:
            return
        
        road_network = self.simulation.road_network
        
        # 更新装载点
        for point_id, point in road_network.loading_points.items():
            if point_id in self.special_point_items:
                item = self.special_point_items[point_id]
                
                if point.is_occupied:
                    color = QColor(46, 125, 50)
                    status = f"Loading V{point.reserved_by}"
                elif point.reserved_by is not None:
                    color = QColor(255, 193, 7)
                    status = f"Reserved V{point.reserved_by}"
                else:
                    color = QColor(129, 199, 132)
                    status = "Available"
                
                item.setBrush(QBrush(color))
                item.setToolTip(f"装载点 {point_id}\n状态: {status}")
        
        # 更新卸载点
        for point_id, point in road_network.unloading_points.items():
            if point_id in self.special_point_items:
                item = self.special_point_items[point_id]
                
                if point.is_occupied:
                    color = QColor(25, 118, 210)
                    status = f"Unloading V{point.reserved_by}"
                elif point.reserved_by is not None:
                    color = QColor(255, 193, 7)
                    status = f"Reserved V{point.reserved_by}"
                else:
                    color = QColor(100, 181, 246)
                    status = "Available"
                
                item.setBrush(QBrush(color))
                item.setToolTip(f"卸载点 {point_id}\n状态: {status}")
    
    def update_reservations(self):
        """更新预留信息显示"""
        # 清除旧的预留显示
        for item in self.reservation_items:
            self.scene.removeItem(item)
        self.reservation_items.clear()
        
        # 重新绘制预留信息
        self.draw_reservations()
    
    def _is_special_node(self, node_id):
        """检查是否为特殊节点"""
        if not self.simulation or not self.simulation.road_network:
            return False
        
        road_network = self.simulation.road_network
        return any(
            point.node_id == node_id 
            for points in [road_network.loading_points, road_network.unloading_points]
            for point in points.values()
        )
    
    def _is_edge_reserved(self, edge_key, current_time):
        """检查边是否被预留"""
        if not self.simulation or not self.simulation.road_network:
            return False
        
        reservations = self.simulation.road_network.edge_reservations.get(edge_key, [])
        return any(r.end_time >= current_time for r in reservations)
    
    def _get_node_reservation_status(self, node_id, current_time):
        """获取节点预留状态"""
        if not self.simulation or not self.simulation.road_network:
            return 'free'
        
        reservations = self.simulation.road_network.node_reservations.get(node_id, [])
        
        for r in reservations:
            if r.start_time <= current_time <= r.end_time:
                return 'occupied'
            elif current_time < r.start_time:
                return 'reserved'
            elif current_time - r.end_time <= 0.3:  # 冷却期
                return 'cooling'
        
        return 'free'
    
    def _get_vehicle_edge_style(self, state):
        """根据车辆状态获取边框样式"""
        if state == VehicleState.LOADING:
            return QColor(46, 125, 50), 1.5  # 绿色
        elif state == VehicleState.UNLOADING:
            return QColor(25, 118, 210), 1.5  # 蓝色
        elif state == VehicleState.MOVING:
            return QColor(255, 255, 255), 1.0  # 白色
        elif state == VehicleState.WAITING:
            return QColor(244, 67, 54), 1.0  # 红色
        elif state == VehicleState.CONFIRMED:
            return QColor(255, 193, 7), 1.5  # 金色
        else:
            return QColor(0, 0, 0), 0.5  # 黑色
    
    def get_vehicle_color(self, vehicle_id):
        """获取车辆颜色"""
        colors = [
            QColor(244, 67, 54),    # 红色
            QColor(33, 150, 243),   # 蓝色
            QColor(76, 175, 80),    # 绿色
            QColor(255, 152, 0),    # 橙色
            QColor(156, 39, 176),   # 紫色
            QColor(121, 85, 72),    # 棕色
            QColor(233, 30, 99),    # 粉色
            QColor(0, 188, 212),    # 青色
            QColor(139, 195, 74),   # 浅绿色
            QColor(255, 87, 34),    # 深橙色
        ]
        return colors[vehicle_id % len(colors)]
    
    def update_status(self):
        """更新状态信息"""
        if not self.simulation:
            self.status_label.setText("等待仿真初始化...")
            return
        
        # 统计信息
        total_vehicles = len(self.simulation.vehicles)
        total_cycles = sum(v.completed_cycles for v in self.simulation.vehicles)
        gnn_mode = "GNN智能调度" if self.simulation.use_gnn else "传统调度"
        
        # 车辆状态统计
        state_counts = {}
        mode_counts = {}
        for vehicle in self.simulation.vehicles:
            state = vehicle.state.value
            mode = vehicle.mode.value
            state_counts[state] = state_counts.get(state, 0) + 1
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        # 特殊点使用情况
        road_network = self.simulation.road_network
        loading_busy = sum(1 for p in road_network.loading_points.values() if p.is_occupied)
        unloading_busy = sum(1 for p in road_network.unloading_points.values() if p.is_occupied)
        
        # 预留统计
        total_edge_reservations = sum(len(reservations) for reservations in 
                                    road_network.edge_reservations.values())
        total_node_reservations = sum(len(reservations) for reservations in 
                                    road_network.node_reservations.values())
        
        status_text = (f"⏱️ {self.simulation.current_time:.1f}s | "
                      f"🚛 {total_vehicles}辆 | "
                      f"🧠 {gnn_mode} | "
                      f"🔄 完成{total_cycles}循环 | "
                      f"🟢 {loading_busy}装载 🔵 {unloading_busy}卸载 | "
                      f"📊 预留: 边{total_edge_reservations} 节点{total_node_reservations}")
        
        self.status_label.setText(status_text)
    
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

class Stage2DemoMainWindow(QMainWindow):
    """第二阶段Demo主窗口"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        print("🚀 第二阶段 GNN多车协同演示系统启动成功")
    
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("第二阶段：基于拓扑感知GNN的多车协同演示系统")
        self.setGeometry(100, 100, 1400, 900)
        
        # 设置应用图标和样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #fafafa;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #f5f5f5;
                border-color: #999999;
            }
            QPushButton:pressed {
                background-color: #e0e0e0;
            }
            QPushButton:disabled {
                background-color: #f5f5f5;
                color: #999999;
            }
        """)
        
        # 中央组件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(8, 8, 8, 8)
        
        # 左侧控制面板
        self.control_widget = Stage2ControlWidget()
        self.control_widget.setMaximumWidth(350)
        self.control_widget.setMinimumWidth(320)
        main_layout.addWidget(self.control_widget)
        
        # 右侧可视化面板 - 使用优化的可视化组件
        self.visualization_widget = OptimizedVisualizationWidget()
        main_layout.addWidget(self.visualization_widget, 1)
        
        # 建立组件间的连接
        self.control_widget.parent = lambda: self
        self.visualization_widget.parent = lambda: self
        
        # 创建状态栏
        self.status_bar = self.statusBar()
        self.status_label = QLabel("🎯 第二阶段GNN多车协同演示系统 - 就绪")
        self.status_bar.addWidget(self.status_label)
        
        # 创建菜单栏
        self.create_menu_bar()
    
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('文件(&F)')
        
        open_action = file_menu.addAction('📁 打开拓扑文件...')
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.control_widget.browse_topology_file)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction('🚪 退出')
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        
        # 仿真菜单
        sim_menu = menubar.addMenu('仿真(&S)')
        
        init_action = sim_menu.addAction('🚀 初始化仿真')
        init_action.setShortcut('Ctrl+I')
        init_action.triggered.connect(self.control_widget.initialize_simulation)
        
        start_action = sim_menu.addAction('▶️ 开始仿真')
        start_action.setShortcut('Space')
        start_action.triggered.connect(self.control_widget.start_simulation)
        
        pause_action = sim_menu.addAction('⏸️ 暂停仿真')
        pause_action.setShortcut('Ctrl+P')
        pause_action.triggered.connect(self.control_widget.pause_simulation)
        
        reset_action = sim_menu.addAction('🔄 重置仿真')
        reset_action.setShortcut('Ctrl+R')
        reset_action.triggered.connect(self.control_widget.reset_simulation)
        
        sim_menu.addSeparator()
        
        toggle_gnn_action = sim_menu.addAction('🧠 切换GNN模式')
        toggle_gnn_action.setShortcut('Ctrl+G')
        toggle_gnn_action.triggered.connect(self.control_widget.toggle_gnn_mode)
        
        # 视图菜单
        view_menu = menubar.addMenu('视图(&V)')
        
        refresh_action = view_menu.addAction('🔄 刷新视图')
        refresh_action.setShortcut('F5')
        refresh_action.triggered.connect(self.visualization_widget.refresh_visualization)
        
        fit_action = view_menu.addAction('📐 适应视图')
        fit_action.setShortcut('Ctrl+F')
        fit_action.triggered.connect(self.visualization_widget.fit_view)
        
        reset_view_action = view_menu.addAction('🏠 重置视图')
        reset_view_action.setShortcut('Ctrl+H')
        reset_view_action.triggered.connect(self.visualization_widget.reset_view)
        
        # 帮助菜单
        help_menu = menubar.addMenu('帮助(&H)')
        
        about_action = help_menu.addAction('ℹ️ 关于...')
        about_action.triggered.connect(self.show_about)
    
    def show_about(self):
        """显示关于对话框"""
        about_text = """
        <h2>🎯 第二阶段：GNN多车协同演示系统</h2>
        <p><b>版本:</b> 1.0.0 - 优化版</p>
        <p><b>描述:</b> 基于拓扑感知GNN架构的露天矿智能调度系统第二阶段演示</p>
        
        <h3>🚛 核心功能:</h3>
        <ul>
        <li>载入第一阶段导出的拓扑结构</li>
        <li>完整循环作业：装载→卸载→装载</li>
        <li>GNN感知路径规划与传统模式对比</li>
        <li>实时多车协同调度与冲突避免</li>
        <li>高质量PyQt可视化系统</li>
        </ul>
        
        <h3>🎨 可视化优化:</h3>
        <ul>
        <li><b>统一节点大小</b>：所有节点保持一致大小，用颜色区分级别</li>
        <li><b>预留状态显示</b>：清晰显示边和节点的预留情况</li>
        <li><b>车辆预测路径</b>：显示规划中的车辆路径</li>
        <li><b>小字体设计</b>：保持界面简洁，信息密度高</li>
        <li><b>颜色分级</b>：根据节点度数使用不同颜色</li>
        </ul>
        
        <h3>🎯 节点颜色方案:</h3>
        <ul>
        <li>浅红色：端点节点 (度数=1)</li>
        <li>浅蓝色：路径节点 (度数=2)</li>
        <li>浅绿色：分支节点 (度数=3)</li>
        <li>浅橙色：枢纽节点 (度数=4)</li>
        <li>浅紫色：重要节点 (度数=5)</li>
        <li>深红色：高度数节点 (度数≥6)</li>
        </ul>
        
        <h3>🖱️ 操作指南:</h3>
        <ul>
        <li>鼠标滚轮: 缩放视图</li>
        <li>中键拖拽: 平移视图</li>
        <li>右键: 切换选择/平移模式</li>
        <li>悬停: 查看详细信息</li>
        </ul>
        
        <h3>⌨️ 快捷键:</h3>
        <ul>
        <li>Ctrl+O: 打开拓扑文件</li>
        <li>Ctrl+I: 初始化仿真</li>
        <li>Space: 开始/暂停仿真</li>
        <li>Ctrl+R: 重置仿真</li>
        <li>Ctrl+G: 切换GNN模式</li>
        <li>F5: 刷新视图</li>
        </ul>
        
        <p><i>基于第一阶段智能拓扑构建结果的多车协同演示系统 - 优化版</i></p>
        """
        
        QMessageBox.about(self, "关于第二阶段GNN演示系统", about_text)
    
    def closeEvent(self, event):
        """关闭事件"""
        reply = QMessageBox.question(
            self, '确认退出',
            '确定要退出第二阶段GNN多车协同演示系统吗？',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 停止动画
            if hasattr(self, 'visualization_widget'):
                self.visualization_widget.stop_animation()
            event.accept()
        else:
            event.ignore()

def main():
    """主函数"""
    if not STAGE2_AVAILABLE:
        print("❌ 第二阶段Demo组件不可用，请确保demo_stage2.py在同一目录下")
        return
    
    app = QApplication(sys.argv)
    app.setApplicationName("第二阶段GNN多车协同演示系统")
    app.setApplicationVersion("Stage 2 - GNN Multi-Vehicle Coordination Demo - Optimized")
    
    try:
        # 创建主窗口
        main_window = Stage2DemoMainWindow()
        main_window.show()
        
        print("🎯 第二阶段GNN多车协同演示系统启动成功 - 优化版")
        print("📋 系统特性:")
        print("\n🚛 多车协同演示:")
        print("  • 完整循环作业：装载→卸载→装载")
        print("  • GNN智能调度 vs 传统调度对比")
        print("  • 实时冲突避免和路径优化")
        print("  • 动态车辆管理（添加/移除）")
        print("\n🎨 可视化优化:")
        print("  • 统一节点大小，颜色区分级别")
        print("  • 清晰显示预留状态（边和节点）")
        print("  • 车辆预测路径和确认路径")
        print("  • 小字体设计，简洁界面")
        print("  • 实时动画：平滑车辆移动")
        print("\n📊 监控功能:")
        print("  • 实时性能统计")
        print("  • 车辆状态分布")
        print("  • 特殊点使用情况")
        print("  • 预留状态监控")
        print("  • 完成循环计数")
        print("\n🔧 操作流程:")
        print("  1. 选择第一阶段导出的拓扑文件")
        print("  2. 配置车辆数量和调度模式")
        print("  3. 初始化仿真系统")
        print("  4. 开始演示并观察协同效果")
        print("  5. 实时调整参数和模式")
        print("\n💡 提示: 使用菜单栏或快捷键进行快速操作")
        print("✨ 新增: 统一节点大小、预留状态显示、预测路径等优化功能")
        
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"❌ 应用程序启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()