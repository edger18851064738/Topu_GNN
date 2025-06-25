"""
optimized_backbone_network.py - 集成增强版ClothoidCubic和EnhancedNodeClusteringConsolidator
✨ 修复版本：解决了端点聚类方法调用问题
"""

import math
import time
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import threading

# 简化的管理器类 - 避免依赖问题
class SimpleInterfaceManager:
    """简化的接口管理器"""
    def __init__(self):
        self.reservations = {}
    
    def reserve_interface(self, interface_id, vehicle_id, current_time, duration):
        self.reservations[interface_id] = {
            'vehicle_id': vehicle_id,
            'start_time': current_time,
            'duration': duration
        }
        return True
    
    def release_interface(self, interface_id, vehicle_id):
        if interface_id in self.reservations:
            del self.reservations[interface_id]
    
    def cleanup_expired_reservations(self, current_time):
        expired = []
        for interface_id, reservation in self.reservations.items():
            if current_time - reservation['start_time'] > reservation['duration']:
                expired.append(interface_id)
        
        for interface_id in expired:
            del self.reservations[interface_id]
    
    def get_interface_congestion_factor(self, interface_id):
        return 1.2 if interface_id in self.reservations else 1.0

class SimpleStabilityManager:
    """简化的稳定性管理器"""
    def __init__(self):
        self.vehicle_switches = {}
    
    def can_vehicle_switch(self, vehicle_id):
        return True  # 简化版总是允许切换
    
    def record_vehicle_switch(self, vehicle_id, path_id):
        self.vehicle_switches[vehicle_id] = path_id
    
    def get_stability_report(self):
        return {
            'overall_stability': 0.9,
            'vehicle_switches': len(self.vehicle_switches)
        }

class SimpleSafetyManager:
    """简化的安全管理器"""
    def __init__(self):
        self.vehicle_safety_params = {}
    
    def is_interface_safe_for_vehicle(self, position, vehicle_id):
        return True  # 简化版总是安全
    
    def calculate_interface_safety_score(self, position, vehicle_id):
        return 0.9  # 简化版返回固定分数

@dataclass
class BiDirectionalPath:
    """双向路径数据结构 - 增强版"""
    path_id: str
    point_a: Dict  # 起点信息
    point_b: Dict  # 终点信息
    forward_path: List[Tuple]  # A->B路径
    reverse_path: List[Tuple]  # B->A路径（自动生成）
    length: float
    quality: float
    planner_used: str
    created_time: float
    usage_count: int = 0
    current_load: int = 0  # 当前使用该路径的车辆数
    max_capacity: int = 5  # 最大容量
    
    # 新增：质量历史追踪
    quality_history: List[float] = None
    last_quality_update: float = 0.0
    
    # 新增：增强版专业设计相关标记
    is_professional_design: bool = False
    is_optimized_design: bool = False
    road_class: Optional[str] = None  # "primary", "secondary", "service"
    design_class: str = "standard"
    consolidation_level: str = "original"  # "original", "consolidated", "enhanced_node_clustering_professional"
    
    # 新增：节点聚类相关属性
    key_nodes: List[str] = None  # 关键节点ID列表
    is_node_clustered: bool = False
    node_reduction_ratio: float = 0.0
    
    # 新增：工程标准
    engineering_standards: Dict = None
    node_spacing: float = 15.0
    safety_rating: float = 1.0
    meets_standards: bool = True
    
    def __post_init__(self):
        if self.quality_history is None:
            self.quality_history = [self.quality]
        if self.engineering_standards is None:
            self.engineering_standards = {}
        if self.key_nodes is None:
            self.key_nodes = []
    
    def get_path(self, from_point_type: str, from_point_id: int, 
                to_point_type: str, to_point_id: int) -> Optional[List[Tuple]]:
        """获取指定方向的路径"""
        # 检查是否匹配A->B方向
        if (self.point_a['type'] == from_point_type and self.point_a['id'] == from_point_id and
            self.point_b['type'] == to_point_type and self.point_b['id'] == to_point_id):
            return self.forward_path
        
        # 检查是否匹配B->A方向
        if (self.point_b['type'] == from_point_type and self.point_b['id'] == from_point_id and
            self.point_a['type'] == to_point_type and self.point_a['id'] == to_point_id):
            return self.reverse_path
        
        return None
    
    def increment_usage(self):
        """增加使用计数"""
        self.usage_count += 1
    
    def add_vehicle(self, vehicle_id: str):
        """添加车辆到路径"""
        self.current_load += 1
    
    def remove_vehicle(self, vehicle_id: str):
        """从路径移除车辆"""
        self.current_load = max(0, self.current_load - 1)
    
    def get_load_factor(self) -> float:
        """获取负载因子"""
        return self.current_load / self.max_capacity
    
    def update_quality_history(self, new_quality: float):
        """更新质量历史"""
        self.quality_history.append(new_quality)
        self.last_quality_update = time.time()
        
        # 限制历史长度
        if len(self.quality_history) > 20:
            self.quality_history = self.quality_history[-10:]
    
    def get_average_quality(self) -> float:
        """获取平均质量"""
        if not self.quality_history:
            return self.quality
        return sum(self.quality_history) / len(self.quality_history)

# 增强版专业道路整合模块导入
try:
    from node_clustering_professional_consolidator import (
        EnhancedNodeClusteringConsolidator,
        NodeClusteringConsolidator,  # 向后兼容
        RoadClass,
        KeyNode,
        EnhancedConsolidatedBackbonePath,
        create_enhanced_node_clustering_consolidator,
        apply_enhanced_consolidation
    )
    ENHANCED_PROFESSIONAL_CONSOLIDATION_AVAILABLE = True
    print("✅ 增强版专业道路整合模块加载成功")
except ImportError as e:
    # 回退到基础版本
    try:
        from node_clustering_professional_consolidator import (
            NodeClusteringConsolidator,
            RoadClass,
            KeyNode
        )
        ENHANCED_PROFESSIONAL_CONSOLIDATION_AVAILABLE = False
        print(f"⚠️ 增强版不可用，使用基础版本: {e}")
    except ImportError:
        ENHANCED_PROFESSIONAL_CONSOLIDATION_AVAILABLE = False
        print(f"❌ 专业道路整合模块完全不可用: {e}")

class OptimizedBackboneNetwork:
    """集成增强版专业道路整合的优化骨干路径网络"""
    
    def __init__(self, env):
        self.env = env
        self.path_planner = None
        
        # 核心数据结构
        self.bidirectional_paths = {}
        self.special_points = {'loading': [], 'unloading': [], 'parking': []}
        
        # 第一阶段状态跟踪 - 增强版
        self.stage1_progress = {
            'current_step': 0,
            'total_steps': 5,
            'step_names': [
                '双向路径智能规划',
                '动态节点密度控制', 
                '关键节点聚类提取',
                '车辆动力学约束拟合',
                '图拓扑标准化输出'
            ],
            'completed_steps': [],
            'step_details': {},
            'step_status': {},  # 新增：每个步骤的状态
            'can_execute_manually': {  # 新增：手动执行能力
                '双向路径智能规划': True,
                '动态节点密度控制': True,
                '关键节点聚类提取': False,  # 需要前两步完成
                '车辆动力学约束拟合': False,  # 需要聚类完成
                '图拓扑标准化输出': False     # 需要拟合完成
            }
        }
        
        # 原始路径状态（聚类前）
        self.raw_paths_state = {
            'generated': False,
            'paths_count': 0,
            'total_nodes': 0,
            'avg_path_length': 0.0,
            'avg_quality': 0.0,
            'generation_time': 0.0,
            'detailed_info': {}  # 新增：详细信息
        }
        
        # 增强版专业整合器
        self.professional_consolidator = None
        self.enhanced_professional_consolidator = None  # 新增：增强版整合器
        self.professional_design_applied = False
        self.original_paths_backup = {}
        self.professional_design_info = {}
        
        # 接口系统
        self.backbone_interfaces = {}
        self.path_interfaces = defaultdict(list)
        
        # 路径查找索引
        self.connection_index = {}
        
        # 负载均衡追踪
        self.vehicle_path_assignments = {}
        self.path_load_history = defaultdict(list)
        
        # 简化的管理器
        self.interface_manager = SimpleInterfaceManager()
        self.stability_manager = SimpleStabilityManager()
        self.safe_interface_manager = SimpleSafetyManager()
        
        # 增强配置
        self.config = {
            'primary_quality_threshold': 0.7,
            'fallback_quality_threshold': 0.4,
            'max_planning_time_per_path': 20.0,
            'enable_progressive_fallback': True,
            'interface_spacing': 8,
            'retry_with_relaxed_params': True,
            'load_balancing_weight': 0.3,
            'path_switching_threshold': 0.8,
            'interface_reservation_duration': 60.0,
            
            # 增强版配置
            'enable_enhanced_professional_consolidation': ENHANCED_PROFESSIONAL_CONSOLIDATION_AVAILABLE,
            'enhanced_design_mode': 'professional',  # 'professional', 'balanced', 'performance'
            'enable_enhanced_curve_fitting': True,
            'force_vehicle_dynamics': True,
            'curve_fitting_quality_threshold': 0.7,
            'preserve_intermediate_states': True,  # 保留中间状态用于可视化
            
            # 传统配置保持兼容
            'professional_design_mode': 'balanced',
            'clustering_radius': 12.0,
            'enable_path_reconstruction': True,
            'enable_node_optimization': True,
            'preserve_original_backup': True,
            'auto_consolidate_after_generation': True,
            
            'enable_safety_optimization': True,
            'enable_stability_management': True,
            'safety_margin': 3.0,
        }
        
        # 增强统计信息
        self.stats = {
            'total_path_pairs': 0,
            'successful_paths': 0,
            'astar_success': 0,
            'rrt_success': 0,
            'direct_fallback': 0,
            'generation_time': 0,
            'loading_to_unloading': 0,
            'loading_to_parking': 0,
            'unloading_to_parking': 0,
            'load_balancing_decisions': 0,
            'path_switches': 0,
            'interface_reservations': 0,
            'safety_optimizations': 0,
            
            # 增强版统计
            'enhanced_consolidation_applied': False,
            'enhanced_curve_fitting_used': False,
            'original_paths_count': 0,
            'consolidated_paths_count': 0,
            'consolidation_time': 0.0,
            'enhanced_professional_design_applied': False,
            
            # 详细的第一阶段统计
            'stage1_step_times': {},
            'stage1_step_quality': {},
            'stage1_intermediate_states': {},
            
            # 节点聚类统计 - 增强版
            'original_nodes_count': 0,
            'key_nodes_count': 0,
            'node_reduction_ratio': 0.0,
            'reconstruction_success_rate': 0.0,
            'enhanced_reconstruction_success_rate': 0.0,
            
            # 曲线拟合统计 - 增强版
            'enhanced_curve_fitting_success': 0,
            'complete_curve_success': 0,
            'segmented_curve_success': 0,
            'fallback_reconstruction': 0,
            'avg_curve_quality': 0.0,
            'dynamics_compliance_rate': 0.0,
            'turning_radius_violations': 0,
            'grade_violations': 0,
            
            # 专业设计统计
            'primary_roads_count': 0,
            'secondary_roads_count': 0,
            'service_roads_count': 0,
            'average_safety_rating': 0.0,
            'standards_compliance_rate': 0.0,
            'total_construction_cost': 0.0,
        }
        
        print("初始化增强版集成的优化骨干路径网络")
        print(f"  增强版专业整合功能: {'✅' if ENHANCED_PROFESSIONAL_CONSOLIDATION_AVAILABLE else '❌'}")
    
    # ==================== 增强版第一阶段核心实现 ====================
    
    def generate_backbone_network(self, quality_threshold: float = None, 
                                enable_consolidation: bool = None) -> bool:
        """
        生成骨干网络 - 增强版第一阶段：智能拓扑构建
        """
        start_time = time.time()
        print("🚀 开始增强版第一阶段：智能拓扑构建")
        
        # 更新配置
        if quality_threshold is not None:
            self.config['primary_quality_threshold'] = quality_threshold
        
        if enable_consolidation is not None:
            self.config['auto_consolidate_after_generation'] = enable_consolidation
        
        try:
            # === 步骤1-2: 双向路径智能规划 + 动态节点密度控制 ===
            success = self._execute_step1_and_step2()
            if not success:
                return False
            
            # 记录原始路径状态（聚类前的状态）
            self._record_enhanced_raw_paths_state(start_time)
            
            # === 可选：执行增强版专业整合（步骤3-5）===
            if self.config['auto_consolidate_after_generation']:
                consolidation_success = self._execute_enhanced_remaining_stage1_steps()
                if not consolidation_success:
                    print("⚠️ 增强版专业整合失败，但原始路径仍可用")
            else:
                print("📋 增强版专业整合已禁用，可手动执行剩余步骤")
                self._mark_steps_ready_for_manual_execution()
            
            # 更新最终统计
            generation_time = time.time() - start_time
            self.stats.update({
                'successful_paths': len(self.bidirectional_paths),
                'generation_time': generation_time
            })
            
            success_rate = len(self.bidirectional_paths) / max(1, self.stats['total_path_pairs'])
            
            print(f"\n🎉 增强版第一阶段：智能拓扑构建完成!")
            print(f"  执行步骤: {len(self.stage1_progress['completed_steps'])}/{self.stage1_progress['total_steps']}")
            print(f"  双向路径: {len(self.bidirectional_paths)} 条")
            print(f"  成功率: {success_rate:.1%}")
            print(f"  总耗时: {generation_time:.2f}s")
            print(f"  增强版整合: {'✅' if self.stats['enhanced_consolidation_applied'] else '❌'}")
            
            return True
        
        except Exception as e:
            print(f"❌ 增强版第一阶段构建失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _execute_step1_and_step2(self) -> bool:
        """执行步骤1和2：双向路径智能规划 + 动态节点密度控制"""
        step_start = time.time()
        
        # === 步骤1: 双向路径智能规划 ===
        self._update_enhanced_stage1_progress(1, "双向路径智能规划", "执行中")
        
        # 加载特殊点
        self._load_special_points()
        if not self._validate_special_points():
            self._update_enhanced_stage1_progress(1, "双向路径智能规划", "失败")
            return False
        
        # 生成所有路径组合
        success_count = self._generate_complete_bidirectional_paths()
        
        if success_count == 0:
            print("❌ 没有成功生成任何骨干路径")
            self._update_enhanced_stage1_progress(1, "双向路径智能规划", "失败")
            return False
        
        step1_time = time.time() - step_start
        self.stats['stage1_step_times']['双向路径智能规划'] = step1_time
        self._update_enhanced_stage1_progress(1, "双向路径智能规划", "完成")
        
        # === 步骤2: 动态节点密度控制 ===
        step2_start = time.time()
        self._update_enhanced_stage1_progress(2, "动态节点密度控制", "执行中")
        
        total_interfaces = self._generate_safe_interfaces()
        
        # 建立连接索引
        self._build_connection_index()
        
        # 初始化质量追踪
        self._initialize_quality_tracking()
        
        step2_time = time.time() - step2_start
        self.stats['stage1_step_times']['动态节点密度控制'] = step2_time
        self._update_enhanced_stage1_progress(2, "动态节点密度控制", "完成")
        
        print(f"\n✅ 第一阶段前两步完成！")
        print(f"  双向路径: {len(self.bidirectional_paths)} 条")
        print(f"  总节点数: 计算中...")
        print(f"  安全接口数量: {total_interfaces} 个")
        print(f"  步骤1耗时: {step1_time:.2f}s, 步骤2耗时: {step2_time:.2f}s")
        
        return True
    
    def _execute_enhanced_remaining_stage1_steps(self) -> bool:
        """执行增强版第一阶段剩余步骤（步骤3-5）"""
        try:
            if len(self.bidirectional_paths) < 2:
                print("路径数量过少，跳过增强版专业整合...")
                return True
            
            print(f"\n🔧 开始应用增强版专业道路整合...")
            consolidation_start = time.time()
            
            # 备份原始路径
            if self.config['preserve_original_backup']:
                self.original_paths_backup = self.bidirectional_paths.copy()
            
            # === 步骤3: 关键节点聚类提取 ===
            step3_success = self._execute_enhanced_step3()
            if not step3_success:
                return False
            
            # === 步骤4: 车辆动力学约束拟合 ===
            step4_success = self._execute_enhanced_step4()
            if not step4_success:
                return False
            
            # === 步骤5: 图拓扑标准化输出 ===
            step5_success = self._execute_enhanced_step5()
            if not step5_success:
                return False
            
            # 更新整合统计信息
            consolidation_time = time.time() - consolidation_start
            self._update_enhanced_consolidation_stats(consolidation_time)
            
            print(f"✅ 增强版第一阶段完整构建成功")
            print(f"   增强版整合耗时: {consolidation_time:.2f}s")
            
            return True
        
        except Exception as e:
            print(f"❌ 增强版第一阶段剩余步骤执行异常: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _execute_enhanced_step3(self) -> bool:
        """增强版步骤3: 关键节点聚类提取"""
        step_start = time.time()
        self._update_enhanced_stage1_progress(3, "关键节点聚类提取", "执行中")
        
        try:
            # 创建增强版整合器
            design_mode = self.config['enhanced_design_mode']
            enhanced_config = self._get_enhanced_professional_config_by_mode(design_mode)
            
            if ENHANCED_PROFESSIONAL_CONSOLIDATION_AVAILABLE:
                self.enhanced_professional_consolidator = EnhancedNodeClusteringConsolidator(
                    self.env, enhanced_config
                )
                # 保持传统接口兼容
                self.professional_consolidator = self.enhanced_professional_consolidator
                
                # ✨ 调试：验证方法存在
                required_methods = [
                    '_extract_original_paths',
                    '_identify_and_cluster_endpoints_optimized',  # ✅ 修复后的方法名
                    '_perform_multi_round_clustering', 
                    '_generate_key_nodes',
                    '_enhanced_reconstruct_backbone_paths',
                    '_apply_consolidation_to_backbone'
                ]
                
                for method_name in required_methods:
                    if not hasattr(self.professional_consolidator, method_name):
                        print(f"❌ 缺少方法: {method_name}")
                        return False
                    else:
                        print(f"✅ 方法存在: {method_name}")
                        
            else:
                # 回退到基础版本
                self.professional_consolidator = NodeClusteringConsolidator(
                    self.env, enhanced_config
                )
            
            # 执行聚类阶段
            success = self._execute_clustering_phase()
            
            if success:
                step3_time = time.time() - step_start
                self.stats['stage1_step_times']['关键节点聚类提取'] = step3_time
                self._update_enhanced_stage1_progress(3, "关键节点聚类提取", "完成")
                print(f"✅ 步骤3完成，耗时: {step3_time:.2f}s")
                return True
            else:
                self._update_enhanced_stage1_progress(3, "关键节点聚类提取", "失败")
                return False
        
        except Exception as e:
            print(f"❌ 增强版步骤3异常: {e}")
            import traceback
            traceback.print_exc()
            self._update_enhanced_stage1_progress(3, "关键节点聚类提取", "失败")
            return False
    
    def _execute_enhanced_step4(self) -> bool:
        """增强版步骤4: 车辆动力学约束拟合"""
        step_start = time.time()
        self._update_enhanced_stage1_progress(4, "车辆动力学约束拟合", "执行中")
        
        try:
            # 执行增强版路径重建
            success = self._execute_enhanced_curve_fitting_phase()
            
            if success:
                step4_time = time.time() - step_start
                self.stats['stage1_step_times']['车辆动力学约束拟合'] = step4_time
                self._update_enhanced_stage1_progress(4, "车辆动力学约束拟合", "完成")
                print(f"✅ 步骤4完成，耗时: {step4_time:.2f}s")
                return True
            else:
                self._update_enhanced_stage1_progress(4, "车辆动力学约束拟合", "失败")
                return False
        
        except Exception as e:
            print(f"❌ 增强版步骤4异常: {e}")
            self._update_enhanced_stage1_progress(4, "车辆动力学约束拟合", "失败")
            return False
    
    def _execute_enhanced_step5(self) -> bool:
        """增强版步骤5: 图拓扑标准化输出"""
        step_start = time.time()
        self._update_enhanced_stage1_progress(5, "图拓扑标准化输出", "执行中")
        
        try:
            # 应用整合结果到骨干网络
            success = self._apply_enhanced_consolidation_results()
            
            if success:
                step5_time = time.time() - step_start
                self.stats['stage1_step_times']['图拓扑标准化输出'] = step5_time
                self._update_enhanced_stage1_progress(5, "图拓扑标准化输出", "完成")
                print(f"✅ 步骤5完成，耗时: {step5_time:.2f}s")
                return True
            else:
                self._update_enhanced_stage1_progress(5, "图拓扑标准化输出", "失败")
                return False
        
        except Exception as e:
            print(f"❌ 增强版步骤5异常: {e}")
            self._update_enhanced_stage1_progress(5, "图拓扑标准化输出", "失败")
            return False
    
    def _execute_clustering_phase(self) -> bool:
        """✅ 修复：执行聚类阶段"""
        if not self.professional_consolidator:
            return False
        
        # 提取原始路径
        success = self.professional_consolidator._extract_original_paths(self)
        if not success:
            return False
        
        # ✨ 修复：使用新的优化端点聚类方法
        success = self.professional_consolidator._identify_and_cluster_endpoints_optimized()
        if not success:
            return False
        
        # 执行多轮聚类
        success = self.professional_consolidator._perform_multi_round_clustering()
        if not success:
            return False
        
        # 生成关键节点
        success = self.professional_consolidator._generate_key_nodes()
        return success
    
    def _execute_enhanced_curve_fitting_phase(self) -> bool:
        """执行增强版曲线拟合阶段"""
        if not self.professional_consolidator:
            return False
        
        # 检查是否是增强版整合器
        if hasattr(self.professional_consolidator, '_enhanced_reconstruct_backbone_paths'):
            # 使用增强版重建方法
            success = self.professional_consolidator._enhanced_reconstruct_backbone_paths()
            if success:
                self.stats['enhanced_curve_fitting_used'] = True
                return True
        
        # 如果没有增强版方法，返回False
        print("       ⚠️ 增强版曲线拟合方法不可用")
        return False
    
    def _apply_enhanced_consolidation_results(self) -> bool:
        """应用增强版整合结果"""
        if not self.professional_consolidator:
            return False
        
        # 应用整合结果到骨干网络
        if hasattr(self.professional_consolidator, '_apply_consolidation_to_backbone'):
            return self.professional_consolidator._apply_consolidation_to_backbone(self)
        
        return False
    
    # ==================== 手动执行单个步骤接口 ====================
    
    def execute_single_stage1_step(self, step_name: str) -> bool:
        """执行单个第一阶段步骤 - 增强版"""
        print(f"🎯 手动执行步骤: {step_name}")
        
        # 检查步骤是否可以执行
        if not self._can_execute_step(step_name):
            print(f"❌ 步骤 '{step_name}' 当前不可执行，请先完成前置步骤")
            return False
        
        try:
            if step_name == "双向路径智能规划":
                return self._manual_execute_step1_2()
            
            elif step_name == "动态节点密度控制":
                return self._manual_execute_step1_2()
            
            elif step_name == "关键节点聚类提取":
                return self._execute_enhanced_step3()
            
            elif step_name == "车辆动力学约束拟合":
                return self._execute_enhanced_step4()
            
            elif step_name == "图拓扑标准化输出":
                return self._execute_enhanced_step5()
            
            else:
                print(f"❌ 未知步骤: {step_name}")
                return False
        
        except Exception as e:
            print(f"❌ 手动执行步骤失败: {e}")
            return False
    
    def _manual_execute_step1_2(self) -> bool:
        """手动执行步骤1和2"""
        if not self.raw_paths_state['generated']:
            return self._execute_step1_and_step2()
        else:
            print("✅ 步骤1-2已完成")
            return True
    
    def _can_execute_step(self, step_name: str) -> bool:
        """检查步骤是否可以执行"""
        if step_name in ["双向路径智能规划", "动态节点密度控制"]:
            return True
        
        if step_name == "关键节点聚类提取":
            return self.raw_paths_state['generated']
        
        if step_name == "车辆动力学约束拟合":
            return (self.professional_consolidator and 
                    hasattr(self.professional_consolidator, 'key_nodes') and
                    self.professional_consolidator.key_nodes)
        
        if step_name == "图拓扑标准化输出":
            return (self.professional_consolidator and 
                    hasattr(self.professional_consolidator, 'consolidated_paths') and
                    self.professional_consolidator.consolidated_paths)
        
        return False
    
    # ==================== 增强版状态管理和查询接口 ====================
    
    def get_raw_backbone_paths_info(self) -> Dict:
        """获取原始骨干路径信息（聚类前状态）- 增强版"""
        if not self.raw_paths_state['generated']:
            return {'status': 'not_generated'}
        
        # 提取路径详细信息
        paths_info = {}
        for path_id, path_data in self.bidirectional_paths.items():
            paths_info[path_id] = {
                'path_id': path_id,
                'start_point': path_data.point_a,
                'end_point': path_data.point_b,
                'forward_path': path_data.forward_path,
                'reverse_path': path_data.reverse_path,
                'length': path_data.length,
                'quality': path_data.quality,
                'planner_used': path_data.planner_used,
                'node_count': len(path_data.forward_path),
                'is_professional': getattr(path_data, 'is_professional_design', False),
                'consolidation_level': getattr(path_data, 'consolidation_level', 'original'),
                'created_time': path_data.created_time
            }
        
        return {
            'status': 'generated',
            'raw_state': self.raw_paths_state,
            'paths_info': paths_info,
            'interfaces_info': self.backbone_interfaces,
            'special_points': self.special_points,
            'generation_stats': {
                'total_path_pairs': self.stats['total_path_pairs'],
                'successful_paths': self.stats['successful_paths'],
                'success_rate': self.stats['successful_paths'] / max(1, self.stats['total_path_pairs']),
                'astar_success': self.stats['astar_success'],
                'rrt_success': self.stats['rrt_success'],
                'direct_fallback': self.stats['direct_fallback'],
                'generation_time': self.raw_paths_state['generation_time']
            }
        }
    
    def get_topology_construction_summary(self) -> Dict:
        """获取拓扑构建摘要信息 - 增强版"""
        summary = {
            'stage1_progress': self.stage1_progress,
            'construction_stats': {
                'paths_generated': len(self.bidirectional_paths),
                'total_nodes': self.raw_paths_state.get('total_nodes', 0),
                'avg_quality': self.raw_paths_state.get('avg_quality', 0.0),
                'interfaces_count': len(self.backbone_interfaces),
                'step_times': self.stats.get('stage1_step_times', {})
            },
            'ready_for_stage2': False,
            'gnn_input_ready': False,
            'enhanced_features': {
                'enhanced_consolidation_available': ENHANCED_PROFESSIONAL_CONSOLIDATION_AVAILABLE,
                'enhanced_consolidation_applied': self.stats['enhanced_consolidation_applied'],
                'enhanced_curve_fitting_used': self.stats['enhanced_curve_fitting_used']
            }
        }
        
        # 检查是否已完成增强版专业整合（准备进入第二阶段）
        if self.stats['enhanced_consolidation_applied']:
            summary.update({
                'ready_for_stage2': True,
                'gnn_input_ready': True,
                'consolidation_stats': self._get_enhanced_consolidation_summary()
            })
        
        return summary
    
    def _get_enhanced_consolidation_summary(self) -> Dict:
        """获取增强版整合摘要"""
        if not self.professional_consolidator:
            return {}
        
        # 从增强版整合器获取统计
        if hasattr(self.professional_consolidator, 'get_consolidation_stats'):
            stats = self.professional_consolidator.get_consolidation_stats()
            
            summary = {
                'key_nodes_count': stats.get('key_nodes_count', 0),
                'node_reduction_ratio': stats.get('node_reduction_ratio', 0.0),
                'reconstruction_success_rate': stats.get('reconstruction_success_rate', 0.0),
                'consolidation_time': self.stats.get('consolidation_time', 0.0)
            }
            
            # 增强版特有统计
            if ENHANCED_PROFESSIONAL_CONSOLIDATION_AVAILABLE:
                summary.update({
                    'enhanced_curve_fitting_success': stats.get('enhanced_curve_fitting_used', 0),
                    'complete_curve_success': stats.get('complete_curve_success', 0),
                    'avg_curve_quality': stats.get('avg_curve_quality', 0.0),
                    'dynamics_compliance_rate': stats.get('dynamics_compliance_rate', 0.0),
                    'turning_radius_violations': stats.get('turning_radius_violations', 0),
                    'grade_violations': stats.get('grade_violations', 0),
                    
                    # ✨ 端点聚类统计
                    'endpoint_reduction_ratio': stats.get('endpoint_reduction_ratio', 0.0),
                    'endpoint_clusters_count': stats.get('endpoint_clusters_count', 0),
                    'merged_endpoint_paths': stats.get('merged_endpoint_paths', 0)
                })
            
            return summary
        
        return {}
    
    # ==================== 辅助方法 ====================
    
    def _update_enhanced_stage1_progress(self, step_num: int, step_description: str, status: str):
        """更新增强版第一阶段进度"""
        self.stage1_progress['current_step'] = step_num
        step_name = self.stage1_progress['step_names'][step_num - 1]
        
        # 更新状态
        self.stage1_progress['step_status'][step_name] = status
        
        if status == "完成" and step_name not in self.stage1_progress['completed_steps']:
            self.stage1_progress['completed_steps'].append(step_name)
        
        self.stage1_progress['step_details'][step_name] = {
            'description': step_description,
            'timestamp': time.time(),
            'status': status
        }
        
        # 更新手动执行能力
        self._update_manual_execution_capabilities()
        
        print(f"📋 第一阶段 步骤{step_num}: {step_name} - {status}")
    
    def _update_manual_execution_capabilities(self):
        """更新手动执行能力"""
        completed = set(self.stage1_progress['completed_steps'])
        
        # 步骤3需要步骤1-2完成
        if '双向路径智能规划' in completed and '动态节点密度控制' in completed:
            self.stage1_progress['can_execute_manually']['关键节点聚类提取'] = True
        
        # 步骤4需要步骤3完成
        if '关键节点聚类提取' in completed:
            self.stage1_progress['can_execute_manually']['车辆动力学约束拟合'] = True
        
        # 步骤5需要步骤4完成
        if '车辆动力学约束拟合' in completed:
            self.stage1_progress['can_execute_manually']['图拓扑标准化输出'] = True
    
    def _mark_steps_ready_for_manual_execution(self):
        """标记步骤为可手动执行"""
        self._update_manual_execution_capabilities()
        print("📋 剩余步骤已准备好手动执行")
    
    def _record_enhanced_raw_paths_state(self, start_time):
        """记录增强版原始路径状态（聚类前）"""
        # 统计节点数
        total_nodes = 0
        total_length = 0
        total_quality = 0
        detailed_info = {}
        
        for path_id, path_data in self.bidirectional_paths.items():
            if path_data.forward_path:
                nodes_count = len(path_data.forward_path)
                total_nodes += nodes_count
                total_length += path_data.length
                total_quality += path_data.quality
                
                detailed_info[path_id] = {
                    'nodes_count': nodes_count,
                    'length': path_data.length,
                    'quality': path_data.quality,
                    'planner_used': path_data.planner_used
                }
        
        paths_count = len(self.bidirectional_paths)
        
        self.raw_paths_state = {
            'generated': True,
            'paths_count': paths_count,
            'total_nodes': total_nodes,
            'avg_path_length': total_length / max(1, paths_count),
            'avg_quality': total_quality / max(1, paths_count),
            'generation_time': time.time() - start_time,
            'detailed_info': detailed_info
        }
        
        # 记录原始路径统计
        self.stats['original_paths_count'] = paths_count
        self.stats['original_nodes_count'] = total_nodes
        
        print(f"📊 增强版原始路径状态已记录: {paths_count}条路径, {total_nodes}个节点")
    
    def _get_enhanced_professional_config_by_mode(self, design_mode: str) -> Dict:
        """根据设计模式获取增强版专业配置"""
        mode_configs = {
            'professional': {
                'enable_enhanced_curve_fitting': True,
                'curve_fitting_quality_threshold': 0.8,
                'force_vehicle_dynamics': True,
                'prefer_complete_curve': True,
                'enable_endpoint_clustering': True,  # ✨ 启用端点聚类
                'endpoint_clustering_radius': 2.0,    # ✨ 端点聚类半径
                'vehicle_dynamics': {
                    'turning_radius': 8.0,
                    'max_grade': 0.12,  # 更严格
                    'safety_margin': 2.0
                },
                'clustering_rounds': [
                    {'radius': 8.0, 'name': '第一轮'},
                    {'radius': 6.0, 'name': '第二轮'},
                    {'radius': 3.0, 'name': '第三轮'}
                ]
            },
            'balanced': {
                'enable_enhanced_curve_fitting': True,
                'curve_fitting_quality_threshold': 0.7,
                'force_vehicle_dynamics': True,
                'prefer_complete_curve': True,
                'enable_endpoint_clustering': True,  # ✨ 启用端点聚类
                'endpoint_clustering_radius': 2.0,    # ✨ 端点聚类半径
                'vehicle_dynamics': {
                    'turning_radius': 8.0,
                    'max_grade': 0.15,
                    'safety_margin': 1.5
                },
                'clustering_rounds': [
                    {'radius': 6.0, 'name': '第一轮'},
                    {'radius': 6.0, 'name': '第二轮'},
                    {'radius': 3.0, 'name': '第三轮'}
                ]
            },
            'performance': {
                'enable_enhanced_curve_fitting': True,
                'curve_fitting_quality_threshold': 0.6,
                'force_vehicle_dynamics': True,
                'prefer_complete_curve': False,
                'enable_endpoint_clustering': True,  # ✨ 启用端点聚类
                'endpoint_clustering_radius': 2.5,    # ✨ 端点聚类半径（更大）
                'vehicle_dynamics': {
                    'turning_radius': 7.0,
                    'max_grade': 0.18,
                    'safety_margin': 1.2
                },
                'clustering_rounds': [
                    {'radius': 6.0, 'name': '第一轮'},
                    {'radius': 3.0, 'name': '第二轮'}
                ]
            }
        }
        
        config = mode_configs.get(design_mode, mode_configs['balanced'])
        
        # 添加通用配置
        config.update({
            'protect_endpoints': True,
            'endpoint_buffer_radius': 3.0,
            'min_cluster_size': 1,
            'importance_threshold': 1.5,
            'preserve_original_on_failure': True,
            'enable_quality_reporting': True,
        })
        
        return config
    
    def _update_enhanced_consolidation_stats(self, consolidation_time: float):
        """更新增强版整合统计信息"""
        if not self.professional_consolidator:
            return
        
        # 获取整合统计
        if hasattr(self.professional_consolidator, 'get_consolidation_stats'):
            consolidation_stats = self.professional_consolidator.get_consolidation_stats()
            
            self.stats.update({
                'enhanced_consolidation_applied': True,
                'consolidated_paths_count': len(self.bidirectional_paths),
                'consolidation_time': consolidation_time,
                'enhanced_professional_design_applied': True,
                'key_nodes_count': consolidation_stats.get('key_nodes_count', 0),
                'node_reduction_ratio': consolidation_stats.get('node_reduction_ratio', 0.0),
                'reconstruction_success_rate': consolidation_stats.get('reconstruction_success_rate', 0.0),
            })
            
            # 增强版特有统计
            if ENHANCED_PROFESSIONAL_CONSOLIDATION_AVAILABLE:
                self.stats.update({
                    'enhanced_curve_fitting_success': consolidation_stats.get('enhanced_curve_fitting_used', 0),
                    'complete_curve_success': consolidation_stats.get('complete_curve_success', 0),
                    'segmented_curve_success': consolidation_stats.get('segmented_curve_success', 0),
                    'fallback_reconstruction': consolidation_stats.get('fallback_reconstruction', 0),
                    'avg_curve_quality': consolidation_stats.get('avg_curve_quality', 0.0),
                    'dynamics_compliance_rate': consolidation_stats.get('dynamics_compliance_rate', 0.0),
                    'turning_radius_violations': consolidation_stats.get('turning_radius_violations', 0),
                    'grade_violations': consolidation_stats.get('grade_violations', 0),
                })
        
        # 存储专业设计信息
        self.professional_design_info = {
            'consolidation_stats': consolidation_stats if 'consolidation_stats' in locals() else {},
            'is_enhanced_professional_design': ENHANCED_PROFESSIONAL_CONSOLIDATION_AVAILABLE,
            'design_type': 'enhanced_node_clustering_professional_consolidation',
            'consolidation_time': consolidation_time,
            'design_mode': self.config['enhanced_design_mode'],
        }
        
        # 获取详细信息
        if hasattr(self.professional_consolidator, 'get_key_nodes_info'):
            self.professional_design_info['key_nodes_info'] = self.professional_consolidator.get_key_nodes_info()
        
        if hasattr(self.professional_consolidator, 'get_consolidated_paths_info'):
            self.professional_design_info['consolidated_paths_info'] = self.professional_consolidator.get_consolidated_paths_info()
        
        # ✨ 获取端点聚类信息
        if hasattr(self.professional_consolidator, 'get_endpoint_clustering_info'):
            self.professional_design_info['endpoint_clustering_info'] = self.professional_consolidator.get_endpoint_clustering_info()
        
        # 标记所有路径为增强专业设计
        for path_data in self.bidirectional_paths.values():
            path_data.is_professional_design = True
            path_data.is_optimized_design = True
            path_data.consolidation_level = "enhanced_node_clustering_professional"
            if hasattr(path_data, 'key_nodes'):
                path_data.is_node_clustered = True
                path_data.node_reduction_ratio = consolidation_stats.get('node_reduction_ratio', 0.0)
        
        self.professional_design_applied = True
        
        print(f"📊 增强版整合统计已更新")
        if ENHANCED_PROFESSIONAL_CONSOLIDATION_AVAILABLE:
            print(f"   原始节点: {consolidation_stats.get('original_nodes_count', 0)} -> 关键节点: {consolidation_stats.get('key_nodes_count', 0)}")
            print(f"   节点减少: {consolidation_stats.get('node_reduction_ratio', 0.0):.1%}")
            print(f"   ✨ 端点减少: {consolidation_stats.get('endpoint_reduction_ratio', 0.0):.1%}")
            print(f"   增强拟合成功: {consolidation_stats.get('enhanced_curve_fitting_used', 0)} 次")
            print(f"   动力学合规率: {consolidation_stats.get('dynamics_compliance_rate', 0.0):.1%}")
    
    # ==================== 保持原有方法（向后兼容） ====================
    
    def _load_special_points(self):
        """加载特殊点 - 保持原有实现"""
        # 装载点
        self.special_points['loading'] = []
        for i, point in enumerate(self.env.loading_points):
            self.special_points['loading'].append({
                'id': i, 'type': 'loading', 'position': self._ensure_3d_point(point)
            })
        
        # 卸载点
        self.special_points['unloading'] = []
        for i, point in enumerate(self.env.unloading_points):
            self.special_points['unloading'].append({
                'id': i, 'type': 'unloading', 'position': self._ensure_3d_point(point)
            })
        
        # 停车点
        self.special_points['parking'] = []
        parking_areas = getattr(self.env, 'parking_areas', [])
        for i, point in enumerate(parking_areas):
            self.special_points['parking'].append({
                'id': i, 'type': 'parking', 'position': self._ensure_3d_point(point)
            })
        
        print(f"加载特殊点: 装载{len(self.special_points['loading'])}个, "
              f"卸载{len(self.special_points['unloading'])}个, "
              f"停车{len(self.special_points['parking'])}个")
    
    def _validate_special_points(self) -> bool:
        """验证特殊点"""
        if not self.special_points['loading'] or not self.special_points['unloading']:
            print("❌ 缺少必要的装载点或卸载点")
            return False
        return True
    
    def _generate_complete_bidirectional_paths(self) -> int:
        """生成完整的双向路径组合 - 保持原有实现但简化"""
        if not self.path_planner:
            print("❌ 未设置路径规划器")
            return 0
        
        success_count = 0
        path_pairs = []
        
        # 定义需要连接的点类型组合
        connection_types = [
            ('loading', 'unloading'),
            ('loading', 'parking'),
            ('unloading', 'parking')
        ]
        
        # 收集所有需要连接的点对
        for type_a, type_b in connection_types:
            if (len(self.special_points[type_a]) == 0 or 
                len(self.special_points[type_b]) == 0):
                continue
                
            for point_a in self.special_points[type_a]:
                for point_b in self.special_points[type_b]:
                    path_pairs.append((point_a, point_b, f"{type_a}_to_{type_b}"))
        
        self.stats['total_path_pairs'] = len(path_pairs)
        print(f"需要生成 {len(path_pairs)} 条双向路径")
        
        # 生成每条双向路径（简化版本）
        for i, (point_a, point_b, connection_type) in enumerate(path_pairs, 1):
            print(f"[{i}/{len(path_pairs)}] 生成路径: {point_a['type'][0].upper()}{point_a['id']} ↔ {point_b['type'][0].upper()}{point_b['id']}")
            
            path_result = self._generate_single_bidirectional_path_simplified(point_a, point_b)
            
            if path_result:
                success_count += 1
                self.stats[connection_type] += 1
                print(f"  ✅ 成功: 长度{len(path_result.forward_path)}, 质量{path_result.quality:.2f}")
            else:
                print(f"  ❌ 失败")
        
        return success_count
    
    def _generate_single_bidirectional_path_simplified(self, point_a: Dict, point_b: Dict) -> Optional['BiDirectionalPath']:
        """生成单条双向路径 - 简化版本"""
        path_id = f"{point_a['type'][0].upper()}{point_a['id']}_to_{point_b['type'][0].upper()}{point_b['id']}"
        
        start_pos = point_a['position']
        end_pos = point_b['position']
        
        # 尝试使用路径规划器
        try:
            result = self.path_planner.plan_path(
                vehicle_id=f"backbone_{path_id}",
                start=start_pos,
                goal=end_pos,
                use_backbone=False,
                context='backbone',
                return_object=True
            )
            
            if result and hasattr(result, 'path') and result.path:
                forward_path = result.path
                reverse_path = self._reverse_path(forward_path)
                quality = getattr(result, 'quality_score', 0.7)
                planner_used = getattr(result, 'planner_used', 'unknown')
                
                # 更新统计
                if 'astar' in planner_used.lower():
                    self.stats['astar_success'] += 1
                elif 'rrt' in planner_used.lower():
                    self.stats['rrt_success'] += 1
                
            else:
                # 回退到直线路径
                forward_path = self._create_direct_path(start_pos, end_pos)
                reverse_path = self._reverse_path(forward_path)
                quality = 0.5
                planner_used = 'direct_fallback'
                self.stats['direct_fallback'] += 1
        
        except Exception as e:
            print(f"  路径规划异常: {e}")
            # 回退到直线路径
            forward_path = self._create_direct_path(start_pos, end_pos)
            reverse_path = self._reverse_path(forward_path)
            quality = 0.4
            planner_used = 'direct_fallback'
            self.stats['direct_fallback'] += 1
        
        if forward_path and len(forward_path) >= 2:
            bidirectional_path = BiDirectionalPath(
                path_id=path_id,
                point_a=point_a,
                point_b=point_b,
                forward_path=forward_path,
                reverse_path=reverse_path,
                length=self._calculate_path_length(forward_path),
                quality=quality,
                planner_used=planner_used,
                created_time=time.time()
            )
            
            self.bidirectional_paths[path_id] = bidirectional_path
            return bidirectional_path
        
        return None
    
    def _create_direct_path(self, start: Tuple, end: Tuple) -> List[Tuple]:
        """创建直线路径"""
        distance = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        steps = max(3, int(distance / 2.0))
        
        path = []
        for i in range(steps + 1):
            t = i / steps
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])
            z = start[2] if len(start) > 2 else 0
            theta = math.atan2(end[1] - start[1], end[0] - start[0])
            path.append((x, y, z, theta))
        
        return path
    
    def _reverse_path(self, path: List[Tuple]) -> List[Tuple]:
        """反转路径方向"""
        if not path:
            return []
        
        reversed_path = []
        for point in reversed(path):
            if len(point) >= 3:
                x, y, theta = point[0], point[1], point[2]
                reverse_theta = (theta + math.pi) % (2 * math.pi)
                reversed_path.append((x, y, reverse_theta))
            else:
                reversed_path.append(point)
        
        return reversed_path
    
    def _generate_safe_interfaces(self) -> int:
        """为双向路径生成安全接口"""
        total_interfaces = 0
        spacing = self.config['interface_spacing']
        
        for path_id, path_data in self.bidirectional_paths.items():
            forward_path = path_data.forward_path
            
            if len(forward_path) < 2:
                continue
            
            interface_count = 0
            
            # 在路径上等间距生成接口
            for i in range(0, len(forward_path), spacing):
                if i >= len(forward_path):
                    break
                
                interface_id = f"{path_id}_if_{interface_count}"
                
                self.backbone_interfaces[interface_id] = {
                    'id': interface_id,
                    'position': forward_path[i],
                    'path_id': path_id,
                    'path_index': i,
                    'is_occupied': False,
                    'reservation_count': 0,
                    'usage_history': [],
                    'consolidation_level': getattr(path_data, 'consolidation_level', 'original'),
                    'is_professional_design': getattr(path_data, 'is_professional_design', False),
                    'spacing_used': spacing
                }
                
                self.path_interfaces[path_id].append(interface_id)
                interface_count += 1
                total_interfaces += 1
        
        return total_interfaces
    
    def _build_connection_index(self):
        """建立连接索引"""
        self.connection_index.clear()
        
        for path_id, path_data in self.bidirectional_paths.items():
            point_a = path_data.point_a
            point_b = path_data.point_b
            
            # 双向索引
            key_ab = (point_a['type'], point_a['id'], point_b['type'], point_b['id'])
            key_ba = (point_b['type'], point_b['id'], point_a['type'], point_a['id'])
            
            self.connection_index[key_ab] = path_id
            self.connection_index[key_ba] = path_id
    
    def _initialize_quality_tracking(self):
        """初始化质量追踪"""
        for path_data in self.bidirectional_paths.values():
            if hasattr(path_data, 'update_quality_history'):
                path_data.update_quality_history(path_data.quality)
    
    def _calculate_distance(self, p1: Tuple, p2: Tuple) -> float:
        """计算两点间距离"""
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    def _calculate_path_length(self, path: List[Tuple]) -> float:
        """计算路径总长度"""
        if not path or len(path) < 2:
            return 0.0
        
        length = 0.0
        for i in range(len(path) - 1):
            length += self._calculate_distance(path[i], path[i + 1])
        return length
    
    def get_network_status(self) -> Dict:
        """获取网络状态"""
        return {
            'bidirectional_paths': len(self.bidirectional_paths),
            'total_interfaces': len(self.backbone_interfaces),
            'generation_stats': self.stats,
            'special_points': {
                'loading': len(self.special_points['loading']),
                'unloading': len(self.special_points['unloading']),
                'parking': len(self.special_points['parking'])
            },
            'enhanced_features': {
                'enhanced_consolidation_available': ENHANCED_PROFESSIONAL_CONSOLIDATION_AVAILABLE,
                'enhanced_consolidation_applied': self.stats['enhanced_consolidation_applied']
            }
        }
    
    def set_path_planner(self, path_planner):
        """设置路径规划器"""
        self.path_planner = path_planner
        print("✅ 已设置路径规划器")
    
    def _ensure_3d_point(self, point) -> Tuple[float, float, float]:
        """确保点坐标为3D"""
        if not point:
            return (0.0, 0.0, 0.0)
        elif len(point) >= 3:
            return (float(point[0]), float(point[1]), float(point[2]))
        elif len(point) == 2:
            return (float(point[0]), float(point[1]), 0.0)
        else:
            return (0.0, 0.0, 0.0)


# ==================== 便捷函数 ====================

def create_enhanced_backbone_network(env, config_mode='balanced'):
    """创建增强版骨干网络"""
    network = OptimizedBackboneNetwork(env)
    
    # 设置增强配置
    if config_mode == 'professional':
        network.config.update({
            'enhanced_design_mode': 'professional',
            'curve_fitting_quality_threshold': 0.8,
            'force_vehicle_dynamics': True
        })
    elif config_mode == 'performance':
        network.config.update({
            'enhanced_design_mode': 'performance',
            'curve_fitting_quality_threshold': 0.6,
            'auto_consolidate_after_generation': True
        })
    
    return network

# 向后兼容性
SimplifiedBackboneNetwork = OptimizedBackboneNetwork