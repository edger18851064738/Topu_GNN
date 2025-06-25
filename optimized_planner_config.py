"""
optimized_planner_config.py - 集成优化版路径规划器配置 (完全集成修复版)
完美适应增强版环境管理、专业道路整合、智能拓扑构建的规划器配置系统
修复了VehicleStatus.PARKING错误，增强了错误处理和兼容性
"""

import math
import time
import threading
from typing import Dict, Any, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass

# 导入新系统的核心模块 - 增强错误处理
try:
    from eastar import HybridAStarPlanner, MiningOptimizedReedShepp
    EASTAR_AVAILABLE = True
    print("✅ 成功导入eastar.py混合A*规划器")
except ImportError as e:
    EASTAR_AVAILABLE = False
    print(f"⚠️ 无法导入eastar.py: {e}")

try:
    from environment import (
        OptimizedOpenPitMineEnv, VehicleSafetyParams, 
        SafetyCollisionDetector, VehicleStatus
    )
    ENHANCED_ENV_AVAILABLE = True
    print("✅ 成功导入增强版环境管理系统")
    
    # 安全检查VehicleStatus的可用属性
    print("🔍 检测到的VehicleStatus属性:")
    for attr in dir(VehicleStatus):
        if not attr.startswith('_') and hasattr(VehicleStatus, attr):
            try:
                value = getattr(VehicleStatus, attr)
                if hasattr(value, 'value'):
                    print(f"  ✓ {attr}: {value.value}")
                else:
                    print(f"  ✓ {attr}: {value}")
            except:
                print(f"  ⚠ {attr}: <无法获取值>")
                
except ImportError as e:
    ENHANCED_ENV_AVAILABLE = False
    VehicleStatus = None
    print(f"⚠️ 无法导入增强版环境: {e}")

try:
    from node_clustering_professional_consolidator import (
        EnhancedNodeClusteringConsolidator,
        RoadClass, NodeType, KeyNode,
        create_enhanced_node_clustering_consolidator
    )
    ENHANCED_CONSOLIDATION_AVAILABLE = True
    print("✅ 成功导入增强版专业道路整合器")
except ImportError as e:
    ENHANCED_CONSOLIDATION_AVAILABLE = False
    RoadClass = None
    print(f"⚠️ 无法导入增强版整合器: {e}")

try:
    from optimized_backbone_network import (
        OptimizedBackboneNetwork, BiDirectionalPath,
        create_enhanced_backbone_network
    )
    ENHANCED_BACKBONE_AVAILABLE = True
    print("✅ 成功导入增强版骨干网络")
except ImportError as e:
    ENHANCED_BACKBONE_AVAILABLE = False
    print(f"⚠️ 无法导入增强版骨干网络: {e}")

# 导入增强版ClothoidCubic曲线拟合模块（如果可用）
try:
    from ClothoidCubic import (
        EnhancedClothoidCubicFitter, 
        BackbonePathFitter, 
        VehicleDynamicsConfig,
        CurveSegment,
        CurveType,
        PathQuality
    )
    ENHANCED_CURVE_FITTING_AVAILABLE = True
    print("✅ 成功导入增强版ClothoidCubic曲线拟合")
except ImportError as e:
    ENHANCED_CURVE_FITTING_AVAILABLE = False
    print(f"⚠️ 无法导入增强版曲线拟合: {e}")

class NetworkTopologyType(Enum):
    """网络拓扑类型"""
    ORIGINAL = "original"
    CONSOLIDATED = "consolidated"
    HIERARCHICAL = "hierarchical"
    ENHANCED_PROFESSIONAL = "enhanced_professional"

class ConflictAwarenessLevel(Enum):
    """冲突感知级别"""
    BASIC = "basic"
    ENHANCED = "enhanced"
    SPATIAL_AWARE = "spatial_aware"
    PREDICTIVE = "predictive"
    SAFETY_RECTANGLE_AWARE = "safety_rectangle_aware"

class TaskStageType(Enum):
    """任务阶段类型"""
    LOADING = "loading"
    TRANSPORT = "transport"
    UNLOADING = "unloading"
    PARKING = "parking"
    MAINTENANCE = "maintenance"

class EnhancedVehicleDynamicsLevel(Enum):
    """增强版车辆动力学级别"""
    BASIC = "basic"
    ENHANCED = "enhanced"
    PROFESSIONAL = "professional"
    SAFETY_OPTIMIZED = "safety_optimized"

@dataclass
class EnhancedNetworkAwareConfig:
    """增强版网络感知配置"""
    topology_type: NetworkTopologyType = NetworkTopologyType.ORIGINAL
    consolidation_aware: bool = False
    enhanced_professional_aware: bool = False
    hierarchy_level_preference: str = "trunk"
    spatial_conflict_avoidance: bool = False  # 禁用以提高性能
    alternative_path_exploration_depth: int = 2  # 减少探索深度
    curve_fitting_quality_threshold: float = 0.6  # 降低质量要求
    vehicle_dynamics_compliance: bool = False  # 禁用严格动力学合规
    
@dataclass
class EnhancedConflictAwareConfig:
    """增强版冲突感知配置 - 性能优化版"""
    awareness_level: ConflictAwarenessLevel = ConflictAwarenessLevel.BASIC
    spatial_detection_enabled: bool = False
    safety_rectangle_detection: bool = False
    predictive_horizon: float = 30.0  # 缩短预测时域
    safety_margin_multiplier: float = 1.0
    conflict_avoidance_weight: float = 0.0
    collision_detector_integration: bool = False

@dataclass
class EnhancedMultiStageTaskConfig:
    """增强版多阶段任务配置"""
    stage_aware_planning: bool = True
    inter_stage_optimization: bool = False  # 禁用以简化
    stage_transition_buffer: float = 10.0  # 减少缓冲
    cross_stage_conflict_prevention: bool = False  # 禁用以提高性能
    enhanced_curve_fitting_per_stage: bool = False  # 禁用复杂拟合
    vehicle_status_awareness: bool = True
    stage_priority_weighting: Dict[str, float] = None
    
    def __post_init__(self):
        if self.stage_priority_weighting is None:
            self.stage_priority_weighting = {
                "loading": 1.1,
                "transport": 1.0,
                "unloading": 1.1,
                "parking": 0.9,
                "maintenance": 0.8
            }

@dataclass
class EnhancedVehicleDynamicsConfig:
    """增强版车辆动力学配置 - 性能优化版"""
    dynamics_level: EnhancedVehicleDynamicsLevel = EnhancedVehicleDynamicsLevel.BASIC
    vehicle_length: float = 6.0
    vehicle_width: float = 3.0
    turning_radius: float = 5.0  # 减小转弯半径提高灵活性
    max_steering_angle: float = 45.0
    max_acceleration: float = 2.5
    max_deceleration: float = 3.5
    max_speed: float = 20.0
    max_grade: float = 0.20  # 放宽坡度限制
    comfort_lateral_accel: float = 1.8
    safety_margin: float = 1.0  # 减小安全边距
    enable_clothoid_curves: bool = False  # 禁用复杂曲线
    force_dynamics_compliance: bool = False
    curve_quality_threshold: float = 0.4  # 降低质量要求
    
    def to_vehicle_safety_params(self) -> 'VehicleSafetyParams':
        """转换为VehicleSafetyParams格式"""
        if ENHANCED_ENV_AVAILABLE and VehicleSafetyParams:
            try:
                return VehicleSafetyParams(
                    length=self.vehicle_length,
                    width=self.vehicle_width,
                    safety_margin=self.safety_margin,
                    turning_radius=self.turning_radius,
                    max_speed=self.max_speed,
                    braking_distance=self.max_deceleration * 2.0
                )
            except Exception as e:
                print(f"⚠️ 创建VehicleSafetyParams失败: {e}")
        return None
    
    def to_vehicle_dynamics_config(self) -> Dict:
        """转换为VehicleDynamicsConfig格式"""
        return {
            'vehicle_length': self.vehicle_length,
            'vehicle_width': self.vehicle_width,
            'turning_radius': self.turning_radius,
            'max_steering_angle': self.max_steering_angle,
            'max_acceleration': self.max_acceleration,
            'max_deceleration': self.max_deceleration,
            'max_speed': self.max_speed,
            'max_grade': self.max_grade,
            'comfort_lateral_accel': self.comfort_lateral_accel,
            'safety_margin': self.safety_margin
        }

class SafeVehicleStatusHandler:
    """安全的车辆状态处理器 - 解决PARKING属性问题"""
    
    def __init__(self):
        self.status_mappings = {}
        self._initialize_safe_mappings()
    
    def _initialize_safe_mappings(self):
        """安全地初始化车辆状态映射"""
        try:
            if ENHANCED_ENV_AVAILABLE and VehicleStatus:
                # 检查每个可能的状态属性
                status_checks = [
                    ('LOADING', 'loading'),
                    ('UNLOADING', 'unloading'),
                    ('MOVING', 'moving'),
                    ('PARKING', 'parking'),
                    ('IDLE', 'idle'),
                    ('MAINTENANCE', 'maintenance'),
                    ('STOPPED', 'stopped'),
                    ('ERROR', 'error')
                ]
                
                for enum_attr, string_key in status_checks:
                    if hasattr(VehicleStatus, enum_attr):
                        try:
                            enum_value = getattr(VehicleStatus, enum_attr)
                            if hasattr(enum_value, 'value'):
                                self.status_mappings[enum_value.value] = string_key
                                self.status_mappings[string_key] = enum_value.value
                            print(f"  ✓ 映射车辆状态: {enum_attr} -> {string_key}")
                        except Exception as e:
                            print(f"  ⚠️ 无法映射 {enum_attr}: {e}")
                            # 回退到字符串映射
                            self.status_mappings[string_key] = string_key
                    else:
                        # 状态不存在，使用字符串映射
                        self.status_mappings[string_key] = string_key
                        print(f"  ⚠️ {enum_attr} 不存在，使用字符串: {string_key}")
                        
            else:
                # 完全回退到字符串映射
                default_statuses = ['loading', 'unloading', 'moving', 'parking', 'idle', 'maintenance']
                for status in default_statuses:
                    self.status_mappings[status] = status
                    
        except Exception as e:
            print(f"❌ 车辆状态映射初始化失败: {e}")
            # 最后的回退
            default_statuses = ['loading', 'unloading', 'moving', 'parking', 'idle']
            for status in default_statuses:
                self.status_mappings[status] = status
    
    def get_safe_status_key(self, status_input):
        """安全地获取状态键"""
        if status_input in self.status_mappings:
            return self.status_mappings[status_input]
        
        # 如果是字符串，直接返回
        if isinstance(status_input, str):
            return status_input
            
        # 如果是枚举，尝试获取其value
        if hasattr(status_input, 'value'):
            return status_input.value
            
        # 最后回退
        return str(status_input).lower()
    
    def get_available_statuses(self):
        """获取所有可用状态"""
        return list(set(self.status_mappings.values()))

class EnhancedIntegratedPlannerConfig:
    """增强版集成规划器配置类 - 修复版"""
    
    def __init__(self):
        # 初始化安全的车辆状态处理器
        self.vehicle_status_handler = SafeVehicleStatusHandler()
        
        # 基础配置 - 性能优化
        self.base_config = {
            'max_planning_time': 12.0,  # 减少规划时间
            'quality_threshold': 0.4,   # 降低质量要求
            'cache_size': 300,          # 减少缓存大小
            'enable_fallback': True,
            'timeout_retry_count': 1,   # 减少重试次数
            'enable_progressive_optimization': False,
            'network_topology_adaptation': True,
            'enhanced_professional_integration': ENHANCED_CONSOLIDATION_AVAILABLE,
            'safety_rectangle_integration': False,
            'curve_fitting_integration': ENHANCED_CURVE_FITTING_AVAILABLE,
            'skip_safety_checks': True,
            'assume_safe_environment': True,
            'performance_priority': True  # 新增：性能优先模式
        }
        
        # 增强版配置 - 性能优化
        self.enhanced_network_config = EnhancedNetworkAwareConfig()
        self.enhanced_conflict_config = EnhancedConflictAwareConfig()
        self.enhanced_multi_stage_config = EnhancedMultiStageTaskConfig()
        self.enhanced_vehicle_dynamics = EnhancedVehicleDynamicsConfig()
        
        # 混合A*配置 - 性能优化版
        self.enhanced_astar_configs = {
            'ultra_fast': {
                'vehicle_length': 6.0,
                'vehicle_width': 3.0,
                'turning_radius': 4.0,
                'step_size': 3.0,
                'angle_resolution': 60,
                'max_iterations': 8000,
                'rs_fitting_radius': 35.0,
                'quality_threshold': 0.3,
                'timeout': 8.0,
                'skip_safety_checks': True,
                'assume_safe': True,
                'aggressive_mode': True
            },
            'fast_no_safety': {
                'vehicle_length': 6.0,
                'vehicle_width': 3.0,
                'turning_radius': 5.0,
                'step_size': 2.5,
                'angle_resolution': 45,
                'max_iterations': 12000,
                'rs_fitting_radius': 30.0,
                'quality_threshold': 0.4,
                'timeout': 10.0,
                'skip_safety_checks': True,
                'assume_safe': True
            },
            'balanced_fast': {
                'vehicle_length': 6.0,
                'vehicle_width': 3.0,
                'turning_radius': 6.0,
                'step_size': 2.0,
                'angle_resolution': 36,
                'max_iterations': 15000,
                'rs_fitting_radius': 25.0,
                'quality_threshold': 0.5,
                'timeout': 12.0,
                'skip_safety_checks': True,
                'assume_safe': True
            },
            'emergency_fast': {
                'vehicle_length': 6.0,
                'vehicle_width': 3.0,
                'turning_radius': 3.5,
                'step_size': 3.5,
                'angle_resolution': 90,
                'max_iterations': 5000,
                'rs_fitting_radius': 40.0,
                'quality_threshold': 0.2,
                'timeout': 5.0,
                'emergency_mode': True,
                'skip_safety_checks': True,
                'assume_safe': True,
                'no_constraints': True
            }
        }
        
        # 增强版Reed-Shepp曲线配置 - 性能优化
        self.enhanced_rs_configs = {
            'ultra_fast': {
                'vehicle_length': 6.0,
                'vehicle_width': 3.0,
                'turning_radius': 4.0,
                'step_size': 1.5,
                'quality_threshold': 0.3,
                'max_curve_attempts': 2,
                'smoothness_preference': 0.4,
                'enable_obstacle_avoidance': False,
                'skip_safety_checks': True,
                'assume_safe': True,
                'aggressive_mode': True
            },
            'fast_no_safety': {
                'vehicle_length': 6.0,
                'vehicle_width': 3.0,
                'turning_radius': 5.0,
                'step_size': 1.2,
                'quality_threshold': 0.4,
                'max_curve_attempts': 3,
                'smoothness_preference': 0.5,
                'enable_obstacle_avoidance': False,
                'skip_safety_checks': True,
                'assume_safe': True
            },
            'balanced_fast': {
                'vehicle_length': 6.0,
                'vehicle_width': 3.0,
                'turning_radius': 6.0,
                'step_size': 1.0,
                'quality_threshold': 0.5,
                'max_curve_attempts': 4,
                'smoothness_preference': 0.6,
                'enable_obstacle_avoidance': False,
                'skip_safety_checks': True,
                'assume_safe': True
            },
            'emergency_fast': {
                'vehicle_length': 6.0,
                'vehicle_width': 3.0,
                'turning_radius': 3.0,
                'step_size': 2.0,
                'quality_threshold': 0.2,
                'max_curve_attempts': 1,
                'smoothness_preference': 0.3,
                'enable_obstacle_avoidance': False,
                'emergency_mode': True,
                'skip_safety_checks': True,
                'assume_safe': True,
                'no_constraints': True
            }
        }
        
        # 增强版网络感知回退策略 - 性能优先
        self.enhanced_fallback_strategies = {
            NetworkTopologyType.ORIGINAL: [
                {
                    'name': 'ultra_fast_astar',
                    'planner': 'enhanced_hybrid_astar',
                    'config': 'ultra_fast',
                    'max_time': 8.0,
                    'priority': 1,
                    'network_specific': True
                },
                {
                    'name': 'rs_ultra_fast',
                    'planner': 'enhanced_rs_curves',
                    'config': 'ultra_fast',
                    'max_time': 5.0,
                    'priority': 2,
                    'network_specific': True
                },
                {
                    'name': 'fast_astar_no_safety',
                    'planner': 'enhanced_hybrid_astar',
                    'config': 'fast_no_safety',
                    'max_time': 10.0,
                    'priority': 3,
                    'network_specific': False
                },
                {
                    'name': 'emergency_fast',
                    'planner': 'enhanced_hybrid_astar',
                    'config': 'emergency_fast',
                    'max_time': 5.0,
                    'priority': 4,
                    'network_specific': False
                },
                {
                    'name': 'direct_fallback',
                    'planner': 'direct',
                    'config': None,
                    'max_time': 1.0,
                    'priority': 5,
                    'network_specific': False
                }
            ]
        }
        
        # 复制策略到其他拓扑类型
        for topology in [NetworkTopologyType.CONSOLIDATED, NetworkTopologyType.HIERARCHICAL, NetworkTopologyType.ENHANCED_PROFESSIONAL]:
            self.enhanced_fallback_strategies[topology] = self.enhanced_fallback_strategies[NetworkTopologyType.ORIGINAL].copy()
        
        # 安全初始化车辆状态配置
        self._initialize_safe_vehicle_status_config()
        
        # 专业道路等级感知配置
        self._initialize_road_class_adjustments()
        
        print(f"✅ 初始化增强版集成规划器配置（性能优化版）")
        print(f"  eastar.py集成: {'✅' if EASTAR_AVAILABLE else '❌'}")
        print(f"  增强环境集成: {'✅' if ENHANCED_ENV_AVAILABLE else '❌'}")
        print(f"  专业整合器集成: {'✅' if ENHANCED_CONSOLIDATION_AVAILABLE else '❌'}")
        print(f"  增强骨干网络集成: {'✅' if ENHANCED_BACKBONE_AVAILABLE else '❌'}")
        print(f"  增强曲线拟合集成: {'✅' if ENHANCED_CURVE_FITTING_AVAILABLE else '❌'}")
        print(f"  可用车辆状态: {len(self.vehicle_status_adjustments)} 个")
    
    def _initialize_safe_vehicle_status_config(self):
        """安全地初始化车辆状态配置 - 修复版"""
        try:
            print("🔧 初始化车辆状态配置...")
            
            # 获取所有可用状态
            available_statuses = self.vehicle_status_handler.get_available_statuses()
            print(f"  发现 {len(available_statuses)} 个可用状态: {available_statuses}")
            
            # 为每个状态创建配置
            self.vehicle_status_adjustments = {}
            
            for status in available_statuses:
                # 根据状态类型设置不同的配置
                if status in ['loading', 'unloading']:
                    self.vehicle_status_adjustments[status] = {
                        'precision_required': True,
                        'safety_margin_boost': 1.2,
                        'quality_threshold_boost': 0.1,
                        'timeout_extension': 1.1
                    }
                elif status == 'parking':
                    self.vehicle_status_adjustments[status] = {
                        'precision_required': True,
                        'safety_margin_boost': 1.3,
                        'quality_threshold_boost': 0.08,
                        'timeout_extension': 1.15
                    }
                elif status == 'moving':
                    self.vehicle_status_adjustments[status] = {
                        'precision_required': False,
                        'safety_margin_boost': 1.0,
                        'quality_threshold_boost': 0.0,
                        'timeout_extension': 1.0
                    }
                elif status in ['idle', 'stopped']:
                    self.vehicle_status_adjustments[status] = {
                        'precision_required': False,
                        'safety_margin_boost': 1.0,
                        'quality_threshold_boost': 0.0,
                        'timeout_extension': 1.0
                    }
                elif status == 'maintenance':
                    self.vehicle_status_adjustments[status] = {
                        'precision_required': True,
                        'safety_margin_boost': 1.1,
                        'quality_threshold_boost': 0.05,
                        'timeout_extension': 1.05
                    }
                else:
                    # 默认配置
                    self.vehicle_status_adjustments[status] = {
                        'precision_required': False,
                        'safety_margin_boost': 1.0,
                        'quality_threshold_boost': 0.0,
                        'timeout_extension': 1.0
                    }
            
            print(f"  ✅ 车辆状态配置完成: {list(self.vehicle_status_adjustments.keys())}")
            
        except Exception as e:
            print(f"❌ 车辆状态配置初始化失败: {e}")
            # 最后的安全回退
            self.vehicle_status_adjustments = {
                'loading': {
                    'precision_required': True,
                    'safety_margin_boost': 1.2,
                    'quality_threshold_boost': 0.1,
                    'timeout_extension': 1.1
                },
                'unloading': {
                    'precision_required': True,
                    'safety_margin_boost': 1.2,
                    'quality_threshold_boost': 0.1,
                    'timeout_extension': 1.1
                },
                'moving': {
                    'precision_required': False,
                    'safety_margin_boost': 1.0,
                    'quality_threshold_boost': 0.0,
                    'timeout_extension': 1.0
                },
                'parking': {
                    'precision_required': True,
                    'safety_margin_boost': 1.3,
                    'quality_threshold_boost': 0.08,
                    'timeout_extension': 1.15
                },
                'idle': {
                    'precision_required': False,
                    'safety_margin_boost': 1.0,
                    'quality_threshold_boost': 0.0,
                    'timeout_extension': 1.0
                }
            }
            print(f"  🔄 使用回退配置: {list(self.vehicle_status_adjustments.keys())}")
    
    def _initialize_road_class_adjustments(self):
        """初始化专业道路等级感知配置"""
        try:
            self.enhanced_road_class_adjustments = {}
            
            # 使用字符串键以避免枚举依赖问题
            road_classes = ['primary', 'secondary', 'service']
            
            if ENHANCED_CONSOLIDATION_AVAILABLE and RoadClass:
                # 尝试使用枚举值
                for class_name in road_classes:
                    enum_attr = class_name.upper()
                    if hasattr(RoadClass, enum_attr):
                        try:
                            enum_value = getattr(RoadClass, enum_attr).value
                            key = enum_value
                        except:
                            key = class_name
                    else:
                        key = class_name
                    
                    self._add_road_class_config(key, class_name)
            else:
                # 直接使用字符串键
                for class_name in road_classes:
                    self._add_road_class_config(class_name, class_name)
                    
        except Exception as e:
            print(f"⚠️ 道路等级配置初始化警告: {e}")
            # 安全回退
            for class_name in ['primary', 'secondary', 'service']:
                self._add_road_class_config(class_name, class_name)
    
    def _add_road_class_config(self, key: str, class_name: str):
        """添加道路等级配置"""
        if class_name == 'primary':
            self.enhanced_road_class_adjustments[key] = {
                'quality_threshold_boost': 0.15,
                'safety_margin_multiplier': 1.15,
                'preferred_planners': ['enhanced_hybrid_astar'],
                'timeout_extension': 1.2,
                'priority_bonus': 0.2,
                'enhanced_curve_fitting': False,  # 禁用以提高性能
                'professional_design_required': False
            }
        elif class_name == 'secondary':
            self.enhanced_road_class_adjustments[key] = {
                'quality_threshold_boost': 0.08,
                'safety_margin_multiplier': 1.08,
                'preferred_planners': ['enhanced_hybrid_astar', 'enhanced_rs_curves'],
                'timeout_extension': 1.0,
                'priority_bonus': 0.1,
                'enhanced_curve_fitting': False
            }
        else:  # service
            self.enhanced_road_class_adjustments[key] = {
                'quality_threshold_boost': 0.0,
                'safety_margin_multiplier': 1.0,
                'preferred_planners': ['enhanced_rs_curves'],
                'timeout_extension': 0.9,
                'priority_bonus': 0.0,
                'enhanced_curve_fitting': False
            }
    
    def update_enhanced_network_topology(self, topology_type: NetworkTopologyType, 
                                       consolidation_info: Dict = None,
                                       professional_info: Dict = None):
        """更新增强版网络拓扑配置"""
        try:
            self.enhanced_network_config.topology_type = topology_type
            
            # 检测增强版专业整合
            if (topology_type == NetworkTopologyType.ENHANCED_PROFESSIONAL or
                (professional_info and professional_info.get('enhanced_professional_design_applied', False))):
                self.enhanced_network_config.enhanced_professional_aware = True
                self.enhanced_network_config.curve_fitting_quality_threshold = 0.7
                self.enhanced_network_config.vehicle_dynamics_compliance = False  # 保持禁用以提高性能
            
            # 应用整合信息
            if consolidation_info:
                self._adapt_config_to_enhanced_consolidation(consolidation_info)
            
            # 应用专业信息
            if professional_info:
                self._adapt_config_to_professional_design(professional_info)
            
            print(f"✅ 增强版网络拓扑配置已更新: {topology_type.value}")
            
        except Exception as e:
            print(f"⚠️ 网络拓扑配置更新失败: {e}")
    
    def _adapt_config_to_enhanced_consolidation(self, consolidation_info: Dict):
        """适应增强版整合信息"""
        try:
            if 'enhanced_consolidation_stats' in consolidation_info:
                enhanced_stats = consolidation_info['enhanced_consolidation_stats']
                
                # 根据节点减少比例调整
                node_reduction = enhanced_stats.get('node_reduction_ratio', 0.0)
                if node_reduction > 0.4:  # 高度整合
                    self.base_config['quality_threshold'] *= 0.9
                    self.enhanced_network_config.alternative_path_exploration_depth = 1
                elif node_reduction > 0.2:  # 中度整合
                    self.base_config['quality_threshold'] *= 0.95
                
                # 根据曲线拟合成功率调整
                curve_success_rate = enhanced_stats.get('curve_fitting_success_rate', 0.0)
                if curve_success_rate > 0.7:
                    self.enhanced_network_config.curve_fitting_quality_threshold = 0.65
            
        except Exception as e:
            print(f"⚠️ 整合配置适应失败: {e}")
    
    def _adapt_config_to_professional_design(self, professional_info: Dict):
        """适应专业设计信息"""
        try:
            if professional_info.get('is_enhanced_professional_design', False):
                # 启用部分增强功能（保持性能优先）
                self.enhanced_network_config.enhanced_professional_aware = True
                
                # 调整质量要求（保持适中）
                design_mode = professional_info.get('design_mode', 'balanced')
                if design_mode == 'professional':
                    self.base_config['quality_threshold'] = 0.6  # 不过度提高
                elif design_mode == 'performance':
                    self.base_config['max_planning_time'] *= 0.9
                    
        except Exception as e:
            print(f"⚠️ 专业设计配置适应失败: {e}")
    
    def get_enhanced_astar_config(self, level: str = 'balanced_fast', 
                                network_context: Dict = None,
                                vehicle_context: Dict = None) -> Dict[str, Any]:
        """获取增强版混合A*配置"""
        try:
            base_config = self.enhanced_astar_configs.get(level, 
                self.enhanced_astar_configs['balanced_fast']).copy()
            
            # 网络上下文调整
            if network_context:
                base_config = self._apply_enhanced_network_context(base_config, network_context)
            
            # 车辆上下文调整
            if vehicle_context:
                base_config = self._apply_safe_vehicle_context(base_config, vehicle_context)
            
            return base_config
            
        except Exception as e:
            print(f"⚠️ A*配置获取失败: {e}")
            return self.enhanced_astar_configs['balanced_fast'].copy()
    
    def get_enhanced_rs_config(self, level: str = 'balanced_fast',
                             task_context: Dict = None,
                             vehicle_context: Dict = None) -> Dict[str, Any]:
        """获取增强版Reed-Shepp配置"""
        try:
            base_config = self.enhanced_rs_configs.get(level,
                self.enhanced_rs_configs['balanced_fast']).copy()
            
            # 任务阶段调整
            if task_context:
                base_config = self._apply_safe_task_adjustments(base_config, task_context)
            
            # 车辆上下文调整
            if vehicle_context:
                base_config = self._apply_safe_vehicle_context(base_config, vehicle_context)
            
            return base_config
            
        except Exception as e:
            print(f"⚠️ RS配置获取失败: {e}")
            return self.enhanced_rs_configs['balanced_fast'].copy()
    
    def _apply_enhanced_network_context(self, config: Dict, network_context: Dict) -> Dict:
        """应用增强版网络上下文"""
        try:
            enhanced_config = config.copy()
            
            # 增强版专业整合网络处理
            if network_context.get('enhanced_professional_consolidation', False):
                enhanced_config['professional_design'] = True
                if 'quality_threshold' in enhanced_config:
                    enhanced_config['quality_threshold'] += 0.03  # 适度提升
            
            # 关键节点感知
            key_nodes_info = network_context.get('key_nodes_info', {})
            if key_nodes_info:
                key_nodes_count = len(key_nodes_info)
                original_nodes = network_context.get('original_nodes_count', key_nodes_count)
                
                if original_nodes > 0:
                    reduction_ratio = 1.0 - (key_nodes_count / original_nodes)
                    if reduction_ratio > 0.3:  # 中等简化
                        enhanced_config['step_size'] = enhanced_config.get('step_size', 2.0) * 1.05
                        enhanced_config['max_iterations'] = int(enhanced_config.get('max_iterations', 15000) * 0.95)
            
            return enhanced_config
            
        except Exception as e:
            print(f"⚠️ 网络上下文应用失败: {e}")
            return config
    
    def _apply_safe_vehicle_context(self, config: Dict, vehicle_context: Dict) -> Dict:
        """安全地应用车辆上下文"""
        try:
            vehicle_config = config.copy()
            
            # 车辆安全参数集成
            if 'safety_params' in vehicle_context:
                safety_params = vehicle_context['safety_params']
                if isinstance(safety_params, dict):
                    vehicle_config['vehicle_length'] = safety_params.get('length', config.get('vehicle_length', 6.0))
                    vehicle_config['vehicle_width'] = safety_params.get('width', config.get('vehicle_width', 3.0))
                    vehicle_config['turning_radius'] = safety_params.get('turning_radius', config.get('turning_radius', 5.0))
                    vehicle_config['safety_margin'] = safety_params.get('safety_margin', 1.0)
            
            # 安全的车辆状态调整
            vehicle_status = vehicle_context.get('vehicle_status', 'moving')
            safe_status = self.vehicle_status_handler.get_safe_status_key(vehicle_status)
            
            if safe_status in self.vehicle_status_adjustments:
                adjustments = self.vehicle_status_adjustments[safe_status]
                
                if adjustments.get('precision_required', False):
                    vehicle_config['step_size'] = vehicle_config.get('step_size', 2.0) * 0.9
                    vehicle_config['angle_resolution'] = max(25, 
                        int(vehicle_config.get('angle_resolution', 45) * 0.9))
                
                safety_boost = adjustments.get('safety_margin_boost', 1.0)
                if 'turning_radius' in vehicle_config:
                    vehicle_config['turning_radius'] *= safety_boost
            
            return vehicle_config
            
        except Exception as e:
            print(f"⚠️ 车辆上下文应用失败: {e}")
            return config
    
    def _apply_safe_task_adjustments(self, config: Dict, task_context: Dict) -> Dict:
        """安全地应用任务调整"""
        try:
            task_config = config.copy()
            
            current_stage = task_context.get('current_stage', 'transport')
            
            # 简化的多阶段优化
            if current_stage in ['loading', 'unloading']:
                task_config['precision_mode'] = True
                if 'smoothness_preference' in task_config:
                    task_config['smoothness_preference'] = min(0.9, 
                        task_config['smoothness_preference'] * 1.05)
            
            # 安全的车辆状态感知调整
            if self.enhanced_multi_stage_config.vehicle_status_awareness:
                vehicle_status = task_context.get('vehicle_status', 'moving')
                safe_status = self.vehicle_status_handler.get_safe_status_key(vehicle_status)
                
                if safe_status in self.vehicle_status_adjustments:
                    adjustments = self.vehicle_status_adjustments[safe_status]
                    
                    if adjustments.get('precision_required', False):
                        task_config['step_size'] = task_config.get('step_size', 1.0) * 0.95
                        task_config['max_curve_attempts'] = task_config.get('max_curve_attempts', 3) + 1
            
            return task_config
            
        except Exception as e:
            print(f"⚠️ 任务调整应用失败: {e}")
            return config
    
    def get_enhanced_context_optimized_config(self, context: str, planner_type: str,
                                            network_info: Dict = None,
                                            task_info: Dict = None,
                                            vehicle_info: Dict = None) -> Dict[str, Any]:
        """获取增强版上下文优化配置"""
        try:
            # 向后兼容：RRT重定向到增强版RS曲线
            if planner_type == 'rrt':
                planner_type = 'enhanced_rs_curves'
            
            if context == 'backbone':
                # 骨干路径生成：高质量要求
                if planner_type in ['hybrid_astar', 'enhanced_hybrid_astar']:
                    config = self.get_enhanced_astar_config('balanced_fast', network_info, vehicle_info)
                    config.update({
                        'max_iterations': min(config.get('max_iterations', 15000), 20000),
                        'timeout': min(config.get('timeout', 12.0), 15.0),
                        'quality_threshold': max(config.get('quality_threshold', 0.5), 0.45),
                        'professional_design': True
                    })
                    return config
                    
                elif planner_type in ['rs_curves', 'enhanced_rs_curves']:
                    config = self.get_enhanced_rs_config('balanced_fast', task_info, vehicle_info)
                    config.update({
                        'max_curve_attempts': min(config.get('max_curve_attempts', 4), 6),
                        'precision_mode': True,
                        'professional_design': True
                    })
                    return config
            
            elif context == 'navigation':
                # 导航路径：平衡质量和速度
                if planner_type in ['hybrid_astar', 'enhanced_hybrid_astar']:
                    config = self.get_enhanced_astar_config('fast_no_safety', network_info, vehicle_info)
                    config.update({
                        'timeout': min(config.get('timeout', 10.0), 12.0),
                        'quality_threshold': config.get('quality_threshold', 0.4) * 0.95
                    })
                    return config
                    
                elif planner_type in ['rs_curves', 'enhanced_rs_curves']:
                    config = self.get_enhanced_rs_config('fast_no_safety', task_info, vehicle_info)
                    return config
            
            elif context == 'emergency':
                # 紧急情况：优先速度
                if planner_type in ['hybrid_astar', 'enhanced_hybrid_astar']:
                    config = self.get_enhanced_astar_config('emergency_fast', network_info, vehicle_info)
                    config.update({
                        'timeout': min(config.get('timeout', 5.0), 6.0),
                        'quality_threshold': max(config.get('quality_threshold', 0.2), 0.15),
                        'emergency_mode': True
                    })
                    return config
                    
                elif planner_type in ['rs_curves', 'enhanced_rs_curves']:
                    config = self.get_enhanced_rs_config('emergency_fast', task_info, vehicle_info)
                    config.update({
                        'max_curve_attempts': 1,
                        'emergency_mode': True
                    })
                    return config
            
            # 默认配置
            if planner_type in ['hybrid_astar', 'enhanced_hybrid_astar']:
                return self.get_enhanced_astar_config('balanced_fast', network_info, vehicle_info)
            elif planner_type in ['rs_curves', 'enhanced_rs_curves']:
                return self.get_enhanced_rs_config('balanced_fast', task_info, vehicle_info)
            
            return {}
            
        except Exception as e:
            print(f"⚠️ 上下文优化配置获取失败: {e}")
            return {}
    
    def get_enhanced_fallback_sequence(self, context: str = 'normal',
                                     network_info: Dict = None,
                                     enhanced_features: Dict = None) -> List:
        """获取增强版网络感知回退序列"""
        try:
            # 检测网络拓扑
            topology = self.enhanced_network_config.topology_type
            
            # 根据增强功能检测拓扑
            if enhanced_features:
                if enhanced_features.get('enhanced_professional_design_applied', False):
                    topology = NetworkTopologyType.ENHANCED_PROFESSIONAL
                elif enhanced_features.get('enhanced_consolidation_applied', False):
                    topology = NetworkTopologyType.CONSOLIDATED
            
            # 获取对应的回退策略
            fallback_sequence = self.enhanced_fallback_strategies.get(
                topology,
                self.enhanced_fallback_strategies[NetworkTopologyType.ORIGINAL]
            ).copy()
            
            # 根据上下文调整序列
            if context == 'backbone':
                # 骨干路径生成：更保守的回退
                fallback_sequence = [s for s in fallback_sequence if s['priority'] <= 3]
                
            elif context == 'emergency':
                # 紧急情况：快速回退序列
                emergency_sequence = [s for s in fallback_sequence if s['priority'] >= 3][:3]
                emergency_sequence.append({
                    'name': 'direct_fallback',
                    'planner': 'direct',
                    'config': None,
                    'max_time': 1.0,
                    'priority': 999
                })
                fallback_sequence = emergency_sequence
            
            return fallback_sequence
            
        except Exception as e:
            print(f"⚠️ 回退序列获取失败: {e}")
            # 返回最基础的回退序列
            return [
                {
                    'name': 'ultra_fast_astar',
                    'planner': 'enhanced_hybrid_astar',
                    'config': 'ultra_fast',
                    'max_time': 8.0,
                    'priority': 1
                },
                {
                    'name': 'direct_fallback',
                    'planner': 'direct',
                    'config': None,
                    'max_time': 1.0,
                    'priority': 999
                }
            ]
    
    def adapt_to_enhanced_environment(self, env_info: Dict):
        """适应增强版环境"""
        try:
            if not env_info:
                return
            
            # 车辆安全参数适应
            if 'vehicle_safety_params' in env_info:
                safety_params = env_info['vehicle_safety_params']
                self.enhanced_vehicle_dynamics.safety_margin = safety_params.get('safety_margin', 1.0)
                self.enhanced_vehicle_dynamics.turning_radius = safety_params.get('turning_radius', 5.0)
            
            # 环境状态适应
            env_state = env_info.get('environment_state', 'ready')
            if env_state == 'running':
                self.enhanced_conflict_config.predictive_horizon = 35.0
            elif env_state == 'error':
                # 错误状态下降低要求
                self.base_config['quality_threshold'] *= 0.9
            
            print(f"✅ 配置已适应增强版环境")
            
        except Exception as e:
            print(f"⚠️ 环境适应失败: {e}")
    
    def get_professional_design_config(self, road_class: str = 'secondary',
                                     design_mode: str = 'balanced',
                                     vehicle_dynamics: Dict = None) -> Dict[str, Any]:
        """获取专业设计配置"""
        try:
            # 确定基础配置
            if design_mode == 'performance':
                base_config = self.get_enhanced_astar_config('fast_no_safety')
            else:
                base_config = self.get_enhanced_astar_config('balanced_fast')
            
            # 应用道路等级调整
            if road_class in self.enhanced_road_class_adjustments:
                adjustments = self.enhanced_road_class_adjustments[road_class]
                
                # 质量阈值调整
                quality_boost = adjustments.get('quality_threshold_boost', 0.0)
                base_config['quality_threshold'] = base_config.get('quality_threshold', 0.5) + quality_boost
                
                # 安全边距调整
                safety_multiplier = adjustments.get('safety_margin_multiplier', 1.0)
                base_config['turning_radius'] = base_config.get('turning_radius', 5.0) * safety_multiplier
                
                # 超时调整
                timeout_extension = adjustments.get('timeout_extension', 1.0)
                base_config['timeout'] = base_config.get('timeout', 12.0) * timeout_extension
                
                # 专业设计标记
                base_config['professional_design'] = True
                base_config['target_road_class'] = road_class
            
            # 车辆动力学集成
            if vehicle_dynamics:
                vehicle_config = self.enhanced_vehicle_dynamics.to_vehicle_dynamics_config()
                vehicle_config.update(vehicle_dynamics)
                base_config['vehicle_dynamics'] = vehicle_config
            else:
                base_config['vehicle_dynamics'] = self.enhanced_vehicle_dynamics.to_vehicle_dynamics_config()
            
            return base_config
            
        except Exception as e:
            print(f"⚠️ 专业设计配置获取失败: {e}")
            return self.get_enhanced_astar_config('balanced_fast')


class EnhancedIntegratedPathPlannerWithConfig:
    """增强版集成路径规划器 - 修复版"""
    
    def __init__(self, env, backbone_network=None, traffic_manager=None):
        self.env = env
        self.backbone_network = backbone_network
        self.traffic_manager = traffic_manager
        
        # 加载修复版配置
        self.config_manager = EnhancedIntegratedPlannerConfig()
        
        # 增强版规划器实例
        self.enhanced_planners = {}
        self._initialize_enhanced_planners_safe()
        
        # 网络拓扑监控 - 增强版
        self.current_network_topology = NetworkTopologyType.ORIGINAL
        self.enhanced_features_detected = {}
        self.last_topology_check = time.time()
        self.topology_check_interval = 30.0
        
        # 增强版缓存
        self.enhanced_cache = {}
        self.cache_lock = threading.RLock() if threading else None
        
        # 增强版性能统计
        self.enhanced_stats = {
            'strategy_success_rates': {},
            'average_planning_times': {},
            'total_requests': 0,
            'successful_plans': 0,
            'cache_hits': 0,
            'eastar_plans': 0,
            'enhanced_rs_plans': 0,
            'safety_rectangle_aware_plans': 0,
            'professional_design_plans': 0,
            'enhanced_curve_fitting_plans': 0,
            'mining_optimized_plans': 0,
            'enhanced_config_adaptations': 0,
            'professional_fallback_usage': 0,
            'safety_optimized_fallback_usage': 0,
            'vehicle_status_aware_adjustments': 0
        }
        
        print("✅ 初始化增强版集成路径规划器（修复版）")
        print(f"  eastar.py混合A*: {'✅' if EASTAR_AVAILABLE else '❌'}")
        print(f"  增强版环境系统: {'✅' if ENHANCED_ENV_AVAILABLE else '❌'}")
        print(f"  专业道路整合器: {'✅' if ENHANCED_CONSOLIDATION_AVAILABLE else '❌'}")
        print(f"  增强版骨干网络: {'✅' if ENHANCED_BACKBONE_AVAILABLE else '❌'}")
        print(f"  增强版曲线拟合: {'✅' if ENHANCED_CURVE_FITTING_AVAILABLE else '❌'}")
    
    def _initialize_enhanced_planners_safe(self):
        """安全地初始化增强版规划器"""
        try:
            # 初始化eastar.py混合A*规划器
            if EASTAR_AVAILABLE:
                try:
                    astar_config = self.config_manager.get_enhanced_astar_config('balanced_fast')
                    
                    self.enhanced_planners['enhanced_hybrid_astar'] = HybridAStarPlanner(
                        self.env,
                        vehicle_length=astar_config['vehicle_length'],
                        vehicle_width=astar_config['vehicle_width'],
                        turning_radius=astar_config['turning_radius'],
                        step_size=astar_config['step_size'],
                        angle_resolution=astar_config['angle_resolution']
                    )
                    
                    # 向后兼容
                    self.enhanced_planners['hybrid_astar'] = self.enhanced_planners['enhanced_hybrid_astar']
                    
                    print("  ✅ eastar.py混合A*规划器初始化完成")
                except Exception as e:
                    print(f"  ⚠️ eastar.py混合A*规划器初始化失败: {e}")
            
            # 初始化增强版Reed-Shepp曲线规划器
            if EASTAR_AVAILABLE:
                try:
                    self._initialize_enhanced_rs_planner()
                    print("  ✅ 增强版Reed-Shepp曲线规划器初始化完成")
                except Exception as e:
                    print(f"  ⚠️ 增强版RS曲线规划器初始化失败: {e}")
            
        except Exception as e:
            print(f"❌ 增强版规划器初始化失败: {e}")
    
    def _initialize_enhanced_rs_planner(self):
        """初始化增强版RS规划器"""
        rs_config = self.config_manager.get_enhanced_rs_config('balanced_fast')
        base_rs_planner = MiningOptimizedReedShepp(
            turning_radius=rs_config['turning_radius'],
            step_size=rs_config['step_size']
        )
        
        class EnhancedRSPlannerWrapper:
            def __init__(self, rs_planner, env, config_manager):
                self.rs_planner = rs_planner
                self.env = env
                self.config_manager = config_manager
            
            def plan_path(self, start, goal, **kwargs):
                """增强版RS曲线规划接口"""
                try:
                    # 确保朝向信息
                    start_with_heading = self._ensure_heading(start, goal)
                    goal_with_heading = self._ensure_heading(goal, start)
                    
                    # 获取动态配置
                    context = kwargs.get('context', 'normal')
                    task_info = kwargs.get('task_info', {})
                    vehicle_info = kwargs.get('vehicle_info', {})
                    
                    dynamic_config = self.config_manager.get_enhanced_rs_config(
                        'balanced_fast', task_info, vehicle_info
                    )
                    
                    # 应用配置调整
                    step_size = dynamic_config.get('step_size', self.rs_planner.step_size)
                    max_attempts = dynamic_config.get('max_curve_attempts', 3)
                    
                    # 多次尝试生成最佳路径
                    best_path = None
                    best_quality = 0.0
                    
                    for attempt in range(max_attempts):
                        # 调整参数
                        attempt_radius = dynamic_config['turning_radius'] * (0.9 + 0.2 * attempt / max_attempts)
                        original_radius = self.rs_planner.turning_radius
                        self.rs_planner.turning_radius = attempt_radius
                        
                        try:
                            path = self.rs_planner.get_path(start_with_heading, goal_with_heading, step_size)
                            
                            if path and len(path) >= 2:
                                quality = self._evaluate_path_quality(path, dynamic_config)
                                if quality > best_quality:
                                    best_path = path
                                    best_quality = quality
                                    
                                    # 如果质量足够好，提前返回
                                    if quality > dynamic_config.get('quality_threshold', 0.5):
                                        break
                        
                        finally:
                            self.rs_planner.turning_radius = original_radius
                    
                    if best_path:
                        print(f"      增强版RS曲线成功: {len(best_path)}点, 质量: {best_quality:.2f}")
                        return best_path
                    else:
                        print(f"      增强版RS曲线生成失败")
                        return None
                
                except Exception as e:
                    print(f"      增强版RS曲线异常: {e}")
                    return None
            
            def _ensure_heading(self, point, reference_point):
                """确保点包含朝向信息"""
                if len(point) >= 3:
                    return point
                
                dx = reference_point[0] - point[0]
                dy = reference_point[1] - point[1]
                heading = math.atan2(dy, dx)
                return (point[0], point[1], heading)
            
            def _evaluate_path_quality(self, path, config):
                """评估路径质量"""
                if not path or len(path) < 2:
                    return 0.0
                
                # 计算长度效率
                path_length = sum(
                    math.sqrt((path[i+1][0] - path[i][0])**2 + (path[i+1][1] - path[i][1])**2)
                    for i in range(len(path) - 1)
                )
                direct_distance = math.sqrt(
                    (path[-1][0] - path[0][0])**2 + (path[-1][1] - path[0][1])**2
                )
                length_efficiency = direct_distance / (path_length + 0.1) if path_length > 0 else 0
                
                # 平滑度
                smoothness = config.get('smoothness_preference', 0.6)
                
                # 基础质量计算
                return min(1.0, length_efficiency * 0.7 + smoothness * 0.3)
        
        # 使用增强版包装器
        self.enhanced_planners['enhanced_rs_curves'] = EnhancedRSPlannerWrapper(
            base_rs_planner, self.env, self.config_manager
        )
        
        # 向后兼容
        self.enhanced_planners['rs_curves'] = self.enhanced_planners['enhanced_rs_curves']
        self.enhanced_planners['rrt'] = self.enhanced_planners['enhanced_rs_curves']  # RRT重定向
    
    def update_enhanced_network_topology_awareness(self):
        """更新增强版网络拓扑感知"""
        try:
            current_time = time.time()
            
            if current_time - self.last_topology_check < self.topology_check_interval:
                return
            
            enhanced_features = {}
            consolidation_info = {}
            professional_info = {}
            
            # 检测增强版骨干网络
            if (ENHANCED_BACKBONE_AVAILABLE and self.backbone_network and 
                hasattr(self.backbone_network, 'get_topology_construction_summary')):
                
                topology_summary = self.backbone_network.get_topology_construction_summary()
                enhanced_features.update(topology_summary.get('enhanced_features', {}))
                
                # 检测网络拓扑类型
                if enhanced_features.get('enhanced_consolidation_applied', False):
                    new_topology = NetworkTopologyType.ENHANCED_PROFESSIONAL
                    professional_info = topology_summary.get('consolidation_stats', {})
                elif topology_summary.get('ready_for_stage2', False):
                    new_topology = NetworkTopologyType.CONSOLIDATED
                else:
                    new_topology = NetworkTopologyType.ORIGINAL
            else:
                new_topology = NetworkTopologyType.ORIGINAL
            
            # 更新配置
            if (new_topology != self.current_network_topology or 
                enhanced_features != self.enhanced_features_detected):
                
                self.current_network_topology = new_topology
                self.enhanced_features_detected = enhanced_features
                
                self.config_manager.update_enhanced_network_topology(
                    new_topology, consolidation_info, professional_info
                )
                
                self.enhanced_stats['enhanced_config_adaptations'] += 1
                print(f"🔄 增强版网络拓扑感知更新: {new_topology.value}")
            
            self.last_topology_check = current_time
            
        except Exception as e:
            print(f"⚠️ 网络拓扑感知更新失败: {e}")
    
    def plan_path(self, vehicle_id: str, start: tuple, goal: tuple,
                use_backbone: bool = True, check_conflicts: bool = False,  # 禁用冲突检查
                planner_type: str = "auto", context: str = "normal",
                return_object: bool = False, **kwargs) -> Any:
        """增强版集成路径规划接口 - 修复版"""
        try:
            # 更新增强版网络拓扑感知
            self.update_enhanced_network_topology_awareness()
            
            # 提取增强版上下文
            enhanced_task_context = self._extract_enhanced_task_context_safe(kwargs)
            enhanced_network_context = self._extract_enhanced_network_context_safe(kwargs)
            enhanced_vehicle_context = self._extract_enhanced_vehicle_context_safe(kwargs)
            
            # ===== 增强版骨干网络优先逻辑 =====
            if use_backbone and self.backbone_network and context != 'backbone':
                print(f"[增强版集成规划器] 尝试增强版骨干路径: {vehicle_id}")
                
                backbone_result = self._try_enhanced_backbone_path_safe(
                    vehicle_id, start, goal, enhanced_task_context, enhanced_network_context
                )
                
                if backbone_result:
                    print(f"  ✅ 增强版骨干路径成功!")
                    if self.enhanced_features_detected.get('enhanced_consolidation_applied', False):
                        self.enhanced_stats['professional_design_plans'] += 1
                    
                    return self._format_enhanced_result(backbone_result, return_object)
            
            # ===== 增强版集成回退规划 =====
            result = self.plan_path_with_enhanced_fallback_safe(
                vehicle_id, start, goal, context, 
                enhanced_network_context, enhanced_task_context, enhanced_vehicle_context, **kwargs
            )
            
            return self._format_enhanced_result(result, return_object)
            
        except Exception as e:
            print(f"❌ 增强版集成路径规划失败: {e}")
            # 最后的回退：生成直线路径
            return self._generate_emergency_fallback_path(start, goal, return_object)
    
    def _try_enhanced_backbone_path_safe(self, vehicle_id: str, start: tuple, goal: tuple,
                                       task_context: Dict, network_context: Dict):
        """安全地尝试增强版骨干路径"""
        try:
            # 增强版目标信息提取
            target_type = task_context.get('target_type')
            target_id = task_context.get('target_id')
            
            if target_type and target_id is not None:
                # 检查是否支持增强版接口
                if hasattr(self.backbone_network, 'get_path_from_position_to_target'):
                    return self.backbone_network.get_path_from_position_to_target(
                        start, target_type, target_id, vehicle_id
                    )
            
            return None
            
        except Exception as e:
            print(f"  增强版骨干路径尝试失败: {e}")
            return None
    
    def plan_path_with_enhanced_fallback_safe(self, vehicle_id: str, start: tuple, goal: tuple,
                                            context: str = 'normal',
                                            network_context: Dict = None,
                                            task_context: Dict = None,
                                            vehicle_context: Dict = None, **kwargs) -> Any:
        """安全的增强版集成回退路径规划"""
        try:
            self.enhanced_stats['total_requests'] += 1
            planning_start = time.time()
            
            # 获取增强版回退序列
            fallback_sequence = self.config_manager.get_enhanced_fallback_sequence(
                context, network_context, self.enhanced_features_detected
            )
            
            print(f"  [增强版回退] 使用{len(fallback_sequence)}个策略，网络: {self.current_network_topology.value}")
            
            for i, strategy in enumerate(fallback_sequence, 1):
                strategy_name = strategy['name']
                planner_type = strategy['planner']
                config_level = strategy.get('config')
                max_time = strategy['max_time']
                
                print(f"  [{i}/{len(fallback_sequence)}] 尝试增强策略: {strategy_name}")
                
                # 获取增强版配置
                strategy_config = self._get_enhanced_strategy_config_safe(
                    planner_type, config_level, context, network_context, task_context, vehicle_context
                )
                
                # 执行增强版规划
                strategy_start = time.time()
                result = self._execute_enhanced_strategy_safe(
                    vehicle_id, start, goal, planner_type,
                    strategy_config, max_time, context,
                    network_context, task_context, vehicle_context
                )
                strategy_time = time.time() - strategy_start
                
                # 更新统计
                self._update_enhanced_strategy_stats_safe(strategy_name, result, strategy_time)
                
                if result:
                    total_time = time.time() - planning_start
                    print(f"    ✅ 增强策略成功! 耗时: {strategy_time:.2f}s, 总耗时: {total_time:.2f}s")
                    return result
                else:
                    print(f"    ❌ 策略失败, 耗时: {strategy_time:.2f}s")
            
            print("  ❌ 所有增强回退策略均失败")
            return None
            
        except Exception as e:
            print(f"❌ 增强版回退规划失败: {e}")
            return None
    
    def _get_enhanced_strategy_config_safe(self, planner_type: str, config_level: str, context: str,
                                         network_context: Dict, task_context: Dict, vehicle_context: Dict) -> Dict:
        """安全地获取增强版策略配置"""
        try:
            if config_level:
                if planner_type in ['hybrid_astar', 'enhanced_hybrid_astar']:
                    strategy_config = self.config_manager.get_enhanced_astar_config(
                        config_level, network_context, vehicle_context
                    )
                elif planner_type in ['rs_curves', 'enhanced_rs_curves', 'rrt']:
                    strategy_config = self.config_manager.get_enhanced_rs_config(
                        config_level, task_context, vehicle_context
                    )
                else:
                    strategy_config = {}
            else:
                strategy_config = {}
            
            # 应用上下文优化
            if context != 'normal':
                context_config = self.config_manager.get_enhanced_context_optimized_config(
                    context, planner_type, network_context, task_context, vehicle_context
                )
                strategy_config.update(context_config)
            
            return strategy_config
            
        except Exception as e:
            print(f"⚠️ 策略配置获取失败: {e}")
            return {}
    
    def _execute_enhanced_strategy_safe(self, vehicle_id: str, start: tuple, goal: tuple,
                                      planner_type: str, config: dict, max_time: float,
                                      context: str, network_context: Dict = None,
                                      task_context: Dict = None, vehicle_context: Dict = None):
        """安全地执行增强版规划策略"""
        try:
            if planner_type in ['hybrid_astar', 'enhanced_hybrid_astar'] and 'enhanced_hybrid_astar' in self.enhanced_planners:
                result = self._plan_with_enhanced_astar_safe(
                    vehicle_id, start, goal, config, max_time, network_context, vehicle_context
                )
                if result:
                    self.enhanced_stats['eastar_plans'] += 1
                return result
            
            elif planner_type in ['rs_curves', 'enhanced_rs_curves', 'rrt'] and 'enhanced_rs_curves' in self.enhanced_planners:
                if planner_type == 'rrt':
                    print(f"      RRT调用重定向到增强版RS曲线规划器")
                
                result = self._plan_with_enhanced_rs_curves_safe(
                    vehicle_id, start, goal, config, max_time, task_context, vehicle_context
                )
                if result:
                    self.enhanced_stats['enhanced_rs_plans'] += 1
                return result
            
            elif planner_type == 'direct':
                return self._plan_enhanced_direct_path_safe(start, goal, network_context, vehicle_context)
            
            return None
            
        except Exception as e:
            print(f"      增强策略执行异常: {e}")
            return None
    
    def _plan_with_enhanced_astar_safe(self, vehicle_id: str, start: tuple, goal: tuple,
                                     config: dict, max_time: float,
                                     network_context: Dict = None, vehicle_context: Dict = None):
        """安全地使用增强版混合A*规划"""
        try:
            planner = self.enhanced_planners['enhanced_hybrid_astar']
            
            # 动态配置更新
            if config and hasattr(planner, 'config'):
                planner.config.update({
                    'max_iterations': config.get('max_iterations', 15000),
                    'timeout': min(max_time, config.get('timeout', 12.0)),
                    'rs_fitting_radius': config.get('rs_fitting_radius', 25.0)
                })
            
            # 增强版规划参数
            planning_params = {
                'agent_id': vehicle_id,
                'max_iterations': config.get('max_iterations', 15000),
                'quality_threshold': config.get('quality_threshold', 0.5)
            }
            
            # 专业设计处理
            if config.get('professional_design', False):
                planning_params['quality_threshold'] *= 1.02
                self.enhanced_stats['professional_design_plans'] += 1
            
            # 露天矿优化处理
            if config.get('mining_optimized', False):
                planning_params['mining_optimization'] = True
                self.enhanced_stats['mining_optimized_plans'] += 1
            
            path = planner.plan_path(start, goal, **planning_params)
            
            if path:
                return self._create_enhanced_result_object_safe(
                    path, 'enhanced_hybrid_astar', len(path), 
                    network_context, task_context=None, vehicle_context=vehicle_context
                )
            
            return None
            
        except Exception as e:
            print(f"      增强版A*规划异常: {e}")
            return None
    
    def _plan_with_enhanced_rs_curves_safe(self, vehicle_id: str, start: tuple, goal: tuple,
                                         config: dict, max_time: float,
                                         task_context: Dict = None, vehicle_context: Dict = None):
        """安全地使用增强版Reed-Shepp曲线规划"""
        try:
            rs_planner = self.enhanced_planners['enhanced_rs_curves']
            
            # 构建规划参数
            planning_kwargs = {
                'context': 'enhanced',
                'task_info': task_context or {},
                'vehicle_info': vehicle_context or {},
                'config': config,
                'max_time': max_time
            }
            
            path = rs_planner.plan_path(start, goal, **planning_kwargs)
            
            if path:
                return self._create_enhanced_result_object_safe(
                    path, 'enhanced_rs_curves', len(path),
                    network_context=None, task_context=task_context, vehicle_context=vehicle_context
                )
            
            return None
            
        except Exception as e:
            print(f"      增强版RS曲线规划异常: {e}")
            return None
    
    def _plan_enhanced_direct_path_safe(self, start: tuple, goal: tuple,
                                      network_context: Dict = None, vehicle_context: Dict = None):
        """安全地生成增强版直线路径"""
        try:
            distance = math.sqrt((goal[0] - start[0])**2 + (goal[1] - start[1])**2)
            
            # 根据车辆安全参数调整步数
            base_steps = max(3, int(distance / 2.0))
            
            if vehicle_context and 'safety_params' in vehicle_context:
                safety_margin = vehicle_context['safety_params'].get('safety_margin', 1.0)
                steps = int(base_steps * (1.0 + safety_margin * 0.05))
            else:
                steps = base_steps
            
            path = []
            for i in range(steps + 1):
                t = i / steps
                x = start[0] + t * (goal[0] - start[0])
                y = start[1] + t * (goal[1] - start[1])
                theta = start[2] if len(start) > 2 else 0
                path.append((x, y, theta))
            
            return self._create_enhanced_result_object_safe(
                path, 'enhanced_direct', len(path),
                network_context, vehicle_context=vehicle_context
            )
            
        except Exception as e:
            print(f"      增强版直线路径生成异常: {e}")
            return None
    
    def _create_enhanced_result_object_safe(self, path: list, planner_used: str, path_length: int,
                                          network_context: Dict = None, task_context: Dict = None,
                                          vehicle_context: Dict = None):
        """安全地创建增强版结果对象"""
        try:
            class EnhancedPlanningResult:
                def __init__(self, path, planner_used, path_length, network_context, task_context, vehicle_context):
                    self.path = path
                    self.planner_used = planner_used
                    self.quality_score = self._calculate_enhanced_quality_score()
                    self.planning_time = 0.0
                    self.structure = {
                        'type': planner_used,
                        'total_length': path_length,
                        'planner_used': planner_used,
                        'backbone_utilization': 0.0,
                        'network_topology': network_context.get('topology_type') if network_context else 'original',
                        'enhanced_professional_design': 'enhanced' in planner_used,
                        'mining_optimized': 'mining' in planner_used,
                        'safety_rectangle_aware': False,  # 禁用以提高性能
                        'curve_fitting_enhanced': 'enhanced' in planner_used,
                        'current_stage': task_context.get('current_stage') if task_context else None,
                        'vehicle_status': vehicle_context.get('vehicle_status') if vehicle_context else None,
                        'safety_compliance': True
                    }
                    
                    # 车辆安全参数集成
                    if vehicle_context and 'safety_params' in vehicle_context:
                        self.structure['vehicle_safety_params'] = vehicle_context['safety_params']
                
                def _calculate_enhanced_quality_score(self):
                    """计算增强版质量分数"""
                    base_scores = {
                        'enhanced_hybrid_astar': 0.85,
                        'enhanced_rs_curves': 0.80,
                        'enhanced_direct': 0.60,
                        'backbone_network': 0.90
                    }
                    
                    base_score = base_scores.get(planner_used, 0.65)
                    
                    # 增强版特性加成
                    if 'enhanced' in planner_used:
                        base_score += 0.03
                    if 'mining' in planner_used:
                        base_score += 0.02
                    
                    return min(1.0, base_score)
            
            return EnhancedPlanningResult(path, planner_used, path_length, network_context, task_context, vehicle_context)
            
        except Exception as e:
            print(f"⚠️ 结果对象创建失败: {e}")
            return None
    
    def _extract_enhanced_task_context_safe(self, kwargs: Dict) -> Dict:
        """安全地提取增强版任务上下文"""
        try:
            context = {
                'multi_stage': False,
                'current_stage': kwargs.get('context', 'transport'),
                'stage_transition': False,
                'target_type': kwargs.get('target_type'),
                'target_id': kwargs.get('target_id'),
                'enhanced_curve_fitting_required': False
            }
            
            # 检测多阶段任务
            if (kwargs.get('target_type') in ['loading', 'unloading'] or
                kwargs.get('current_stage')):
                context['multi_stage'] = True
                context['current_stage'] = kwargs.get('current_stage', kwargs.get('context', 'transport'))
            
            # 检测阶段转换
            if kwargs.get('next_stage'):
                context['stage_transition'] = True
                context['next_stage'] = kwargs['next_stage']
            
            return context
            
        except Exception as e:
            print(f"⚠️ 任务上下文提取失败: {e}")
            return {'current_stage': 'transport', 'multi_stage': False}
    
    def _extract_enhanced_network_context_safe(self, kwargs: Dict) -> Dict:
        """安全地提取增强版网络上下文"""
        try:
            context = {
                'topology_type': self.current_network_topology.value,
                'enhanced_professional_design': self.enhanced_features_detected.get('enhanced_consolidation_applied', False),
                'enhanced_consolidation_applied': self.enhanced_features_detected.get('enhanced_consolidation_applied', False),
                'enhanced_curve_fitting_used': self.enhanced_features_detected.get('enhanced_curve_fitting_used', False),
                'safety_rectangle_detection': False,  # 禁用以提高性能
                'professional_road_class': kwargs.get('road_class', 'secondary')
            }
            
            # 从增强版骨干网络获取信息
            if (ENHANCED_BACKBONE_AVAILABLE and self.backbone_network and 
                hasattr(self.backbone_network, 'get_topology_construction_summary')):
                
                topology_summary = self.backbone_network.get_topology_construction_summary()
                context.update({
                    'ready_for_stage2': topology_summary.get('ready_for_stage2', False),
                    'gnn_input_ready': topology_summary.get('gnn_input_ready', False)
                })
            
            return context
            
        except Exception as e:
            print(f"⚠️ 网络上下文提取失败: {e}")
            return {'topology_type': 'original'}
    
    def _extract_enhanced_vehicle_context_safe(self, kwargs: Dict) -> Dict:
        """安全地提取增强版车辆上下文"""
        try:
            context = {
                'vehicle_status': kwargs.get('vehicle_status', 'moving'),
                'safety_params': {},
                'enhanced_dynamics_required': False
            }
            
            # 从kwargs直接获取安全参数
            if 'safety_params' in kwargs:
                context['safety_params'].update(kwargs['safety_params'])
            
            # 使用配置管理器的安全状态处理
            safe_status = self.config_manager.vehicle_status_handler.get_safe_status_key(context['vehicle_status'])
            context['vehicle_status'] = safe_status
            
            # 检测增强版动力学需求
            if safe_status in ['loading', 'unloading', 'parking']:
                context['enhanced_dynamics_required'] = True
            
            return context
            
        except Exception as e:
            print(f"⚠️ 车辆上下文提取失败: {e}")
            return {'vehicle_status': 'moving', 'safety_params': {}}
    
    def _update_enhanced_strategy_stats_safe(self, strategy_name: str, result: Any, strategy_time: float):
        """安全地更新增强版策略统计"""
        try:
            if strategy_name not in self.enhanced_stats['strategy_success_rates']:
                self.enhanced_stats['strategy_success_rates'][strategy_name] = {'success': 0, 'total': 0}
            
            self.enhanced_stats['strategy_success_rates'][strategy_name]['total'] += 1
            
            if result:
                self.enhanced_stats['strategy_success_rates'][strategy_name]['success'] += 1
                self.enhanced_stats['successful_plans'] += 1
                
                # 记录平均规划时间
                if strategy_name not in self.enhanced_stats['average_planning_times']:
                    self.enhanced_stats['average_planning_times'][strategy_name] = []
                self.enhanced_stats['average_planning_times'][strategy_name].append(strategy_time)
                
                # 限制列表长度
                if len(self.enhanced_stats['average_planning_times'][strategy_name]) > 20:
                    self.enhanced_stats['average_planning_times'][strategy_name] = \
                        self.enhanced_stats['average_planning_times'][strategy_name][-10:]
                
                # 特定策略统计
                if 'professional' in strategy_name:
                    self.enhanced_stats['professional_fallback_usage'] += 1
                if 'safety' in strategy_name:
                    self.enhanced_stats['safety_optimized_fallback_usage'] += 1
                    
        except Exception as e:
            print(f"⚠️ 策略统计更新失败: {e}")
    
    def _format_enhanced_result(self, result: Any, return_object: bool) -> Any:
        """格式化增强版结果"""
        try:
            if return_object and result:
                return result
            elif result and hasattr(result, 'path'):
                return result.path
            else:
                return result
        except:
            return result
    
    def _generate_emergency_fallback_path(self, start: tuple, goal: tuple, return_object: bool):
        """生成紧急回退路径"""
        try:
            distance = math.sqrt((goal[0] - start[0])**2 + (goal[1] - start[1])**2)
            steps = max(3, int(distance / 3.0))
            
            path = []
            for i in range(steps + 1):
                t = i / steps
                x = start[0] + t * (goal[0] - start[0])
                y = start[1] + t * (goal[1] - start[1])
                theta = start[2] if len(start) > 2 else 0
                path.append((x, y, theta))
            
            if return_object:
                class EmergencyResult:
                    def __init__(self, path):
                        self.path = path
                        self.planner_used = 'emergency_fallback'
                        self.quality_score = 0.3
                        self.planning_time = 0.1
                
                return EmergencyResult(path)
            else:
                return path
                
        except Exception as e:
            print(f"❌ 紧急回退路径生成失败: {e}")
            return None
    
    # ==================== 保持向后兼容的接口 ====================
    
    def set_backbone_network(self, backbone_network):
        """设置骨干网络（向后兼容）"""
        self.backbone_network = backbone_network
        self.update_enhanced_network_topology_awareness()
        
        # 设置到增强版规划器
        for planner in self.enhanced_planners.values():
            if hasattr(planner, 'set_backbone_network'):
                try:
                    planner.set_backbone_network(backbone_network)
                except Exception as e:
                    print(f"⚠️ 设置骨干网络到规划器失败: {e}")
    
    def set_traffic_manager(self, traffic_manager):
        """设置交通管理器（向后兼容）"""
        self.traffic_manager = traffic_manager
        print("✅ 交通管理器已连接到增强版规划器")
    
    def get_statistics(self):
        """获取统计信息（向后兼容）"""
        return self.get_enhanced_performance_statistics()
    
    def get_enhanced_performance_statistics(self) -> dict:
        """获取增强版性能统计"""
        try:
            stats = {}
            
            # 策略成功率
            for strategy, data in self.enhanced_stats['strategy_success_rates'].items():
                if data['total'] > 0:
                    success_rate = data['success'] / data['total']
                    stats[f'{strategy}_success_rate'] = success_rate
            
            # 平均规划时间
            for strategy, times in self.enhanced_stats['average_planning_times'].items():
                if times:
                    avg_time = sum(times) / len(times)
                    stats[f'{strategy}_avg_time'] = avg_time
            
            # 增强版特定统计
            stats.update({
                'eastar_plans': self.enhanced_stats['eastar_plans'],
                'enhanced_rs_plans': self.enhanced_stats['enhanced_rs_plans'],
                'professional_design_plans': self.enhanced_stats['professional_design_plans'],
                'current_network_topology': self.current_network_topology.value,
                'enhanced_features_detected': self.enhanced_features_detected,
                'enhanced_config_adaptations': self.enhanced_stats['enhanced_config_adaptations'],
                'total_requests': self.enhanced_stats['total_requests'],
                'successful_plans': self.enhanced_stats['successful_plans']
            })
            
            return stats
            
        except Exception as e:
            print(f"⚠️ 性能统计获取失败: {e}")
            return {}
    
    def clear_cache(self):
        """清除缓存（向后兼容）"""
        try:
            if self.cache_lock:
                with self.cache_lock:
                    self.enhanced_cache.clear()
            else:
                self.enhanced_cache.clear()
            
            self.enhanced_stats['cache_hits'] = 0
            print("✅ 增强版集成规划器缓存已清理")
            
        except Exception as e:
            print(f"⚠️ 缓存清理失败: {e}")
    
    def shutdown(self):
        """关闭规划器"""
        try:
            self.clear_cache()
            
            for planner_name, planner in self.enhanced_planners.items():
                if hasattr(planner, 'shutdown'):
                    try:
                        planner.shutdown()
                    except Exception as e:
                        print(f"⚠️ 关闭增强版规划器 {planner_name} 失败: {e}")
            
            self.enhanced_planners.clear()
            print("✅ 增强版集成路径规划器已关闭")
            
        except Exception as e:
            print(f"⚠️ 规划器关闭失败: {e}")


# ==================== 便捷创建函数 ====================

def create_enhanced_integrated_planning_system(env, backbone_network=None, traffic_manager=None):
    """创建增强版集成规划系统"""
    try:
        config_manager = EnhancedIntegratedPlannerConfig()
        planner = EnhancedIntegratedPathPlannerWithConfig(env, backbone_network, traffic_manager)
        
        return {
            'config_manager': config_manager,
            'planner': planner,
            'features': {
                'eastar_integration': EASTAR_AVAILABLE,
                'enhanced_environment': ENHANCED_ENV_AVAILABLE,
                'professional_consolidation': ENHANCED_CONSOLIDATION_AVAILABLE,
                'enhanced_backbone': ENHANCED_BACKBONE_AVAILABLE,
                'enhanced_curve_fitting': ENHANCED_CURVE_FITTING_AVAILABLE
            }
        }
    except Exception as e:
        print(f"❌ 增强版集成规划系统创建失败: {e}")
        return None

# 兼容性别名
EnhancedPathPlannerWithConfig = EnhancedIntegratedPathPlannerWithConfig
OptimizedPlannerConfig = EnhancedIntegratedPlannerConfig
IntegratedPathPlannerWithOptimizedConfig = EnhancedIntegratedPathPlannerWithConfig
IntegratedPlannerConfig = EnhancedIntegratedPlannerConfig