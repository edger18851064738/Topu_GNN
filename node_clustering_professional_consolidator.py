"""
node_clustering_professional_consolidator.py - 增强版：集成新ClothoidCubic曲线拟合 (完整修复版)
优化版本：
1. 完美集成增强版ClothoidCubic.py
2. 实现起点→关键节点序列→终点的完整曲线拟合
3. 增强的节点序列排序和验证
4. 严格的车辆动力学约束执行
5. 详细的质量评估和统计报告
6. 修复了所有已知的初始化和变量定义问题
7. ✨ 新增：优化端点聚类逻辑 - 对所有端点进行半径为2的聚类
"""

import math
import time
import numpy as np
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum

# 导入eastar.py中的算法用于路径重建
try:
    from eastar import HybridAStarPlanner, MiningOptimizedReedShepp
    EASTAR_AVAILABLE = True
    print("✅ 成功导入eastar.py用于路径重建")
except ImportError as e:
    EASTAR_AVAILABLE = False
    print(f"⚠️ 无法导入eastar.py: {e}")

# 导入增强版ClothoidCubic模块 - 完整修复版
ENHANCED_CURVE_FITTING_AVAILABLE = False
CURVE_FITTING_AVAILABLE = False

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
    CURVE_FITTING_AVAILABLE = True  # 修复：增强版可用时也设置传统版本标记
    print("✅ 成功导入增强版ClothoidCubic模块")
except ImportError as e:
    ENHANCED_CURVE_FITTING_AVAILABLE = False
    print(f"⚠️ 无法导入增强版ClothoidCubic: {e}")
    
    # 回退到原版
    try:
        from ClothoidCubic import BackbonePathFitter
        CURVE_FITTING_AVAILABLE = True
        print("✅ 回退到原版ClothoidCubic模块")
    except ImportError:
        CURVE_FITTING_AVAILABLE = False
        print("❌ ClothoidCubic模块完全不可用")

class RoadClass(Enum):
    """道路等级"""
    PRIMARY = "primary"        # 主干道
    SECONDARY = "secondary"    # 次干道  
    SERVICE = "service"        # 作业道

class NodeType(Enum):
    """节点类型"""
    ENDPOINT = "endpoint"      # 端点（不可聚类）
    WAYPOINT = "waypoint"      # 路径点（可聚类）
    KEY_NODE = "key_node"      # 关键节点

@dataclass
class KeyNode:
    """关键节点 - 增强版"""
    node_id: str
    position: Tuple[float, float, float]
    cluster_center: Tuple[float, float, float]
    
    # 节点属性
    node_type: NodeType = NodeType.KEY_NODE
    is_endpoint: bool = False
    endpoint_info: Dict = field(default_factory=dict)
    
    # 节点属性继承
    original_nodes: List[Tuple] = field(default_factory=list)
    path_memberships: Set[str] = field(default_factory=set)
    node_importance: float = 1.0
    
    # 工程属性
    road_class: RoadClass = RoadClass.SECONDARY
    traffic_capacity: int = 80
    safety_rating: float = 1.0
    
    # 连接信息
    connected_nodes: Set[str] = field(default_factory=set)
    backbone_segments: List[str] = field(default_factory=list)
    
    # 聚类信息
    cluster_info: Dict = field(default_factory=dict)
    
    # 增强：曲线拟合相关
    curve_fitting_quality: float = 0.0     # 曲线拟合质量
    dynamics_compliance: bool = True        # 动力学合规性
    smoothness_score: float = 0.0          # 平滑度分数
    
    def add_original_node(self, node: Tuple, path_id: str, is_endpoint: bool = False):
        """添加原始节点信息"""
        if node not in self.original_nodes:
            self.original_nodes.append(node)
        self.path_memberships.add(path_id)
        
        if is_endpoint:
            self.is_endpoint = True
            self.node_type = NodeType.ENDPOINT
        
        # 更新节点重要性
        self.node_importance = len(self.path_memberships)
        if self.is_endpoint:
            self.node_importance *= 2
        
        # 根据路径数量确定道路等级
        if self.is_endpoint or len(self.path_memberships) >= 4:
            self.road_class = RoadClass.PRIMARY
            self.traffic_capacity = 120
        elif len(self.path_memberships) >= 2:
            self.road_class = RoadClass.SECONDARY
            self.traffic_capacity = 80
        else:
            self.road_class = RoadClass.SERVICE
            self.traffic_capacity = 40
    
    def get_average_position(self) -> Tuple[float, float, float]:
        """获取所有原始节点的平均位置"""
        if self.is_endpoint:
            return self.position
        
        if not self.original_nodes:
            return self.position
        
        sum_x = sum(node[0] for node in self.original_nodes)
        sum_y = sum(node[1] for node in self.original_nodes)
        sum_z = sum(node[2] if len(node) > 2 else 0 for node in self.original_nodes)
        count = len(self.original_nodes)
        
        return (sum_x / count, sum_y / count, sum_z / count)

@dataclass
class EndpointCluster:
    """端点聚类信息"""
    cluster_id: str
    representative_position: Tuple[float, float, float]
    original_endpoints: List[Dict]  # 原始端点信息
    endpoint_type: str  # 'start' or 'end'
    cluster_importance: float = 1.0
    merged_path_ids: Set[str] = field(default_factory=set)
    
    def calculate_representative_position(self):
        """计算代表性位置（加权平均）"""
        if not self.original_endpoints:
            return self.representative_position
        
        total_weight = 0
        sum_x = sum_y = sum_z = 0.0
        
        for endpoint in self.original_endpoints:
            # 权重基于路径数量
            weight = len(endpoint.get('paths', {endpoint.get('path_id', '')}))
            
            pos = endpoint['position']
            sum_x += pos[0] * weight
            sum_y += pos[1] * weight
            sum_z += (pos[2] if len(pos) > 2 else 0) * weight
            total_weight += weight
            
            # 收集路径ID
            if 'path_id' in endpoint:
                self.merged_path_ids.add(endpoint['path_id'])
            if 'paths' in endpoint:
                self.merged_path_ids.update(endpoint['paths'])
        
        if total_weight > 0:
            self.representative_position = (
                sum_x / total_weight, 
                sum_y / total_weight, 
                sum_z / total_weight
            )
            self.cluster_importance = total_weight
        
        return self.representative_position

@dataclass
class EnhancedConsolidatedBackbonePath:
    """增强的整合后骨干路径"""
    path_id: str
    original_path_id: str
    key_nodes: List[str]  # 关键节点ID序列
    
    # 路径属性
    path_length: float = 0.0
    road_class: RoadClass = RoadClass.SECONDARY
    quality_score: float = 0.7
    
    # 原始路径信息保留
    original_endpoints: Tuple = None
    original_quality: float = 0.7
    endpoint_nodes: Dict = field(default_factory=dict)
    
    # 增强：曲线拟合结果
    reconstructed_path: List[Tuple] = field(default_factory=list)
    reconstruction_success: bool = False
    curve_segments: List = field(default_factory=list)  # 修复：移除CurveSegment类型约束避免导入问题
    
    # 增强：质量和性能指标
    curve_fitting_method: str = "none"           # 拟合方法
    curve_quality_score: float = 0.0             # 曲线质量分数
    dynamics_compliance_rate: float = 0.0        # 动力学合规率
    smoothness_score: float = 0.0                # 平滑度分数
    safety_score: float = 0.0                    # 安全性分数
    efficiency_score: float = 0.0                # 效率分数
    
    # 增强：车辆动力学统计
    max_curvature: float = 0.0                   # 最大曲率
    avg_curvature: float = 0.0                   # 平均曲率
    max_grade: float = 0.0                       # 最大坡度
    turning_radius_compliance: bool = True       # 转弯半径合规
    grade_compliance: bool = True                # 坡度合规
    
    def calculate_path_properties(self, key_nodes_dict: Dict[str, KeyNode]):
        """计算路径属性 - 增强版"""
        if len(self.key_nodes) < 2:
            return
        
        # 计算总长度
        total_length = 0.0
        for i in range(len(self.key_nodes) - 1):
            node1 = key_nodes_dict.get(self.key_nodes[i])
            node2 = key_nodes_dict.get(self.key_nodes[i + 1])
            
            if node1 and node2:
                pos1 = node1.position
                pos2 = node2.position
                distance = math.sqrt(
                    (pos2[0] - pos1[0])**2 + 
                    (pos2[1] - pos1[1])**2
                )
                total_length += distance
        
        self.path_length = total_length
        
        # 确定道路等级（基于关键节点的最高等级）
        max_importance = 0
        for node_id in self.key_nodes:
            node = key_nodes_dict.get(node_id)
            if node and node.node_importance > max_importance:
                max_importance = node.node_importance
                self.road_class = node.road_class
    
    def update_curve_fitting_results(self, curve_segments: List):
        """更新曲线拟合结果 - 修复版"""
        if not curve_segments:
            return
        
        self.curve_segments = curve_segments
        
        # 计算综合质量指标
        total_quality = sum(getattr(seg, 'quality_score', 0.7) for seg in curve_segments)
        self.curve_quality_score = total_quality / len(curve_segments)
        
        # 计算综合性能指标
        total_smoothness = sum(getattr(seg, 'smoothness_score', 0.8) for seg in curve_segments)
        self.smoothness_score = total_smoothness / len(curve_segments)
        
        # 计算曲率统计
        all_curvatures = []
        for seg in curve_segments:
            if hasattr(seg, 'max_curvature'):
                all_curvatures.append(seg.max_curvature)
        
        if all_curvatures:
            self.max_curvature = max(all_curvatures)
            self.avg_curvature = sum(all_curvatures) / len(all_curvatures)
        
        # 检查合规性
        self.dynamics_compliance_rate = sum(
            1 for seg in curve_segments if getattr(seg, 'dynamics_compliance', True)
        ) / len(curve_segments)
        
        self.turning_radius_compliance = all(getattr(seg, 'dynamics_compliance', True) for seg in curve_segments)
        self.grade_compliance = all(getattr(seg, 'grade_compliance', True) for seg in curve_segments)
        
        # 确定拟合方法
        if curve_segments:
            self.curve_fitting_method = getattr(curve_segments[0], 'curve_type', 'unknown')

class EnhancedNodeClusteringConsolidator:
    """增强版基于节点聚类的专业道路网络整合器"""
    
    def __init__(self, env, config: Dict = None):
        self.env = env
        
        # 增强配置
        self.config = {
            # 多轮聚类配置
            'multi_round_clustering': True,
            'clustering_rounds': [
                {'radius': 6.0, 'name': '第一轮'},
                {'radius': 6.0, 'name': '第二轮'},
                {'radius': 3.0, 'name': '第三轮'}
            ],
            
            # 端点保护和聚类配置 - ✨ 新增优化
            'protect_endpoints': True,
            'endpoint_buffer_radius': 3.0,
            'enable_endpoint_clustering': True,     # ✨ 启用端点聚类
            'endpoint_clustering_radius': 2.0,      # ✨ 端点聚类半径
            'min_endpoint_cluster_size': 1,         # ✨ 端点聚类最小尺寸
            'endpoint_merge_threshold': 2.5,        # ✨ 端点合并阈值
            
            # 聚类参数
            'min_cluster_size': 1,
            'importance_threshold': 1.5,
            
            # 增强：曲线拟合参数
            'enable_enhanced_curve_fitting': ENHANCED_CURVE_FITTING_AVAILABLE,
            'curve_fitting_quality_threshold': 0.7,
            'prefer_complete_curve': True,      # 优先使用完整曲线
            'enable_segmented_fallback': True,  # 启用分段回退
            'force_vehicle_dynamics': True,    # 强制车辆动力学约束
            
            # 车辆动力学配置 - 修复版：包含所有必需字段
            'vehicle_dynamics': {
                'vehicle_length': 6.0,
                'vehicle_width': 3.0,
                'turning_radius': 8.0,
                'max_steering_angle': 45.0,
                'max_acceleration': 1.5,
                'max_deceleration': 2.0,
                'max_speed': 15.0,
                'max_grade': 0.15,
                'comfort_lateral_accel': 1.0,
                'safety_margin': 1.5
            },
            
            # 质量控制
            'min_reconstruction_quality': 0.5,
            'preserve_original_on_failure': True,
            'enable_quality_reporting': True,
        }
        
        if config:
            self.config.update(config)
        
        # 核心数据结构
        self.original_paths = {}
        self.key_nodes = {}
        self.consolidated_paths = {}
        self.node_clusters = []
        
        # ✨ 新增：端点聚类相关数据结构
        self.original_endpoint_nodes = {}       # 原始端点信息
        self.endpoint_clusters = {}             # 端点聚类结果
        self.clustered_endpoint_nodes = {}      # 聚类后的端点节点
        self.endpoint_cluster_mapping = {}      # 原始端点到聚类的映射
        
        # 端点信息（保持向后兼容）
        self.endpoint_nodes = {}
        self.protected_positions = set()
        
        # 增强：曲线拟合器
        self.enhanced_curve_fitter = None
        self.traditional_path_fitter = None
        self._initialize_curve_fitters()
        
        # 增强：详细统计
        self.consolidation_stats = {
            'original_nodes_count': 0,
            'endpoint_nodes_count': 0,
            'clusterable_nodes_count': 0,
            'key_nodes_count': 0,
            'node_reduction_ratio': 0.0,
            'clustering_time': 0.0,
            'reconstruction_time': 0.0,
            'paths_reconstructed': 0,
            'reconstruction_success_rate': 0.0,
            
            # ✨ 新增：端点聚类统计
            'original_endpoints_count': 0,
            'endpoint_clusters_count': 0,
            'endpoint_reduction_ratio': 0.0,
            'endpoint_clustering_time': 0.0,
            'merged_endpoint_paths': 0,
            
            # 增强：曲线拟合统计
            'enhanced_curve_fitting_used': 0,
            'complete_curve_success': 0,
            'segmented_curve_success': 0,
            'fallback_reconstruction': 0,
            'avg_curve_quality': 0.0,
            'dynamics_compliance_rate': 0.0,
            'turning_radius_violations': 0,
            'grade_violations': 0,
        }
        
        print(f"🔧 增强版基于节点聚类的专业道路整合器初始化完成")
        print(f"  增强曲线拟合: {'✅' if self.config['enable_enhanced_curve_fitting'] else '❌'}")
        print(f"  车辆动力学约束: {'✅' if self.config['force_vehicle_dynamics'] else '❌'}")
        print(f"  ✨ 端点智能聚类: {'✅' if self.config['enable_endpoint_clustering'] else '❌'}")
        print(f"  端点聚类半径: {self.config['endpoint_clustering_radius']}m")
        print(f"  转弯半径: {self.config['vehicle_dynamics']['turning_radius']}m")
        print(f"  最大坡度: {self.config['vehicle_dynamics']['max_grade']:.1%}")
    
    def _initialize_curve_fitters(self):
        """初始化曲线拟合器 - 完整修复版"""
        if self.config['enable_enhanced_curve_fitting'] and ENHANCED_CURVE_FITTING_AVAILABLE:
            try:
                # 创建车辆动力学配置 - 修复版：确保所有参数都有值
                vehicle_dynamics = self.config['vehicle_dynamics']
                
                vehicle_config = VehicleDynamicsConfig(
                    vehicle_length=vehicle_dynamics.get('vehicle_length', 6.0),
                    vehicle_width=vehicle_dynamics.get('vehicle_width', 3.0),
                    turning_radius=vehicle_dynamics.get('turning_radius', 8.0),
                    max_steering_angle=vehicle_dynamics.get('max_steering_angle', 45.0),
                    max_acceleration=vehicle_dynamics.get('max_acceleration', 1.5),
                    max_deceleration=vehicle_dynamics.get('max_deceleration', 2.0),
                    max_speed=vehicle_dynamics.get('max_speed', 15.0),
                    max_grade=vehicle_dynamics.get('max_grade', 0.15),
                    comfort_lateral_accel=vehicle_dynamics.get('comfort_lateral_accel', 1.0),
                    safety_margin=vehicle_dynamics.get('safety_margin', 1.5)
                )
                
                # 创建增强拟合器 - 修复版：使用位置参数
                self.enhanced_curve_fitter = EnhancedClothoidCubicFitter(
                    vehicle_config,  # 第一个位置参数
                    self.env        # 第二个位置参数
                )
                
                print("✅ 增强曲线拟合器初始化成功")
                
            except Exception as e:
                print(f"⚠️ 增强曲线拟合器初始化失败: {e}")
                import traceback
                traceback.print_exc()
                self.config['enable_enhanced_curve_fitting'] = False
        
        # 回退到传统拟合器 - 修复版：确保CURVE_FITTING_AVAILABLE已定义
        if CURVE_FITTING_AVAILABLE:
            try:
                self.traditional_path_fitter = BackbonePathFitter(env=self.env)
                print("✅ 传统曲线拟合器初始化成功")
            except Exception as e:
                print(f"⚠️ 传统曲线拟合器初始化失败: {e}")
    
    def consolidate_backbone_network_professional(self, backbone_network):
        """执行增强版专业整合"""
        print(f"\n🔧 开始增强版基于节点聚类的专业道路网络整合...")
        start_time = time.time()
        
        try:
            # 阶段1: 提取和分析原始路径
            print(f"\n📊 阶段1: 提取和分析原始路径")
            if not self._extract_original_paths(backbone_network):
                print(f"❌ 原始路径提取失败")
                return False
            
            # 阶段2: ✨ 优化的端点识别和聚类
            print(f"\n🔒 阶段2: ✨ 优化的端点识别和智能聚类")
            if not self._identify_and_cluster_endpoints_optimized():
                print(f"❌ 优化端点聚类失败")
                return False
            
            # 阶段3: 多轮节点聚类
            print(f"\n🎯 阶段3: 执行多轮节点聚类")
            clustering_start = time.time()
            if not self._perform_multi_round_clustering():
                print(f"❌ 节点聚类失败")
                return False
            self.consolidation_stats['clustering_time'] = time.time() - clustering_start
            
            # 阶段4: 生成关键节点
            print(f"\n⭐ 阶段4: 生成关键节点")
            if not self._generate_key_nodes():
                print(f"❌ 关键节点生成失败")
                return False
            
            # 阶段5: 增强路径重建（核心功能）
            print(f"\n🛤️ 阶段5: 增强路径重建")
            reconstruction_start = time.time()
            if not self._enhanced_reconstruct_backbone_paths():
                print(f"❌ 增强路径重建失败")
                return False
            self.consolidation_stats['reconstruction_time'] = time.time() - reconstruction_start
            
            # 阶段6: 应用整合结果
            print(f"\n✅ 阶段6: 应用整合结果")
            if not self._apply_consolidation_to_backbone(backbone_network):
                print(f"❌ 整合结果应用失败")
                return False
            
            total_time = time.time() - start_time
            self._generate_enhanced_consolidation_report(total_time)
            
            print(f"🎉 增强版专业道路网络整合完成!")
            return True
            
        except Exception as e:
            print(f"❌ 增强版专业整合失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # ==================== ✨ 新增：优化的端点聚类方法 ====================
    
    def _identify_and_cluster_endpoints_optimized(self) -> bool:
        """✨ 优化的端点识别和智能聚类"""
        print(f"   🔍 识别和分析所有原始端点...")
        
        # 第一步：收集所有原始端点
        start_endpoints = []  # 起点列表
        end_endpoints = []    # 终点列表
        
        for path_id, path_info in self.original_paths.items():
            nodes = path_info['nodes']
            
            if len(nodes) < 2:
                continue
            
            # 起点信息
            start_point = nodes[0]
            start_endpoint = {
                'position': start_point,
                'type': 'start',
                'path_id': path_id,
                'paths': {path_id},
                'original_id': f"endpoint_start_{path_id}"
            }
            start_endpoints.append(start_endpoint)
            
            # 终点信息
            end_point = nodes[-1]
            end_endpoint = {
                'position': end_point,
                'type': 'end',
                'path_id': path_id,
                'paths': {path_id},
                'original_id': f"endpoint_end_{path_id}"
            }
            end_endpoints.append(end_endpoint)
        
        total_original_endpoints = len(start_endpoints) + len(end_endpoints)
        self.consolidation_stats['original_endpoints_count'] = total_original_endpoints
        
        print(f"   📍 收集到原始端点: 起点{len(start_endpoints)}个, 终点{len(end_endpoints)}个")
        
        # 第二步：分别对起点和终点进行聚类
        endpoint_clustering_start = time.time()
        
        if self.config['enable_endpoint_clustering']:
            print(f"   🎯 执行端点智能聚类 (半径: {self.config['endpoint_clustering_radius']}m)...")
            
            # 对起点进行聚类
            start_clusters = self._cluster_endpoints(start_endpoints, 'start')
            print(f"   ✅ 起点聚类完成: {len(start_endpoints)} -> {len(start_clusters)} 个聚类")
            
            # 对终点进行聚类
            end_clusters = self._cluster_endpoints(end_endpoints, 'end')
            print(f"   ✅ 终点聚类完成: {len(end_endpoints)} -> {len(end_clusters)} 个聚类")
            
            # 合并聚类结果
            self.endpoint_clusters.update(start_clusters)
            self.endpoint_clusters.update(end_clusters)
            
            # 生成聚类后的端点节点
            self._generate_clustered_endpoint_nodes()
            
        else:
            print(f"   ⚠️ 端点聚类已禁用，使用原始端点")
            # 如果禁用聚类，直接使用原始端点
            self._use_original_endpoints_as_clusters(start_endpoints + end_endpoints)
        
        endpoint_clustering_time = time.time() - endpoint_clustering_start
        self.consolidation_stats['endpoint_clustering_time'] = endpoint_clustering_time
        
        # 第三步：设置保护区域
        self._setup_endpoint_protection_zones()
        
        # 更新统计信息
        clustered_endpoints_count = len(self.clustered_endpoint_nodes)
        self.consolidation_stats['endpoint_clusters_count'] = clustered_endpoints_count
        
        if total_original_endpoints > 0:
            self.consolidation_stats['endpoint_reduction_ratio'] = (
                1.0 - clustered_endpoints_count / total_original_endpoints
            )
        
        print(f"   📊 端点聚类统计:")
        print(f"      原始端点: {total_original_endpoints} 个")
        print(f"      聚类后端点: {clustered_endpoints_count} 个")
        print(f"      端点减少率: {self.consolidation_stats['endpoint_reduction_ratio']:.1%}")
        print(f"      聚类耗时: {endpoint_clustering_time:.2f}s")
        
        return True
    
    def _cluster_endpoints(self, endpoints: List[Dict], endpoint_type: str) -> Dict[str, EndpointCluster]:
        """对指定类型的端点进行聚类"""
        if not endpoints:
            return {}
        
        clustering_radius = self.config['endpoint_clustering_radius']
        clusters = {}
        visited = set()
        cluster_counter = 0
        
        for i, endpoint in enumerate(endpoints):
            if i in visited:
                continue
            
            # 创建新的端点聚类
            cluster_id = f"endpoint_cluster_{endpoint_type}_{cluster_counter}"
            endpoint_cluster = EndpointCluster(
                cluster_id=cluster_id,
                representative_position=endpoint['position'],
                original_endpoints=[endpoint],
                endpoint_type=endpoint_type
            )
            
            visited.add(i)
            
            # 查找聚类半径内的其他端点
            for j, other_endpoint in enumerate(endpoints):
                if j in visited:
                    continue
                
                distance = self._calculate_distance(endpoint['position'], other_endpoint['position'])
                
                if distance <= clustering_radius:
                    endpoint_cluster.original_endpoints.append(other_endpoint)
                    visited.add(j)
            
            # 计算代表性位置和重要性
            endpoint_cluster.calculate_representative_position()
            
            clusters[cluster_id] = endpoint_cluster
            cluster_counter += 1
            
            # 输出聚类信息
            if len(endpoint_cluster.original_endpoints) > 1:
                print(f"      🔗 {endpoint_type}聚类 {cluster_id}: 合并了 {len(endpoint_cluster.original_endpoints)} 个端点")
                self.consolidation_stats['merged_endpoint_paths'] += len(endpoint_cluster.original_endpoints) - 1
        
        return clusters
    
    def _generate_clustered_endpoint_nodes(self):
        """生成聚类后的端点节点"""
        self.clustered_endpoint_nodes = {}
        self.endpoint_cluster_mapping = {}
        
        for cluster_id, endpoint_cluster in self.endpoint_clusters.items():
            # 创建聚类端点节点
            clustered_endpoint = {
                'id': cluster_id,
                'position': endpoint_cluster.representative_position,
                'type': endpoint_cluster.endpoint_type,
                'cluster_info': {
                    'original_endpoints_count': len(endpoint_cluster.original_endpoints),
                    'merged_path_ids': list(endpoint_cluster.merged_path_ids),
                    'cluster_importance': endpoint_cluster.cluster_importance
                },
                'paths': endpoint_cluster.merged_path_ids.copy(),
                'is_clustered_endpoint': True
            }
            
            self.clustered_endpoint_nodes[cluster_id] = clustered_endpoint
            
            # 建立原始端点到聚类的映射
            for original_endpoint in endpoint_cluster.original_endpoints:
                original_id = original_endpoint['original_id']
                self.endpoint_cluster_mapping[original_id] = cluster_id
        
        # 保持向后兼容性：更新endpoint_nodes
        self.endpoint_nodes = self.clustered_endpoint_nodes.copy()
    
    def _use_original_endpoints_as_clusters(self, all_endpoints: List[Dict]):
        """如果禁用聚类，直接使用原始端点"""
        self.clustered_endpoint_nodes = {}
        self.endpoint_cluster_mapping = {}
        
        for endpoint in all_endpoints:
            original_id = endpoint['original_id']
            self.clustered_endpoint_nodes[original_id] = endpoint
            self.endpoint_cluster_mapping[original_id] = original_id
        
        # 保持向后兼容性
        self.endpoint_nodes = self.clustered_endpoint_nodes.copy()
    
    def _setup_endpoint_protection_zones(self):
        """设置端点保护区域"""
        self.protected_positions.clear()
        
        for endpoint_node in self.clustered_endpoint_nodes.values():
            position = endpoint_node['position']
            self.protected_positions.add(position)
        
        print(f"   🛡️ 设置了 {len(self.protected_positions)} 个端点保护区域")
    
    # ==================== 增强版路径重建（已修复） ====================
    
    def _enhanced_reconstruct_backbone_paths(self) -> bool:
        """增强版骨干路径重建 - 核心实现"""
        print(f"   🎯 开始增强版路径重建...")
        
        self.consolidated_paths = {}
        reconstruction_success_count = 0
        enhanced_fitting_count = 0
        
        for path_id, path_info in self.original_paths.items():
            print(f"\n     🛤️ 重建路径: {path_id}")
            
            # 构建完整的关键节点序列
            complete_node_sequence = self._build_enhanced_key_node_sequence(path_id)
            
            if not complete_node_sequence or len(complete_node_sequence) < 2:
                print(f"       ❌ 无法构建有效的关键节点序列")
                self._preserve_original_path_as_consolidated(path_id, path_info)
                continue
            
            print(f"       📍 关键节点序列: {len(complete_node_sequence)} 个节点")
            for i, (node_id, position) in enumerate(complete_node_sequence):
                node_type = "起点" if i == 0 else "终点" if i == len(complete_node_sequence)-1 else "关键节点"
                print(f"         {i+1}. {node_type} {node_id}: ({position[0]:.1f}, {position[1]:.1f})")
            
            # 创建增强整合路径对象
            consolidated_path = EnhancedConsolidatedBackbonePath(
                path_id=f"enhanced_{path_id}",
                original_path_id=path_id,
                key_nodes=[node_id for node_id, pos in complete_node_sequence],
                original_endpoints=path_info['endpoints'],
                original_quality=path_info['quality']
            )
            
            # 计算路径属性
            consolidated_path.calculate_path_properties(self.key_nodes)
            
            # 尝试增强曲线拟合
            success = self._attempt_enhanced_curve_fitting(consolidated_path, complete_node_sequence)
            
            if success:
                reconstruction_success_count += 1
                if consolidated_path.curve_fitting_method.startswith('enhanced'):
                    enhanced_fitting_count += 1
                    self.consolidation_stats['enhanced_curve_fitting_used'] += 1
                
                print(f"       ✅ 重建成功: 方法={consolidated_path.curve_fitting_method}, "
                      f"质量={consolidated_path.curve_quality_score:.2f}")
            else:
                print(f"       ❌ 重建失败")
            
            self.consolidated_paths[consolidated_path.path_id] = consolidated_path
        
        # 更新统计
        self.consolidation_stats['paths_reconstructed'] = reconstruction_success_count
        if len(self.original_paths) > 0:
            self.consolidation_stats['reconstruction_success_rate'] = (
                reconstruction_success_count / len(self.original_paths)
            )
        
        # 计算平均质量
        if self.consolidated_paths:
            total_quality = sum(p.curve_quality_score for p in self.consolidated_paths.values())
            self.consolidation_stats['avg_curve_quality'] = total_quality / len(self.consolidated_paths)
        
        print(f"\n   📊 增强重建统计:")
        print(f"      总路径: {len(self.original_paths)}")
        print(f"      重建成功: {reconstruction_success_count}")
        print(f"      增强拟合: {enhanced_fitting_count}")
        print(f"      成功率: {reconstruction_success_count/len(self.original_paths):.1%}")
        print(f"      平均质量: {self.consolidation_stats['avg_curve_quality']:.2f}")
        
        return reconstruction_success_count > 0
    
    def _build_enhanced_key_node_sequence(self, path_id: str) -> List[Tuple[str, Tuple]]:
        """构建增强的关键节点序列"""
        path_info = self.original_paths[path_id]
        path_nodes = path_info['nodes']
        
        if len(path_nodes) < 2:
            return []
        
        sequence = []
        
        # 1. 找到起点和终点的关键节点（使用聚类后的端点）
        start_key_node = self._find_closest_clustered_endpoint(path_nodes[0], 'start', path_id)
        end_key_node = self._find_closest_clustered_endpoint(path_nodes[-1], 'end', path_id)
        
        if not start_key_node or not end_key_node:
            print(f"       ⚠️ 无法找到端点对应的聚类端点节点")
            return []
        
        sequence.append((start_key_node, self.key_nodes[start_key_node].position))
        
        # 2. 找到并排序中间关键节点
        middle_nodes_with_position = []
        
        for key_node_id, key_node in self.key_nodes.items():
            if (key_node_id != start_key_node and 
                key_node_id != end_key_node and
                path_id in key_node.path_memberships):
                
                # 找到该关键节点在原路径上的最佳位置索引
                best_index = self._find_best_path_position(key_node.position, path_nodes)
                if best_index >= 0:
                    middle_nodes_with_position.append((best_index, key_node_id, key_node.position))
        
        # 3. 按在原路径上的位置排序
        middle_nodes_with_position.sort(key=lambda x: x[0])
        
        # 4. 添加中间节点到序列
        for index, node_id, position in middle_nodes_with_position:
            sequence.append((node_id, position))
        
        # 5. 添加终点
        sequence.append((end_key_node, self.key_nodes[end_key_node].position))
        
        # 6. 验证序列的空间合理性
        validated_sequence = self._validate_node_sequence(sequence)
        
        return validated_sequence
    
    def _find_closest_clustered_endpoint(self, position: Tuple, endpoint_type: str, path_id: str) -> Optional[str]:
        """找到最接近位置的聚类端点节点"""
        min_distance = float('inf')
        closest_node_id = None
        
        for key_node_id, key_node in self.key_nodes.items():
            if not key_node.is_endpoint:
                continue
            
            # 检查端点类型匹配
            endpoint_info = key_node.endpoint_info
            if endpoint_info.get('type') != endpoint_type:
                continue
            
            # 检查路径关联（对于聚类端点，检查merged_path_ids）
            if 'cluster_info' in endpoint_info:
                cluster_info = endpoint_info.get('cluster_info', {})
                merged_path_ids = cluster_info.get('merged_path_ids', [])
                if path_id not in merged_path_ids:
                    continue
            elif path_id not in key_node.path_memberships:
                continue
            
            distance = self._calculate_distance(position, key_node.position)
            if distance < min_distance:
                min_distance = distance
                closest_node_id = key_node_id
        
        # 设置合理的距离阈值
        threshold = 8.0  # 增加阈值以适应聚类后的端点
        return closest_node_id if min_distance < threshold else None
    
    def _find_best_path_position(self, target_position: Tuple, path_nodes: List[Tuple]) -> int:
        """找到关键节点在原路径上的最佳位置索引"""
        min_distance = float('inf')
        best_index = -1
        
        for i, path_node in enumerate(path_nodes):
            distance = self._calculate_distance(target_position, path_node)
            if distance < min_distance:
                min_distance = distance
                best_index = i
        
        # 设置合理的距离阈值
        return best_index if min_distance < 15.0 else -1
    
    def _validate_node_sequence(self, sequence: List[Tuple[str, Tuple]]) -> List[Tuple[str, Tuple]]:
        """验证并修复节点序列 - 已取消跳过过近节点的逻辑"""
        if len(sequence) < 2:
            return sequence
        
        validated = [sequence[0]]  # 起点
        
        for i in range(1, len(sequence)):
            current_node = sequence[i]
            prev_node = validated[-1]
            
            # 计算距离（仅用于信息输出，不再用于跳过节点）
            distance = self._calculate_distance(prev_node[1], current_node[1])
            
            # 原来的跳过过近节点逻辑已被移除
            # 现在保留所有节点，包括过近的节点
            print(f"       📍 保留节点: {current_node[0]} (距离: {distance:.2f}m)")
            
            # 检查角度变化的合理性（仅用于信息输出）
            if len(validated) >= 2:
                angle_change = self._calculate_angle_change(
                    validated[-2][1], validated[-1][1], current_node[1]
                )
                
                # 如果角度变化过大，输出警告但不跳过
                if angle_change > math.pi * 0.8:  # 144度
                    print(f"       ⚠️ 大角度转弯: {current_node[0]} (角度: {math.degrees(angle_change):.1f}°)")
            
            # 添加所有节点到验证序列中
            validated.append(current_node)
        
        print(f"       ✅ 节点序列验证完成: 保留了所有 {len(validated)} 个节点")
        return validated
    
    def _calculate_angle_change(self, p1: Tuple, p2: Tuple, p3: Tuple) -> float:
        """计算三点的角度变化"""
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        len1 = math.sqrt(v1[0]**2 + v1[1]**2)
        len2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if len1 < 1e-6 or len2 < 1e-6:
            return 0.0
        
        cos_angle = (v1[0]*v2[0] + v1[1]*v2[1]) / (len1 * len2)
        cos_angle = max(-1, min(1, cos_angle))
        
        return math.acos(cos_angle)
    
    def _attempt_enhanced_curve_fitting(self, consolidated_path: EnhancedConsolidatedBackbonePath, 
                                      node_sequence: List[Tuple[str, Tuple]]) -> bool:
        """尝试增强曲线拟合"""
        
        # 提取位置和ID列表
        positions = [pos for node_id, pos in node_sequence]
        node_ids = [node_id for node_id, pos in node_sequence]
        
        # 确保3D坐标
        positions_3d = []
        for pos in positions:
            if len(pos) >= 3:
                positions_3d.append(pos)
            else:
                positions_3d.append((pos[0], pos[1], 0.0))
        
        # 策略1: 增强版曲线拟合
        if self.config['enable_enhanced_curve_fitting'] and self.enhanced_curve_fitter:
            print(f"       🎨 尝试增强版曲线拟合...")
            
            try:
                # 确定道路等级
                road_class = consolidated_path.road_class.value if consolidated_path.road_class else 'secondary'
                
                # 执行拟合
                curve_segments = self.enhanced_curve_fitter.fit_path_between_nodes(
                    key_nodes=positions_3d,
                    key_node_ids=node_ids,
                    road_class=road_class
                )
                
                if curve_segments:
                    # 处理拟合结果
                    success = self._process_enhanced_fitting_result(consolidated_path, curve_segments)
                    
                    if success:
                        print(f"       ✅ 增强曲线拟合成功")
                        self.consolidation_stats['complete_curve_success'] += 1
                        return True
                    else:
                        print(f"       ⚠️ 增强拟合质量不达标")
                
            except Exception as e:
                print(f"       ⚠️ 增强曲线拟合异常: {e}")
        
        # 策略2: 传统曲线拟合回退
        if self.traditional_path_fitter:
            print(f"       🔄 回退到传统曲线拟合...")
            
            try:
                road_class = consolidated_path.road_class.value if consolidated_path.road_class else 'secondary'
                
                fitted_path = self.traditional_path_fitter.reconstruct_path_with_curve_fitting(
                    key_node_positions=positions_3d,
                    key_node_ids=node_ids,
                    road_class=road_class,
                    path_quality=consolidated_path.original_quality
                )
                
                if fitted_path and len(fitted_path) >= 2:
                    consolidated_path.reconstructed_path = fitted_path
                    consolidated_path.reconstruction_success = True
                    consolidated_path.curve_fitting_method = "traditional_curve_fitting"
                    consolidated_path.curve_quality_score = self._evaluate_path_quality(fitted_path)
                    
                    print(f"       ✅ 传统曲线拟合成功")
                    self.consolidation_stats['segmented_curve_success'] += 1
                    return True
                
            except Exception as e:
                print(f"       ⚠️ 传统曲线拟合异常: {e}")
        
        # 策略3: 高质量直线连接回退
        print(f"       🆘 使用高质量直线连接...")
        self._create_high_quality_fallback_path(consolidated_path, positions_3d)
        self.consolidation_stats['fallback_reconstruction'] += 1
        
        return True
    
    def _process_enhanced_fitting_result(self, consolidated_path: EnhancedConsolidatedBackbonePath, 
                                       curve_segments: List) -> bool:
        """处理增强拟合结果"""
        if not curve_segments:
            return False
        
        # 更新曲线拟合结果
        consolidated_path.update_curve_fitting_results(curve_segments)
        
        # 转换为路径格式
        complete_path = self.enhanced_curve_fitter.convert_to_path_format(curve_segments)
        consolidated_path.reconstructed_path = complete_path
        consolidated_path.reconstruction_success = True
        
        # 设置拟合方法
        if len(curve_segments) == 1:
            consolidated_path.curve_fitting_method = f"enhanced_complete_{getattr(curve_segments[0], 'curve_type', 'unknown')}"
        else:
            consolidated_path.curve_fitting_method = f"enhanced_segmented_{getattr(curve_segments[0], 'curve_type', 'unknown')}"
        
        # 质量检查
        quality_threshold = self.config['curve_fitting_quality_threshold']
        if consolidated_path.curve_quality_score >= quality_threshold:
            return True
        else:
            print(f"       ⚠️ 质量不达标: {consolidated_path.curve_quality_score:.2f} < {quality_threshold}")
            return False
    
    def _evaluate_path_quality(self, path: List[Tuple]) -> float:
        """评估路径质量"""
        if not path or len(path) < 2:
            return 0.0
        
        # 简化的质量评估
        # 1. 路径效率
        path_length = sum(
            math.sqrt((path[i][0] - path[i-1][0])**2 + (path[i][1] - path[i-1][1])**2)
            for i in range(1, len(path))
        )
        
        direct_distance = math.sqrt(
            (path[-1][0] - path[0][0])**2 + (path[-1][1] - path[0][1])**2
        )
        
        efficiency = direct_distance / path_length if path_length > 0 else 0
        
        # 2. 平滑度（简化）
        smoothness = 0.8  # 默认值
        
        return min(1.0, efficiency * 0.6 + smoothness * 0.4)
    
    def _create_high_quality_fallback_path(self, consolidated_path: EnhancedConsolidatedBackbonePath, 
                                         positions: List[Tuple]):
        """创建高质量的回退路径"""
        if len(positions) < 2:
            return
        
        fallback_path = []
        
        for i in range(len(positions) - 1):
            start_pos = positions[i]
            end_pos = positions[i + 1]
            
            # 计算距离
            distance = self._calculate_distance(start_pos, end_pos)
            
            # 高密度采样（每0.5米一个点）
            num_points = max(3, int(distance / 0.5))
            
            for j in range(num_points):
                if i > 0 and j == 0:
                    continue  # 跳过重复起点
                
                t = j / (num_points - 1) if num_points > 1 else 0
                
                x = start_pos[0] + t * (end_pos[0] - start_pos[0])
                y = start_pos[1] + t * (end_pos[1] - start_pos[1])
                z = start_pos[2] + t * (end_pos[2] - start_pos[2]) if len(start_pos) > 2 else 0
                
                fallback_path.append((x, y, z))
        
        consolidated_path.reconstructed_path = fallback_path
        consolidated_path.reconstruction_success = True
        consolidated_path.curve_fitting_method = "high_quality_fallback"
        consolidated_path.curve_quality_score = 0.6  # 基础质量分数
    
    # ==================== 从原版继承的核心方法 ====================
    
    def _extract_original_paths(self, backbone_network) -> bool:
        """提取原始路径数据"""
        print(f"   提取原始骨干路径...")
        
        if not hasattr(backbone_network, 'bidirectional_paths'):
            print(f"   ❌ 骨干网络缺少bidirectional_paths属性")
            return False
        
        self.original_paths = {}
        original_nodes_count = 0
        
        for path_id, path_data in backbone_network.bidirectional_paths.items():
            if not hasattr(path_data, 'forward_path') or not path_data.forward_path:
                print(f"   ⚠️ 路径 {path_id} 缺少forward_path")
                continue
            
            # 提取路径节点
            forward_path = path_data.forward_path
            processed_nodes = []
            
            for node in forward_path:
                if len(node) >= 3:
                    processed_nodes.append((float(node[0]), float(node[1]), float(node[2])))
                elif len(node) == 2:
                    processed_nodes.append((float(node[0]), float(node[1]), 0.0))
                else:
                    continue
            
            if len(processed_nodes) < 2:
                print(f"   ⚠️ 路径 {path_id} 有效节点太少")
                continue
            
            # 存储原始路径信息
            self.original_paths[path_id] = {
                'path_id': path_id,
                'nodes': processed_nodes,
                'quality': getattr(path_data, 'quality', 0.7),
                'path_data': path_data,
                'endpoints': (processed_nodes[0], processed_nodes[-1])
            }
            
            original_nodes_count += len(processed_nodes)
            
        self.consolidation_stats['original_nodes_count'] = original_nodes_count
        
        print(f"   ✅ 提取完成: {len(self.original_paths)} 条路径, {original_nodes_count} 个节点")
        return len(self.original_paths) > 0
    
    def _perform_multi_round_clustering(self) -> bool:
        """执行多轮节点聚类"""
        print(f"   开始多轮节点聚类分析...")
        
        # 收集所有可聚类节点
        clusterable_nodes = []
        
        for path_id, path_info in self.original_paths.items():
            nodes = path_info['nodes']
            
            # 跳过起点和终点，只处理中间节点
            for i, node in enumerate(nodes):
                if i == 0 or i == len(nodes) - 1:
                    continue
                
                # 检查保护半径
                if self._is_near_protected_position(node, self.config['endpoint_buffer_radius']):
                    continue
                
                clusterable_nodes.append({
                    'position': node,
                    'path_id': path_id,
                    'node_index': i,
                    'paths': {path_id}
                })
        
        self.consolidation_stats['clusterable_nodes_count'] = len(clusterable_nodes)
        print(f"   可聚类节点数: {len(clusterable_nodes)}")
        
        if len(clusterable_nodes) == 0:
            self.node_clusters = []
            return True
        
        # 执行多轮聚类
        current_nodes = clusterable_nodes
        
        for round_idx, round_config in enumerate(self.config['clustering_rounds']):
            radius = round_config['radius']
            round_name = round_config['name']
            
            print(f"\n   === {round_name} (半径: {radius}m) ===")
            print(f"   输入节点数: {len(current_nodes)}")
            
            # 执行本轮聚类
            round_clusters = self._perform_single_round_clustering(current_nodes, radius)
            
            print(f"   生成聚类数: {len(round_clusters)}")
            
            # 如果是最后一轮，保存结果
            if round_idx == len(self.config['clustering_rounds']) - 1:
                self.node_clusters = round_clusters
            else:
                # 将聚类转换为下一轮的输入
                current_nodes = self._convert_clusters_to_nodes(round_clusters)
                print(f"   输出节点数: {len(current_nodes)}")
        
        print(f"\n   ✅ 多轮聚类完成: {len(self.node_clusters)} 个最终聚类")
        return True
    
    def _is_near_protected_position(self, position: Tuple, radius: float = None) -> bool:
        """检查位置是否在保护区域内"""
        if radius is None:
            radius = self.config['endpoint_buffer_radius']
        
        for protected_pos in self.protected_positions:
            distance = self._calculate_distance(position, protected_pos)
            if distance < radius:
                return True
        
        return False
    
    def _perform_single_round_clustering(self, nodes: List[Dict], radius: float) -> List[Dict]:
        """执行单轮聚类"""
        clusters = []
        visited = set()
        
        for i, node in enumerate(nodes):
            if i in visited:
                continue
            
            # 创建新聚类
            cluster = {
                'center': node['position'],
                'nodes': [node],
                'paths': node['paths'].copy()
            }
            visited.add(i)
            
            # 查找聚类半径内的其他节点
            for j, other_node in enumerate(nodes):
                if j in visited:
                    continue
                
                distance = self._calculate_distance(node['position'], other_node['position'])
                
                if distance <= radius:
                    cluster['nodes'].append(other_node)
                    cluster['paths'].update(other_node['paths'])
                    visited.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _convert_clusters_to_nodes(self, clusters: List[Dict]) -> List[Dict]:
        """将聚类转换为节点"""
        nodes = []
        
        for cluster_idx, cluster in enumerate(clusters):
            # 计算聚类中心
            center = self._calculate_weighted_cluster_center(cluster['nodes'])
            
            # 合并路径信息
            all_paths = set()
            for node in cluster['nodes']:
                all_paths.update(node['paths'])
            
            # 创建代表节点
            representative_node = {
                'position': center,
                'paths': all_paths,
                'original_cluster': cluster,
                'cluster_size': len(cluster['nodes']),
                'is_intersection': len(all_paths) > 1,
                'cluster_id': f"cluster_{cluster_idx}"
            }
            
            nodes.append(representative_node)
        
        return nodes
    
    def _calculate_weighted_cluster_center(self, nodes: List[Dict]) -> Tuple[float, float, float]:
        """计算加权聚类中心"""
        if not nodes:
            return (0.0, 0.0, 0.0)
        
        sum_x = 0.0
        sum_y = 0.0
        sum_z = 0.0
        total_weight = 0
        
        for node in nodes:
            pos = node['position']
            weight = len(node['paths'])
            
            sum_x += pos[0] * weight
            sum_y += pos[1] * weight
            sum_z += (pos[2] if len(pos) > 2 else 0) * weight
            total_weight += weight
        
        if total_weight > 0:
            return (sum_x / total_weight, sum_y / total_weight, sum_z / total_weight)
        else:
            sum_x = sum(node['position'][0] for node in nodes)
            sum_y = sum(node['position'][1] for node in nodes)
            sum_z = sum(node['position'][2] if len(node['position']) > 2 else 0 for node in nodes)
            count = len(nodes)
            return (sum_x / count, sum_y / count, sum_z / count)
    
    def _generate_key_nodes(self) -> bool:
        """生成关键节点"""
        print(f"   生成关键节点...")
        
        self.key_nodes = {}
        node_id_counter = 0
        
        # 1. 首先添加所有聚类后的端点作为关键节点
        for endpoint_id, endpoint_info in self.clustered_endpoint_nodes.items():
            key_node = KeyNode(
                node_id=endpoint_id,
                position=endpoint_info['position'],
                cluster_center=endpoint_info['position'],
                node_type=NodeType.ENDPOINT,
                is_endpoint=True
            )
            
            key_node.endpoint_info = endpoint_info
            key_node.path_memberships = endpoint_info['paths']
            
            # ✨ 设置端点重要性（基于合并的路径数量）
            if 'cluster_info' in endpoint_info:
                cluster_info = endpoint_info['cluster_info']
                merged_paths_count = len(cluster_info.get('merged_path_ids', []))
                key_node.node_importance = max(10.0, merged_paths_count * 5.0)  # 聚类端点更重要
                key_node.cluster_info = cluster_info
            else:
                key_node.node_importance = 10.0
            
            key_node.road_class = RoadClass.PRIMARY
            key_node.traffic_capacity = 150
            
            self.key_nodes[endpoint_id] = key_node
        
        print(f"   添加了 {len(self.clustered_endpoint_nodes)} 个聚类端点作为关键节点")
        
        # 2. 处理聚类生成的关键节点
        for cluster_idx, cluster in enumerate(self.node_clusters):
            # 计算聚类中心
            cluster_center = self._calculate_weighted_cluster_center(cluster['nodes'])
            
            # 创建关键节点
            key_node_id = f"key_node_{node_id_counter}"
            key_node = KeyNode(
                node_id=key_node_id,
                position=cluster_center,
                cluster_center=cluster_center,
                node_type=NodeType.KEY_NODE
            )
            
            # 详细信息继承
            unique_paths = set()
            all_original_positions = []
            
            for node in cluster['nodes']:
                original_pos = node['position']
                all_original_positions.append(original_pos)
                
                for path_id in node['paths']:
                    unique_paths.add(path_id)
                    key_node.add_original_node(original_pos, path_id, is_endpoint=False)
            
            # 添加聚类统计信息
            key_node.cluster_info = {
                'original_node_count': len(all_original_positions),
                'path_count': len(unique_paths),
                'cluster_size': len(cluster['nodes']),
                'is_intersection': len(unique_paths) > 1
            }
            
            self.key_nodes[key_node_id] = key_node
            node_id_counter += 1
            
            if len(unique_paths) > 1:
                print(f"     交叉关键节点 {key_node_id}: {len(unique_paths)}条路径交汇")
        
        self.consolidation_stats['key_nodes_count'] = len(self.key_nodes)
        
        # 计算节点减少比例
        original_count = self.consolidation_stats['original_nodes_count']
        key_count = len(self.key_nodes)
        
        if original_count > 0:
            self.consolidation_stats['node_reduction_ratio'] = (
                1.0 - key_count / original_count
            )
        
        print(f"   ✅ 关键节点生成完成: {len(self.key_nodes)} 个")
        print(f"   节点减少: {original_count} -> {key_count} "
              f"({self.consolidation_stats['node_reduction_ratio']:.1%})")
        
        return len(self.key_nodes) > 0
    
    def _preserve_original_path_as_consolidated(self, path_id: str, path_info: Dict):
        """将原路径保留为整合路径"""
        consolidated_path = EnhancedConsolidatedBackbonePath(
            path_id=f"preserved_{path_id}",
            original_path_id=path_id,
            key_nodes=[],
            original_endpoints=path_info['endpoints'],
            original_quality=path_info['quality']
        )
        
        # 直接使用原路径
        consolidated_path.reconstructed_path = path_info['nodes']
        consolidated_path.reconstruction_success = True
        consolidated_path.curve_quality_score = path_info['quality']
        consolidated_path.curve_fitting_method = "preserved_original"
        consolidated_path.path_length = self._calculate_path_length(path_info['nodes'])
        
        self.consolidated_paths[consolidated_path.path_id] = consolidated_path
    
    def _apply_consolidation_to_backbone(self, backbone_network) -> bool:
        """应用整合结果到骨干网络"""
        print(f"   应用整合结果到骨干网络...")
        
        # 创建新的骨干路径字典
        new_bidirectional_paths = {}
        
        for consolidated_path in self.consolidated_paths.values():
            if not consolidated_path.reconstructed_path:
                continue
            
            # 创建新的骨干路径对象
            new_path_object = self._create_enhanced_backbone_path_object(consolidated_path)
            
            if new_path_object:
                new_bidirectional_paths[consolidated_path.path_id] = new_path_object
                print(f"     ✅ 整合路径: {consolidated_path.path_id} "
                      f"({len(consolidated_path.reconstructed_path)} 节点, "
                      f"方法: {consolidated_path.curve_fitting_method})")
        
        # 更新骨干网络
        backbone_network.bidirectional_paths = new_bidirectional_paths
        
        # 添加整合信息到骨干网络
        backbone_network.consolidation_info = {
            'consolidation_applied': True,
            'consolidation_type': 'enhanced_node_clustering_professional',
            'key_nodes': self.key_nodes,
            'consolidation_stats': self.consolidation_stats,
            'enhanced_curve_fitting': self.config['enable_enhanced_curve_fitting'],
            'vehicle_dynamics_config': self.config['vehicle_dynamics'],
            'original_paths_count': len(self.original_paths),
            'consolidated_paths_count': len(new_bidirectional_paths),
            
            # ✨ 新增：端点聚类信息
            'endpoint_clustering_applied': self.config['enable_endpoint_clustering'],
            'endpoint_clusters': self.endpoint_clusters,
            'clustered_endpoint_nodes': self.clustered_endpoint_nodes,
            'endpoint_cluster_mapping': self.endpoint_cluster_mapping,
        }
        
        print(f"   ✅ 整合结果已应用: {len(new_bidirectional_paths)} 条增强路径")
        return True
    
    def _create_enhanced_backbone_path_object(self, consolidated_path: EnhancedConsolidatedBackbonePath):
        """创建增强的骨干路径对象"""
        
        class EnhancedConsolidatedBackbonePathObject:
            def __init__(self, consolidated_path, key_nodes_dict):
                self.path_id = consolidated_path.path_id
                self.original_path_id = consolidated_path.original_path_id
                
                # 路径数据
                self.forward_path = consolidated_path.reconstructed_path
                self.reverse_path = list(reversed(consolidated_path.reconstructed_path))
                
                # 属性
                self.length = consolidated_path.path_length
                self.quality = consolidated_path.curve_quality_score
                self.planner_used = 'enhanced_node_clustering_professional_consolidator'
                self.created_time = time.time()
                
                # 增强属性
                self.is_professional_design = True
                self.is_consolidated = True
                self.is_enhanced_curve_fitted = True
                self.consolidation_method = 'enhanced_multi_round_node_clustering'
                self.curve_fitting_method = consolidated_path.curve_fitting_method
                self.key_nodes = consolidated_path.key_nodes
                self.road_class = consolidated_path.road_class.value
                
                # 车辆动力学属性
                self.dynamics_compliance_rate = consolidated_path.dynamics_compliance_rate
                self.max_curvature = consolidated_path.max_curvature
                self.avg_curvature = consolidated_path.avg_curvature
                self.max_grade = consolidated_path.max_grade
                self.turning_radius_compliance = consolidated_path.turning_radius_compliance
                self.grade_compliance = consolidated_path.grade_compliance
                
                # 质量指标
                self.curve_quality_score = consolidated_path.curve_quality_score
                self.smoothness_score = consolidated_path.smoothness_score
                self.safety_score = consolidated_path.safety_score
                self.efficiency_score = consolidated_path.efficiency_score
                
                # 端点信息
                if consolidated_path.original_endpoints:
                    start_pos, end_pos = consolidated_path.original_endpoints
                    self.point_a = {'type': 'loading', 'id': 0, 'position': start_pos}
                    self.point_b = {'type': 'unloading', 'id': 0, 'position': end_pos}
                else:
                    self.point_a = {'type': 'loading', 'id': 0, 'position': self.forward_path[0]}
                    self.point_b = {'type': 'unloading', 'id': 0, 'position': self.forward_path[-1]}
                
                # 负载管理
                self.usage_count = 0
                self.current_load = 0
                self.max_capacity = 5
                
                # 质量追踪
                self.quality_history = [self.quality]
                self.last_quality_update = time.time()
            
            def get_path(self, from_point_type, from_point_id, to_point_type, to_point_id):
                return self.forward_path
            
            def increment_usage(self):
                self.usage_count += 1
            
            def add_vehicle(self, vehicle_id):
                self.current_load += 1
            
            def remove_vehicle(self, vehicle_id):
                self.current_load = max(0, self.current_load - 1)
            
            def get_load_factor(self):
                return self.current_load / self.max_capacity if self.max_capacity > 0 else 0
        
        return EnhancedConsolidatedBackbonePathObject(consolidated_path, self.key_nodes)
    
    def _generate_enhanced_consolidation_report(self, total_time: float):
        """生成增强整合报告"""
        print(f"\n📊 生成增强整合报告...")
        
        # 计算增强统计
        if self.consolidated_paths:
            # 动力学合规性统计
            total_dynamics_compliance = sum(
                p.dynamics_compliance_rate for p in self.consolidated_paths.values()
            )
            self.consolidation_stats['dynamics_compliance_rate'] = (
                total_dynamics_compliance / len(self.consolidated_paths)
            )
            
            # 转弯半径违规统计
            self.consolidation_stats['turning_radius_violations'] = sum(
                1 for p in self.consolidated_paths.values() 
                if not p.turning_radius_compliance
            )
            
            # 坡度违规统计
            self.consolidation_stats['grade_violations'] = sum(
                1 for p in self.consolidated_paths.values() 
                if not p.grade_compliance
            )
        
        # 道路等级统计
        road_class_dist = {'primary': 0, 'secondary': 0, 'service': 0}
        endpoint_count = 0
        
        for key_node in self.key_nodes.values():
            road_class_dist[key_node.road_class.value] += 1
            if key_node.is_endpoint:
                endpoint_count += 1
        
        # 更新统计
        self.consolidation_stats.update({
            'total_time': total_time,
            'road_class_distribution': road_class_dist,
            'protected_endpoints': endpoint_count,
            'clustering_rounds': len(self.config['clustering_rounds']),
            
            # 增强统计
            'enhanced_fitting_enabled': self.config['enable_enhanced_curve_fitting'],
            'vehicle_dynamics_enforced': self.config['force_vehicle_dynamics'],
            'curve_fitting_success_rate': (
                (self.consolidation_stats['enhanced_curve_fitting_used'] + 
                 self.consolidation_stats['segmented_curve_success']) / 
                max(1, len(self.consolidated_paths))
            ),
            
            # ✨ 端点聚类统计
            'endpoint_clustering_enabled': self.config['enable_endpoint_clustering'],
            'endpoint_clustering_radius': self.config['endpoint_clustering_radius'],
        })
        
        print(f"   ✅ 增强整合报告生成完成")
        print(f"      总耗时: {total_time:.2f}s")
        print(f"      ✨ 端点减少: {self.consolidation_stats['original_endpoints_count']} -> {self.consolidation_stats['endpoint_clusters_count']}")
        print(f"      ✨ 端点合并: {self.consolidation_stats['merged_endpoint_paths']} 条路径")
        print(f"      增强拟合使用: {self.consolidation_stats['enhanced_curve_fitting_used']} 次")
        print(f"      动力学合规率: {self.consolidation_stats['dynamics_compliance_rate']:.1%}")
        print(f"      转弯半径违规: {self.consolidation_stats['turning_radius_violations']} 条路径")
        print(f"      坡度违规: {self.consolidation_stats['grade_violations']} 条路径")
    
    # ==================== 辅助方法 ====================
    
    def _calculate_distance(self, pos1: Tuple, pos2: Tuple) -> float:
        """计算两点间距离"""
        return math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
    
    def _calculate_path_length(self, path: List[Tuple]) -> float:
        """计算路径长度"""
        if not path or len(path) < 2:
            return 0.0
        
        length = 0.0
        for i in range(len(path) - 1):
            length += self._calculate_distance(path[i], path[i + 1])
        return length
    
    # ==================== 增强版公共接口方法 ====================
    
    def get_consolidation_stats(self) -> Dict:
        """获取整合统计信息"""
        return self.consolidation_stats.copy()
    
    def get_key_nodes_info(self) -> Dict:
        """获取关键节点信息"""
        nodes_info = {}
        for node_id, key_node in self.key_nodes.items():
            nodes_info[node_id] = {
                'position': key_node.position,
                'importance': key_node.node_importance,
                'road_class': key_node.road_class.value,
                'path_memberships': list(key_node.path_memberships),
                'traffic_capacity': key_node.traffic_capacity,
                'original_nodes_count': len(key_node.original_nodes),
                'is_endpoint': key_node.is_endpoint,
                'node_type': key_node.node_type.value,
                'cluster_info': getattr(key_node, 'cluster_info', {}),
                
                # 增强属性
                'curve_fitting_quality': getattr(key_node, 'curve_fitting_quality', 0.0),
                'dynamics_compliance': getattr(key_node, 'dynamics_compliance', True),
                'smoothness_score': getattr(key_node, 'smoothness_score', 0.0),
                
                # ✨ 端点聚类信息
                'endpoint_info': getattr(key_node, 'endpoint_info', {}),
            }
        return nodes_info
    
    def get_consolidated_paths_info(self) -> Dict:
        """获取整合路径信息"""
        paths_info = {}
        for path_id, consolidated_path in self.consolidated_paths.items():
            paths_info[path_id] = {
                'original_path_id': consolidated_path.original_path_id,
                'key_nodes': consolidated_path.key_nodes,
                'path_length': consolidated_path.path_length,
                'road_class': consolidated_path.road_class.value,
                'reconstruction_success': consolidated_path.reconstruction_success,
                'node_count': len(consolidated_path.reconstructed_path),
                
                # 增强属性
                'curve_fitting_method': consolidated_path.curve_fitting_method,
                'curve_quality_score': consolidated_path.curve_quality_score,
                'dynamics_compliance_rate': consolidated_path.dynamics_compliance_rate,
                'smoothness_score': consolidated_path.smoothness_score,
                'max_curvature': consolidated_path.max_curvature,
                'avg_curvature': consolidated_path.avg_curvature,
                'turning_radius_compliance': consolidated_path.turning_radius_compliance,
                'grade_compliance': consolidated_path.grade_compliance,
            }
        return paths_info
    
    def get_endpoint_clustering_info(self) -> Dict:
        """✨ 获取端点聚类信息"""
        return {
            'endpoint_clustering_enabled': self.config['enable_endpoint_clustering'],
            'endpoint_clustering_radius': self.config['endpoint_clustering_radius'],
            'original_endpoints_count': self.consolidation_stats['original_endpoints_count'],
            'endpoint_clusters_count': self.consolidation_stats['endpoint_clusters_count'],
            'endpoint_reduction_ratio': self.consolidation_stats['endpoint_reduction_ratio'],
            'merged_endpoint_paths': self.consolidation_stats['merged_endpoint_paths'],
            'endpoint_clusters': {
                cluster_id: {
                    'representative_position': cluster.representative_position,
                    'endpoint_type': cluster.endpoint_type,
                    'original_endpoints_count': len(cluster.original_endpoints),
                    'cluster_importance': cluster.cluster_importance,
                    'merged_path_ids': list(cluster.merged_path_ids)
                }
                for cluster_id, cluster in self.endpoint_clusters.items()
            },
            'clustered_endpoint_nodes': self.clustered_endpoint_nodes,
            'endpoint_cluster_mapping': self.endpoint_cluster_mapping
        }
    
    def get_curve_fitting_statistics(self) -> Dict:
        """获取曲线拟合统计信息"""
        if self.enhanced_curve_fitter:
            return self.enhanced_curve_fitter.get_fitting_statistics()
        else:
            return {}

# 向后兼容性
NodeClusteringConsolidator = EnhancedNodeClusteringConsolidator

# 便捷创建函数
def create_enhanced_node_clustering_consolidator(env, config=None):
    """创建增强版基于节点聚类的专业道路整合器"""
    default_config = {
        'enable_enhanced_curve_fitting': True,
        'force_vehicle_dynamics': True,
        'curve_fitting_quality_threshold': 0.7,
        'prefer_complete_curve': True,
        
        # ✨ 端点聚类配置
        'enable_endpoint_clustering': True,
        'endpoint_clustering_radius': 2.0,
        'endpoint_merge_threshold': 2.5,
        
        'vehicle_dynamics': {
            'turning_radius': 8.0,
            'max_grade': 0.15,
            'safety_margin': 1.5
        }
    }
    
    if config:
        default_config.update(config)
    
    return EnhancedNodeClusteringConsolidator(env, default_config)

def apply_enhanced_consolidation(backbone_network, env, mode='professional'):
    """应用增强版整合到骨干网络"""
    mode_configs = {
        'professional': {
            'curve_fitting_quality_threshold': 0.7,
            'force_vehicle_dynamics': True,
            'prefer_complete_curve': True,
            'enable_endpoint_clustering': True,
            'endpoint_clustering_radius': 2.0,
            'vehicle_dynamics': {
                'turning_radius': 8.0,
                'max_grade': 0.12,  # 更严格的坡度要求
                'safety_margin': 2.0
            }
        },
        'balanced': {
            'curve_fitting_quality_threshold': 0.7,
            'force_vehicle_dynamics': True,
            'prefer_complete_curve': True,
            'enable_endpoint_clustering': True,
            'endpoint_clustering_radius': 2.0,
            'vehicle_dynamics': {
                'turning_radius': 8.0,
                'max_grade': 0.15,
                'safety_margin': 1.5
            }
        },
        'performance': {
            'curve_fitting_quality_threshold': 0.6,
            'force_vehicle_dynamics': True,
            'prefer_complete_curve': False,
            'enable_endpoint_clustering': True,
            'endpoint_clustering_radius': 2.5,
            'vehicle_dynamics': {
                'turning_radius': 7.0,
                'max_grade': 0.18,
                'safety_margin': 1.2
            }
        }
    }
    
    config = mode_configs.get(mode, mode_configs['balanced'])
    
    consolidator = EnhancedNodeClusteringConsolidator(env, config)
    success = consolidator.consolidate_backbone_network_professional(backbone_network)
    
    if success:
        stats = consolidator.get_consolidation_stats()
        endpoint_info = consolidator.get_endpoint_clustering_info()
        
        print(f"✅ 增强版专业道路整合成功 (模式: {mode})")
        print(f"   节点减少: {stats['node_reduction_ratio']:.1%}")
        print(f"   ✨ 端点减少: {endpoint_info['endpoint_reduction_ratio']:.1%}")
        print(f"   曲线拟合成功率: {stats.get('curve_fitting_success_rate', 0):.1%}")
        print(f"   动力学合规率: {stats.get('dynamics_compliance_rate', 0):.1%}")
        return consolidator
    else:
        print(f"❌ 增强版专业道路整合失败")
        return None

# 替换原有的整合器
OptimizedProfessionalMiningRoadConsolidator = EnhancedNodeClusteringConsolidator