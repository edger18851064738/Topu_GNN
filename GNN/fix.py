#!/usr/bin/env python3
"""
环境验证脚本
用于验证MAGEC Demo运行环境是否正确配置
"""

import sys
import subprocess
import pkg_resources
from packaging import version

def check_python_version():
    """检查Python版本"""
    current_version = sys.version_info
    required_version = (3, 12, 7)
    
    print(f"Python版本检查:")
    print(f"  当前版本: {current_version.major}.{current_version.minor}.{current_version.micro}")
    print(f"  要求版本: {required_version[0]}.{required_version[1]}.{required_version[2]}")
    
    if current_version >= required_version:
        print("  ✅ Python版本满足要求")
        return True
    else:
        print("  ❌ Python版本过低")
        return False

def check_package_version(package_name, required_version=None):
    """检查包版本"""
    try:
        installed_version = pkg_resources.get_distribution(package_name).version
        print(f"  {package_name}: {installed_version}", end="")
        
        if required_version:
            if version.parse(installed_version) >= version.parse(required_version):
                print(" ✅")
                return True
            else:
                print(f" ❌ (需要 >= {required_version})")
                return False
        else:
            print(" ✅")
            return True
            
    except pkg_resources.DistributionNotFound:
        print(f"  {package_name}: 未安装 ❌")
        return False

def check_torch_cuda():
    """检查PyTorch和CUDA"""
    try:
        import torch
        print(f"\nPyTorch检查:")
        print(f"  PyTorch版本: {torch.__version__}")
        print(f"  CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  CUDA版本: {torch.version.cuda}")
            device_count = torch.cuda.device_count()
            print(f"  GPU设备数量: {device_count}")
            
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                print(f"  GPU {i}: {device_name}")
                
                # 测试GPU内存
                try:
                    device = torch.device(f'cuda:{i}')
                    test_tensor = torch.randn(1000, 1000, device=device)
                    memory_allocated = torch.cuda.memory_allocated(i) / 1024**2  # MB
                    print(f"    内存测试: ✅ ({memory_allocated:.1f} MB 已分配)")
                    del test_tensor
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"    内存测试: ❌ ({str(e)})")
            return True
        else:
            print("  ❌ CUDA不可用")
            return False
            
    except ImportError:
        print("  ❌ PyTorch未安装")
        return False

def check_torch_geometric():
    """检查PyTorch Geometric"""
    try:
        import torch_geometric
        from torch_geometric.nn import GCNConv
        from torch_geometric.data import Data, Batch
        
        print(f"\nPyTorch Geometric检查:")
        print(f"  版本: {torch_geometric.__version__}")
        
        # 测试基本功能
        try:
            import torch
            # 创建测试数据
            x = torch.randn(4, 3)
            edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
            data = Data(x=x, edge_index=edge_index)
            
            # 测试GCN层
            conv = GCNConv(3, 16)
            out = conv(data.x, data.edge_index)
            
            print("  基本功能测试: ✅")
            return True
            
        except Exception as e:
            print(f"  基本功能测试: ❌ ({str(e)})")
            return False
            
    except ImportError:
        print("  ❌ PyTorch Geometric未安装")
        return False

def check_other_packages():
    """检查其他必要包"""
    print(f"\n其他依赖包检查:")
    
    packages = {
        'numpy': '1.26.0',
        'matplotlib': '3.7.0',
        'networkx': '3.0',
        'scipy': None,
        'tqdm': None
    }
    
    all_ok = True
    for package, min_version in packages.items():
        if not check_package_version(package, min_version):
            all_ok = False
    
    return all_ok

def run_functionality_test():
    """运行功能测试"""
    print(f"\n功能测试:")
    
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import GCNConv
        from torch_geometric.data import Data, Batch
        import numpy as np
        import matplotlib.pyplot as plt
        import networkx as nx
        
        # 测试1: 创建简单图
        print("  创建NetworkX图: ", end="")
        G = nx.erdos_renyi_graph(10, 0.3)
        print("✅")
        
        # 测试2: 转换为PyTorch Geometric格式
        print("  转换为PyG格式: ", end="")
        edge_index = torch.tensor(list(G.edges())).t().contiguous()
        x = torch.randn(10, 3)
        data = Data(x=x, edge_index=edge_index)
        print("✅")
        
        # 测试3: 创建GNN模型
        print("  创建GNN模型: ", end="")
        class TestGNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = GCNConv(3, 16)
                self.conv2 = GCNConv(16, 8)
                
            def forward(self, data):
                x, edge_index = data.x, data.edge_index
                x = F.relu(self.conv1(x, edge_index))
                x = self.conv2(x, edge_index)
                return x
        
        model = TestGNN()
        print("✅")
        
        # 测试4: 前向传播
        print("  模型前向传播: ", end="")
        with torch.no_grad():
            output = model(data)
        print("✅")
        
        # 测试5: GPU测试（如果可用）
        if torch.cuda.is_available():
            print("  GPU计算测试: ", end="")
            device = torch.device('cuda:0')
            model = model.to(device)
            data = data.to(device)
            with torch.no_grad():
                output = model(data)
            print("✅")
        
        # 测试6: 可视化测试
        print("  Matplotlib可视化测试: ", end="")
        plt.figure(figsize=(6, 4))
        plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
        plt.title("Test Plot")
        plt.close()
        print("✅")
        
        print("  🎉 所有功能测试通过!")
        return True
        
    except Exception as e:
        print(f"  ❌ 功能测试失败: {str(e)}")
        import traceback
        print(f"\n详细错误信息:")
        traceback.print_exc()
        return False

def print_installation_commands():
    """打印安装命令"""
    print(f"\n" + "="*60)
    print("如果环境验证失败，请按以下步骤安装:")
    print("="*60)
    
    print("\n1. 创建Conda环境:")
    print("conda create -n magec_demo python=3.12.7")
    print("conda activate magec_demo")
    
    print("\n2. 安装PyTorch:")
    print("pip install torch==2.7.0+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    print("\n3. 安装PyTorch Geometric:")
    print("pip install torch-geometric")
    
    print("\n4. 安装其他依赖:")
    print("pip install numpy==1.26.4 matplotlib>=3.7.0 networkx>=3.0 scipy tqdm")
    
    print("\n5. 验证安装:")
    print("python verify_environment.py")

def main():
    """主函数"""
    print("="*60)
    print("MAGEC Demo 环境验证")
    print("="*60)
    
    all_checks_passed = True
    
    # 检查Python版本
    if not check_python_version():
        all_checks_passed = False
    
    # 检查PyTorch和CUDA
    if not check_torch_cuda():
        all_checks_passed = False
    
    # 检查PyTorch Geometric
    if not check_torch_geometric():
        all_checks_passed = False
    
    # 检查其他包
    if not check_other_packages():
        all_checks_passed = False
    
    # 运行功能测试
    if all_checks_passed:
        if not run_functionality_test():
            all_checks_passed = False
    
    # 总结
    print(f"\n" + "="*60)
    if all_checks_passed:
        print("🎉 环境验证通过! 可以运行MAGEC Demo了!")
        print("\n运行命令:")
        print("python demo.py")
        print("\n或者带参数运行:")
        print("python demo.py --num_episodes 100 --num_agents 4")
    else:
        print("❌ 环境验证失败! 请检查上述问题。")
        print_installation_commands()
    
    print("="*60)

if __name__ == "__main__":
    main()