#!/usr/bin/env python3
"""
快速启动脚本 - Enhanced GNN Vehicle Coordination System
修复所有已知问题并直接启动系统
"""

import sys
import subprocess
import pkg_resources
from pathlib import Path

def check_and_install_dependencies():
    """检查并安装依赖"""
    required_packages = [
        'fastapi>=0.100.0',
        'uvicorn[standard]>=0.20.0',
        'websockets>=11.0',
        'python-multipart>=0.0.5'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            pkg_name = package.split('>=')[0].split('[')[0]
            pkg_resources.get_distribution(pkg_name)
            print(f"✅ {pkg_name} is installed")
        except pkg_resources.DistributionNotFound:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"📦 Installing missing packages: {missing_packages}")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install'
            ] + missing_packages)
            print("✅ All dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            print("\n💡 Please manually install:")
            for pkg in missing_packages:
                print(f"   pip install {pkg}")
            return False
    
    return True

def create_minimal_files():
    """创建最小化的必要文件"""
    
    # 创建目录
    Path("static").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    
    # 检查主要Python文件
    required_files = ["road_network.py", "vehicle_manager.py", "simulation_server.py"]
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Please ensure all Python modules are in the current directory.")
        return False
    
    print("✅ All required files found")
    return True

def main():
    print("🚛 Enhanced GNN Vehicle Coordination System")
    print("🚀 Starting server directly...")
    print("📡 WebSocket: ws://localhost:8000/ws")
    print("🌐 Interface: http://localhost:8000")
    print("⌨️  Press Ctrl+C to stop")
    print("=" * 50)
    
    try:
        # 直接启动服务器
        import uvicorn
        uvicorn.run(
            "simulation_server:app",
            host="127.0.0.1",
            port=8000,
            reload=False,
            log_level="info"
        )
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\n💡 Please install required packages:")
        print("pip install fastapi uvicorn[standard] websockets python-multipart")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure all Python files are in the current directory:")
        print("   - road_network.py")
        print("   - vehicle_manager.py") 
        print("   - simulation_server.py")

if __name__ == "__main__":
    main()