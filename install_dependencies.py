#!/usr/bin/env python3
"""
安装 MatSci-ML Studio 所需的依赖包
Installation script for MatSci-ML Studio dependencies
"""

import subprocess
import sys
import importlib

def check_package(package_name):
    """检查包是否已安装"""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def install_package(package):
    """安装包"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """主安装函数"""
    print("🚀 MatSci-ML Studio 依赖安装程序")
    print("=" * 50)
    
    # 必需的依赖包
    required_packages = [
        # 基础科学计算
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        
        # 机器学习框架
        "scikit-learn>=1.0.0",
        "xgboost>=1.5.0",
        "lightgbm>=3.3.0",
        
        # GUI框架
        "PyQt5>=5.15.0",
        
        # 可视化
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        
        # 系统监控
        "psutil>=5.8.0",
        
        # 数据处理
        "openpyxl>=3.0.0",
        "xlrd>=2.0.0",
        
        # 其他工具
        "joblib>=1.1.0",
        "tqdm>=4.60.0"
    ]
    
    print(f"需要安装 {len(required_packages)} 个依赖包...")
    print()
    
    failed_packages = []
    
    for i, package in enumerate(required_packages, 1):
        package_name = package.split(">=")[0].split("==")[0]
        
        print(f"[{i}/{len(required_packages)}] 检查 {package_name}...")
        
        if check_package(package_name):
            print(f"  ✅ {package_name} 已安装")
        else:
            print(f"  📦 正在安装 {package}...")
            if install_package(package):
                print(f"  ✅ {package} 安装成功")
            else:
                print(f"  ❌ {package} 安装失败")
                failed_packages.append(package)
    
    print()
    print("=" * 50)
    
    if failed_packages:
        print("❌ 安装完成，但以下包安装失败：")
        for package in failed_packages:
            print(f"  - {package}")
        print()
        print("请手动安装失败的包：")
        for package in failed_packages:
            print(f"  pip install {package}")
    else:
        print("✅ 所有依赖包安装成功！")
        print()
        print("现在可以运行 MatSci-ML Studio：")
        print("  python main.py")
    
    print()
    print("如果遇到问题，请检查：")
    print("1. Python 版本 >= 3.8")
    print("2. pip 是否为最新版本：python -m pip install --upgrade pip")
    print("3. 网络连接是否正常")

if __name__ == "__main__":
    main() 