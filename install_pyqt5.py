#!/usr/bin/env python3
"""
简单的PyQt5安装脚本
解决网络代理问题的PyQt5安装
"""

import subprocess
import sys
import os

def install_package(package_name, use_mirror=True):
    """安装单个包"""
    print(f"正在安装 {package_name}...")
    
    cmd = [sys.executable, "-m", "pip", "install", package_name]
    
    if use_mirror:
        # 尝试多个镜像源
        mirrors = [
            "https://pypi.tuna.tsinghua.edu.cn/simple",
            "https://mirrors.aliyun.com/pypi/simple/",
            "https://pypi.douban.com/simple/",
        ]
        
        for mirror in mirrors:
            try:
                print(f"尝试使用镜像源: {mirror}")
                result = subprocess.run(
                    cmd + ["-i", mirror], 
                    capture_output=True, 
                    text=True, 
                    timeout=300
                )
                if result.returncode == 0:
                    print(f"✓ {package_name} 安装成功")
                    return True
                else:
                    print(f"镜像源 {mirror} 失败: {result.stderr}")
            except subprocess.TimeoutExpired:
                print(f"镜像源 {mirror} 超时")
            except Exception as e:
                print(f"镜像源 {mirror} 错误: {e}")
    
    # 尝试官方源
    try:
        print("尝试使用官方源...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✓ {package_name} 安装成功")
            return True
        else:
            print(f"官方源失败: {result.stderr}")
    except Exception as e:
        print(f"官方源错误: {e}")
    
    return False

def main():
    print("=" * 50)
    print("MatSci-ML Studio PyQt5 安装脚本")
    print("=" * 50)
    
    # 核心依赖包
    packages = [
        "PyQt5",
        "pandas", 
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "joblib",
        "scipy",
        "openpyxl"
    ]
    
    success_count = 0
    failed_packages = []
    
    for package in packages:
        if install_package(package):
            success_count += 1
        else:
            failed_packages.append(package)
            print(f"✗ {package} 安装失败")
    
    print("\n" + "=" * 50)
    print("安装结果:")
    print(f"成功安装: {success_count}/{len(packages)} 个包")
    
    if failed_packages:
        print(f"失败的包: {', '.join(failed_packages)}")
        print("\n手动安装建议:")
        for pkg in failed_packages:
            print(f"pip install {pkg}")
    else:
        print("所有包安装成功！")
    
    print("=" * 50)

if __name__ == "__main__":
    main() 