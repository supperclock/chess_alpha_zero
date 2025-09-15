#!/usr/bin/env python3
"""
中国象棋游戏启动脚本
"""

import subprocess
import sys
import os

def install_requirements():
    """安装依赖包"""
    print("正在安装依赖包...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("依赖包安装完成！")
    except subprocess.CalledProcessError as e:
        print(f"安装依赖包失败: {e}")
        return False
    return True

def start_server():
    """启动服务器"""
    print("正在启动中国象棋游戏服务器...")
    print("游戏将在 http://localhost:5000 运行")
    print("按 Ctrl+C 停止服务器")
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"启动服务器失败: {e}")

if __name__ == "__main__":
    if not os.path.exists("requirements.txt"):
        print("错误: 找不到 requirements.txt 文件")
        sys.exit(1)
    
    if not install_requirements():
        sys.exit(1)
    
    start_server()
