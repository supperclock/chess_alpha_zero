"""
测试脚本 - 验证拆分后的文件是否能正常工作
"""

import os
import sys

def test_imports():
    """测试所有模块是否能正常导入"""
    print("Testing imports...")
    
    try:
        import game_utils
        print("✓ game_utils imported successfully")
    except Exception as e:
        print(f"✗ Failed to import game_utils: {e}")
        return False
    
    try:
        import data_generator
        print("✓ data_generator imported successfully")
    except Exception as e:
        print(f"✗ Failed to import data_generator: {e}")
        return False
    
    try:
        import model_trainer
        print("✓ model_trainer imported successfully")
    except Exception as e:
        print(f"✗ Failed to import model_trainer: {e}")
        return False
    
    return True

def test_game_utils():
    """测试游戏工具函数"""
    print("\nTesting game_utils functions...")
    
    try:
        from game_utils import initial_board, print_board, generate_legal_moves, is_terminal
        
        # 测试初始化棋盘
        board = initial_board()
        print(f"✓ Board initialized: {type(board)}")
        
        # 测试生成合法走法
        moves = generate_legal_moves(board, 'red')
        print(f"✓ Generated {len(moves)} legal moves for red")
        
        # 测试终局检查
        terminal, outcome = is_terminal(board, 'red')
        print(f"✓ Terminal check: {terminal}, outcome: {outcome}")
        
        return True
    except Exception as e:
        print(f"✗ game_utils test failed: {e}")
        return False

def test_data_generator():
    """测试数据生成器"""
    print("\nTesting data_generator...")
    
    try:
        from data_generator import MCTS, self_play_game
        from game_utils import initial_board
        
        # 测试MCTS初始化
        mcts = MCTS(net=None, sims=10, device='cpu')
        print("✓ MCTS initialized successfully")
        
        # 测试自对弈（短游戏）
        board = initial_board()
        traj, outcome = self_play_game(mcts, max_moves=5, temp=1.0)
        print(f"✓ Self-play completed: {len(traj)} positions, outcome: {outcome}")
        
        return True
    except Exception as e:
        print(f"✗ data_generator test failed: {e}")
        return False

def test_model_trainer():
    """测试模型训练器"""
    print("\nTesting model_trainer...")
    
    try:
        from model_trainer import AlphaNet, load_dataset_into_memory
        import torch
        
        # 测试神经网络初始化
        net = AlphaNet()
        print("✓ AlphaNet initialized successfully")
        
        # 测试前向传播
        dummy_input = torch.randn(1, 15, 10, 9)
        with torch.no_grad():
            policy, value = net(dummy_input)
        print(f"✓ Forward pass successful: policy shape {policy.shape}, value shape {value.shape}")
        
        return True
    except Exception as e:
        print(f"✗ model_trainer test failed: {e}")
        return False

def test_command_line_help():
    """测试命令行帮助"""
    print("\nTesting command line help...")
    
    try:
        import subprocess
        
        # 测试数据生成器帮助
        result = subprocess.run([sys.executable, 'data_generator.py', '--help'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ data_generator.py --help works")
        else:
            print(f"✗ data_generator.py --help failed: {result.stderr}")
            
        # 测试模型训练器帮助
        result = subprocess.run([sys.executable, 'model_trainer.py', '--help'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ model_trainer.py --help works")
        else:
            print(f"✗ model_trainer.py --help failed: {result.stderr}")
            
        return True
    except Exception as e:
        print(f"✗ Command line test failed: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("Testing AlphaZero Xiangqi Code Separation")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_game_utils,
        test_data_generator,
        test_model_trainer,
        test_command_line_help,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 50)
    
    if passed == total:
        print("🎉 All tests passed! Code separation successful.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

