# 象棋AlphaZero项目

这是一个基于AlphaZero算法的中国象棋AI训练项目，使用PyTorch和MCTS（蒙特卡洛树搜索）实现。

## 项目特点

- 🎯 基于AlphaZero算法的策略价值网络
- 🌳 高效的MCTS搜索算法
- 🔄 经验回放缓冲区
- 📊 完整的训练监控和日志
- 🛡️ 健壮的错误处理和验证
- ⚙️ 可配置的训练参数

## 项目结构

```
chess_alpha_zero/
├── config.py              # 配置文件
├── model.py               # 神经网络模型
├── mcts.py                # MCTS搜索算法
├── env_wrapper.py         # 环境包装器
├── replay_buffer.py       # 经验回放缓冲区
├── train.py               # 原始训练脚本
├── train_improved.py      # 改进的训练脚本
├── utils.py               # 工具函数
├── run.py                 # 主运行脚本
├── requirements.txt       # 依赖包
└── README.md             # 项目说明
```

## 主要改进

### 1. 错误处理和健壮性
- 添加了全面的异常处理
- 环境状态验证
- 输入数据验证
- 优雅的降级处理

### 2. 代码质量
- 添加了类型提示
- 完整的文档字符串
- 日志记录系统
- 代码模块化

### 3. 训练稳定性
- 梯度裁剪防止梯度爆炸
- 概率分布归一化
- 缓冲区溢出保护
- 检查点保存和恢复

### 4. 配置管理
- 集中化的配置管理
- 可调整的超参数
- 设备自动检测（CPU/GPU）

### 5. 监控和调试
- 训练进度跟踪
- 损失函数分解
- 缓冲区统计信息
- 模型参数统计

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 快速开始

```bash
# 使用改进的训练脚本
python train_improved.py

# 或使用原始训练脚本
python train.py
```

### 配置训练参数

编辑 `config.py` 文件来调整训练参数：

```python
@dataclass
class TrainingConfig:
    num_epochs: int = 100          # 训练轮数
    batch_size: int = 32           # 批次大小
    learning_rate: float = 1e-3    # 学习率
    mcts_simulations: int = 100    # MCTS模拟次数
    # ... 更多参数
```

### 自定义模型

在 `model.py` 中修改网络架构：

```python
class PolicyValueNet(nn.Module):
    def __init__(self, action_size, board_height=10, board_width=9):
        # 自定义网络层
        pass
```

## 训练过程

1. **环境初始化**: 创建象棋环境
2. **模型创建**: 初始化策略价值网络
3. **MCTS搜索**: 使用当前策略进行游戏树搜索
4. **经验收集**: 收集游戏状态、动作概率和奖励
5. **网络训练**: 使用收集的经验训练网络
6. **模型保存**: 定期保存检查点和最佳模型

## 输出文件

训练过程中会生成以下文件：

- `models/`: 模型保存目录
  - `best_model.pth`: 最佳模型
  - `final_model.pth`: 最终模型
  - `checkpoint_epoch_X.pth`: 检查点文件
- `training_*.log`: 训练日志
- `training_config.json`: 训练配置
- `training_history.json`: 训练历史

## 性能优化建议

1. **GPU加速**: 确保安装了CUDA版本的PyTorch
2. **内存管理**: 调整缓冲区大小和批次大小
3. **并行化**: 考虑使用多进程进行MCTS搜索
4. **模型压缩**: 使用知识蒸馏或模型剪枝

## 故障排除

### 常见问题

1. **环境创建失败**
   - 检查gym-xiangqi是否正确安装
   - 验证Python版本兼容性

2. **内存不足**
   - 减少批次大小
   - 减少MCTS模拟次数
   - 使用CPU训练

3. **训练不收敛**
   - 调整学习率
   - 增加训练轮数
   - 检查数据预处理

### 日志分析

查看训练日志文件来诊断问题：

```bash
tail -f training_*.log
```

## 贡献指南

欢迎提交Issue和Pull Request来改进项目！

## 许可证

本项目采用MIT许可证。

## 致谢

- 基于DeepMind的AlphaZero算法
- 使用gym-xiangqi象棋环境
- PyTorch深度学习框架
