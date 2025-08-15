"""
项目配置文件
"""
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    """训练配置"""
    # 基本参数
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    max_steps_per_game: int = 1000
    
    # MCTS参数
    mcts_simulations: int = 100
    mcts_c_puct: float = 1.0
    
    # 模型参数
    board_height: int = 10
    board_width: int = 9
    conv_channels: int = 128
    hidden_size: int = 64
    dropout_rate: float = 0.1
    
    # 缓冲区参数
    replay_buffer_capacity: int = 10000
    
    # 保存参数
    save_interval: int = 10
    model_save_dir: str = "models"
    
    # 设备配置
    device: str = "auto"  # "auto", "cpu", "cuda"
    
    def __post_init__(self):
        """初始化后处理"""
        # 创建模型保存目录
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        # 自动选择设备
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class ModelConfig:
    """模型配置"""
    input_channels: int = 2
    conv_layers: int = 3
    conv_channels: int = 128
    policy_head_size: int = 64
    value_head_size: int = 64

@dataclass
class EnvironmentConfig:
    """环境配置"""
    env_name: str = "gym_xiangqi:xiangqi-v0"
    backup_env_name: str = "gym_xiangqi:xiangqi-v1"
    render_mode: Optional[str] = None

# 默认配置
DEFAULT_CONFIG = TrainingConfig()
