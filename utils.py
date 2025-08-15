"""
工具函数
"""
import torch
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
import json
import os

logger = logging.getLogger(__name__)

def set_random_seed(seed: int = 42):
    """设置随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

def save_config(config: Any, filepath: str):
    """保存配置到文件"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config.__dict__, f, indent=2, ensure_ascii=False)
        logger.info(f"Config saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save config: {e}")

def load_config(filepath: str) -> Dict[str, Any]:
    """从文件加载配置"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"Config loaded from {filepath}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}

def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """计算准确率"""
    if predictions.dim() == 2:
        predictions = torch.argmax(predictions, dim=1)
    if targets.dim() == 2:
        targets = torch.argmax(targets, dim=1)
    
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    return correct / total if total > 0 else 0.0

def normalize_probabilities(probs: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """归一化概率分布"""
    if np.sum(probs) == 0:
        return np.ones_like(probs) / len(probs)
    
    probs = probs + epsilon
    return probs / np.sum(probs)

def create_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                     epoch: int, loss: float, filepath: str):
    """创建检查点"""
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")

def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   filepath: str) -> Tuple[int, float]:
    """加载检查点"""
    try:
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        logger.info(f"Checkpoint loaded from {filepath}")
        return epoch, loss
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return 0, float('inf')

def count_parameters(model: torch.nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_summary(model: torch.nn.Module) -> str:
    """获取模型摘要"""
    summary = []
    summary.append(f"Model: {model.__class__.__name__}")
    summary.append(f"Total parameters: {count_parameters(model):,}")
    
    # 计算可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    summary.append(f"Trainable parameters: {trainable_params:,}")
    
    # 计算模型大小（MB）
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    summary.append(f"Model size: {size_mb:.2f} MB")
    
    return "\n".join(summary)

def validate_board_state(board: np.ndarray) -> bool:
    """验证棋盘状态是否有效"""
    try:
        if board is None:
            return False
        
        if board.ndim != 2:
            return False
        
        # 检查棋子值是否合理
        unique_values = np.unique(board)
        for val in unique_values:
            if val not in [-1, 0, 1]:  # 黑棋、空位、红棋
                return False
        
        return True
    except Exception:
        return False

def print_board(board: np.ndarray, symbols: Dict[int, str] = None):
    """打印棋盘"""
    if symbols is None:
        symbols = {-1: "●", 0: "·", 1: "○"}
    
    print("  " + " ".join([str(i) for i in range(board.shape[1])]))
    for i, row in enumerate(board):
        row_str = " ".join([symbols.get(val, "?") for val in row])
        print(f"{i} {row_str}")
    print()
