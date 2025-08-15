import random
import numpy as np
import logging
from collections import deque
from typing import Tuple, List, Optional

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.total_added = 0
        self.total_sampled = 0

    def add(self, state, prob, reward):
        """添加经验到缓冲区"""
        try:
            # 类型和形状验证
            if not isinstance(state, np.ndarray):
                raise ValueError(f"State must be numpy array, got {type(state)}")
            
            if not isinstance(prob, np.ndarray):
                raise ValueError(f"Probability must be numpy array, got {type(prob)}")
            
            if not isinstance(reward, (int, float, np.number)):
                raise ValueError(f"Reward must be numeric, got {type(reward)}")
            
            # 验证概率和为1
            if np.abs(np.sum(prob) - 1.0) > 1e-6:
                logger.warning(f"Probability sum is not 1.0: {np.sum(prob)}")
                # 归一化概率
                if np.sum(prob) > 0:
                    prob = prob / np.sum(prob)
                else:
                    prob = np.ones_like(prob) / len(prob)
            
            # 验证状态形状
            if state.ndim != 3 or state.shape[0] != 2:
                raise ValueError(f"State must have shape (2, H, W), got {state.shape}")
            
            # 添加到缓冲区
            self.buffer.append((state.copy(), prob.copy(), float(reward)))
            self.total_added += 1
            
            logger.debug(f"Added experience to buffer. Buffer size: {len(self.buffer)}")
            
        except Exception as e:
            logger.error(f"Error adding experience to buffer: {e}")
            raise

    def sample(self, batch_size: int) -> Optional[List[Tuple]]:
        """从缓冲区采样经验"""
        try:
            if batch_size <= 0:
                raise ValueError(f"Batch size must be positive, got {batch_size}")
            
            if len(self.buffer) == 0:
                logger.warning("Buffer is empty, cannot sample")
                return None
            
            # 确保batch_size不超过缓冲区大小
            actual_batch_size = min(batch_size, len(self.buffer))
            
            # 采样
            batch = random.sample(list(self.buffer), actual_batch_size)
            self.total_sampled += actual_batch_size
            
            logger.debug(f"Sampled {actual_batch_size} experiences from buffer")
            return batch
            
        except Exception as e:
            logger.error(f"Error sampling from buffer: {e}")
            return None

    def get_stats(self) -> dict:
        """获取缓冲区统计信息"""
        return {
            "buffer_size": len(self.buffer),
            "capacity": self.capacity,
            "utilization": len(self.buffer) / self.capacity,
            "total_added": self.total_added,
            "total_sampled": self.total_sampled
        }

    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
        self.total_added = 0
        self.total_sampled = 0
        logger.info("Buffer cleared")

    def is_full(self) -> bool:
        """检查缓冲区是否已满"""
        return len(self.buffer) >= self.capacity

    def get_random_sample(self, n: int = 1) -> Optional[List[Tuple]]:
        """获取随机样本（不删除）"""
        try:
            if n <= 0 or n > len(self.buffer):
                return None
            return random.sample(list(self.buffer), n)
        except Exception as e:
            logger.error(f"Error getting random sample: {e}")
            return None
