import gym
import numpy as np
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def make_env():
    """创建象棋环境"""
    try:
        env = gym.make('gym_xiangqi:xiangqi-v0')
        logger.info("Successfully created XiangQi environment")
        return env
    except Exception as e:
        logger.error(f"Failed to create XiangQi environment: {e}")
        # 尝试创建备用环境
        try:
            env = gym.make('gym_xiangqi:xiangqi-v1')
            logger.info("Successfully created XiangQi environment (v1)")
            return env
        except Exception as e2:
            logger.error(f"Failed to create backup environment: {e2}")
            raise RuntimeError("Could not create any XiangQi environment")

def state_to_tensor(obs):
    """将观察转换为模型输入张量"""
    try:
        if obs is None:
            raise ValueError("Observation is None")
        
        # 确保obs是numpy数组
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        
        # 检查形状
        if obs.ndim != 2:
            raise ValueError(f"Expected 2D observation, got {obs.ndim}D")
        
        # 获取棋盘尺寸
        height, width = obs.shape
        
        # 转换为int8类型
        board = obs.astype(np.int8)
        
        # 创建两个通道：红棋和黑棋
        red = (board > 0).astype(np.float32)
        black = (board < 0).astype(np.float32)
        
        # 堆叠通道
        tensor = np.stack([red, black], axis=0)  # shape (2, height, width)
        
        # 验证输出
        if tensor.shape != (2, height, width):
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
        
        logger.debug(f"Converted observation to tensor with shape: {tensor.shape}")
        return tensor
        
    except Exception as e:
        logger.error(f"Error converting state to tensor: {e}")
        # 返回零张量作为fallback
        height, width = obs.shape if obs is not None else (10, 9)
        return np.zeros((2, height, width), dtype=np.float32)

def get_board_info(obs):
    """获取棋盘信息"""
    try:
        if obs is None:
            return {"red_pieces": 0, "black_pieces": 0, "total_pieces": 0}
        
        board = np.array(obs)
        red_count = np.sum(board > 0)
        black_count = np.sum(board < 0)
        total_count = red_count + black_count
        
        return {
            "red_pieces": int(red_count),
            "black_pieces": int(black_count),
            "total_pieces": int(total_count),
            "board_shape": board.shape
        }
    except Exception as e:
        logger.error(f"Error getting board info: {e}")
        return {"red_pieces": 0, "black_pieces": 0, "total_pieces": 0}
