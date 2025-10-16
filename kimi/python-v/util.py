import logging
import random

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('log.log', mode='a', encoding='utf-8')
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
file_handler.setFormatter(formatter)

# 避免重复添加 handler
logger.addHandler(file_handler)

def log(msg):
    logger.info(msg)

# Constants from ai.py
ROWS = 10
COLS = 9
ZOBRIST = {}
ZOBRIST_SIDE = None
# Piece types
PIECES = ['車', '俥', '车', '馬', '傌', '马', '炮', '砲', '相', '象', '仕', '士', '帥', '將', '兵', '卒']
SIDES = ['red', 'black']

def init_zobrist():
    """Initialize Zobrist hashing table"""
    global ZOBRIST, ZOBRIST_SIDE
    random.seed(123456)
    for p in PIECES:
        for s in SIDES:
            for y in range(ROWS):
                for x in range(COLS):
                    ZOBRIST[(p, s, y, x)] = random.getrandbits(64)
    ZOBRIST_SIDE = random.getrandbits(64)

def compute_zobrist(board_state, side_to_move):
    """Compute Zobrist hash for a given board state"""
    h = 0
    for y in range(ROWS):
        for x in range(COLS):
            piece = board_state[y][x]
            if piece:
                key = (piece['type'], piece['side'], y, x)
                h ^= ZOBRIST[key]
    if side_to_move == 'black':
        h ^= ZOBRIST_SIDE
    return h


import re
def convert_move_string(s):
    """
    将"Move(from_x=1, from_y=2, to_x=4, to_y=2)"格式的字符串
    转换为{'from': {'x': 1, 'y': 2}, 'to': {'x': 4, 'y': 2}}格式的字典
    通用处理任意符合格式的输入字符串
    """    
    # 使用正则表达式提取字符串中的数值
    pattern = r"Move\(from_x=(\d+), from_y=(\d+), to_x=(\d+), to_y=(\d+)\)"
    match = re.match(pattern, s)
    
    if not match:        
        raise ValueError("输入字符串格式不正确，应为: Move(from_x=数字, from_y=数字, to_x=数字, to_y=数字)")
    
    # 提取匹配到的数值并转换为整数
    from_x, from_y, to_x, to_y = map(int, match.groups())
    
    # 构建并返回对应的字典
    return {
        'from': {'x': from_x, 'y': from_y},
        'to': {'x': to_x, 'y': to_y}
    }