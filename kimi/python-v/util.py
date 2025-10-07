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