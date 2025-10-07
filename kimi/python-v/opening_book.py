# opening_book.py —— 修复版：保持与 ai.py 一致（红方在上，黑方在下）
from collections import namedtuple
import logging
import random
import sqlite3

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s')


class Move:
    __slots__ = ('fy', 'fx', 'ty', 'tx', 'score', 'captured')
    def __init__(self, fy, fx, ty, tx):
        self.fy, self.fx, self.ty, self.tx = fy, fx, ty, tx
        self.score = 0
        self.captured = None
    def to_dict(self):
        return {'from': {'y': self.fy, 'x': self.fx}, 'to': {'y': self.ty, 'x': self.tx}}
    def __eq__(self, other):
        if not isinstance(other, Move):
            if isinstance(other, dict) and 'from' in other and 'to' in other:
                return self.fy == other['from']['y'] and self.fx == other['from']['x'] and \
                       self.ty == other['to']['y'] and self.tx == other['to']['x']
            return NotImplemented
        return self.fy == other.fy and self.fx == other.fx and self.ty == other.ty and self.tx == other.tx
    def __hash__(self):
        return hash((self.fy, self.fx, self.ty, self.tx))

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
def find_from_position(current_board, side_to_move):
    conn = sqlite3.connect('chess_games.db')
    cursor = conn.cursor()
    init_zobrist()
    board_state = compute_zobrist(current_board, side_to_move)
    # cursor.execute('''
    #     SELECT best_move FROM positions
    #     WHERE zobrist = ? order by visits desc
    # ''', (str(board_state),))
    cursor.execute('''
        SELECT best_move FROM ttxq_positions
        WHERE zobrist = ? order by visits desc
    ''', (str(board_state),))
    rows = cursor.fetchone()
    if not rows:
        return None
    return rows[0]
    #随机选取rows中的一条记录
    # if rows:
    #     result = random.choice(rows)
    #     # 确保返回的是字符串而不是元组
    #     return result[0] if isinstance(result, tuple) else result
