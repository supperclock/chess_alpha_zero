# opening_book.py —— 修复版：保持与 ai.py 一致（红方在上，黑方在下）
from collections import namedtuple
import logging
import random
import sqlite3
import re
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s')


class Move:
    __slots__ = ('fy', 'fx', 'ty', 'tx', 'score', 'captured')
    
    def __init__(self, fy, fx, ty, tx):
        self.fy, self.fx, self.ty, self.tx = fy, fx, ty, tx
        self.score = 0
        self.captured = None

    def to_dict(self):
        return {'from': {'y': self.fy, 'x': self.fx}, 'to': {'y': self.ty, 'x': self.tx}}
    
    @classmethod
    def from_str(cls, move_str: str):
        """
        [已修改] 类方法：从字符串构建 Move 对象。
        
        接受的字符串格式示例:
        - "Move(from_x=1, from_y=2, to_x=1, to_y=4)"
        - "Move(fx=1, fy=2, tx=1, ty=4)"
        """
        
        # 使用正则表达式查找所有 key=value 对
        # \s* 允许等号周围有空格
        pattern = re.compile(r"(\w+)\s*=\s*(\d+)")
        matches = pattern.findall(move_str)
        
        params = {key: int(value) for key, value in matches}
        
        try:
            # 检查是 'from_x' 格式还是 'fx' 格式
            if 'from_x' in params:
                fy = params['from_y']
                fx = params['from_x']
                ty = params['to_y']
                tx = params['to_x']
            elif 'fx' in params:
                fy = params['fy']
                fx = params['fx']
                ty = params['ty']
                tx = params['tx']
            else:
                raise KeyError("字符串中未找到 'fx'/'fy' 或 'from_x'/'from_y' 键。")
                
        except KeyError as e:
            # 捕获 params['...'] 失败
            raise ValueError(f"解析 Move 字符串 '{move_str}' 失败。缺少键: {e}")
        
        # 使用默认构造函数创建并返回实例
        # cls 指向的就是 Move 类本身
        return cls(fy=fy, fx=fx, ty=ty, tx=tx)

    def __eq__(self, other):
        if not isinstance(other, Move):
            if isinstance(other, dict) and 'from' in other and 'to' in other:
                return self.fy == other['from']['y'] and self.fx == other['from']['x'] and \
                       self.ty == other['to']['y'] and self.tx == other['to']['x']
            return NotImplemented
        return self.fy == other.fy and self.fx == other.fx and self.ty == other.ty and self.tx == other.tx

    def __hash__(self):
        return hash((self.fy, self.fx, self.ty, self.tx))

    def __repr__(self):
        """
        [建议添加] 返回一个清晰的字符串表示，方便调试。
        """
        return f"Move(fy={self.fy}, fx={self.fx}, ty={self.ty}, tx={self.tx})"

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
    cursor.execute('''
        SELECT best_move FROM positions
        WHERE zobrist = ? order by black_wins desc
    ''', (str(board_state),))
    # cursor.execute('''
    #     SELECT best_move FROM ttxq_positions
    #     WHERE zobrist = ? 
    #     union
    #     select best_move from positions
    #     where zobrist = ? 
    # ''', (str(board_state),str(board_state),))
    rows = cursor.fetchone()
    if not rows:
        return None
    return rows[0]
    #随机选取rows中的一条记录
    # if rows:
    #     result = random.choice(rows)
    #     # 确保返回的是字符串而不是元组
    #     return result[0] if isinstance(result, tuple) else result
