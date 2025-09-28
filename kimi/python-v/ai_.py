"""
Optimized Chinese Chess AI (ai_optimized.py)
Enhancements over ai_improved.py:
- Full Iterative Deepening (ID) at the root search.
- Aspiration Windows implementation for faster root search.
- Time management checks within the search.
- Killer Heuristic for better non-capture move ordering.
- Deeper Quiescence Search: includes checks/check evasion (though move generation logic remains simple).
"""

import random
import copy
import logging
import time
from collections import defaultdict

# --- Configuration ---
MAX_DEPTH = 6  # Maximum search depth for the engine (used for ID)
TIME_LIMIT = 5.0  # Time limit in seconds per move
ASPIRATION_WINDOW_DELTA = 50 # Size of the aspiration window around the previous best score

logging.basicConfig(
    filename='kimi_backend_optimized.log',
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)

def log(msg):
    logging.info(msg)

# ---- constants ----
ROWS = 10
COLS = 9

# Score for being in check / checkmate (simple)
CHECK_SCORE = 200
MATE_SCORE = 1000000

PIECE_VALUES = {
    '將': MATE_SCORE, '帥': MATE_SCORE,
    '車': 900, '俥': 900, '车': 900,
    '馬': 450, '傌': 450, '马': 450,
    '炮': 400, '砲': 400,
    '相': 200, '象': 200,
    '仕': 200, '士': 200,
    '兵': 100, '卒': 100
}

# Initial board setup
INITIAL_SETUP = [
    [{'type': '車', 'side': 'red'}, {'type': '馬', 'side': 'red'}, {'type': '相', 'side': 'red'}, {'type': '仕', 'side': 'red'}, {'type': '帥', 'side': 'red'}, {'type': '仕', 'side': 'red'}, {'type': '相', 'side': 'red'}, {'type': '馬', 'side': 'red'}, {'type': '車', 'side': 'red'}],
    [None]*9,
    [None, {'type': '炮', 'side': 'red'}, None, None, None, None, None, {'type': '炮', 'side': 'red'}, None],
    [{'type': '兵', 'side': 'red'}, None, {'type': '兵', 'side': 'red'}, None, {'type': '兵', 'side': 'red'}, None, {'type': '兵', 'side': 'red'}, None, {'type': '兵', 'side': 'red'}],
    [None]*9,
    [None]*9,
    [{'type': '卒', 'side': 'black'}, None, {'type': '卒', 'side': 'black'}, None, {'type': '卒', 'side': 'black'}, None, {'type': '卒', 'side': 'black'}, None, {'type': '卒', 'side': 'black'}],
    [None, {'type': '炮', 'side': 'black'}, None, None, None, None, None, {'type': '炮', 'side': 'black'}, None],
    [None]*9,
    [{'type': '車', 'side': 'black'}, {'type': '馬', 'side': 'black'}, {'type': '象', 'side': 'black'}, {'type': '士', 'side': 'black'}, {'type': '將', 'side': 'black'}, {'type': '士', 'side': 'black'}, {'type': '象', 'side': 'black'}, {'type': '馬', 'side': 'black'}, {'type': '車', 'side': 'black'}],
]

# Simple piece-square tables (PST) - slightly expanded
PST = defaultdict(lambda: [[0]*COLS for _ in range(ROWS)])
# Encourage soldiers to advance and central files
for y in range(ROWS):
    for x in range(COLS):
        # Soldiers advancement (more aggressive)
        PST[('兵','red')][y][x] = (y - 3) * 5  # red soldiers: bigger when y increases
        PST[('卒','black')][y][x] = (6 - y) * 5
        # Soldiers center control
        if 3 <= x <= 5:
             PST[('兵','red')][y][x] += 5
             PST[('卒','black')][y][x] += 5

# Encourage central files for Rooks
for y in range(ROWS):
    for x in range(COLS):
        val = 15 if 3 <= x <= 5 else 0
        PST[('車','black')][y][x] += val
        PST[('車','red')][y][x] += val
        
# Cannon positions (high on river bank)
for x in range(COLS):
    PST[('炮','black')][7][x] += 10 # black cannon on 8th rank
    PST[('炮','red')][2][x] += 10 # red cannon on 3rd rank


# ---- Zobrist hashing for transposition table ----
PIECES = list(set([k for k in PIECE_VALUES.keys()]))
SIDES = ['red','black']
random.seed(123456)  # deterministic for reproducibility
ZOBRIST = {}
for p in PIECES:
    for s in SIDES:
        for y in range(ROWS):
            for x in range(COLS):
                ZOBRIST[(p,s,y,x)] = random.getrandbits(64)
ZOBRIST_SIDE = random.getrandbits(64)

# Transposition table entry: (value, depth, flag, best_move)
TT = {}
HISTORY = defaultdict(int)
# 清空历史表（每新局调用一次，可选）
def clear_history():
    global HISTORY
    HISTORY = defaultdict(int)

# Killer Move heuristic: stores 2 non-capture moves that caused a cutoff at a specific depth
# Format: {depth: [move1, move2]}
KILLER_MOVES = defaultdict(lambda: [None, None])

# ---- small opening book (same format as original) ----
FIRST_MOVES = [
    {'from': {'y': 2, 'x': 1}, 'to': {'y': 2, 'x': 4}},
    {'from': {'y': 3, 'x': 6}, 'to': {'y': 4, 'x': 6}},
    {'from': {'y': 0, 'x': 7}, 'to': {'y': 2, 'x': 6}},
    {'from': {'y': 3, 'x': 2}, 'to': {'y': 4, 'x': 2}},
    {'from': {'y': 0, 'x': 1}, 'to': {'y': 2, 'x': 2}},
]


# 方向常量
D4   = ((1,0),(-1,0),(0,1),(0,-1))          # 车、炮直线
D4_O = ((1,1),(1,-1),(-1,1),(-1,-1))        # 士
D8   = D4 + D4_O                            # 将在九宫内
H8   = ((1,2),(2,1),(-1,2),(-2,1),
        (1,-2),(2,-1),(-1,-2),(-2,-1))      # 马
E4   = ((2,2),(2,-2),(-2,2),(-2,-2))       # 象/相

# 快速边界判断
def inside(x,y): return 0 <= x < 9 and 0 <= y < 10

# 快速"将帅对脸"检测（只判断同一列且中间无子）
def kings_facing(board, red_king_pos, black_king_pos):
    if red_king_pos['x'] != black_king_pos['x']: return False
    x = red_king_pos['x']
    y1,y2 = sorted((red_king_pos['y'], black_king_pos['y']))
    for y in range(y1+1, y2):
        if board[y][x] is not None: return False
    return True

# 快速“己方是否被将军”——只扫描对方子力
def in_check_fast(board, side, king_pos=None):
    if king_pos is None:
        king_pos = find_general_in_board_state(board, side)
        if king_pos is None: return False
    x,y = king_pos['x'], king_pos['y']
    opp = 'red' if side=='black' else 'black'
    # 1. 直线（车、炮、将）
    for dx,dy in D4:
        nx,ny = x+dx, y+dy
        step = 0
        while inside(nx,ny):
            p = board[ny][nx]
            if p is None: nx+=dx; ny+=dy; continue
            if p['side']==opp:
                t = p['type']
                if t in {'車','俥','车','帥','將'} and step==0: return True
                if t in {'炮','砲'} and step==1: return True
            break
        # 炮需要“隔一个”才能打，所以继续扫
        nx,ny = x+dx, y+dy
        step  = 0
        while inside(nx,ny):
            p = board[ny][nx]
            if p is None:
                nx+=dx; ny+=dy; continue
            if step==1 and p['side']==opp and p['type'] in {'炮','砲'}:
                return True
            step += 1
            if step>1: break
            nx+=dx; ny+=dy
    # 2. 马
    for dx,dy in H8:
        nx,ny = x+dx, y+dy
        if not inside(nx,ny): continue
        # 绊马腿
        mx,my = x+dx//2, y+dy//2
        if board[my][mx] is not None: continue
        p = board[ny][nx]
        if p and p['side']==opp and p['type'] in {'馬','傌','马'}:
            return True
    return False

def gen_chariot(board, x, y, side):
    """车/俥/车"""
    moves = []
    for dx,dy in D4:
        nx,ny = x+dx, y+dy
        while inside(nx,ny):
            t = board[ny][nx]
            if t is None:
                moves.append({'from':{'x':x,'y':y}, 'to':{'x':nx,'y':ny}})
            else:
                if t['side']!=side:
                    moves.append({'from':{'x':x,'y':y}, 'to':{'x':nx,'y':ny}})
                break
            nx+=dx; ny+=dy
    return moves

def gen_cannon(board, x, y, side):
    """炮/砲"""
    moves = []
    for dx,dy in D4:
        # 第一段：空位可平移
        nx,ny = x+dx, y+dy
        while inside(nx,ny) and board[ny][nx] is None:
            moves.append({'from':{'x':x,'y':y}, 'to':{'x':nx,'y':ny}})
            nx+=dx; ny+=dy
        # 跳过第一个障碍
        if not inside(nx,ny): continue
        nx+=dx; ny+=dy
        # 第二段：必须吃到对方棋子
        while inside(nx,ny):
            t = board[ny][nx]
            if t is not None:
                if t['side']!=side:
                    moves.append({'from':{'x':x,'y':y}, 'to':{'x':nx,'y':ny}})
                break
            nx+=dx; ny+=dy
    return moves

def gen_horse(board, x, y, side):
    """马/傌/马"""
    moves = []
    for dx,dy in H8:
        nx,ny = x+dx, y+dy
        if not inside(nx,ny): continue
        # 绊马腿
        mx,my = x+dx//2, y+dy//2
        if board[my][mx] is not None: continue
        t = board[ny][nx]
        if t is None or t['side']!=side:
            moves.append({'from':{'x':x,'y':y}, 'to':{'x':nx,'y':ny}})
    return moves

def gen_soldier(board, x, y, side):
    """兵/卒"""
    moves = []
    forward = 1 if side=='red' else -1
    # 向前一格
    ny = y+forward
    if inside(x,ny):
        t = board[ny][x]
        if t is None or t['side']!=side:
            moves.append({'from':{'x':x,'y':y}, 'to':{'x':x,'y':ny}})
    # 过河后可横移
    river_crossed = (side=='red' and y>=5) or (side=='black' and y<=4)
    if river_crossed:
        for dx in (-1,1):
            nx = x+dx
            if not inside(nx,y): continue
            t = board[y][nx]
            if t is None or t['side']!=side:
                moves.append({'from':{'x':x,'y':y}, 'to':{'x':nx,'y':y}})
    return moves

def gen_general(board, x, y, side):
    """帥/將"""
    palace = range(3,6)
    y_range = (0,1,2) if side=='red' else (7,8,9)
    moves = []
    for dx,dy in D4:
        nx,ny = x+dx, y+dy
        if nx not in palace or ny not in y_range: continue
        t = board[ny][nx]
        if t is None or t['side']!=side:
            moves.append({'from':{'x':x,'y':y}, 'to':{'x':nx,'y':ny}})
    # 将帅对脸飞将（直线吃对方将）
    opp = 'red' if side=='black' else 'black'
    k2 = find_general_in_board_state(board, opp)
    if k2 and k2['x']==x:
        y1,y2 = sorted((y, k2['y']))
        blocked = False
        for yy in range(y1+1, y2):
            if board[yy][x] is not None:
                blocked = True
                break
        if not blocked:
            moves.append({'from':{'x':x,'y':y}, 'to':{'x':k2['x'],'y':k2['y']}})
    return moves

def gen_advisor(board, x, y, side):
    """仕/士"""
    palace = range(3,6)
    y_range = (0,1,2) if side=='red' else (7,8,9)
    moves = []
    for dx,dy in D4_O:
        nx,ny = x+dx, y+dy
        if nx not in palace or ny not in y_range: continue
        t = board[ny][nx]
        if t is None or t['side']!=side:
            moves.append({'from':{'x':x,'y':y}, 'to':{'x':nx,'y':ny}})
    return moves

def gen_elephant(board, x, y, side):
    """相/象"""
    y_limit = 5 if side=='red' else 4
    moves = []
    for dx,dy in E4:
        nx,ny = x+dx, y+dy
        if ny>y_limit if side=='red' else ny<y_limit: continue
        if not inside(nx,ny): continue
        # 塞象眼
        mx,my = x+dx//2, y+dy//2
        if board[my][mx] is not None: continue
        t = board[ny][nx]
        if t is None or t['side']!=side:
            moves.append({'from':{'x':x,'y':y}, 'to':{'x':nx,'y':ny}})
    return moves

GEN_MAP = {
    '車': gen_chariot, '俥': gen_chariot, '车': gen_chariot,
    '炮': gen_cannon, '砲': gen_cannon,
    '馬': gen_horse, '傌': gen_horse, '马': gen_horse,
    '兵': gen_soldier, '卒': gen_soldier,
    '帥': gen_general, '將': gen_general,
    '仕': gen_advisor, '士': gen_advisor,
    '相': gen_elephant, '象': gen_elephant,
}

def generate_moves_fast(board_state, side, depth=0):
    """
    返回与原代码格式完全一致的走法列表，带 score 字段（仅 MVV/LVA + PST）
    深度参数 depth 仅用于 Killer 排序，可保留原逻辑。
    """
    pseudo = []          # 伪合法
    for y in range(10):
        for x in range(9):
            p = board_state[y][x]
            if p and p['side']==side:
                pseudo += GEN_MAP[p['type']](board_state, x, y, side)

    # 统一合法性过滤
    legal = []
    red_king   = find_general_in_board_state(board_state, 'red')
    black_king = find_general_in_board_state(board_state, 'black')
    for m in pseudo:
        # 快速复制 + 落子
        nb = [row[:] for row in board_state]
        fx,fy = m['from']['x'], m['from']['y']
        tx,ty = m['to']['x'], m['to']['y']
        nb[ty][tx] = nb[fy][fx]
        nb[fy][fx] = None
        # 飞将检测
        if kings_facing(nb, red_king, black_king):
            continue
        # 自杀将检测
        if in_check_fast(nb, side):
            continue
        # 附加评分（与原逻辑一致）
        captured = board_state[ty][tx]
        score = 0
        if captured:
            score += 10*PIECE_VALUES[captured['type']] - PIECE_VALUES[nb[ty][tx]['type']]
        score += PST[(nb[ty][tx]['type'], side)][ty][tx]
        m['score'] = score
        legal.append(m)

    # Killer 排序（与原代码一致）
    km = KILLER_MOVES[depth]
    for m in legal:
        if m==km[0]: m['score'] += 1000
        elif m==km[1]: m['score'] += 900
    legal.sort(key=lambda x: x['score'], reverse=True)
    return legal
# ---- helper utilities (unchanged utility functions) ----

def compute_zobrist(board_state, side_to_move):
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

def copy_board(board_state):
    return [row[:] for row in board_state]

# --- Move Legality Functions (same as original, omitted for brevity) ---
def in_board_coords(x, y):
    return 0 <= x < COLS and 0 <= y < ROWS
def can_move_chariot_on(board_state, from_x, from_y, to_x, to_y):
    # ... (same implementation)
    if from_x == to_x:
        min_y, max_y = min(from_y, to_y), max(from_y, to_y)
        for i in range(min_y + 1, max_y):
            if board_state[i][from_x] is not None: return False
        return True
    elif from_y == to_y:
        min_x, max_x = min(from_x, to_x), max(from_x, to_x)
        for i in range(min_x + 1, max_x):
            if board_state[from_y][i] is not None: return False
        return True
    return False
def can_move_cannon_on(board_state, from_x, from_y, to_x, to_y, is_capture):
    # ... (same implementation)
    obstacle_count = 0
    if from_x == to_x:
        min_y, max_y = min(from_y, to_y), max(from_y, to_y)
        for i in range(min_y + 1, max_y):
            if board_state[i][from_x] is not None: obstacle_count += 1
    elif from_y == to_y:
        min_x, max_x = min(from_x, to_x), max(from_x, to_x)
        for i in range(min_x + 1, max_x):
            if board_state[from_y][i] is not None: obstacle_count += 1
    else: return False
    return (is_capture and obstacle_count == 1) or (not is_capture and obstacle_count == 0)
def can_move_horse_on(board_state, from_x, from_y, to_x, to_y):
    # ... (same implementation)
    dx, dy = abs(to_x - from_x), abs(to_y - from_y)
    if not ((dx == 1 and dy == 2) or (dx == 2 and dy == 1)): return False
    if dx == 1: check_y = from_y + (1 if to_y > from_y else -1)
    else: check_x = from_x + (1 if to_x > from_x else -1)
    if (dx == 1 and board_state[check_y][from_x] is not None) or \
       (dy == 1 and board_state[from_y][check_x] is not None): return False
    return True
def can_move_soldier_on(board_state, from_x, from_y, to_x, to_y, side):
    # ... (same implementation)
    dx = abs(to_x - from_x)
    dy = to_y - from_y
    is_across_river = (side == 'red' and from_y >= 5) or (side == 'black' and from_y <= 4)
    if (side == 'red' and dy < 0) or (side == 'black' and dy > 0): return False
    if dx + abs(dy) != 1: return False
    if is_across_river: return True
    else: return dx == 0 and ((side == 'red' and dy == 1) or (side == 'black' and dy == -1))
def can_move_general_on(board_state, from_x, from_y, to_x, to_y, side):
    # ... (same implementation)
    dx, dy = abs(to_x - from_x), abs(to_y - from_y)
    if dx + dy != 1: return False
    if not (3 <= to_x <= 5): return False
    if side == 'red' and not (0 <= to_y <= 2): return False
    if side == 'black' and not (7 <= to_y <= 9): return False
    return True
def can_move_elephant_on(board_state, from_x, from_y, to_x, to_y, side):
    # ... (same implementation)
    dx, dy = abs(to_x - from_x), abs(to_y - from_y)
    if dx != 2 or dy != 2: return False
    if (side == 'red' and to_y > 4) or (side == 'black' and to_y < 5): return False
    mid_x, mid_y = (from_x + to_x) // 2, (from_y + to_y) // 2
    if board_state[mid_y][mid_x] is not None: return False
    return True
def can_move_advisor_on(board_state, from_x, from_y, to_x, to_y, side):
    # ... (same implementation)
    dx, dy = abs(to_x - from_x), abs(to_y - from_y)
    if dx != 1 or dy != 1: return False
    if not (3 <= to_x <= 5): return False
    if side == 'red' and not (0 <= to_y <= 2): return False
    if side == 'black' and not (7 <= to_y <= 9): return False
    return True
def can_move_on(board_state, from_pos, to_pos):
    fx, fy = from_pos['x'], from_pos['y']
    tx, ty = to_pos['x'], to_pos['y']
    if not in_board_coords(fx, fy) or not in_board_coords(tx, ty): return False
    piece_obj = board_state[fy][fx]
    if not piece_obj: return False
    name, side = piece_obj['type'], piece_obj['side']
    target = board_state[ty][tx]
    if target and target['side'] == side: return False
    if name in ['車', '俥', '车']: return can_move_chariot_on(board_state, fx, fy, tx, ty)
    elif name in ['炮', '砲']: return can_move_cannon_on(board_state, fx, fy, tx, ty, bool(target))
    elif name in ['馬', '傌', '马']: return can_move_horse_on(board_state, fx, fy, tx, ty)
    elif name in ['兵', '卒']: return can_move_soldier_on(board_state, fx, fy, tx, ty, side)
    elif name in ['帥', '將']: return can_move_general_on(board_state, fx, fy, tx, ty, side)
    elif name in ['相', '象']: return can_move_elephant_on(board_state, fx, fy, tx, ty, side)
    elif name in ['仕', '士']: return can_move_advisor_on(board_state, fx, fy, tx, ty, side)
    return False

# ---- board utilities (same as original) ----
def find_general_in_board_state(board_state, side):
    for y in range(ROWS):
        for x in range(COLS):
            piece = board_state[y][x]
            if piece and (piece['type'] in ['帥', '將']) and piece['side'] == side:
                return {'x': x, 'y': y}
    return None
def is_in_check_board(board_state, side):
    general = find_general_in_board_state(board_state, side)
    if not general: return False
    opponent = 'red' if side == 'black' else 'black'
    for y in range(ROWS):
        for x in range(COLS):
            piece = board_state[y][x]
            if piece and piece['side'] == opponent:
                if can_move_on(board_state, {'x': x, 'y': y}, general): return True
    return False
def is_king_facing_king_board(board_state):
    red_king, black_king = None, None
    for y in range(ROWS):
        for x in range(COLS):
            piece = board_state[y][x]
            if not piece: continue
            if piece['type'] == '帥': red_king = {'x': x, 'y': y}
            if piece['type'] == '將': black_king = {'x': x, 'y': y}
    if red_king and black_king and red_king['x'] == black_king['x']:
        min_y, max_y = min(red_king['y'], black_king['y']), max(red_king['y'], black_king['y'])
        for y in range(min_y + 1, max_y):
            if board_state[y][red_king['x']] is not None: return False
        return True
    return False

# ---- move generation with legality filtering and ordering ----

def get_piece_value(piece_type):
    return PIECE_VALUES.get(piece_type, 0)

def generate_moves(board_state, side, tt_best_move=None, depth=0):
    moves = generate_moves_fast(board_state, side, depth)
    if tt_best_move:
        # 提到最前面
        for i,m in enumerate(moves):
            if m['from']==tt_best_move['from'] and m['to']==tt_best_move['to']:
                moves.insert(0, moves.pop(i))
                break
    return moves

# ---- evaluation ----

def evaluate_board(board_state):
    score = 0

    # --- 子力、位置、基础分 ---
    for y in range(ROWS):
        for x in range(COLS):
            piece = board_state[y][x]
            if not piece:
                continue
            val = PIECE_VALUES.get(piece['type'], 0)
            pstv = PST[(piece['type'], piece['side'])][y][x]
            side_mult = 1 if piece['side'] == 'black' else -1

            # 兵卒过河翻倍
            if piece['type'] in ['兵', '卒']:
                if (piece['side'] == 'red' and y >= 5) or (piece['side'] == 'black' and y <= 4):
                    val *= 2

            score += (val + pstv) * side_mult

    # --- 士象完整性 ---
    red_advisors = sum(1 for y in range(ROWS) for x in range(COLS)
                       if (p := board_state[y][x]) and p['side'] == 'red' and p['type'] == '仕')
    red_elephants = sum(1 for y in range(ROWS) for x in range(COLS)
                        if (p := board_state[y][x]) and p['side'] == 'red' and p['type'] == '相')
    black_advisors = sum(1 for y in range(ROWS) for x in range(COLS)
                         if (p := board_state[y][x]) and p['side'] == 'black' and p['type'] == '士')
    black_elephants = sum(1 for y in range(ROWS) for x in range(COLS)
                          if (p := board_state[y][x]) and p['side'] == 'black' and p['type'] == '象')

    score += (2 - red_advisors) * -50
    score += (2 - red_elephants) * -30
    score += (2 - black_advisors) * 50
    score += (2 - black_elephants) * 30

    # --- 将帅安全 ---
    red_king = find_general_in_board_state(board_state, 'red')
    black_king = find_general_in_board_state(board_state, 'black')
    if red_king:
        score -= 20 * (4 - red_advisors - red_elephants)
    if black_king:
        score += 20 * (4 - black_advisors - black_elephants)

    # --- 威胁评估（简化版）---
    def count_attacker(board, x, y, side):
        return sum(1 for fy in range(ROWS) for fx in range(COLS)
                   if (p := board[fy][fx]) and p['side'] == side and can_move_on(board, {'x': fx, 'y': fy}, {'x': x, 'y': y}))

    for y in range(ROWS):
        for x in range(COLS):
            piece = board_state[y][x]
            if not piece:
                continue
            side = piece['side']
            attackers = count_attacker(board_state, x, y, 'red' if side == 'black' else 'black')
            defenders = count_attacker(board_state, x, y, side)
            val = PIECE_VALUES.get(piece['type'], 0)
            if attackers > defenders:
                penalty = (val // 2) * (attackers - defenders)
                score += penalty if side == 'black' else -penalty

    # --- 棋子协同（车炮同线、马后炮）---
    def synergy_bonus():
        bonus = 0
        # 车炮同线
        for y in range(ROWS):
            red_chariot = None
            red_cannon = None
            black_chariot = None
            black_cannon = None
            for x in range(COLS):
                p = board_state[y][x]
                if p and p['side'] == 'red':
                    if p['type'] in ['車', '俥', '车']:
                        red_chariot = x
                    elif p['type'] in ['炮', '砲']:
                        red_cannon = x
                if p and p['side'] == 'black':
                    if p['type'] in ['車', '俥', '车']:
                        black_chariot = x
                    elif p['type'] in ['炮', '砲']:
                        black_cannon = x
            if red_chariot is not None and red_cannon is not None:
                bonus -= 30
            if black_chariot is not None and black_cannon is not None:
                bonus += 30
        return bonus

    score += synergy_bonus()

    # --- 机动性 ---
    black_moves = len(generate_moves(board_state, 'black'))
    red_moves = len(generate_moves(board_state, 'red'))
    score += (black_moves - red_moves) * 5

    # --- 将军惩罚 ---
    if is_in_check_board(board_state, 'red'):
        score += 200
    if is_in_check_board(board_state, 'black'):
        score -= 200

    return score

# ---- quiescence search (captures/checks only) ----

def is_forcing_move(move, board_state):
    # Forcing: Capture or Check (simple)
    
    # 1. Capture?
    if board_state[move['to']['y']][move['to']['x']] is not None:
        return True
    
    # 2. Check? (More expensive, simplified here)
    side = board_state[move['from']['y']][move['from']['x']]['side']
    opponent = 'red' if side == 'black' else 'black'
    
    # Check if the move puts the opponent in check
    newb = copy_board(board_state)
    newb[move['to']['y']][move['to']['x']] = newb[move['from']['y']][move['from']['x']]
    newb[move['from']['y']][move['from']['x']] = None
    if is_in_check_board(newb, opponent):
        return True
        
    return False


def quiescence(board_state, alpha, beta, side_to_move):
    # Stand-pat evaluation
    stand_pat = evaluate_board(board_state)
    
    # Black is maximizing (standard Negamax sign is implicitly applied by score structure)
    if side_to_move == 'black':
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
    # Red is minimizing
    else:
        # Flip signs for red's perspective (Red minimizes the score, which is Black's score)
        # Check if Black's score (stand_pat) is so low that Red can accept it (beta)
        if stand_pat <= alpha:
            return alpha
        if beta > stand_pat:
            beta = stand_pat

    moves = generate_moves(board_state, side_to_move)
    
    # Filter for forcing moves: captures or checks
    forcing_moves = [m for m in moves if is_forcing_move(m, board_state)]
    
    # Re-sort forcing moves: captures (MVV/LVA) first
    forcing_moves.sort(key=lambda m: m['score'], reverse=True)

    for m in forcing_moves:
        newb = copy_board(board_state)
        newb[m['to']['y']][m['to']['x']] = newb[m['from']['y']][m['from']['x']]
        newb[m['from']['y']][m['from']['x']] = None
        
        # Recursive call: The score returned is always from Black's perspective
        val = quiescence(newb, alpha, beta, 'red' if side_to_move == 'black' else 'black')
        
        if side_to_move == 'black':
            if val >= beta:
                return beta
            if val > alpha:
                alpha = val
        else: # Red minimizing side
            if val <= alpha:
                return alpha
            if val < beta:
                beta = val
                
    return alpha if side_to_move == 'black' else beta

# ---- alpha-beta with transposition table & quiescence ----

def negamax(board_state, depth, alpha, beta, side_to_move, current_depth, start_time=None, time_limit=None):
    # Time check: Throw exception to break out of all recursion
    if start_time and time_limit and (time.time() - start_time) > time_limit:
        raise TimeoutError

    # Check for mate/stalemate (though checkmate detection is primarily by PIECE_VALUES)
    # If no legal moves, it's checkmate (if in check) or stalemate (not in check) - simple end game check
    if depth == 0:
        return quiescence(board_state, alpha, beta, side_to_move)

    zob = compute_zobrist(board_state, side_to_move)
    tt_entry = TT.get(zob)
    
    # TT Lookup (Read)
    if tt_entry and tt_entry['depth'] >= depth:
        flag = tt_entry['flag']
        val = tt_entry['value']
        if flag == 'EXACT':
            return val
        if flag == 'LOWER' and val > alpha:
            alpha = val
        if flag == 'UPPER' and val < beta:
            beta = val
        if alpha >= beta:
            return val
            
    # Move Generation and Ordering
    tt_best_move = tt_entry.get('best_move') if tt_entry else None
    # pass current_depth for Killer Move access
    moves = generate_moves(board_state, side_to_move, tt_best_move, current_depth)
    
    # TT Best Move Ordering (move to front)
    if tt_best_move:
        # move it to front if present
        for i,m in enumerate(moves):
            if m['from']==tt_best_move['from'] and m['to']==tt_best_move['to']:
                moves.insert(0, moves.pop(i))
                break

    best_val = -MATE_SCORE if side_to_move == 'black' else MATE_SCORE
    best_move = None
    
    # Check for No Legal Moves (Game Over)
    if not moves:
        # If no moves and in check, it's checkmate (very low score for red, high for black)
        if is_in_check_board(board_state, side_to_move):
            return -MATE_SCORE if side_to_move == 'black' else MATE_SCORE
        # Otherwise, stalemate/draw (score 0, though draw handling is complex)
        return 0

    original_alpha = alpha
    original_beta = beta

    for m in moves:
        newb = copy_board(board_state)
        newb[m['to']['y']][m['to']['x']] = newb[m['from']['y']][m['from']['x']]
        newb[m['from']['y']][m['from']['x']] = None
        
        try:
            # Recursive call. current_depth increases.
            val = negamax(newb, depth-1, alpha, beta, 'red' if side_to_move=='black' else 'black', current_depth + 1, start_time, time_limit)
        except TimeoutError:
            raise # Re-raise to immediately stop the entire ID loop

        # Update best value and move
        if side_to_move == 'black':
            if val > best_val:
                best_val = val
                best_move = m
            alpha = max(alpha, best_val)
        else:
            if val < best_val:
                best_val = val
                best_move = m
            beta = min(beta, best_val)
            
        # Beta Cutoff (Principal Variation Search/Pruning)
        if alpha >= beta:
            # Killer Move Heuristic (Store non-capture move that caused cutoff)
            if board_state[m['to']['y']][m['to']['x']] is None:
                km = KILLER_MOVES[current_depth]
                if m != km[0]:
                    km[1] = km[0]
                    km[0] = m
            break # Cutoff!

    # TT Store (Write)
    flag = 'EXACT'
    if side_to_move == 'black':
        if best_val <= original_alpha:
            flag = 'UPPER'
        elif best_val >= original_beta:
            flag = 'LOWER'
    else: # Red minimizing side (opposite logic for lower/upper bound)
        if best_val >= original_beta: # Red found a value that's too high for black to accept
            flag = 'LOWER' # Black's score is >= beta -> Lower Bound
        elif best_val <= original_alpha: # Red found a value that's too low for black to accept
            flag = 'UPPER' # Black's score is <= alpha -> Upper Bound
            
    TT[zob] = {'value': best_val, 'depth': depth, 'flag': flag, 'best_move': best_move}
    return best_val

# ---- root search helper (Iterative Deepening + Aspiration Windows) ----

def minimax_root(board_state, side, time_limit=TIME_LIMIT):
    # opening book (kept as is)
    if board_state == INITIAL_SETUP and side == 'red':
        return random.choice(FIRST_MOVES)

    moves = generate_moves(board_state, side)
    if not moves:
        return None

    start_time = time.time()
    best_move_so_far = random.choice(moves) # fallback move
    best_val_so_far = 0
    
    # --- Iterative Deepening Loop ---
    for depth in range(1, MAX_DEPTH + 1):
        log(f"Starting ID search for depth {depth}...")
        
        current_best_val = -float('inf') if side == 'black' else float('inf')
        current_best_moves = []
        
        # Aspiration Window: Use the best score from the previous iteration
        alpha = best_val_so_far - ASPIRATION_WINDOW_DELTA
        beta = best_val_so_far + ASPIRATION_WINDOW_DELTA
        
        # TT best move ordering: use the best move from the last completed depth
        tt_best_move = TT.get(compute_zobrist(board_state, side), {}).get('best_move')
        if tt_best_move:
             # move the best move from the previous depth to the front for this depth's search
             for i,m in enumerate(moves):
                 if m['from']==tt_best_move['from'] and m['to']==tt_best_move['to']:
                     moves.insert(0, moves.pop(i))
                     break
        
        # Main search loop (with Aspiration Window handling)
        while True:
            # Re-run negamax with the current window
            try:
                # Store original alpha/beta for window logic
                original_alpha = alpha
                original_beta = beta
                
                # Use current depth as current_depth (0) for killer moves
                val = negamax(board_state, depth, alpha, beta, side, 0, start_time, time_limit)
                
                # Check for failure: If the true score is outside the window, re-search with a wider window
                if side == 'black':
                    # Fail High (True score > beta)
                    if val >= original_beta:
                        log(f"Depth {depth}: Fail High. Re-search with alpha={val}")
                        alpha = val
                        beta = float('inf')
                        continue # Re-run search
                    # Fail Low (True score <= alpha)
                    elif val <= original_alpha:
                        log(f"Depth {depth}: Fail Low. Re-search with beta={val}")
                        alpha = -float('inf')
                        beta = val
                        continue # Re-run search
                else: # Red minimizing side (opposite logic)
                    if val <= original_alpha:
                        log(f"Depth {depth}: Fail High (for black). Re-search with alpha={val}")
                        alpha = -float('inf')
                        beta = val
                        continue # Re-run search
                    elif val >= original_beta:
                        log(f"Depth {depth}: Fail Low (for black). Re-search with beta={val}")
                        alpha = val
                        beta = float('inf')
                        continue # Re-run search
                        
                # Success: Found within the window (or full window search)
                break 

            except TimeoutError:
                log(f"Timeout at depth {depth}. Returning move from depth {depth-1}.")
                # If search timed out, exit the ID loop and return the best move from the last COMPLETED depth
                return best_move_so_far

        # Find the best move for the *current* successful search
        tt_entry = TT.get(compute_zobrist(board_state, side))
        if tt_entry and tt_entry['depth'] >= depth:
            best_move_found = tt_entry.get('best_move')
            
            if best_move_found:
                best_move_so_far = best_move_found
                best_val_so_far = val
                log(f"Depth {depth} completed. Best Score: {val}. Best Move: {best_move_so_far}")
                
        # Time check after completing a full depth search
        if time.time() - start_time > time_limit * 0.95: # 95% threshold
            log(f"Time limit reached after completing depth {depth}. Stopping ID.")
            break
            
    return best_move_so_far

# ---- game over simple check (unchanged) ----

def check_game_over(board_state):
    red_general = find_general_in_board_state(board_state, 'red')
    black_general = find_general_in_board_state(board_state, 'black')
    if not red_general:
        return {'game_over': True, 'message': '黑方胜利！'}
    if not black_general:
        return {'game_over': True, 'message': '红方胜利！'}
    return {'game_over': False, 'message': ''}