"""
Optimized Chinese Chess AI
Streamlined version with only essential functions and improved structure.
"""

import random
import copy
import logging
import time
from collections import defaultdict
from opening_book import get_opening_move, Move

# --- Configuration ---
MAX_DEPTH = 6
TIME_LIMIT = 5.0
INITIAL_WINDOW = 800  # 第一层使用更大的初始窗口
ASPIRATION_WINDOW_DELTA = 300  # 后续层级的初始窗口
MAX_RESEARCH_COUNT = 3  # 最大重新搜索次数
MIN_TIME_LEFT = 0.5  # 剩余最小时间阈值（秒）

logging.basicConfig(
    filename='kimi_backend_optimized.log',
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)

def log(msg):
    logging.info(msg)

# ---- Constants ----
ROWS = 10
COLS = 9
MATE_SCORE = 1000000
MAX_HISTORY_SCORE = 1000000 # For history heuristic

PIECE_VALUES = {
    '將': MATE_SCORE, '帥': MATE_SCORE,
    '車': 900, '俥': 900, '车': 900,
    '馬': 450, '傌': 450, '马': 450,
    '炮': 400, '砲': 400,
    '相': 200, '象': 200,
    '仕': 200, '士': 200,
    '兵': 100, '卒': 100
}

# Initial board setup (Simplified to match original list structure)
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

# Piece-square tables
PST = defaultdict(lambda: [[0]*COLS for _ in range(ROWS)])
# Soldiers advancement and center control
for y in range(ROWS):
    for x in range(COLS):
        # Red Soldier
        PST[('兵','red')][y][x] = (y - 3) * 5 
        if 3 <= x <= 5: PST[('兵','red')][y][x] += 5
        # Black Soldier
        PST[('卒','black')][y][x] = (6 - y) * 5
        if 3 <= x <= 5: PST[('卒','black')][y][x] += 5

# Central files for Rooks
for y in range(ROWS):
    for x in range(COLS):
        val = 15 if 3 <= x <= 5 else 0
        PST[('車','black')][y][x] += val
        PST[('車','red')][y][x] += val

# Cannon positions
for x in range(COLS):
    PST[('炮','black')][7][x] += 10
    PST[('炮','red')][2][x] += 10

# ---- Zobrist Hashing ----
PIECES = list(set(PIECE_VALUES.keys()))
SIDES = ['red','black']
random.seed(123456)
ZOBRIST = {}
for p in PIECES:
    for s in SIDES:
        for y in range(ROWS):
            for x in range(COLS):
                ZOBRIST[(p,s,y,x)] = random.getrandbits(64)
ZOBRIST_SIDE = random.getrandbits(64)

# Transposition table, Killer moves, and History Heuristic
TT = {}
KILLER_MOVES = defaultdict(lambda: [None, None])
HISTORY_TABLE = defaultdict(lambda: 0)

# Opening book
# Converted to Move class

# ---- Movement Direction Constants ----
D4 = ((1,0),(-1,0),(0,1),(0,-1))          # Straight lines
D4_O = ((1,1),(1,-1),(-1,1),(-1,-1))      # Diagonal
H8 = ((1,2),(2,1),(-1,2),(-2,1),          # Horse
      (1,-2),(2,-1),(-1,-2),(-2,-1))
E4 = ((2,2),(2,-2),(-2,2),(-2,-1))        # Elephant (Fixed typo in original E4 definition)

def inside(x, y):
    return 0 <= x < 9 and 0 <= y < 10

def find_general(board_state, side):
    """Find the general/king position"""
    for y in range(ROWS):
        for x in range(COLS):
            piece = board_state[y][x]
            if piece and piece['type'] in ['帥', '將'] and piece['side'] == side:
                return {'x': x, 'y': y}
    return None

def kings_facing(board, red_king, black_king):
    """Check if kings are facing each other"""
    if red_king['x'] != black_king['x']:
        return False
    x = red_king['x']
    y1, y2 = sorted((red_king['y'], black_king['y']))
    for y in range(y1+1, y2):
        if board[y][x] is not None:
            return False
    return True

def in_check(board, side, king_pos=None):
    """Check if side is in check (Simplified and optimized logic)"""
    if king_pos is None:
        king_pos = find_general(board, side)
        if king_pos is None: return False
    x, y = king_pos['x'], king_pos['y']
    opp = 'red' if side == 'black' else 'black'
    
    # 1. Chariot/General/Cannon/Soldier attacks
    for dx, dy in D4:
        # Straight-line attack (Chariot, General)
        steps = 0
        for i in range(1, 10):
            nx, ny = x + dx * i, y + dy * i
            if not inside(nx, ny): break
            p = board[ny][nx]
            
            if p is not None:
                if p['side'] == opp:
                    t = p['type']
                    # Chariot (must be the first piece)
                    if t in {'車','俥','车'} and steps == 0: return True
                    # Cannon (must jump once)
                    if t in {'炮','砲'} and steps == 1: return True
                    # General (must be the first piece, handles flying general check too)
                    if t in {'帥','將'} and steps == 0: return True
                    # Soldier (handles adjacent check)
                    if t in {'兵','卒'} and steps == 0:
                        is_forward = (t == '兵' and dy == 1 and dx == 0) or \
                                     (t == '卒' and dy == -1 and dx == 0)
                        is_sideways = (y >= 5 if side == 'red' else y <= 4) and dy == 0 and abs(dx) == 1
                        if is_forward or is_sideways: return True
                steps += 1
                if steps > 1: break # Stop searching after the second piece is found
    
    # 2. Horse attack
    for dx, dy in H8:
        nx, ny = x + dx, y + dy
        if not inside(nx, ny): continue
        # Check horse leg blocking
        mx, my = x + dx - (dx//abs(dx)) if dx else x, y + dy - (dy//abs(dy)) if dy else y
        if board[my][mx] is not None: continue
        p = board[ny][nx]
        if p and p['side'] == opp and p['type'] in {'馬','傌','马'}: return True
        
    return False

# ---- Move Generation Functions ----
def gen_chariot(board, x, y, side):
    """Generate chariot moves"""
    moves = []
    for dx, dy in D4:
        nx, ny = x + dx, y + dy
        while inside(nx, ny):
            t = board[ny][nx]
            if t is None:
                moves.append(Move(fy=y, fx=x, ty=ny, tx=nx))
            else:
                if t['side'] != side:
                    moves.append(Move(fy=y, fx=x, ty=ny, tx=nx))
                break
            nx += dx
            ny += dy
    return moves

def gen_cannon(board, x, y, side):
    """Generate cannon moves"""
    moves = []
    for dx, dy in D4:
        # Phase 1: Move through empty squares
        nx, ny = x + dx, y + dy
        while inside(nx, ny) and board[ny][nx] is None:
            moves.append(Move(fy=y, fx=x, ty=ny, tx=nx))
            nx += dx
            ny += dy
        # Phase 2: Jump over first obstacle to capture
        if not inside(nx, ny): continue
        
        # nx, ny is the first obstacle (cannon "eye")
        
        nx += dx
        ny += dy
        
        while inside(nx, ny):
            t = board[ny][nx]
            if t is not None:
                if t['side'] != side:
                    moves.append(Move(fy=y, fx=x, ty=ny, tx=nx))
                break # Stop after the second piece (the target)
            nx += dx
            ny += dy
    return moves

def gen_horse(board, x, y, side):
    """Generate horse moves"""
    moves = []
    # Simplified check for horse leg blocking
    BLOCKERS = {
        (1, 2): (0, 1), (-1, 2): (0, 1), 
        (1, -2): (0, -1), (-1, -2): (0, -1),
        (2, 1): (1, 0), (-2, 1): (-1, 0),
        (2, -1): (1, 0), (-2, -1): (-1, 0),
    }
    
    for dx, dy in H8:
        nx, ny = x + dx, y + dy
        if not inside(nx, ny): continue
        
        # Check horse leg blocking
        leg_dx, leg_dy = BLOCKERS.get((dx, dy), (dx//2, dy//2)) # Fallback just in case
        if board[y + leg_dy][x + leg_dx] is not None: continue
            
        t = board[ny][nx]
        if t is None or t['side'] != side:
            moves.append(Move(fy=y, fx=x, ty=ny, tx=nx))
    return moves

def gen_soldier(board, x, y, side):
    """Generate soldier moves"""
    moves = []
    forward = 1 if side == 'red' else -1
    
    # Move forward
    ny = y + forward
    if inside(x, ny):
        t = board[ny][x]
        if t is None or t['side'] != side:
            moves.append(Move(fy=y, fx=x, ty=ny, tx=x))
            
    # Move sideways after crossing river
    river_crossed = (side == 'red' and y >= 5) or (side == 'black' and y <= 4)
    if river_crossed:
        for dx in (-1, 1):
            nx = x + dx
            if not inside(nx, y): continue
            t = board[y][nx]
            if t is None or t['side'] != side:
                moves.append(Move(fy=y, fx=x, ty=y, tx=nx))
    return moves

def gen_general(board, x, y, side):
    """Generate general moves"""
    palace = range(3, 6)
    y_range = range(0, 3) if side == 'red' else range(7, 10)
    moves = []
    
    # Normal moves
    for dx, dy in D4:
        nx, ny = x + dx, y + dy
        if nx not in palace or ny not in y_range: continue
        t = board[ny][nx]
        if t is None or t['side'] != side:
            moves.append(Move(fy=y, fx=x, ty=ny, tx=nx))
            
    # Flying general (capture enemy general in same file) - Note: This capture move is usually illegal 
    # if it doesn't resolve an existing check, but we generate it and filter later.
    opp = 'red' if side == 'black' else 'black'
    k2 = find_general(board, opp)
    if k2 and k2['x'] == x:
        y1, y2 = sorted((y, k2['y']))
        blocked = False
        for yy in range(y1+1, y2):
            if board[yy][x] is not None:
                blocked = True
                break
        if not blocked:
            moves.append(Move(fy=y, fx=x, ty=k2['y'], tx=k2['x']))
    return moves

def gen_advisor(board, x, y, side):
    """Generate advisor moves"""
    palace = range(3, 6)
    y_range = range(0, 3) if side == 'red' else range(7, 10)
    moves = []
    for dx, dy in D4_O:
        nx, ny = x + dx, y + dy
        if nx not in palace or ny not in y_range: continue
        t = board[ny][nx]
        if t is None or t['side'] != side:
            moves.append(Move(fy=y, fx=x, ty=ny, tx=nx))
    return moves

def gen_elephant(board, x, y, side):
    """Generate elephant moves"""
    y_limit = 5 if side == 'red' else 4
    moves = []
    for dx, dy in E4:
        nx, ny = x + dx, y + dy
        
        # Check river crossing
        if (side == 'red' and ny >= y_limit) or (side == 'black' and ny <= y_limit):
            pass # Keep it simple, just check the range
        else:
            continue
            
        if not inside(nx, ny): continue
        
        # Check elephant eye blocking
        mx, my = x + dx//2, y + dy//2
        if board[my][mx] is not None: continue
        
        t = board[ny][nx]
        if t is None or t['side'] != side:
            moves.append(Move(fy=y, fx=x, ty=ny, tx=nx))
    return moves

# Move generation map
GEN_MAP = {
    '車': gen_chariot, '俥': gen_chariot, '车': gen_chariot,
    '炮': gen_cannon, '砲': gen_cannon,
    '馬': gen_horse, '傌': gen_horse, '马': gen_horse,
    '兵': gen_soldier, '卒': gen_soldier,
    '帥': gen_general, '將': gen_general,
    '仕': gen_advisor, '士': gen_advisor,
    '相': gen_elephant, '象': gen_elephant,
}

def make_move(board, move):
    """Applies a move and returns the captured piece for unmaking."""
    fy, fx = move.fy, move.fx
    ty, tx = move.ty, move.tx
    
    captured = board[ty][tx]
    move.captured = captured # Store captured piece in move object
    
    board[ty][tx] = board[fy][fx]
    board[fy][fx] = None
    return captured

def unmake_move(board, move, captured):
    """Undoes a move."""
    fy, fx = move.fy, move.fx
    ty, tx = move.ty, move.tx
    
    board[fy][fx] = board[ty][tx]
    board[ty][tx] = captured
    move.captured = None # Clear captured piece

def generate_moves(board_state, side, tt_best_move=None, depth=0):
    """Generate all legal moves for a side"""
    pseudo = []
    for y in range(10):
        for x in range(9):
            p = board_state[y][x]
            if p and p['side'] == side:
                pseudo += GEN_MAP[p['type']](board_state, x, y, side)

    # Filter and score legal moves
    legal = []
    
    for m in pseudo:
        # Make move on current board (fast application)
        captured = make_move(board_state, m)
        
        # Check legality
        red_king = find_general(board_state, 'red')
        black_king = find_general(board_state, 'black')
        
        is_legal = True
        if red_king and black_king and kings_facing(board_state, red_king, black_king):
            is_legal = False
        elif in_check(board_state, side):
            is_legal = False
        
        # Unmake move
        unmake_move(board_state, m, captured)
        
        if is_legal:
            # Score the move (MVV/LVA for captures + PST)
            piece = board_state[m.fy][m.fx]
            
            score = 0
            # 历史启发值基准分
            piece_type = piece['type']
            key = (piece_type, side, m.fy, m.fx, m.ty, m.tx)
            history_score = HISTORY_TABLE.get(key, 0)
            
            # MVV-LVA (Most Valuable Victim - Least Valuable Aggressor)
            if captured:
                victim_value = PIECE_VALUES[captured['type']]
                aggressor_value = PIECE_VALUES[piece_type]
                score = 1000000  # 基础分：确保吃子走法优先
                score += victim_value * 100 - aggressor_value  # MVV-LVA评分
                
                # 特殊情况：吃兵时考虑过河因素
                if captured['type'] in ('兵', '卒'):
                    if (captured['type'] == '兵' and m.ty >= 5) or \
                       (captured['type'] == '卒' and m.ty <= 4):
                        score += 50  # 优先吃过河兵
            else:
                # 非吃子走法评分
                # 1. 位置价值变化
                old_pst = PST[(piece_type, side)][m.fy][m.fx]
                new_pst = PST[(piece_type, side)][m.ty][m.tx]
                score += (new_pst - old_pst) * 10
                
                # 2. 历史启发
                score += history_score // 2  # 降低历史表权重，避免过度依赖
                
                # 3. 中心控制（针对马、炮、车）
                if piece_type in ('馬', '炮', '車'):
                    center_dist = abs(4 - m.tx) + abs(4 - m.ty)
                    score += (7 - center_dist) * 5
            
            # 杀手启发
            km = KILLER_MOVES[depth]
            if m == km[0]: score += 5000
            elif m == km[1]: score += 3000
            
            m.score = score
            legal.append(m)
    
    # Sort by score
    legal.sort(key=lambda x: x.score, reverse=True)
    
    # Move TT best move to front
    if tt_best_move:
        for i, m in enumerate(legal):
            if m == tt_best_move:
                legal.insert(0, legal.pop(i))
                break
    
    return legal

def copy_board(board_state):
    """Create a copy of the board (only used for root search/initial state)"""
    return [row[:] for row in board_state]

def compute_zobrist(board_state, side_to_move):
    """Compute Zobrist hash"""
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

# ========= 新版 evaluate_board =========
def evaluate_board(board_state):
    """
    静态评估函数（黑方视角，越大越优）
    1. 子力 + 位置
    2. 过河兵深度加权
    3. 被攻击/保护次数
    4. 基础协同
    5. 士象完整 & 将帅安全
    6. 将军持续威胁
    """
    from collections import defaultdict

    # --- 子力、位置、兵深度 ---
    score = 0
    red_mob = black_mob = 0

    # --- 威胁统计 ---
    attacked = defaultdict(int)   # (y,x) 被敌方攻击次数
    guarded  = defaultdict(int)   # (y,x) 被己方保护次数

    # 先扫描一遍，统计攻击/保护
    for y in range(ROWS):
        for x in range(COLS):
            p = board_state[y][x]
            if not p:
                continue
            side = p['side']
            # 用原有 gen_xxx 函数生成“攻击范围”
            att_list = GEN_MAP[p['type']](board_state, x, y, side)
            for m in att_list:
                tx, ty = m.tx, m.ty
                tgt = board_state[ty][tx]
                if tgt is None:               # 空格子 → 保护
                    guarded[(ty,tx)] += 1 if side == 'black' else -1
                elif tgt['side'] != side:     # 敌方子 → 攻击
                    attacked[(ty,tx)] += 1 if side == 'black' else -1

    # 正式扫描棋子
    for y in range(ROWS):
        for x in range(COLS):
            p = board_state[y][x]
            if not p:
                continue
            side = p['side']
            mult = 1 if side == 'black' else -1

            # 1. 子力
            val = PIECE_VALUES.get(p['type'], 0)

            # 2. 过河兵深度
            if p['type'] in ('兵', '卒'):
                if side == 'red' and y >= 5:
                    val += (y - 5) * 10          # 每前进一格 +10
                elif side == 'black' and y <= 4:
                    val += (4 - y) * 10

            # 3. 位置分
            pst_val = PST[(p['type'], side)][y][x]

            # 4. 威胁分
            threat = (attacked[(y,x)] - guarded[(y,x)]) * 20
            val += threat

            score += (val + pst_val) * mult

            # 5. 机动性（同原）
            if p['type'] not in ('帥', '將'):
                mob = len(GEN_MAP[p['type']](board_state, x, y, side))
                if side == 'red':
                    red_mob += mob * 2
                else:
                    black_mob += mob * 2

    # 机动性汇总
    score += black_mob - red_mob

    # 6. 士象完整
    red_advisor = sum(1 for y in range(ROWS) for x in range(COLS)
                      if (q := board_state[y][x]) and q['side'] == 'red' and q['type'] == '仕')
    red_elephant = sum(1 for y in range(ROWS) for x in range(COLS)
                       if (q := board_state[y][x]) and q['side'] == 'red' and q['type'] == '相')
    black_advisor = sum(1 for y in range(ROWS) for x in range(COLS)
                        if (q := board_state[y][x]) and q['side'] == 'black' and q['type'] == '士')
    black_elephant = sum(1 for y in range(ROWS) for x in range(COLS)
                         if (q := board_state[y][x]) and q['side'] == 'black' and q['type'] == '象')

    score -= (2 - red_advisor) * 50
    score -= (2 - red_elephant) * 30
    score += (2 - black_advisor) * 50
    score += (2 - black_elephant) * 30

    # 7. 将帅安全（被困）
    red_king = find_general(board_state, 'red')
    black_king = find_general(board_state, 'black')
    if red_king:
        kx, ky = red_king['x'], red_king['y']
        # 九宫格内被攻击次数
        palace_att = sum(attacked[(y, x)] for y in range(0, 3) for x in range(3, 6)
                         if 0 <= y < ROWS and 0 <= x < COLS)
        score += palace_att * 25      # 红方被攻击，黑方加分

    if black_king:
        kx, ky = black_king['x'], black_king['y']
        palace_att = sum(attacked[(y, x)] for y in range(7, 10) for x in range(3, 6)
                         if 0 <= y < ROWS and 0 <= x < COLS)
        score -= palace_att * 25      # 黑方被攻击，黑方减分

    # 8. 将军持续威胁（非简单将军惩罚）
    if in_check(board_state, 'red'):
        score += 200
    if in_check(board_state, 'black'):
        score -= 200

    return score

def is_forcing_move(move, board_state):
    """Check if move is forcing (capture or check)"""
    if board_state[move.ty][move.tx] is not None:
        return True
    
    # Fast check for check after move
    # Note: A full check requires applying/unapplying the move, which is done in quiescence
    # This is a good place for simple *pre-check* heuristics, but for full accuracy, we proceed to Q-search.
    return False

def quiescence(board_state, alpha, beta, side_to_move, color_multiplier):
    """Quiescence search for captures and checks (Black always maximizing from their perspective)"""
    stand_pat = evaluate_board(board_state) * color_multiplier # Current score from the perspective of the *current* player
    
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat
    
    moves = generate_moves(board_state, side_to_move)
    forcing_moves = [m for m in moves if is_forcing_move(m, board_state)]
    forcing_moves.sort(key=lambda m: m.score, reverse=True)
    
    for m in forcing_moves:
        captured = make_move(board_state, m)
        
        # Check legality after move (only for general facing and check)
        red_king = find_general(board_state, 'red')
        black_king = find_general(board_state, 'black')
        
        if red_king and black_king and kings_facing(board_state, red_king, black_king):
            unmake_move(board_state, m, captured)
            continue
        if in_check(board_state, side_to_move):
            unmake_move(board_state, m, captured)
            continue
            
        val = -quiescence(board_state, -beta, -alpha, 'red' if side_to_move == 'black' else 'black', -color_multiplier)
        
        unmake_move(board_state, m, captured)
        
        if val >= beta:
            return beta
        if val > alpha:
            alpha = val
                
    return alpha

def negamax(board_state, depth, alpha, beta, side_to_move, color_multiplier, current_depth, start_time=None, time_limit=None):
    """Negamax search with alpha-beta pruning"""
    if start_time and time_limit and (time.time() - start_time) > time_limit:
        raise TimeoutError
    
    if depth == 0:
        return quiescence(board_state, alpha, beta, side_to_move, color_multiplier)
    
    # Transposition table lookup
    zob = compute_zobrist(board_state, side_to_move)
    tt_entry = TT.get(zob)
    
    if tt_entry and tt_entry['depth'] >= depth:
        flag = tt_entry['flag']
        val = tt_entry['value']
        
        # Adjust value from TT to current player's perspective
        tt_val = val * color_multiplier
        
        if flag == 'EXACT': return tt_val
        if flag == 'LOWER' and tt_val > alpha: alpha = tt_val
        if flag == 'UPPER' and tt_val < beta: beta = tt_val
        if alpha >= beta: return tt_val
    
    # Move generation
    tt_best_move = tt_entry.get('best_move') if tt_entry else None
    moves = generate_moves(board_state, side_to_move, tt_best_move, current_depth)
    
    best_val = -MATE_SCORE - 1 # Sentinel value
    best_move = None
    
    if not moves:
        # Checkmate or Stalemate
        if in_check(board_state, side_to_move):
            return -MATE_SCORE + current_depth # Checkmate (current player loses)
        return 0 # Stalemate
    
    original_alpha = alpha
    
    for m in moves:
        # Apply move
        captured = make_move(board_state, m)
        
        try:
            val = -negamax(board_state, depth-1, -beta, -alpha, 
                           'red' if side_to_move == 'black' else 'black', 
                           -color_multiplier, current_depth + 1, start_time, time_limit)
        except TimeoutError:
            unmake_move(board_state, m, captured)
            raise
        
        # Unmake move
        unmake_move(board_state, m, captured)
        
        # Update best value and move
        if val > best_val:
            best_val = val
            best_move = m
            
        alpha = max(alpha, best_val)
            
        # Beta cutoff
        if alpha >= beta:
            # Store killer move (non-capture only)
            if captured is None:
                km = KILLER_MOVES[current_depth]
                if m != km[0]:
                    km[1] = km[0]
                    km[0] = m
                
                # Update history heuristic (for non-capture moves that caused cutoff)
                piece_type = board_state[m.fy][m.fx]['type']
                key = (piece_type, side_to_move, m.fy, m.fx, m.ty, m.tx)
                HISTORY_TABLE[key] += depth * depth 
                HISTORY_TABLE[key] = min(HISTORY_TABLE[key], MAX_HISTORY_SCORE)
            
            break
    
    # Store in transposition table (store from Black's perspective, always maximizing)
    tt_val_black_perspective = best_val * color_multiplier 
    
    flag = 'EXACT'
    if best_val <= original_alpha:
        flag = 'UPPER'
    elif best_val >= beta:
        flag = 'LOWER'
            
    TT[zob] = {'value': tt_val_black_perspective, 'depth': depth, 'flag': flag, 'best_move': best_move}
    return best_val

def minimax_root(board_state, side, time_limit=TIME_LIMIT):
    """Root search with iterative deepening"""
    log(f"[搜索] 开始搜索，执棋方: {side}")
    
    # Opening book
    log(f"[搜索] 查询开局库...")
    move = get_opening_move(board_state, side)
    if move:
        log(f"[搜索] 开局库命中，返回走法: {move}")
        return move
    else:
        log(f"[搜索] 开局库未命中，开始搜索引擎")
    
    moves = generate_moves(board_state, side)
    if not moves: return None
    
    start_time = time.time()
    best_move_so_far = moves[0]
    best_val_so_far = -MATE_SCORE - 1
    research_count = 0
    
    color_multiplier = 1 if side == 'black' else -1
    
    # Iterative deepening
    for depth in range(1, MAX_DEPTH + 1):
        # 时间管理：检查剩余时间
        time_spent = time.time() - start_time
        time_left = time_limit - time_spent
        if time_left < MIN_TIME_LEFT:
            log(f"剩余时间不足 ({time_left:.2f}s)，停止搜索")
            break
            
        log(f"Starting ID search for depth {depth}...")
        
        # 第一层使用更大的初始窗口
        if depth == 1:
            alpha = -INITIAL_WINDOW
            beta = INITIAL_WINDOW
        else:
            # 后续层级使用渐进窗口
            alpha = best_val_so_far - ASPIRATION_WINDOW_DELTA
            beta = best_val_so_far + ASPIRATION_WINDOW_DELTA
        
        # TT best move ordering is handled inside generate_moves
        
        # Main search loop with re-search for aspiration window failure
        while True:
            try:
                original_alpha = alpha
                original_beta = beta
                
                # Recalculate move list with updated scores/ordering for new search
                moves = generate_moves(board_state, side, tt_best_move=best_move_so_far, depth=0)
                
                # Initialize best move/value for this depth
                current_best_val = -MATE_SCORE - 1 
                current_best_move = None
                
                for m in moves:
                    captured = make_move(board_state, m)
                    
                    val = -negamax(board_state, depth-1, -beta, -alpha, 
                                   'red' if side == 'black' else 'black', 
                                   -color_multiplier, 1, start_time, time_limit * 0.95) # Allocate 95% of time
                    
                    unmake_move(board_state, m, captured)
                    
                    if val > current_best_val:
                        current_best_val = val
                        current_best_move = m
                    
                    alpha = max(alpha, current_best_val)
                    if alpha >= beta: break # Fail-soft cutoff
                
                final_val = current_best_val
                
                # Check for aspiration window failure and re-search
                if final_val >= original_beta:
                    research_count += 1
                    if research_count >= MAX_RESEARCH_COUNT:
                        log(f"Depth {depth}: 达到最大重搜次数，使用全窗口")
                        alpha = final_val
                        beta = MATE_SCORE
                    else:
                        window_size = ASPIRATION_WINDOW_DELTA * (2 ** research_count)
                        log(f"Depth {depth}: Fail High. Re-search with wider window {window_size}")
                        alpha = final_val
                        beta = min(final_val + window_size, MATE_SCORE)
                    continue
                elif final_val <= original_alpha:
                    research_count += 1
                    if research_count >= MAX_RESEARCH_COUNT:
                        log(f"Depth {depth}: 达到最大重搜次数，使用全窗口")
                        alpha = -MATE_SCORE
                        beta = final_val
                    else:
                        window_size = ASPIRATION_WINDOW_DELTA * (2 ** research_count)
                        log(f"Depth {depth}: Fail Low. Re-search with wider window {window_size}")
                        alpha = max(final_val - window_size, -MATE_SCORE)
                        beta = final_val
                    continue
                        
                break # Aspiration window successful
                
            except TimeoutError:
                log(f"Timeout at depth {depth}. Returning best move from last completed depth.")
                return best_move_so_far.to_dict()
        
        # Successful depth completion
        best_val_so_far = final_val
        best_move_so_far = current_best_move
        log(f"Depth {depth} completed. Best Score (current player): {best_val_so_far}. Best Move: {best_move_so_far.to_dict()}")
        
        # Time check
        if time.time() - start_time > time_limit * 0.95:
            log(f"Time limit reached after completing depth {depth}. Stopping ID.")
            break
            
    return best_move_so_far.to_dict()

def check_game_over(board_state):
    """Check if game is over"""
    red_general = find_general(board_state, 'red')
    black_general = find_general(board_state, 'black')
    if not red_general:
        log("[游戏] 游戏结束：黑方胜利！")
        from opening_book import print_opening_stats
        print_opening_stats()
        return {'game_over': True, 'message': '黑方胜利！'}
    if not black_general:
        log("[游戏] 游戏结束：红方胜利！")
        from opening_book import print_opening_stats
        print_opening_stats()
        return {'game_over': True, 'message': '红方胜利！'}
    return {'game_over': False, 'message': ''}
