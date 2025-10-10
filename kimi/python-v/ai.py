"""
Advanced Chinese Chess AI (完整文件)
包含：PVS + Extensions + SEE + 分阶段评估 + 残局特判
兼容原有 make_move/unmake_move/generate_moves/GEN_MAP 等接口
"""

import random
import time
from collections import defaultdict
from opening_book import find_from_position, Move
from util import *

# --- Configuration ---
MAX_DEPTH = 2
TIME_LIMIT = 100.0
MIN_TIME_LEFT = 0.5

# ---- Constants ----
ROWS = 10
COLS = 9
MATE_SCORE = 1000000
MAX_HISTORY_SCORE = 1000000

PIECE_VALUES = {
    '將': MATE_SCORE, '帥': MATE_SCORE,
    '車': 900, '俥': 900, '车': 900,
    '馬': 450, '傌': 450, '马': 450,
    '炮': 800, '砲': 800,
    '相': 200, '象': 200,
    '仕': 200, '士': 200,
    '兵': 100, '卒': 100
}

# Initial setup
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
for y in range(ROWS):
    for x in range(COLS):
        PST[('兵','red')][y][x] = (y - 3) * 5
        if 3 <= x <= 5: PST[('兵','red')][y][x] += 5
        PST[('卒','black')][y][x] = (6 - y) * 5
        if 3 <= x <= 5: PST[('卒','black')][y][x] += 5
        val = 15 if 3 <= x <= 5 else 0
        PST[('車','black')][y][x] += val
        PST[('車','red')][y][x] += val
for x in range(COLS):
    PST[('炮','black')][7][x] += 10
    PST[('炮','red')][2][x] += 10

# ---- Zobrist ----
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

# Tables

KILLER_MOVES = defaultdict(lambda: [None, None])
HISTORY_TABLE = defaultdict(lambda: 0)

# Directions
D4 = ((1,0),(-1,0),(0,1),(0,-1))
D4_O = ((1,1),(1,-1),(-1,1),(-1,-1))
H8 = ((1,2),(2,1),(-1,2),(-2,1),(1,-2),(2,-1),(-1,-2),(-2,-1))
E4 = ((2,2),(2,-2),(-2,2),(-2,-2))

def inside(x, y):
    return 0 <= x < COLS and 0 <= y < ROWS

def find_general(board_state, side):
    for y in range(ROWS):
        for x in range(COLS):
            piece = board_state[y][x]
            if piece and piece['type'] in ['帥', '將'] and piece['side'] == side:
                return {'x': x, 'y': y}
    return None

def kings_facing(board, red_king, black_king):
    if red_king['x'] != black_king['x']:
        return False
    x = red_king['x']
    y1, y2 = sorted((red_king['y'], black_king['y']))
    for y in range(y1+1, y2):
        if board[y][x] is not None:
            return False
    return True

def in_check(board, side, king_pos=None):
    if king_pos is None:
        king_pos = find_general(board, side)
        if king_pos is None: return False
    x, y = king_pos['x'], king_pos['y']
    opp = 'red' if side == 'black' else 'black'
    # straight lines (chariot, cannon, general, soldier)
    for dx, dy in D4:
        steps = 0
        for i in range(1, 10):
            nx, ny = x + dx * i, y + dy * i
            if not inside(nx, ny): break
            p = board[ny][nx]
            if p is not None:
                if p['side'] == opp:
                    t = p['type']
                    if t in {'車','俥','车'} and steps == 0: return True
                    if t in {'炮','砲'} and steps == 1: return True
                    if t in {'帥','將'} and steps == 0: return True
                    if t in {'兵','卒'} and steps == 0:
                        is_forward = (t == '兵' and dy == 1 and dx == 0) or (t == '卒' and dy == -1 and dx == 0)
                        is_sideways = (y >= 5 if side == 'red' else y <= 4) and dy == 0 and abs(dx) == 1
                        if is_forward or is_sideways: return True
                steps += 1
                if steps > 1: break
    # horse
    for dx, dy in H8:
        nx, ny = x + dx, y + dy
        if not inside(nx, ny): continue
        if abs(dx) == 2:
            leg_x, leg_y = x + (dx // 2), y
        else:
            leg_x, leg_y = x, y + (dy // 2)
        if not inside(leg_x, leg_y): continue
        if board[leg_y][leg_x] is not None: continue
        p = board[ny][nx]
        if p and p['side'] == opp and p['type'] in {'馬','傌','马'}:
            return True
    return False

# ---- Move generation functions ----
def gen_chariot(board, x, y, side):
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
    moves = []
    for dx, dy in D4:
        nx, ny = x + dx, y + dy
        while inside(nx, ny) and board[ny][nx] is None:
            moves.append(Move(fy=y, fx=x, ty=ny, tx=nx))
            nx += dx
            ny += dy
        if not inside(nx, ny): continue
        nx += dx
        ny += dy
        while inside(nx, ny):
            t = board[ny][nx]
            if t is not None:
                if t['side'] != side:
                    moves.append(Move(fy=y, fx=x, ty=ny, tx=nx))
                break
            nx += dx
            ny += dy
    return moves

def gen_horse(board, x, y, side):
    moves = []
    for dx, dy in H8:
        nx, ny = x + dx, y + dy
        if not inside(nx, ny): continue
        if abs(dx) == 2:
            leg_x, leg_y = x + (dx // 2), y
        else:
            leg_x, leg_y = x, y + (dy // 2)
        if not inside(leg_x, leg_y): continue
        if board[leg_y][leg_x] is not None: continue
        t = board[ny][nx]
        if t is None or t['side'] != side:
            moves.append(Move(fy=y, fx=x, ty=ny, tx=nx))
    return moves

def gen_soldier(board, x, y, side):
    moves = []
    forward = 1 if side == 'red' else -1
    ny = y + forward
    if inside(x, ny):
        t = board[ny][x]
        if t is None or t['side'] != side:
            moves.append(Move(fy=y, fx=x, ty=ny, tx=x))
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
    palace = range(3, 6)
    y_range = range(0, 3) if side == 'red' else range(7, 10)
    moves = []
    for dx, dy in D4:
        nx, ny = x + dx, y + dy
        if nx not in palace or ny not in y_range: continue
        t = board[ny][nx]
        if t is None or t['side'] != side:
            moves.append(Move(fy=y, fx=x, ty=ny, tx=nx))
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
    y_limit = 5 if side == 'red' else 4
    moves = []
    for dx, dy in E4:
        nx, ny = x + dx, y + dy
        if (side == 'red' and ny >= y_limit) or (side == 'black' and ny <= y_limit):
            pass
        else:
            continue
        if not inside(nx, ny): continue
        mx, my = x + dx//2, y + dy//2
        if board[my][mx] is not None: continue
        t = board[ny][nx]
        if t is None or t['side'] != side:
            moves.append(Move(fy=y, fx=x, ty=ny, tx=nx))
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

def make_move(board, move):
    fy, fx = move.fy, move.fx
    ty, tx = move.ty, move.tx
    captured = board[ty][tx]
    move.captured = captured
    board[ty][tx] = board[fy][fx]
    board[fy][fx] = None
    return captured

def unmake_move(board, move, captured):
    fy, fx = move.fy, move.fx
    ty, tx = move.ty, move.tx
    board[fy][fx] = board[ty][tx]
    board[ty][tx] = captured
    move.captured = None

def generate_moves(board_state, side, depth=0):
    pseudo = []
    for y in range(ROWS):
        for x in range(COLS):
            p = board_state[y][x]
            if p and p['side'] == side:
                pseudo += GEN_MAP[p['type']](board_state, x, y, side)

    if in_check(board_state, side):
        legal = []
        for m in pseudo:
            piece = board_state[m.fy][m.fx]
            if piece is None:
                continue
            captured = make_move(board_state, m)
            if not in_check(board_state, side):
                score = 0
                piece_type = piece['type']
                key = (piece_type, side, m.fy, m.fx, m.ty, m.tx)
                history_score = HISTORY_TABLE.get(key, 0)
                if captured:
                    victim_value = PIECE_VALUES[captured['type']]
                    aggressor_value = PIECE_VALUES[piece_type]
                    score = 1000000
                    score += victim_value * 100 - aggressor_value
                    if captured['type'] in ('兵', '卒'):
                        if (captured['type'] == '兵' and m.ty >= 5) or (captured['type'] == '卒' and m.ty <= 4):
                            score += 50
                else:
                    old_pst = PST[(piece_type, side)][m.fy][m.fx]
                    new_pst = PST[(piece_type, side)][m.ty][m.tx]
                    score += (new_pst - old_pst) * 10
                    score += history_score // 2
                    if piece_type in ('馬', '炮', '車'):
                        center_dist = abs(4 - m.tx) + abs(4 - m.ty)
                        score += (7 - center_dist) * 5
                km = KILLER_MOVES[depth]
                if m == km[0]: score += 5000
                elif m == km[1]: score += 3000
                m.score = score
                legal.append(m)
            unmake_move(board_state, m, captured)
        legal.sort(key=lambda x: x.score, reverse=True)       
        return legal

    legal = []
    for m in pseudo:
        piece = board_state[m.fy][m.fx]
        if piece is None:
            continue
        captured = make_move(board_state, m)
        red_king = find_general(board_state, 'red')
        black_king = find_general(board_state, 'black')
        is_legal = True
        if red_king and black_king and kings_facing(board_state, red_king, black_king):
            is_legal = False
        elif in_check(board_state, side):
            is_legal = False
        unmake_move(board_state, m, captured)
        if is_legal:
            score = 0
            piece_type = piece['type']
            key = (piece_type, side, m.fy, m.fx, m.ty, m.tx)
            history_score = HISTORY_TABLE.get(key, 0)
            if captured:
                victim_value = PIECE_VALUES[captured['type']]
                aggressor_value = PIECE_VALUES[piece_type]
                score = 1000000
                score += victim_value * 100 - aggressor_value
                if captured['type'] in ('兵', '卒'):
                    if (captured['type'] == '兵' and m.ty >= 5) or (captured['type'] == '卒' and m.ty <= 4):
                        score += 50
            else:
                old_pst = PST[(piece_type, side)][m.fy][m.fx]
                new_pst = PST[(piece_type, side)][m.ty][m.tx]
                score += (new_pst - old_pst) * 10
                score += history_score // 2
                if piece_type in ('馬', '炮', '車'):
                    center_dist = abs(4 - m.tx) + abs(4 - m.ty)
                    score += (7 - center_dist) * 5
            km = KILLER_MOVES[depth]
            if m == km[0]: score += 5000
            elif m == km[1]: score += 3000
            m.score = score
            legal.append(m)
    legal.sort(key=lambda x: x.score, reverse=True)    
    return legal

def copy_board(board_state):
    return [row[:] for row in board_state]

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

# ---------------- Static Exchange Evaluation (SEE) ----------------
def static_exchange_evaluation(board_state, tx, ty, side):
    attackers = []
    defenders = []
    for y in range(ROWS):
        for x in range(COLS):
            p = board_state[y][x]
            if not p: continue
            for m in GEN_MAP[p['type']](board_state, x, y, p['side']):
                if m.tx == tx and m.ty == ty:
                    item = (PIECE_VALUES.get(p['type'],0), p['type'], y, x)
                    if p['side']==side: attackers.append(item)
                    else: defenders.append(item)
                    break
    if not attackers: return 0
    attackers.sort(key=lambda x:x[0])
    defenders.sort(key=lambda x:x[0])
    gain = []
    tgt = board_state[ty][tx]
    value_on_square = PIECE_VALUES[tgt['type']] if tgt else 0
    gain.append(value_on_square)
    a_idx=d_idx=0
    side_to_move=side
    while True:
        if side_to_move==side:
            if a_idx>=len(attackers): break
            gain.append(PIECE_VALUES[attackers[a_idx][1]])
            a_idx+=1
        else:
            if d_idx>=len(defenders): break
            gain.append(PIECE_VALUES[defenders[d_idx][1]])
            d_idx+=1
        side_to_move = 'red' if side_to_move=='black' else 'black'
    for i in range(len(gain)-2,-1,-1):
        gain[i] = max(-gain[i+1], gain[i])
    return gain[0]

# ------------------ 分阶段评估（高级版） ------------------
def evaluate_board(board_state):
    """
    分阶段评估（黑方视角，越大越优）
    包含：子力、PST、机动性、王安全、威胁按价值、兵结构、协同、残局特判
    """
    total_material = 0
    black_material = 0
    red_material = 0

    # --- MODIFIED: 升级攻击信息统计 ---
    # attack_info 记录每个格子受到的攻击者信息
    # 格式: {(y, x): [{'value': piece_value, 'side': piece_side}, ...]}
    attack_info = defaultdict(list)

    # 材料统计
    for y in range(ROWS):
        for x in range(COLS):
            p = board_state[y][x]
            if not p: continue
            v = PIECE_VALUES.get(p['type'], 0)
            total_material += v
            if p['side'] == 'black':
                black_material += v
            else:
                red_material += v

    phase = min(1.0, total_material / 16000.0)

    # 填充攻击信息
    for y in range(ROWS):
        for x in range(COLS):
            p = board_state[y][x]
            if not p: continue
            side = p['side']
            att_list = GEN_MAP[p['type']](board_state, x, y, side)
            for m in att_list:
                attack_info[(m.ty, m.tx)].append({'value': PIECE_VALUES.get(p['type'], 0), 'side': side})


    score = 0
    black_mob = 0
    red_mob = 0
    # --- NEW: 用于累加棋子安全度罚分 ---
    safety_penalty = 0

    for y in range(ROWS):
        for x in range(COLS):
            p = board_state[y][x]
            if not p: continue
            side = p['side']
            mult = 1 if side == 'black' else -1
            base = PIECE_VALUES.get(p['type'], 0)
            piece_val = base # 保存原始子力价值，用于安全度计算

            # 过河兵奖励
            if p['type'] in ('卒','兵'):
                if side == 'black':
                    base += max(0, 4 - y) * 12
                else:
                    base += max(0, y - 5) * 12

            # 位置分
            pst_val = PST[(p['type'], side)][y][x]

            # --- MODIFIED: 降低单纯威胁权重 ---
            # 计算该位置的净威胁值
            net_threat = sum(atk['value'] for atk in attack_info[(y,x)] if atk['side'] == 'black') - \
                         sum(atk['value'] for atk in attack_info[(y,x)] if atk['side'] == 'red')
            # 权重从 0.15 降低到 0.08
            base += int(net_threat * 0.08 * mult) 
            
            # --- NEW SECTION: PIECE SAFETY EVALUATION ---
            # 检查当前棋子是否受到对方低价值棋子的攻击
            attackers = [atk for atk in attack_info[(y, x)] if atk['side'] != side]
            if attackers:
                # 找到价值最低的攻击者
                min_attacker_val = min(atk['value'] for atk in attackers)

                # 如果被一个价值更低的棋子攻击，施加惩罚
                if min_attacker_val < piece_val:
                    # 罚分正比于被攻击棋子自身的价值，例如其价值的40%
                    # 这意味着一个车被兵攻击，比一个马被兵攻击，罚分要高得多
                    penalty = int(piece_val * 0.4) 
                    safety_penalty += penalty * mult # 黑子受罚，总分降低；红子受罚，总分升高
            # --- PIECE SAFETY EVALUATION END ---

            # 机动性
            if p['type'] not in ('將','帥'):
                mob = len(GEN_MAP[p['type']](board_state, x, y, side))
                if side == 'black':
                    black_mob += mob
                else:
                    red_mob += mob

            score += (base + pst_val) * mult

    # 将安全度罚分计入总分
    score -= safety_penalty # 从黑方视角看，罚分越高，总分越低

    score += int((black_mob - red_mob) * (10 * (0.6 + 0.4 * phase)))

    # 王安全
    def king_safety_score(side):
        k = find_general(board_state, side)
        if not k:
            return -2000 if side=='black' else 2000
        kx, ky = k['x'], k['y']
        palace_attack_val = 0
        y_range = range(7, 10) if side == 'black' else range(0, 3)
        
        for yy in y_range:
            for xx in range(3, 6):
                # 计算九宫内每个点受到的对方攻击总价值
                for attacker in attack_info.get((yy, xx), []):
                    if attacker['side'] != side:
                        palace_attack_val += attacker['value']
        
        advisors = sum(1 for r in range(ROWS) for c in range(COLS) if (q:=board_state[r][c]) and q['side']==side and q['type'] in ('士','仕'))
        elephants = sum(1 for r in range(ROWS) for c in range(COLS) if (q:=board_state[r][c]) and q['side']==side and q['type'] in ('象','相'))
        shield = advisors + elephants

        # 惩罚系数可以调整，这里用攻击方棋子价值的 1/20 作为基础惩罚
        val = (palace_attack_val // 20) * 18 - (2 - shield) * 60
        
        center_attack_val = 0
        for attacker in attack_info.get((ky, kx), []):
            if attacker['side'] != side:
                center_attack_val += attacker['value']

        val += (center_attack_val // 20) * 25
        
        return val if side == 'black' else -val

    score += king_safety_score('black')
    score += king_safety_score('red')

    # 协同: 双车同线
    def rook_bonus(side):
        rooks = [(y,x) for y in range(ROWS) for x in range(COLS)
                 if (q:=board_state[y][x]) and q['side']==side and q['type'] in ('車','车')]
        if len(rooks) >= 2:
            if rooks[0][0] == rooks[1][0] or rooks[0][1] == rooks[1][1]:
                return 120 if side=='black' else -120
        return 0
    score += rook_bonus('black')
    score += rook_bonus('red')

    # 炮架马
    def cannon_horse_bonus(side):
        bonus = 0
        for y in range(ROWS):
            for x in range(COLS):
                p = board_state[y][x]
                if not p or p['side'] != side: continue
                if p['type'] in ('炮','砲'):
                    for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nx, ny = x+dx, y+dy
                        if 0<=nx<COLS and 0<=ny<ROWS:
                            q = board_state[ny][nx]
                            if q and q['side']==side and q['type'] in ('馬','马'):
                                bonus += 60 if side=='black' else -60
        return bonus
    score += cannon_horse_bonus('black')
    score += cannon_horse_bonus('red')

    # 兵结构
    def pawn_structure(side, piece_type):
        bonus = 0
        pawns = [(y,x) for y in range(ROWS) for x in range(COLS)
                 if (q:=board_state[y][x]) and q['side']==side and q['type']==piece_type]
        for (y,x) in pawns:
            for dx in (-1,1):
                nx = x+dx
                if 0<=nx<COLS and board_state[y][nx] and board_state[y][nx]['side']==side and board_state[y][nx]['type']==piece_type:
                    bonus += 30 if side=='black' else -30
            left = (0<=x-1<COLS and board_state[y][x-1] and board_state[y][x-1]['side']==side and board_state[y][x-1]['type']==piece_type)
            right = (0<=x+1<COLS and board_state[y][x+1] and board_state[y][x+1]['side']==side and board_state[y][x+1]['type']==piece_type)
            if not left and not right:
                bonus -= 18 if side=='black' else -18
        return bonus
    score += pawn_structure('black','卒')
    score += pawn_structure('red','兵')

    # 残局特判
    def count(side, types):
        return sum(1 for y in range(ROWS) for x in range(COLS)
                   if (q:=board_state[y][x]) and q['side']==side and q['type'] in types)
    bp = count('black',['卒'])
    rp = count('red',['兵'])
    br = count('black',['車','车'])
    rr = count('red',['車','车'])
    bc = count('black',['炮','砲'])
    rc = count('red',['炮','砲'])
    bh = count('black',['馬','马'])
    rh = count('red',['馬','马'])

    if bp == 1 and br == 1 and (bc+bh)==0 and (rr+rc+rh+rp)==1:
        score += 140
    if rp == 1 and rr == 1 and (rc+rh)==0 and (br+bc+bh+bp)==1:
        score -= 140

    if bp >= 1 and bc >= 1 and rh == 1 and rr == 0:
        score += 90
    if rp >= 1 and rc >= 1 and bh == 1 and br == 0:
        score -= 90

    if bc == 2 and rh == 1 and (rr+rp+rc)==0:
        score += 110
    if rc == 2 and bh == 1 and (br+bp+bc)==0:
        score -= 110

    if br >= 2 and (rr+rh+rc) <= 1:
        score += 180
    if rr >= 2 and (br+bh+bc) <= 1:
        score -= 180

    if count('black',['將'])==1 and count('black',['士','象'])==0:
        score -= 300
    if count('red',['帥'])==1 and count('red',['仕','相'])==0:
        score += 300

    return score

# ----------------- Quiescence with SEE filter -----------------
def quiescence(board_state, alpha, beta, side_to_move, color_multiplier, start_time=None, time_limit=None):
    if start_time and time_limit and (time.time() - start_time) > time_limit:
        raise TimeoutError

    stand_pat = evaluate_board(board_state) * color_multiplier
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat

    moves = generate_moves(board_state, side_to_move)
    forcing = []
    for m in moves:
        if board_state[m.ty][m.tx] is not None:
            see = static_exchange_evaluation(board_state, m.tx, m.ty, side_to_move)
            # keep capture if SEE + aggressor_value >= 0 (conservative)
            if see + PIECE_VALUES.get(board_state[m.fy][m.fx]['type'], 0) >= 0:
                forcing.append(m)
        else:
            captured = make_move(board_state, m)
            is_chk = in_check(board_state, 'red' if side_to_move=='black' else 'black')
            unmake_move(board_state, m, captured)
            if is_chk:
                forcing.append(m)

    forcing.sort(key=lambda m: (PIECE_VALUES.get(board_state[m.ty][m.tx]['type'], 0) if board_state[m.ty][m.tx] else 0), reverse=True)

    for m in forcing:
        captured = make_move(board_state, m)
        red_king = find_general(board_state, 'red')
        black_king = find_general(board_state, 'black')
        if red_king and black_king and kings_facing(board_state, red_king, black_king):
            unmake_move(board_state, m, captured)
            continue
        if in_check(board_state, side_to_move):
            unmake_move(board_state, m, captured)
            continue

        val = -quiescence(board_state, -beta, -alpha, 'red' if side_to_move=='black' else 'black', -color_multiplier, start_time, time_limit)
        unmake_move(board_state, m, captured)

        if val >= beta:
            return beta
        if val > alpha:
            alpha = val
    return alpha

# ----------------- PVS with extensions -----------------
def pvs_search(board_state, depth, alpha, beta, side_to_move, color_multiplier, current_depth=0, start_time=None, time_limit=None):
    if start_time and time_limit and (time.time() - start_time) > time_limit:
        raise TimeoutError


    if depth <= 0:
        return quiescence(board_state, alpha, beta, side_to_move, color_multiplier, start_time, time_limit)

    is_in_check = in_check(board_state, side_to_move)
    # 通常还会检查子力数量，这里为了简化，暂时省略，但在完整引擎中很重要
    if not is_in_check and depth >= 3:
        # 2. 执行空步搜索
        # R 是深度削减因子，通常取 2 或 3
        R = 2
        
        # 假设我方“空着”，轮到对方走棋
        opponent_side = 'red' if side_to_move == 'black' else 'black'
        
        # 以削减后的深度进行一次PVS搜索
        # 注意这里的 alpha 和 beta 变成了 (-beta, -beta + 1)
        val = -pvs_search(board_state, depth - 1 - R, -beta, -beta + 1, opponent_side, -color_multiplier, current_depth + 1, start_time, time_limit)

        # 3. 判断是否可以裁剪
        if val >= beta:
            # 如果分数很高，说明我方局面优势巨大，可以直接返回beta进行裁剪
            return beta
    # --- 空步裁剪结束 ---

    # --- 原有的 Null Move Reduction (不同于 Pruning) 逻辑，保持不变 ---

    if depth >= 3 and not is_in_check:
        R = 2
        try:
            val = -pvs_search(board_state, depth - 1 - R, -beta, -beta + 1, 'red' if side_to_move=='black' else 'black', -color_multiplier, current_depth+1, start_time, time_limit)
            if val >= beta:
                return beta
        except TimeoutError:
            pass

    moves = generate_moves(board_state, side_to_move, current_depth)
    if not moves:
        if is_in_check:
            return -MATE_SCORE + current_depth
        return 0

    best_val = -MATE_SCORE - 1
    best_move = None
    first = True

    for i, m in enumerate(moves):
        # extensions
        ext = 0
        captured_piece = board_state[m.ty][m.tx]
        if captured_piece is not None:
            victim_val = PIECE_VALUES.get(captured_piece['type'], 0)
            if victim_val >= 800:
                ext = 1

        # check extension
        captured = make_move(board_state, m)
        gives_check = in_check(board_state, 'red' if side_to_move=='black' else 'black')
        unmake_move(board_state, m, captured)
        if gives_check:
            ext = max(ext, 1)

        try:
            captured = make_move(board_state, m)
            if first:
                val = -pvs_search(board_state, depth-1+ext, -beta, -alpha, 'red' if side_to_move=='black' else 'black', -color_multiplier, current_depth+1, start_time, time_limit)
            else:
                val = -pvs_search(board_state, depth-1+ext, -alpha-1, -alpha, 'red' if side_to_move=='black' else 'black', -color_multiplier, current_depth+1, start_time, time_limit)
                if alpha < val < beta:
                    val = -pvs_search(board_state, depth-1+ext, -beta, -alpha, 'red' if side_to_move=='black' else 'black', -color_multiplier, current_depth+1, start_time, time_limit)
            unmake_move(board_state, m, captured)
        except TimeoutError:
            unmake_move(board_state, m, captured)
            raise

        if val > best_val:
            best_val = val

        if val > alpha:
            alpha = val

        if alpha >= beta:
            if board_state[m.ty][m.tx] is None:
                km = KILLER_MOVES[current_depth]
                if m != km[0]:
                    km[1] = km[0]
                    km[0] = m
                piece_type = board_state[m.fy][m.fx]['type'] if board_state[m.fy][m.fx] else None
                if piece_type:
                    key = (piece_type, side_to_move, m.fy, m.fx, m.ty, m.tx)
                    HISTORY_TABLE[key] += depth * depth
                    HISTORY_TABLE[key] = min(HISTORY_TABLE[key], MAX_HISTORY_SCORE)
            break

        first = False
    return best_val

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


def nn_interface(board_state, side):
    from nn_interface import NN_Interface
    nn_player = NN_Interface(model_path="ckpt/latest.pth") 
    _, policy = nn_player.predict(board_state, side)
    # 按概率从高到低排序并打印
    sorted_policy = sorted(policy.items(), key=lambda item: item[1], reverse=True)
    if not sorted_policy: # 确保列表非空
        return None 
    for move, prob in sorted_policy[:5]: # 打印前5个最可能的走法
        print(f"  - 走法: {move.to_dict()}, 概率: {prob:.4f}")
    log(sorted_policy[0][0].to_dict())
    return sorted_policy[0][0].to_dict()

# ----------------- Root Iterative Deepening using PVS -----------------
def minimax_root(board_state, side, time_limit=TIME_LIMIT):    
    move = find_from_position(board_state, side)    
    if move:
        log("[搜索] 棋谱库命中，直接返回棋谱走法"+move)
        return convert_move_string(move)
        #将move格式由字符串"Move(from_x=1, from_y=2, to_x=4, to_y=2)"改为{'from': {'y': 7, 'x': 7}, 'to': {'y': 7, 'x': 6}}                
        # rlt = convert_move_string(move)
        # log(f"[搜索] 格式转换后：{rlt}")
        # return rlt

    log(f"[搜索] Root PVS 开始，执棋方: {side}")
    moves = generate_moves(board_state, side)
    if not moves:
        return None

    start_time = time.time()
    best_move = moves[0]
    best_val = -MATE_SCORE - 1
    color_multiplier = 1 if side == 'black' else -1

    for depth in range(2, MAX_DEPTH + 1):
        time_spent = time.time() - start_time
        if time_spent > time_limit * 0.98:
            break

        try:
            current_best_val = -MATE_SCORE - 1
            current_best_move = None
            alpha = -MATE_SCORE
            beta = MATE_SCORE
          
            moves = generate_moves(board_state, side, depth=0)            

            for m in moves:
                log(f"[搜索] 迭代深度 {depth}，当前走法: {m.to_dict()}")
                captured = make_move(board_state, m)
                try:
                    val = -pvs_search(board_state, depth-1, -beta, -alpha, 'red' if side=='black' else 'black', -color_multiplier, 1, start_time, time_limit * 0.95)
                except TimeoutError:
                    unmake_move(board_state, m, captured)
                    raise
                unmake_move(board_state, m, captured)

                if val > current_best_val:
                    current_best_val = val
                    current_best_move = m

                alpha = max(alpha, current_best_val)
                if alpha >= beta:
                    break

            if current_best_move:
                best_move = current_best_move
                best_val = current_best_val
                log(f"[搜索] 深度 {depth} 完成，最佳估值: {best_val}，最佳走法: {best_move.to_dict() if best_move else None}")

            if time.time() - start_time > time_limit * 0.95:
                break

        except TimeoutError:
            log(f"[搜索] 超时，在深度 {depth} 中断，返回上次最佳走法")
            break

    return best_move.to_dict()

def check_game_over(board_state):
    red_general = find_general(board_state, 'red')
    black_general = find_general(board_state, 'black')
    if not red_general:
        log("[游戏] 游戏结束：黑方胜利！")
        try:
            from opening_book import print_opening_stats
            print_opening_stats()
        except Exception:
            pass
        return {'game_over': True, 'message': '黑方胜利！'}
    if not black_general:
        log("[游戏] 游戏结束：红方胜利！")
        try:
            from opening_book import print_opening_stats
            print_opening_stats()
        except Exception:
            pass
        return {'game_over': True, 'message': '红方胜利！'}
    return {'game_over': False, 'message': ''}
