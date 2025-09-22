# 改进版 ai.py
# 基于用户原始文件（参考）做出若干增强：置换表、静态搜(quiescence)、走子排序、历史表、移动性评估等
import random
import copy
import sys
import logging
import time

logging.basicConfig(
    filename='kimi_backend.log',
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
def log(msg):
    logging.info(msg)

# 统一 piece value 表（保留你原来的数值）
PIECE_VALUES_STD = {
    '將': 10000, '帥': 10000,
    '車': 900, '俥': 900, '车': 900,
    '馬': 450, '傌': 450, '马': 450,
    '炮': 450, '砲': 450,
    '相': 200, '象': 200,
    '仕': 200, '士': 200,
    '兵': 100, '卒': 100
}

# 棋盘尺寸
ROWS = 10
COLS = 9

# --- 简易开局库（保留） ---
FIRST_MOVES = [
    {'from': {'y': 2, 'x': 1}, 'to': {'y': 2, 'x': 4}},  # 中炮: 炮二平五
    {'from': {'y': 3, 'x': 6}, 'to': {'y': 4, 'x': 6}},  # 仙人指路: 兵七进一
    {'from': {'y': 0, 'x': 7}, 'to': {'y': 2, 'x': 6}},  # 起马: 马八进七
    {'from': {'y': 3, 'x': 2}, 'to': {'y': 4, 'x': 2}},  # 兵三进一
    {'from': {'y': 0, 'x': 1}, 'to': {'y': 2, 'x': 2}},  # 起马: 马一平二
]

# --- Zobrist 哈希 ---
ZOBRIST_TABLE = {}
ZOBRIST_TURN = random.getrandbits(64)

def init_zobrist():
    pieces = list(PIECE_VALUES_STD.keys())
    sides = ['red', 'black']
    for y in range(ROWS):
        for x in range(COLS):
            for p in pieces:
                for s in sides:
                    ZOBRIST_TABLE[(y, x, p, s)] = random.getrandbits(64)
init_zobrist()

def compute_hash(board_state, side_to_move):
    h = 0
    for y in range(ROWS):
        for x in range(COLS):
            piece = board_state[y][x]
            if piece:
                h ^= ZOBRIST_TABLE[(y, x, piece['type'], piece['side'])]
    if side_to_move == 'black':
        h ^= ZOBRIST_TURN
    return h

def update_hash(h, from_pos, to_pos, moving_piece, captured_piece, side_to_move):
    fx, fy = from_pos['x'], from_pos['y']
    tx, ty = to_pos['x'], to_pos['y']
    h ^= ZOBRIST_TABLE[(fy, fx, moving_piece['type'], moving_piece['side'])]
    if captured_piece:
        h ^= ZOBRIST_TABLE[(ty, tx, captured_piece['type'], captured_piece['side'])]
    h ^= ZOBRIST_TABLE[(ty, tx, moving_piece['type'], moving_piece['side'])]
    h ^= ZOBRIST_TURN
    return h

# --- 工具函数 ---
def board_to_tuple(board_state):
    return tuple(tuple((item['type'] + '_' + item['side']) if item else None for item in row) for row in board_state)

# --- 基本走法判断（保留原有实现） ---
def can_move_on(board_state, from_pos, to_pos):
    fx, fy = from_pos['x'], from_pos['y']
    tx, ty = to_pos['x'], to_pos['y']

    if not (0 <= fx < COLS and 0 <= fy < ROWS and 0 <= tx < COLS and 0 <= ty < ROWS):
        return False

    piece_obj = board_state[fy][fx]
    if not piece_obj:
        return False

    name = piece_obj['type']
    side = piece_obj['side']
    target = board_state[ty][tx]
    if target and target['side'] == side:
        return False

    if name in ['車', '俥', '车']:
        return can_move_chariot_on(board_state, fx, fy, tx, ty)
    elif name in ['炮', '砲']:
        return can_move_cannon_on(board_state, fx, fy, tx, ty, bool(target))
    elif name in ['馬', '傌', '马']:
        return can_move_horse_on(board_state, fx, fy, tx, ty)
    elif name in ['兵', '卒']:
        return can_move_soldier_on(board_state, fx, fy, tx, ty, side)
    elif name in ['帥', '將']:
        return can_move_general_on(board_state, fx, fy, tx, ty, side)
    elif name in ['相', '象']:
        return can_move_elephant_on(board_state, fx, fy, tx, ty, side)
    elif name in ['仕', '士']:
        return can_move_advisor_on(board_state, fx, fy, tx, ty, side)
    else:
        return False

def can_move_chariot_on(board_state, from_x, from_y, to_x, to_y):
    if from_x == to_x:
        min_y, max_y = min(from_y, to_y), max(from_y, to_y)
        for i in range(min_y + 1, max_y):
            if board_state[i][from_x] is not None:
                return False
        return True
    elif from_y == to_y:
        min_x, max_x = min(from_x, to_x), max(from_x, to_x)
        for i in range(min_x + 1, max_x):
            if board_state[from_y][i] is not None:
                return False
        return True
    return False

def can_move_cannon_on(board_state, from_x, from_y, to_x, to_y, is_capture):
    obstacle_count = 0
    if from_x == to_x:
        min_y, max_y = min(from_y, to_y), max(from_y, to_y)
        for i in range(min_y + 1, max_y):
            if board_state[i][from_x] is not None:
                obstacle_count += 1
    elif from_y == to_y:
        min_x, max_x = min(from_x, to_x), max(from_x, to_x)
        for i in range(min_x + 1, max_x):
            if board_state[from_y][i] is not None:
                obstacle_count += 1
    else:
        return False

    return (is_capture and obstacle_count == 1) or (not is_capture and obstacle_count == 0)

def can_move_horse_on(board_state, from_x, from_y, to_x, to_y):
    dx = abs(to_x - from_x)
    dy = abs(to_y - from_y)

    if not ((dx == 1 and dy == 2) or (dx == 2 and dy == 1)):
        return False

    if dx == 1:
        check_y = from_y + (1 if to_y > from_y else -1)
        if board_state[check_y][from_x] is not None:
            return False
    else:
        check_x = from_x + (1 if to_x > from_x else -1)
        if board_state[from_y][check_x] is not None:
            return False
    return True

def can_move_soldier_on(board_state, from_x, from_y, to_x, to_y, side):
    dx = abs(to_x - from_x)
    dy = to_y - from_y
    is_across_river = (side == 'red' and from_y >= 5) or (side == 'black' and from_y <= 4)

    if (side == 'red' and dy < 0) or (side == 'black' and dy > 0):
        return False

    if dx + abs(dy) != 1:
        return False

    if is_across_river:
        if side == 'red':
            return (dy == 1 and dx == 0) or (dy == 0 and dx == 1)
        else:
            return (dy == -1 and dx == 0) or (dy == 0 and dx == 1)
    else:
        if side == 'red':
            return dx == 0 and dy == 1
        else:
            return dx == 0 and dy == -1

def can_move_general_on(board_state, from_x, from_y, to_x, to_y, side):
    dx = abs(to_x - from_x)
    dy = abs(to_y - from_y)

    if dx + dy != 1:
        return False

    if not (3 <= to_x <= 5):
        return False
    if side == 'red' and not (0 <= to_y <= 2):
        return False
    if side == 'black' and not (7 <= to_y <= 9):
        return False

    # 移动后“面对面”不可允许
    if from_x == to_x:
        opponent_general_y = -1
        for y in range(ROWS):
            piece = board_state[y][to_x]
            if piece and (piece['type'] in ['將', '帥']) and piece['side'] != side:
                opponent_general_y = y
                break

        if opponent_general_y != -1:
            blocked = False
            min_y, max_y = min(to_y, opponent_general_y), max(to_y, opponent_general_y)
            for i in range(min_y + 1, max_y):
                if board_state[i][to_x] is not None:
                    blocked = True
                    break
            if not blocked:
                return False
    return True

def can_move_elephant_on(board_state, from_x, from_y, to_x, to_y, side):
    dx = abs(to_x - from_x)
    dy = abs(to_y - from_y)

    if dx != 2 or dy != 2:
        return False

    if (side == 'red' and to_y > 4) or (side == 'black' and to_y < 5):
        return False

    mid_x = (from_x + to_x) // 2
    mid_y = (from_y + to_y) // 2
    if board_state[mid_y][mid_x] is not None:
        return False
    return True

def can_move_advisor_on(board_state, from_x, from_y, to_x, to_y, side):
    dx = abs(to_x - from_x)
    dy = abs(to_y - from_y)

    if dx != 1 or dy != 1:
        return False

    if not (3 <= to_x <= 5):
        return False
    if side == 'red' and not (0 <= to_y <= 2):
        return False
    if side == 'black' and not (7 <= to_y <= 9):
        return False

    return True

# --- 将帅检测（保留） ---
def find_general_in_board_state(board_state, side):
    for y in range(ROWS):
        for x in range(COLS):
            piece = board_state[y][x]
            if piece and (piece['type'] in ['帥', '將']) and piece['side'] == side:
                return {'x': x, 'y': y}
    return None

def is_in_check_board(board_state, side):
    general = find_general_in_board_state(board_state, side)
    if not general:
        return False

    opponent = 'red' if side == 'black' else 'black'
    for y in range(ROWS):
        for x in range(COLS):
            piece = board_state[y][x]
            if piece and piece['side'] == opponent:
                if can_move_on(board_state, {'x': x, 'y': y}, general):
                    return True
    return False

def is_king_facing_king_board(board_state):
    red_king = None
    black_king = None

    for y in range(ROWS):
        for x in range(COLS):
            piece = board_state[y][x]
            if not piece:
                continue
            if piece['type'] == '帥':
                red_king = {'x': x, 'y': y}
            if piece['type'] == '將':
                black_king = {'x': x, 'y': y}

    if red_king and black_king and red_king['x'] == black_king['x']:
        min_y, max_y = min(red_king['y'], black_king['y']), max(red_king['y'], black_king['y'])
        for y in range(min_y + 1, max_y):
            if board_state[y][red_king['x']] is not None:
                return False
        return True
    return False

# --- 历史（长将）检测（保留） ---
def is_perpetual_check(history, required_repeat=3):
    counts = {}
    for board_hash, is_check, checking_side in history:
        if not is_check:
            continue
        key = (board_hash, checking_side)
        counts[key] = counts.get(key, 0) + 1
        if counts[key] >= required_repeat:
            return checking_side
    return None

# --- 评估函数（增强：PST、移动性、被将状态、历史惩罚/奖励） ---
def evaluate_board(board_state, history):
    score = 0
    for y in range(ROWS):
        for x in range(COLS):
            piece = board_state[y][x]
            if not piece:
                continue

            val = PIECE_VALUES_STD.get(piece['type'], 0)
            side_mult = 1 if piece['side'] == 'black' else -1
            add = val * side_mult

            # PST 简单示例
            if piece['type'] in ['兵', '卒']:
                add += PST_SOLDIER[y][x] * side_mult
                is_across = (piece['side'] == 'red' and y >= 5) or (piece['side'] == 'black' and y <= 4)
                if is_across:
                    add += 25 * side_mult

            if 3 <= x <= 5:
                add += 5 * side_mult

            score += add

    # 🚀 新增：轻量化 mobility 计算（避免递归）
    def count_legal_moves(board_state, side):
        count = 0
        for yy in range(ROWS):
            for xx in range(COLS):
                piece = board_state[yy][xx]
                if not piece or piece['side'] != side:
                    continue
                for ty in range(ROWS):
                    for tx in range(COLS):
                        if can_move_on(board_state, {'x': xx, 'y': yy}, {'x': tx, 'y': ty}):
                            tmp_board = copy.deepcopy(board_state)
                            tmp_board[ty][tx] = tmp_board[yy][xx]
                            tmp_board[yy][xx] = None
                            if not is_in_check_board(tmp_board, side) and not is_king_facing_king_board(tmp_board):
                                count += 1
        return count

    mobility_black = count_legal_moves(board_state, 'black')
    mobility_red = count_legal_moves(board_state, 'red')
    score += (mobility_black - mobility_red) * 2

    # 将军奖励/惩罚
    if is_in_check_board(board_state, 'red'):
        score += 200
    if is_in_check_board(board_state, 'black'):
        score -= 200

    offender = is_perpetual_check(history)
    if offender is not None:
        if offender == 'black':
            score -= 1000000
        else:
            score += 1000000

    return score


# --- 置换表（简单实现） ---
TRANSP_TABLE = {}  # key: board_tuple, value: (depth, score, flag) flag: 'EXACT'/'LOWER'/'UPPER'

# --- 历史表（history heuristic）: key: (from_x,from_y,to_x,to_y) -> score ---
HISTORY_HEUR = {}

# --- 走子评分（用于走子排序） ---
def score_move_board(from_pos, to_pos, board_state, history, side):
    # 保留你的局部模拟并加上排序信息
    tmp_board = copy.deepcopy(board_state)
    moving_piece = tmp_board[from_pos['y']][from_pos['x']]
    captured_piece = tmp_board[to_pos['y']][to_pos['x']]

    base = 0
    if captured_piece:
        base += PIECE_VALUES_STD.get(captured_piece['type'], 0)

    tmp_board[to_pos['y']][to_pos['x']] = moving_piece
    tmp_board[from_pos['y']][from_pos['x']] = None

    if is_in_check_board(tmp_board, side):
        return -999999

    opponent = 'red' if side == 'black' else 'black'
    check_bonus = 0
    if is_in_check_board(tmp_board, opponent):
        check_bonus = 150

    opponent_general_type = '帥' if opponent == 'red' else '將'
    general_captured = False
    for y in range(ROWS):
        for x in range(COLS):
            piece = tmp_board[y][x]
            if piece and piece['type'] == opponent_general_type and piece['side'] == opponent:
                general_captured = True
                break
        if general_captured:
            break

    if general_captured:
        check_bonus += 10000

    is_check_now = is_in_check_board(tmp_board, opponent)
    new_history = history + [(board_to_tuple(tmp_board), is_check_now, side)]

    offender = is_perpetual_check(new_history)
    if offender is not None:
        if offender == 'black':
            return -1000000
        else:
            return 1000000

    # 被吃/保护评估（保留）
    penalty = 0
    for y in range(ROWS):
        for x in range(COLS):
            piece = tmp_board[y][x]
            if piece and piece['side'] == opponent:
                if can_move_on(tmp_board, {'x': x, 'y': y}, to_pos):
                    sim_board = copy.deepcopy(tmp_board)
                    sim_board[to_pos['y']][to_pos['x']] = sim_board[y][x]
                    sim_board[y][x] = None

                    if not is_in_check_board(sim_board, opponent):
                        attacker_value = PIECE_VALUES_STD.get(piece['type'], 0)
                        victim_value = PIECE_VALUES_STD.get(moving_piece['type'], 0)

                        can_recapture = False
                        recapture_value = 0
                        for yy in range(ROWS):
                            for xx in range(COLS):
                                ally = sim_board[yy][xx]
                                if ally and ally['side'] == side:
                                    if can_move_on(sim_board, {'x': xx, 'y': yy}, to_pos):
                                        can_recapture = True
                                        recapture_value = PIECE_VALUES_STD.get(ally['type'], 0)
                                        break
                            if can_recapture:
                                break

                        if can_recapture:
                            trade_loss = victim_value - attacker_value
                            if trade_loss > 0:
                                penalty = trade_loss
                        else:
                            penalty = victim_value
                        break
        if penalty > 0:
            break

    eval_score = evaluate_board(tmp_board, new_history)
    total_score = eval_score + check_bonus + base - penalty
    return total_score

# --- 生成所有合法走法（改进：返回额外信息，便于排序） ---
def get_all_legal_moves_board(board_state, side, history):
    moves = []
    for y in range(ROWS):
        for x in range(COLS):
            piece = board_state[y][x]
            if not piece or piece['side'] != side:
                continue

            for ty in range(ROWS):
                for tx in range(COLS):
                    if not can_move_on(board_state, {'x': x, 'y': y}, {'x': tx, 'y': ty}):
                        continue

                    tmp_board = copy.deepcopy(board_state)
                    tmp_board[ty][tx] = tmp_board[y][x]
                    tmp_board[y][x] = None

                    # 不能走出会被自己将军的走法
                    if is_in_check_board(tmp_board, side):
                        continue

                    # 不能造成双方将帅面对面
                    if is_king_facing_king_board(tmp_board):
                        continue

                    score = score_move_board({'x': x, 'y': y}, {'x': tx, 'y': ty}, board_state, history, side)

                    # 附加排序键： capture_value, promotion-like bonus（这里没有升变），history heuristic
                    capture = board_state[ty][tx]
                    capture_value = PIECE_VALUES_STD.get(capture['type'], 0) if capture else 0
                    history_bonus = HISTORY_HEUR.get((x,y,tx,ty), 0)

                    moves.append({
                        'from': {'x': x, 'y': y},
                        'to': {'x': tx, 'y': ty},
                        'score': score,
                        'capture_value': capture_value,
                        'history_bonus': history_bonus
                    })

    # 排序：以 capture_value（高先） -> history_bonus -> heuristic score
    moves.sort(key=lambda m: (m['capture_value'], m['history_bonus'], m['score']), reverse=True)
    # 如果你希望 black/ red 按不同偏好排序，可以在调用处再反转
    return moves

# --- 生成仅“吃子”走法（用于静态延伸） ---
def generate_capture_moves(board_state, side):
    captures = []
    for y in range(ROWS):
        for x in range(COLS):
            piece = board_state[y][x]
            if not piece or piece['side'] != side:
                continue
            for ty in range(ROWS):
                for tx in range(COLS):
                    target = board_state[ty][tx]
                    if not target:
                        continue
                    if target['side'] == side:
                        continue
                    if can_move_on(board_state, {'x': x, 'y': y}, {'x': tx, 'y': ty}):
                        capture_value = PIECE_VALUES_STD.get(target['type'], 0)
                        captures.append({'from': {'x': x, 'y': y}, 'to': {'x': tx, 'y': ty}, 'capture_value': capture_value})
    # MVV-LVA style: higher captured value first
    captures.sort(key=lambda m: m['capture_value'], reverse=True)
    return captures

# --- 检查胜负（保留） ---
def check_game_over(board_state):
    red_general = find_general_in_board_state(board_state, 'red')
    black_general = find_general_in_board_state(board_state, 'black')

    if not red_general:
        return {'game_over': True, 'message': '黑方胜利！'}
    if not black_general:
        return {'game_over': True, 'message': '红方胜利！'}
    return {'game_over': False, 'message': ''}

# --- Quiescence Search（静态搜索） ---
def quiescence_search(board_state, alpha, beta, side, history):
    stand_pat = evaluate_board(board_state, history)
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    captures = generate_capture_moves(board_state, side)
    for mv in captures:
        from_pos, to_pos = mv['from'], mv['to']
        tmp_board = copy.deepcopy(board_state)
        tmp_board[to_pos['y']][to_pos['x']] = tmp_board[from_pos['y']][from_pos['x']]
        tmp_board[from_pos['y']][from_pos['x']] = None

        if is_in_check_board(tmp_board, side):
            continue

        opponent = 'red' if side == 'black' else 'black'
        is_check_now = is_in_check_board(tmp_board, opponent)
        new_history = history + [(board_to_tuple(tmp_board), is_check_now, side)]

        score = -quiescence_search(tmp_board, -beta, -alpha, opponent, new_history)
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha

# --- Minimax（带 alpha-beta、置换表、静态搜索与历史启发） ---
def minimax(board_state, depth, alpha, beta, side_to_move, history):
    key = board_to_tuple(board_state)
    tt_entry = TRANSP_TABLE.get(key)
    if tt_entry:
        tt_depth, tt_score, tt_flag = tt_entry
        if tt_depth >= depth:
            if tt_flag == 'EXACT':
                return tt_score
            elif tt_flag == 'LOWER' and tt_score > alpha:
                alpha = tt_score
            elif tt_flag == 'UPPER' and tt_score < beta:
                beta = tt_score
            if alpha >= beta:
                return tt_score

    if depth == 0:
        # 使用静态搜索（quiescence）
        qscore = quiescence_search(board_state, alpha, beta, side_to_move, history)
        return qscore

    moves = get_all_legal_moves_board(board_state, side_to_move, history)
    if not moves:
        if is_in_check_board(board_state, side_to_move):
            return -1000000 if side_to_move == 'black' else 1000000
        else:
            return 0

    best_score = -float('inf')
    best_flag = 'UPPER'
    opponent = 'red' if side_to_move == 'black' else 'black'

    # 对 moves 已经按吃子与历史排序；我们仍然在循环中尝试 alpha-beta
    for move in moves:
        from_pos = move['from']
        to_pos = move['to']

        tmp_board = copy.deepcopy(board_state)
        moving = tmp_board[from_pos['y']][from_pos['x']]
        captured = tmp_board[to_pos['y']][to_pos['x']]
        tmp_board[to_pos['y']][to_pos['x']] = moving
        tmp_board[from_pos['y']][from_pos['x']] = None

        if is_in_check_board(tmp_board, side_to_move):
            continue

        is_check_now = is_in_check_board(tmp_board, opponent)
        new_history = history + [(board_to_tuple(tmp_board), is_check_now, side_to_move)]

        score = -minimax(tmp_board, depth - 1, -beta, -alpha, opponent, new_history)

        # 如果该走子导致更好结果，更新历史表（简单增量）
        if score > best_score:
            best_score = score
        if score > alpha:
            alpha = score
            best_flag = 'EXACT'
            # history heuristic 增强
            keyh = (from_pos['x'], from_pos['y'], to_pos['x'], to_pos['y'])
            HISTORY_HEUR[keyh] = HISTORY_HEUR.get(keyh, 0) + (1 << (depth))  # 深度奖励更高
        if alpha >= beta:
            # 失败-剪枝 -> 增强历史表（杀手/促动）
            keyh = (from_pos['x'], from_pos['y'], to_pos['x'], to_pos['y'])
            HISTORY_HEUR[keyh] = HISTORY_HEUR.get(keyh, 0) + (1 << (depth+1))
            break

    # 存入置换表
    if best_score <= alpha:
        flag = 'UPPER'
    elif best_score >= beta:
        flag = 'LOWER'
    else:
        flag = 'EXACT'
    TRANSP_TABLE[key] = (depth, best_score, flag)
    return best_score

# --- 根节点调用（带迭代加深，可按需限制 depth） ---
def minimax_root(board_state, depth, side, history=None, use_iterative=False):
    if history is None:
        history = []

    log(f"minimax_root: side={side}, depth={depth}, history_len={len(history)}")
    # 开局库判断（保留）
    starting_position = [[{'type': '車', 'side': 'red'}, {'type': '馬', 'side': 'red'}, {'type': '相', 'side': 'red'}, {'type': '仕', 'side': 'red'}, {'type': '帥', 'side': 'red'}, {'type': '仕', 'side': 'red'}, {'type': '相', 'side': 'red'}, {'type': '馬', 'side': 'red'}, {'type': '車', 'side': 'red'}],
                         [None]*9,
                         [None, {'type': '炮', 'side': 'red'}, None, None, None, None, None, {'type': '炮', 'side': 'red'}, None],
                         [{'type': '兵', 'side': 'red'}, None, {'type': '兵', 'side': 'red'}, None, {'type': '兵', 'side': 'red'}, None, {'type': '兵', 'side': 'red'}, None, {'type': '兵', 'side': 'red'}],
                         [None]*9,
                         [None]*9,
                         [{'type': '卒', 'side': 'black'}, None, {'type': '卒', 'side': 'black'}, None, {'type': '卒', 'side': 'black'}, None, {'type': '卒', 'side': 'black'}, None, {'type': '卒', 'side': 'black'}],
                         [None, {'type': '炮', 'side': 'black'}, None, None, None, None, None, {'type': '炮', 'side': 'black'}, None],
                         [None]*9,
                         [{'type': '車', 'side': 'black'}, {'type': '馬', 'side': 'black'}, {'type': '象', 'side': 'black'}, {'type': '士', 'side': 'black'}, {'type': '將', 'side': 'black'}, {'type': '士', 'side': 'black'}, {'type': '象', 'side': 'black'}, {'type': '馬', 'side': 'black'}, {'type': '車', 'side': 'black'}]]

    if board_state == starting_position and side == 'red':
        log("minimax_root: using opening book move")
        return random.choice(FIRST_MOVES)

    moves = get_all_legal_moves_board(board_state, side, history)
    if not moves:
        log("minimax_root: no legal moves")
        return None

    # 如果采用迭代加深，先浅层到深层循环（可按需启用）
    best_moves = []
    if use_iterative:
        best_val = -float('inf') if side == 'black' else float('inf')
        for d in range(1, depth + 1):
            TRANSP_TABLE.clear()  # 每次深度重用置换表（可保留）或保留以提升
            current_best_moves = []
            if side == 'black':
                local_best = -float('inf')
                for move in moves:
                    tmp_board = copy.deepcopy(board_state)
                    tmp_board[move['to']['y']][move['to']['x']] = tmp_board[move['from']['y']][move['from']['x']]
                    tmp_board[move['from']['y']][move['from']['x']] = None

                    opponent = 'red'
                    is_check_now = is_in_check_board(tmp_board, opponent)
                    new_history = history + [(board_to_tuple(tmp_board), is_check_now, side)]

                    val = minimax(tmp_board, d - 1, -float('inf'), float('inf'), opponent, new_history)
                    if val > local_best:
                        local_best = val
                        current_best_moves = [move]
                    elif val == local_best:
                        current_best_moves.append(move)
                best_val = local_best
                best_moves = current_best_moves
            else:
                local_best = float('inf')
                for move in moves:
                    tmp_board = copy.deepcopy(board_state)
                    tmp_board[move['to']['y']][move['to']['x']] = tmp_board[move['from']['y']][move['from']['x']]
                    tmp_board[move['from']['y']][move['from']['x']] = None

                    opponent = 'black'
                    is_check_now = is_in_check_board(tmp_board, opponent)
                    new_history = history + [(board_to_tuple(tmp_board), is_check_now, side)]

                    val = minimax(tmp_board, d - 1, -float('inf'), float('inf'), opponent, new_history)
                    if val < local_best:
                        local_best = val
                        current_best_moves = [move]
                    elif val == local_best:
                        current_best_moves.append(move)
                best_val = local_best
                best_moves = current_best_moves
            # log per-depth
            log(f"minimax_root: iterative depth={d} best_val={best_val} best_moves_count={len(best_moves)}")
        return random.choice(best_moves) if best_moves else random.choice(moves)
    else:
        if side == 'black':
            best_val = -float('inf')
            best_moves = []
            for move in moves:
                tmp_board = copy.deepcopy(board_state)
                tmp_board[move['to']['y']][move['to']['x']] = tmp_board[move['from']['y']][move['from']['x']]
                tmp_board[move['from']['y']][move['from']['x']] = None

                opponent = 'red'
                is_check_now = is_in_check_board(tmp_board, opponent)
                new_history = history + [(board_to_tuple(tmp_board), is_check_now, side)]

                val = minimax(tmp_board, depth - 1, -float('inf'), float('inf'), opponent, new_history)
                log(f"minimax_root: black move {move} val={val}")
                if val > best_val:
                    best_val = val
                    best_moves = [move]
                elif val == best_val:
                    best_moves.append(move)
            log(f"minimax_root: black best_val={best_val} best_moves={best_moves}")
            return random.choice(best_moves)
        else:  # red
            best_val = float('inf')
            best_moves = []
            for move in moves:
                tmp_board = copy.deepcopy(board_state)
                tmp_board[move['to']['y']][move['to']['x']] = tmp_board[move['from']['y']][move['from']['x']]
                tmp_board[move['from']['y']][move['from']['x']] = None

                opponent = 'black'
                is_check_now = is_in_check_board(tmp_board, opponent)
                new_history = history + [(board_to_tuple(tmp_board), is_check_now, side)]

                val = minimax(tmp_board, depth - 1, -float('inf'), float('inf'), opponent, new_history)
                log(f"minimax_root: red move {move} val={val}")
                if val < best_val:
                    best_val = val
                    best_moves = [move]
                elif val == best_val:
                    best_moves.append(move)
            log(f"minimax_root: red best_val={best_val} best_moves={best_moves}")
            return random.choice(best_moves)
