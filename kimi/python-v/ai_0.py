"""
Optimized Chinese Chess AI
Streamlined version with only essential functions and improved structure.
"""

import random
import copy
import logging
import time
from collections import defaultdict

# --- Configuration ---
MAX_DEPTH = 6
TIME_LIMIT = 5.0
ASPIRATION_WINDOW_DELTA = 50

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

# Piece-square tables
PST = defaultdict(lambda: [[0]*COLS for _ in range(ROWS)])
# Soldiers advancement and center control
for y in range(ROWS):
    for x in range(COLS):
        PST[('兵','red')][y][x] = (y - 3) * 5
        PST[('卒','black')][y][x] = (6 - y) * 5
        if 3 <= x <= 5:
            PST[('兵','red')][y][x] += 5
            PST[('卒','black')][y][x] += 5

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

# Transposition table and Killer moves
TT = {}
KILLER_MOVES = defaultdict(lambda: [None, None])

# Opening book
FIRST_MOVES = [
    {'from': {'y': 2, 'x': 1}, 'to': {'y': 2, 'x': 4}},
    {'from': {'y': 3, 'x': 6}, 'to': {'y': 4, 'x': 6}},
    {'from': {'y': 0, 'x': 7}, 'to': {'y': 2, 'x': 6}},
    {'from': {'y': 3, 'x': 2}, 'to': {'y': 4, 'x': 2}},
    {'from': {'y': 0, 'x': 1}, 'to': {'y': 2, 'x': 2}},
]

# ---- Movement Direction Constants ----
D4 = ((1,0),(-1,0),(0,1),(0,-1))          # Straight lines
D4_O = ((1,1),(1,-1),(-1,1),(-1,-1))      # Diagonal
H8 = ((1,2),(2,1),(-1,2),(-2,1),          # Horse
      (1,-2),(2,-1),(-1,-2),(-2,-1))
E4 = ((2,2),(2,-2),(-2,2),(-2,-2))        # Elephant

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
    """Check if side is in check"""
    if king_pos is None:
        king_pos = find_general(board, side)
        if king_pos is None:
            return False
    x, y = king_pos['x'], king_pos['y']
    opp = 'red' if side == 'black' else 'black'
    
    # Check straight lines (chariot, cannon, general)
    for dx, dy in D4:
        nx, ny = x + dx, y + dy
        step = 0
        while inside(nx, ny):
            p = board[ny][nx]
            if p is None:
                nx += dx
                ny += dy
                continue
            if p['side'] == opp:
                t = p['type']
                if t in {'車','俥','车','帥','將'} and step == 0:
                    return True
                if t in {'炮','砲'} and step == 1:
                    return True
            break
        
        # Check cannon jumps
        nx, ny = x + dx, y + dy
        step = 0
        while inside(nx, ny):
            p = board[ny][nx]
            if p is None:
                nx += dx
                ny += dy
                continue
            if step == 1 and p['side'] == opp and p['type'] in {'炮','砲'}:
                return True
            step += 1
            if step > 1:
                break
            nx += dx
            ny += dy
    
    # Check horse
    for dx, dy in H8:
        nx, ny = x + dx, y + dy
        if not inside(nx, ny):
            continue
        # Check horse leg blocking
        mx, my = x + dx//2, y + dy//2
        if board[my][mx] is not None:
            continue
        p = board[ny][nx]
        if p and p['side'] == opp and p['type'] in {'馬','傌','马'}:
            return True
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
                moves.append({'from': {'x': x, 'y': y}, 'to': {'x': nx, 'y': ny}})
            else:
                if t['side'] != side:
                    moves.append({'from': {'x': x, 'y': y}, 'to': {'x': nx, 'y': ny}})
                break
            nx += dx
            ny += dy
    return moves

def gen_cannon(board, x, y, side):
    """Generate cannon moves"""
    moves = []
    for dx, dy in D4:
        # First phase: move through empty squares
        nx, ny = x + dx, y + dy
        while inside(nx, ny) and board[ny][nx] is None:
            moves.append({'from': {'x': x, 'y': y}, 'to': {'x': nx, 'y': ny}})
            nx += dx
            ny += dy
        # Jump over first obstacle
        if not inside(nx, ny):
            continue
        nx += dx
        ny += dy
        # Second phase: capture enemy piece
        while inside(nx, ny):
            t = board[ny][nx]
            if t is not None:
                if t['side'] != side:
                    moves.append({'from': {'x': x, 'y': y}, 'to': {'x': nx, 'y': ny}})
                break
            nx += dx
            ny += dy
    return moves

def gen_horse(board, x, y, side):
    """Generate horse moves"""
    moves = []
    for dx, dy in H8:
        nx, ny = x + dx, y + dy
        if not inside(nx, ny):
            continue
        # Check horse leg blocking
        mx, my = x + dx//2, y + dy//2
        if board[my][mx] is not None:
            continue
        t = board[ny][nx]
        if t is None or t['side'] != side:
            moves.append({'from': {'x': x, 'y': y}, 'to': {'x': nx, 'y': ny}})
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
            moves.append({'from': {'x': x, 'y': y}, 'to': {'x': x, 'y': ny}})
    # Move sideways after crossing river
    river_crossed = (side == 'red' and y >= 5) or (side == 'black' and y <= 4)
    if river_crossed:
        for dx in (-1, 1):
            nx = x + dx
            if not inside(nx, y):
                continue
            t = board[y][nx]
            if t is None or t['side'] != side:
                moves.append({'from': {'x': x, 'y': y}, 'to': {'x': nx, 'y': y}})
    return moves

def gen_general(board, x, y, side):
    """Generate general moves"""
    palace = range(3, 6)
    y_range = (0, 1, 2) if side == 'red' else (7, 8, 9)
    moves = []
    for dx, dy in D4:
        nx, ny = x + dx, y + dy
        if nx not in palace or ny not in y_range:
            continue
        t = board[ny][nx]
        if t is None or t['side'] != side:
            moves.append({'from': {'x': x, 'y': y}, 'to': {'x': nx, 'y': ny}})
    # Flying general (capture enemy general in same file)
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
            moves.append({'from': {'x': x, 'y': y}, 'to': {'x': k2['x'], 'y': k2['y']}})
    return moves

def gen_advisor(board, x, y, side):
    """Generate advisor moves"""
    palace = range(3, 6)
    y_range = (0, 1, 2) if side == 'red' else (7, 8, 9)
    moves = []
    for dx, dy in D4_O:
        nx, ny = x + dx, y + dy
        if nx not in palace or ny not in y_range:
            continue
        t = board[ny][nx]
        if t is None or t['side'] != side:
            moves.append({'from': {'x': x, 'y': y}, 'to': {'x': nx, 'y': ny}})
    return moves

def gen_elephant(board, x, y, side):
    """Generate elephant moves"""
    y_limit = 5 if side == 'red' else 4
    moves = []
    for dx, dy in E4:
        nx, ny = x + dx, y + dy
        if (ny > y_limit if side == 'red' else ny < y_limit):
            continue
        if not inside(nx, ny):
            continue
        # Check elephant eye blocking
        mx, my = x + dx//2, y + dy//2
        if board[my][mx] is not None:
            continue
        t = board[ny][nx]
        if t is None or t['side'] != side:
            moves.append({'from': {'x': x, 'y': y}, 'to': {'x': nx, 'y': ny}})
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

def generate_moves(board_state, side, tt_best_move=None, depth=0):
    """Generate all legal moves for a side"""
    # Generate pseudo-legal moves
    pseudo = []
    for y in range(10):
        for x in range(9):
            p = board_state[y][x]
            if p and p['side'] == side:
                pseudo += GEN_MAP[p['type']](board_state, x, y, side)

    # Filter for legal moves
    legal = []
    red_king = find_general(board_state, 'red')
    black_king = find_general(board_state, 'black')
    
    for m in pseudo:
        # Make move on temporary board
        nb = [row[:] for row in board_state]
        fx, fy = m['from']['x'], m['from']['y']
        tx, ty = m['to']['x'], m['to']['y']
        nb[ty][tx] = nb[fy][fx]
        nb[fy][fx] = None
        
        # Check legality
        if kings_facing(nb, red_king, black_king):
            continue
        if in_check(nb, side):
            continue
            
        # Score the move
        captured = board_state[ty][tx]
        score = 0
        if captured:
            score += 10 * PIECE_VALUES[captured['type']] - PIECE_VALUES[nb[ty][tx]['type']]
        score += PST[(nb[ty][tx]['type'], side)][ty][tx]
        m['score'] = score
        legal.append(m)

    # Apply killer move heuristic
    km = KILLER_MOVES[depth]
    for m in legal:
        if m == km[0]:
            m['score'] += 1000
        elif m == km[1]:
            m['score'] += 900
    
    # Sort by score
    legal.sort(key=lambda x: x['score'], reverse=True)
    
    # Move TT best move to front
    if tt_best_move:
        for i, m in enumerate(legal):
            if m['from'] == tt_best_move['from'] and m['to'] == tt_best_move['to']:
                legal.insert(0, legal.pop(i))
                break
    
    return legal

def copy_board(board_state):
    """Create a copy of the board"""
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

def evaluate_board(board_state):
    """Evaluate board position"""
    score = 0
    
    # Material and position
    for y in range(ROWS):
        for x in range(COLS):
            piece = board_state[y][x]
            if not piece:
                continue
            val = PIECE_VALUES.get(piece['type'], 0)
            pstv = PST[(piece['type'], piece['side'])][y][x]
            side_mult = 1 if piece['side'] == 'black' else -1
            
            # Double value for soldiers across river
            if piece['type'] in ['兵', '卒']:
                if (piece['side'] == 'red' and y >= 5) or (piece['side'] == 'black' and y <= 4):
                    val *= 2
            
            score += (val + pstv) * side_mult
    
    # Advisor and elephant integrity
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
    
    # King safety
    red_king = find_general(board_state, 'red')
    black_king = find_general(board_state, 'black')
    if red_king:
        score -= 20 * (4 - red_advisors - red_elephants)
    if black_king:
        score += 20 * (4 - black_advisors - black_elephants)
    
    # Check penalties
    if in_check(board_state, 'red'):
        score += 200
    if in_check(board_state, 'black'):
        score -= 200
    
    return score

def is_forcing_move(move, board_state):
    """Check if move is forcing (capture or check)"""
    # Check for capture
    if board_state[move['to']['y']][move['to']['x']] is not None:
        return True
    
    # Check for check
    side = board_state[move['from']['y']][move['from']['x']]['side']
    opponent = 'red' if side == 'black' else 'black'
    
    newb = copy_board(board_state)
    newb[move['to']['y']][move['to']['x']] = newb[move['from']['y']][move['from']['x']]
    newb[move['from']['y']][move['from']['x']] = None
    return in_check(newb, opponent)

def quiescence(board_state, alpha, beta, side_to_move):
    """Quiescence search for captures and checks"""
    stand_pat = evaluate_board(board_state)
    
    if side_to_move == 'black':
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
    else:
        if stand_pat <= alpha:
            return alpha
        if beta > stand_pat:
            beta = stand_pat
    
    moves = generate_moves(board_state, side_to_move)
    forcing_moves = [m for m in moves if is_forcing_move(m, board_state)]
    forcing_moves.sort(key=lambda m: m['score'], reverse=True)
    
    for m in forcing_moves:
        newb = copy_board(board_state)
        newb[m['to']['y']][m['to']['x']] = newb[m['from']['y']][m['from']['x']]
        newb[m['from']['y']][m['from']['x']] = None
        
        val = quiescence(newb, alpha, beta, 'red' if side_to_move == 'black' else 'black')
        
        if side_to_move == 'black':
            if val >= beta:
                return beta
            if val > alpha:
                alpha = val
        else:
            if val <= alpha:
                return alpha
            if val < beta:
                beta = val
                
    return alpha if side_to_move == 'black' else beta

def negamax(board_state, depth, alpha, beta, side_to_move, current_depth, start_time=None, time_limit=None):
    """Negamax search with alpha-beta pruning"""
    # Time check
    if start_time and time_limit and (time.time() - start_time) > time_limit:
        raise TimeoutError
    
    if depth == 0:
        return quiescence(board_state, alpha, beta, side_to_move)
    
    # Transposition table lookup
    zob = compute_zobrist(board_state, side_to_move)
    tt_entry = TT.get(zob)
    
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
    
    # Move generation
    tt_best_move = tt_entry.get('best_move') if tt_entry else None
    moves = generate_moves(board_state, side_to_move, tt_best_move, current_depth)
    
    best_val = -MATE_SCORE if side_to_move == 'black' else MATE_SCORE
    best_move = None
    
    if not moves:
        if in_check(board_state, side_to_move):
            return -MATE_SCORE if side_to_move == 'black' else MATE_SCORE
        return 0
    
    original_alpha = alpha
    original_beta = beta
    
    for m in moves:
        newb = copy_board(board_state)
        newb[m['to']['y']][m['to']['x']] = newb[m['from']['y']][m['from']['x']]
        newb[m['from']['y']][m['from']['x']] = None
        
        try:
            val = negamax(newb, depth-1, alpha, beta, 'red' if side_to_move == 'black' else 'black', 
                         current_depth + 1, start_time, time_limit)
        except TimeoutError:
            raise
        
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
            
        # Beta cutoff
        if alpha >= beta:
            # Store killer move
            if board_state[m['to']['y']][m['to']['x']] is None:
                km = KILLER_MOVES[current_depth]
                if m != km[0]:
                    km[1] = km[0]
                    km[0] = m
            break
    
    # Store in transposition table
    flag = 'EXACT'
    if side_to_move == 'black':
        if best_val <= original_alpha:
            flag = 'UPPER'
        elif best_val >= original_beta:
            flag = 'LOWER'
    else:
        if best_val >= original_beta:
            flag = 'LOWER'
        elif best_val <= original_alpha:
            flag = 'UPPER'
            
    TT[zob] = {'value': best_val, 'depth': depth, 'flag': flag, 'best_move': best_move}
    return best_val

def minimax_root(board_state, side, time_limit=TIME_LIMIT):
    """Root search with iterative deepening"""
    # Opening book
    if board_state == INITIAL_SETUP and side == 'red':
        return random.choice(FIRST_MOVES)
    
    moves = generate_moves(board_state, side)
    if not moves:
        return None
    
    start_time = time.time()
    best_move_so_far = random.choice(moves)
    best_val_so_far = 0
    
    # Iterative deepening
    for depth in range(1, MAX_DEPTH + 1):
        log(f"Starting ID search for depth {depth}...")
        
        # Aspiration window
        alpha = best_val_so_far - ASPIRATION_WINDOW_DELTA
        beta = best_val_so_far + ASPIRATION_WINDOW_DELTA
        
        # TT best move ordering
        tt_best_move = TT.get(compute_zobrist(board_state, side), {}).get('best_move')
        if tt_best_move:
            for i, m in enumerate(moves):
                if m['from'] == tt_best_move['from'] and m['to'] == tt_best_move['to']:
                    moves.insert(0, moves.pop(i))
                    break
        
        # Main search loop
        while True:
            try:
                original_alpha = alpha
                original_beta = beta
                
                val = negamax(board_state, depth, alpha, beta, side, 0, start_time, time_limit)
                
                # Check for aspiration window failure
                if side == 'black':
                    if val >= original_beta:
                        log(f"Depth {depth}: Fail High. Re-search with alpha={val}")
                        alpha = val
                        beta = float('inf')
                        continue
                    elif val <= original_alpha:
                        log(f"Depth {depth}: Fail Low. Re-search with beta={val}")
                        alpha = -float('inf')
                        beta = val
                        continue
                else:
                    if val <= original_alpha:
                        log(f"Depth {depth}: Fail High (for black). Re-search with alpha={val}")
                        alpha = -float('inf')
                        beta = val
                        continue
                    elif val >= original_beta:
                        log(f"Depth {depth}: Fail Low (for black). Re-search with beta={val}")
                        alpha = val
                        beta = float('inf')
                        continue
                        
                break
                
            except TimeoutError:
                log(f"Timeout at depth {depth}. Returning move from depth {depth-1}.")
                return best_move_so_far
        
        # Update best move
        tt_entry = TT.get(compute_zobrist(board_state, side))
        if tt_entry and tt_entry['depth'] >= depth:
            best_move_found = tt_entry.get('best_move')
            if best_move_found:
                best_move_so_far = best_move_found
                best_val_so_far = val
                log(f"Depth {depth} completed. Best Score: {val}. Best Move: {best_move_so_far}")
        
        # Time check
        if time.time() - start_time > time_limit * 0.95:
            log(f"Time limit reached after completing depth {depth}. Stopping ID.")
            break
            
    return best_move_so_far

def check_game_over(board_state):
    """Check if game is over"""
    red_general = find_general(board_state, 'red')
    black_general = find_general(board_state, 'black')
    if not red_general:
        return {'game_over': True, 'message': '黑方胜利！'}
    if not black_general:
        return {'game_over': True, 'message': '红方胜利！'}
    return {'game_over': False, 'message': ''}
