import random
# 统一 piece value 表
PIECE_VALUES_STD = {
  '將': 10000, '帥': 10000,
  '車': 900, '俥': 900, '车': 900,
  '馬': 450, '傌': 450, '马': 450,
  '炮': 400, '砲': 400,
  '相': 200, '象': 200,
  '仕': 200, '士': 200,
  '兵': 100, '卒': 100
}

# 棋盘尺寸
ROWS = 10
COLS = 9


# 新增全局变量：简单的第一步开局库
# 走法以 (from_y, from_x, to_y, to_x) 格式表示
FIRST_MOVES = [
    # 1. 中炮局（炮二平五）
    # 炮从 (y=2, x=1) 移动到 (y=2, x=4)
    {'from': {'y': 2, 'x': 1}, 'to': {'y': 2, 'x': 4}},
    
    # 2. 仙人指路（兵七进一）
    # 兵从 (y=3, x=6) 移动到 (y=4, x=6)
    {'from': {'y': 3, 'x': 6}, 'to': {'y': 4, 'x': 6}},

    # 3. 起马局（马八进七）
    # 马从 (y=0, x=7) 移动到 (y=2, x=6)
    {'from': {'y': 0, 'x': 7}, 'to': {'y': 2, 'x': 6}},

    # 4. 兵三进一
    # 兵从 (y=3, x=2) 移动到 (y=4, x=2)
    {'from': {'y': 3, 'x': 2}, 'to': {'y': 4, 'x': 2}},

    # 5. 起马局（马一平二）
    # 马从 (y=0, x=1) 移动到 (y=2, x=2)
    {'from': {'y': 0, 'x': 1}, 'to': {'y': 2, 'x': 2}},
]


# --- Minimax 核心算法 ---
def minimax_root(board_state, depth, side):
  #根据board_state是初始棋盘和side是红方来决定是否使用开局库
  #如果board_state等于[[{'type': '車', 'side': 'red'}, {'type': '馬', 'side': 'red'}, {'type': '相', 'side': 'red'}, {'type': '仕', 'side': 'red'}, {'type': '帥', 'side': 'red'}, {'type': '仕', 'side': 'red'}, {'type': '相', 'side': 'red'}, {'type': '馬', 'side': 'red'}, {'type': '車', 'side': 'red'}], [None, None, None, None, None, None, None, None, None], [None, {'type': '炮', 'side': 'red'}, None, None, None, None, None, {'type': '炮', 'side': 'red'}, None], [{'type': '兵', 'side': 'red'}, None, {'type': '兵', 'side': 'red'}, None, {'type': '兵', 'side': 'red'}, None, {'type': '兵', 'side': 'red'}, None, {'type': '兵', 'side': 'red'}], [None, None, None, None, None, None, None, None, None], [None, None, None, None, None, None, None, None, None], [{'type': '卒', 'side': 'black'}, None, {'type': '卒', 'side': 'black'}, None, {'type': '卒', 'side': 'black'}, None, {'type': '卒', 'side': 'black'}, None, {'type': '卒', 'side': 'black'}], [None, {'type': '炮', 'side': 'black'}, None, None, None, None, None, {'type': '炮', 'side': 'black'}, None], [None, None, None, None, None, None, None, None, None], [{'type': '車', 'side': 'black'}, {'type': '馬', 'side': 'black'}, {'type': '象', 'side': 'black'}, {'type': '士', 'side': 'black'}, {'type': '將', 'side': 'black'}, {'type': '士', 'side': 'black'}, {'type': '象', 'side': 'black'}, {'type': '馬', 'side': 'black'}, {'type': '車', 'side': 'black'}]]
  #且side等于'red'，则使用开局库
  if board_state == [[{'type': '車', 'side': 'red'}, {'type': '馬', 'side': 'red'}, {'type': '相', 'side': 'red'}, {'type': '仕', 'side': 'red'}, {'type': '帥', 'side': 'red'}, {'type': '仕', 'side': 'red'}, {'type': '相', 'side': 'red'}, {'type': '馬', 'side': 'red'}, {'type': '車', 'side': 'red'}], [None, None, None, None, None, None, None, None, None], [None, {'type': '炮', 'side': 'red'}, None, None, None, None, None, {'type': '炮', 'side': 'red'}, None], [{'type': '兵', 'side': 'red'}, None, {'type': '兵', 'side': 'red'}, None, {'type': '兵', 'side': 'red'}, None, {'type': '兵', 'side': 'red'}, None, {'type': '兵', 'side': 'red'}], [None, None, None, None, None, None, None, None, None], [None, None, None, None, None, None, None, None, None], [{'type': '卒', 'side': 'black'}, None, {'type': '卒', 'side': 'black'}, None, {'type': '卒', 'side': 'black'}, None, {'type': '卒', 'side': 'black'}, None, {'type': '卒', 'side': 'black'}], [None, {'type': '炮', 'side': 'black'}, None, None, None, None, None, {'type': '炮', 'side': 'black'}, None], [None, None, None, None, None, None, None, None, None], [{'type': '車', 'side': 'black'}, {'type': '馬', 'side': 'black'}, {'type': '象', 'side': 'black'}, {'type': '士', 'side': 'black'}, {'type': '將', 'side': 'black'}, {'type': '士', 'side': 'black'}, {'type': '象', 'side': 'black'}, {'type': '馬', 'side': 'black'}, {'type': '車', 'side': 'black'}]] and side == 'red':
    return random.choice(FIRST_MOVES)
  moves = get_all_legal_moves_board(board_state, side)
  if not moves:
    return None
  
  best_val = -float('inf') if side == 'black' else float('inf')
  best_moves = []
  
  for move in moves:
    tmp_board = [row[:] for row in board_state]
    tmp_board[move['to']['y']][move['to']['x']] = tmp_board[move['from']['y']][move['from']['x']]
    tmp_board[move['from']['y']][move['from']['x']] = None
    
    val = minimax(tmp_board, depth - 1, -float('inf'), float('inf'), 'red' if side == 'black' else 'black')
    
    if side == 'black':
        if val > best_val:
            best_val = val
            best_moves = [move]
        elif val == best_val:
            best_moves.append(move)
    else: # red
        if val < best_val:
            best_val = val
            best_moves = [move]
        elif val == best_val:
            best_moves.append(move)
  
  return random.choice(best_moves)

def minimax(board_state, depth, alpha, beta, side_to_move):
  if depth == 0:
    return evaluate_board(board_state)
  
  moves = get_all_legal_moves_board(board_state, side_to_move)
  if not moves:
    if is_in_check_board(board_state, side_to_move):
      return -1000000 if side_to_move == 'black' else 1000000
    else:
      return 0
  
  if side_to_move == 'black':
    value = -float('inf')
    for move in moves:
      tmp_board = [row[:] for row in board_state]
      tmp_board[move['to']['y']][move['to']['x']] = tmp_board[move['from']['y']][move['from']['x']]
      tmp_board[move['from']['y']][move['from']['x']] = None
      
      value = max(value, minimax(tmp_board, depth - 1, alpha, beta, 'red'))
      alpha = max(alpha, value)
      if alpha >= beta:
        break
    return value
  else:  # 'red'
    value = float('inf')
    for move in moves:
      tmp_board = [row[:] for row in board_state]
      tmp_board[move['to']['y']][move['to']['x']] = tmp_board[move['from']['y']][move['from']['x']]
      tmp_board[move['from']['y']][move['from']['x']] = None
      
      value = min(value, minimax(tmp_board, depth - 1, alpha, beta, 'black'))
      beta = min(beta, value)
      if alpha >= beta:
        break
    return value
    
# --- 走法合法性判断 (纯数据版本) ---
def can_move_on(board_state, from_pos, to_pos):
  fx, fy = from_pos['x'], from_pos['y']
  tx, ty = to_pos['x'], to_pos['y']
  
  piece_obj = board_state[fy][fx]
  if not piece_obj:
    return False
  
  name = piece_obj['type']
  side = piece_obj['side']
  
  def in_board_coords(x, y):
    return 0 <= x < COLS and 0 <= y < ROWS
  
  if not in_board_coords(tx, ty):
    return False
  
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

# --- 棋盘评估函数 ---
def evaluate_board(board_state):
  score = 0
  for y in range(ROWS):
    for x in range(COLS):
      piece = board_state[y][x]
      if not piece:
        continue
      
      val = PIECE_VALUES_STD.get(piece['type'], 0)
      side_mult = 1 if piece['side'] == 'black' else -1
      add = val * side_mult
      
      if piece['type'] in ['兵', '卒']:
        is_across = (piece['side'] == 'red' and y >= 5) or (piece['side'] == 'black' and y <= 4)
        if is_across:
          add += 20 * side_mult
      
      if 3 <= x <= 5:
        add += 5 * side_mult
      
      score += add
  
  if is_in_check_board(board_state, 'red'):
    score += 200
  if is_in_check_board(board_state, 'black'):
    score -= 200
    
  return score

def score_move_board(from_pos, to_pos, board_state):
  tmp_board = [row[:] for row in board_state]
  moving_piece = tmp_board[from_pos['y']][from_pos['x']]
  captured_piece = tmp_board[to_pos['y']][to_pos['x']]
  
  base = 0
  if captured_piece:
    base += PIECE_VALUES_STD.get(captured_piece['type'], 0)
  
  tmp_board[to_pos['y']][to_pos['x']] = moving_piece
  tmp_board[from_pos['y']][from_pos['x']] = None
  
  if is_in_check_board(tmp_board, moving_piece['side']):
    return -999999
  
  opponent = 'red' if moving_piece['side'] == 'black' else 'black'
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
  
  return evaluate_board(tmp_board) + check_bonus + base

# --- 辅助函数 ---
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

def get_all_legal_moves_board(board_state, side):
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
            
          tmp_board = [row[:] for row in board_state]
          tmp_board[ty][tx] = tmp_board[y][x]
          tmp_board[y][x] = None
          
          if is_in_check_board(tmp_board, side):
            continue
          
          if is_king_facing_king_board(tmp_board):
            continue
            
          score = score_move_board({'x': x, 'y': y}, {'x': tx, 'y': ty}, board_state)
          moves.append({'from': {'x': x, 'y': y}, 'to': {'x': tx, 'y': ty}, 'score': score})
          
  moves.sort(key=lambda m: m['score'], reverse=True if side == 'black' else False)
  return moves

def check_game_over(board_state):
    red_general = find_general_in_board_state(board_state, 'red')
    black_general = find_general_in_board_state(board_state, 'black')
    
    if not red_general:
        return {'game_over': True, 'message': '黑方胜利！'}
    if not black_general:
        return {'game_over': True, 'message': '红方胜利！'}
    return {'game_over': False, 'message': ''}