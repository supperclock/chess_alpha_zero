"""
共享游戏工具模块 - 包含象棋游戏规则和通用函数
"""

import numpy as np
from collections import namedtuple

# 常量定义
BOARD_ROWS, BOARD_COLS = 10, 9
PIECE_TYPES = ['r','h','c','e','a','g','s']  # 棋子类型：车、马、炮、象、士、将、兵
ACTION_SPACE_SIZE = BOARD_ROWS * BOARD_COLS * BOARD_ROWS * BOARD_COLS

# 数据转换函数
def pos_to_idx(r, c):
    """将位置(r, c)映射到1D索引0-89"""
    return r * BOARD_COLS + c

def move_to_idx(move):
    """将走法((r0, c0), (r1, c1))映射到1D索引0-8099"""
    (r0, c0), (r1, c1) = move
    from_idx = pos_to_idx(r0, c0)
    to_idx = pos_to_idx(r1, c1)
    return from_idx * BOARD_ROWS * BOARD_COLS + to_idx

def initial_board():
    """初始化棋盘"""
    b = [['.' for _ in range(BOARD_COLS)] for __ in range(BOARD_ROWS)]
    # 黑方（上方，小写）
    b[0] = list("rheagaehr")
    b[2][1] = 'c'; b[2][7] = 'c'
    b[3][0] = 's'; b[3][2] = 's'; b[3][4] = 's'; b[3][6] = 's'; b[3][8] = 's'
    # 红方（下方，大写）镜像位置
    b[9] = [c.upper() for c in b[0]]
    b[7][1] = 'C'; b[7][7] = 'C'
    b[6][0] = 'S'; b[6][2] = 'S'; b[6][4] = 'S'; b[6][6] = 'S'; b[6][8] = 'S'
    return tuple(tuple(row) for row in b)  # 返回不可变元组

def print_board(board):
    """打印棋盘"""
    for r in range(BOARD_ROWS):
        print(''.join(board[r]))
    print()

def in_bounds(r, c):
    """检查位置是否在棋盘范围内"""
    return 0 <= r < BOARD_ROWS and 0 <= c < BOARD_COLS

def is_red(piece):
    """检查棋子是否为红方"""
    return piece != '.' and piece.isupper()

def is_black(piece):
    """检查棋子是否为黑方"""
    return piece != '.' and piece.islower()

def same_side(p1, p2):
    """检查两个棋子是否为同一方"""
    if p1 == '.' or p2 == '.': 
        return False
    return (p1.isupper() and p2.isupper()) or (p1.islower() and p2.islower())

def find_general(board, side):
    """查找指定方的将/帅位置"""
    piece = 'G' if side == 'red' else 'g'
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            if board[r][c] == piece:
                return r, c
    return None  # 将/帅被吃

def apply_move(board, move):
    """应用走法，返回新棋盘和吃子信息"""
    (r0, c0), (r1, c1) = move
    # 将元组转换为列表进行修改
    b2 = [list(row) for row in board]
    moved = b2[r0][c0]
    captured = b2[r1][c1]
    b2[r1][c1] = moved
    b2[r0][c0] = '.'
    return tuple(tuple(row) for row in b2), captured  # 返回不可变元组

def generate_piece_moves(board, r, c):
    """生成指定位置棋子的所有可能走法"""
    p = board[r][c]
    moves = []
    if p == '.': 
        return moves
    isRed = p.isupper()
    side = 'red' if isRed else 'black'
    
    # 车/車
    if p.lower() == 'r':  
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r+dr, c+dc
            while in_bounds(nr, nc):
                if board[nr][nc] == '.':
                    moves.append(((r,c),(nr,nc)))
                else:
                    if not same_side(p, board[nr][nc]):
                        moves.append(((r,c),(nr,nc)))
                    break
                nr += dr; nc += dc
                
    # 马/馬
    elif p.lower() == 'h':  
        horse_dirs = [(-2,-1),(-2,1),(2,-1),(2,1),(-1,-2),(1,-2),(-1,2),(1,2)]
        for dr, dc in horse_dirs:
            if abs(dr) == 2:
                leg_r, leg_c = r + (dr//2), c
            else:
                leg_r, leg_c = r, c + (dc//2)
            if not in_bounds(leg_r, leg_c) or board[leg_r][leg_c] != '.': 
                continue
            nr, nc = r+dr, c+dc
            if not in_bounds(nr, nc): 
                continue
            if board[nr][nc] == '.' or not same_side(p, board[nr][nc]):
                moves.append(((r,c),(nr,nc)))
                
    # 炮
    elif p.lower() == 'c':  
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r+dr, c+dc
            # 非吃子滑动走法
            while in_bounds(nr, nc) and board[nr][nc] == '.':
                moves.append(((r,c),(nr,nc)))
                nr += dr; nc += dc
            # 吃子
            if in_bounds(nr, nc) and board[nr][nc] != '.':
                nr += dr; nc += dc
                while in_bounds(nr, nc):
                    if board[nr][nc] != '.':
                        if not same_side(p, board[nr][nc]):
                            moves.append(((r,c),(nr,nc)))
                        break
                    nr += dr; nc += dc
                    
    # 象/相
    elif p.lower() == 'e':  
        deltas = [(-2,-2),(-2,2),(2,-2),(2,2)]
        for dr, dc in deltas:
            nr, nc = r+dr, c+dc
            eye_r, eye_c = r+dr//2, c+dc//2
            if not in_bounds(nr, nc): 
                continue
            if p.islower() and nr > 4: 
                continue  # 黑象不能过河
            if p.isupper() and nr < 5: 
                continue  # 红象不能过河
            if board[eye_r][eye_c] != '.': 
                continue  # 象眼被塞
            if board[nr][nc] == '.' or not same_side(p, board[nr][nc]):
                moves.append(((r,c),(nr,nc)))
                
    # 士/仕
    elif p.lower() == 'a':  
        deltas = [(-1,-1),(-1,1),(1,-1),(1,1)]
        for dr, dc in deltas:
            nr, nc = r+dr, c+dc
            if not in_bounds(nr, nc): 
                continue
            if p.islower():
                if not (0 <= nr <= 2 and 3 <= nc <= 5): 
                    continue
            else:
                if not (7 <= nr <= 9 and 3 <= nc <= 5): 
                    continue
            if board[nr][nc] == '.' or not same_side(p, board[nr][nc]):
                moves.append(((r,c),(nr,nc)))
                
    # 将/帅
    elif p.lower() == 'g':  
        # 标准走法
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r+dr, c+dc
            if not in_bounds(nr, nc): 
                continue
            if p.islower():
                if not (0 <= nr <= 2 and 3 <= nc <= 5): 
                    continue
            else:
                if not (7 <= nr <= 9 and 3 <= nc <= 5): 
                    continue
            if board[nr][nc] == '.' or not same_side(p, board[nr][nc]):
                moves.append(((r,c),(nr,nc)))
        # 飞将
        other_piece = 'G' if p.islower() else 'g'
        for rr in range(BOARD_ROWS):
            if rr == r: 
                continue
            if board[rr][c] == other_piece:
                blocked = False
                for mid in range(min(r,rr)+1, max(r,rr)):
                    if board[mid][c] != '.':
                        blocked = True; break
                if not blocked:
                    moves.append(((r,c),(rr,c)))
                    
    # 兵/卒
    elif p.lower() == 's':  
        dirs = []
        if p.isupper():  # 红方向上移动
            dirs.append((-1,0))
            if r <= 4:  # 过河后可以横向移动
                dirs += [(0,-1),(0,1)]
        else:  # 黑方向下移动
            dirs.append((1,0))
            if r >= 5:  # 过河后可以横向移动
                dirs += [(0,-1),(0,1)]
        for dr, dc in dirs:
            nr, nc = r+dr, c+dc
            if not in_bounds(nr, nc): 
                continue
            if board[nr][nc] == '.' or not same_side(p, board[nr][nc]):
                moves.append(((r,c),(nr,nc)))
    
    return moves

def generate_all_possible_moves(board, side):
    """生成指定方的所有可能走法（不检查将军状态）"""
    moves = []
    is_upper = side == 'red'
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            p = board[r][c]
            if p == '.': 
                continue
            if is_upper != p.isupper(): 
                continue
            moves.extend(generate_piece_moves(board, r, c))
    return moves

def is_in_check(board, side):
    """检查指定方的将/帅是否被将军"""
    gen_pos = find_general(board, side)
    if gen_pos is None: 
        return True  # 将/帅被吃，肯定被将军

    opp_side = 'red' if side == 'black' else 'black'
    # 生成对方的所有走法
    opp_moves = generate_all_possible_moves(board, opp_side)
    
    # 检查是否有对方走法攻击将/帅位置
    for move in opp_moves:
        (_, _), (tr, tc) = move
        if (tr, tc) == gen_pos:
            return True
    return False

def generate_legal_moves(board, side):
    """生成指定方的所有合法走法（过滤掉导致被将军的走法）"""
    all_moves = generate_all_possible_moves(board, side)
    legal_moves = []
    
    for move in all_moves:
        new_board, _ = apply_move(board, move)
        if not is_in_check(new_board, side):
            legal_moves.append(move)
            
    return legal_moves

def is_terminal(board, side):
    """
    检查是否为终局状态
    返回 (True/False, 结果)
    结果: +1 红方胜, -1 黑方胜, 0 平局/继续
    """
    # 1. 检查将/帅是否被吃
    red_gen_pos = find_general(board, 'red')
    black_gen_pos = find_general(board, 'black')
    
    if red_gen_pos is None: 
        return True, -1.0  # 黑方胜
    if black_gen_pos is None: 
        return True, 1.0   # 红方胜

    # 2. 检查无子可动（困毙）
    legal = generate_legal_moves(board, side)
    if len(legal) == 0:
        # 当前方无合法走法 -> 当前方败
        return True, (-1.0 if side == 'red' else 1.0)
        
    return False, 0.0

def board_to_tensor(board, side_to_move):
    """将棋盘转换为(15, 10, 9)的numpy数组"""
    planes = np.zeros((15, BOARD_ROWS, BOARD_COLS), dtype=np.float32)
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            p = board[r][c]
            if p == '.': 
                continue
            idx = PIECE_TYPES.index(p.lower())
            # 平面0-6: 黑方（小写）；平面7-13: 红方（大写）
            plane = idx + (0 if p.islower() else 7)
            planes[plane, r, c] = 1.0
    # 平面14: 当前行棋方（1.0为红方，0.0为黑方）
    planes[14,:,:] = 1.0 if side_to_move == 'red' else 0.0
    return planes

# 数据转换函数
Transition = namedtuple('Transition', ['state','pi','value'])

