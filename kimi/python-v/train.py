AlphaZero-style 训练脚本（中国象棋） - 支持多卡并行训练和早停
"""
import os, time, random, json, torch, torch.nn as nn
from ai import (
    make_move, unmake_move, generate_moves,
DB_PATH          = 'chess_games.db' # 数据库文件路径
LR               = 5e-4
MAX_EPOCHS       = 100             # <<-- 修改：最大训练轮数，早停可能会提前结束
CHECKPOINT_EVERY = 10              # 每 N 盘存一次权重
SAVE_EVERY_N_BATCHES = 1000   
VALIDATION_SPLIT = 0.1             # 验证集占总数据量的比例 (例如 0.1 表示 10%)
PATIENCE         = 3               # 连续 N 个 epoch 验证损失没有改善就停止
# ---------------- MCTS 节点 ----------------
class MCTSNode:
    # (此部分代码未修改，保持原样)
    def __init__(self, board, side, parent=None, prior=0.0):
        self.parent  = parent
        self.P       = prior          # 网络给出的先验概率
        self.N      = 0               # 访问次数
        self.W      = 0.0             # 累计价值
        self.Q      = 0.0             # 平均价值
        self.children = {}            # move -> MCTSNode
    def is_leaf(self): return len(self.children) == 0
    def select(self):
        best_score = -float('inf')
        best_move, best_child = None, None
        for move, child in self.children.items():
            score = child.Q + C_PUCT * child.P * (self.N**0.5) / (1 + child.N)
            if score > best_score:
                best_score, best_move, best_child = score, move, child
        return best_move, best_child
    def expand(self, move_priors):
        legal_moves = generate_moves(self.board, self.side)
        if self.parent: self.parent.backup(-v)

# ---------------- MCTS 搜索 ----------------
def mcts_policy(net, board, side, simuls=MCTS_SIMULS, temperature=TAU):
    # (此部分代码未修改，保持原样)
    root = MCTSNode(board, side)
    legal_moves = generate_moves(board, side)
    if not legal_moves: return {}, 0.0
    _, prior_dict = net.predict(board, side)
    root.expand(prior_dict)
        if node.parent:
            captured = make_move(node.board, move)
            game_over = check_game_over(node.board)
            if game_over['game_over']:
                v = 1.0 if game_over['message'][0] == '黑' else -1.0
                node.backup(v)
                unmake_move(node.board, move, captured)
                continue
        else: captured = None
        
        value, move_priors = net.predict(node.board, node.side)
        legal = generate_moves(node.board, node.side)
        node.backup(value)
        for move, child in root.children.items():
            pi[move] = child.N**(1/temperature) / (sum_N or 1.0)
    return pi, root.Q

# -------------- 其他函数 (保持不变) --------------
def print_board_color(board, current_player):
    log("\n   a  b  c  d  e  f  g  h  i  (列)")
    log(" --------------------------")
    for y, row in enumerate(board):
        row_str = f"{y}|"
        for piece in row: row_str += " ・ " if piece is None else (f"\033[1m {piece['type']} \033[0m" if piece['side'] == 'black' else f"\033[91m {piece['type']} \033[0m")
        log(row_str)
    log(" --------------------------")
