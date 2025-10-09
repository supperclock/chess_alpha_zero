#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AlphaZero-style 自对弈训练脚本（中国象棋）
"""
import os, time, random, json, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, deque
from nn_interface import NN_Interface
from nn_data_representation import board_to_tensor, MOVE_TO_INDEX
from ai import (
    make_move, unmake_move, generate_moves,
    in_check, find_general, check_game_over, copy_board
)
from util import * 
# ---------------- 超参数 ----------------
MODEL_DIR        = "ckpt"          # 权重保存目录
SELFPLAY_GAMES   = 1         # 总对局数（可 Ctrl-C 随时停）
MCTS_SIMULS      = 400             # 每步 MCTS 模拟次数
C_PUCT           = 2.0
TAU              = 1.0             # 温度，前 30 步用 1.0，之后 0.1
BATCH_SIZE       = 256
LR               = 5e-4
EPOCHS_PER_GAME  = 1
CHECKPOINT_EVERY = 10              # 每 N 盘存一次权重
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- MCTS 节点 ----------------
class MCTSNode:
    def __init__(self, board, side, parent=None, prior=0.0):
        self.board   = copy_board(board)
        self.side    = side
        self.parent  = parent
        self.P       = prior          # 网络给出的先验概率
        self.N      = 0               # 访问次数
        self.W      = 0.0             # 累计价值
        self.Q      = 0.0             # 平均价值
        self.children = {}            # move -> MCTSNode

    def is_leaf(self):
        return len(self.children) == 0

    def select(self):
        """UCB 选择，返回 (move, child)"""
        best_score = -float('inf')
        best_move  = None
        best_child = None
        for move, child in self.children.items():
            score = child.Q + C_PUCT * child.P * (self.N**0.5) / (1 + child.N)
            if score > best_score:
                best_score, best_move, best_child = score, move, child
        return best_move, best_child

    def expand(self, move_priors):
        """用网络输出的 (move, prior) 展开子节点"""
        legal_moves = generate_moves(self.board, self.side)
        for move in legal_moves:
            key = (move.fy, move.fx, move.ty, move.tx)
            prior = move_priors.get(move, 0.0)
            self.children[move] = MCTSNode(self.board,
                                         'red' if self.side=='black' else 'black',
                                         parent=self, prior=prior)

    def backup(self, v):
        """反向更新"""
        self.N += 1
        self.W += v
        self.Q = self.W / self.N
        if self.parent:
            self.parent.backup(-v)    # 对手视角相反

# ---------------- MCTS 搜索 ----------------
def mcts_policy(net, board, side, simuls=MCTS_SIMULS, temperature=TAU):
    root = MCTSNode(board, side)
    legal_moves = generate_moves(board, side)
    if not legal_moves:
        return {}, 0.0

    # 网络给出先验概率
    _, prior_dict = net.predict(board, side)

    root.expand(prior_dict)

    for _ in range(simuls):
        node = root
        # 1. 选择
        while not node.is_leaf():
            move, node = node.select()

        # 2. 评估 / 展开
        if node.parent:     # 不是根
            # 先真正走这一步
            captured = make_move(node.board, move)
            game_over = check_game_over(node.board)
            if game_over['game_over']:
                v = 1.0 if game_over['message'][0] == '黑' else -1.0
                node.backup(v)
                unmake_move(node.board, move, captured)
                continue
        else:
            captured = None

        # 网络评估
        value, move_priors = net.predict(node.board, node.side)
        legal = generate_moves(node.board, node.side)
        filtered = {m: move_priors.get(m, 0.) for m in legal}
        total = sum(filtered.values()) or 1.0
        filtered = {m: p/total for m, p in filtered.items()}

        node.expand(filtered)
        node.backup(value)

        if node.parent:
            unmake_move(node.board, move, captured)

    # 输出 π ∝ N^(1/tau)
    if temperature == 0.0:
        # 取 argmax
        best_move = max(root.children.items(), key=lambda x: x[1].N)[0]
        pi = {move: 1.0 if move == best_move else 0.0 for move in root.children}
    else:
        pi = {}
        sum_N = sum(child.N**(1/temperature) for child in root.children.values())
        for move, child in root.children.items():
            pi[move] = child.N**(1/temperature) / (sum_N or 1.0)
    return pi, root.Q

# -------------- 彩色棋盘打印 --------------
def print_board_color(board, current_player):
    """
    按用户提供的 ANSI 风格打印棋盘
    board: 10×9 列表
    current_player: 'red'/'black'
    """
    log("\n   a  b  c  d  e  f  g  h  i  (列)")
    log(" --------------------------")
    for y, row in enumerate(board):
        row_str = f"{y}|"
        for piece in row:
            if piece is None:
                row_str += " ・ "
            else:
                piece_char = piece['type']
                if piece['side'] == 'black':
                    row_str += f"\033[1m {piece_char} \033[0m"
                else:
                    row_str += f"\033[91m {piece_char} \033[0m"
        log(row_str)
    log(" --------------------------")
    log(f"当前走棋方: {current_player}\n")

def wait_key():
    input(">>> 按回车继续下一步...")
# ----------------------------------------------


def self_play_one_game(net, pause=True):
    """
    自对弈一盘，带彩色棋盘打印与人工回车控制
    pause: True=每步等待回车，False=直接往下跑
    """
    from ai import INITIAL_SETUP, make_move, unmake_move, generate_moves, check_game_over
    board = copy_board(INITIAL_SETUP)
    side = 'red'
    examples = []
    step = 0

    while True:
        log(f"\n======== 第 {step+1} 步 （{side} 方）========")
        print_board_color(board, side)

        # MCTS 给出策略 π 与价值
        pi, v = mcts_policy(net, board, side,
                          temperature=1.0 if step < 30 else 0.1)
        tensor = board_to_tensor(board, side).squeeze(0)
        pi_vec = torch.zeros(len(MOVE_TO_INDEX))
        for move, prob in pi.items():
            key = (move.fy, move.fx, move.ty, move.tx)
            if key in MOVE_TO_INDEX:
                pi_vec[MOVE_TO_INDEX[key]] = prob
        examples.append((tensor, pi_vec, side))

        # 按 π 随机落子
        moves, probs = list(pi.keys()), list(pi.values())
        move = random.choices(moves, weights=probs)[0]
        log(f"AI 选择：{move.fy}{move.fx} → {move.ty}{move.tx}")
        captured = make_move(board, move)
        step += 1

        # 胜负判定
        game_over = check_game_over(board)
        if game_over['game_over']:
            print_board_color(board, side)
            log(f"对局结束：{game_over['message']}")
            z = 1.0 if game_over['message'][0] == '黑' else -1.0
            return [(tensor, pi_vec, z if who=='black' else -z)
                    for tensor, pi_vec, who in examples]

        side = 'red' if side == 'black' else 'black'
        if pause:
            wait_key()

# ---------------- 数据集 ----------------
class GameDataSet(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer
    def __len__(self):
        return len(self.buffer)
    def __getitem__(self, idx):
        tensor, pi, z = self.buffer[idx]
        return tensor, pi, torch.tensor(z, dtype=torch.float32)

# ---------------- 训练 ----------------
def train(net, examples):
    dataset   = GameDataSet(examples)
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(net.model.parameters(), lr=LR, weight_decay=1e-4)
    loss_pi   = nn.CrossEntropyLoss()
    loss_v    = nn.MSELoss()

    net.model.train()
    for epoch in range(EPOCHS_PER_GAME):
        tot_pi, tot_v, n = 0.0, 0.0, 0
        for board_tensor, pi_vec, z in loader:
            board_tensor, pi_vec, z = board_tensor.to(DEVICE), pi_vec.to(DEVICE), z.to(DEVICE)

            pred_pi, pred_v = net.model(board_tensor.unsqueeze(1)[:, :-1, :, :])  # 去掉辅助通道
            # 计算损失
            L_pi = -torch.sum(pi_vec * torch.log_softmax(pred_pi, dim=1)) / pi_vec.size(0)
            L_v  = loss_v(pred_v.squeeze(), z)
            loss = L_pi + L_v

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tot_pi += L_pi.item()
            tot_v  += L_v.item()
            n += 1
        log(f"  epoch {epoch+1}/{EPOCHS_PER_GAME}  loss_pi={tot_pi/n:.4f}  loss_v={tot_v/n:.4f}")

# ---------------- 主循环 ----------------
def main():
    net = NN_Interface(model_path=os.path.join(MODEL_DIR, "latest.pth"))
    replay_buffer = deque(maxlen=200000)   # 经验回放缓冲

    for game in range(1, SELFPLAY_GAMES+1):
        log(f"\n===== 自对弈第 {game} 盘 =====")
        examples = self_play_one_game(net, pause=False)
        replay_buffer.extend(examples)
        log(f"  本盘 {len(examples)} 步，缓冲共 {len(replay_buffer)} 步")

        if len(replay_buffer) >= 10000:      # 缓冲够大再训练
            log("  开始训练...")
            train(net, list(replay_buffer))

        if game % CHECKPOINT_EVERY == 0:
            path = os.path.join(MODEL_DIR, f"ckpt_{game}.pth")
            torch.save(net.model.state_dict(), path)
            torch.save(net.model.state_dict(), os.path.join(MODEL_DIR, "latest.pth"))
            log(f"  已保存 {path}")

        # 动态降温
        global TAU
        TAU = max(0.1, 1.0 - game*2/SELFPLAY_GAMES)

if __name__ == "__main__":
    main()