#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版 AlphaZero-style 训练脚本（针对“击败固定 AI”目标的改造）
- 保存策略：优先保存模型赢棋的数据；输棋数据可选择丢弃或降低权重
- 根节点加入 Dirichlet 噪声以提高探索性
- 数据库采用滑动窗口（保留最近 N 条记录）以避免旧数据拖累
- 训练时根据胜负信号放大 value 的学习率（reward scaling）
- 每代输出对固定 AI 的胜率统计，便于追踪进展

说明：本文件基于你现有的 train.py 做最小侵入式改造，保持原有流程不变，
但在关键位置增强以提升对固定 AI 学习的效果。

请在运行前确保你的 NN_Interface 与 ai_bridge 等接口兼容。
"""

import os, time, random, json, torch, torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
import sqlite3, pickle
from collections import defaultdict, deque
from nn_interface import NN_Interface
from nn_data_representation import board_to_tensor, MOVE_TO_INDEX
from ai import (
    make_move, unmake_move, generate_moves,
    check_game_over, copy_board, INITIAL_SETUP, Move
)
from util import *
from tqdm import tqdm
import math
import numpy as np

# ---------------- 超参数（增强） ----------------
MODEL_DIR        = "ckpt"
DB_PATH          = 'chess_games.db'
MCTS_SIMULS      = 800
C_PUCT           = 2.0
TAU              = 1.0
BATCH_SIZE       = 256
LR               = 5e-4
MAX_EPOCHS       = 100
CHECKPOINT_EVERY = 10
SAVE_EVERY_N_BATCHES = 1000
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 早停
VALIDATION_SPLIT = 0.1
PATIENCE         = 3

# 强化学习循环
NUM_GENERATIONS      = 100
GAMES_PER_GENERATION = 20

# === 针对固定AI学习的新增参数 ===
SAVE_ONLY_WIN_GAMES = False   # 只保存模型赢棋的(s, pi, z)
LOSS_EXAMPLE_WEIGHT = 1    # 当不保存输局时无效；若保存则权重
PRUNE_DB_MAX_ROWS   = 50000  # 数据库中保留的最大 self_play_moves 行数，超过则删除最旧

# MCTS 根节点探索噪声（Dirichlet）
DIRICHLET_EPSILON = 0.25
DIRICHLET_ALPHA   = 0.3

# 记录胜率计算窗口
WINRATE_WINDOW = GAMES_PER_GENERATION

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- 数据库初始化 ----------------
def setup_database():
    log(f"正在初始化数据库... {DB_PATH}")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS self_play_moves (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tensor BLOB,
            pi BLOB,
            z REAL,
            weight REAL DEFAULT 1.0
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS self_play_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            steps INTEGER,
            winner TEXT,
            update_time TEXT
        )
        """)
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM self_play_moves").fetchone()[0]
        log(f"数据库初始化完毕。当前总数据量: {count} 条")

# ---------------- MCTS 节点（保持） ----------------
class MCTSNode:
    def __init__(self, board, side, parent=None, prior=0.0):
        self.board   = copy_board(board)
        self.side    = side
        self.parent  = parent
        self.P       = prior
        self.N       = 0
        self.W       = 0.0
        self.Q       = 0.0
        self.children = {}
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
        for move in legal_moves:
            prior = move_priors.get(move, 0.0)
            self.children[move] = MCTSNode(self.board, 'red' if self.side=='black' else 'black', parent=self, prior=prior)
    def backup(self, v):
        self.N += 1
        self.W += v
        self.Q = self.W / self.N
        if self.parent: self.parent.backup(-v)

# ---------------- MCTS 搜索（加入 Dirichlet 噪声） ----------------
def mcts_policy(net, board, side, simuls=MCTS_SIMULS, temperature=TAU, add_root_noise=True):
    root = MCTSNode(board, side)
    legal_moves = generate_moves(board, side)
    if not legal_moves: return {}, 0.0

    # 使用网络预测先验
    _, prior_dict = net.predict(board, side)

    # 只保留合法走法的先验
    filtered = {m: prior_dict.get(m, 0.) for m in legal_moves}
    total = sum(filtered.values()) or 1.0
    filtered = {m: p/total for m, p in filtered.items()}

    # 根节点加入 Dirichlet 噪声以增强探索性（尤其在和固定对手学习时很有用）
    if add_root_noise and DIRICHLET_EPSILON > 0:
        probs = np.array([filtered[m] for m in legal_moves], dtype=float)
        # 避免全零
        if probs.sum() == 0:
            probs = np.ones_like(probs) / len(probs)
        noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(legal_moves))
        mixed = (1 - DIRICHLET_EPSILON) * probs + DIRICHLET_EPSILON * noise
        mixed = mixed / mixed.sum()
        filtered = {m: float(mixed[i]) for i, m in enumerate(legal_moves)}

    root.expand(filtered)

    for _ in range(simuls):
        node = root
        # 向下选择直到叶子
        while not node.is_leaf():
            move, child = node.select()
            if child is None:
                break
            node = child
        # 模拟/扩展
        if node.parent:
            captured = make_move(node.board, move)
            game_over = check_game_over(node.board)
            if game_over['game_over']:
                v = 1.0 if game_over['message'][0] == '黑' else -1.0
                node.backup(v)
                unmake_move(node.board, move, captured)
                continue
        else:
            captured = None

        value, move_priors = net.predict(node.board, node.side)
        legal = generate_moves(node.board, node.side)
        filtered2 = {m: move_priors.get(m, 0.) for m in legal}
        total2 = sum(filtered2.values()) or 1.0
        filtered2 = {m: p/total2 for m, p in filtered2.items()}
        node.expand(filtered2)
        node.backup(value)
        if node.parent: unmake_move(node.board, move, captured)

    if temperature == 0.0:
        best_move = max(root.children.items(), key=lambda x: x[1].N)[0]
        pi = {move: 1.0 if move == best_move else 0.0 for move in root.children}
    else:
        pi, sum_N = {}, sum(child.N**(1/temperature) for child in root.children.values())
        if sum_N == 0:
            num_children = len(root.children)
            if num_children > 0:
                prob = 1.0 / num_children
                pi = {move: prob for move in root.children}
            else:
                return {}, root.Q
        else:
            for move, child in root.children.items():
                pi[move] = child.N**(1/temperature) / sum_N
    return pi, root.Q

# -------------- 打印棋盘 (保持不变) --------------
def print_board_color(board, current_player):
    log("\n   a  b  c  d  e  f  g  h  i  (列)")
    log(" --------------------------")
    for y, row in enumerate(board):
        row_str = f"{y}|"
        for piece in row: row_str += " ・ " if piece is None else (f"\033[1m {piece['type']} \033[0m" if piece['side'] == 'black' else f"\033[91m {piece['type']} \033[0m")
        log(row_str)
    log(" --------------------------")
    log(f"当前走棋方: {current_player}\n")

from ai_bridge import find_best_move_c_for_train

# ---------------- [可替换] 对手接口 ----------------
def get_opponent_move(board, side) -> Move:
    """
    这里仍然使用固定 AI 接口（C 版或其它）。
    如果你将来想用不同强度的固定 AI，可在此处添加参数选择。
    """
    log(f"  (对手) 正在为 {side} 方思考...")
    return find_best_move_c_for_train(board, side)

# ---------------- 对局运行 ----------------
def play_against_opponent(net, model_plays_as='red'):
    board = copy_board(INITIAL_SETUP)
    side = 'red'
    step = 0
    model_examples = []
    opponent_moves_record = []

    def end_game(winner, loser, reason):
        log(f"【对局结束】{reason}")
        # 为了让模型从输棋中吸取教训，我们对每一步的 z 值做“分步衰减/加重”处理：
        # - 对于赢棋：正向信号略有衰减（靠近终局的步作用更大）
        # - 对于输棋：靠近终局的步被赋予更强的负信号（更应该被记住）
        total_steps = len(model_examples) if len(model_examples) > 0 else 1
        final_examples = []

        for idx, (tensor, pi_vec, who_played) in enumerate(model_examples):
            progress = (idx + 1) / total_steps  # 取值 (0,1], 越接近 1 表示越靠近终局
            # 基础最终 z（相对于胜方为 +1）
            game_z = 1.0 if winner == 'black' else -1.0
            # 将 z 映射到相对于当时下子方的值
            base_z = game_z if who_played == 'black' else -game_z

            # 对 z 做衰减/加重：赢时轻微衰减，输时近终局步惩罚更强
            if base_z > 0:
                # 赢局：保持正向信号，但靠近终局的步稍微更重要
                z = float(base_z * (1.0 - 0.15 * (1.0 - progress)))
            else:
                # 输局：越靠近终局，负向信号越强（幂次控制形状）
                z = float(base_z * (progress ** 0.8))

            final_examples.append((tensor, pi_vec, z))

        str_model_win = 'model_win'
        if model_plays_as == winner:
            log("  模型 胜利！")
        else:
            log("  模型 失败！")
            str_model_win = 'model_lose'

        log(f"本局为模型 ({model_plays_as}) 收集到 {len(final_examples)} 条训练数据（经分步加权）。")
        # 记录结果到数据库
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                         INSERT INTO self_play_results (steps,winner,update_time) VALUES (?, ?, datetime('now', 'localtime'))
                         """,
                         (len(final_examples), str_model_win))
        return final_examples

    while True:
        print_board_color(board, side)

        legal_moves = generate_moves(board, side)
        if not legal_moves:
            winner = 'red' if side == 'black' else 'black'
            return end_game(winner, side, f"{side} 方无合法走法，判负。")

        if side == model_plays_as:
            current_temp = TAU if step < 30 else 0.1
            log(f"  (我方 MCTS) 正在为 {side} 方思考... (Temp={current_temp})")
            pi, v = mcts_policy(net, board, side, temperature=current_temp, add_root_noise=True)
            if not pi:
                winner = 'red' if side == 'black' else 'black'
                return end_game(winner, side, f"{side} 方 MCTS 无棋可走。")

            tensor = board_to_tensor(board, side).squeeze(0)
            pi_vec = torch.zeros(len(MOVE_TO_INDEX))
            for m, prob in pi.items():
                key = (m.fy, m.fx, m.ty, m.tx)
                if key in MOVE_TO_INDEX:
                    pi_vec[MOVE_TO_INDEX[key]] = prob
            model_examples.append((tensor, pi_vec, side))

            moves, probs = list(pi.keys()), list(pi.values())
            move = random.choices(moves, weights=probs)[0]
            log(f"  (我方 MCTS) 选择：{move.fy}{move.fx} → {move.ty}{move.tx}")

        else:
            move = get_opponent_move(board, side)
            if move is None:
                winner = 'red' if side == 'black' else 'black'
                return end_game(winner, side, f"{side} 方接口无走法。")
            # 记录对手走法（用于未来分析/对手建模）
            opponent_moves_record.append((copy_board(board), side, move))

        make_move(board, move)
        step += 1
        side = 'red' if side == 'black' else 'black'

        game_over = check_game_over(board)
        if game_over['game_over']:
            print_board_color(board, side)
            winner = 'black' if game_over['message'][0] == '黑' else 'red'
            loser = 'red' if winner == 'black' else 'black'
            return end_game(winner, loser, game_over['message'])

# ---------------- 存储数据（含滑动窗口与输局权重） ----------------
def save_game_to_db(game_data, db_path, save_only_wins=SAVE_ONLY_WIN_GAMES, loss_weight=LOSS_EXAMPLE_WEIGHT):
    if not game_data:
        return

    # game_data 是 (tensor, pi_vec, z) 的列表
    log(f"正在将 {len(game_data)} 条新数据存入数据库 {db_path}...")
    serialized_data = []
    for tensor, pi_vec, z in game_data:
        # 根据 z 判断是否为赢局样本（z 相对于下子方的 +/-1）
        is_win = (z == 1.0)
        if save_only_wins and not is_win:
            # 跳过保存输局样本
            continue
        weight = 1.0 if is_win else loss_weight
        s_tensor = sqlite3.Binary(pickle.dumps(tensor))
        s_pi = sqlite3.Binary(pickle.dumps(pi_vec))
        serialized_data.append((s_tensor, s_pi, float(z), float(weight)))

    if not serialized_data:
        log("没有满足保存条件的新数据（或全部为输局且配置为不保存）。")
        return

    try:
        with sqlite3.connect(db_path) as conn:
            conn.executemany(
                "INSERT INTO self_play_moves (tensor, pi, z, weight) VALUES (?, ?, ?, ?)",
                serialized_data
            )
            conn.commit()
            # prune
            cur = conn.execute("SELECT COUNT(*) FROM self_play_moves")
            total = cur.fetchone()[0]
            if total > PRUNE_DB_MAX_ROWS:
                to_delete = total - PRUNE_DB_MAX_ROWS
                log(f"数据库行数 {total} 超过上限 {PRUNE_DB_MAX_ROWS}，将删除最旧 {to_delete} 条记录。")
                conn.execute("DELETE FROM self_play_moves WHERE id IN (SELECT id FROM self_play_moves ORDER BY id ASC LIMIT ?)", (to_delete,))
                conn.commit()
            count = conn.execute("SELECT COUNT(*) FROM self_play_moves").fetchone()[0]
            log(f"数据保存完毕。数据库中数据总量: {count} 条。")
    except Exception as e:
        log(f"[!!!] 数据库保存失败: {e}")

# ---------------- 数据集（带权重采样） ----------------
class SQLiteChessDataset(IterableDataset):
    def __init__(self, db_path, split='train', split_ratio=VALIDATION_SPLIT, shuffle=True):
        self.db_path = db_path
        self.split = split
        self.shuffle = shuffle
        try:
            with sqlite3.connect(self.db_path) as conn:
                total_rows = conn.execute("SELECT COUNT(*) FROM self_play_moves WHERE tensor IS NOT NULL").fetchone()[0]
        except sqlite3.Error as e:
            log(f"数据库错误: {e}. 假设总行数为 0.")
            total_rows = 0

        if total_rows == 0:
            log("警告：数据库为空，无法加载数据。")
            self.val_size = 0
            self.train_size = 0
        else:
            self.val_size = int(total_rows * split_ratio)
            self.train_size = total_rows - self.val_size
            if self.val_size == 0 and total_rows > BATCH_SIZE:
                self.val_size = BATCH_SIZE
                self.train_size = max(0, total_rows - self.val_size)

    def __len__(self):
        if self.split == 'train':
            return (self.train_size + BATCH_SIZE - 1) // BATCH_SIZE
        else:
            return (self.val_size + BATCH_SIZE - 1) // BATCH_SIZE

    def __iter__(self):
        if self.train_size == 0 and self.val_size == 0:
            return iter([])
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        if self.split == 'train':
            # 按时间倒序取最新训练样本
            sql = f"SELECT tensor, pi, z, weight FROM (SELECT * FROM self_play_moves WHERE tensor IS NOT NULL ORDER BY id DESC LIMIT {self.train_size}) AS T"
            if self.shuffle:
                sql += " ORDER BY RANDOM()"
        else:
            sql = f"SELECT tensor, pi, z, weight FROM self_play_moves WHERE tensor IS NOT NULL ORDER BY id DESC LIMIT {self.val_size} OFFSET {self.train_size}"
        cur = conn.execute(sql)
        for row in cur:
            weight_val = float(row['weight']) if 'weight' in row.keys() else 1.0
            yield (pickle.loads(row['tensor']),
                   pickle.loads(row['pi']),
                   torch.tensor(row['z'], dtype=torch.float32),
                   weight_val)
        conn.close()

# ---------------- 训练（使用样本权重 & reward scaling） ----------------
def train(net, model_dir):
    log("正在创建数据加载器...")
    train_dataset = SQLiteChessDataset(DB_PATH, split='train')
    val_dataset = SQLiteChessDataset(DB_PATH, split='val', shuffle=False)

    if train_dataset.train_size == 0:
        log("没有可用的训练数据。跳过训练。")
        return

    # DataLoader expects each yield to be consistent; we will collate manually in loop
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, prefetch_factor=2)

    optimizer = torch.optim.Adam(net.model.parameters(), lr=LR, weight_decay=1e-4)
    loss_v = nn.MSELoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0

    best_loss_path = os.path.join(model_dir, "best_val_loss.txt")
    if os.path.exists(best_loss_path):
        try:
            with open(best_loss_path, 'r') as f:
                best_val_loss = float(f.read())
                log(f"已加载上次的最佳验证损失: {best_val_loss:.4f}")
        except:
            pass

    log("开始训练循环...")
    for epoch in range(1, MAX_EPOCHS + 1):
        net.model.train()
        total_loss_pi, total_loss_v = 0.0, 0.0
        batches = 0

        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS} [训练]", leave=True, ncols=120, unit="batch", total=len(train_dataset))
        for item in train_progress_bar:
            # item 是 (tensor, pi, z, weight)
            board_tensor, pi_vec, z, weight = item
            board_tensor, pi_vec, z = board_tensor.to(DEVICE), pi_vec.to(DEVICE), z.to(DEVICE)
            weight = torch.tensor([weight], device=DEVICE)

            pred_pi, pred_v = net.model(board_tensor)
            # 原始逐样本策略损失（未平均）
            L_pi_elem = -torch.sum(pi_vec * torch.log_softmax(pred_pi, dim=1), dim=1)
            # 当样本为输局时，放大策略损失以强制模型更积极地纠正错误
            # z 的形状应与 L_pi_elem 对齐；(z < 0).float() 会产生掩码
            loss_mask = (z < 0).float()
            # 对于输局样本，策略损失额外放大 50%
            L_pi = L_pi_elem * (1.0 + 0.5 * loss_mask)
            # 将样本权重（scalar 或可广播）应用到每个样本
            L_pi = (L_pi * weight).mean()

            # reward scaling：放大小样本中胜负信号的影响
            # 统计当前 batch 中 z 的绝对值平均
            z_abs_mean = z.abs().mean().item()
            reward_scale = 2.0 if z_abs_mean < 0.3 else 1.0
            L_v = reward_scale * loss_v(pred_v.squeeze(), z)
            loss = L_pi + L_v

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_pi += L_pi.item()
            total_loss_v += L_v.item()
            batches += 1

            if batches > 0 and batches % SAVE_EVERY_N_BATCHES == 0:
                ckpt_path = os.path.join(model_dir, f"ckpt_epoch_{epoch}_batch_{batches}.pth")
                torch.save(net.model.state_dict(), ckpt_path)
                torch.save(net.model.state_dict(), os.path.join(model_dir, "latest.pth"))
                train_progress_bar.write(f"  -> Checkpoint saved to {ckpt_path}")

            avg_loss_pi = total_loss_pi / batches
            avg_loss_v = total_loss_v / batches
            train_progress_bar.set_postfix(loss_pi=f"{avg_loss_pi:.4f}", loss_v=f"{avg_loss_v:.4f}")

        if batches == 0:
            log("训练阶段没有数据，跳过。")
            continue

        # 验证阶段
        net.model.eval()
        total_val_loss_pi, total_val_loss_v = 0.0, 0.0
        val_batches = 0

        if val_dataset.val_size > 0:
            with torch.no_grad():
                val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS} [验证]", leave=True, ncols=120, unit="batch", total=len(val_dataset))
                for item in val_progress_bar:
                    board_tensor, pi_vec, z, weight = item
                    board_tensor, pi_vec, z = board_tensor.to(DEVICE), pi_vec.to(DEVICE), z.to(DEVICE)
                    pred_pi, pred_v = net.model(board_tensor)
                    L_pi = -torch.sum(pi_vec * torch.log_softmax(pred_pi, dim=1), dim=1).mean()
                    L_v = loss_v(pred_v.squeeze(), z)
                    total_val_loss_pi += L_pi.item()
                    total_val_loss_v += L_v.item()
                    val_batches += 1
                    avg_val_loss_pi = total_val_loss_pi / val_batches
                    avg_val_loss_v = total_val_loss_v / val_batches
                    val_progress_bar.set_postfix(val_loss_pi=f"{avg_val_loss_pi:.4f}", val_loss_v=f"{avg_val_loss_v:.4f}")

        if val_batches == 0:
            log("警告：没有验证数据。将使用训练损失来保存模型。")
            avg_train_loss = (total_loss_pi + total_loss_v) / batches
            log(f"Epoch {epoch} Summary: Avg Train Loss(pi/v): {total_loss_pi/batches:.4f}/{total_loss_v/batches:.4f}")
            log(f"  (无验证集) 保存 latest.pth 和 best_model.pth")
            torch.save(net.model.state_dict(), os.path.join(model_dir, "latest.pth"))
            torch.save(net.model.state_dict(), os.path.join(model_dir, "best_model.pth"))
            continue

        avg_val_loss = (total_val_loss_pi + total_val_loss_v) / val_batches
        log(f"Epoch {epoch} Summary: Avg Train Loss(pi/v): {total_loss_pi/batches:.4f}/{total_loss_v/batches:.4f} | Avg Val Loss(pi/v): {total_val_loss_pi/val_batches:.4f}/{total_val_loss_v/val_batches:.4f}")

        torch.save(net.model.state_dict(), os.path.join(model_dir, "latest.pth"))

        if avg_val_loss < best_val_loss:
            log(f"  Validation loss improved ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving best model...")
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(net.model.state_dict(), os.path.join(model_dir, "best_model.pth"))
            with open(best_loss_path, 'w') as f:
                f.write(str(best_val_loss))
        else:
            epochs_no_improve += 1
            log(f"  Validation loss did not improve for {epochs_no_improve} epoch(s). Best was {best_val_loss:.4f}.")

        if epochs_no_improve >= PATIENCE:
            log(f"Early stopping triggered after {PATIENCE} epochs with no improvement.")
            break

# ---------------- 统计最近胜率 ----------------
def compute_recent_winrate(n=WINRATE_WINDOW):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT winner FROM self_play_results ORDER BY id DESC LIMIT ?", (n,))
        rows = cur.fetchall()
        if not rows:
            return None
        wins = sum(1 for r in rows if r[0] == 'model_win')
        return wins / len(rows)

# ---------------- 主循环（保持逻辑，但每代输出胜率并可调整策略） ----------------
def main():
    setup_database()

    model_path = os.path.join(MODEL_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, "latest.pth")
        if not os.path.exists(model_path):
            log("未找到任何模型权重。将从头开始训练 (如果 NN_Interface 支持)。")
            model_path = None
        else:
            log(f"未找到 'best_model.pth'，将从 '{model_path}' 开始。")
    else:
        log(f"将从 '{model_path}' 开始。")

    for gen in range(1, NUM_GENERATIONS + 1):
        log(f"\n" + "="*60)
        log(f"===== 开始第 {gen} / {NUM_GENERATIONS} 代强化学习 =====")
        log("="*60)

        log(f"正在加载模型: {model_path or '新模型'}")
        net = NN_Interface(model_path=model_path)
        net.model.to(DEVICE)

        log(f"--- [第 {gen} 代] 对弈阶段 (共 {GAMES_PER_GENERATION} 盘) ---")
        all_new_game_data = []

        for i in range(GAMES_PER_GENERATION):
            model_side = 'red' if i % 2 == 0 else 'black'
            log(f"\n--- 开始第 {i+1} / {GAMES_PER_GENERATION} 盘 (模型执 {model_side}) ---")
            try:
                game_data = play_against_opponent(net, model_plays_as=model_side)
                # 根据配置保存数据
                save_game_to_db(game_data, DB_PATH, save_only_wins=SAVE_ONLY_WIN_GAMES, loss_weight=LOSS_EXAMPLE_WEIGHT)
                all_new_game_data.extend(game_data)
                log(f"对局 {i+1} 结束。目前共收集到 {len(all_new_game_data)} 条新数据（包含未保存的输局）。")
            except Exception as e:
                log(f"[!!!] 对局 {i+1} 发生严重错误: {e}")
                import traceback
                traceback.print_exc()

        # 每代结束后，统计最近胜率
        winrate = compute_recent_winrate(WINRATE_WINDOW)
        if winrate is not None:
            log(f"最近 {WINRATE_WINDOW} 盘模型对固定 AI 的胜率: {winrate*100:.2f}%")

        log(f"\n--- [第 {gen} 代] 训练阶段 ---")
        train(net, MODEL_DIR)

        model_path = os.path.join(MODEL_DIR, "best_model.pth")
        log(f"第 {gen} 代训练完成。下一代将使用 '{model_path}' 进行对弈。")

    log("="*60)
    log(f"所有 {NUM_GENERATIONS} 代强化学习已完成。")
    log("="*60)

if __name__ == "__main__":
    main()
