#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AlphaZero-style 训练脚本（中国象棋） - 支持多卡并行训练和早停
"""
import os, time, random, json, torch, torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import sqlite3, pickle, torch
from collections import defaultdict, deque, OrderedDict
from nn_interface import NN_Interface
from nn_data_representation import board_to_tensor, MOVE_TO_INDEX
from ai import (
    make_move, unmake_move, generate_moves,
    check_game_over, copy_board
)
from util import * 
from tqdm import tqdm

# ---------------- 超参数 ----------------
MODEL_DIR        = "ckpt"          # 权重保存目录
DB_PATH          = 'chess_games.db' # 数据库文件路径
SELFPLAY_GAMES   = 1         # 总对局数（可 Ctrl-C 随时停）
MCTS_SIMULS      = 400             # 每步 MCTS 模拟次数
C_PUCT           = 2.0
TAU              = 1.0             # 温度，前 30 步用 1.0，之后 0.1
BATCH_SIZE       = 256
LR               = 5e-4
MAX_EPOCHS       = 100             # <<-- 修改：最大训练轮数，早停可能会提前结束
CHECKPOINT_EVERY = 10              # 每 N 盘存一次权重
SAVE_EVERY_N_BATCHES = 1000   
# --- 修改：设备检测，优先使用 CUDA ---
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 新增：早停相关超参数 ---
VALIDATION_SPLIT = 0.1             # 验证集占总数据量的比例 (例如 0.1 表示 10%)
PATIENCE         = 3               # 连续 N 个 epoch 验证损失没有改善就停止

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- MCTS 节点 ----------------
class MCTSNode:
    # (此部分代码未修改，保持原样)
    def __init__(self, board, side, parent=None, prior=0.0):
        self.board   = copy_board(board)
        self.side    = side
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
        for move in legal_moves:
            prior = move_priors.get(move, 0.0)
            self.children[move] = MCTSNode(self.board, 'red' if self.side=='black' else 'black', parent=self, prior=prior)
    def backup(self, v):
        self.N += 1
        self.W += v
        self.Q = self.W / self.N
        if self.parent: self.parent.backup(-v)

# ---------------- MCTS 搜索 ----------------
def mcts_policy(net, board, side, simuls=MCTS_SIMULS, temperature=TAU):
    # (此部分代码未修改，保持原样)
    root = MCTSNode(board, side)
    legal_moves = generate_moves(board, side)
    if not legal_moves: return {}, 0.0
    _, prior_dict = net.predict(board, side)
    root.expand(prior_dict)
    for _ in range(simuls):
        node = root
        move, selected_node = None, node 
        while not selected_node.is_leaf():
            move, selected_node = selected_node.select()
        
        node = selected_node
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
        filtered = {m: move_priors.get(m, 0.) for m in legal}
        total = sum(filtered.values()) or 1.0
        filtered = {m: p/total for m, p in filtered.items()}
        node.expand(filtered)
        node.backup(value)
        if node.parent: unmake_move(node.board, move, captured)
        
    if temperature == 0.0:
        best_move = max(root.children.items(), key=lambda x: x[1].N)[0]
        pi = {move: 1.0 if move == best_move else 0.0 for move in root.children}
    else:
        pi, sum_N = {}, sum(child.N**(1/temperature) for child in root.children.values())
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
    log(f"当前走棋方: {current_player}\n")
def wait_key(): input(">>> 按回车继续下一步...")
def self_play_one_game(net, pause=True):
    from ai import INITIAL_SETUP, make_move, unmake_move, generate_moves, check_game_over
    board, side, examples, step = copy_board(INITIAL_SETUP), 'red', [], 0
    while True:
        log(f"\n======== 第 {step+1} 步 （{side} 方）========")
        print_board_color(board, side)
        pi, v = mcts_policy(net, board, side, temperature=1.0 if step < 30 else 0.1)
        tensor, pi_vec = board_to_tensor(board, side).squeeze(0), torch.zeros(len(MOVE_TO_INDEX))
        for move, prob in pi.items():
            key = (move.fy, move.fx, move.ty, move.tx)
            if key in MOVE_TO_INDEX: pi_vec[MOVE_TO_INDEX[key]] = prob
        examples.append((tensor, pi_vec, side))
        moves, probs = list(pi.keys()), list(pi.values())
        move = random.choices(moves, weights=probs)[0]
        log(f"AI 选择：{move.fy}{move.fx} → {move.ty}{move.tx}")
        captured = make_move(board, move)
        step += 1
        game_over = check_game_over(board)
        if game_over['game_over']:
            print_board_color(board, side)
            log(f"对局结束：{game_over['message']}")
            z = 1.0 if game_over['message'][0] == '黑' else -1.0
            return [(tensor, pi_vec, z if who=='black' else -z) for tensor, pi_vec, who in examples]
        side = 'red' if side == 'black' else 'black'
        if pause: wait_key()

# ---------------- 数据集 (保持不变) ----------------
class SQLiteChessDataset(IterableDataset):
    def __init__(self, db_path, split='train', split_ratio=VALIDATION_SPLIT, shuffle=True):
        self.db_path = db_path
        self.split = split
        self.shuffle = shuffle

        with sqlite3.connect(self.db_path) as conn:
            total_rows = conn.execute("SELECT COUNT(*) FROM moves WHERE tensor IS NOT NULL").fetchone()[0]
        
        self.val_size = int(total_rows * split_ratio)
        self.train_size = total_rows - self.val_size

    def __iter__(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        
        if self.split == 'train':
            sql = f"SELECT tensor, pi, z FROM moves WHERE tensor IS NOT NULL"
            if self.shuffle:
                sql += " ORDER BY RANDOM()"
            sql += f" LIMIT {self.train_size}"
        else: # validation
            sql = f"SELECT tensor, pi, z FROM moves WHERE tensor IS NOT NULL ORDER BY rowid DESC LIMIT {self.val_size}"

        cur = conn.execute(sql)
        for row in cur:
            yield (pickle.loads(row['tensor']),
                   pickle.loads(row['pi']),
                   torch.tensor(row['z'], dtype=torch.float32))
        conn.close()

# ---------------- 训练 (核心修改) ----------------
def train(net, model_dir):
    log("正在创建数据加载器...")
    # 增加 DataLoader workers 数量以更好地利用多 GPU
    num_workers = min(os.cpu_count(), 8) 
    train_dataset = SQLiteChessDataset(DB_PATH, split='train')
    val_dataset = SQLiteChessDataset(DB_PATH, split='val', shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=num_workers, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=num_workers, prefetch_factor=2)

    optimizer = torch.optim.Adam(net.model.parameters(), lr=LR, weight_decay=1e-4)
    loss_v = nn.MSELoss()

    # <<< NEW: Add Learning Rate Scheduler >>>
    # It will anneal the learning rate from LR down to nearly zero over MAX_EPOCHS.
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    log("开始训练循环...")
    for epoch in range(1, MAX_EPOCHS + 1):
        # --- 1. 训练阶段 ---
        net.model.train()
        total_loss_pi, total_loss_v = 0.0, 0.0
        batches = 0
        
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS} [训练]", leave=True, ncols=120, unit="batch")
        for board_tensor, pi_vec, z in train_progress_bar:
            board_tensor, pi_vec, z = board_tensor.to(DEVICE), pi_vec.to(DEVICE), z.to(DEVICE)
            pred_pi, pred_v = net.model(board_tensor)
            
            L_pi = -torch.sum(pi_vec * torch.log_softmax(pred_pi, dim=1), dim=1).mean()
            L_v = loss_v(pred_v.squeeze(), z)
            loss = L_pi + L_v

            optimizer.zero_grad()
            loss.backward()
            # <<< NEW: Add Gradient Clipping >>>
            torch.nn.utils.clip_grad_norm_(net.model.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_loss_pi += L_pi.item()
            total_loss_v += L_v.item()
            batches += 1

            # --- 修改：保存模型时，处理 DataParallel 的情况 ---
            if batches > 0 and batches % SAVE_EVERY_N_BATCHES == 0:
                ckpt_path = os.path.join(model_dir, f"ckpt_epoch_{epoch}_batch_{batches}.pth")
                # 如果是多卡训练，保存 .module.state_dict()
                model_state = net.model.module.state_dict() if isinstance(net.model, nn.DataParallel) else net.model.state_dict()
                torch.save(model_state, ckpt_path)
                torch.save(model_state, os.path.join(model_dir, "latest.pth"))
                train_progress_bar.write(f"  -> Checkpoint saved to {ckpt_path}")

            avg_loss_pi = total_loss_pi / batches
            avg_loss_v = total_loss_v / batches
            # --- 修改：在进度条中增加学习率显示 ---
            current_lr = optimizer.param_groups[0]['lr']
            train_progress_bar.set_postfix(loss_pi=f"{avg_loss_pi:.4f}", loss_v=f"{avg_loss_v:.4f}", lr=f"{current_lr:.1e}")

        # --- 2. 验证阶段 ---
        net.model.eval()
        total_val_loss_pi, total_val_loss_v = 0.0, 0.0
        val_batches = 0
        
        with torch.no_grad():
            val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS} [验证]", leave=True, ncols=120, unit="batch")
            for board_tensor, pi_vec, z in val_progress_bar:
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

        # <<< MODIFIED: Step the scheduler after each epoch >>>
        scheduler.step()
        # --- 3. 早停逻辑判断 ---
        if val_batches > 0:
            avg_val_loss = (total_val_loss_pi + total_val_loss_v) / val_batches
            log(f"Epoch {epoch} Summary: Avg Train Loss(pi/v): {total_loss_pi/batches:.4f}/{total_loss_v/batches:.4f} | Avg Val Loss(pi/v): {total_val_loss_pi/val_batches:.4f}/{total_val_loss_v/val_batches:.4f}")
            
            if avg_val_loss < best_val_loss:
                log(f"  Validation loss improved ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving best model...")
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                # --- 修改：同样处理 DataParallel 的情况 ---
                model_state = net.model.module.state_dict() if isinstance(net.model, nn.DataParallel) else net.model.state_dict()
                torch.save(model_state, os.path.join(model_dir, "best_model.pth"))
            else:
                epochs_no_improve += 1
                log(f"  Validation loss did not improve for {epochs_no_improve} epoch(s). Best was {best_val_loss:.4f}.")

            if epochs_no_improve >= PATIENCE:
                log(f"Early stopping triggered after {PATIENCE} epochs with no improvement.")
                break

# ---------------- 主循环 (核心修改) ----------------
def main():
    # --- 新增：设备检测和模型加载 ---
    log("="*50)
    # 初始化网络接口
    net = NN_Interface() 
    model_path = os.path.join(MODEL_DIR, "latest.pth")

    # 鲁棒地加载权重，处理 DataParallel 保存的权重
    if os.path.exists(model_path):
        try:
            log(f"正在从 {model_path} 加载权重...")
            state_dict = torch.load(model_path, map_location=DEVICE)
            # 检查权重是否由 nn.DataParallel 保存（键以 'module.' 开头）
            if next(iter(state_dict)).startswith('module.'):
                log("  检测到 DataParallel 权重，将自动转换。")
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # 移除 `module.` 前缀
                    new_state_dict[name] = v
                state_dict = new_state_dict
            
            net.model.load_state_dict(state_dict)
            log("权重加载成功。")
        except Exception as e:
            log(f"加载权重失败: {e}。将从头开始训练。")
    else:
        log("未找到现有权重，将从头开始训练。")

    # --- 新增：多卡训练设置 ---
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        log(f"发现 {num_gpus} 个可用的 GPU。")
        if num_gpus > 1:
            log("启用 DataParallel 进行多卡训练。")
            net.model = nn.DataParallel(net.model)
        else:
            log("使用单个 GPU 进行训练。")
    else:
        log("未发现可用 GPU，使用 CPU 进行训练。")
    
    net.model.to(DEVICE)

    # 直接开始训练
    train(net, MODEL_DIR)

    log("训练结束。")

if __name__ == "__main__":
    main()