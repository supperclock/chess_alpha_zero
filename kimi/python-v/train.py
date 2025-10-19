#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AlphaZero-style 训练脚本（中国象棋）
将模型与外部接口对弈，并使用对弈数据进行强化学习
"""
import os, time, random, json, torch, torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
import sqlite3, pickle, torch
from collections import defaultdict, deque
from nn_interface import NN_Interface
from nn_data_representation import board_to_tensor, MOVE_TO_INDEX
from ai import (
    make_move, unmake_move, generate_moves,
    check_game_over, copy_board, INITIAL_SETUP, Move
)
from util import * 
from tqdm import tqdm

# ---------------- 超参数 ----------------
MODEL_DIR        = "ckpt"          # 权重保存目录
DB_PATH          = 'chess_games.db' # 数据库文件路径
MCTS_SIMULS      = 800             # 每步 MCTS 模拟次数
C_PUCT           = 2.0
TAU              = 1.0             # 温度，前 30 步用 1.0，之后 0.1
BATCH_SIZE       = 256
LR               = 5e-4
MAX_EPOCHS       = 100             # 每轮训练的最大 epoch 数
CHECKPOINT_EVERY = 10              # (此参数在
SAVE_EVERY_N_BATCHES = 1000   
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 早停相关超参数 ---
VALIDATION_SPLIT = 0.1             # 验证集占总数据量的比例 (例如 0.1 表示 10%)
PATIENCE         = 3               # 连续 N 个 epoch 验证损失没有改善就停止

# --- 新增：强化学习循环超参数 ---
NUM_GENERATIONS      = 100         # 总共进行多少代 "对弈-训练" 循环
GAMES_PER_GENERATION = 20          # 每一代（每轮）对弈多少盘棋
MAX_TRAIN_ROWS       = 500000      # [!!!] 新增：只使用最新的 50 万条数据进行训练

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- 数据库初始化 ----------------
def setup_database():
    """确保数据库和表存在"""
    log(f"正在初始化数据库... {DB_PATH}")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS self_play_moves (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tensor BLOB,
            pi BLOB,
            z REAL
        )
        """)
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM self_play_moves").fetchone()[0]
        log(f"数据库初始化完毕。当前总数据量: {count} 条")
        #创建数据库表，保存模型与接口对弈的结果，字段包括自增ID，走棋步数，胜利方，更新时间
        conn.execute("""
        CREATE TABLE IF NOT EXISTS self_play_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            steps INTEGER,
            winner TEXT,
            update_time TEXT
        )
        """)
        conn.commit()
    

# ---------------- MCTS 节点 ----------------
class MCTSNode:
    # (此部分代码未修改，保持原样)
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
        move, node_child = node.select() # Renamed to avoid confusion
        if node_child is None: # 如果没有可选的子节点 (比如只有一步棋)
            if node.is_leaf():
                pass # 已经是叶子了，啥也不用做
            else:
                # 这种情况理论上不应该在 select 后发生，但作为保险
                continue 
        else:
            node = node_child # 移动到子节点
            
        if node.parent:
            captured = make_move(node.board, move)
            game_over = check_game_over(node.board)
            if game_over['game_over']:
                # 游戏结束，根据黑方胜(1.0)或红方胜(-1.0)来 backup
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
        if sum_N == 0: # 避免除零，如果所有 N 都为 0 (例如模拟次数过少)
            # 均匀分配概率
            num_children = len(root.children)
            if num_children > 0:
                prob = 1.0 / num_children
                pi = {move: prob for move in root.children}
            else:
                return {}, root.Q # 没有合法走法
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
# ---------------- [!!!] 您的走棋接口 (请您实现) ----------------
def get_opponent_move(board, side) -> Move:
    """
    *** 这是您需要实现的部分 ***
    
    调用您的 "自动走棋接口"。
    
    参数:
    - board: 当前棋盘状态 (与 ai.py 中的格式相同)
    - side: 您的接口需要走的棋方 ('red' 或 'black')
    
    返回:
    - move: 一个 Move 对象 (必须包含 fy, fx, ty, tx 属性)
             例如: Move(fy=0, fx=1, ty=2, tx=2)
    """
    
    # 【【【 请在这里替换为您的接口调用 】】】
    
    # --- 示例：使用随机走法作为占位符 ---
    log(f"  (对手) 正在为 {side} 方思考...")
    return find_best_move_c_for_train(board, side)

    # legal_moves = generate_moves(board, side)
    # if not legal_moves:
    #     return None # 游戏已结束或无棋可走
    
    # # 模拟接口思考时间
    # time.sleep(0.1) 
    
    # opponent_move = random.choice(legal_moves)
    # log(f"  (对手) 决定走: {opponent_move.fy}{opponent_move.fx} -> {opponent_move.ty}{opponent_move.tx}")
    # return opponent_move
    # --- 示例结束 ---


def play_against_opponent(net, model_plays_as='red'):
    """
    运行一盘对局：我们的 MCTS 模型 vs 对手接口
    当任意一方无合法走法时自动判负。
    """
    board = copy_board(INITIAL_SETUP)
    side = 'red'
    step = 0
    model_examples = []

    def end_game(winner, loser, reason):
        """统一结束对局逻辑"""
        log(f"【对局结束】{reason}")
        game_z = 1.0 if winner == 'black' else -1.0
        final_examples = [
            (tensor, pi_vec, game_z if who_played == 'black' else -game_z)
            for tensor, pi_vec, who_played in model_examples
        ]
        log(f"  获胜方: {winner}")
        log(f"本局为模型 ({model_plays_as}) 收集到 {len(final_examples)} 条训练数据。")
        #记录结果到数据库
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                         INSERT INTO self_play_results (steps,winner,update_time) VALUES (?, ?, datetime('now', 'localtime'))
                         """, 
                         (len(final_examples), winner))
        return final_examples

    while True:
        print_board_color(board, side)

        # === 1. 检查是否有合法走法 ===
        legal_moves = generate_moves(board, side)
        if not legal_moves:
            winner = 'red' if side == 'black' else 'black'
            return end_game(winner, side, f"{side} 方无合法走法，判负。")

        # === 2. 执行走棋逻辑 ===
        # if side == model_plays_as:
        # 动态设置温度
        current_temp = TAU if step < 30 else 0.1 # <--- 修改点：前30步用TAU (1.0)

        log(f"   正在为 {side} 方思考... (Temp={current_temp})")
        pi, v = mcts_policy(net, board, side, temperature=current_temp) # <--- 修改点
        if not pi:  # 理论上不会发生，但保险
            winner = 'red' if side == 'black' else 'black'
            return end_game(winner, side, f"{side} 方 MCTS 无棋可走。")

        # 存储训练样本
        tensor = board_to_tensor(board, side).squeeze(0)
        pi_vec = torch.zeros(len(MOVE_TO_INDEX))
        for m, prob in pi.items():
            key = (m.fy, m.fx, m.ty, m.tx)
            if key in MOVE_TO_INDEX:
                pi_vec[MOVE_TO_INDEX[key]] = prob
        model_examples.append((tensor, pi_vec, side))

        # 按概率选择走法
        moves, probs = list(pi.keys()), list(pi.values())
        move = random.choices(moves, weights=probs)[0]
        log(f"   选择：{move.fy}{move.fx} → {move.ty}{move.tx}")

        # else:
        #     move = get_opponent_move(board, side)
        #     if move is None:
        #         winner = 'red' if side == 'black' else 'black'
        #         return end_game(winner, side, f"{side} 方接口无走法。")

        # === 3. 执行走法并检测终局 ===
        make_move(board, move)
        step += 1
        side = 'red' if side == 'black' else 'black'

        game_over = check_game_over(board)
        if game_over['game_over']:
            print_board_color(board, side)
            winner = 'black' if game_over['message'][0] == '黑' else 'red'
            loser = 'red' if winner == 'black' else 'black'
            return end_game(winner, loser, game_over['message'])


# ---------------- [新增] 存储数据 ----------------
def save_game_to_db(game_data, db_path):
    """
    将一局游戏产生的所有 (s, pi, z) 数据批量存入数据库
    """
    if not game_data:
        return

    log(f"正在将 {len(game_data)} 条新数据存入数据库 {db_path}...")
    
    # 序列化数据
    serialized_data = []
    for tensor, pi_vec, z in game_data:
        s_tensor = sqlite3.Binary(pickle.dumps(tensor))
        s_pi = sqlite3.Binary(pickle.dumps(pi_vec))
        serialized_data.append((s_tensor, s_pi, z))

    try:
        with sqlite3.connect(db_path) as conn:
            conn.executemany(
                "INSERT INTO self_play_moves (tensor, pi, z) VALUES (?, ?, ?)",
                serialized_data
            )
            conn.commit()
            
        count = conn.execute("SELECT COUNT(*) FROM self_play_moves").fetchone()[0]
        log(f"数据保存完毕。数据库中数据总量: {count} 条。")
        
    except Exception as e:
        log(f"[!!!] 数据库保存失败: {e}")


# ---------------- 数据集 (修改) ----------------
class SQLiteChessDataset(IterableDataset):
    def __init__(self, db_path, split='train', split_ratio=VALIDATION_SPLIT, shuffle=True):
        self.db_path = db_path
        self.split = split
        self.shuffle = shuffle

        # 连接数据库，计算训练集和验证集的大小
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
            # 1. 确定我们最多使用多少数据
            total_usable_rows = min(total_rows, MAX_TRAIN_ROWS)
            log(f"数据库总行数: {total_rows}。将使用最新的 {total_usable_rows} 条数据进行训练/验证。")

            # 2. 从这部分数据中切分验证集
            self.val_size = int(total_usable_rows * split_ratio)
            self.train_size = total_usable_rows - self.val_size
            
            # 3. 确保验证集至少有一批数据（如果数据总量允许）
            if self.val_size == 0 and total_usable_rows > BATCH_SIZE * 2: # 保证训练集也至少有一批
                self.val_size = BATCH_SIZE
                self.train_size = total_usable_rows - self.val_size
            
            # 4. 计算需要跳过多少条 "非常旧" 的数据
            self.offset = max(0, total_rows - total_usable_rows)
            # --- 修改结束 ---

    def __len__(self):
        # 估算长度（用于tqdm）
        if self.split == 'train':
            return (self.train_size + BATCH_SIZE - 1) // BATCH_SIZE
        else:
            return (self.val_size + BATCH_SIZE - 1) // BATCH_SIZE

    def __iter__(self):
        if self.train_size == 0 and self.val_size == 0:
             # 如果没有数据，返回一个空迭代器
            return iter([])
            
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        
        # --- 修改开始：调整 SQL 查询以使用 offset ---
        
        # 我们先构建一个基础查询，它只选择 "滑动窗口" 内的数据
        # (注意：OFFSET 在 LIMIT 之前)
        base_sql = f"FROM self_play_moves WHERE tensor IS NOT NULL ORDER BY id DESC LIMIT {self.train_size + self.val_size} OFFSET {self.offset}"

        if self.split == 'train':
            # 训练集：取窗口中的最新 N 条
            sql = f"SELECT tensor, pi, z FROM (SELECT * {base_sql}) AS T ORDER BY id DESC LIMIT {self.train_size}"
            if self.shuffle:
                # 在窗口数据上随机化
                sql = f"SELECT * FROM ({sql}) AS T_SHUFFLE ORDER BY RANDOM()"
        else: # validation
            # 验证集：取窗口中紧邻训练集的 M 条
            sql = f"SELECT tensor, pi, z FROM (SELECT * {base_sql}) AS T ORDER BY id DESC LIMIT {self.val_size} OFFSET {self.train_size}"
        # --- 修改结束 ---

        cur = conn.execute(sql)
        for row in cur:
            yield (pickle.loads(row['tensor']),
                   pickle.loads(row['pi']),
                   torch.tensor(row['z'], dtype=torch.float32))
        conn.close()

# ---------------- 训练 (核心修改) ----------------
def train(net, model_dir):
    log("正在创建数据加载器...")
    train_dataset = SQLiteChessDataset(DB_PATH, split='train')
    val_dataset = SQLiteChessDataset(DB_PATH, split='val', shuffle=False) # 验证集不需要随机化

    if train_dataset.train_size == 0:
        log("没有可用的训练数据。跳过训练。")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, prefetch_factor=2)

    optimizer = torch.optim.Adam(net.model.parameters(), lr=LR, weight_decay=1e-4)
    loss_v = nn.MSELoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # 尝试加载上一次的最佳损失
    best_loss_path = os.path.join(model_dir, "best_val_loss.txt")
    if os.path.exists(best_loss_path):
        try:
            with open(best_loss_path, 'r') as f:
                best_val_loss = float(f.read())
                log(f"已加载上次的最佳验证损失: {best_val_loss:.4f}")
        except:
            pass # 加载失败则使用 inf

    log("开始训练循环...")
    for epoch in range(1, MAX_EPOCHS + 1):
        # --- 1. 训练阶段 ---
        net.model.train()
        total_loss_pi, total_loss_v = 0.0, 0.0
        batches = 0
        
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS} [训练]", leave=True, ncols=120, unit="batch", total=len(train_dataset))
        for board_tensor, pi_vec, z in train_progress_bar:
            board_tensor, pi_vec, z = board_tensor.to(DEVICE), pi_vec.to(DEVICE), z.to(DEVICE)
            pred_pi, pred_v = net.model(board_tensor)
            
            L_pi = -torch.sum(pi_vec * torch.log_softmax(pred_pi, dim=1), dim=1).mean()
            L_v = loss_v(pred_v.squeeze(), z)
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

        # --- 2. 验证阶段 ---
        net.model.eval()
        total_val_loss_pi, total_val_loss_v = 0.0, 0.0
        val_batches = 0
        
        if val_dataset.val_size > 0:
            with torch.no_grad(): # 关闭梯度计算
                val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS} [验证]", leave=True, ncols=120, unit="batch", total=len(val_dataset))
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
        
        if val_batches == 0:
            log("警告：没有验证数据。将使用训练损失来保存模型。")
            # 如果没有验证集，我们退而求其次，总是保存最新的模型
            avg_train_loss = (total_loss_pi + total_loss_v) / batches
            log(f"Epoch {epoch} Summary: Avg Train Loss(pi/v): {total_loss_pi/batches:.4f}/{total_loss_v/batches:.4f}")
            log(f"  (无验证集) 保存 latest.pth 和 best_model.pth")
            torch.save(net.model.state_dict(), os.path.join(model_dir, "latest.pth"))
            torch.save(net.model.state_dict(), os.path.join(model_dir, "best_model.pth"))
            continue

        # --- 3. 早停逻辑判断 ---
        avg_val_loss = (total_val_loss_pi + total_val_loss_v) / val_batches
        log(f"Epoch {epoch} Summary: Avg Train Loss(pi/v): {total_loss_pi/batches:.4f}/{total_loss_v/batches:.4f} | Avg Val Loss(pi/v): {total_val_loss_pi/val_batches:.4f}/{total_val_loss_v/val_batches:.4f}")
        
        # 总是保存最新的模型
        torch.save(net.model.state_dict(), os.path.join(model_dir, "latest.pth"))

        if avg_val_loss < best_val_loss:
            log(f"  Validation loss improved ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving best model...")
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(net.model.state_dict(), os.path.join(model_dir, "best_model.pth"))
            # 保存最佳损失值
            with open(best_loss_path, 'w') as f:
                f.write(str(best_val_loss))
        else:
            epochs_no_improve += 1
            log(f"  Validation loss did not improve for {epochs_no_improve} epoch(s). Best was {best_val_loss:.4f}.")

        if epochs_no_improve >= PATIENCE:
            log(f"Early stopping triggered after {PATIENCE} epochs with no improvement.")
            break # 退出训练循环

# ---------------- [!!!] 主循环 (已重写) ----------------
def main():
    
    # 1. 初始化数据库
    setup_database()

    # 2. 确定初始模型路径
    # 优先使用 'best_model.pth'，如果不存在，则使用 'latest.pth'
    model_path = os.path.join(MODEL_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, "latest.pth")
        if not os.path.exists(model_path):
            log("未找到任何模型权重。将从头开始训练 (如果 NN_Interface 支持)。")
            model_path = None # NN_Interface 内部会处理 None
        else:
            log(f"未找到 'best_model.pth'，将从 '{model_path}' 开始。")
    else:
        log(f"将从 '{model_path}' 开始。")

    # 3. 开始 "对弈-训练" 强化学习循环
    for gen in range(1, NUM_GENERATIONS + 1):
        log(f"\n" + "="*60)
        log(f"===== 开始第 {gen} / {NUM_GENERATIONS} 代强化学习 =====")
        log("="*60)

        # --- 步骤 1: 加载当前最佳模型 ---
        # NN_Interface 会在内部加载模型
        log(f"正在加载模型: {model_path or '新模型'}")
        net = NN_Interface(model_path=model_path)
        net.model.to(DEVICE) # 确保模型在正确的设备上

        # --- 步骤 2: 对弈 & 收集数据 ---
        log(f"--- [第 {gen} 代] 对弈阶段 (共 {GAMES_PER_GENERATION} 盘) ---")
        all_new_game_data = []
        
        for i in range(GAMES_PER_GENERATION):
            # 交替执红/黑
            model_side = 'red' if i % 2 == 0 else 'black'
            log(f"\n--- 开始第 {i+1} / {GAMES_PER_GENERATION} 盘 (模型执 {model_side}) ---")
            
            try:
                # 运行一盘对局
                game_data = play_against_opponent(net)
                # ---存储新数据 ---
                log(f"\n--- [第 {gen} 代] 存储阶段 ---")
                save_game_to_db(game_data, DB_PATH)
                all_new_game_data.extend(game_data)
                log(f"对局 {i+1} 结束。目前共收集到 {len(all_new_game_data)} 条新数据。")
                
            except Exception as e:
                log(f"[!!!] 对局 {i+1} 发生严重错误: {e}")
                import traceback
                traceback.print_exc()
        
        # --- 步骤 4: 训练模型 ---
        log(f"\n--- [第 {gen} 代] 训练阶段 ---")
        # 'net' 对象包含了已加载的模型，train 函数将在此基础上继续训练
        train(net, MODEL_DIR)
        
        # 训练结束后，'best_model.pth' 会被更新（如果验证损失有改善）
        # 我们将 'model_path' 更新为 'best_model.pth'，供下一代使用
        model_path = os.path.join(MODEL_DIR, "best_model.pth")
        log(f"第 {gen} 代训练完成。下一代将使用 '{model_path}' 进行对弈。")

    log("="*60)
    log(f"所有 {NUM_GENERATIONS} 代强化学习已完成。")
    log("="*60)

if __name__ == "__main__":
    main()