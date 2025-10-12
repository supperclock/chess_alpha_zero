#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AlphaZero-style 自对弈循环训练脚本（中国象棋）
"""
import os, time, random, json, torch, torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
import sqlite3, pickle, torch
from collections import defaultdict, deque
from nn_interface import NN_Interface
from nn_data_representation import board_to_tensor, MOVE_TO_INDEX
from collections import Counter # 确保在文件顶部引入
from ai import (
    make_move, unmake_move, generate_moves,
    check_game_over, copy_board, INITIAL_SETUP
)
from util import * 
from tqdm import tqdm

# ---------------- 超参数 (修改和新增) ----------------
MODEL_DIR        = "ckpt"          # 权重保存目录
DB_PATH          = 'chess_games.db' # 数据库文件路径
TOTAL_GAMES      = 50000           # <<-- 修改：整个训练过程要产生的总对局数
GAMES_PER_CYCLE  = 25              # <<-- 修改：每产生 N 盘对局，就启动一次训练
EPOCHS_PER_CYCLE = 2               # <<-- 新增：每次训练时，在当前数据集上迭代的轮数
MCTS_SIMULS      = 400             # 每步 MCTS 模拟次数
C_PUCT           = 2.0
TAU              = 1.0             # 温度，前 30 步用 1.0，之后 0.1
BATCH_SIZE       = 256
LR               = 2e-4
MAX_DB_ROWS      = 200000          # <<-- 新增：数据库中最多保存的局面数量 (防止无限增长)
MAX_GAME_STEPS   = 200             # 新增：一局棋的最大步数
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- MCTS 节点与搜索 (保持不变) ----------------
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

def mcts_policy(net, board, side, simuls=MCTS_SIMULS, temperature=TAU):
    # (此部分代码未修改，保持原样)
    root = MCTSNode(board, side)
    legal_moves = generate_moves(board, side)
    if not legal_moves: return {}, 0.0
    _, prior_dict = net.predict(board, side)
    root.expand(prior_dict)
    for _ in range(simuls):
        node = root
        move = None
        while not node.is_leaf():
            move, node = node.select()
        if node.parent:
            captured = make_move(node.board, move)
        game_over = check_game_over(node.board)
        if game_over['game_over']:
            v = 1.0 if game_over['message'][0] == '黑' else -1.0
            node.backup(v)
            if node.parent: unmake_move(node.board, move, captured)
            continue
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

def board_to_key(board, side):
    """将棋盘和走棋方转换为一个唯一的、可哈希的键"""
    board_tuple = tuple(tuple(str(p) for p in row) for row in board)
    return (board_tuple, side)
# ---------------- 自对弈 & 数据存储 (逻辑微调) ----------------
def self_play_one_game(net):
    # (此函数基本不变，只是去掉了 pause 参数)
    board, side, examples, step = copy_board(INITIAL_SETUP), 'red', [], 0
    position_history = Counter()
    # 记录初始局面
    position_history[board_to_key(board, side)] += 1
    while True:
        # 在自对弈时，温度参数控制探索程度
        temp = 1.0 if step < 30 else 0.1
        pi, v = mcts_policy(net, board, side, temperature=temp)
        
        # 存储训练样本
        tensor, pi_vec = board_to_tensor(board, side).squeeze(0), torch.zeros(len(MOVE_TO_INDEX))
        for move, prob in pi.items():
            key = (move.fy, move.fx, move.ty, move.tx)
            if key in MOVE_TO_INDEX: pi_vec[MOVE_TO_INDEX[key]] = prob
        examples.append({'tensor': tensor, 'pi': pi_vec, 'side': side})
        
        # 从策略中采样走棋
        moves, probs = list(pi.keys()), list(pi.values())
        if not moves: 
            log("没有可移动的棋子，判当前走棋方输。")
            if side == 'black':
                z = -1.0
            else:
                z = 1.0
            break
        move = random.choices(moves, weights=probs)[0]
        
        make_move(board, move)
        step += 1
        side = 'red' if side == 'black' else 'black'
        log(f"第 {step} 步：{move.to_dict()}")

        # --- 游戏结束判断 ---
        
        # 1. 规则杀棋判断
        game_over = check_game_over(board)
        if game_over['game_over']:
            z = 1.0 if game_over['message'][0] == '黑' else -1.0
            log(f"游戏结束：{game_over['message']}")
            break

        # 2. 局面重复判断 (和棋)
        current_key = board_to_key(board, side)
        position_history[current_key] += 1
        if position_history[current_key] >= 3:
            log("局面重复3次，判定为和棋。")
            z = 0.0
            break

        # 3. 步数上限判断 (和棋)
        if step > MAX_GAME_STEPS:
            log(f"对局超过 {MAX_GAME_STEPS} 步，强制判为和棋。")
            z = 0.0
            break
                    
    final_examples = []    
    for ex in examples:
        final_z = 0.0 if z == 0.0 else (z if ex['side'] == 'black' else -z)
        final_examples.append((ex['tensor'], ex['pi'], final_z))
    
    log(f"本局共计 {len(final_examples)} 个训练样本...")
    return final_examples    

# ---------------- 新增：数据库操作函数 ----------------
def save_examples_to_db(examples):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS self_play_moves (tensor BLOB, pi BLOB, z REAL)")
    
    for tensor, pi, z in examples:
        cur.execute("INSERT INTO self_play_moves (tensor, pi, z) VALUES (?, ?, ?)",
                    (pickle.dumps(tensor), pickle.dumps(pi), z))
    conn.commit()
    
    # 数据库裁剪，防止无限增大
    cur.execute("SELECT COUNT(*) FROM self_play_moves")
    count = cur.fetchone()[0]
    if count > MAX_DB_ROWS:
        num_to_delete = count - MAX_DB_ROWS
        log(f"数据库达到上限 {MAX_DB_ROWS}，正在删除最旧的 {num_to_delete} 条记录...")
        # 注意: 'rowid' 是 sqlite 的隐藏自增列
        cur.execute(f"DELETE FROM self_play_moves WHERE rowid IN (SELECT rowid FROM moves ORDER BY rowid ASC LIMIT {num_to_delete})")
        conn.commit()
        log("删除完成。")

    conn.close()
    
# ---------------- 数据集 (简化) ----------------
class SQLiteChessDataset(IterableDataset):
    def __init__(self, db_path):
        self.db_path = db_path

    def __iter__(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # 在自对弈循环中，我们通常在整个数据集上随机训练
        sql = "SELECT tensor, pi, z FROM self_play_moves WHERE tensor IS NOT NULL ORDER BY RANDOM()"
        cur = conn.execute(sql)
        for row in cur:
            yield (pickle.loads(row['tensor']),
                   pickle.loads(row['pi']),
                   torch.tensor(row['z'], dtype=torch.float32))
        conn.close()

# ---------------- 训练函数 (简化，移除早停) ----------------
def train(net, model_dir):
    log("="*20 + " 训练阶段 " + "="*20)
    dataset = SQLiteChessDataset(DB_PATH)
    # 如果数据量太少，可以跳过训练
    with sqlite3.connect(DB_PATH) as conn:
        try:
            total_rows = conn.execute("SELECT COUNT(*) FROM self_play_moves").fetchone()[0]
            if total_rows < BATCH_SIZE * 10: # 至少有10个batch的数据
                log(f"数据量过少 ({total_rows} 条)，跳过本次训练。")
                return
        except sqlite3.OperationalError:
            log("数据库为空，跳过本次训练。")
            return
            
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, prefetch_factor=2)
    optimizer = torch.optim.Adam(net.model.parameters(), lr=LR, weight_decay=1e-4)
    loss_v = nn.MSELoss()

    for epoch in range(1, EPOCHS_PER_CYCLE + 1):
        net.model.train()
        total_loss_pi, total_loss_v = 0.0, 0.0
        batches = 0
        
        progress_bar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS_PER_CYCLE}", leave=True, ncols=120, unit="batch")
        for board_tensor, pi_vec, z in progress_bar:
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
            if batches == 0: continue
            avg_loss_pi = total_loss_pi / batches
            avg_loss_v = total_loss_v / batches
            progress_bar.set_postfix(loss_pi=f"{avg_loss_pi:.4f}", loss_v=f"{avg_loss_v:.4f}")
            
    log(f"训练完成。保存最新模型到 {os.path.join(model_dir, 'latest.pth')}")
    torch.save(net.model.state_dict(), os.path.join(model_dir, "latest.pth"))

# ---------------- 主循环 (核心修改) ----------------
def main():
    # 初始化或加载最新模型
    net = NN_Interface(model_path=os.path.join(MODEL_DIR, "latest.pth"))
    
    game_num = 0
    while game_num < TOTAL_GAMES:
        log("="*20 + " 自对弈阶段 " + "="*20)
        
        # 1. 自对弈生成数据        
        for i in tqdm(range(GAMES_PER_CYCLE), desc=f"自对弈循环 ({game_num+1}-{game_num+GAMES_PER_CYCLE})"):
            log(f"开始第 {game_num+1} 局对弈...")
            examples_one_game = self_play_one_game(net)            
            save_examples_to_db(examples_one_game)     
            log(f"保存 {len(examples_one_game)} 个训练样本到数据库中...")     
            game_num += 1                

        # 3. 训练网络
        # NN_Interface 会在每次使用时自动加载最新的 'latest.pth'
        # 所以我们只需要调用 train 函数即可
        train(net, MODEL_DIR)

    log(f"已完成全部 {TOTAL_GAMES} 局对弈训练。")

if __name__ == "__main__":
    main()