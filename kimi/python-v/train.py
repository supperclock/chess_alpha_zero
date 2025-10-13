#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AlphaZero-style 自对弈循环训练脚本（中国象棋）- 带模型评估
"""
import os, time, random, json, torch, torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
import sqlite3, pickle, torch, shutil
from collections import defaultdict, deque
from nn_interface import NN_Interface
from nn_data_representation import board_to_tensor, MOVE_TO_INDEX
from collections import Counter
from ai import (
    make_move, unmake_move, generate_moves,
    check_game_over, copy_board, INITIAL_SETUP
)
from util import * 
from tqdm import tqdm

# ---------------- 超参数 ----------------
MODEL_DIR        = "ckpt"
DB_PATH          = 'chess_games.db'
TOTAL_GAMES      = 50000
GAMES_PER_CYCLE  = 100             # <<-- 建议增加
EPOCHS_PER_CYCLE = 2
MCTS_SIMULS      = 400
C_PUCT           = 2.0
TAU              = 1.0
BATCH_SIZE       = 256
LR               = 5e-5            # <<-- 建议降低
MAX_DB_ROWS      = 200000
MAX_GAME_STEPS   = 200
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 新增：评估阶段超参数 ---
EVALUATE_GAMES   = 40              # 每次评估时，新旧模型对战的局数
EVALUATE_WIN_RATE = 0.55           # 新模型胜率超过此阈值，才能被接受
MCTS_SIMULS_EVAL = 100             # 评估时 MCTS 模拟次数可以少一些，以加快速度

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
    
# ---------------- 自对弈 & 数据存储 (保持不变) ----------------
def self_play_one_game(net):
    board, side, examples, step = copy_board(INITIAL_SETUP), 'red', [], 0
    position_history = Counter()
    position_history[board_to_key(board, side)] += 1
    z = 0.0 # Default to draw

    while True:
        temp = 1.0 if step < 30 else 0.1
        pi, v = mcts_policy(net, board, side, temperature=temp)
        tensor, pi_vec = board_to_tensor(board, side).squeeze(0), torch.zeros(len(MOVE_TO_INDEX))
        for move, prob in pi.items():
            key = (move.fy, move.fx, move.ty, move.tx)
            if key in MOVE_TO_INDEX: pi_vec[MOVE_TO_INDEX[key]] = prob
        examples.append({'tensor': tensor, 'pi': pi_vec, 'side': side})
        
        moves, probs = list(pi.keys()), list(pi.values())
        if not moves: 
            z = -1.0 if side == 'black' else 1.0
            break
        move = random.choices(moves, weights=probs)[0]
        
        make_move(board, move)
        step += 1
        side = 'red' if side == 'black' else 'black'

        game_over = check_game_over(board)
        if game_over['game_over']:
            z = 1.0 if game_over['message'][0] == '黑' else -1.0
            break

        current_key = board_to_key(board, side)
        position_history[current_key] += 1
        if position_history[current_key] >= 3:
            z = 0.0
            break

        if step > MAX_GAME_STEPS:
            z = 0.0
            break
                    
    final_examples = []    
    for ex in examples:
        final_z = 0.0 if z == 0.0 else (z if ex['side'] == 'black' else -z)
        final_examples.append((ex['tensor'], ex['pi'], final_z))
    return final_examples    

def save_examples_to_db(examples):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS self_play_moves (tensor BLOB, pi BLOB, z REAL)")
    
    for tensor, pi, z in examples:
        cur.execute("INSERT INTO self_play_moves (tensor, pi, z) VALUES (?, ?, ?)",
                    (pickle.dumps(tensor), pickle.dumps(pi), z))
    conn.commit()
    
    cur.execute("SELECT COUNT(*) FROM self_play_moves")
    count = cur.fetchone()[0]
    if count > MAX_DB_ROWS:
        num_to_delete = count - MAX_DB_ROWS
        log(f"数据库达到上限 {MAX_DB_ROWS}，正在删除最旧的 {num_to_delete} 条记录...")
        cur.execute(f"DELETE FROM self_play_moves WHERE rowid IN (SELECT rowid FROM self_play_moves ORDER BY rowid ASC LIMIT {num_to_delete})")
        conn.commit()
        log("删除完成。")

    conn.close()
    
class SQLiteChessDataset(IterableDataset):
    def __init__(self, db_path):
        self.db_path = db_path
    def __iter__(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        sql = "SELECT tensor, pi, z FROM self_play_moves WHERE tensor IS NOT NULL ORDER BY RANDOM()"
        cur = conn.execute(sql)
        for row in cur:
            yield (pickle.loads(row['tensor']),
                   pickle.loads(row['pi']),
                   torch.tensor(row['z'], dtype=torch.float32))
        conn.close()

# ---------------- 训练函数 (保持不变) ----------------
def train(net, model_dir):
    log("="*20 + " 训练阶段 " + "="*20)
    dataset = SQLiteChessDataset(DB_PATH)
    with sqlite3.connect(DB_PATH) as conn:
        try:
            total_rows = conn.execute("SELECT COUNT(*) FROM self_play_moves").fetchone()[0]
            if total_rows < BATCH_SIZE * 10:
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
            
    log(f"训练完成。保存挑战者模型到 {os.path.join(model_dir, 'latest.pth')}")
    torch.save(net.model.state_dict(), os.path.join(model_dir, "latest.pth"))


# ---------------- 新增：模型评估函数 ----------------
def evaluate_models(candidate_net, best_net, num_games):
    log("="*20 + " 评估阶段 " + "="*20)
    candidate_wins = 0
    
    for i in tqdm(range(num_games), desc="模型评估对战"):
        # 为了公平，轮流执红
        if i % 2 == 0:
            red_player, black_player = candidate_net, best_net
        else:
            red_player, black_player = best_net, candidate_net
            
        board = copy_board(INITIAL_SETUP)
        side = 'red'
        step = 0
        position_history = Counter()
        position_history[board_to_key(board, side)] += 1
        
        while True:
            current_player = red_player if side == 'red' else black_player
            # 评估时 temperature=0，总是选择最优走法
            pi, _ = mcts_policy(current_player, board, side, simuls=MCTS_SIMULS_EVAL, temperature=0.0)
            
            moves = list(pi.keys())
            if not moves:
                # 当前方无棋可走，判负
                winner = 'black' if side == 'red' else 'red'
                break
            
            # 选择访问次数最多的走法
            best_move = max(pi.items(), key=lambda item: item[1])[0]
            make_move(board, best_move)
            side = 'red' if side == 'black' else 'black'
            step += 1
            
            game_over = check_game_over(board)
            if game_over['game_over']:
                winner = 'black' if game_over['message'][0] == '黑' else 'red'
                break

            current_key = board_to_key(board, side)
            position_history[current_key] += 1
            if position_history[current_key] >= 3 or step > MAX_GAME_STEPS:
                winner = 'draw'
                break
        
        # 计分
        if winner == 'draw':
            candidate_wins += 0.5
        elif (winner == 'red' and red_player == candidate_net) or \
             (winner == 'black' and black_player == candidate_net):
            candidate_wins += 1
            
    return candidate_wins / num_games


# ---------------- 主循环 (核心修改) ----------------
def main():
    latest_path = os.path.join(MODEL_DIR, "latest.pth")
    best_path = os.path.join(MODEL_DIR, "best.pth")

    # --- 初始化模型 ---
    # 如果 best.pth 不存在，说明是第一次运行，将 latest.pth 复制为 best.pth
    if not os.path.exists(best_path) and os.path.exists(latest_path):
        log(f"未找到冠军模型 (best.pth)，将初始模型 {latest_path} 设为冠军。")
        shutil.copyfile(latest_path, best_path)

    # 加载两个模型
    # best_net 用于生成数据，candidate_net 用于训练和挑战
    best_net = NN_Interface(model_path=best_path)
    candidate_net = NN_Interface(model_path=latest_path)
    
    game_num = 0
    # 尝试从数据库获取已有的游戏数量
    try:
        with sqlite3.connect(DB_PATH) as conn:
            game_num = conn.execute("SELECT COUNT(DISTINCT game_id) FROM moves").fetchone()[0] # 假设你有 game_id
    except: # 如果表或列不存在，从0开始
        game_num = 0
        
    while game_num < TOTAL_GAMES:
        # 1. 自对弈阶段：使用 best_net 生成数据
        log("="*20 + f" 自对弈阶段 (使用模型: {best_path}) " + "="*20)
        for i in tqdm(range(GAMES_PER_CYCLE), desc=f"自对弈循环 ({game_num+1}-{game_num+GAMES_PER_CYCLE})"):
            examples_one_game = self_play_one_game(best_net)
            if examples_one_game:
                save_examples_to_db(examples_one_game)
            game_num += 1

        # 2. 训练阶段：训练 candidate_net
        train(candidate_net, MODEL_DIR)

        # 3. 评估阶段：candidate_net 挑战 best_net
        win_rate = evaluate_models(candidate_net, best_net, EVALUATE_GAMES)
        log(f"评估完成。挑战者模型胜率: {win_rate:.2%}")

        if win_rate > EVALUATE_WIN_RATE:
            log(f"挑战成功！新模型成为冠军。胜率 {win_rate:.2%} > {EVALUATE_WIN_RATE:.2%}")
            # 更新 best.pth
            shutil.copyfile(latest_path, best_path)
            # best_net 实例也需要重新加载模型
            best_net = NN_Interface(model_path=best_path)
        else:
            log(f"挑战失败。继续使用旧的冠军模型。胜率 {win_rate:.2%} <= {EVALUATE_WIN_RATE:.2%}")

    log(f"已完成全部 {TOTAL_GAMES} 局对弈训练。")

if __name__ == "__main__":
    main()