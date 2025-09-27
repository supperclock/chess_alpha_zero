"""
数据生成器 - 用于生成AlphaZero象棋训练数据
包含MCTS算法和自对弈功能
"""

import argparse
import os
import random
import math
import json
from collections import defaultdict
import numpy as np

from game_utils import *

# ----------------------------- 传统评估函数 -----------------------------
def get_piece_value(piece):
    """根据棋子类型返回分值"""
    # 假设piece为字符串，如 'rK'（红将）、'bC'（黑车）等
    piece_type = piece[1] if len(piece) == 2 else piece[-1]
    values = {'K': 1000, 'A': 2, 'B': 2, 'N': 4, 'R': 9, 'C': 5, 'P': 1}
    return values.get(piece_type, 0)

def get_piece_side(piece):
    """返回棋子所属方"""
    # 假设piece为字符串，首字母'r'为红，'b'为黑
    if piece[0] == 'r':
        return 'red'
    elif piece[0] == 'b':
        return 'black'
    return None

def evaluate_board(board, side):
    """简单评估函数：统计棋子价值总和"""
    value = 0
    for row in board:
        for piece in row:
            if piece is not None:
                v = get_piece_value(piece)
                if get_piece_side(piece) == side:
                    value += v
                else:
                    value -= v
    return value / 100.0  # 归一化

# ----------------------------- MCTS -----------------------------
class MCTSNode:
    def __init__(self, board, side):
        self.board = board  # 不可变元组，可用于字典键
        self.side = side
        self.children = {}  # move -> node
        self.P = {}  # move -> prior prob
        self.N = defaultdict(int)  # 访问次数
        self.W = defaultdict(float)  # 总动作价值
        self.Q = defaultdict(float)  # 平均动作价值 (W/N)
        self.is_expanded = False

class MCTS:
    def __init__(self, net=None, sims=100, c_puct=1.0, device='cpu'):
        self.net = None  # 禁用神经网络
        self.sims = sims
        self.c_puct = c_puct
        self.device = device
        self.node_cache = {}

    def get_or_create_node(self, board, side):
        key = (board, side)
        if key not in self.node_cache:
            self.node_cache[key] = MCTSNode(board, side)
        return self.node_cache[key]

    def policy_value(self, board, side):
        """返回 move -> prior 字典和 value（传统评估）"""
        legal = generate_legal_moves(board, side)
        if len(legal) == 0:
            return {}, 0.0
        priors = {m: 1.0/len(legal) for m in legal}  # 均匀分布
        value = evaluate_board(board, side)  # 传统评估
        return priors, value

    def run(self, root_board, root_side):
        # 为新搜索清除缓存
        self.node_cache = {} 
        root = self.get_or_create_node(root_board, root_side)
        
        if not root.is_expanded:
            priors, _ = self.policy_value(root_board, root_side)
            root.P = priors
            root.is_expanded = True
        
        for _ in range(self.sims):
            node = root
            path = []
            
            # 选择
            while node.is_expanded and len(node.P) > 0:
                total_N = sum(node.N[m] for m in node.P)
                # 使用小epsilon防止ZeroDivisionError当total_N=0时
                total_N_sqrt = math.sqrt(total_N + 1)
                
                best_move = None; best_ucb = -float('inf')
                
                for m, p in node.P.items():
                    N_m = node.N[m]
                    Q_m = node.Q[m]
                    # PUCT公式: Q(m) + c_puct * P(m) * sqrt(Total_N) / (1 + N(m))
                    u = self.c_puct * p * total_N_sqrt / (1 + N_m)
                    uc = Q_m + u
                    if uc > best_ucb:
                        best_ucb = uc; best_move = m
                        
                if best_move is None: 
                    break  # 无走法，如果展开检查正确则不应发生
                
                path.append((node, best_move))
                
                # 下降或展开
                if best_move in node.children:
                    node = node.children[best_move]
                else:
                    # 创建子节点，检查终局，展开/评估
                    child_board, _ = apply_move(node.board, best_move)
                    child_side = 'red' if node.side == 'black' else 'black'
                    child_node = self.get_or_create_node(child_board, child_side)
                    
                    # 链接
                    node.children[best_move] = child_node
                    node = child_node
                    break  # 转到展开/回传
                    
            # 展开和评估
            terminal, z = is_terminal(node.board, node.side)
            if terminal:
                # 从*当前*节点方的视角看价值：
                # 如果当前方获胜则为+1，如果当前方失败则为-1
                value = z if node.side == 'red' else -z 
            else:
                priors, value = self.policy_value(node.board, node.side)
                node.P = priors
                node.is_expanded = True
                
            # 回传
            for parent, move in reversed(path):
                # 更新统计
                parent.N[move] += 1
                parent.W[move] += value
                parent.Q[move] = parent.W[move] / parent.N[move]
                
                # 为父节点视角翻转价值：如果子节点对其方是失败(-1)，
                # 则对父节点方是胜利(+1)
                value = -value 
                
        # 返回根节点走法的访问次数作为策略
        pi = {m: root.N[m] for m in root.P if root.N[m] > 0} 
        return pi

    def select_move(self, board, side, temperature=1e-3):
        pi = self.run(board, side)
        if not pi:
            return None, {}
        moves = list(pi.keys()); counts = np.array([pi[m] for m in moves], dtype=np.float64)
        
        # 处理数值稳定性边界情况
        if temperature <= 0 or temperature < 1e-5:
            # 选择最大值
            idx = np.argmax(counts)
            total = counts.sum()
            if total > 0:
                return moves[idx], {m: counts[i]/total for i, m in enumerate(moves)}
            else:
                return moves[idx], {m: 1.0/len(moves) for m in moves}
        
        # 限制温度防止溢出
        temperature = max(temperature, 1e-5)
        temperature = min(temperature, 10.0)  # 防止极值
        
        # 数值稳定的温度应用
        try:
            # 添加小epsilon防止log(0)
            counts = np.maximum(counts, 1e-10)
            log_counts = np.log(counts)
            scaled_log = log_counts / temperature
            
            # 通过减去最大值防止溢出
            max_scaled = np.max(scaled_log)
            exp_scaled = np.exp(scaled_log - max_scaled)
            probs = exp_scaled / np.sum(exp_scaled)
            
            # 确保概率是有限的
            if not np.all(np.isfinite(probs)):
                # 回退到均匀分布
                probs = np.ones_like(counts) / len(counts)
            
            m = random.choices(moves, weights=probs.tolist(), k=1)[0]
            return m, {mm: float(p) for mm, p in zip(moves, probs)}
        except (OverflowError, ValueError, ZeroDivisionError):
            # 如果出现数值问题则回退到argmax
            idx = np.argmax(counts)
            return moves[idx], {m: counts[i]/counts.sum() for i, m in enumerate(moves)}

# ----------------------------- 自对弈 -----------------------------
def self_play_game(mcts, max_moves=200, temp=1.0):
    """进行一局自对弈游戏"""
    board = initial_board()  # 从不可变棋盘开始
    side = 'red'
    traj = []
    terminal = False
    z = 0.0

    for turn in range(max_moves):
        # 在MCTS中使用可变棋盘，但将不可变棋盘作为状态输入
        move, pi = mcts.select_move(board, side, temperature=temp) 
        
        if move is None:
            terminal, z = is_terminal(board, side)  # 重新检查终局以确保安全
            if not terminal:
                 # 如果select_move返回None，应该是当前方败
                z = -1.0 if side == 'red' else 1.0
            break
            
        # 记录状态和映射到动作空间向量的pi分布
        state = board_to_tensor(board, side)
        pi_vec = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        for m, p in pi.items():
            idx = move_to_idx(m)
            pi_vec[idx] = p
        traj.append(Transition(state, pi_vec, None))
        
        # 应用走法（返回新的不可变棋盘）
        board, captured = apply_move(board, move)
        side = 'red' if side == 'black' else 'black'
        
        # 检查终局
        terminal, ztmp = is_terminal(board, side)
        if terminal:
            z = ztmp
            break

    # 为所有转换分配最终z
    if not terminal:
        z = 0.0  # 如果达到max_moves则为平局
    
    filled = []
    for t in traj:
        # 我们的行棋方平面编码：红方为1.0，黑方为0.0
        side_at_state = 'red' if t.state[14,0,0] == 1.0 else 'black' 
        # 从该状态行棋方的视角看价值
        value = z if side_at_state == 'red' else -z 
        filled.append(Transition(t.state, t.pi, value))
        
    return filled, z

# ----------------------------- 数据I/O -----------------------------
def write_transitions_jsonl(transitions, file_path):
    """将转换数据写入JSONL文件"""
    with open(file_path, 'a', encoding='utf-8') as f:
        for t in transitions:
            rec = {
                'state': t.state.tolist(),
                'pi': t.pi.tolist(),
                'value': float(t.value),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ----------------------------- 数据生成主函数 -----------------------------
def generate_selfplay_data(output_path, model_path=None, num_games=10, sims=20, temp=1.0, device='cpu'):
    """生成自对弈数据（不使用神经网络）"""
    print("Info: 使用传统AI评估，不加载神经网络模型。")
    mcts = MCTS(net=None, sims=sims, c_puct=1.0, device=device)
    total_positions = 0
    for i in range(num_games):
        traj, outcome = self_play_game(mcts, max_moves=200, temp=temp)
        write_transitions_jsonl(traj, output_path)
        total_positions += len(traj)
        print(f"Generated game {i+1}/{num_games}: {len(traj)} positions, outcome {outcome}")
    print(f"Done. Wrote {total_positions} positions to '{output_path}'.")

def main():
    parser = argparse.ArgumentParser(description='AlphaZero象棋数据生成器')
    
    # 通用参数
    parser.add_argument('--device', type=str, default='cpu', 
                        help='设备选择: "cuda" 或 "cpu"')
    parser.add_argument('--model_path', type=str, default='alphazero_xiangqi.pt', 
                        help='神经网络模型路径（用于加载预训练模型）')
    
    # 数据生成参数
    parser.add_argument('--output', type=str, default='selfplay.jsonl', help='输出JSONL文件路径')
    parser.add_argument('--num_games', type=int, default=10, help='生成的自对弈游戏数量')
    parser.add_argument('--sims', type=int, default=20, help='每步MCTS模拟次数')
    parser.add_argument('--temp', type=float, default=1.0, help='走法选择温度')

    args = parser.parse_args()
    
    generate_selfplay_data(
        output_path=args.output,
        model_path=args.model_path,
        num_games=args.num_games,
        sims=args.sims,
        temp=args.temp,
        device=args.device,
    )

if __name__ == '__main__':
    main()

