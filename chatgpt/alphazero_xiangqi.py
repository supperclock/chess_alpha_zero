
"""
AlphaZero-like minimal implementation for Xiangqi (Chinese Chess)

Single-file runnable demo. Not production-grade, but complete and end-to-end:
- Xiangqi game logic (move generation, basic rules, terminal detection)
- Small PyTorch residual network (policy + value head)
- MCTS using neural network priors (PUCT)
- Self-play data collection & single mini-training loop example

Usage:
    python alphazero_xiangqi.py --demo

Notes:
- This is a compact educational implementation. For large-scale training you should:
  - Use efficient vectorized data pipelines, GPUs, mixed precision
  - Implement full move legality checks including check/checkmate detection robustly
  - Add model checkpointing, evaluation, hyperparameter tuning
"""

import argparse
import os
import random
import math
import time
import json
from collections import defaultdict, deque, namedtuple
import numpy as np

# Try to import torch; if unavailable, we'll fall back to random play
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception as e:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Will run limited demo without neural MCTS. Error:", e)

# ----------------------------- Xiangqi Game -----------------------------
# Board coordinates: rows 0..9, cols 0..8 (10x9). We'll use 'red' at bottom (rows 7-9 initial)
# Piece encoding: single letters with case for side: upper-case = Red, lower-case = Black
# R/r = chariot(車), H/h = horse(馬), C/c = cannon(炮), E/e = elephant(相/象), A/a = advisor(仕/士), G/g = general(將/帥), S/s = soldier(卒/兵)

def initial_board():
    # Return a 10x9 array of characters or '.' for empty
    b = [['.' for _ in range(9)] for __ in range(10)]
    # Black side (top, lowercase)
    b[0] = list("rheagaehr")  # row 0: chariot, horse, elephant, advisor, general, advisor, elephant, horse, chariot (use 'g' for general? but we'll use 'a' above)
    # adjust because classical Xiangqi: r h e a g a e h r
    b[2][1] = 'c'; b[2][7] = 'c'  # cannons at row2 col1 and col7
    b[3][0] = 's'; b[3][2] = 's'; b[3][4] = 's'; b[3][6] = 's'; b[3][8] = 's'
    # Place Black general at row0 col4 explicitly as 'g'
    b[0][4] = 'g'
    # Place advisors at (0,3) and (0,5), elephants at (0,2) and (0,6), horses at (0,1),(0,7), chariots at (0,0),(0,8)
    b[0][0] = 'r'; b[0][1] = 'h'; b[0][2] = 'e'; b[0][3] = 'a'; b[0][5] = 'a'; b[0][6] = 'e'; b[0][7] = 'h'; b[0][8] = 'r'

    # Red side (bottom, uppercase) mirror positions
    b[9] = [c.upper() for c in b[0]]
    b[7][1] = 'C'; b[7][7] = 'C'
    b[6][0] = 'S'; b[6][2] = 'S'; b[6][4] = 'S'; b[6][6] = 'S'; b[6][8] = 'S'
    b[9][4] = 'G'  # general
    return b

def print_board(board):
    for r in range(10):
        print(''.join(board[r]))
    print()

def in_bounds(r,c):
    return 0 <= r < 10 and 0 <= c < 9

def is_red(piece):
    return piece != '.' and piece.isupper()

def is_black(piece):
    return piece != '.' and piece.islower()

def same_side(p1,p2):
    if p1=='.' or p2=='.': return False
    return (p1.isupper() and p2.isupper()) or (p1.islower() and p2.islower())

# Helpers to find king positions
def find_generals(board):
    pos = {}
    for r in range(10):
        for c in range(9):
            if board[r][c] in ('G','g'):
                pos['red' if board[r][c]=='G' else 'black'] = (r,c)
    return pos

# Move representation: ((r0,c0),(r1,c1))
# Basic move generation (not checking check status). Implements standard Xiangqi moves including cannon screen rule.
def generate_legal_moves(board, side):
    # side: 'red' or 'black'
    moves = []
    for r in range(10):
        for c in range(9):
            p = board[r][c]
            if p == '.': continue
            if side=='red' and not p.isupper(): continue
            if side=='black' and not p.islower(): continue
            moves.extend(generate_piece_moves(board, r, c))
    # filter moves to avoid capturing same-side piece is already done by generate_piece_moves
    moves = filter_moves_no_self_check(board, moves, side)
    return moves

def filter_moves_no_self_check(board, moves, side):
    """过滤掉导致自己将军的走法"""
    filtered = []
    opp = 'red' if side=='black' else 'black'
    for m in moves:
        nb, _ = apply_move(board, m)
        gens = find_generals(nb)
        my_gen_pos = gens.get(side, None)
        if my_gen_pos is None:
            # 自己的将没了，说明非法
            continue
        # 检查对方是否能攻击到我的将
        opp_moves = generate_legal_moves_basic(nb, opp)  # 用不含自检过滤的版本
        attacked = False
        for om in opp_moves:
            (_, _), (tr, tc) = om
            if (tr, tc) == my_gen_pos:
                attacked = True
                break
        if not attacked:
            filtered.append(m)
    return filtered

def generate_legal_moves_basic(board, side):
    """基础版，不做‘走后被将’检测，用于 filter 检查"""
    moves = []
    for r in range(10):
        for c in range(9):
            p = board[r][c]
            if p == '.': continue
            if side=='red' and not p.isupper(): continue
            if side=='black' and not p.islower(): continue
            moves.extend(generate_piece_moves(board, r, c))
    return moves

def generate_piece_moves(board, r, c):
    p = board[r][c]
    moves = []
    if p=='.': return moves
    isRed = p.isupper()
    side = 'red' if isRed else 'black'
    # directions
    if p.lower() == 'r':  # chariot: rook-like
        for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr,nc = r+dr, c+dc
            while in_bounds(nr,nc):
                if board[nr][nc]=='.':
                    moves.append(((r,c),(nr,nc)))
                else:
                    if not same_side(p, board[nr][nc]):
                        moves.append(((r,c),(nr,nc)))
                    break
                nr += dr; nc += dc
    elif p.lower() == 'h':  # horse: L-shape with blocking
        # horse moves with leg-block check
        horse_dirs = [(-2,-1),(-2,1),(2,-1),(2,1),(-1,-2),(1,-2),(-1,2),(1,2)]
        # leg_checks = [(-1,0),(-1,0),(1,0),(1,0),(0,-1,0,),(0,-1),(0,1),(0,1)]
        # we'll check each with corresponding leg
        legs = [(-1,0),(-1,0),(1,0),(1,0),(0,-1),(0,-1),(0,1),(0,1)]
        for (dr,dc),(lr,lc) in zip(horse_dirs, legs):
            lr = r + (lr if abs(dr)==2 else lr)
            lc = c + (lc if abs(dc)==2 else lc)
            # Actually simpler: for each move, leg is the orthogonal neighbor toward that move
            # We'll compute leg as (r + sign(dr), c) if abs(dr)==2 else (r, c+sign(dc))
            if abs(dr)==2:
                leg_r, leg_c = r + (dr//2), c
            else:
                leg_r, leg_c = r, c + (dc//2)
            if not in_bounds(leg_r, leg_c): continue
            if board[leg_r][leg_c] != '.': continue  # blocked
            nr, nc = r+dr, c+dc
            if not in_bounds(nr,nc): continue
            if board[nr][nc]=='.' or not same_side(p, board[nr][nc]):
                moves.append(((r,c),(nr,nc)))
    elif p.lower() == 'c':  # cannon
        # moves like rook when not capturing; to capture must have exactly one piece between
        for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr,nc = r+dr, c+dc
            # non-capture sliding moves
            while in_bounds(nr,nc) and board[nr][nc]=='.':
                moves.append(((r,c),(nr,nc)))
                nr += dr; nc += dc
            # now try to find a capture by jumping one screen
            # First check if there's a screen (piece) at the current position
            if in_bounds(nr,nc) and board[nr][nc] != '.':
                # There's a screen, now look for target beyond it
                nr += dr; nc += dc
                while in_bounds(nr,nc):
                    if board[nr][nc] != '.':
                        # Found a piece at target position
                        if not same_side(p, board[nr][nc]):
                            moves.append(((r,c),(nr,nc)))
                        break
                    nr += dr; nc += dc
    elif p.lower() == 'e':  # elephant/minister (diagonal two, cannot cross river)
        deltas = [(-2,-2),(-2,2),(2,-2),(2,2)]
        for dr,dc in deltas:
            nr, nc = r+dr, c+dc
            eye_r, eye_c = r+dr//2, c+dc//2
            if not in_bounds(nr,nc): continue
            # cannot cross river: black elephants (lowercase) must stay rows 0-4; red rows 5-9
            if p.islower() and nr > 4: continue
            if p.isupper() and nr < 5: continue
            if board[eye_r][eye_c] != '.': continue
            if board[nr][nc]=='.' or not same_side(p, board[nr][nc]):
                moves.append(((r,c),(nr,nc)))
    elif p.lower() == 'a':  # advisor (palace diagonal one)
        deltas = [(-1,-1),(-1,1),(1,-1),(1,1)]
        for dr,dc in deltas:
            nr, nc = r+dr, c+dc
            if not in_bounds(nr,nc): continue
            # palace zone: for black rows 0-2 cols 3-5; for red rows 7-9 cols 3-5
            if p.islower():
                if not (0 <= nr <= 2 and 3 <= nc <= 5): continue
            else:
                if not (7 <= nr <= 9 and 3 <= nc <= 5): continue
            if board[nr][nc]=='.' or not same_side(p, board[nr][nc]):
                moves.append(((r,c),(nr,nc)))
    elif p.lower() == 'g':  # general: one orthogonal within palace; flying general capture included
        for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr,nc = r+dr, c+dc
            if not in_bounds(nr,nc): continue
            if p.islower():
                if not (0 <= nr <= 2 and 3 <= nc <= 5): continue
            else:
                if not (7 <= nr <= 9 and 3 <= nc <= 5): continue
            if board[nr][nc]=='.' or not same_side(p, board[nr][nc]):
                moves.append(((r,c),(nr,nc)))
        # flying general: if same file and no pieces between, they can capture each other
        # find other general along column
        for rr in range(10):
            if rr==r: continue
            if board[rr][c] in ('G','g'):
                # check no pieces between r and rr
                blocked = False
                for mid in range(min(r,rr)+1, max(r,rr)):
                    if board[mid][c] != '.':
                        blocked = True; break
                if not blocked:
                    moves.append(((r,c),(rr,c)))
    elif p.lower() == 's':  # soldier
        dirs = []
        if p.isupper():  # red moves up (decreasing row)
            dirs.append((-1,0))
            if r <= 4:  # crossed river? red starts at rows 6-9; after crossing row<=4 can move sideways
                dirs += [(0,-1),(0,1)]
        else:
            dirs.append((1,0))
            if r >=5:
                dirs += [(0,-1),(0,1)]
        for dr,dc in dirs:
            nr,nc = r+dr, c+dc
            if not in_bounds(nr,nc): continue
            if board[nr][nc]=='.' or not same_side(p, board[nr][nc]):
                moves.append(((r,c),(nr,nc)))
    # filter out moves that land on same side pieces already prevented above
    return moves

def apply_move(board, move):
    # returns new board (deep copy) and capture info
    (r0,c0),(r1,c1) = move
    b2 = [row.copy() for row in board]
    moved = b2[r0][c0]
    captured = b2[r1][c1]
    b2[r1][c1] = moved
    b2[r0][c0] = '.'
    return b2, captured

def is_terminal(board, side):
    # For simplicity: terminal if one general missing or no legal moves for side to move (basic check)
    generals = {'red':False,'black':False}
    for r in range(10):
        for c in range(9):
            if board[r][c]=='G': generals['red']=True
            if board[r][c]=='g': generals['black']=True
    if not generals['red'] or not generals['black']:
        return True, (1 if generals['red'] and not generals['black'] else -1 if generals['black'] and not generals['red'] else 0)
    legal = generate_legal_moves(board, side)
    if len(legal)==0:
        return True, (-1 if side=='red' else 1)
    return False, 0

# ----------------------------- Neural Network -----------------------------
# Small residual conv net that maps board to (policy over moves, value)
# We'll represent board as 14x10x9 tensor: 7 piece types x 2 sides + side-to-move plane (optional)
PIECE_TYPES = ['r','h','c','e','a','g','s']  # base types (lowercase)
def board_to_tensor(board, side_to_move):
    # returns shape (15,10,9)
    planes = np.zeros((15,10,9), dtype=np.float32)
    for r in range(10):
        for c in range(9):
            p = board[r][c]
            if p == '.': continue
            idx = PIECE_TYPES.index(p.lower())
            plane = idx + (0 if p.islower() else 7)  # black side first
            planes[plane, r, c] = 1.0
    # side plane
    planes[14,:,:] = 1.0 if side_to_move=='red' else 0.0
    return planes


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)

class AlphaNet(nn.Module):
    def __init__(self, in_channels=15, channels=64, n_resblocks=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.resblocks = nn.ModuleList([ResidualBlock(channels) for _ in range(n_resblocks)])
        # policy head: we will produce policy over move logits by encoding moves as (from_r, from_c, to_r, to_c) -> 10*9*10*9 ~ 8100
        self.policy_conv = nn.Conv2d(channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32*10*9, 9*10*9*10)  # big but manageable for demo
        # value head
        self.value_conv = nn.Conv2d(channels, 8, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8*10*9, 64)
        self.value_fc2 = nn.Linear(64,1)

    def forward(self, x):
        # x: (B,14,10,9)
        out = F.relu(self.bn(self.conv(x)))
        for r in self.resblocks:
            out = r(out)
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)  # (B, action_space)
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).view(-1)
        return p, v

# ----------------------------- MCTS -----------------------------
# We'll implement a simple PUCT MCTS that queries the neural net for prior probabilities and values.
class MCTSNode:
    def __init__(self, board, side):
        self.board = board
        self.side = side
        self.children = {}  # move -> node
        self.P = {}  # move -> prior prob
        self.N = defaultdict(int)
        self.W = defaultdict(float)
        self.Q = defaultdict(float)
        self.is_expanded = False

class MCTS:
    def __init__(self, net=None, sims=100, c_puct=1.0, device='cpu'):
        self.net = net
        self.sims = sims
        self.c_puct = c_puct
        self.device = device

    def policy_value(self, board, side):
        # returns dict move -> prior, and value
        legal = generate_legal_moves(board, side)
        if len(legal)==0:
            # no moves
            return {}, 0.0
        priors = {m: 1.0/len(legal) for m in legal}  # default uniform
        value = 0.0
        if TORCH_AVAILABLE and self.net is not None:
            # convert to tensor and query net
            x = board_to_tensor(board, side)
            xt = torch.tensor(x).unsqueeze(0).to(self.device)  # (1,14,10,9)
            with torch.no_grad():
                logits, v = self.net(xt)
                logits = logits.cpu().numpy().reshape(-1)
                v = float(v.cpu().numpy()[0])
            # map logits to moves by consistent ordering
            # We'll use ordering (from_r,from_c,to_r,to_c)
            # Build mapping for legal moves and pick softmax over those logits
            idxs = []
            logits_for_moves = []
            for m in legal:
                (r0,c0),(r1,c1) = m
                idx = ((r0*9 + c0) * 10*9) + (r1*9 + c1)
                logits_for_moves.append(logits[idx])
            # softmax
            exps = np.exp(np.array(logits_for_moves) - np.max(logits_for_moves))
            probs = exps / (exps.sum() + 1e-9)
            priors = {m: float(p) for m,p in zip(legal, probs)}
            return priors, v
        return priors, value

    def run(self, root_board, root_side):
        root = MCTSNode(root_board, root_side)
        # initialize root with priors
        priors, _ = self.policy_value(root_board, root_side)
        root.P = priors
        root.is_expanded = True
        for _ in range(self.sims):
            node = root
            path = []
            # selection
            while node.is_expanded and len(node.P)>0:
                # select move maximizing UCB
                total_N = sum(node.N[m] for m in node.P) + 1e-9
                best_move = None; best_ucb = -1e9
                for m,p in node.P.items():
                    u = self.c_puct * p * math.sqrt(total_N) / (1 + node.N[m])
                    q = node.Q[m]
                    uc = q + u
                    if uc > best_ucb:
                        best_ucb = uc; best_move = m
                path.append((node, best_move))
                # descend
                if best_move in node.children:
                    node = node.children[best_move]
                else:
                    # create child with applied move
                    child_board, _ = apply_move(node.board, best_move)
                    child_side = 'red' if node.side=='black' else 'black'
                    node = MCTSNode(child_board, child_side)
                    node.is_expanded = False
                    node_parent = path[-1][0]
                    node_parent.children[best_move] = node
                    break
            # expansion & evaluation
            terminal, z = is_terminal(node.board, node.side)
            if terminal:
                value = z if node.side=='red' else -z
            else:
                priors, value = self.policy_value(node.board, node.side)
                node.P = priors
                node.is_expanded = True
            # backpropagate
            for parent, move in reversed(path):
                parent.N[move] += 1
                parent.W[move] += value
                parent.Q[move] = parent.W[move] / parent.N[move]
                value = -value  # alternate perspective
        # return visit counts for root moves as policy
        pi = {m: root.N[m] for m in root.P}
        return pi

    def select_move(self, board, side, temperature=1e-3):
        pi = self.run(board, side)
        if not pi:
            return None, {}
        moves = list(pi.keys()); counts = np.array([pi[m] for m in moves], dtype=np.float64)
        
        # Handle edge cases for numerical stability
        if temperature <= 0 or temperature < 1e-5:
            # pick max
            idx = np.argmax(counts)
            total = counts.sum()
            if total > 0:
                return moves[idx], {m: counts[i]/total for i,m in enumerate(moves)}
            else:
                return moves[idx], {m: 1.0/len(moves) for m in moves}
        
        # Clamp temperature to prevent overflow
        temperature = max(temperature, 1e-5)
        temperature = min(temperature, 10.0)  # Prevent extreme values
        
        # Apply temperature with numerical stability
        try:
            # Add small epsilon to prevent log(0)
            counts = np.maximum(counts, 1e-10)
            log_counts = np.log(counts)
            scaled_log = log_counts / temperature
            
            # Prevent overflow by subtracting max
            max_scaled = np.max(scaled_log)
            exp_scaled = np.exp(scaled_log - max_scaled)
            probs = exp_scaled / np.sum(exp_scaled)
            
            # Ensure probabilities are finite
            if not np.all(np.isfinite(probs)):
                # Fallback to uniform distribution
                probs = np.ones_like(counts) / len(counts)
            
            m = random.choices(moves, weights=probs.tolist(), k=1)[0]
            return m, {mm: float(p) for mm,p in zip(moves, probs)}
        except (OverflowError, ValueError, ZeroDivisionError):
            # Fallback to argmax if numerical issues occur
            idx = np.argmax(counts)
            return moves[idx], {m: counts[i]/counts.sum() for i,m in enumerate(moves)}

# ----------------------------- Replay Buffer & Self-play -----------------------------
Transition = namedtuple('Transition', ['state','pi','value'])

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buf = deque(maxlen=capacity)
    def push(self, traj):
        # traj is list of Transition
        self.buf.extend(traj)
    def sample(self, batch_size):
        idx = np.random.choice(len(self.buf), batch_size, replace=False)
        batch = [self.buf[i] for i in idx]
        states = np.stack([b.state for b in batch])
        pis = np.stack([b.pi for b in batch])
        values = np.array([b.value for b in batch], dtype=np.float32)
        return states, pis, values
    def __len__(self):
        return len(self.buf)

def self_play_game(mcts, max_moves=200, temp=1.0):
    board = initial_board()
    side = 'red'
    traj = []
    terminal = False
    z = 0.0

    for turn in range(max_moves):
        move, pi = mcts.select_move(board, side, temperature=temp)
        if move is None:
            # no legal move -> terminal; treat as loss for side to move
            terminal = True
            z = -1.0 if side == 'red' else 1.0
            break
        # record state and pi distribution mapped to action-space vector
        state = board_to_tensor(board, side)
        # convert pi dict to action vector of length 9*10*9*10
        action_size = 9*10*9*10
        pi_vec = np.zeros(action_size, dtype=np.float32)
        for m,p in pi.items():
            (r0,c0),(r1,c1) = m
            idx = ((r0*9 + c0) * 10*9) + (r1*9 + c1)
            pi_vec[idx] = p
        traj.append(Transition(state, pi_vec, None))
        board, captured = apply_move(board, move)
        side = 'red' if side == 'black' else 'black'
        # check terminal
        terminal, ztmp = is_terminal(board, side)
        if terminal:
            z = ztmp
            break
    # assign final z for all transitions from perspective of player who moved at that state
    if not terminal:
        z = 0.0
    # z is outcome from perspective of current side? is_terminal earlier returned +1 for red win, -1 for black win
    # We need value for each state from perspective of the side to move at that state.
    # We'll use final outcome z_final = +1 red win, -1 black win
    
    # fill values
    filled = []
    for t in traj:
        # If side at that t.state was 'red' when we recorded side plane=1. We'll interpret value = terminal_outcome for red, -terminal_outcome for black? 
        # Actually terminal_outcome = +1 if red wins. Value from perspective of side_to_move at that state = terminal_outcome if side was red else -terminal_outcome
        side_at_state = 'red' if t.state[14,0,0]==1.0 else 'black'  # our side plane encoding
        value = z if side_at_state == 'red' else -z
        filled.append(Transition(t.state, t.pi, value))
    return filled, z

# ----------------------------- Training -----------------------------
def train_step(net, optimizer, batch, device=None):
    net.train()
    states, pis, values = batch
    # Ensure tensors are created on the same device as the model to avoid CPU/GPU dtype mismatch
    if device is None:
        device = next(net.parameters()).device
    states_t = torch.tensor(states, dtype=torch.float32, device=device)
    pis_t = torch.tensor(pis, dtype=torch.float32, device=device)
    vals_t = torch.tensor(values, dtype=torch.float32, device=device)
    logits, pred_vals = net(states_t)
    # policy loss (cross-entropy with pi distribution treated as soft targets -> use KL or cross-entropy)
    # We'll use mean squared between logits softmax and target pi (simple)
    log_probs = F.log_softmax(logits, dim=1)
    policy_loss = - (pis_t * log_probs).sum(dim=1).mean()
    value_loss = F.mse_loss(pred_vals, vals_t)
    loss = policy_loss + value_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().numpy()), float(policy_loss.detach().cpu().numpy()), float(value_loss.detach().cpu().numpy())

# ----------------------------- Demo / Main -----------------------------
def demo_selfplay_and_train(device='cuda'):
    print("Demo: building network and running a tiny self-play + train cycle...")
    if not TORCH_AVAILABLE:
        print("PyTorch not found. Running a purely random-play demo of Xiangqi move generation.")
        board = initial_board(); print_board(board)
        moves = generate_legal_moves(board, 'red')
        print("Legal moves for red at start:", len(moves))
        return

    net = AlphaNet()
    net.to(device)
    # Attempt to load existing model weights if available
    default_model_path = 'alphazero_xiangqi.pt'
    model_path = os.environ.get('ALPHAXIANGQI_MODEL_PATH', default_model_path)
    if os.path.exists(model_path):
        try:
            state = torch.load(model_path, map_location=device)
            net.load_state_dict(state)
            print(f"Loaded model parameters from '{model_path}'.")
        except Exception as e:
            print(f"Warning: Failed to load model from '{model_path}': {e}")
    mcts = MCTS(net=net, sims=20, c_puct=1.0, device=device)
    buffer = ReplayBuffer(capacity=1000)
    # generate 2 self-play games
    for i in range(2):
        traj, outcome = self_play_game(mcts, max_moves=200, temp=1.0)
        buffer.push(traj)
        print(f"Self-play game {i} produced {len(traj)} positions, outcome {outcome}")
    # train for a handful of steps
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    if len(buffer) >= 4:
        for step in range(10):
            states, pis, vals = buffer.sample(min(4, len(buffer)))
            loss, pl, vl = train_step(net, optimizer, (states, pis, vals))
            print(f"Train step {step}: loss={loss:.4f}, policy_loss={pl:.4f}, value_loss={vl:.4f}")
    # Save model parameters for future runs
    try:
        torch.save(net.state_dict(), model_path)
        print(f"Saved model parameters to '{model_path}'.")
    except Exception as e:
        print(f"Warning: Failed to save model to '{model_path}': {e}")
    print("Demo complete.")

# ----------------------------- Data I/O (JSONL) -----------------------------
def write_transitions_jsonl(transitions, file_path):
    # Each line: {"state": [...], "pi": [...], "value": float}
    with open(file_path, 'a', encoding='utf-8') as f:
        for t in transitions:
            rec = {
                'state': t.state.tolist(),
                'pi': t.pi.tolist(),
                'value': float(t.value),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def read_transitions_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            state = np.array(rec['state'], dtype=np.float32)
            pi = np.array(rec['pi'], dtype=np.float32)
            value = float(rec['value'])
            yield Transition(state, pi, value)

# ----------------------------- Mode 1: Generate Self-play Data -----------------------------
def generate_selfplay_data(output_path, num_games=10, sims=20, temp=1.0, device='cuda'):
    if not TORCH_AVAILABLE:
        print("Warning: PyTorch not available. Generation will use uniform priors (no NN).")
    net = AlphaNet().to(device) if TORCH_AVAILABLE else None
    mcts = MCTS(net=net if TORCH_AVAILABLE else None, sims=sims, c_puct=1.0, device=device)
    total_positions = 0
    for i in range(num_games):
        traj, outcome = self_play_game(mcts, max_moves=200, temp=temp)
        # Fill values already handled in self_play_game
        write_transitions_jsonl(traj, output_path)
        total_positions += len(traj)
        print(f"Generated game {i+1}/{num_games}: {len(traj)} positions, outcome {outcome}")
    print(f"Done. Wrote {total_positions} positions to '{output_path}'.")

# ----------------------------- Mode 2: Train From File -----------------------------
def load_dataset_into_memory(data_path):
    states = []
    pis = []
    vals = []
    for t in read_transitions_jsonl(data_path):
        states.append(t.state)
        pis.append(t.pi)
        vals.append(t.value)
    if not states:
        raise RuntimeError(f"No data found in '{data_path}'.")
    states = np.stack(states)
    pis = np.stack(pis)
    vals = np.array(vals, dtype=np.float32)
    return states, pis, vals

def iterate_minibatches(states, pis, vals, batch_size, shuffle=True):
    n = states.shape[0]
    idxs = np.arange(n)
    if shuffle:
        np.random.shuffle(idxs)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = idxs[start:end]
        yield states[batch_idx], pis[batch_idx], vals[batch_idx]

def train_from_file(data_path, model_path=None, device='cuda', epochs=1, batch_size=64, lr=1e-3):
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Training mode cannot proceed.")
        return
    print(f"Loading dataset from '{data_path}' ...")
    states, pis, vals = load_dataset_into_memory(data_path)
    net = AlphaNet().to(device)
    if model_path and os.path.exists(model_path):
        try:
            state = torch.load(model_path, map_location=device)
            net.load_state_dict(state)
            print(f"Loaded model parameters from '{model_path}'.")
        except Exception as e:
            print(f"Warning: Failed to load model from '{model_path}': {e}")
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    steps = 0
    for ep in range(epochs):
        ep_loss = []
        for b_states, b_pis, b_vals in iterate_minibatches(states, pis, vals, batch_size, shuffle=True):
            loss, pl, vl = train_step(net, optimizer, (b_states, b_pis, b_vals), device=device)
            ep_loss.append(loss)
            steps += 1
        print(f"Epoch {ep+1}/{epochs}: avg_loss={np.mean(ep_loss):.4f}, batches={len(ep_loss)}")
    # Save
    if model_path is None:
        model_path = os.environ.get('ALPHAXIANGQI_MODEL_PATH', 'alphazero_xiangqi.pt')
    try:
        torch.save(net.state_dict(), model_path)
        print(f"Saved model parameters to '{model_path}'.")
    except Exception as e:
        print(f"Warning: Failed to save model to '{model_path}': {e}")

def main():
    parser = argparse.ArgumentParser()
    # Demo (legacy)
    parser.add_argument('--demo', action='store_true', help='Run a short demo')
    parser.add_argument('--model_path', type=str, default=None, help='Path to save/load model parameters (.pt). Overrides ALPHAXIANGQI_MODEL_PATH')
    # Mode 1: generate data
    parser.add_argument('--generate_data', action='store_true', help='Generate self-play data and save to a JSONL text file')
    parser.add_argument('--output', type=str, default='selfplay.jsonl', help='Output JSONL file for generated data')
    parser.add_argument('--num_games', type=int, default=10, help='Number of self-play games to generate')
    parser.add_argument('--sims', type=int, default=20, help='MCTS simulations per move for generation')
    parser.add_argument('--temp', type=float, default=1.0, help='Temperature for move selection during generation')
    # Mode 2: train from file
    parser.add_argument('--train_from_file', action='store_true', help='Train the model using data from a JSONL text file')
    parser.add_argument('--data_path', type=str, default=None, help='Path to JSONL data file for training')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    # Device
    parser.add_argument('--device', type=str, default='cuda', help='Device to use: "cuda" or "cpu"')
    args = parser.parse_args()

    # Device selection
    device = args.device

    if args.demo:
        if args.model_path:
            os.environ['ALPHAXIANGQI_MODEL_PATH'] = args.model_path
        demo_selfplay_and_train(device=device)
        return

    if args.generate_data:
        generate_selfplay_data(
            output_path=args.output,
            num_games=args.num_games,
            sims=args.sims,
            temp=args.temp,
            device=device,
        )
        return

    if args.train_from_file:
        if args.model_path:
            os.environ['ALPHAXIANGQI_MODEL_PATH'] = args.model_path
        if not args.data_path:
            print('Error: --data_path is required when --train_from_file is set')
            return
        train_from_file(
            data_path=args.data_path,
            model_path=args.model_path,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        return

    print("No mode selected. Use one of: --generate_data, --train_from_file, or --demo.")

if __name__ == '__main__':
    main()
