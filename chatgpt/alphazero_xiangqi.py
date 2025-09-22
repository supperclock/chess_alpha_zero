
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
import random
import math
import time
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
        leg_checks = [(-1,0),(-1,0),(1,0),(1,0),(0,-1,0,),(0,-1),(0,1),(0,1)]
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
            nr += dr; nc += dc
            while in_bounds(nr,nc):
                if board[nr-dr][nc-dc] == '.':
                    break  # no screen -> cannot capture beyond
                if board[nr][nc] != '.':
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

def is_terminal(board):
    # For simplicity: terminal if one general missing or no legal moves for side to move (basic check)
    generals = {'red':False,'black':False}
    for r in range(10):
        for c in range(9):
            if board[r][c]=='G': generals['red']=True
            if board[r][c]=='g': generals['black']=True
    if not generals['red'] or not generals['black']:
        return True, (1 if generals['red'] and not generals['black'] else -1 if generals['black'] and not generals['red'] else 0)
    return False, 0

# ----------------------------- Neural Network -----------------------------
# Small residual conv net that maps board to (policy over moves, value)
# We'll represent board as 14x10x9 tensor: 7 piece types x 2 sides + side-to-move plane (optional)
PIECE_TYPES = ['r','h','c','e','a','g','s']  # base types (lowercase)
def board_to_tensor(board, side_to_move):
    # returns shape (14,10,9)
    planes = np.zeros((14,10,9), dtype=np.float32)
    for r in range(10):
        for c in range(9):
            p = board[r][c]
            if p == '.': continue
            idx = PIECE_TYPES.index(p.lower())
            plane = idx + (0 if p.islower() else 7)  # black side first
            planes[plane, r, c] = 1.0
    # side plane
    planes[13,:,:] = 1.0 if side_to_move=='red' else 0.0
    return planes

if TORCH_AVAILABLE:
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
        def __init__(self, in_channels=14, channels=64, n_resblocks=3):
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
            terminal, z = is_terminal(node.board)
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
        moves = list(pi.keys()); counts = np.array([pi[m] for m in moves], dtype=np.float32)
        if temperature<=0 or temperature<1e-5:
            # pick max
            idx = np.argmax(counts)
            return moves[idx], {m: counts[i]/counts.sum() for i,m in enumerate(moves)}
        counts = counts ** (1/temperature)
        probs = counts / counts.sum()
        m = random.choices(moves, weights=probs.tolist(), k=1)[0]
        return m, {mm: float(p) for mm,p in zip(moves, probs)}

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
    for turn in range(max_moves):
        move, pi = mcts.select_move(board, side, temperature=temp)
        if move is None:
            # no legal move -> terminal; treat as loss for side to move
            terminal, z = True, -1.0
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
        # check terminal
        terminal, z = is_terminal(board)
        if terminal:
            break
        side = 'red' if side=='black' else 'black'
    # assign final z for all transitions from perspective of player who moved at that state
    if not terminal:
        z = 0.0
    # z is outcome from perspective of current side? is_terminal earlier returned +1 for red win, -1 for black win
    # We need value for each state from perspective of the side to move at that state.
    # We'll use final outcome z_final = +1 red win, -1 black win
    terminal_outcome = z
    # fill values
    filled = []
    for t in traj:
        # If side at that t.state was 'red' when we recorded side plane=1. We'll interpret value = terminal_outcome for red, -terminal_outcome for black? 
        # Actually terminal_outcome = +1 if red wins. Value from perspective of side_to_move at that state = terminal_outcome if side was red else -terminal_outcome
        side_at_state = 'red' if t.state[13,0,0]==1.0 else 'black'  # our side plane encoding
        value = terminal_outcome if side_at_state=='red' else -terminal_outcome
        filled.append(Transition(t.state, t.pi, value))
    return filled, terminal_outcome

# ----------------------------- Training -----------------------------
def train_step(net, optimizer, batch):
    net.train()
    states, pis, values = batch
    states_t = torch.tensor(states, dtype=torch.float32)
    pis_t = torch.tensor(pis, dtype=torch.float32)
    vals_t = torch.tensor(values, dtype=torch.float32)
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
def demo_selfplay_and_train(device='cpu'):
    print("Demo: building network and running a tiny self-play + train cycle...")
    if not TORCH_AVAILABLE:
        print("PyTorch not found. Running a purely random-play demo of Xiangqi move generation.")
        board = initial_board(); print_board(board)
        moves = generate_legal_moves(board, 'red')
        print("Legal moves for red at start:", len(moves))
        return

    net = AlphaNet()
    net.to(device)
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
    print("Demo complete. Save model if desired.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='Run a short demo')
    args = parser.parse_args()
    if args.demo:
        demo_selfplay_and_train()
    else:
        print("This script provides a demo. Run with --demo to execute a short self-play + train cycle.")

if __name__ == '__main__':
    main()
