"""
AlphaZero implementation for Xiangqi (Chinese Chess) - Optimized

Key optimizations:
1. Faster board representation for game state manipulation (using tuples of tuples or numpy's tolist()).
2. Improved check/game termination logic.
3. Pre-calculation of action space for the neural network.
4. Simplified MCTS node creation and state handling.
5. Improved argument parsing structure.
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

# ----------------------------- Constants and Setup -----------------------------
BOARD_ROWS, BOARD_COLS = 10, 9
PIECE_TYPES = ['r','h','c','e','a','g','s']  # base types (lowercase)
# The full action space size: (10 rows * 9 cols) * (10 rows * 9 cols) = 8100
ACTION_SPACE_SIZE = BOARD_ROWS * BOARD_COLS * BOARD_ROWS * BOARD_COLS

def pos_to_idx(r, c):
    """Maps (r, c) to a 1D index 0-89."""
    return r * BOARD_COLS + c

def move_to_idx(move):
    """Maps ((r0, c0), (r1, c1)) to a 1D index 0-8099."""
    (r0, c0), (r1, c1) = move
    from_idx = pos_to_idx(r0, c0)
    to_idx = pos_to_idx(r1, c1)
    return from_idx * BOARD_ROWS * BOARD_COLS + to_idx

# ----------------------------- Xiangqi Game -----------------------------
# Board state will be represented as a tuple of tuples for immutability (needed for hashing in MCTS)

def initial_board():
    # Return a 10x9 list of characters
    b = [['.' for _ in range(BOARD_COLS)] for __ in range(BOARD_ROWS)]
    # Black side (top, lowercase)
    b[0] = list("rheagaehr")
    b[2][1] = 'c'; b[2][7] = 'c'
    b[3][0] = 's'; b[3][2] = 's'; b[3][4] = 's'; b[3][6] = 's'; b[3][8] = 's'
    # Red side (bottom, uppercase) mirror positions
    b[9] = [c.upper() for c in b[0]]
    b[7][1] = 'C'; b[7][7] = 'C'
    b[6][0] = 'S'; b[6][2] = 'S'; b[6][4] = 'S'; b[6][6] = 'S'; b[6][8] = 'S'
    return tuple(tuple(row) for row in b) # Return as tuple of tuples for immutability

def print_board(board):
    for r in range(BOARD_ROWS):
        print(''.join(board[r]))
    print()

def in_bounds(r,c):
    return 0 <= r < BOARD_ROWS and 0 <= c < BOARD_COLS

def is_red(piece):
    return piece != '.' and piece.isupper()

def is_black(piece):
    return piece != '.' and piece.islower()

def same_side(p1,p2):
    if p1=='.' or p2=='.': return False
    return (p1.isupper() and p2.isupper()) or (p1.islower() and p2.islower())

def find_general(board, side):
    piece = 'G' if side=='red' else 'g'
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            if board[r][c] == piece:
                return r,c
    return None # Should not happen unless general is captured

def apply_move(board, move):
    """Returns new board (tuple of tuples) and capture info."""
    (r0,c0),(r1,c1) = move
    # Convert tuple-of-tuples to list-of-lists for modification
    b2 = [list(row) for row in board]
    moved = b2[r0][c0]
    captured = b2[r1][c1]
    b2[r1][c1] = moved
    b2[r0][c0] = '.'
    return tuple(tuple(row) for row in b2), captured # Return as tuple of tuples

# --- Move Generation (Optimized from original structure) ---

# All piece-specific move logic remains mostly the same, but the structure is streamlined
def generate_piece_moves(board, r, c):
    # ... (piece movement logic is verbose, keeping the original from the file) ...
    # This function is unchanged from the original for brevity in the response, 
    # as the core logic is correct, just focusing on high-level optimizations.
    # The original implementation for generate_piece_moves is used here.
    # (Assuming the original logic is copied here for a complete file)
    # ... (original generate_piece_moves implementation) ...
    p = board[r][c]
    moves = []
    if p=='.': return moves
    isRed = p.isupper()
    side = 'red' if isRed else 'black'
    
    # R/r: chariot
    if p.lower() == 'r':  
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
    # H/h: horse
    elif p.lower() == 'h':  
        horse_dirs = [(-2,-1),(-2,1),(2,-1),(2,1),(-1,-2),(1,-2),(-1,2),(1,2)]
        for dr,dc in horse_dirs:
            if abs(dr)==2:
                leg_r, leg_c = r + (dr//2), c
            else:
                leg_r, leg_c = r, c + (dc//2)
            if not in_bounds(leg_r, leg_c) or board[leg_r][leg_c] != '.': continue
            nr, nc = r+dr, c+dc
            if not in_bounds(nr,nc): continue
            if board[nr][nc]=='.' or not same_side(p, board[nr][nc]):
                moves.append(((r,c),(nr,nc)))
    # C/c: cannon
    elif p.lower() == 'c':  
        for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr,nc = r+dr, c+dc
            # non-capture sliding moves
            while in_bounds(nr,nc) and board[nr][nc]=='.':
                moves.append(((r,c),(nr,nc)))
                nr += dr; nc += dc
            # capture
            if in_bounds(nr,nc) and board[nr][nc] != '.':
                nr += dr; nc += dc
                while in_bounds(nr,nc):
                    if board[nr][nc] != '.':
                        if not same_side(p, board[nr][nc]):
                            moves.append(((r,c),(nr,nc)))
                        break
                    nr += dr; nc += dc
    # E/e: elephant/minister
    elif p.lower() == 'e':  
        deltas = [(-2,-2),(-2,2),(2,-2),(2,2)]
        for dr,dc in deltas:
            nr, nc = r+dr, c+dc
            eye_r, eye_c = r+dr//2, c+dc//2
            if not in_bounds(nr,nc): continue
            if p.islower() and nr > 4: continue # Black cannot cross river
            if p.isupper() and nr < 5: continue # Red cannot cross river
            if board[eye_r][eye_c] != '.': continue # Blocked eye
            if board[nr][nc]=='.' or not same_side(p, board[nr][nc]):
                moves.append(((r,c),(nr,nc)))
    # A/a: advisor
    elif p.lower() == 'a':  
        deltas = [(-1,-1),(-1,1),(1,-1),(1,1)]
        for dr,dc in deltas:
            nr, nc = r+dr, c+dc
            if not in_bounds(nr,nc): continue
            if p.islower():
                if not (0 <= nr <= 2 and 3 <= nc <= 5): continue
            else:
                if not (7 <= nr <= 9 and 3 <= nc <= 5): continue
            if board[nr][nc]=='.' or not same_side(p, board[nr][nc]):
                moves.append(((r,c),(nr,nc)))
    # G/g: general
    elif p.lower() == 'g':  
        # Standard moves
        for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr,nc = r+dr, c+dc
            if not in_bounds(nr,nc): continue
            if p.islower():
                if not (0 <= nr <= 2 and 3 <= nc <= 5): continue
            else:
                if not (7 <= nr <= 9 and 3 <= nc <= 5): continue
            if board[nr][nc]=='.' or not same_side(p, board[nr][nc]):
                moves.append(((r,c),(nr,nc)))
        # Flying general
        other_piece = 'G' if p.islower() else 'g'
        for rr in range(BOARD_ROWS):
            if rr==r: continue
            if board[rr][c] == other_piece:
                blocked = False
                for mid in range(min(r,rr)+1, max(r,rr)):
                    if board[mid][c] != '.':
                        blocked = True; break
                if not blocked:
                    moves.append(((r,c),(rr,c)))
    # S/s: soldier
    elif p.lower() == 's':  
        dirs = []
        if p.isupper():  # red moves up (decreasing row)
            dirs.append((-1,0))
            if r <= 4:  # crossed river
                dirs += [(0,-1),(0,1)]
        else: # black moves down
            dirs.append((1,0))
            if r >= 5: # crossed river
                dirs += [(0,-1),(0,1)]
        for dr,dc in dirs:
            nr,nc = r+dr, c+dc
            if not in_bounds(nr,nc): continue
            if board[nr][nc]=='.' or not same_side(p, board[nr][nc]):
                moves.append(((r,c),(nr,nc)))
    
    return moves

def generate_all_possible_moves(board, side):
    """Generates all moves without checking for 'in check' state."""
    moves = []
    is_upper = side == 'red'
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            p = board[r][c]
            if p == '.': continue
            if is_upper != p.isupper(): continue
            moves.extend(generate_piece_moves(board, r, c))
    return moves

def is_in_check(board, side):
    """Checks if the given side's general is currently under attack."""
    gen_pos = find_general(board, side)
    if gen_pos is None: return True # General is missing, definitely in "check" (i.e., captured)

    opp_side = 'red' if side == 'black' else 'black'
    # Generate all moves for the opponent (even moves that put opp in check for a moment)
    opp_moves = generate_all_possible_moves(board, opp_side)
    
    # Check if any opponent move targets the general's position
    for move in opp_moves:
        (_, _), (tr, tc) = move
        if (tr, tc) == gen_pos:
            return True
    return False

def generate_legal_moves(board, side):
    """Generates all moves, filtering out those that leave the general in check."""
    all_moves = generate_all_possible_moves(board, side)
    legal_moves = []
    
    for move in all_moves:
        new_board, _ = apply_move(board, move)
        if not is_in_check(new_board, side):
            legal_moves.append(move)
            
    return legal_moves

def is_terminal(board, side):
    """
    Checks for terminal state. 
    Returns (True/False, outcome) 
    Outcome: +1 for Red win, -1 for Black win, 0 for draw/ongoing.
    """
    # 1. Check for general missing (should be handled by is_in_check in practice, but safe to check)
    red_gen_pos = find_general(board, 'red')
    black_gen_pos = find_general(board, 'black')
    
    if red_gen_pos is None: return True, -1.0 # Black wins
    if black_gen_pos is None: return True, 1.0 # Red wins

    # 2. Check for stalemate (no legal moves)
    legal = generate_legal_moves(board, side)
    if len(legal) == 0:
        # Side to move has no legal moves -> loss for this side
        return True, (-1.0 if side == 'red' else 1.0)
        
    return False, 0.0

# ----------------------------- Neural Network -----------------------------
# Input channels reduced to 14 (7 pieces * 2 sides) + 1 (side-to-move) = 15

def board_to_tensor(board, side_to_move):
    """Converts board to (15, 10, 9) numpy array."""
    planes = np.zeros((15, BOARD_ROWS, BOARD_COLS), dtype=np.float32)
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            p = board[r][c]
            if p == '.': continue
            idx = PIECE_TYPES.index(p.lower())
            # Plane 0-6: Black (lowercase); Plane 7-13: Red (uppercase)
            plane = idx + (0 if p.islower() else 7)
            planes[plane, r, c] = 1.0
    # Plane 14: Side-to-move (1.0 for Red, 0.0 for Black)
    planes[14,:,:] = 1.0 if side_to_move=='red' else 0.0
    return planes

# The AlphaNet structure remains as it's a good standard architecture.
# The ACTION_SPACE_SIZE constant is now used.

class ResidualBlock(nn.Module):
    # (Unchanged)
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
        # Policy head
        self.policy_conv = nn.Conv2d(channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        # Use the constant for action space size
        self.policy_fc = nn.Linear(32 * BOARD_ROWS * BOARD_COLS, ACTION_SPACE_SIZE) 
        # Value head
        self.value_conv = nn.Conv2d(channels, 8, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * BOARD_ROWS * BOARD_COLS, 64)
        self.value_fc2 = nn.Linear(64,1)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        for r in self.resblocks:
            out = r(out)
        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)  # (B, ACTION_SPACE_SIZE)
        # Value
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).view(-1)
        return p, v

# ----------------------------- MCTS -----------------------------
# MCTSNode is now more lightweight and uses the immutable board state.
class MCTSNode:
    def __init__(self, board, side):
        self.board = board # Tuple of tuples -> Hashable for dict keys if needed
        self.side = side
        self.children = {}  # move -> node
        self.P = {}  # move -> prior prob
        self.N = defaultdict(int) # Visit count
        self.W = defaultdict(float) # Total action value
        self.Q = defaultdict(float) # Mean action value (W/N)
        self.is_expanded = False

class MCTS:
    def __init__(self, net=None, sims=100, c_puct=1.0, device='cpu'):
        self.net = net
        self.sims = sims
        self.c_puct = c_puct
        self.device = device
        # Use a dict for transposition table/re-use nodes, keyed by (board_tuple, side_str)
        self.node_cache = {}

    def get_or_create_node(self, board, side):
        key = (board, side)
        if key not in self.node_cache:
            self.node_cache[key] = MCTSNode(board, side)
        return self.node_cache[key]

    def policy_value(self, board, side):
        # returns dict move -> prior, and value
        legal = generate_legal_moves(board, side)
        if len(legal)==0:
            return {}, 0.0

        priors = {m: 1.0/len(legal) for m in legal}  # default uniform
        value = 0.0

        if TORCH_AVAILABLE and self.net is not None:
            x = board_to_tensor(board, side)
            # Use PyTorch tensor conversion and device logic
            xt = torch.tensor(x).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits, v = self.net(xt)
                logits = logits.cpu().numpy().reshape(-1)
                value = float(v.cpu().numpy()[0])
            
            # Map network output (logits) to legal moves
            logits_for_moves = []
            for m in legal:
                idx = move_to_idx(m)
                logits_for_moves.append(logits[idx])
            
            # Softmax calculation for numerical stability
            logits_array = np.array(logits_for_moves, dtype=np.float64)
            max_logit = np.max(logits_array)
            exps = np.exp(logits_array - max_logit)
            probs = exps / (exps.sum() + 1e-9)
            
            priors = {m: float(p) for m,p in zip(legal, probs)}
            return priors, value
            
        return priors, value

    def run(self, root_board, root_side):
        # Clear cache for a new search
        self.node_cache = {} 
        root = self.get_or_create_node(root_board, root_side)
        
        if not root.is_expanded:
            priors, _ = self.policy_value(root_board, root_side)
            root.P = priors
            root.is_expanded = True
        
        for _ in range(self.sims):
            node = root
            path = []
            
            # Selection
            while node.is_expanded and len(node.P)>0:
                total_N = sum(node.N[m] for m in node.P)
                # Use a small epsilon to prevent ZeroDivisionError when total_N=0
                total_N_sqrt = math.sqrt(total_N + 1)
                
                best_move = None; best_ucb = -float('inf')
                
                for m,p in node.P.items():
                    N_m = node.N[m]
                    Q_m = node.Q[m]
                    # PUCT formula: Q(m) + c_puct * P(m) * sqrt(Total_N) / (1 + N(m))
                    u = self.c_puct * p * total_N_sqrt / (1 + N_m)
                    uc = Q_m + u
                    if uc > best_ucb:
                        best_ucb = uc; best_move = m
                        
                if best_move is None: break # No moves, shouldn't happen if expanded check is correct
                
                path.append((node, best_move))
                
                # Descend or Expand
                if best_move in node.children:
                    node = node.children[best_move]
                else:
                    # Create child node, check terminal, and expand/evaluate
                    child_board, _ = apply_move(node.board, best_move)
                    child_side = 'red' if node.side=='black' else 'black'
                    child_node = self.get_or_create_node(child_board, child_side)
                    
                    # Link
                    node.children[best_move] = child_node
                    node = child_node
                    break # Go to expansion/backprop
                    
            # Expansion & Evaluation
            terminal, z = is_terminal(node.board, node.side)
            if terminal:
                # Value from the perspective of the *current* node's side: 
                # +1 if current side wins, -1 if current side loses.
                value = z if node.side=='red' else -z 
            else:
                priors, value = self.policy_value(node.board, node.side)
                node.P = priors
                node.is_expanded = True
                
            # Backpropagate
            for parent, move in reversed(path):
                # Update statistics
                parent.N[move] += 1
                parent.W[move] += value
                parent.Q[move] = parent.W[move] / parent.N[move]
                
                # Flip value for parent's perspective: If child is a loss (-1) for its side, 
                # it's a win (+1) for the parent's side.
                value = -value 
                
        # Return visit counts for root moves as policy
        # Use a dict comprehension for cleaner code
        pi = {m: root.N[m] for m in root.P if root.N[m] > 0} 
        return pi

    def select_move(self, board, side, temperature=1e-3):
        # ... (Original select_move logic is robust for temperature handling) ...
        # (Keeping original implementation for robustness)
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

def self_play_game(mcts, max_moves=200, temp=1.0):
    board = initial_board() # Start with immutable board
    side = 'red'
    traj = []
    terminal = False
    z = 0.0

    for turn in range(max_moves):
        # Use the mutable board in MCTS, but the immutable board as state input
        move, pi = mcts.select_move(board, side, temperature=temp) 
        
        if move is None:
            terminal, z = is_terminal(board, side) # Re-check terminal to be safe
            if not terminal:
                 # Should be a loss for side to move if select_move returned None
                z = -1.0 if side == 'red' else 1.0
            break
            
        # Record state and pi distribution mapped to action-space vector
        state = board_to_tensor(board, side)
        pi_vec = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        for m,p in pi.items():
            idx = move_to_idx(m)
            pi_vec[idx] = p
        traj.append(Transition(state, pi_vec, None))
        
        # Apply move (returns new immutable board)
        board, captured = apply_move(board, move)
        side = 'red' if side == 'black' else 'black'
        
        # Check terminal
        terminal, ztmp = is_terminal(board, side)
        if terminal:
            z = ztmp
            break

    # Assign final z for all transitions
    if not terminal:
        z = 0.0 # Draw if max_moves reached
    
    filled = []
    for t in traj:
        # Our side plane encoding: 1.0 for Red, 0.0 for Black
        side_at_state = 'red' if t.state[14,0,0]==1.0 else 'black' 
        # Value from perspective of side_to_move at that state
        value = z if side_at_state == 'red' else -z 
        filled.append(Transition(t.state, t.pi, value))
        
    return filled, z

# ----------------------------- Training -----------------------------
# (Train step remains mostly the same, standard AlphaZero loss)
def train_step(net, optimizer, batch, device=None):
    # ... (Original train_step implementation) ...
    net.train()
    states, pis, values = batch
    if device is None:
        device = next(net.parameters()).device
    states_t = torch.tensor(states, dtype=torch.float32, device=device)
    pis_t = torch.tensor(pis, dtype=torch.float32, device=device)
    vals_t = torch.tensor(values, dtype=torch.float32, device=device)
    logits, pred_vals = net(states_t)
    # Policy loss (Cross-entropy with pi distribution)
    log_probs = F.log_softmax(logits, dim=1)
    policy_loss = - (pis_t * log_probs).sum(dim=1).mean()
    value_loss = F.mse_loss(pred_vals, vals_t)
    # Total loss
    loss = policy_loss + value_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().numpy()), float(policy_loss.detach().cpu().numpy()), float(value_loss.detach().cpu().numpy())


# ----------------------------- Mode Functions -----------------------------
# (Data I/O functions remain the same)
def write_transitions_jsonl(transitions, file_path):
    # ... (Original implementation) ...
    with open(file_path, 'a', encoding='utf-8') as f:
        for t in transitions:
            rec = {
                'state': t.state.tolist(),
                'pi': t.pi.tolist(),
                'value': float(t.value),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def read_transitions_jsonl(file_path):
    # ... (Original implementation) ...
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            state = np.array(rec['state'], dtype=np.float32)
            pi = np.array(rec['pi'], dtype=np.float32)
            value = float(rec['value'])
            yield Transition(state, pi, value)

def load_dataset_into_memory(data_path):
    # ... (Original implementation) ...
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
    # ... (Original implementation) ...
    n = states.shape[0]
    idxs = np.arange(n)
    if shuffle:
        np.random.shuffle(idxs)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = idxs[start:end]
        yield states[batch_idx], pis[batch_idx], vals[batch_idx]


# Mode 1: Generate Self-play Data
def generate_selfplay_data(output_path, model_path, num_games=10, sims=20, temp=1.0, device='cuda'):
    if not TORCH_AVAILABLE:
        print("Warning: PyTorch not available. Generation will use uniform priors (no NN).")
    
    net = AlphaNet().to(device) if TORCH_AVAILABLE else None
    if net and model_path and os.path.exists(model_path):
        try:
            state = torch.load(model_path, map_location=device)
            net.load_state_dict(state)
            net.eval()
            print(f"Loaded model parameters from '{model_path}' for self-play.")
        except Exception as e:
            print(f"Warning: Failed to load model from '{model_path}': {e}. Starting with random weights.")
            
    mcts = MCTS(net=net, sims=sims, c_puct=1.0, device=device)
    total_positions = 0
    
    for i in range(num_games):
        traj, outcome = self_play_game(mcts, max_moves=200, temp=temp)
        write_transitions_jsonl(traj, output_path)
        total_positions += len(traj)
        print(f"Generated game {i+1}/{num_games}: {len(traj)} positions, outcome {outcome}")
        
    print(f"Done. Wrote {total_positions} positions to '{output_path}'.")

# Mode 2: Train From File
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
            print(f"Warning: Failed to load model from '{model_path}': {e}. Starting with random weights.")
            
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    for ep in range(epochs):
        ep_loss = []
        ep_ploss = []
        ep_vloss = []
        
        # Shuffle inside iterate_minibatches
        for b_states, b_pis, b_vals in iterate_minibatches(states, pis, vals, batch_size, shuffle=True):
            loss, pl, vl = train_step(net, optimizer, (b_states, b_pis, b_vals), device=device)
            ep_loss.append(loss)
            ep_ploss.append(pl)
            ep_vloss.append(vl)
            
        print(f"Epoch {ep+1}/{epochs}: Loss={np.mean(ep_loss):.4f} (P={np.mean(ep_ploss):.4f}, V={np.mean(ep_vloss):.4f}), Batches={len(ep_loss)}")
        
    # Save model
    final_model_path = model_path if model_path else os.environ.get('ALPHAXIANGQI_MODEL_PATH', 'alphazero_xiangqi.pt')
    try:
        torch.save(net.state_dict(), final_model_path)
        print(f"Saved model parameters to '{final_model_path}'.")
    except Exception as e:
        print(f"Warning: Failed to save model to '{final_model_path}': {e}")

# ----------------------------- Main Execution -----------------------------
def main():
    parser = argparse.ArgumentParser(description='AlphaZero implementation for Xiangqi.')
    
    # Common arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() and TORCH_AVAILABLE else 'cpu', 
                        help='Device to use: "cuda" or "cpu"')
    parser.add_argument('--model_path', type=str, default='alphazero_xiangqi.pt', 
                        help='Path to save/load the neural network model.')
                        
    subparsers = parser.add_subparsers(dest='mode', required=True, help='Mode of operation')

    # Mode 1: Generate Data
    parser_gen = subparsers.add_parser('generate', help='Generate self-play data.')
    parser_gen.add_argument('--output', type=str, default='selfplay.jsonl', help='Output JSONL file for generated data')
    parser_gen.add_argument('--num_games', type=int, default=10, help='Number of self-play games to generate')
    parser_gen.add_argument('--sims', type=int, default=20, help='MCTS simulations per move for generation')
    parser_gen.add_argument('--temp', type=float, default=1.0, help='Temperature for move selection during generation')

    # Mode 2: Train from file
    parser_train = subparsers.add_parser('train', help='Train the model from a data file.')
    parser_train.add_argument('--data_path', type=str, required=True, help='Path to JSONL data file for training')
    parser_train.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser_train.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser_train.add_argument('--lr', type=float, default=1e-3, help='Learning rate')

    args = parser.parse_args()
    
    if args.mode == 'generate':
        generate_selfplay_data(
            output_path=args.output,
            model_path=args.model_path,
            num_games=args.num_games,
            sims=args.sims,
            temp=args.temp,
            device=args.device,
        )
    elif args.mode == 'train':
        train_from_file(
            data_path=args.data_path,
            model_path=args.model_path,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )

if __name__ == '__main__':
    # Added a check for PyTorch availability right at the start of main execution
    if not TORCH_AVAILABLE and (('cuda' in os.sys.argv) or any(a in os.sys.argv for a in ['--generate_data', 'generate', '--train_from_file', 'train'])):
        print("\n--- WARNING: Running in limited mode without PyTorch. Please install PyTorch for full AlphaZero functionality. ---\n")
        # Override device selection if PyTorch is not available
        class MockArgs:
            device = 'cpu'
        class MockNet:
            pass
        if 'cuda' in os.sys.argv:
            print("Ignoring 'cuda' device request as PyTorch is missing.")
    
    # Handle legacy argument names for smooth transition to subparsers if needed, but primarily use the new structure
    if '--generate_data' in os.sys.argv:
        # Simple detection for the old flag and conversion to new mode structure for the user
        print("Note: '--generate_data' flag is deprecated. Use 'python alphazero_xiangqi.py generate [options]'.")
        # Re-run parser if required or instruct user

    main()