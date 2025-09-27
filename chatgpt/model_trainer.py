"""
模型训练器 - 用于训练AlphaZero象棋神经网络模型
包含神经网络定义、训练循环和数据处理功能
"""

import argparse
import os
import json
import numpy as np

# 尝试导入torch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception as e:
    TORCH_AVAILABLE = False
    print("Error: PyTorch not available. Training cannot proceed. Error:", e)
    exit(1)

from game_utils import *

# ----------------------------- 神经网络 -----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)

class AlphaNet(nn.Module):
    def __init__(self, in_channels=15, channels=64, n_resblocks=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.resblocks = nn.ModuleList([ResidualBlock(channels) for _ in range(n_resblocks)])
        # 策略头
        self.policy_conv = nn.Conv2d(channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * BOARD_ROWS * BOARD_COLS, ACTION_SPACE_SIZE) 
        # 价值头
        self.value_conv = nn.Conv2d(channels, 8, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * BOARD_ROWS * BOARD_COLS, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        for r in self.resblocks:
            out = r(out)
        # 策略
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)  # (B, ACTION_SPACE_SIZE)
        # 价值
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).view(-1)
        return p, v

# ----------------------------- 数据处理 -----------------------------
def read_transitions_jsonl(file_path):
    """从JSONL文件读取转换数据"""
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
    """将数据集加载到内存中"""
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
    """生成小批次数据"""
    n = states.shape[0]
    idxs = np.arange(n)
    if shuffle:
        np.random.shuffle(idxs)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = idxs[start:end]
        yield states[batch_idx], pis[batch_idx], vals[batch_idx]

# ----------------------------- 训练 -----------------------------
def train_step(net, optimizer, batch, device=None):
    """执行一步训练"""
    net.train()
    states, pis, values = batch
    if device is None:
        device = next(net.parameters()).device
    states_t = torch.tensor(states, dtype=torch.float32, device=device)
    pis_t = torch.tensor(pis, dtype=torch.float32, device=device)
    vals_t = torch.tensor(values, dtype=torch.float32, device=device)
    logits, pred_vals = net(states_t)
    # 策略损失（与pi分布的交叉熵）
    log_probs = F.log_softmax(logits, dim=1)
    policy_loss = - (pis_t * log_probs).sum(dim=1).mean()
    value_loss = F.mse_loss(pred_vals, vals_t)
    # 总损失
    loss = policy_loss + value_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().numpy()), float(policy_loss.detach().cpu().numpy()), float(value_loss.detach().cpu().numpy())

def train_from_file(data_path, model_path=None, device='cuda', epochs=1, batch_size=64, lr=1e-3):
    """从文件训练模型"""
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Training mode cannot proceed.")
        return
    
    print(f"Loading dataset from '{data_path}' ...")
    states, pis, vals = load_dataset_into_memory(data_path)
    print(f"Loaded {len(states)} training samples")
    
    net = AlphaNet().to(device)
    
    if model_path and os.path.exists(model_path):
        try:
            state = torch.load(model_path, map_location=device)
            net.load_state_dict(state)
            print(f"Loaded model parameters from '{model_path}'.")
        except Exception as e:
            print(f"Warning: Failed to load model from '{model_path}': {e}. Starting with random weights.")
            
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    print(f"Starting training for {epochs} epochs...")
    for ep in range(epochs):
        ep_loss = []
        ep_ploss = []
        ep_vloss = []
        
        # 在iterate_minibatches内部打乱
        for b_states, b_pis, b_vals in iterate_minibatches(states, pis, vals, batch_size, shuffle=True):
            loss, pl, vl = train_step(net, optimizer, (b_states, b_pis, b_vals), device=device)
            ep_loss.append(loss)
            ep_ploss.append(pl)
            ep_vloss.append(vl)
            
        print(f"Epoch {ep+1}/{epochs}: Loss={np.mean(ep_loss):.4f} (P={np.mean(ep_ploss):.4f}, V={np.mean(ep_vloss):.4f}), Batches={len(ep_loss)}")
        
    # 保存模型
    final_model_path = model_path if model_path else os.environ.get('ALPHAXIANGQI_MODEL_PATH', 'alphazero_xiangqi.pt')
    try:
        torch.save(net.state_dict(), final_model_path)
        print(f"Saved model parameters to '{final_model_path}'.")
    except Exception as e:
        print(f"Warning: Failed to save model to '{final_model_path}': {e}")

# ----------------------------- 模型评估 -----------------------------
def evaluate_model(model_path, data_path, device='cuda', batch_size=64):
    """评估模型性能"""
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Evaluation cannot proceed.")
        return
    
    print(f"Loading model from '{model_path}' ...")
    net = AlphaNet().to(device)
    try:
        state = torch.load(model_path, map_location=device)
        net.load_state_dict(state)
        print(f"Loaded model parameters from '{model_path}'.")
    except Exception as e:
        print(f"Error: Failed to load model from '{model_path}': {e}")
        return
    
    print(f"Loading evaluation data from '{data_path}' ...")
    states, pis, vals = load_dataset_into_memory(data_path)
    print(f"Loaded {len(states)} evaluation samples")
    
    net.eval()
    total_loss = 0
    total_ploss = 0
    total_vloss = 0
    num_batches = 0
    
    with torch.no_grad():
        for b_states, b_pis, b_vals in iterate_minibatches(states, pis, vals, batch_size, shuffle=False):
            states_t = torch.tensor(b_states, dtype=torch.float32, device=device)
            pis_t = torch.tensor(b_pis, dtype=torch.float32, device=device)
            vals_t = torch.tensor(b_vals, dtype=torch.float32, device=device)
            
            logits, pred_vals = net(states_t)
            # 策略损失
            log_probs = F.log_softmax(logits, dim=1)
            policy_loss = - (pis_t * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(pred_vals, vals_t)
            loss = policy_loss + value_loss
            
            total_loss += float(loss.detach().cpu().numpy())
            total_ploss += float(policy_loss.detach().cpu().numpy())
            total_vloss += float(value_loss.detach().cpu().numpy())
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_ploss = total_ploss / num_batches
    avg_vloss = total_vloss / num_batches
    
    print(f"Evaluation Results:")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Average Policy Loss: {avg_ploss:.4f}")
    print(f"  Average Value Loss: {avg_vloss:.4f}")
    print(f"  Number of Batches: {num_batches}")

def main():
    parser = argparse.ArgumentParser(description='AlphaZero象棋模型训练器')
    
    # 通用参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() and TORCH_AVAILABLE else 'cpu', 
                        help='设备选择: "cuda" 或 "cpu"')
    parser.add_argument('--model_path', type=str, default='alphazero_xiangqi.pt', 
                        help='神经网络模型路径')
    
    subparsers = parser.add_subparsers(dest='mode', required=True, help='操作模式')

    # 训练模式
    parser_train = subparsers.add_parser('train', help='训练模型')
    parser_train.add_argument('--data_path', type=str, required=True, help='训练数据JSONL文件路径')
    parser_train.add_argument('--epochs', type=int, default=1, help='训练轮数')
    parser_train.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser_train.add_argument('--lr', type=float, default=1e-3, help='学习率')

    # 评估模式
    parser_eval = subparsers.add_parser('evaluate', help='评估模型')
    parser_eval.add_argument('--data_path', type=str, required=True, help='评估数据JSONL文件路径')
    parser_eval.add_argument('--batch_size', type=int, default=64, help='批次大小')

    args = parser.parse_args()
    
    if args.mode == 'train':
        train_from_file(
            data_path=args.data_path,
            model_path=args.model_path,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
    elif args.mode == 'evaluate':
        evaluate_model(
            model_path=args.model_path,
            data_path=args.data_path,
            device=args.device,
            batch_size=args.batch_size,
        )

if __name__ == '__main__':
    main()

