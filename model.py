import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyValueNet(nn.Module):
    def __init__(self, action_size, board_height=10, board_width=9):
        super().__init__()
        self.board_height = board_height
        self.board_width = board_width
        self.action_size = action_size
        
        self.conv = nn.Sequential(
            nn.Conv2d(2, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # 动态计算全连接层的输入维度
        conv_output_size = 128 * board_height * board_width
        self.policy_head = nn.Linear(conv_output_size, action_size)
        self.value_head = nn.Sequential(
            nn.Linear(conv_output_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # 输入验证
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (batch, channels, height, width), got {x.dim()}D")
        
        if x.size(1) != 2:
            raise ValueError(f"Expected 2 channels, got {x.size(1)}")
            
        if x.size(2) != self.board_height or x.size(3) != self.board_width:
            raise ValueError(f"Expected board size ({self.board_height}, {self.board_width}), got ({x.size(2)}, {x.size(3)})")
        
        batch = x.size(0)
        x = self.conv(x)
        x_flat = x.view(batch, -1)
        
        logits = self.policy_head(x_flat)
        value = self.value_head(x_flat)
        
        return F.log_softmax(logits, dim=1), value
    
    def get_action_probs(self, x, temperature=1.0):
        """获取动作概率分布"""
        with torch.no_grad():
            logp, _ = self.forward(x)
            probs = torch.exp(logp / temperature)
            return probs
