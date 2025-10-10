import torch
import torch.nn as nn
import torch.nn.functional as F
from nn_data_representation import NUM_CHANNELS, get_move_maps

class ResidualBlock(nn.Module):
    """一个残差块，AlphaZero中的核心组件"""
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity # 残差连接
        return F.relu(out)

class XiangqiNet(nn.Module):
    """
    对接你的象棋逻辑的神经网络
    """
    def __init__(self, num_filters=128, num_res_blocks=7):
        super(XiangqiNet, self).__init__()
        
        # 1. 初始卷积层 (身体)
        self.conv_in = nn.Conv2d(NUM_CHANNELS + 1, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(num_filters)
        
        # 2. 残差块
        self.res_blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_res_blocks)])
        
        # 3. 策略头
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, stride=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.move_to_idx, self.idx_to_move = get_move_maps()
        self.policy_fc = nn.Linear(2 * 10 * 9, len(self.idx_to_move))
        
        # 4. 价值头
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, stride=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 10 * 9, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # 身体部分
        x = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            x = block(x)
        
        # 策略头
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1) # Flatten
        policy = self.policy_fc(policy)
        # 输出原始logits，softmax将在接口中处理
        
        # 价值头
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1) # Flatten
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value)) # 输出范围在[-1, 1]
        
        return policy, value