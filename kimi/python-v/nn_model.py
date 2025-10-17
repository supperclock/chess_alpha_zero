import torch
import torch.nn as nn
import torch.nn.functional as F
from nn_data_representation import NUM_CHANNELS, get_move_maps

# <<< NEW: Squeeze-and-Excitation Block >>>
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block that adaptively recalibrates channel-wise feature responses.
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# <<< MODIFIED: ResidualBlock now includes an SEBlock >>>
class SEResidualBlock(nn.Module):
    """A residual block with a Squeeze-and-Excitation module."""
    def __init__(self, num_filters):
        super(SEResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.se = SEBlock(num_filters) # Add the SE block

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)  # Apply SE module
        out += identity     # Residual connection
        return F.relu(out)

# <<< MODIFIED: Main network with increased capacity and SEResidualBlocks >>>
class XiangqiNet(nn.Module):
    """
    The optimized neural network for Xiangqi (Chinese Chess).
    """
    def __init__(self, num_filters=256, num_res_blocks=12): # Increased defaults
        super(XiangqiNet, self).__init__()
        
        # 1. Initial convolution layer (body)
        self.conv_in = nn.Conv2d(NUM_CHANNELS+1, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(num_filters)
        
        # 2. Residual blocks
        self.res_blocks = nn.ModuleList([SEResidualBlock(num_filters) for _ in range(num_res_blocks)])
        
        # 3. Policy head
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, stride=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.move_to_idx, self.idx_to_move = get_move_maps()
        self.policy_fc = nn.Linear(2 * 10 * 9, len(self.idx_to_move))
        
        # 4. Value head
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, stride=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 10 * 9, 512) # Increased hidden layer size
        self.value_fc2 = nn.Linear(512, 1)

    def forward(self, x):
        # Body
        x = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1) # Flatten
        policy = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1) # Flatten
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value)) # Output range in [-1, 1]
        
        return policy, value