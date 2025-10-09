import torch
from opening_book import Move  
from ai import GEN_MAP, ROWS, COLS, inside  

# --- 1. 棋子到网络通道的映射 ---
# 我们为双方的7种不同棋子创建独立的通道
PIECE_TO_CHANNEL = {
    ('帥', 'red'): 0, ('仕', 'red'): 1, ('相', 'red'): 2, ('馬', 'red'): 3, ('車', 'red'): 4, ('炮', 'red'): 5, ('兵', 'red'): 6,
    ('將', 'black'): 7, ('士', 'black'): 8, ('象', 'black'): 9, ('馬', 'black'): 10, ('車', 'black'): 11, ('砲', 'black'): 12, ('卒', 'black'): 13,
}
# 统一棋子名称，方便查找
CANONICAL_PIECE_NAMES = {
    '帥': '帥', '將': '將',
    '仕': '仕', '士': '士',
    '相': '相', '象': '象',
    '馬': '馬', '傌': '馬', '马': '馬',
    '車': '車', '俥': '車', '车': '車',
    '炮': '炮', '砲': '砲',
    '兵': '兵', '卒': '卒',
}
NUM_CHANNELS = 14 # 7种棋子 * 2个阵营

def board_to_tensor(board_state, side_to_move):
    """
    将你的 board_state 转换为一个 PyTorch 张量。
    张量维度: (1, num_channels + 1, 10, 9)
    - num_channels: 14个通道代表双方的棋子
    - +1: 额外的1个通道代表当前轮到谁走棋
    """
    tensor = torch.zeros(1, NUM_CHANNELS + 1, ROWS, COLS)

    for r in range(ROWS):
        for c in range(COLS):
            piece = board_state[r][c]
            if piece:
                canonical_type = CANONICAL_PIECE_NAMES[piece['type']]
                channel_key = (canonical_type, piece['side'])
                if channel_key in PIECE_TO_CHANNEL:
                    channel_idx = PIECE_TO_CHANNEL[channel_key]
                    tensor[0, channel_idx, r, c] = 1
    
    # 添加额外的一个通道来表示轮到谁走
    # 如果是黑方走，该通道全为1；如果是红方走，全为0
    if side_to_move == 'black':
        tensor[0, NUM_CHANNELS, :, :] = 1.0
        
    return tensor

# --- 2. 走法到网络输出索引的映射 ---
# 我们需要一个固定的方式来表示所有可能的走法
MOVE_TO_INDEX = {}
INDEX_TO_MOVE = []

def create_move_maps():
    """
    生成所有伪合法走法到索引的映射。
    这个函数只需要在启动时运行一次。
    """
    if MOVE_TO_INDEX: # 防止重复生成
        return
        
    # 这是一个简化的版本，它为棋盘上每个点到每个点都创建一个映射
    # 更高效的方式是只生成伪合法移动（如马走日，象走田）
    # 但为了简单和通用，我们先用一个全连接的映射
    # 注意：在真实AlphaZero中，走法表示更复杂，但这对于入门是完美的。
    
    # 策略1：生成所有起点到终点的映射 (90*90 = 8100个输出)
    # 简单但稀疏，对于象棋来说有点大。
    
    # 策略2：生成所有棋子在所有位置上的伪合法走法（推荐）
    # 这会更复杂，但输出空间更紧凑。我们在这里实现这个。
    
    # 伪棋盘，用于生成所有可能的移动
    pseudo_board = [[None for _ in range(COLS)] for _ in range(ROWS)]
    
    idx = 0
    # 遍历所有可能的起点
    for r_from in range(ROWS):
        for c_from in range(COLS):
            # 模拟每种棋子在这个位置
            for piece_type, move_gen_func in GEN_MAP.items():
                # 只需要一个代表性的棋子类型
                if piece_type in ['俥', '车', '傌', '马', '砲', '卒', '士', '象', '將']:
                    continue
                
                # 模拟双方
                for side in ['red', 'black']:
                    pseudo_board[r_from][c_from] = {'type': piece_type, 'side': side}
                    
                    # 生成该棋子从该位置的伪合法移动
                    moves = move_gen_func(pseudo_board, c_from, r_from, side)
                    for move in moves:
                        key = (move.fy, move.fx, move.ty, move.tx)
                        if key not in MOVE_TO_INDEX:
                            MOVE_TO_INDEX[key] = idx
                            INDEX_TO_MOVE.append(key)
                            idx += 1
                            
                    pseudo_board[r_from][c_from] = None # 清理

    print(f"走法映射创建完成，总共有 {len(MOVE_TO_INDEX)} 种可能的走法。")


# 在模块加载时自动创建映射
create_move_maps()