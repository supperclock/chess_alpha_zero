import torch
from nn_model import XiangqiNet
from nn_data_representation import board_to_tensor, MOVE_TO_INDEX, INDEX_TO_MOVE
from ai import generate_moves  # 导入你现有的走法生成器
import os

class NN_Interface:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"正在使用设备: {self.device}")
        self.model = XiangqiNet().to(self.device)
        
        if model_path:
            if os.path.isfile(model_path):
                state = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state)
                print(f"模型已加载：{model_path}")
            else:
                print(f"权重文件不存在，将创建并初始化：{model_path}")
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(self.model.state_dict(), model_path)
                print("已保存初始权重。")
        else:
            print("警告: 未提供模型路径。将使用随机初始化的模型。")
            
        self.model.eval() # 设置为评估模式

    def predict(self, board_state, side_to_move):
        """
        对当前局面进行预测。
        返回: (价值, 策略字典)
        - 价值 (float): 当前局面的评估值，从当前玩家视角看 (-1必败, 1必胜)
        - 策略字典 (dict): {Move: probability}，只包含当前局面的合法走法及其概率
        """
        with torch.no_grad():
            # 1. 将棋盘转换为张量
            tensor = board_to_tensor(board_state, side_to_move).to(self.device)
            
            # 2. 通过模型获取原始输出
            policy_logits, value_tensor = self.model(tensor)
            
            # 3. 处理价值输出
            value = value_tensor.item()
            
            # 4. 处理策略输出
            # 首先，获取当前局面下的所有合法走法
            legal_moves = generate_moves(board_state, side_to_move)
            if not legal_moves:
                return value, {}

            # 将logits转换为概率
            policy_probs = torch.softmax(policy_logits, dim=1).squeeze().cpu()
            
            # 筛选出合法走法的概率
            legal_policy = {}
            total_prob = 0.0
            
            for move in legal_moves:
                move_key = (move.fx, move.fy, move.tx, move.ty)
                if move_key in MOVE_TO_INDEX:
                    move_idx = MOVE_TO_INDEX[move_key]
                    prob = policy_probs[move_idx].item()
                    legal_policy[move] = prob
                    total_prob += prob
                else:
                    # 如果一个合法走法不在我们的映射中，这是一个警告
                    print(f"警告：合法走法 {move_key} 未在走法映射中找到！")
            
            # 5. 重新归一化，确保合法走法概率和为1
            if total_prob > 0:
                for move in legal_policy:
                    legal_policy[move] /= total_prob
            else:
                # 如果所有合法走法的概率都为0（极少见），则平均分配
                num_legal_moves = len(legal_policy)
                if num_legal_moves > 0:
                    prob = 1.0 / num_legal_moves
                    for move in legal_policy:
                        legal_policy[move] = prob

            return value, legal_policy