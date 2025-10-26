from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from ai import check_game_over,minimax_root
from util import log
# from ai_bridge import find_best_move_c
# from ai_bridge2 import find_best_move_c2
# from ai_gpt_bridge import find_best_move_c
from pikafish_wrapper import PikafishEngine

with PikafishEngine("pikafish.exe") as engine:
    # 1. 初始化 UCI
    print("Initializing UCI...")
    uci_resp = engine.uci()
    for line in uci_resp:
        print(line)
    
app = Flask(__name__)
CORS(app)

# 定义中国象棋棋子到FEN字符的映射
# (根据您数据中的中文和side)
XIANGQI_PIECE_MAP = {
    'red': {
        '帥': 'K', # 帅 (King)
        '仕': 'A', # 仕 (Advisor)
        '相': 'B', # 相 (Elephant/Bishop)
        '馬': 'N', # 马 (Horse/Knight)
        '車': 'R', # 车 (Rook/Chariot)
        '炮': 'C', # 炮 (Cannon)
        '兵': 'P', # 兵 (Pawn)
    },
    'black': {
        '將': 'k', # 将 (General)
        '士': 'a', # 士 (Advisor)
        '象': 'b', # 象 (Elephant)
        '馬': 'n', # 马 (Horse)
        '車': 'r', # 车 (Rook)
        '炮': 'c', # 炮 (Cannon)
        '卒': 'p', # 卒 (Pawn)
    }
}

def convert_to_fen(board_data: list, side_to_move: str) -> str:
    """
    将前端的棋盘数据结构转换为中国象棋的FEN字符串。

    :param board_data: 10x9 的列表，[0] 为红方底线, [9] 为黑方底线。
    :param side_to_move: 'red' 或 'black'。
    :return: FEN 字符串。
    """
    
    fen_rows = []
    
    # FEN 从棋盘顶部 (黑方) 开始，所以我们倒序遍历 board_data
    # reversed(board_data) 会从 index 9 遍历到 index 0
    for row in reversed(board_data):
        fen_row_str = ""
        empty_squares = 0
        
        # 遍历行中的每一格 (9列)
        for cell in row:
            if cell is None:
                empty_squares += 1
            else:
                # 如果有连续的空格，先添加数字
                if empty_squares > 0:
                    fen_row_str += str(empty_squares)
                    empty_squares = 0
                
                # 添加棋子字符
                try:
                    piece_side = cell['side']
                    piece_type = cell['type']
                    fen_char = XIANGQI_PIECE_MAP[piece_side][piece_type]
                    fen_row_str += fen_char
                except KeyError:
                    raise ValueError(f"未知的棋子类型或阵营: {cell}")

        # 处理行尾的连续空格
        if empty_squares > 0:
            fen_row_str += str(empty_squares)
            
        fen_rows.append(fen_row_str)

    # 1. 棋盘部分：用 '/' 连接每一行
    board_fen = "/".join(fen_rows)

    # 2. 走棋方： 'red' -> 'w', 'black' -> 'b'
    side_fen = 'w' if side_to_move == 'red' else 'b'
    
    # 3. 其他FEN部分 (吃过路兵、回合数等)
    # 对于中国象棋，通常用 "- - 0 1" 作为默认值
    remaining_fen = "- - 0 1"

    return f"{board_fen} {side_fen} {remaining_fen}"

def ucci_move_to_coord(ucci_move: str) -> dict:
    """
    将 UCCI 着法（如 'h2e2'）转换为坐标。
    假设棋盘：x=0~8 (a~i), y=0~9 (0~9)
    """
    if len(ucci_move) < 4:
        raise ValueError(f"无效 UCCI 着法: {ucci_move}")
    
    from_sq = ucci_move[0:2]  # e.g., "h2"
    to_sq = ucci_move[2:4]    # e.g., "e2"

    def sq_to_coord(sq):
        file, rank = sq[0], sq[1]
        if file not in 'abcdefghi' or rank not in '0123456789':
            raise ValueError(f"无效中国象棋格: {sq}")
        x = ord(file) - ord('a')  # a=0, ..., i=8
        y = int(rank)             # 0~9 直接对应 y
        return {'x': x, 'y': y}

    return {
        'from': sq_to_coord(from_sq),
        'to': sq_to_coord(to_sq)
    }

@app.route('/ai_move', methods=['POST'])
def ai_move():    
    # log.info(f"/ai_move {request.method} data={request.json}")
    
    data = request.json
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    if 'board' not in data:
        return jsonify({"error": "Missing 'board' field"}), 400
    if 'side' not in data:
        return jsonify({"error": "Missing 'side' field"}), 400
        
    board_state = data['board']
    side_to_move = data['side']

    # log(f"[LOG] /ai_move {request.method} data={request.json}")
    fen = convert_to_fen(board_state, side_to_move)
    # is_flipped = is_fen_flipped(fen)
    # if is_flipped:  # 翻转棋盘
    #     fen = flip_xiangqi_fen(fen)
    # log(f"[LOG] FEN: {fen}")
    move = engine.get_bestmove(
        fen=fen,
        movetime=1500
    )
    log(f"[LOG] Best move: {move}")
    best_move = ucci_move_to_coord(move)
    # if is_flipped:  # 翻转回正
    #     #坐标左右、上下翻转
    #     best_move = {
    #         'from': {'x': 8 - best_move['from']['x'], 'y': 9 - best_move['from']['y']},
    #         'to': {'x': 8 - best_move['to']['x'], 'y': 9 - best_move['to']['y']}
    #     }
    log(f"[LOG] Best move: {best_move}")
  

    # if side_to_move == 'red':
    #     # best_move = minimax_root(board_state, side_to_move)
    #     best_move = find_best_move_c2(board_state, side_to_move)
    # else:
    #     best_move = find_best_move_c(board_state, side_to_move)
    
    # # best_move = nn_interface(board_state, side_to_move)
    # log(f"Best move: {best_move}")
    return jsonify(best_move)    
    

# from nn_interface import NN_Interface
# from train import mcts_policy
# from nn_data_representation import board_to_tensor, MOVE_TO_INDEX
# import torch
import random

# nn_player = NN_Interface(model_path="ckpt/latest.pth") 
# def nn_interface(board_state, side):      
    # pi, v = mcts_policy(nn_player, board_state, side, temperature=0)  
    # tensor, pi_vec = board_to_tensor(board_state, side).squeeze(0), torch.zeros(len(MOVE_TO_INDEX))
    # for move, prob in pi.items():
    #     key = (move.fy, move.fx, move.ty, move.tx)
    #     if key in MOVE_TO_INDEX: pi_vec[MOVE_TO_INDEX[key]] = prob   
    # moves, probs = list(pi.keys()), list(pi.values())
    # if not moves: # 确保列表非空
    #     return None
    # move = random.choices(moves, weights=probs)[0]
    # return move.to_dict()

    # _, policy = nn_player.predict(board_state, side)
    # # 按概率从高到低排序并打印
    # sorted_policy = sorted(policy.items(), key=lambda item: item[1], reverse=True)
    # if not sorted_policy: # 确保列表非空
    #     return None 
    # for move, prob in sorted_policy[:5]: # 打印前5个最可能的走法
    #     log(f"  - 走法: {move.to_dict()}, 概率: {prob:.4f}")
    # log(sorted_policy[0][0].to_dict())
    # return sorted_policy[0][0].to_dict()
@app.route('/check_game_over', methods=['POST'])
def check_game_over_endpoint():
    try:
    #   log.info(f"[LOG] /check_game_over {request.method} data={request.json}")
      data = request.json
      if not data:
          return jsonify({"error": "No JSON data provided"}), 400
      
      if 'board' not in data:
          return jsonify({"error": "Missing 'board' field"}), 400
          
      board_state = data['board']
      return jsonify(check_game_over(board_state))
    except Exception as e:
      log(f"Error in /check_game_over: {e}")
      return jsonify({"error": str(e)}), 500

@app.route('/save_position', methods=['POST'])
def save_position():
    """
    保存棋盘位置信息到数据库
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # 检查必需字段
        required_fields = ['board', 'side', 'move']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing '{field}' field"}), 400
        
        board_state = data['board']
        side_to_move = data['side']
        move = data['move']
        
        # 保存位置信息到数据库
        from record_positions import save_position_to_db
        save_position_to_db(board_state, side_to_move, move)
        
        return jsonify({"message": "Position saved successfully"})
    except Exception as e:
        log(f"Error in /save_position: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/opening_stats', methods=['GET'])
def get_opening_stats_endpoint():
    """获取开局库统计信息"""
    try:
        from opening_book import get_opening_stats
        stats = get_opening_stats()
        return jsonify(stats)
    except Exception as e:
        log(f"Error in /opening_stats: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/reset_opening_stats', methods=['POST'])
def reset_opening_stats_endpoint():
    """重置开局库统计信息"""
    try:
        from opening_book import reset_opening_stats
        reset_opening_stats()
        return jsonify({"message": "开局库统计已重置"})
    except Exception as e:
        log(f"Error in /reset_opening_stats: {e}")
        return jsonify({"error": str(e)}), 500

# 创建static目录
static_dir = 'static'
if not os.path.exists(static_dir):
    os.makedirs(static_dir)


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    """处理静态文件请求"""
    # 首先尝试从static目录获取文件
    if os.path.exists(os.path.join(static_dir, filename)):
        return send_from_directory(static_dir, filename)
    # 如果static目录中没有，尝试从根目录获取
    elif os.path.exists(filename):
        return send_from_directory('.', filename)
    else:
        return "File not found", 404


if __name__ == '__main__':
    log("[后端] 启动Flask服务器...")
    app.run(host='0.0.0.0', port=5000, threaded=True)
