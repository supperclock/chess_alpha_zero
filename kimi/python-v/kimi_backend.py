from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from ai import check_game_over,minimax_root
from util import log
from ai_bridge import find_best_move_c
from ai_bridge2 import find_best_move_c2
# from ai_gpt_bridge import find_best_move_c

app = Flask(__name__)
CORS(app)

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
    
    if side_to_move == 'red':
        # best_move = minimax_root(board_state, side_to_move)
        best_move = find_best_move_c2(board_state, side_to_move)
    else:
        best_move = find_best_move_c(board_state, side_to_move)
    
    # best_move = nn_interface(board_state, side_to_move)
    log(f"Best move: {best_move}")
    return jsonify(best_move)    
    

from nn_interface import NN_Interface
# from train import mcts_policy
from nn_data_representation import board_to_tensor, MOVE_TO_INDEX
import torch
import random

nn_player = NN_Interface(model_path="ckpt/latest.pth") 
def nn_interface(board_state, side):      
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

    _, policy = nn_player.predict(board_state, side)
    # 按概率从高到低排序并打印
    sorted_policy = sorted(policy.items(), key=lambda item: item[1], reverse=True)
    if not sorted_policy: # 确保列表非空
        return None 
    for move, prob in sorted_policy[:5]: # 打印前5个最可能的走法
        print(f"  - 走法: {move.to_dict()}, 概率: {prob:.4f}")
    log(sorted_policy[0][0].to_dict())
    return sorted_policy[0][0].to_dict()
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
