from flask import Flask, request, jsonify, send_from_directory, stream_with_context
from flask_cors import CORS
import os
from ai import minimax_root, check_game_over,logging


app = Flask(__name__)
CORS(app)


@app.route('/ai_move', methods=['POST'])
def ai_move():    
    # logging.info(f"/ai_move {request.method} data={request.json}")
    data = request.json
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    if 'board' not in data:
        return jsonify({"error": "Missing 'board' field"}), 400
    if 'side' not in data:
        return jsonify({"error": "Missing 'side' field"}), 400
        
    board_state = data['board']
    side_to_move = data['side']
    
    best_move = minimax_root(board_state, side_to_move, time_limit=10)
    # logging.info(f"Best move: {best_move}")
    return jsonify(best_move)
  
    

@app.route('/check_game_over', methods=['POST'])
def check_game_over_endpoint():
    try:
    #   logging.info(f"[LOG] /check_game_over {request.method} data={request.json}")
      data = request.json
      if not data:
          return jsonify({"error": "No JSON data provided"}), 400
      
      if 'board' not in data:
          return jsonify({"error": "Missing 'board' field"}), 400
          
      board_state = data['board']
      return jsonify(check_game_over(board_state))
    except Exception as e:
      logging.error(f"Error in /check_game_over: {e}")
      return jsonify({"error": str(e)}), 500

@app.route('/opening_stats', methods=['GET'])
def get_opening_stats_endpoint():
    """获取开局库统计信息"""
    try:
        from opening_book import get_opening_stats
        stats = get_opening_stats()
        return jsonify(stats)
    except Exception as e:
        logging.error(f"Error in /opening_stats: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/reset_opening_stats', methods=['POST'])
def reset_opening_stats_endpoint():
    """重置开局库统计信息"""
    try:
        from opening_book import reset_opening_stats
        reset_opening_stats()
        return jsonify({"message": "开局库统计已重置"})
    except Exception as e:
        logging.error(f"Error in /reset_opening_stats: {e}")
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
    print("[后端] 启动Flask服务器...", flush=True)
    app.run(host='0.0.0.0', port=5000, threaded=True)
