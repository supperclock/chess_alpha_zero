from flask import Flask, request, jsonify, Response, send_from_directory, stream_with_context
from flask_cors import CORS
import requests
import os
from ai import minimax_root, check_game_over
import logging

logging.basicConfig(
    filename='kimi_backend.log',
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)

app = Flask(__name__)
CORS(app)

@app.route('/ai_move', methods=['POST'])
def ai_move():
    try:
      logging.info(f"/ai_move {request.method} data={request.json}")
      data = request.json
      board_state = data['board']
      side_to_move = data['side']
        
      depth = 4
      best_move = minimax_root(board_state, depth, side_to_move)
      return jsonify(best_move)
    except Exception as e:
      logging.error(f"Error in /ai_move: {e}")
      return jsonify({"error": str(e)}), 500

@app.route('/check_game_over', methods=['POST'])
def check_game_over_endpoint():
    try:
      logging.info(f"[LOG] /check_game_over {request.method} data={request.json}")
      data = request.json
      data = request.json
      board_state = data['board']
      return jsonify(check_game_over(board_state))
    except Exception as e:
      logging.error(f"Error in /check_game_over: {e}")
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
