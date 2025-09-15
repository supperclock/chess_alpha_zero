from flask import Flask, request, jsonify, send_from_directory
import os

app = Flask(__name__, static_folder=os.path.dirname(os.path.abspath(__file__)))

@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'index.html')

# 初始化棋盘状态
initial_board = [
    ["车", "马", "相", "士", "帅", "士", "相", "马", "车"],
    ["", "", "", "", "", "", "", "", ""],
    ["", "炮", "", "", "", "", "", "炮", ""],
    ["兵", "", "兵", "", "兵", "", "兵", "", "兵"],
    ["", "", "", "", "", "", "", "", ""],
    ["", "", "", "", "", "", "", "", ""],
    ["卒", "", "卒", "", "卒", "", "卒", "", "卒"],
    ["", "炮", "", "", "", "", "", "炮", ""],
    ["", "", "", "", "", "", "", "", ""],
    ["车", "马", "象", "仕", "将", "仕", "象", "马", "车"]
]

# 当前棋盘状态
current_board = [row[:] for row in initial_board]

@app.route('/move', methods=['POST'])
def move_piece():
    data = request.json
    from_pos = data.get('from')
    to_pos = data.get('to')
    
    # 检查移动是否合法（待实现）
    # 更新棋盘状态
    piece = current_board[from_pos[0]][from_pos[1]]
    current_board[from_pos[0]][from_pos[1]] = ""
    current_board[to_pos[0]][to_pos[1]] = piece
    
    return jsonify({"status": "success", "board": current_board})

@app.route('/reset', methods=['POST'])
def reset_board():
    global current_board
    current_board = [row[:] for row in initial_board]
    return jsonify({"status": "success", "board": current_board})

if __name__ == '__main__':
    app.run(debug=True)