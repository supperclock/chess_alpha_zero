from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# 棋盘初始状态（简化版，后续完善）
INITIAL_BOARD = [
    ["rC", "rM", "rX", "rS", "rJ", "rS", "rX", "rM", "rC"],
    [None, None, None, None, None, None, None, None, None],
    [None, "rP", None, None, None, None, None, "rP", None],
    ["rZ", None, "rZ", None, "rZ", None, "rZ", None, "rZ"],
    [None, None, None, None, None, None, None, None, None],
    [None, None, None, None, None, None, None, None, None],
    ["bZ", None, "bZ", None, "bZ", None, "bZ", None, "bZ"],
    [None, "bP", None, None, None, None, None, "bP", None],
    [None, None, None, None, None, None, None, None, None],
    ["bC", "bM", "bX", "bS", "bJ", "bS", "bX", "bM", "bC"]
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/board', methods=['GET'])
def get_board():
    # TODO: 支持多局面/存盘
    return jsonify({'board': INITIAL_BOARD})

@app.route('/api/move', methods=['POST'])
def move():
    data = request.json
    # TODO: 检查走棋规则，更新棋盘
    # 返回新棋盘和是否走棋成功
    return jsonify({'success': True, 'board': INITIAL_BOARD})

if __name__ == '__main__':
    app.run(debug=True)
