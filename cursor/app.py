from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json

app = Flask(__name__, template_folder='templates')
CORS(app)

class ChineseChess:
    def __init__(self):
        # 初始化棋盘，0表示空位，正数表示红方，负数表示黑方
        # 1-7: 红方 (帅、仕、相、马、车、炮、兵)
        # -1到-7: 黑方 (将、士、象、马、车、炮、卒)
        self.board = [
            [-4, -5, -3, -2, -1, -2, -3, -5, -4],  # 黑方底线
            [0, 0, 0, 0, 0, 0, 0, 0, 0],            # 空行
            [0, -6, 0, 0, 0, 0, 0, -6, 0],          # 黑方炮位
            [-7, 0, -7, 0, -7, 0, -7, 0, -7],       # 黑方兵位
            [0, 0, 0, 0, 0, 0, 0, 0, 0],            # 空行
            [0, 0, 0, 0, 0, 0, 0, 0, 0],            # 空行
            [7, 0, 7, 0, 7, 0, 7, 0, 7],            # 红方兵位
            [0, 6, 0, 0, 0, 0, 0, 6, 0],            # 红方炮位
            [0, 0, 0, 0, 0, 0, 0, 0, 0],            # 空行
            [4, 5, 3, 2, 1, 2, 3, 5, 4]             # 红方底线
        ]
        self.current_player = 1  # 1表示红方，-1表示黑方
        self.game_over = False
        self.winner = None
        
    def get_piece_name(self, piece):
        """获取棋子名称"""
        piece_names = {
            1: "帅", 2: "仕", 3: "相", 4: "马", 5: "车", 6: "炮", 7: "兵",
            -1: "将", -2: "士", -3: "象", -4: "马", -5: "车", -6: "炮", -7: "卒"
        }
        return piece_names.get(piece, "")
    
    def is_valid_position(self, x, y):
        """检查位置是否在棋盘内"""
        return 0 <= x < 10 and 0 <= y < 9
    
    def is_own_piece(self, piece, player):
        """检查是否是己方棋子"""
        if player == 1:
            return piece > 0
        else:
            return piece < 0
    
    def is_enemy_piece(self, piece, player):
        """检查是否是敌方棋子"""
        if player == 1:
            return piece < 0
        else:
            return piece > 0
    
    def is_empty(self, piece):
        """检查位置是否为空"""
        return piece == 0
    
    def get_piece_type(self, piece):
        """获取棋子类型（绝对值）"""
        return abs(piece)
    
    def is_in_palace(self, x, y, player):
        """检查是否在九宫格内"""
        if player == 1:  # 红方九宫格
            return 7 <= x <= 9 and 3 <= y <= 5
        else:  # 黑方九宫格
            return 0 <= x <= 2 and 3 <= y <= 5
    
    def is_valid_move(self, from_x, from_y, to_x, to_y):
        """检查移动是否合法"""
        if not self.is_valid_position(from_x, from_y) or not self.is_valid_position(to_x, to_y):
            return False
        
        if from_x == to_x and from_y == to_y:
            return False
        
        piece = self.board[from_x][from_y]
        target = self.board[to_x][to_y]
        
        # 检查是否移动己方棋子
        if not self.is_own_piece(piece, self.current_player):
            return False
        
        # 检查目标位置是否是己方棋子
        if self.is_own_piece(target, self.current_player):
            return False
        
        piece_type = self.get_piece_type(piece)
        
        # 根据棋子类型检查移动规则
        if piece_type == 1:  # 帅/将
            return self.is_valid_king_move(from_x, from_y, to_x, to_y)
        elif piece_type == 2:  # 仕/士
            return self.is_valid_advisor_move(from_x, from_y, to_x, to_y)
        elif piece_type == 3:  # 相/象
            return self.is_valid_elephant_move(from_x, from_y, to_x, to_y)
        elif piece_type == 4:  # 马
            return self.is_valid_horse_move(from_x, from_y, to_x, to_y)
        elif piece_type == 5:  # 车
            return self.is_valid_rook_move(from_x, from_y, to_x, to_y)
        elif piece_type == 6:  # 炮
            return self.is_valid_cannon_move(from_x, from_y, to_x, to_y)
        elif piece_type == 7:  # 兵/卒
            return self.is_valid_pawn_move(from_x, from_y, to_x, to_y)
        
        return False
    
    def is_valid_king_move(self, from_x, from_y, to_x, to_y):
        """帅/将的移动规则"""
        # 必须在九宫格内
        if not self.is_in_palace(to_x, to_y, self.current_player):
            return False
        
        # 只能移动一格
        if abs(from_x - to_x) + abs(from_y - to_y) != 1:
            return False
        
        return True
    
    def is_valid_advisor_move(self, from_x, from_y, to_x, to_y):
        """仕/士的移动规则"""
        # 必须在九宫格内
        if not self.is_in_palace(to_x, to_y, self.current_player):
            return False
        
        # 只能斜向移动一格
        if abs(from_x - to_x) != 1 or abs(from_y - to_y) != 1:
            return False
        
        return True
    
    def is_valid_elephant_move(self, from_x, from_y, to_x, to_y):
        """相/象的移动规则"""
        # 不能过河
        if self.current_player == 1 and to_x < 5:
            return False
        if self.current_player == -1 and to_x > 4:
            return False
        
        # 斜向移动两格
        if abs(from_x - to_x) != 2 or abs(from_y - to_y) != 2:
            return False
        
        # 检查象眼是否被堵
        eye_x = (from_x + to_x) // 2
        eye_y = (from_y + to_y) // 2
        if not self.is_empty(self.board[eye_x][eye_y]):
            return False
        
        return True
    
    def is_valid_horse_move(self, from_x, from_y, to_x, to_y):
        """马的移动规则"""
        dx = abs(from_x - to_x)
        dy = abs(from_y - to_y)
        
        # 马走日字
        if not ((dx == 2 and dy == 1) or (dx == 1 and dy == 2)):
            return False
        
        # 检查马腿是否被堵
        if dx == 2:  # 上下移动
            if from_x < to_x:  # 向下
                if not self.is_empty(self.board[from_x + 1][from_y]):
                    return False
            else:  # 向上
                if not self.is_empty(self.board[from_x - 1][from_y]):
                    return False
        else:  # 左右移动
            if from_y < to_y:  # 向右
                if not self.is_empty(self.board[from_x][from_y + 1]):
                    return False
            else:  # 向左
                if not self.is_empty(self.board[from_x][from_y - 1]):
                    return False
        
        return True
    
    def is_valid_rook_move(self, from_x, from_y, to_x, to_y):
        """车的移动规则"""
        # 只能直线移动
        if from_x != to_x and from_y != to_y:
            return False
        
        # 检查路径是否被堵
        if from_x == to_x:  # 水平移动
            start_y = min(from_y, to_y)
            end_y = max(from_y, to_y)
            for y in range(start_y + 1, end_y):
                if not self.is_empty(self.board[from_x][y]):
                    return False
        else:  # 垂直移动
            start_x = min(from_x, to_x)
            end_x = max(from_x, to_x)
            for x in range(start_x + 1, end_x):
                if not self.is_empty(self.board[x][from_y]):
                    return False
        
        return True
    
    def is_valid_cannon_move(self, from_x, from_y, to_x, to_y):
        """炮的移动规则"""
        # 只能直线移动
        if from_x != to_x and from_y != to_y:
            return False
        
        # 检查路径
        if from_x == to_x:  # 水平移动
            start_y = min(from_y, to_y)
            end_y = max(from_y, to_y)
            pieces_in_path = 0
            for y in range(start_y + 1, end_y):
                if not self.is_empty(self.board[from_x][y]):
                    pieces_in_path += 1
        else:  # 垂直移动
            start_x = min(from_x, to_x)
            end_x = max(from_x, to_x)
            pieces_in_path = 0
            for x in range(start_x + 1, end_x):
                if not self.is_empty(self.board[x][from_y]):
                    pieces_in_path += 1
        
        # 如果目标位置有棋子，必须翻山（中间有一个棋子）
        if not self.is_empty(self.board[to_x][to_y]):
            return pieces_in_path == 1
        else:
            # 如果目标位置为空，路径必须为空
            return pieces_in_path == 0
    
    def is_valid_pawn_move(self, from_x, from_y, to_x, to_y):
        """兵/卒的移动规则"""
        if self.current_player == 1:  # 红方
            if from_x > 4:  # 未过河
                return to_x == from_x - 1 and to_y == from_y
            else:  # 已过河
                return (to_x == from_x - 1 and to_y == from_y) or \
                       (to_x == from_x and abs(to_y - from_y) == 1)
        else:  # 黑方
            if from_x < 5:  # 未过河
                return to_x == from_x + 1 and to_y == from_y
            else:  # 已过河
                return (to_x == from_x + 1 and to_y == from_y) or \
                       (to_x == from_x and abs(to_y - from_y) == 1)
    
    def make_move(self, from_x, from_y, to_x, to_y):
        """执行移动"""
        if not self.is_valid_move(from_x, from_y, to_x, to_y):
            return False
        
        # 记录移动
        piece = self.board[from_x][from_y]
        captured = self.board[to_x][to_y]
        
        # 执行移动
        self.board[to_x][to_y] = piece
        self.board[from_x][from_y] = 0
        
        # 检查游戏是否结束
        self.check_game_over()
        
        # 切换玩家
        self.current_player *= -1
        
        return True
    
    def check_game_over(self):
        """检查游戏是否结束"""
        # 检查是否有帅/将
        red_king = False
        black_king = False
        
        for row in self.board:
            for piece in row:
                if piece == 1:
                    red_king = True
                elif piece == -1:
                    black_king = True
        
        if not red_king:
            self.game_over = True
            self.winner = -1  # 黑方获胜
        elif not black_king:
            self.game_over = True
            self.winner = 1   # 红方获胜
    
    def get_board_state(self):
        """获取棋盘状态"""
        return {
            'board': self.board,
            'current_player': self.current_player,
            'game_over': self.game_over,
            'winner': self.winner
        }

# 创建游戏实例
game = ChineseChess()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/board', methods=['GET'])
def get_board():
    return jsonify(game.get_board_state())

@app.route('/api/move', methods=['POST'])
def make_move():
    data = request.json
    from_x = data.get('from_x')
    from_y = data.get('from_y')
    to_x = data.get('to_x')
    to_y = data.get('to_y')
    
    if game.make_move(from_x, from_y, to_x, to_y):
        return jsonify({'success': True, 'board': game.get_board_state()})
    else:
        return jsonify({'success': False, 'message': 'Invalid move'})

@app.route('/api/reset', methods=['POST'])
def reset_game():
    global game
    game = ChineseChess()
    return jsonify({'success': True, 'board': game.get_board_state()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
