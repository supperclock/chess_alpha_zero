import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from io import BytesIO

class ChessGame:
    def __init__(self):
        # 初始化棋盘 - 使用中文字符表示棋子
        # 红方在下(南方)，黑方在上(北方)
        self.board = [
            ['r_chariot', 'r_horse', 'r_elephant', 'r_advisor', 'r_general', 'r_advisor', 'r_elephant', 'r_horse', 'r_chariot'],
            ['', '', '', '', '', '', '', '', ''],
            ['', 'r_cannon', '', '', '', '', '', 'r_cannon', ''],
            ['r_soldier', '', 'r_soldier', '', 'r_soldier', '', 'r_soldier', '', 'r_soldier'],
            ['', '', '', '', '', '', '', '', ''],
            ['', '', '', '', '', '', '', '', ''],
            ['b_soldier', '', 'b_soldier', '', 'b_soldier', '', 'b_soldier', '', 'b_soldier'],
            ['', 'b_cannon', '', '', '', '', '', 'b_cannon', ''],
            ['', '', '', '', '', '', '', '', ''],
            ['b_chariot', 'b_horse', 'b_elephant', 'b_advisor', 'b_general', 'b_advisor', 'b_elephant', 'b_horse', 'b_chariot']
        ]
        self.current_player = 'red'  # 红方先走
        self.game_status = 'playing'  # playing, red_won, black_won

    def get_board(self):
        return self.board

    def move_piece(self, from_row, from_col, to_row, to_col):
        # 检查游戏是否仍在进行中
        if self.game_status != 'playing':
            return False, "Game is over"

        # 检查起点是否有棋子
        piece = self.board[from_row][from_col]
        if not piece:
            return False, "No piece at starting position"

        # 检查是否是当前玩家的棋子
        piece_color = piece[0]  # 'r' for red, 'b' for black
        if (piece_color == 'r' and self.current_player != 'red') or \
           (piece_color == 'b' and self.current_player != 'black'):
            return False, "Not your piece"

        # 检查目标位置是否是己方棋子
        target_piece = self.board[to_row][to_col]
        if target_piece and target_piece[0] == piece_color:
            return False, "Cannot capture your own piece"

        # 根据棋子类型检查移动是否合法
        is_valid_move = self.is_valid_move(piece, from_row, from_col, to_row, to_col)
        if not is_valid_move:
            return False, "Invalid move for this piece"

        # 执行移动
        captured_piece = self.board[to_row][to_col]
        self.board[to_row][to_col] = piece
        self.board[from_row][from_col] = ''

        # 检查是否将军或将死
        if self.is_checkmate('red' if self.current_player == 'black' else 'black'):
            self.game_status = 'red_won' if self.current_player == 'red' else 'black_won'

        # 切换玩家
        self.current_player = 'black' if self.current_player == 'red' else 'red'

        return True, "Move successful"

    def is_valid_move(self, piece, from_row, from_col, to_row, to_col):
        # 获取棋子类型（去掉颜色前缀）
        piece_type = piece[2:]

        # 检查目标位置是否在棋盘范围内
        if not (0 <= to_row <= 9 and 0 <= to_col <= 8):
            return False

        # 根据不同棋子类型检查移动规则
        if piece_type == 'general':
            return self._is_valid_general_move(from_row, from_col, to_row, to_col)
        elif piece_type == 'advisor':
            return self._is_valid_advisor_move(from_row, from_col, to_row, to_col)
        elif piece_type == 'elephant':
            return self._is_valid_elephant_move(from_row, from_col, to_row, to_col)
        elif piece_type == 'horse':
            return self._is_valid_horse_move(from_row, from_col, to_row, to_col)
        elif piece_type == 'chariot':
            return self._is_valid_chariot_move(from_row, from_col, to_row, to_col)
        elif piece_type == 'cannon':
            return self._is_valid_cannon_move(from_row, from_col, to_row, to_col)
        elif piece_type == 'soldier':
            return self._is_valid_soldier_move(from_row, from_col, to_row, to_col)
        else:
            return False

    def _is_valid_general_move(self, from_row, from_col, to_row, to_col):
        # 将/帅移动规则：只能在九宫格内，每次只能移动一格水平或垂直
        # 红方九宫格：(0,3)-(2,5), 黑方九宫格：(7,3)-(9,5)
        if self.board[from_row][from_col][0] == 'r':
            # 红方将
            if not (0 <= to_row <= 2 and 3 <= to_col <= 5):
                return False
        else:
            # 黑方将
            if not (7 <= to_row <= 9 and 3 <= to_col <= 5):
                return False

        # 只能移动一格水平或垂直
        row_diff = abs(to_row - from_row)
        col_diff = abs(to_col - from_col)
        if not ((row_diff == 1 and col_diff == 0) or (row_diff == 0 and col_diff == 1)):
            return False

        return True

    def _is_valid_advisor_move(self, from_row, from_col, to_row, to_col):
        # 士/仕移动规则：只能在九宫格内斜线移动一格
        if self.board[from_row][from_col][0] == 'r':
            # 红方士
            if not (0 <= to_row <= 2 and 3 <= to_col <= 5):
                return False
        else:
            # 黑方士
            if not (7 <= to_row <= 9 and 3 <= to_col <= 5):
                return False

        # 斜线移动一格
        row_diff = abs(to_row - from_row)
        col_diff = abs(to_col - from_col)
        if not (row_diff == 1 and col_diff == 1):
            return False

        return True

    def _is_valid_elephant_move(self, from_row, from_col, to_row, to_col):
        # 象/相移动规则：斜线走两格，不能过河，不能塞象眼
        # 红方不能过河(>4)，黑方不能过河(<5)
        if self.board[from_row][from_col][0] == 'r' and to_row > 4:
            return False
        if self.board[from_row][from_col][0] == 'b' and to_row < 5:
            return False

        # 斜线走两格
        row_diff = abs(to_row - from_row)
        col_diff = abs(to_col - from_col)
        if not (row_diff == 2 and col_diff == 2):
            return False

        # 检查象眼是否被塞住（移动路径的中心点）
        eye_row = (from_row + to_row) // 2
        eye_col = (from_col + to_col) // 2
        if self.board[eye_row][eye_col]:
            return False

        return True

    def _is_valid_horse_move(self, from_row, from_col, to_row, to_col):
        # 马/馬移动规则：日字形走法，不能蹩马腿
        row_diff = abs(to_row - from_row)
        col_diff = abs(to_col - from_col)
        
        if not ((row_diff == 2 and col_diff == 1) or (row_diff == 1 and col_diff == 2)):
            return False

        # 检查马腿是否被蹩住
        if row_diff == 2:  # 竖着走日字
            leg_row = (from_row + to_row) // 2
            leg_col = from_col
        else:  # 横着走日字
            leg_row = from_row
            leg_col = (from_col + to_col) // 2

        if self.board[leg_row][leg_col]:
            return False

        return True

    def _is_valid_chariot_move(self, from_row, from_col, to_row, to_col):
        # 车/車移动规则：水平或垂直直线移动，路径上不能有棋子
        if from_row != to_row and from_col != to_col:
            return False

        # 检查路径上是否有棋子阻挡
        if from_row == to_row:  # 水平移动
            start_col = min(from_col, to_col)
            end_col = max(from_col, to_col)
            for col in range(start_col + 1, end_col):
                if self.board[from_row][col]:
                    return False
        else:  # 垂直移动
            start_row = min(from_row, to_row)
            end_row = max(from_row, to_row)
            for row in range(start_row + 1, end_row):
                if self.board[row][from_col]:
                    return False

        return True

    def _is_valid_cannon_move(self, from_row, from_col, to_row, to_col):
        # 炮/砲移动规则：水平或垂直直线移动，吃子时必须隔一个棋子
        if from_row != to_row and from_col != to_col:
            return False

        # 计算路径上的棋子数
        pieces_between = 0
        if from_row == to_row:  # 水平移动
            start_col = min(from_col, to_col)
            end_col = max(from_col, to_col)
            for col in range(start_col + 1, end_col):
                if self.board[from_row][col]:
                    pieces_between += 1
        else:  # 垂直移动
            start_row = min(from_row, to_row)
            end_row = max(from_row, to_row)
            for row in range(start_row + 1, end_row):
                if self.board[row][from_col]:
                    pieces_between += 1

        # 移动时不能隔子，吃子时必须隔一个子
        target_piece = self.board[to_row][to_col]
        if target_piece:  # 吃子
            return pieces_between == 1
        else:  # 移动
            return pieces_between == 0

    def _is_valid_soldier_move(self, from_row, from_col, to_row, to_col):
        # 兵/卒移动规则：未过河只能向前，过河后可横移或前移
        piece_color = self.board[from_row][from_col][0]
        
        # 只能移动一格
        row_diff = abs(to_row - from_row)
        col_diff = abs(to_col - from_col)
        if not ((row_diff == 1 and col_diff == 0) or (row_diff == 0 and col_diff == 1)):
            return False

        if piece_color == 'r':  # 红方兵
            # 未过河只能向前(向上移动行数减少)
            if from_row > 4:  # 未过河
                if not (to_row == from_row - 1 and to_col == from_col):
                    return False
            else:  # 已过河
                # 只能向前或横移，不能后退
                if to_row > from_row:
                    return False
        else:  # 黑方卒
            # 未过河只能向前(向下移动行数增加)
            if from_row < 5:  # 未过河
                if not (to_row == from_row + 1 and to_col == from_col):
                    return False
            else:  # 已过河
                # 只能向前或横移，不能后退
                if to_row < from_row:
                    return False

        return True

    def is_checkmate(self, player):
        # 简化版将军检测 - 实际游戏中应实现完整规则
        # 这里仅作示例，返回False
        return False

class ChessHandler(BaseHTTPRequestHandler):
    game = ChessGame()

    def do_GET(self):
        if self.path == '/':
            self.serve_static_file('index.html', 'text/html')
        elif self.path == '/style.css':
            self.serve_static_file('style.css', 'text/css')
        elif self.path == '/script.js':
            self.serve_static_file('script.js', 'application/javascript')
        elif self.path == '/board':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            board_data = {
                'board': self.game.get_board(),
                'current_player': self.game.current_player,
                'game_status': self.game.game_status
            }
            
            self.wfile.write(json.dumps(board_data).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/move':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            move_data = json.loads(post_data.decode())
            
            from_row = move_data['from_row']
            from_col = move_data['from_col']
            to_row = move_data['to_row']
            to_col = move_data['to_col']
            
            success, message = self.game.move_piece(from_row, from_col, to_row, to_col)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {
                'success': success,
                'message': message,
                'current_player': self.game.current_player,
                'game_status': self.game.game_status
            }
            
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def serve_static_file(self, filename, content_type):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.send_response(200)
            self.send_header('Content-type', content_type)
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()

if __name__ == '__main__':
    server = HTTPServer(('localhost', 8000), ChessHandler)
    print('Chess server running on http://localhost:8000')
    server.serve_forever()