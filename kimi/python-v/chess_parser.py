import collections
import copy
import sqlite3

# 棋盘的初始布局 (无变化)
INITIAL_SETUP = [
    [{'type': '車', 'side': 'red'}, {'type': '馬', 'side': 'red'}, {'type': '相', 'side': 'red'}, {'type': '仕', 'side': 'red'}, {'type': '帥', 'side': 'red'}, {'type': '仕', 'side': 'red'}, {'type': '相', 'side': 'red'}, {'type': '馬', 'side': 'red'}, {'type': '車', 'side': 'red'}],
    [None, None, None, None, None, None, None, None, None],
    [None, {'type': '炮', 'side': 'red'}, None, None, None, None, None, {'type': '炮', 'side': 'red'}, None],
    [{'type': '兵', 'side': 'red'}, None, {'type': '兵', 'side': 'red'}, None, {'type': '兵', 'side': 'red'}, None, {'type': '兵', 'side': 'red'}, None, {'type': '兵', 'side': 'red'}],
    [None, None, None, None, None, None, None, None, None],
    [None, None, None, None, None, None, None, None, None],
    [{'type': '卒', 'side': 'black'}, None, {'type': '卒', 'side': 'black'}, None, {'type': '卒', 'side': 'black'}, None, {'type': '卒', 'side': 'black'}, None, {'type': '卒', 'side': 'black'}],
    [None, {'type': '炮', 'side': 'black'}, None, None, None, None, None, {'type': '炮', 'side': 'black'}, None],
    [None, None, None, None, None, None, None, None, None],
    [{'type': '車', 'side': 'black'}, {'type': '馬', 'side': 'black'}, {'type': '象', 'side': 'black'}, {'type': '士', 'side': 'black'}, {'type': '將', 'side': 'black'}, {'type': '士', 'side': 'black'}, {'type': '象', 'side': 'black'}, {'type': '馬', 'side': 'black'}, {'type': '車', 'side': 'black'}],
]

# 初始FEN字符串（红方在上)
INITIAL_FEN_W = "RHEAKAEHR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rheakaehr w - - 0 1"

PIECE_MAP = {
    # Red Pieces
    'R': '車', 'H': '馬', 'E': '相', 'A': '仕', 'K': '帥', 'C': '炮', 'P': '兵',
    # Black Pieces
    'r': '車', 'h': '馬', 'e': '象', 'a': '士', 'k': '將', 'c': '炮', 'p': '卒',
    # Alternative notation (sometimes N for Knight, B for Bishop/Elephant)
    'B': '相', 'N': '馬', 
    'b': '象', 'n': '馬'
}
def parse_fen(fen):
    """
    解析FEN字符串，根据要求，此版本将左右方向进行翻转。
    FEN字符串中靠左的棋子将被放置在棋盘的右侧（i路）。
    """
    parts = fen.strip().split()
    if not parts:
        raise ValueError("FEN字符串为空。")

    board_fen = parts[0]
    rows = board_fen.split('/')

    if len(rows) != 10:
        raise ValueError(f"FEN格式错误：应有10行，实际为 {len(rows)} 行。")

    board = [[None for _ in range(9)] for _ in range(10)]

    # 自动检测FEN的方向（红上或黑上）
    # first_piece_char = next((char for char in rows[0] if char.isalpha()), None)
    # is_standard_fen = first_piece_char is not None and first_piece_char.islower()
    is_standard_fen = True
    
    for y, row_str in enumerate(rows):
        # 确定正确的垂直方向（行）索引
        board_y = (9 - y) if is_standard_fen else y
        
        # --- 核心修改：先生成再翻转 ---
        temp_row = []
        for ch in row_str:
            if ch.isdigit():
                # 扩展N个空位
                temp_row.extend([None] * int(ch))
            else:
                if ch in PIECE_MAP:
                    piece_type = PIECE_MAP[ch]
                    side = 'red' if ch.isupper() else 'black'
                    piece = {'type': piece_type, 'side': side}
                    temp_row.append(piece)
        
        # 检查临时行长度是否正确
        if len(temp_row) != 9:
            raise ValueError(f"FEN行解析错误: '{row_str}' -> 长度不为9")

        # 将解析出的行翻转后存入棋盘
        board[board_y] = temp_row[::-1]
    
    # 解析走棋方
    current_player = 'red'
    if len(parts) > 1 and parts[1] == 'b':
        current_player = 'black'
    
    return board, current_player

Move = collections.namedtuple('Move', ['from_x', 'from_y', 'to_x', 'to_y'])

class XiangqiGame:
    def __init__(self, init_fen=''):
        if init_fen == '':
            self.board = copy.deepcopy(INITIAL_SETUP)
            self.current_player = 'red'
        else:
            self.board, self.current_player = parse_fen(init_fen)
            # self.print_board()

    def print_board(self):
        print("   a  b  c  d  e  f  g  h  i  (列)")
        print(" --------------------------")
        for y, row in enumerate(self.board):
            row_str = f"{y}|"
            for piece in row:
                if piece is None:
                    row_str += " ・ "
                else:
                    piece_char = piece['type']
                    if piece['side'] == 'black':
                        row_str += f"\033[1m {piece_char} \033[0m"
                    else:
                        row_str += f"\033[91m {piece_char} \033[0m"
            print(row_str)
        print(" --------------------------")
        print(f"当前走棋方: {self.current_player}")

    def _parse_iccs(self, iccs_str):
        """
        ICCS 坐标解析
        例如: 'b2e2' -> (1,2) -> (4,2)
        """
        if len(iccs_str) != 4:
            raise ValueError("ICCS 走法必须是4个字符，比如 'b2e2'")
        
        cols = "abcdefghi"
        try:
            from_x = 8 - cols.index(iccs_str[0]) 
            from_y = int(iccs_str[1])
            to_x   = 8 - cols.index(iccs_str[2])  
            to_y   = int(iccs_str[3])
        except Exception:
            raise ValueError(f"非法 ICCS 字符串: {iccs_str}")
        
        return Move(from_x, from_y, to_x, to_y)

    def move(self, iccs, game_id=None):
        # print(f"\n{game_id}执行走棋: {self.current_player} -> '{iccs}'")        
        move_coords = self._parse_iccs(iccs)
        from_x, from_y, to_x, to_y = move_coords
        piece_to_move = self.board[from_y][from_x]
        if piece_to_move is None:
            raise RuntimeError(f"起点 {iccs[:2]} 没有棋子！")
        
        target_piece = self.board[to_y][to_x]
        # if target_piece:
            # print(f"  吃子: {piece_to_move['type']} 吃掉 {target_piece['type']}")
        
        self.board[to_y][to_x] = piece_to_move
        self.board[from_y][from_x] = None
        self.current_player = 'black' if self.current_player == 'red' else 'red'
        # print(f"  成功: {move_coords}")
        return True, move_coords
        

    def process_moves(self, iccs_list, game_id=None):
        for iccs in iccs_list:
            ok, _ = self.move(iccs, game_id)
            if not ok:
                print("由于上一步走棋失败，棋局终止。")
                break
            #等待用户输入回车
            input("请按回车继续...")
            self.print_board()


# --- 使用示例 ---
if __name__ == '__main__':

    conn = sqlite3.connect('chess_games.db')
    cursor = conn.cursor()

    sql = """
        SELECT game_id, result,init_fen FROM games where game_id=1602
    """
    cursor.execute(sql)
    games = cursor.fetchall()    
    
    # Process each game
    for game_id, result,init_fen in games:                
        game = XiangqiGame(init_fen)    
        print("--- 初始棋盘 ---")
        game.print_board()
        # 从数据库获取走子 (假设存的是 ICCS 字符串)
        cursor.execute('''
            SELECT iccs
            FROM moves 
            WHERE game_id = ? 
            ORDER BY move_index
        ''', (1602,))
            
        moves = [row[0] for row in cursor.fetchall()]
            
        print(f"\n--- 从数据库读取到 {len(moves)} 步棋 ---")
        
        # 按顺序走棋
        game.process_moves(moves, game_id=201)
        
        print("\n--- 最终棋盘 ---")
        game.print_board()