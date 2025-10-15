import ctypes
import os
import platform

# --- 1. 加载编译好的 C 共享库 ---

def _load_library():
    """根据操作系统加载正确的共享库文件。"""
    lib_name = "gemini"
    if platform.system() == "Windows":
        lib_ext = ".dll"
    else:
        lib_ext = ".so"
    
    # 假设 C 引擎编译后的库在 'c_engine' 子目录中
    lib_path = os.path.join(os.path.dirname(__file__), 'c_engine/gemini', lib_name + lib_ext)
    
    try:
        return ctypes.CDLL(lib_path)
    except OSError as e:
        print(f"Error: Could not load the C library from {lib_path}")
        print("Please ensure you have compiled the C code first.")
        print(f"Compilation command for your system:")
        if platform.system() == "Windows":
            print("gcc -shared -o c_engine\\chess_engine.dll c_engine\\*.c -O3")
        else:
            print("gcc -shared -o c_engine/chess_engine.so -fPIC c_engine/*.c -O3")
        raise e

c_engine = _load_library()

# --- 2. 在 Python 中重新定义 C 的枚举和结构体 ---

# C enum 'Piece'
class Piece:
    EMPTY, r_king, r_advisor, r_elephant, r_horse, r_chariot, r_cannon, r_pawn, \
    b_king, b_advisor, b_elephant, b_horse, b_chariot, b_cannon, b_pawn = range(15)

# C enum 'Side'
class Side:
    RED, BLACK = range(2)

# C struct 'Move'
class CMove(ctypes.Structure):
    _fields_ = [("from_y", ctypes.c_int),
                ("from_x", ctypes.c_int),
                ("to_y", ctypes.c_int),
                ("to_x", ctypes.c_int),
                ("captured", ctypes.c_int), # This is a 'Piece' enum
                ("score", ctypes.c_int)]

# C struct 'BoardState'
class CBoardState(ctypes.Structure):
    _fields_ = [("board", (ctypes.c_int * 9) * 10), # (c_int * COLS) * ROWS
                ("side_to_move", ctypes.c_int)] # This is a 'Side' enum

# --- 3. 定义 C 函数的原型 (参数类型和返回类型) ---

# void init_board_from_initial_setup(BoardState* state);
c_engine.init_board_from_initial_setup.argtypes = [ctypes.POINTER(CBoardState)]
c_engine.init_board_from_initial_setup.restype = None

# void print_board(const BoardState* state);
c_engine.print_board.argtypes = [ctypes.POINTER(CBoardState)]
c_engine.print_board.restype = None

# Move find_best_move(BoardState* state, int max_depth, double time_limit);
c_engine.find_best_move.argtypes = [ctypes.POINTER(CBoardState), ctypes.c_int, ctypes.c_double]
c_engine.find_best_move.restype = CMove


# --- 4. 编写 Python <=> C 的数据转换函数 ---

# Python 字符串表示到 C 枚举整数的映射
PIECE_MAP = {
    'red': {
        '帥': Piece.r_king, '仕': Piece.r_advisor, '相': Piece.r_elephant,
        '馬': Piece.r_horse, '車': Piece.r_chariot, '炮': Piece.r_cannon,
        '兵': Piece.r_pawn
    },
    'black': {
        '將': Piece.b_king, '士': Piece.b_advisor, '象': Piece.b_elephant,
        '馬': Piece.b_horse, '車': Piece.b_chariot, '炮': Piece.b_cannon,
        '卒': Piece.b_pawn
    }
}

def python_to_c_board(board_py, side_py):
    """将 Python 的棋盘表示 (字典列表) 转换为 C 的 BoardState 结构体。"""
    c_board = CBoardState()
    c_board.side_to_move = Side.RED if side_py == 'red' else Side.BLACK
    
    for r in range(10):
        for c in range(9):
            piece_py = board_py[r][c]
            if piece_py:
                side = piece_py['side']
                piece_type = piece_py['type']
                # 兼容 Python 代码中的多种棋子名称
                if piece_type in ('俥', '车'): piece_type = '車'
                if piece_type in ('傌', '马'): piece_type = '馬'
                if piece_type in ('砲'): piece_type = '炮'
                
                c_board.board[r][c] = PIECE_MAP[side][piece_type]
            else:
                c_board.board[r][c] = Piece.EMPTY
    return c_board

def c_move_to_python_dict(c_move):
    """将 C 的 Move 结构体转换为 Python 的字典格式。"""
    if c_move.from_x == c_move.to_x and c_move.from_y == c_move.to_y: return None
    return {
        'from': {'y': c_move.from_y, 'x': c_move.from_x},
        'to': {'y': c_move.to_y, 'x': c_move.to_x}
    }

# --- 5. 创建新的顶层AI入口函数 ---

def find_best_move_c(board_state_py, side_py, max_depth=16, time_limit=5):
    """
    这是新的AI入口函数，它将替换掉 Python ai.py 中的 minimax_root。
    
    Args:
        board_state_py: Python格式的棋盘 (字典列表)。
        side_py: 当前走棋方 ('red' or 'black')。
        max_depth: C引擎迭代加深的最大深度。
        time_limit: C引擎的思考时间限制（秒）。

    Returns:
        一个Python字典格式的最佳走法。
    """
    # 步骤1: 将 Python 数据转换为 C 结构体
    c_board_state = python_to_c_board(board_state_py, side_py)
    
    # 步骤2: 调用 C 引擎的核心函数
    best_c_move = c_engine.find_best_move(
        ctypes.byref(c_board_state), 
        ctypes.c_int(max_depth), 
        ctypes.c_double(time_limit)
    )
    
    # 步骤3: 将 C 的返回结果转换为 Python 格式
    return c_move_to_python_dict(best_c_move)


# --- 6. 测试块 ---

if __name__ == '__main__':
    print("--- Testing Python-C Bridge ---")
    
    # 从您的 ai.py 文件中复制初始棋盘定义
    INITIAL_SETUP = [
        [{'type': '車', 'side': 'red'}, {'type': '馬', 'side': 'red'}, {'type': '相', 'side': 'red'}, {'type': '仕', 'side': 'red'}, {'type': '帥', 'side': 'red'}, {'type': '仕', 'side': 'red'}, {'type': '相', 'side': 'red'}, {'type': '馬', 'side': 'red'}, {'type': '車', 'side': 'red'}],
        [None]*9,
        [None, {'type': '炮', 'side': 'red'}, None, None, None, None, None, {'type': '炮', 'side': 'red'}, None],
        [{'type': '兵', 'side': 'red'}, None, {'type': '兵', 'side': 'red'}, None, {'type': '兵', 'side': 'red'}, None, {'type': '兵', 'side': 'red'}, None, {'type': '兵', 'side': 'red'}],
        [None]*9,
        [None]*9,
        [{'type': '卒', 'side': 'black'}, None, {'type': '卒', 'side': 'black'}, None, {'type': '卒', 'side': 'black'}, None, {'type': '卒', 'side': 'black'}, None, {'type': '卒', 'side': 'black'}],
        [None, {'type': '炮', 'side': 'black'}, None, None, None, None, None, {'type': '炮', 'side': 'black'}, None],
        [None]*9,
        [{'type': '車', 'side': 'black'}, {'type': '馬', 'side': 'black'}, {'type': '象', 'side': 'black'}, {'type': '士', 'side': 'black'}, {'type': '將', 'side': 'black'}, {'type': '士', 'side': 'black'}, {'type': '象', 'side': 'black'}, {'type': '馬', 'side': 'black'}, {'type': '車', 'side': 'black'}],
    ]

    print("\n1. Testing C-side board initialization and printing...")
    c_board_test = CBoardState()
    c_engine.init_board_from_initial_setup(ctypes.byref(c_board_test))
    c_engine.print_board(ctypes.byref(c_board_test)) # 这会打印到你的终端
    
    print("\n2. Testing a search from the initial position (RED to move)...")
    time_to_think = 3.0
    print(f"   Thinking for {time_to_think} seconds...")
    
    best_move = find_best_move_c(INITIAL_SETUP, 'red', time_limit=time_to_think)
    
    print("\n--- Test Result ---")
    print(f"C Engine recommended move: {best_move}")
    # 一个常见的开局走法是炮二平五: {'from': {'y': 2, 'x': 1}, 'to': {'y': 2, 'x': 4}}
    # 或者当头炮: {'from': {'y': 2, 'x': 7}, 'to': {'y': 2, 'x': 4}}
    print("Test complete. If you see a valid move, the bridge is working!")