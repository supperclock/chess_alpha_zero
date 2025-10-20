import ctypes
import os
import platform

# ==========================================================
# xqai_bridge.py - Python ↔ C 中国象棋引擎接口
# ==========================================================

def _load_library():
    """根据操作系统加载编译好的象棋引擎动态库。"""
    lib_name = "gpt" if platform.system() != "Windows" else "gpt"
    lib_ext = ".dll" if platform.system() == "Windows" else ".so"
    lib_path = os.path.join(os.path.dirname(__file__), "c_engine/gpt", lib_name + lib_ext)

    try:
        return ctypes.CDLL(lib_path)
    except OSError as e:
        print(f"❌ 无法加载动态库: {lib_path}")
        print("请先编译 C 引擎。例如：")
        if platform.system() == "Windows":
            print("gcc -shared -o c_engine\\xqai.dll c_engine\\*.c -O3")
        else:
            print("gcc -shared -o c_engine/libxqai.so -fPIC c_engine/*.c -O3")
        raise e


c_engine = _load_library()

# ==========================================================
# 一、C 结构体与常量定义
# ==========================================================

ROWS, COLS = 10, 9

class Side:
    RED, BLACK = range(2)

class CMove(ctypes.Structure):
    _fields_ = [
        ("fy", ctypes.c_int),
        ("fx", ctypes.c_int),
        ("ty", ctypes.c_int),
        ("tx", ctypes.c_int),
        ("score", ctypes.c_int),
        ("captured_type", ctypes.c_int),
        ("captured_side", ctypes.c_int),
    ]

class CPiece(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("side", ctypes.c_int)]

class CBoard(ctypes.Structure):
    _fields_ = [("sq", (CPiece * COLS) * ROWS), ("side_to_move", ctypes.c_int)]

# ==========================================================
# 二、C 函数原型定义
# ==========================================================

# void init_board(Board* b);
c_engine.init_board.argtypes = [ctypes.POINTER(CBoard)]
c_engine.init_board.restype = None

# void search_best(const Board* b_in, Move* best_move, int depth, int time_limit_ms);
c_engine.search_best.argtypes = [
    ctypes.POINTER(CBoard),
    ctypes.POINTER(CMove),
    ctypes.c_int,
    ctypes.c_int,
]
c_engine.search_best.restype = None

# int evaluate_board_c(const Board* b);
c_engine.evaluate_board_c.argtypes = [ctypes.POINTER(CBoard)]
c_engine.evaluate_board_c.restype = ctypes.c_int

# ==========================================================
# 三、Python ↔ C 数据转换
# ==========================================================

PIECE_NAME_TO_TYPE = {
    "帥": 6, "將": 6,
    "仕": 5, "士": 5,
    "相": 4, "象": 4,
    "馬": 3, "马": 3,
    "車": 1, "车": 1,
    "炮": 2, "砲": 2,
    "兵": 7, "卒": 7,
}

def python_to_c_board(board_py, side_py):
    """将 Python 的二维棋盘(dict) 转换为 C 的 Board 结构体。"""
    c_board = CBoard()
    c_board.side_to_move = Side.RED if side_py == "red" else Side.BLACK
    for y in range(ROWS):
        for x in range(COLS):
            cell = board_py[y][x]
            if cell:
                p = CPiece()
                p.type = PIECE_NAME_TO_TYPE.get(cell["type"], 0)
                p.side = Side.RED if cell["side"] == "red" else Side.BLACK
                c_board.sq[y][x] = p
            else:
                c_board.sq[y][x] = CPiece(0, 0)
    return c_board


def c_move_to_dict(c_move):
    return {
        "from": {"y": c_move.fy, "x": c_move.fx},
        "to": {"y": c_move.ty, "x": c_move.tx},
        "score": c_move.score,
    }

# ==========================================================
# 四、高层 Python 调用接口
# ==========================================================

def find_best_move_c(board_state_py, side_py, depth=16, time_limit_ms=12000):
    """
    从 Python 调用 C 引擎搜索最佳走法。
    Args:
        board_state_py: Python 的棋盘表示（10x9列表）
        side_py: 当前走棋方 ('red' 或 'black')
        depth: 搜索深度
        time_limit_ms: 时间限制（毫秒）
    """
    board_c = python_to_c_board(board_state_py, side_py)
    best_c_move = CMove()
    c_engine.search_best(ctypes.byref(board_c), ctypes.byref(best_c_move), depth, time_limit_ms)
    return c_move_to_dict(best_c_move)


def evaluate_board_c(board_state_py, side_py):
    board_c = python_to_c_board(board_state_py, side_py)
    return c_engine.evaluate_board_c(ctypes.byref(board_c))

# ==========================================================
# 五、测试模块
# ==========================================================

if __name__ == "__main__":
    print("✅ 测试 Python ↔ C 象棋引擎接口")

    from pprint import pprint

    # 简化版初始布局
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

    print("\n🧠 调用 evaluate_board_c...")
    score = evaluate_board_c(INITIAL_SETUP, "red")
    print("评估分数:", score)

    print("\n🔍 调用 find_best_move_c...")
    move = find_best_move_c(INITIAL_SETUP, "red", depth=6, time_limit_ms=2000)
    pprint(move)
