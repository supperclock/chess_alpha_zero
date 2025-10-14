import ctypes
import os

# Define Python classes to mirror C structs
class CMove(ctypes.Structure):
    _fields_ = [("from_y", ctypes.c_int),
                ("from_x", ctypes.c_int),
                ("to_y", ctypes.c_int),
                ("to_x", ctypes.c_int),
                ("captured", ctypes.c_int),
                ("score", ctypes.c_int)]

class CBoardState(ctypes.Structure):
    _fields_ = [("board", (ctypes.c_int * 9) * 10),
                ("side_to_move", ctypes.c_int)] # 0 for RED, 1 for BLACK

# Load the compiled C library
lib_path = os.path.join(os.path.dirname(__file__), 'chess_engine.so')
c_engine = ctypes.CDLL(lib_path)

# Define function prototypes for type safety
c_engine.find_best_move.argtypes = [ctypes.POINTER(CBoardState), ctypes.c_int, ctypes.c_double]
c_engine.find_best_move.restype = CMove

def python_to_c_board(board_state_py, side_py):
    """Converts the Python board representation to the C struct."""
    c_board = CBoardState()
    # Map Python piece strings to C integer constants
    piece_map = {'帥': 1, '仕': 2, ...}
    for r in range(10):
        for c in range(9):
            piece = board_state_py[r][c]
            if piece:
                # This assumes a mapping from (type, side) to a single integer
                c_board.board[r][c] = ... # Your mapping logic here
            else:
                c_board.board[r][c] = 0 # EMPTY

    c_board.side_to_move = 0 if side_py == 'red' else 1
    return c_board

def c_move_to_python_dict(c_move):
    """Converts the C Move struct back to a Python dictionary."""
    return {
        'from': {'y': c_move.from_y, 'x': c_move.from_x},
        'to': {'y': c_move.to_y, 'x': c_move.to_x}
    }

# This becomes your new entry point, replacing minimax_root
def find_best_move_c(board_state, side, time_limit):
    # 1. Convert Python board state to C struct
    c_board_state = python_to_c_board(board_state, side)

    # 2. Call the C function
    best_c_move = c_engine.find_best_move(ctypes.byref(c_board_state), 3, time_limit)

    # 3. Convert result back to Python format
    return c_move_to_python_dict(best_c_move)