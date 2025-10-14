# py_xq_wrapper.py
import ctypes
from ctypes import c_int, c_double, POINTER, byref
import os

# load compiled lib
_libname = './libxqai.so'  # change path as needed (Windows: .dll)
lib = ctypes.CDLL(_libname)

# c prototypes
lib.xq_minimax_root.argtypes = [POINTER(c_int), c_int, c_double, POINTER(c_int)]
lib.xq_minimax_root.restype = c_int
lib.xq_evaluate.argtypes = [POINTER(c_int)]
lib.xq_evaluate.restype = c_int
lib.xq_init.argtypes = []
lib.xq_init.restype = None

# mapping from your ai.py board cell to C encoding
# ensure matches your piece 'type' and 'side' strings
PIECE_CODE = {
    # black positive
    ('車','black'): 1, ('俥','black'):1, ('车','black'):1,
    ('馬','black'):2, ('傌','black'):2, ('马','black'):2,
    ('相','black'):3, ('象','black'):3,
    ('仕','black'):4, ('士','black'):4,
    ('將','black'):5, ('将','black'):5, ('帥','black'):5,
    ('炮','black'):6, ('砲','black'):6,
    ('卒','black'):7, ('兵','black'):7,
    # red negative
    ('車','red'): -1, ('俥','red'):-1, ('车','red'):-1,
    ('馬','red'):-2, ('傌','red'):-2, ('马','red'):-2,
    ('相','red'):-3, ('象','red'):-3,
    ('仕','red'):-4, ('士','red'):-4,
    ('將','red'):-5, ('将','red'):-5, ('帥','red'):-5,
    ('炮','red'):-6, ('砲','red'):-6,
    ('卒','red'):-7, ('兵','red'):-7,
}

def board_to_c_array(board_state):
    arr = (c_int * 90)()
    for y in range(10):
        for x in range(9):
            cell = board_state[y][x]
            code = 0
            if cell is not None:
                t = cell.get('type')
                s = cell.get('side')
                code = PIECE_CODE.get((t,s), 0)
            arr[y*9 + x] = int(code)
    return arr

# public minimax_root to replace original Python function
def minimax_root(board_state, side, time_limit=5.0):
    """
    board_state: 10x9 list-of-lists from your ai.py
    side: 'black' or 'red'
    time_limit: seconds
    returns: {'from': {'y': fy,'x':fx}, 'to': {'y':ty,'x':tx}} or None
    """
    lib.xq_init()
    cboard = board_to_c_array(board_state)
    out = (c_int * 4)()
    side_int = 0 if side=='black' else 1
    ok = lib.xq_minimax_root(cboard, side_int, c_double(time_limit), out)
    if not ok:
        return None
    fy, fx, ty, tx = int(out[0]), int(out[1]), int(out[2]), int(out[3])
    return {'from': {'y': fy, 'x': fx}, 'to': {'y': ty, 'x': tx}}
