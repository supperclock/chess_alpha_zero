# py_search_wrapper_efficient.py
# 高效的 ctypes wrapper：复用 Move 对象 + captured_stack (LIFO)
# 适配 search_core C 库（提供 search_root / compute_eval_c）
#
# 依赖：ai.py 中定义：
#   - generate_moves(board_state, side) -> list of Move-like objects with .fy/.fx/.ty/.tx/.score (score 可选)
#   - make_move(board_state, move) -> returns captured object or None
#   - unmake_move(board_state, move, captured) -> restore
#   - board_state is a 10x9 list-of-lists with either None or {'type':..., 'side':...}
#
# 用法示例在文件底部

import ctypes
from ctypes import c_int, c_void_p, Structure, POINTER, CFUNCTYPE, py_object
import time
import ai  # 你原有的 Python 引擎文件，包含 generate_moves/make_move/unmake_move/evaluate_board

# ---------- load C library ----------
# 修改路径为你的库文件名（Linux: .so，Windows: .dll）
lib = ctypes.CDLL('./libsearch_core.so')  # or 'search_core.dll' on Windows

# ---------- CMove struct definition (must match C side) ----------
class CMove(Structure):
    _fields_ = [("fy", c_int), ("fx", c_int), ("ty", c_int), ("tx", c_int), ("score", c_int)]

# ---------- callback types ----------
MOVEGEN_CB = CFUNCTYPE(c_int, c_void_p, c_int, POINTER(CMove), c_int)
MAKE_CB = CFUNCTYPE(c_int, c_void_p, c_int, c_int, c_int, c_int)
UNMAKE_CB = CFUNCTYPE(None, c_void_p, c_int, c_int, c_int, c_int, c_int)
TIMEUP_CB = CFUNCTYPE(c_int, c_void_p)
SNAPSHOT_CB = CFUNCTYPE(c_int, c_void_p, POINTER(c_int), c_int)

# ---------- Python-side SearchCtx structure (mirror C header) ----------
class SearchCtx(ctypes.Structure):
    _fields_ = [
        ("ctx", c_void_p),
        ("movegen", MOVEGEN_CB),
        ("make_move", MAKE_CB),
        ("unmake_move", UNMAKE_CB),
        ("timeup", TIMEUP_CB),
        ("snapshot", SNAPSHOT_CB),
        ("max_nodes", c_int),
        ("nodes", c_int),
    ]

# ---------- Helper: create a C pointer to a Python object for ctx ----------
def make_ctx_ptr(pyobj):
    # create a stable py_object and return its pointer as c_void_p
    return ctypes.cast(ctypes.pointer(ctypes.py_object(pyobj)), c_void_p)

def pyctx_from_voidp(ctx_ptr):
    if not ctx_ptr:
        return None
    return ctypes.cast(ctx_ptr, ctypes.POINTER(py_object)).contents.value

# ---------- Efficient wrappers state (will be stored in pyctx) ----------
class WrapperState:
    def __init__(self, board):
        self.board = board
        # Reusable Move object for calling ai.make_move/unmake_move to avoid alloc
        self._reuse_move = type("MoveReusable", (), {})()
        self._captured_stack = []  # store captured objects (could be None); LIFO
        # Optionally: store last move for debugging
        self.nodes = 0
        self.start_time = None
        self.time_limit = None

    def make_move_push(self, fy, fx, ty, tx):
        """
        Use reusable move object, call ai.make_move(board, move),
        push captured onto stack and return stack index (int).
        """
        m = self._reuse_move
        m.fy = int(fy)
        m.fx = int(fx)
        m.ty = int(ty)
        m.tx = int(tx)
        # call ai.make_move(board, move) which returns captured obj or None
        captured = ai.make_move(self.board, m)
        # push captured onto stack and return its index
        self._captured_stack.append(captured)
        return len(self._captured_stack) - 1

    def unmake_move_pop(self, fy, fx, ty, tx, captured_id):
        """
        Pop and unmake. We expect LIFO ordering: captured_id should equal last index.
        If not, we try to recover by indexing directly (less efficient but safe).
        """
        m = self._reuse_move
        m.fy = int(fy)
        m.fx = int(fx)
        m.ty = int(ty)
        m.tx = int(tx)

        if not self._captured_stack:
            # nothing to pop - try to call unmake with None
            ai.unmake_move(self.board, m, None)
            return

        last_index = len(self._captured_stack) - 1
        if captured_id == last_index:
            captured = self._captured_stack.pop()
            ai.unmake_move(self.board, m, captured)
        else:
            # mismatch (shouldn't normally happen). Attempt to fetch by id.
            if 0 <= captured_id < len(self._captured_stack):
                captured = self._captured_stack[captured_id]
                # mark slot as used (set to None) to keep indices stable
                self._captured_stack[captured_id] = None
                ai.unmake_move(self.board, m, captured)
                # we won't shrink stack here because indices may still be referenced
            else:
                # fallback: unmake with None
                ai.unmake_move(self.board, m, None)

# ---------- PIECE_CODE mapping for snapshot (covering synonyms) ----------
# MUST match your ai.py's piece 'type' strings and 'side' values ('black'/'red')
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

# ---------- Callback implementations ----------

# movegen_cb: call ai.generate_moves(board, side_str)
@MOVEGEN_CB
def movegen_cb(ctx_ptr, side_int, moves_out_ptr, max_moves):
    pyctx = pyctx_from_voidp(ctx_ptr)
    if pyctx is None:
        return 0
    state = pyctx  # WrapperState object
    board = state.board
    side_str = 'black' if int(side_int) == 0 else 'red'
    # call your ai.generate_moves (which returns a list of move objects)
    try:
        moves_py = ai.generate_moves(board, side_str)
    except Exception as e:
        # if generate_moves expects different signature, adapt accordingly
        # print('movegen exception', e)
        return 0
    n = min(len(moves_py), int(max_moves))
    # fill CMove array
    for i in range(n):
        mv = moves_py[i]
        # assume mv has attributes .fy .fx .ty .tx, otherwise adapt
        moves_out_ptr[i].fy = int(mv.fy)
        moves_out_ptr[i].fx = int(mv.fx)
        moves_out_ptr[i].ty = int(mv.ty)
        moves_out_ptr[i].tx = int(mv.tx)
        # use existing score if present, else 0
        moves_out_ptr[i].score = int(getattr(mv, 'score', 0))
    return n

# make_move callback: use reusable move and push captured onto stack; return captured_id (int)
@MAKE_CB
def make_cb(ctx_ptr, fy, fx, ty, tx):
    pyctx = pyctx_from_voidp(ctx_ptr)
    if pyctx is None:
        return -1
    state = pyctx
    try:
        captured_id = state.make_move_push(fy, fx, ty, tx)
        # return captured_id as int (>=0)
        return int(captured_id)
    except Exception as e:
        # fallback: attempt naive make via ai.make_move
        m = state._reuse_move
        m.fy = int(fy); m.fx = int(fx); m.ty = int(ty); m.tx = int(tx)
        captured = ai.make_move(state.board, m)
        state._captured_stack.append(captured)
        return len(state._captured_stack) - 1

# unmake_move callback: pop and restore using captured_id
@UNMAKE_CB
def unmake_cb(ctx_ptr, fy, fx, ty, tx, captured_id):
    pyctx = pyctx_from_voidp(ctx_ptr)
    if pyctx is None:
        return
    state = pyctx
    try:
        state.unmake_move_pop(fy, fx, ty, tx, int(captured_id))
    except Exception:
        # fallback: try ai.unmake_move with None
        m = state._reuse_move
        m.fy = int(fy); m.fx = int(fx); m.ty = int(ty); m.tx = int(tx)
        try:
            ai.unmake_move(state.board, m, None)
        except Exception:
            pass

# simple timeup callback (0 never timeout); you can replace with custom logic
@TIMEUP_CB
def timeup_cb(ctx_ptr):
    pyctx = pyctx_from_voidp(ctx_ptr)
    if pyctx is None:
        return 0
    state = pyctx
    if state.start_time is None or state.time_limit is None:
        return 0
    return 1 if (time.time() - state.start_time) > state.time_limit else 0

# snapshot callback: fill 90 ints row-major according to PIECE_CODE mapping
@SNAPSHOT_CB
def snapshot_cb(ctx_ptr, buf_ptr, buf_len):
    pyctx = pyctx_from_voidp(ctx_ptr)
    if pyctx is None:
        return 1
    state = pyctx
    board = state.board
    if int(buf_len) < 90:
        return 2
    # write into buf_ptr which supports assignment like array
    for y in range(10):
        for x in range(9):
            cell = board[y][x]
            code = 0
            if cell is not None:
                t = cell.get('type')
                s = cell.get('side')
                code = PIECE_CODE.get((t, s), 0)
            # buf_ptr is POINTER(c_int) so index assignment works
            buf_ptr[y*9 + x] = int(code)
    return 0

# ---------- Bind lib functions arg/return types ----------
lib.search_root.argtypes = [POINTER(SearchCtx), c_int, c_int, POINTER(CMove)]
lib.search_root.restype = c_int

lib.compute_eval_c.argtypes = [POINTER(SearchCtx), c_int]
lib.compute_eval_c.restype = c_int

# ---------- High-level helper to run search ----------
def run_search(board_state, side_str='black', depth=4, time_limit=None, max_nodes=0):
    """
    board_state: your ai.py board object (10x9 list of lists)
    side_str: 'black' or 'red' (which side to move)
    depth: search depth
    time_limit: seconds or None
    max_nodes: 0 means unlimited
    Returns: (fy,fx,ty,tx) of best move or None
    """
    # create wrapper state and ctx ptr
    state = WrapperState(board_state)
    state.start_time = None
    state.time_limit = None
    if time_limit:
        state.start_time = time.time()
        state.time_limit = float(time_limit)

    ctx_ptr = make_ctx_ptr(state)
    S = SearchCtx()
    S.ctx = ctx_ptr
    S.movegen = movegen_cb
    S.make_move = make_cb
    S.unmake_move = unmake_cb
    S.timeup = timeup_cb
    S.snapshot = snapshot_cb
    S.max_nodes = int(max_nodes)
    S.nodes = 0

    out = CMove()
    side_int = 0 if side_str == 'black' else 1
    ok = lib.search_root(ctypes.byref(S), int(depth), side_int, ctypes.byref(out))
    if ok:
        return (out.fy, out.fx, out.ty, out.tx)
    else:
        return None

# ---------- Example usage ----------
if __name__ == "__main__":
    # quick smoke test: use ai.INITIAL_SETUP or your current board
    try:
        board = ai.INITIAL_SETUP  # your ai.py's board variable
    except Exception:
        # fallback: create empty board
        board = [[None]*9 for _ in range(10)]

    print("Running shallow test search (depth=2)...")
    best = run_search(board_state=board, side_str='black', depth=2, time_limit=2.0, max_nodes=100000)
    print("Best move returned:", best)

    # test evaluation via C compute_eval_c
    state = WrapperState(board)
    ctx_ptr = make_ctx_ptr(state)
    S = SearchCtx()
    S.ctx = ctx_ptr
    S.movegen = movegen_cb
    S.make_move = make_cb
    S.unmake_move = unmake_cb
    S.timeup = timeup_cb
    S.snapshot = snapshot_cb
    S.max_nodes = 0
    S.nodes = 0
    val = lib.compute_eval_c(ctypes.byref(S), 0)  # evaluate for black-to-move perspective
    print("C eval (black-perspective):", val)
