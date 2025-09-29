# opening_book.py —— 修复版：保持与 ai.py 一致（红方在上，黑方在下）
from collections import namedtuple
import logging
import random

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s')

Opening = namedtuple('Opening', ['name', 'fen', 'candidates'])

class Move:
    __slots__ = ('fy', 'fx', 'ty', 'tx', 'score', 'captured')
    def __init__(self, fy, fx, ty, tx):
        self.fy, self.fx, self.ty, self.tx = fy, fx, ty, tx
        self.score = 0
        self.captured = None
    def to_dict(self):
        return {'from': {'y': self.fy, 'x': self.fx}, 'to': {'y': self.ty, 'x': self.tx}}
    def __eq__(self, other):
        if not isinstance(other, Move):
            if isinstance(other, dict) and 'from' in other and 'to' in other:
                return self.fy == other['from']['y'] and self.fx == other['from']['x'] and \
                       self.ty == other['to']['y'] and self.tx == other['to']['x']
            return NotImplemented
        return self.fy == other.fy and self.fx == other.fx and self.ty == other.ty and self.tx == other.tx
    def __hash__(self):
        return hash((self.fy, self.fx, self.ty, self.tx))


# =====================================================================
# 工具函数
# =====================================================================

piece_dict = {
    'K':'帥','A':'仕','E':'相','H':'馬','R':'車','C':'炮','P':'兵',
    'k':'將','a':'士','e':'象','h':'马','r':'车','c':'砲','p':'卒'
}
piece_reverse = {v:k for k,v in piece_dict.items()}

def fen_to_board(fen: str):
    rows = fen.split()[0].split('/')
    board = [[None]*9 for _ in range(10)]
    for y, row in enumerate(rows):  # 红方在上
        x = 0
        for ch in row:
            if ch.isdigit():
                x += int(ch)
            else:
                side = 'red' if ch.isupper() else 'black'
                board[y][x] = {'type': piece_dict[ch], 'side': side}
                x += 1
    return board

def board_to_fen(board_state, side_to_move):
    fen_rows = []
    piece_reverse = {
        '帥':'K','仕':'A','相':'E','馬':'H','車':'R','炮':'C','兵':'P',
        '將':'k','士':'a','象':'e','马':'h','车':'r','砲':'c','卒':'p'
    }
    for y in range(10):  # 红方在上
        cnt = 0
        row = ''
        for x in range(9):
            p = board_state[y][x]
            if p is None:
                cnt += 1
            else:
                if cnt:
                    row += str(cnt)
                    cnt = 0
                ch = piece_reverse[p['type']]
                row += ch.upper() if p['side'] == 'red' else ch.lower()
        if cnt:   # ✅ 行尾补上剩余空格
            row += str(cnt)
        fen_rows.append(row)
    turn = 'w' if side_to_move == 'red' else 'b'
    return '/'.join(fen_rows) + f' {turn} - - 0 1'


def match_fen(fen1, fen2):
    return ' '.join(fen1.split()[:2]) == ' '.join(fen2.split()[:2])

def flip_fen(fen: str):
    """把黑方在上 → 红方在上的 FEN 翻转"""
    parts = fen.split()
    rows = parts[0].split('/')
    flipped = []
    for row in reversed(rows):
        new_row = ""
        for ch in row:
            if ch.isalpha():
                new_row += ch.swapcase()
            else:
                new_row += ch
        flipped.append(new_row)
    parts[0] = '/'.join(flipped)
    return ' '.join(parts)


# =====================================================================
# 开局库
# =====================================================================

# 初始棋局 FEN（红方在上）
INITIAL_FEN_W = "RHEAKAEHR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rheakaehr w - - 0 1"


RED_OPENING_START = [
    Opening("红方第一步选择", INITIAL_FEN_W, [
        Move(2, 4, 4, 4), # 炮二平五 - 中炮开局系列
        Move(3, 6, 4, 6), # 兵七进一 - 仙人指路系列
        Move(3, 2, 4, 2), # 兵三进一 - 对兵局系列
        Move(2, 1, 4, 1), # 炮二平四 - 士角炮系列
        Move(2, 8, 4, 8), # 炮八平五 - 过宫炮系列
        Move(0, 1, 2, 2), # 马二进三 - 边马局系列
        Move(0, 7, 2, 6), # 马八进七
        Move(0, 2, 2, 4), # 相三进五
        Move(0, 6, 2, 4), # 相七进五
    ])
]

# ---------- 红方开局库 ----------
RED_BOOK = [
    Opening("马二进三对左右炮",
            "R1EAKAEHR/9/1CH4C1/P1P1P1P1P/9/9/p1p1p1p1p/1c2c4/9/rheakaehr w - - 0 1",
            [Move(2,4,4,4),  Move(3,2,4,2)]),  # 中炮平五、马三进四、兵三进一
    Opening("中炮对屏风马",
            "RHEAKAEHR/9/1C2C4/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rheakaehr w - - 0 1",
            [Move(2,4,2,7), Move(3,6,4,6), Move(0,7,2,6)]),
    Opening("中炮对反宫马",
            "RHEAKAEHR/9/1C2C4/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rheakaehr w - - 0 1",
            [Move(2,4,2,7), Move(0,1,2,2), Move(3,2,4,2)]),
    Opening("仙人指路",
            "RHEAKAEHR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c2c4/9/rheakaehr w - - 0 1",
            [Move(3,6,4,6), Move(0,7,2,6), Move(2,4,2,7)]),
    Opening("起马局",
            flip_fen("rheakaeh1/9/1c5c1/p1p1p1p1p/9/2P6/P3P1P1P/1c5c1/3h5/rheaka1hr w - - 0 3"),
            [Move(0,1,2,2), Move(0,7,2,6), Move(3,6,4,6)]),
    Opening("飞相局",
            "RHEAKAEHR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/2c3c1/9/rheakaehr w - - 0 1",
            [Move(0,4,4,4), Move(3,6,4,6), Move(0,7,2,6)]),
    Opening("过宫炮",
            "RHEAKAEHR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/2c3c1/9/rheakaehr w - - 0 1",
            [Move(2,8,2,6), Move(0,1,2,2), Move(3,2,4,2)]),
    Opening("士角炮",
            "RHEAKAEHR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/3c2c1/9/rheakaehr w - - 0 1",
            [Move(2,1,4,1), Move(3,6,4,6), Move(0,7,2,6)]),
    Opening("进兵对卒底炮",
            "RHEAKAEHR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/3c2c1/9/rheakaehr w - - 0 1",
            [Move(3,2,4,2), Move(0,7,2,6), Move(2,4,2,7)]),
    Opening("中炮对左炮封车",
            "RHEAKAEHR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/c3c3/9/rheakaehr w - - 0 1",
            [Move(2,4,2,7), Move(0,0,0,2), Move(0,1,2,2)]),
    Opening("横车对直车",
            "RHEAKAEHR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/c3c3/9/rheakaehr w - - 0 1",
            [Move(2,4,2,7), Move(0,0,0,4), Move(0,4,0,2)]),
    Opening("五七炮",
            "RHEAKAEHR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/c4c2/9/rheakaehr w - - 0 1",
            [Move(2,4,2,7), Move(2,8,2,5), Move(0,7,2,6)]),
    Opening("五六炮",
            "RHEAKAEHR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/c4c2/9/rheakaehr w - - 0 1",
            [Move(2,4,2,7), Move(2,1,2,4), Move(0,7,2,6)]),
    Opening("急进中兵",
            "RHEAKAEHR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/c5c1/9/rheakaehr w - - 0 1",
            [Move(2,4,2,7), Move(3,2,4,2), Move(4,2,5,2)]),
    Opening("边马局",
            "RHEAKAEHR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/c5c1/9/rheakaehr w - - 0 1",
            [Move(0,1,2,2), Move(0,7,2,8), Move(3,6,4,6)]),
    Opening("金钩炮",
            "RHEAKAEHR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/c5c1/9/rheakaehr w - - 0 1",
            [Move(2,8,0,6), Move(0,7,2,6), Move(3,6,4,6)]),
]

# ---------- 黑方开局库 ----------
BLACK_BOOK = [
    # 红方炮二平五的应招
    Opening("屏风马应中炮变例",
            "RHEAKAEHR/9/4C2C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rheakaehr b - - 0 1",
            [Move(7,4,7,7), Move(9,7,7,6), Move(9,1,7,2)]),
    Opening("屏风马应中炮",
            "RHEAKAEHR/9/1C2C4/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rheakaehr b - - 0 1",
            [Move(7,4,7,7), Move(9,7,7,6), Move(9,1,7,2)]),
    Opening("反宫马应中炮",
            "RHEAKAEHR/9/1C2C4/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rheakaehr b - - 0 1",
            [Move(9,1,7,2), Move(7,1,7,4), Move(9,7,7,6)]),
    Opening("卒底炮应仙人指路",
            "RHEAKAEHR/9/1C5C1/P1P1P1P1P/9/4P4/p1p1p1p1p/1c5c1/9/rheakaehr b - - 0 1",
            [Move(6,2,5,2), Move(9,7,7,6), Move(9,1,7,2)]),
    Opening("对兵局",
            "RHEAKAEHR/9/1C5C1/P1P1P1P1P/9/2P1P4/p1p1p1p1p/1c5c1/9/rheakaehr b - - 0 1",
            [Move(6,2,5,2), Move(9,1,7,2), Move(7,4,7,7)]),
    Opening("顺炮",
            "RHEAKAEHR/9/1C2C4/P1P1P1P1P/9/2P6/p1p1p1p1p/1c5c1/9/rheakaehr b - - 0 1",
            [Move(7,4,7,7), Move(9,0,7,2), Move(9,8,7,8)]),
    Opening("列炮",
            "RHEAKAEHR/9/1C2C4/P1P1P1P1P/9/2P6/p1p1p1p1p/1c5c1/9/rheakaehr b - - 0 1",
            [Move(7,4,7,7), Move(9,8,7,8), Move(9,1,7,2)]),
    Opening("左炮封车",
            "RHEAKAEHR/9/1C5C1/P1P1P1P1P/9/2P6/p1p1p1p1p/1c5c1/9/rheakaehr b - - 0 1",
            [Move(7,7,7,4), Move(9,1,7,2), Move(9,7,7,6)]),
    Opening("右炮封车",
            "RHEAKAEHR/9/1C5C1/P1P1P1P1P/9/2P6/p1p1p1p1p/1c5c1/9/rheakaehr b - - 0 1",
            [Move(7,1,7,4), Move(9,1,7,2), Move(9,7,7,6)]),
    Opening("三步虎",
            "RHEAKAEHR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rheakaehr b - - 0 1",
            [Move(9,1,7,2), Move(7,1,5,1), Move(9,0,7,0)]),
    Opening("后补列炮",
            "RHEAKAEHR/9/1C2C4/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rheakaehr b - - 0 1",
            [Move(9,1,7,2), Move(6,8,6,5), Move(7,7,7,4)]),
    Opening("半途列炮",
            "RHEAKAEHR/9/1C5C1/P1P1P1P1P/9/2P1P4/p1p1p1p1p/1c5c1/9/rheakaehr b - - 0 1",
            [Move(9,1,7,2), Move(3,2,4,2), Move(7,7,7,4)]),
    Opening("进马对挺兵",
            "RHEAKAEHR/9/1C5C1/P1P1P1P1P/9/2P6/p1p1p1p1p/1c5c1/9/rheakaehr b - - 0 1",
            [Move(9,1,7,2), Move(6,2,5,2), Move(9,7,7,6)]),
    Opening("飞象应中炮",
            "RHEAKAEHR/9/1C5C1/P1P1P1P1P/9/2P6/p1p1p1p1p/1c5c1/9/rheakaehr b - - 0 1",
            [Move(9,4,5,4), Move(9,1,7,2), Move(9,7,7,6)]),
    Opening("进炮应飞相",
            "RHEAKAEHR/9/1C5C1/P1P1P1P1P/9/4E4/p1p1p1p1p/1c5c1/9/rheakaehr b - - 0 1",
            [Move(7,1,5,1), Move(9,1,7,2), Move(9,7,7,6)]),
    Opening("卒底炮对兵局",
            "RHEAKAEHR/9/1C5C1/P1P1P1P1P/9/4P4/p1p1p1p1p/1c5c1/9/rheakaehr b - - 0 1",
            [Move(6,2,5,2), Move(9,8,7,8), Move(9,1,7,2)]),
    
    # 以下是对应 RED_OPENING_START 中其他走法的应招
    Opening("中炮对士角炮",
            "RHEAKAEHR/9/1C5C1/P1P1P1P1P/9/4C4/p1p1p1p1p/1c5c1/9/rheakaehr b - - 0 1",
            [Move(7,4,4,4), Move(9,1,7,2), Move(9,7,7,6)]),
    Opening("中炮对八路炮",
            "RHEAKAEHR/9/1C5C1/P1P1P1P1P/9/8C/p1p1p1p1p/1c5c1/9/rheakaehr b - - 0 1",
            [Move(7,4,4,4), Move(9,1,7,2), Move(9,7,7,6)]),
    Opening("屏风马应马八进七",
            "RHEAK1EHR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rheakaehr b - - 0 1",
            [Move(7,4,7,7), Move(9,7,7,6), Move(9,1,7,2)]),
    Opening("中炮应相三进五",
            "RHEAKAEHR/9/1C5C1/P1P1P1P1P/9/4E4/p1p1p1p1p/1c5c1/9/rheakaehr b - - 0 1",
            [Move(7,4,4,4), Move(9,1,7,2), Move(9,7,7,6)]),
    Opening("中炮应相七进五",
            "RHEAKAEHR/9/1C5C1/P1P1P1P1P/9/4E4/p1p1p1p1p/1c5c1/9/rheakaehr b - - 0 1",
            [Move(7,4,4,4), Move(9,1,7,2), Move(9,7,7,6)]),
    
    # 以下是对应马二进三+炮二平五的应招
    Opening("马炮联合对进七马",
            "R1EAKAEHR/9/2H1C2C1/P1P1P1P1P/9/9/p1p1p1p1p/1c4hc1/9/rheakae1r b - - 0 1",
            [Move(9,1,7,2), Move(7,2,5,1), Move(9,0,7,0)]),  # 马二进三、马二进四、车一平三
            
    Opening("互移车马战术",
            "1REAKAEHR/9/2H1C2C1/P1P1P1P1P/9/9/p1p1p1p1p/rc4hc1/9/1heakae1r b - - 0 1",
            [Move(9,1,7,2),  Move(9,8,8,8)]),  # 将五进一、车六平三
    
    # 以下是对应马二进三的应招
    Opening("中炮封马",
            "R1EAKAEHR/9/1CH4C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rheakaehr b - - 0 1",
            [Move(7,4,4,4), Move(9,1,7,2), Move(7,7,7,4)]),  # 中炮平五
    Opening("对马局",
            "R1EAKAEHR/9/1CH4C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rheakaehr b - - 0 1",
            [Move(9,1,7,2), Move(7,1,5,1), Move(9,7,7,6)]),  # 马二进三
    Opening("顺炮对马",
            "R1EAKAEHR/9/1CH4C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rheakaehr b - - 0 1",
            [Move(7,7,7,4), Move(9,1,7,2), Move(9,7,7,6)])   # 炮八平五
]


# =====================================================================
# 统计 & 查询
# =====================================================================

BLACK_BOOK_ACTIVE = True

OPENING_STATS = {
    'red_hits': 0, 'black_hits': 0, 'red_misses': 0, 'black_misses': 0, 
    'total_queries': 0, 'opening_usage': {}
}

def get_opening_stats(): return OPENING_STATS.copy()
def reset_opening_stats():
    global OPENING_STATS
    OPENING_STATS = {'red_hits': 0, 'black_hits': 0, 'red_misses': 0, 'black_misses': 0, 'total_queries': 0, 'opening_usage': {}}
    logging.info("[开局库] 统计信息已重置")

def print_opening_stats():
    stats = OPENING_STATS
    total_hits = stats['red_hits'] + stats['black_hits']
    total_misses = stats['red_misses'] + stats['black_misses']
    total_queries = stats['total_queries']
    logging.info("=" * 50)
    logging.info("开局库使用统计报告")
    logging.info(f"总查询次数: {total_queries}. 命中次数: {total_hits}. 未命中次数: {total_misses}")
    if total_queries > 0:
        hit_rate = total_hits / total_queries * 100
        logging.info(f"总命中率: {hit_rate:.1f}%")
    logging.info("=" * 50)

def get_opening_move(current_board, side_to_move):
    OPENING_STATS['total_queries'] += 1
    current_fen = board_to_fen(current_board, side_to_move)
    logging.info(f"[开局库] 当前局面: {current_fen}")

    if side_to_move == 'red':
        if match_fen(current_fen, INITIAL_FEN_W):
            books_to_check = RED_OPENING_START
        else:
            books_to_check = RED_BOOK
    elif BLACK_BOOK_ACTIVE:
        books_to_check = BLACK_BOOK
    else:
        return None

    matched_openings = [(i, op) for i, op in enumerate(books_to_check) if match_fen(current_fen, op.fen)]
            
    if matched_openings:
        opening_idx, opening = random.choice(matched_openings)
        selected_move = random.choice(opening.candidates)
        
        key = 'red_hits' if side_to_move == 'red' else 'black_hits'
        OPENING_STATS[key] += 1
        if opening.name not in OPENING_STATS['opening_usage']:
            OPENING_STATS['opening_usage'][opening.name] = 0
        OPENING_STATS['opening_usage'][opening.name] += 1
        
        logging.info(f"[开局库] 命中开局：{opening.name}")
        return selected_move.to_dict()
    else:
        key = 'red_misses' if side_to_move == 'red' else 'black_misses'
        OPENING_STATS[key] += 1
        logging.info(f"[开局库] 未找到匹配的开局")
        return None
def print_board(board):
    """打印棋盘（红方在上，黑方在下）"""
    for y in range(10):
        row = ""
        for x in range(9):
            piece = board[y][x]
            if piece is None:
                row += "． "   # 用全角点表示空格
            else:
                row += piece['type'] + " "
        print(f"{9-y:2d} | {row}")
    print("    -------------------")
    print("     0 1 2 3 4 5 6 7 8 (列号)")

if __name__ == "__main__":
    print("检查 FEN:", INITIAL_FEN_W)
    board = fen_to_board(INITIAL_FEN_W)
    print_board(board)
    #检查RED_BOOK和BLACK_BOOK里的FEN是否正确
    for opening in RED_BOOK:
        print(f"[开局库] 开局: {opening.name}")
        board = fen_to_board(opening.fen)
        print_board(board)
        print("-" * 50)
    #检查BLACK_BOOK
    for opening in BLACK_BOOK:
        print(f"[开局库] 开局: {opening.name}")
        board = fen_to_board(opening.fen)
        print_board(board)
        print("-" * 50)