import sqlite3
from collections import defaultdict
from chess_parser import XiangqiGame,INITIAL_SETUP
from util import compute_zobrist, init_zobrist

# ---------- 初始化 ----------
init_zobrist()

DB_PATH = 'chess_games.db'

# ---------- 每盘棋内部缓存结构 ----------
# key = (zobrist_hash, best_move)
# value = {'visits', 'red_wins', 'black_wins', 'draws'}
PositionCache = lambda: defaultdict(lambda: {
    'visits': 0,
    'red_wins': 0,
    'black_wins': 0,
    'draws': 0
})

def generate_positions(db_path='chess_games.db'):
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all games
    sql = """
        SELECT game_id, result,init_fen FROM games where tag =0 order by game_id
    """
    cursor.execute(sql)
    games = cursor.fetchall()    
    
    # Process each game
    #显示进度
    for game_id, result,init_fen in games:        
        print("Processing game: %s" % game_id)
        game = XiangqiGame(init_fen)
        # Determine result values
        red_result = 0  # Loss
        black_result = 0  # Loss
        draw_result = 0
        if result == '红先胜':
            red_result = 1  # Win
        elif result == '红先负':
            black_result = 1  # Win
        elif result == '红先和':
            draw_result = 1
        
        # Get moves for this game
        cursor.execute('''
            SELECT iccs 
            FROM moves 
            WHERE game_id = ? 
            ORDER BY move_index
        ''', (game_id,))
        
        moves = cursor.fetchall()   
        # 逐着推演，缓存局面
        cache = PositionCache()
       
        # Process each move
        for move_str in moves:      
            zobrist_hash = compute_zobrist(game.board, game.current_player)        
            try:             
                rlt, move = game.move(move_str[0],game_id)                      
                if not rlt:
                    print("由于上一步走棋失败，棋局终止。")
                    cursor.execute('update games set tag=2 where game_id=?',(game_id,))
                    conn.commit()
                    break
                key = (str(zobrist_hash), str(move))
                cache[key]['visits'] += 1
                cache[key]['red_wins'] += red_result
                cache[key]['black_wins'] += black_result
                cache[key]['draws'] += draw_result    
            except Exception:            
                cursor.execute('update games set tag=2 where game_id=?',(game_id,))
                conn.commit()
                break
        # 一盘棋结束，批量写库
        flush_positions(game_id, cache)                                    

# ---------- 单盘棋落库 ----------
def flush_positions(game_id: int, cache: dict):
    """
    把一盘棋产生的所有局面一次性写进 positions 表
    """
    if not cache:          # 空缓存直接返回
        return

    data = [
        (
            zobrist,
            best_move,
            stats['visits'],
            stats['red_wins'],
            stats['black_wins'],
            stats['draws'],
            str(game_id)     # 用于拼接 game_ids
        )
        for (zobrist, best_move), stats in cache.items()
    ]

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 批量 Upsert
    sql = """
        INSERT INTO positions (zobrist, best_move, visits,
                               red_wins, black_wins, draws, game_ids)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(zobrist, best_move) DO UPDATE SET
            visits     = visits + excluded.visits,
            red_wins   = red_wins + excluded.red_wins,
            black_wins = black_wins + excluded.black_wins,
            draws      = draws + excluded.draws,
            game_ids   = game_ids || '-' || excluded.game_ids
    """
    cursor.executemany(sql, data)
    conn.commit()
    cursor.execute('update games set tag=1 where game_id=?',(game_id,))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    generate_positions()