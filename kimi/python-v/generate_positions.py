import sqlite3
from collections import defaultdict
from chess_parser import XiangqiGame,INITIAL_SETUP
from util import compute_zobrist, init_zobrist

def generate_positions(db_path='chess_games.db'):
    """Generate positions table data from games and moves tables"""
    # Initialize Zobrist hashing
    init_zobrist()
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all games
    sql = """
        SELECT game_id, result,init_fen FROM games where result IN ('红先胜', '红先负', '红先和') and game_id > 2000 order by game_id
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
       
        # Process each move
        for move_str in moves:      
            zobrist_hash = compute_zobrist(game.board, game.current_player)                     
            rlt, move = game.move(move_str[0],game_id)                      
            if not rlt:
                print("由于上一步走棋失败，棋局终止。")
                break
            insert_position(str(zobrist_hash), game_id, str(move), 1, red_result, black_result, draw_result)                                           

def insert_position(zobrist_hash, game_id, best_move, visits, red_wins, black_wins, draws):
    """Insert a position into the database"""
    conn = sqlite3.connect('chess_games.db')
    cursor = conn.cursor()
    sql = '''INSERT INTO positions 
        (zobrist, game_ids, best_move, visits, red_wins, black_wins, draws)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(zobrist, best_move) DO UPDATE SET
            visits = visits + 1,
            red_wins = red_wins + excluded.red_wins,
            black_wins = black_wins + excluded.black_wins,
            draws = draws + excluded.draws,
            game_ids = game_ids || '-' || excluded.game_ids
    '''
    #如果zobrist和best_move已经存在，则更新visits+1,red_wins+red_wins, black_wins+black_wins, draws+draws
    cursor.execute(sql,  (
        zobrist_hash,
        game_id,
        best_move,
        visits,
        red_wins,
        black_wins,
        draws        
    )
    )
    # Commit changes and close connection
    conn.commit()
    conn.close()

if __name__ == "__main__":
    generate_positions()