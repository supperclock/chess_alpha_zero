import sqlite3
import os

def init_chess_database(db_path='chess_games.db'):
    """
    Initialize a SQLite database for storing Chinese chess game records.
    
    The database will contain:
    - games table: stores game metadata
    - moves table: stores individual moves for each game
    - positions table: stores position statistics
    """
    
    # Check if database already exists
    # if os.path.exists(db_path):
    #     print(f"Database {db_path} already exists.")
    #     response = input("Do you want to overwrite it? (y/N): ")
    #     if response.lower() != 'y':
    #         print("Database initialization cancelled.")
    #         return False
    
    # Create/connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create games table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS games (
            game_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            event       TEXT,           -- 比赛/来源
            site        TEXT,           -- 地点
            date        TEXT,           -- 日期
            round       TEXT,           -- 轮次
            red_player  TEXT,
            black_player TEXT,
            result      TEXT,           -- 结果: "1-0", "0-1", "1/2-1/2"
            moves_count INTEGER,
            source      TEXT,            -- 来源文件名
            init_fen     TEXT
        )
    ''')
    
    # Create moves table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS moves (
            game_id     INTEGER,
            move_index  INTEGER,        -- 第几步
            side        TEXT,           -- red/black
            move        TEXT,           -- 走法 (UCI 或自定义格式)
            iccs        TEXT,           -- iccs
            PRIMARY KEY (game_id, move_index)
        )
    ''')
    
    # Create positions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS positions (
            zobrist TEXT,
            game_ids TEXT,
            best_move TEXT,
            visits INTEGER,
            red_wins INTEGER,
            black_wins INTEGER,
            draws INTEGER,
            -- 定义zobrist和best_move为联合主键
            PRIMARY KEY (zobrist, best_move)
        )
    ''')  

     # Create positions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ttxq_positions (
            zobrist TEXT,          
            best_move TEXT,    
            visits INTEGER,       
            update_time TEXT,     -- 最后更新时间
            -- 定义zobrist和best_move为联合主键
            PRIMARY KEY (zobrist, best_move)
        )
    ''')  
    
    # Create indexes for better query performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_games_date ON games (date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_games_result ON games (result)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_games_red_player ON games (red_player)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_games_black_player ON games (black_player)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_moves_game_id ON moves (game_id)')    
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_positions_zobrist ON positions (zobrist)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_ttxq_positions_zobrist ON ttxq_positions (zobrist)')
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print(f"Database {db_path} initialized successfully.")
    return True

if __name__ == "__main__":
    init_chess_database()