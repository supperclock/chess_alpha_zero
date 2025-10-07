import sqlite3
import json
from util import log,compute_zobrist,init_zobrist
import json

def convert_move_string(move_json):
    """
    将一个代表移动的JSON对象转换为一个格式化的字符串。

    例如:
        输入: {"from": {"x": 1, "y": 2}, "to": {"x": 4, "y": 2}}
        输出: "Move(from_x=1, from_y=2, to_x=4, to_y=2)"
    """
    # 如果输入是JSON字符串，先将其解析为Python字典
    if isinstance(move_json, str):
        move_data = json.loads(move_json)
    else:
        move_data = move_json

    from_coords = move_data.get("from", {})
    to_coords = move_data.get("to", {})

    from_x = from_coords.get("x")
    from_y = from_coords.get("y")
    to_x = to_coords.get("x")
    to_y = to_coords.get("y")

    return f"Move(from_x={from_x}, from_y={from_y}, to_x={to_x}, to_y={to_y})"

def save_position_to_db(board_state, side_to_move, move, db_path='chess_games.db'):
    """
    将当前棋盘状态和走子信息保存到positions表中
    
    Args:
        board_state: 当前棋盘状态 (二维数组)
        side_to_move: 当前行棋方
        move: 走子信息
        db_path: 数据库路径
    """
    try:
        init_zobrist()
        # 计算zobrist哈希值作为position的唯一标识
        zobrist = compute_zobrist(board_state, side_to_move)
        #将move由{"from": {"x": 1, "y": 2}, "to": {"x": 4, "y": 2}}转换为Move(from_x=1, from_y=2, to_x=4, to_y=2)
        move = convert_move_string(move)
        
        # 连接数据库
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        sql = '''INSERT INTO ttxq_positions 
        (zobrist, best_move, visits, update_time)
        VALUES (?, ?, ?, datetime('now', 'localtime'))
        ON CONFLICT(zobrist, best_move) DO UPDATE SET
            visits = visits + 1           
        '''
        cursor.execute(sql,  (str(zobrist),move,1))        
        conn.commit()
        conn.close()
        
        log(f"Position saved to database. Zobrist: {zobrist}, Move: {move}")
        
    except Exception as e:
        log(f"Error saving position to database: {e}")

def record_game_positions(moves_history, db_path='chess_games.db'):
    """
    记录整局棋的positions信息
    
    Args:
        moves_history: 走子历史列表
        db_path: 数据库路径
    """
    for move_record in moves_history:
        board_state = move_record['board']
        side_to_move = move_record['side']
        move = move_record['move']
        save_position_to_db(board_state, side_to_move, move, db_path)