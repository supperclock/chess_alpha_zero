import sqlite3
import os
import shutil
from bs4 import BeautifulSoup
import re

def parse_chess_html(file_path, db_cursor=None):
    """
    解析象棋HTML文件并将信息写入数据库
    
    Args:
        file_path: HTML文件路径
        db_cursor: 数据库游标（可选）
    """
    # 连接数据库检查是否已存在该源文件的记录
    conn = None
    if db_cursor is None:
        conn = sqlite3.connect('chess_games.db')
        cursor = conn.cursor()
    else:
        cursor = db_cursor
        
    # 获取文件名用于source字段
    source_filename = os.path.basename(file_path)
    cursor.execute('SELECT game_id FROM games WHERE source = ?', (source_filename,))
    if cursor.fetchone():
        print(f"文件 {source_filename} 已经处理过，跳过...")
        if conn:
            conn.close()        
        return False  # 返回False表示未处理
    
    # 读取HTML文件
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # 使用BeautifulSoup解析HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 提取基本信息
    red_player = soup.find('div', class_='field--label', string='红方棋手')
    if red_player:
        red_player = red_player.find_next('div', class_='field--item').get_text(strip=True)
    
    black_player = soup.find('div', class_='field--label', string='黑方棋手')
    if black_player:
        black_player = black_player.find_next('div', class_='field--item').get_text(strip=True)
    
    event = soup.find('div', class_='field--label', string='比赛名称')
    if event:
        event = event.find_next('div', class_='field--item').get_text(strip=True)
    
    site = soup.find('div', class_='field--label', string='比赛地点')
    if site:
        site = site.find_next('div', class_='field--item').get_text(strip=True)
    
    date = soup.find('div', class_='field--label', string='日期')
    if date:
        date = date.find_next('div', class_='field--item').get_text(strip=True)
    
    round_info = soup.find('div', class_='field--label', string='轮次')
    if round_info:
        round_info = round_info.find_next('div', class_='field--item').get_text(strip=True)
    
    result_text = soup.find('div', class_='field--label', string='棋局结果')
    if result_text:
        result_text = result_text.find_next('div', class_='field--item').get_text(strip=True)
    
    moves_count = soup.find('div', class_='field--label', string='步数')
    if moves_count:
        moves_count = int(moves_count.find_next('div', class_='field--item').get_text(strip=True))
    
    # 转换结果格式
    result = result_text     
       
    # 提取走法
    moves = []
    moves_container = soup.find('ul', id='moves_text')
    if moves_container:
        round_elements = moves_container.find_all('li', class_='round')
        for round_li in round_elements:
            moves_spans = round_li.find_all('span', class_='move')
            for move_span in moves_spans:
                move_name = move_span.get('name')
                if move_name and move_name != "0":  # 跳过初始标记
                    move_text = move_span.get_text(strip=True)
                    move_index = int(move_name)
                    # 确定是红方还是黑方走棋
                    side = 'red' if move_index % 2 == 1 else 'black'
                    moves.append({
                        'index': move_index,
                        'side': side,
                        'move': move_text
                    })
    
    # 提取ICCS信息
    iccs_moves = []
    iccs_container = soup.find('div', id='qipu-moves-iccs')
    if iccs_container:
        iccs_text = iccs_container.get_text(strip=True)
        # 按4位一组截取ICCS信息
        for i in range(0, len(iccs_text), 4):
            if i + 4 <= len(iccs_text):
                iccs_moves.append(iccs_text[i:i+4])
    #提取init_fen信息
    init_fen = ''
    init_fen_container = soup.find('div', id='qipu-init-fen')
    if init_fen_container:
        init_fen = init_fen_container.get_text(strip=True)
    
    # 插入游戏信息
    cursor.execute('''
        INSERT INTO games (event, site, date, round, red_player, black_player, result, moves_count, source,init_fen)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (event, site, date, round_info, red_player, black_player, result, moves_count, source_filename,init_fen))
    
    game_id = cursor.lastrowid
    
    # 插入走法信息
    for i, move in enumerate(moves):
        # 如果有对应的ICCS信息，则使用它
        iccs = ''
        if i < len(iccs_moves):
            iccs = iccs_moves[i]
            
        cursor.execute('''
            INSERT INTO moves (game_id, move_index, side, move, iccs)
            VALUES (?, ?, ?, ?, ?)
        ''', (game_id, move['index'], move['side'], move['move'], iccs))
    
    # 如果是我们自己创建的连接，则提交事务并关闭连接
    if conn:
        conn.commit()
        conn.close()
       
    print(f"成功解析并插入棋局: {red_player} vs {black_player}")
    print(f"比赛: {event}")
    print(f"日期: {date}")
    print(f"结果: {result}")
    print(f"共 {len(moves)} 步棋")
    print(f"来源: {source_filename}")
    
    return True  # 返回True表示已处理


if __name__ == "__main__":
    # 示例用法
    # 解析单个文件
    # parse_chess_html('0000baa8-6d3b-4d0d-a88e-fb908b5a98d0')
    
    # 批量解析当前目录下的所有HTML文件
    dir = 'D:\\chinese-chess-qp\\www.xqipu.com\\qipu\\'
    # 连接数据库检查是否已存在该源文件的记录
    conn = sqlite3.connect('chess_games.db')
    cursor = conn.cursor()
    # 从数据库表qipu_file_ids中获取file_id，每次取100条
    cursor.execute('SELECT file_id FROM qipu_file_ids WHERE tag = 0 LIMIT 2000')
    file_ids = cursor.fetchall()
    for file_id in file_ids:
        file_path = os.path.join(dir, file_id[0])    
        parse_chess_html(file_path, cursor) 
        # 更新数据库表qipu_file_ids的tag字段为1
        cursor.execute('UPDATE qipu_file_ids SET tag = 1 WHERE file_id = ?', (file_id[0],))
        
    # 完成所有操作后提交并关闭连接
    conn.commit()
    conn.close()