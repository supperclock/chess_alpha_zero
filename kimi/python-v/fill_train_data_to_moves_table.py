#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sqlite3, pickle, torch, tqdm
from pathlib import Path
from opening_book import Move  
from util import * 

# 你的项目目录
PROJECT = Path(__file__).resolve().parent
import sys; sys.path.append(str(PROJECT.parent))   # 若脚本放子目录
from ai import make_move, unmake_move, copy_board, INITIAL_SETUP, find_general
from nn_data_representation import board_to_tensor, MOVE_TO_INDEX,get_move_maps

DB_FILE = PROJECT / 'chess_games.db'
BATCH   = 1000      # 每 N 步 commit 一次

def compute_z(game_id, conn):
    """
    根据 games.result 返回黑方视角 z：
    红先胜 -> 红赢 -> z = -1
    红先负 -> 黑赢 -> z = +1
    红先和 -> 和棋 -> z =  0
    """
    row = conn.execute("SELECT result FROM games WHERE game_id=?", (game_id,)).fetchone()
    if row is None:
        return 0.0
    res = row['result']
    if res == '红先胜':
        return -1.0
    if res == '红先负':
        return 1.0
    return 0.0          # 红先和

def build_pi(move):
    """返回 one-hot 向量，长度=合法动作空间"""
    pi = torch.zeros(len(MOVE_TO_INDEX))
    key = (move.fx, move.fy, move.tx, move.ty)
    pi[MOVE_TO_INDEX[key]] = 1.0
    return pi

def str_to_move(iccs_str):
    """
    ICCS 坐标解析
    例如: 'b2e2' -> (1,2) -> (4,2)
    """
    if len(iccs_str) != 4:
        raise ValueError("ICCS 走法必须是4个字符，比如 'b2e2'")
    
    cols = "abcdefghi"
    try:
        from_x = 8 - cols.index(iccs_str[0]) 
        from_y = int(iccs_str[1])
        to_x   = 8 - cols.index(iccs_str[2])  
        to_y   = int(iccs_str[3])
    except Exception:
        raise ValueError(f"非法 ICCS 字符串: {iccs_str}")
    
    return Move(from_y, from_x, to_y, to_x)
def process_game(game_id, moves, conn):
    """moves: List[Row] 同一盘棋按 move_index 排序"""
    try:
        board = copy_board(INITIAL_SETUP)
        side  = 'red'
        z     = compute_z(game_id, conn)
        for row in moves:
            move = str_to_move(row['iccs'])
            log(f"{game_id}:{row['iccs']}:{row['move_index']}: {side}: {move.to_dict()}")

            # 计算 tensor（走子前的局面）
            tensor = board_to_tensor(board, side).squeeze(0)
            pi     = build_pi(move)
            # 落子
            make_move(board, move)
            side = 'red' if side == 'black' else 'black'
            # 保存
            yield (pickle.dumps(tensor),
                pickle.dumps(pi),
                z,
                row['game_id'],
                row['move_index'])
    except Exception as e:
        log(f"{game_id} 棋盘解析错误: {e}")
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('delete from moves where game_id=?',(game_id,))
        conn.commit()   
        return

def main():
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        # 1. 所有待处理记录
        todo = list(cur.execute(
            "SELECT game_id, move_index, iccs FROM moves WHERE tensor IS NULL ORDER BY game_id, move_index"))
        if not todo:
            print('没有需要更新的记录。')
            return

        # 2. 按 game_id 分组
        from itertools import groupby
        groups = [(k, list(g)) for k, g in groupby(todo, key=lambda r: r['game_id'])]
        get_move_maps()

        buf = []
        for game_id, moves in tqdm.tqdm(groups, desc='Games'):
            for tensor_b, pi_b, z, gid, mid in process_game(game_id, moves, conn):
                if tensor_b is None: continue
                buf.append((tensor_b, pi_b, z, gid, mid))
            cur.executemany(
                "UPDATE moves SET tensor=?, pi=?, z=? WHERE game_id=? AND move_index=?", buf)
            conn.commit()
            buf.clear()
        print('全部更新完成！')

if __name__ == '__main__':
    main()