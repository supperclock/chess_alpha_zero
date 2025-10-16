#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sqlite3, pickle, torch, tqdm
from pathlib import Path
from opening_book import Move
from util import *
import multiprocessing as mp  # 导入多进程模块
import os  # 用于获取CPU核心数

# 你的项目目录
PROJECT = Path(__file__).resolve().parent
import sys; sys.path.append(str(PROJECT.parent))  # 若脚本放子目录
from ai import make_move, unmake_move, copy_board, INITIAL_SETUP, find_general
from nn_data_representation import board_to_tensor, MOVE_TO_INDEX, get_move_maps

DB_FILE = PROJECT / 'chess_games.db'
BATCH = 2000  # 可以适当增大批处理大小以提高数据库写入效率

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
    return 0.0  # 红先和

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
        to_x = 8 - cols.index(iccs_str[2])
        to_y = int(iccs_str[3])
    except Exception:
        raise ValueError(f"非法 ICCS 字符串: {iccs_str}")
    
    return Move(from_y, from_x, to_y, to_x)

# ==============================================================================
# 新增：这是将在每个子进程中运行的工作函数
# ==============================================================================
def worker_process_game(args):
    """
    处理单个棋局的计算任务。此函数不进行任何数据库操作。
    它接收一个棋局的所有信息，并返回计算结果。
    """
    game_id, moves, z = args  # 接收主进程打包好的参数
    results = []
    try:
        board = copy_board(INITIAL_SETUP)
        side = 'red'
        for move_data in moves:
            iccs_str = move_data['iccs']
            move_index = move_data['move_index']
            
            move = str_to_move(iccs_str)
            # log(f"{game_id}:{iccs_str}:{move_index}: {side}: {move.to_dict()}") # 在多进程中打印会混乱，建议注释掉

            # 计算 tensor（走子前的局面）
            tensor = board_to_tensor(board, side).squeeze(0)
            pi = build_pi(move)
            
            # 落子
            make_move(board, move)
            side = 'red' if side == 'black' else 'black'
            
            # 准备结果，注意序列化(pickle)也在这里完成，因为这是CPU密集型操作
            results.append((
                pickle.dumps(tensor),
                pickle.dumps(pi),
                z,
                game_id,
                move_index
            ))
    except Exception as e:
        log(f"!!! Worker Error:  {game_id} 解析错误: {e}")
        # 如果一个棋局处理失败，返回一个空列表，主进程将跳过它
        return []
    
    return results

def main():
    # 设置多进程启动方式，'fork'在某些环境下可能更高效，但'spawn'更安全稳定
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass # 已经设置过

    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        # 1. 所有待处理记录
        print("正在查询所有待处理的走法记录...")
        todo = list(cur.execute(
            "SELECT game_id, move_index, iccs FROM moves WHERE tensor IS NULL ORDER BY game_id, move_index"))
        if not todo:
            print('没有需要更新的记录。')
            return
        print(f"查询到 {len(todo)} 条记录需要处理。")

        # 2. 按 game_id 分组
        from itertools import groupby
        groups = [(k, list(g)) for k, g in groupby(todo, key=lambda r: r['game_id'])]
        # get_move_maps()

        # 3. 为每个棋局预先计算z值，并打包成工作任务
        print("正在为每个棋局准备任务...")
        worker_args = []
        for game_id, moves in tqdm.tqdm(groups, desc='准备任务'):
            # Convert each sqlite3.Row object into a standard dictionary. Dictionaries are picklable.
            moves_as_dicts = [dict(row) for row in moves]
            z = compute_z(game_id, conn)
            worker_args.append((game_id, moves_as_dicts, z))

        # 4. 创建并运行进程池
        # 使用 os.cpu_count() 来自动确定进程数，可以减1以保留一个核心给系统
        num_workers = max(1, os.cpu_count() - 1 if os.cpu_count() else 1)
        print(f"启动 {num_workers} 个工作进程进行并行计算...")
        
        buf = []
        with mp.Pool(processes=num_workers, initializer=get_move_maps) as pool:
            # 使用 imap_unordered 来获得最佳性能，它会按完成顺返回结果
            # tqdm 用于显示总体进度
            progress_bar = tqdm.tqdm(
                pool.imap_unordered(worker_process_game, worker_args),
                total=len(groups),
                desc='处理棋局'
            )
            
            for result_list in progress_bar:
                # 收集计算结果
                if result_list: # 仅当worker成功返回结果时才添加
                    buf.extend(result_list)
                
                # 当缓冲区达到批处理大小时，写入数据库
                if len(buf) >= BATCH:
                    cur.executemany(
                        "UPDATE moves SET tensor=?, pi=?, z=? WHERE game_id=? AND move_index=?", buf)
                    conn.commit()
                    print(f" -> 已提交 {len(buf)} 条更新。")
                    buf.clear()

        # 5. 处理并提交缓冲区中剩余的最后一批数据
        if buf:
            cur.executemany(
                "UPDATE moves SET tensor=?, pi=?, z=? WHERE game_id=? AND move_index=?", buf)
            conn.commit()
            print(f" -> 已提交最后 {len(buf)} 条更新。")
            buf.clear()

        print('全部更新完成！')

if __name__ == '__main__':
    main()