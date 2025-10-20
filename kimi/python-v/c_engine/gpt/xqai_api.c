#include "xqai_api.h"
#include <string.h>
#include <stdio.h>

/* ---------------------------------------------------------------
 *  xqai_api.c - 中国象棋引擎 Python 接口层
 * ---------------------------------------------------------------
 *  提供 ctypes 可直接加载的 C API：
 *    - init_board()
 *    - search_best()
 *    - evaluate_board_c()
 * --------------------------------------------------------------- */

/* 初始化棋盘 */
EXPORT void init_board(Board *b) {
    if (!b) return;
    memset(b, 0, sizeof(Board));
    b->side_to_move = SIDE_RED;
}

/* 搜索最佳走法 */
EXPORT void search_best(const Board *b_in, Move *best_move, int depth, int time_limit) {
    if (!b_in || !best_move) return;

    Board board = *b_in; /* 拷贝以避免修改原对象 */

    zobrist_init();
    search_init();

    Move m = search_root(&board, depth, time_limit);
    *best_move = m;
}

/* 调用评估函数 */
EXPORT int evaluate_board_c(const Board *b) {
    if (!b) return 0;
    return evaluate_board(b);
}