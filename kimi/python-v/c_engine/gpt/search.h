// search.h
#ifndef SEARCH_H
#define SEARCH_H

#include "board.h"
#include <stdint.h>

#define MAX_PLY 64
#define SQ_COUNT (ROWS*COLS)
#define TT_SIZE (1<<20) /* 1M entries, 调整为机器内存允许 */

typedef enum { TT_EMPTY = 0, TT_EXACT = 1, TT_LOWER = 2, TT_UPPER = 3 } TTFlag;

typedef struct {
    uint64_t key;
    int depth;
    int value;
    int flag; /* TTFlag */
    int move_from;
    int move_to;
} TTEntry;

/* 初始化 */
void search_init(void);

/* 根节点迭代加深搜索，返回最佳走法（Move） */
Move search_root(Board *board, int max_depth, int time_ms);

/* 公开的计时器取消（如果需要外部强制中断） */
void search_stop(void);

#endif
