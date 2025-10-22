// chess.h (最小定义)
#pragma once

#include <stdbool.h>

#define ROWS 10
#define COLS 9
#define MATE_SCORE 1000000 // 将死分值
#define MAX_MOVES 256 // 评估函数中也用到了

// 棋子类型 (必须从 0 开始或提供映射)
// 为 NNUE 方便, 我们让 EMPTY=0, r_king=1 ... b_pawn=14
typedef enum {
    EMPTY,
    r_king, r_advisor, r_elephant, r_horse, r_chariot, r_cannon, r_pawn,
    b_king, b_advisor, b_elephant, b_horse, b_chariot, b_cannon, b_pawn,
    PIECE_TYPE_NB // 15
} Piece;

// 阵营
typedef enum {
    RED, BLACK
} Side;

// 走法结构体 (Move)
// 存储走法的起点、终点、被吃掉的棋子（用于悔棋）和评分（用于排序）
typedef struct {
    int from_y, from_x;
    int to_y, to_x;
    Piece captured;
    int score;
} Move;

// 棋盘状态
typedef struct {
    Piece board[ROWS][COLS];
    Side side_to_move;
    // ... 其他状态 ...
    NnueAccumulator accumulator;
} BoardState;

// --- 必需的辅助函数 (NNUE 核心依赖它们) ---

/**
 * @brief 获取棋子阵营
 */
static inline Side get_piece_side(Piece p) {
    if (p >= r_king && p <= r_pawn) return RED;
    if (p >= b_king && p <= b_pawn) return BLACK;
    return -1; // Error
}

/**
 * @brief 查找指定阵营的 "将" (King/General)
 * @return 找到则返回 true, 并填充 y, x
 */
static inline bool find_general(const BoardState* state, Side side, int* y, int* x) {
    Piece king_to_find = (side == RED) ? r_king : b_king;
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            if (state->board[r][c] == king_to_find) {
                *y = r;
                *x = c;
                return true;
            }
        }
    }
    *y = -1; *x = -1;
    return false; // 找不到将 (局面非法)
}