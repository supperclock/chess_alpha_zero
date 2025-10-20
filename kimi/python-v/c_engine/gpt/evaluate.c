#include "evaluate.h"
#include "board.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "rules.h"

/* ========== 常量定义 ========== */

#define BONUS_CANNON_HORSE 60
#define BONUS_PAWN_STRUCTURE 20
#define BONUS_ROOK_CONNECTION 40
#define PENALTY_WEAK_KING 80

const int PIECE_VALUES_STD[] = {
    0, 900, 500, 450, 250, 250, 10000, 100
};

/* 简单 PST（位置加分表，可按需微调） */
static const int PST_PAWN[10][9] = {
    {0,0,0,0,0,0,0,0,0},
    {5,10,10,20,20,10,10,10,5},
    {4,8,16,32,32,16,16,8,4},
    {3,6,12,24,24,12,12,6,3},
    {2,4,8,16,16,8,8,4,2},
    {1,2,4,8,8,4,4,2,1},
    {0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0}
};

/* ========== 辅助函数 ========== */

static inline int inside(int x, int y) {
    return x >= 0 && x < COLS && y >= 0 && y < ROWS;
}

/* ========== 各种附加评估项 ========== */

/* 炮马组合加分 */
static int cannon_horse_bonus(const Board *b, Side side) {
    int bonus = 0;
    static const int dirs[4][2] = {{-1,0},{1,0},{0,-1},{0,1}};
    for (int y=0;y<ROWS;y++) {
        for (int x=0;x<COLS;x++) {
            Piece p = b->sq[y][x];
            if (p.side != side || p.type != PT_CANNON) continue;
            for (int d=0; d<4; d++) {
                int nx = x + dirs[d][0];
                int ny = y + dirs[d][1];
                if (!inside(nx, ny)) continue;
                Piece q = b->sq[ny][nx];
                if (q.side == side && q.type == PT_HORSE)
                    bonus += (side == SIDE_BLACK) ? BONUS_CANNON_HORSE : -BONUS_CANNON_HORSE;
            }
        }
    }
    return bonus;
}

/* 兵形结构（同线兵） */
static int pawn_structure_bonus(const Board *b, Side side) {
    int bonus = 0;
    for (int x=0; x<COLS; x++) {
        int count = 0;
        for (int y=0; y<ROWS; y++) {
            Piece p = b->sq[y][x];
            if (p.side == side && p.type == PT_PAWN) count++;
        }
        if (count >= 2) bonus -= BONUS_PAWN_STRUCTURE * (count - 1);
    }
    return (side == SIDE_BLACK) ? bonus : -bonus;
}

/* 车连线 */
static int rook_connection_bonus(const Board *b, Side side) {
    int bonus = 0;
    for (int y=0;y<ROWS;y++) {
        int rx[2] = {-1, -1};
        int cnt = 0;
        for (int x=0;x<COLS;x++) {
            Piece p = b->sq[y][x];
            if (p.side == side && p.type == PT_ROOK) {
                if (cnt < 2) rx[cnt++] = x;
            }
        }
        if (cnt == 2) {
            int dist = abs(rx[0] - rx[1]);
            if (dist <= 3)
                bonus += (side == SIDE_BLACK) ? BONUS_ROOK_CONNECTION : -BONUS_ROOK_CONNECTION;
        }
    }
    return bonus;
}

/* 王安全评估 */
static int king_safety_penalty(const Board *b, Side side) {
    int penalty = 0;
    int gx = -1, gy = -1;
    for (int y=0;y<ROWS;y++)
        for (int x=0;x<COLS;x++)
            if (b->sq[y][x].side == side && b->sq[y][x].type == PT_GENERAL) {
                gx = x; gy = y;
            }
    if (gx < 0) return 0;

    /* 检查九宫内守卫数量 */
    int guards = 0;
    for (int dy=-1; dy<=1; dy++) {
        for (int dx=-1; dx<=1; dx++) {
            int nx = gx + dx;
            int ny = gy + dy;
            if (!inside(nx, ny)) continue;
            Piece p = b->sq[ny][nx];
            if (p.side == side && (p.type == PT_ADVISOR || p.type == PT_ELEPHANT))
                guards++;
        }
    }

    if (guards < 2) penalty += PENALTY_WEAK_KING * (2 - guards);
    return (side == SIDE_BLACK) ? penalty : -penalty;
}

/* 子力和位置分数 */
static int material_and_pst(const Board *b) {
    int score = 0;
    for (int y=0; y<ROWS; y++) {
        for (int x=0; x<COLS; x++) {
            Piece p = b->sq[y][x];
            if (p.type == PT_NONE) continue;
            int val = PIECE_VALUES_STD[p.type];
            if (p.type == PT_PAWN)
                val += (p.side == SIDE_RED) ? PST_PAWN[9 - y][x] : PST_PAWN[y][x];
            if (p.side == SIDE_RED) score += val;
            else score -= val;
        }
    }
    return score;
}

/* ========== 主评估函数 ========== */

int evaluate_board(const Board *board_state) {
    int score = 0;

    /* 子力 + PST */
    score += material_and_pst(board_state);

    /* 炮马组合 */
    score += cannon_horse_bonus(board_state, SIDE_BLACK);
    score += cannon_horse_bonus(board_state, SIDE_RED);

    /* 兵结构 */
    score += pawn_structure_bonus(board_state, SIDE_BLACK);
    score += pawn_structure_bonus(board_state, SIDE_RED);

    /* 车连线 */
    score += rook_connection_bonus(board_state, SIDE_BLACK);
    score += rook_connection_bonus(board_state, SIDE_RED);

    /* 王安全 */
    score -= king_safety_penalty(board_state, SIDE_BLACK);
    score -= king_safety_penalty(board_state, SIDE_RED);

    for (int y=0; y<ROWS; y++) {
        for (int x=0; x<COLS; x++) {
            Piece p = board_state->sq[y][x];
            if (p.type == PT_NONE) continue;
            Side opp = (p.side == SIDE_RED) ? SIDE_BLACK : SIDE_RED;
            if (square_attacked(board_state, x, y, opp)) {
                score -= PIECE_VALUES_STD[p.type] / 4; // 暴露风险
            }
        }
    }

    return score;
}
