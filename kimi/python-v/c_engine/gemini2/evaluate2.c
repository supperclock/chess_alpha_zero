#include "chess.h"
#include <stddef.h>


// --- 兵/卒位置表 (PST) ---
static const int PST_R_PAWN[ROWS][COLS] = {
    {  0,   0,   0,   0,   0,   0,   0,   0,   0},
    {  0,   0,   0,   0,   0,   0,   0,   0,   0},
    {  0,   0,   0,   0,   0,   0,   0,   0,   0},
    { -5,  -5,  -5, -10, -10, -10,  -5,  -5,  -5},
    {-10, -10, -10, -15, -15, -15, -10, -10, -10},
    {-15, -15, -15, -20, -20, -20, -15, -15, -15},
    {-20, -20, -20, -25, -25, -25, -20, -20, -20},
    {-25, -25, -25, -30, -30, -30, -25, -25, -25},
    {-30, -30, -30, -35, -35, -35, -30, -30, -30},
    {-35, -35, -35, -40, -40, -40, -35, -35, -35}
};

static const int PST_B_PAWN[ROWS][COLS] = {
    { 35,  35,  35,  40,  40,  40,  35,  35,  35},
    { 30,  30,  30,  35,  35,  35,  30,  30,  30},
    { 25,  25,  25,  30,  30,  30,  25,  25,  25},
    { 20,  20,  20,  25,  25,  25,  20,  20,  20},
    { 15,  15,  15,  20,  20,  20,  15,  15,  15},
    { 10,  10,  10,  15,  15,  15,  10,  10,  10},
    {  5,   5,   5,  10,  10,  10,   5,   5,   5},
    {  0,   0,   0,   0,   0,   0,   0,   0,   0},
    {  0,   0,   0,   0,   0,   0,   0,   0,   0},
    {  0,   0,   0,   0,   0,   0,   0,   0,   0}
};

// --- 辅助函数 ---
const int (*get_pst_for_piece(Piece p))[COLS] {
    if (p == r_pawn) return PST_R_PAWN;
    if (p == b_pawn) return PST_B_PAWN;
    return NULL;
}

int get_mobility_score(const BoardState* state, int y, int x, Piece p) {
    Move moves[MAX_MOVES];
    Side side = (p >= b_king) ? BLACK : RED;
    int count = 0;
    switch(p) {
        case r_king: case b_king: count = gen_king_moves(state, y, x, side, moves); break;
        case r_advisor: case b_advisor: count = gen_advisor_moves(state, y, x, side, moves); break;
        case r_elephant: case b_elephant: count = gen_elephant_moves(state, y, x, side, moves); break;
        case r_horse: case b_horse: count = gen_horse_moves(state, y, x, side, moves); break;
        case r_chariot: case b_chariot: count = gen_chariot_moves(state, y, x, side, moves); break;
        case r_cannon: case b_cannon: count = gen_cannon_moves(state, y, x, side, moves); break;
        case r_pawn: case b_pawn: count = gen_pawn_moves(state, y, x, side, moves); break;
        default: break;
    }
    return count * 2; // 保持低权重，稳定搜索
}

// --- 主评估函数 ---
int evaluate_board(const BoardState* state) {
    int score = 0;

    for (int y = 0; y < ROWS; ++y) {
        for (int x = 0; x < COLS; ++x) {
            Piece p = state->board[y][x];
            if (p == EMPTY) continue;

            const int (*pst)[COLS] = get_pst_for_piece(p);
            int pst_score = (pst != NULL) ? pst[y][x] : 0;

            int piece_value = PIECE_VALUES[p];
            int mobility_score = get_mobility_score(state, y, x, p);

            int center_bonus = (x >= 3 && x <= 5 && y >= 3 && y <= 6) ? 5 : 0;

            int shield_bonus = 0;
            if (p == r_advisor || p == r_elephant) shield_bonus = -30;
            if (p == b_advisor || p == b_elephant) shield_bonus = 30;

            if (get_piece_side(p) == BLACK) {
                score += piece_value + pst_score + mobility_score + center_bonus + shield_bonus;
            } else {
                score -= piece_value + pst_score + mobility_score + center_bonus + shield_bonus;
            }
        }
    }

    // 稍微放大分数差异，提高搜索果断性
    return (int)(score * 1.2);
}
