#include "chess.h"
#include <stddef.h> 

// --- 评估常量和位置表 (PST) ---

// 棋子基础价值
// 数组的索引对应 Piece 枚举值
static const int PIECE_VALUES[] = {
    0,         // EMPTY
    MATE_SCORE, // r_king
    200,       // r_advisor
    200,       // r_elephant
    450,       // r_horse
    900,       // r_chariot
    800,       // r_cannon
    100,       // r_pawn
    MATE_SCORE, // b_king
    200,       // b_advisor
    200,       // b_elephant
    450,       // b_horse
    900,       // b_chariot
    800,       // b_cannon
    100        // b_pawn
};

// 兵/卒的位置附加值 (Piece-Square Table)
// 为了简化，我们只实现兵的PST，其他棋子在Python中比较简单，可以直接翻译。
// 红兵 PST (黑方视角，所以红兵位置越好，分数越负)
static const int PST_R_PAWN[ROWS][COLS] = {
    {  0,   0,   0,   0,   0,   0,   0,   0,   0},
    {  0,   0,   0,   0,   0,   0,   0,   0,   0},
    {  0,   0,   0,   0,   0,   0,   0,   0,   0},
    { -5,  -5,  -5, -10, -10, -10,  -5,  -5,  -5}, // base
    {-10, -10, -10, -15, -15, -15, -10, -10, -10},
    {-15, -15, -15, -20, -20, -20, -15, -15, -15}, // river crossed
    {-20, -20, -20, -25, -25, -25, -20, -20, -20},
    {-25, -25, -25, -30, -30, -30, -25, -25, -25},
    {-30, -30, -30, -35, -35, -35, -30, -30, -30},
    {-35, -35, -35, -40, -40, -40, -35, -35, -35}
};

// 黑卒 PST (黑方视角，所以黑卒位置越好，分数越正)
static const int PST_B_PAWN[ROWS][COLS] = {
    { 35,  35,  35,  40,  40,  40,  35,  35,  35},
    { 30,  30,  30,  35,  35,  35,  30,  30,  30},
    { 25,  25,  25,  30,  30,  30,  25,  25,  25},
    { 20,  20,  20,  25,  25,  25,  20,  20,  20},
    { 15,  15,  15,  20,  20,  20,  15,  15,  15}, // river crossed
    { 10,  10,  10,  15,  15,  15,  10,  10,  10},
    {  5,   5,   5,  10,  10,  10,   5,   5,   5}, // base
    {  0,   0,   0,   0,   0,   0,   0,   0,   0},
    {  0,   0,   0,   0,   0,   0,   0,   0,   0},
    {  0,   0,   0,   0,   0,   0,   0,   0,   0}
};

// 辅助函数：根据棋子返回对应的PST
const int (*get_pst_for_piece(Piece p))[COLS] {
    if (p == r_pawn) return PST_R_PAWN;
    if (p == b_pawn) return PST_B_PAWN;
    return NULL; // 其他棋子暂时没有PST
}

// 辅助函数：计算机动性得分
int get_mobility_score(const BoardState* state, int y, int x, Piece p) {
    Move moves[MAX_MOVES];
    Side side = (p >= b_king) ? BLACK : RED;
    int count = 0;
    // C语言没有Python中优雅的GEN_MAP，我们用一个switch语句实现
    switch(p) {
        case r_king: case b_king:
            count = gen_king_moves(state, y, x, side, moves); break;
        case r_advisor: case b_advisor:
            count = gen_advisor_moves(state, y, x, side, moves); break;
        case r_elephant: case b_elephant:
            count = gen_elephant_moves(state, y, x, side, moves); break;
        case r_horse: case b_horse:
            count = gen_horse_moves(state, y, x, side, moves); break;
        case r_chariot: case b_chariot:
            count = gen_chariot_moves(state, y, x, side, moves); break;
        case r_cannon: case b_cannon:
            count = gen_cannon_moves(state, y, x, side, moves); break;
        case r_pawn: case b_pawn:
            count = gen_pawn_moves(state, y, x, side, moves); break;
        default: break;
    }
    return count;
}


// --- 主评估函数 ---
// 返回一个整数分数。正数表示黑方优势，负数表示红方优势。
int evaluate_board(const BoardState* state) {
    int score = 0;
    int total_material = 0; // 用于计算游戏阶段

    for (int y = 0; y < ROWS; ++y) {
        for (int x = 0; x < COLS; ++x) {
            Piece p = state->board[y][x];
            if (p == EMPTY) continue;

            int piece_value = PIECE_VALUES[p];
            total_material += piece_value;

            int pst_score = 0;
            const int (*pst)[COLS] = get_pst_for_piece(p);
            if (pst != NULL) {
                pst_score = pst[y][x];
            }
            
            // 机动性得分 (这里简化，只给一个较低的权重)
            int mobility_score = get_mobility_score(state, y, x, p) * 2;

            // 根据阵营加分或减分
            if (get_piece_side(p) == BLACK) {
                score += piece_value;
                score += pst_score;
                score += mobility_score;
            } else {
                score -= piece_value;
                score -= pst_score;
                score -= mobility_score;
            }
        }
    }
    
    // 可以在这里添加更多复杂的评估逻辑，如王的安全、兵结构、残局特判等
    // 为了保持本步骤的简洁性，我们暂时只实现子力和位置分。
    
    return score;
}