#include <stdio.h>
#include <string.h> // for memset
#include "chess.h"
// evaluation.h 已经被 "chess.h" 包含

// --- PIECE_VALUES 和 PST 数组 (保持不变) ---

const int PST_PAWN[2][ROWS][COLS] = {
    {{0,0,0,0,0,0,0,0,0}, {-5,-5,-5,-10,-10,-10,-5,-5,-5}, {-5,-5,-5,-10,-10,-10,-5,-5,-5}, {-10,-10,-10,-15,-15,-15,-10,-10,-10}, {-15,-15,-15,-20,-20,-20,-15,-15,-15}, {-20,-20,-20,-25,-25,-25,-20,-20,-20}, {-25,-25,-25,-30,-30,-30,-25,-25,-25}, {-30,-30,-30,-35,-35,-35,-30,-30,-30}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}},
    {{0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {30,30,30,35,35,35,30,30,30}, {25,25,25,30,30,30,25,25,25}, {20,20,20,25,25,25,20,20,20}, {15,15,15,20,20,20,15,15,15}, {10,10,10,15,15,15,10,10,10}, {5,5,5,10,10,10,5,5,5}, {5,5,5,10,10,10,5,5,5}, {0,0,0,0,0,0,0,0,0}}
};
const int PST_CHARIOT[2][ROWS][COLS] = {
    {{-15,-15,-15,-20,-20,-20,-15,-15,-15}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}},
    {{0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {15,15,15,20,20,20,15,15,15}}
};
const int PST_CANNON[2][ROWS][COLS] = {
    {{0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {-10,-10,-10,-10,-10,-10,-10,-10,-10}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}},
    {{0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {10,10,10,10,10,10,10,10,10}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}}
};
const int PST_HORSE[2][ROWS][COLS] = {{0}};
const int PST_ELEPHANT[2][ROWS][COLS] = {{0}};
const int PST_ADVISOR[2][ROWS][COLS] = {{0}};
const int PST_KING[2][ROWS][COLS] = {{0}};

// --- 辅助函数 (不变) ---
static int get_pst_score(Piece p, int y, int x) {
    Side side = get_piece_side(p);
    if (side == -1) return 0;
    switch(p) {
        case r_pawn: case b_pawn:     return PST_PAWN[side][y][x];
        case r_chariot: case b_chariot: return PST_CHARIOT[side][y][x];
        case r_cannon: case b_cannon:   return PST_CANNON[side][y][x];
        default: return 0;
    }
}

static int get_mobility_count(const BoardState* state, int y, int x, Piece p, Move* move_list) {
    Side side = get_piece_side(p);
    if (side == -1) return 0;
    switch(p) {
        case r_king: case b_king:       return gen_king_moves(state, y, x, side, move_list);
        case r_advisor: case b_advisor: return gen_advisor_moves(state, y, x, side, move_list);
        case r_elephant: case b_elephant: return gen_elephant_moves(state, y, x, side, move_list);
        case r_horse: case b_horse:     return gen_horse_moves(state, y, x, side, move_list);
        case r_chariot: case b_chariot: return gen_chariot_moves(state, y, x, side, move_list);
        case r_cannon: case b_cannon:   return gen_cannon_moves(state, y, x, side, move_list);
        case r_pawn: case b_pawn:       return gen_pawn_moves(state, y, x, side, move_list);
        default: return 0;
    }
}

// --- VVVV 优化后的函数 VVVV ---

/**
 * @brief 优化: 削弱王安全和缺士象的惩罚
 */
static int get_king_safety_score(const BoardState* state, Side side,
                                 int attack_sum[2][ROWS][COLS], const int piece_counts[15]) {
    int ky, kx;
    if (!find_general(state, side, &ky, &kx)) return -MATE_SCORE; 

    int palace_attack_val = 0;
    int y_start = (side == RED) ? 0 : 7;
    int y_end = (side == RED) ? 3 : 10;
    int* opp_attack_map = (int*)attack_sum[1 - side];

    for (int y = y_start; y < y_end; ++y) {
        for (int x = 3; x <= 5; ++x) {
            palace_attack_val += opp_attack_map[y * COLS + x];
        }
    }

    int advisors = (side == RED) ? piece_counts[r_advisor] : piece_counts[b_advisor];
    int elephants = (side == RED) ? piece_counts[r_elephant] : piece_counts[b_elephant];
    int shield_count = advisors + elephants;
    
    // --- 优化: 削弱惩罚 ---
    // (2 - shield) * 60 -> (2 - shield) * 30
    int val = (palace_attack_val / 20) * 18 - (2 - shield_count) * 30;
    
    // 削弱对 "将" 本身被攻击的惩罚 (25 -> 15)
    val += (opp_attack_map[ky * COLS + kx] / 20) * 15;
    
    return val;
}

/**
 * @brief 优化: 削弱协同奖励
 */
static int get_coordination_score(const BoardState* state) {
    int score = 0;
    
    // --- 炮架马 (60 -> 30) ---
    for (int y = 0; y < ROWS; ++y) {
        for (int x = 0; x < COLS; ++x) {
            Piece p = state->board[y][x];
            Side side;
            Piece friendly_horse;
            int mult;

            if (p == r_cannon) { side = RED; mult = -1; friendly_horse = r_horse; }
            else if (p == b_cannon) { side = BLACK; mult = 1; friendly_horse = b_horse; }
            else continue;

            const int DIRS[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (int i = 0; i < 4; ++i) {
                int ny = y + DIRS[i][0];
                int nx = x + DIRS[i][1];
                if (is_inside(ny, nx) && state->board[ny][nx] == friendly_horse) {
                    score += 30 * mult; // 60 -> 30
                }
            }
        }
    }

    // --- 双车同线 (120 -> 50) ---
    int rook_y[2][2]; // [side][index]
    int rook_x[2][2];
    int rook_count[2] = {0, 0};
    for (int y = 0; y < ROWS; ++y) {
        for (int x = 0; x < COLS; ++x) {
            Piece p = state->board[y][x];
            if (p == r_chariot && rook_count[RED] < 2) {
                rook_y[RED][rook_count[RED]] = y;
                rook_x[RED][rook_count[RED]] = x;
                rook_count[RED]++;
            } else if (p == b_chariot && rook_count[BLACK] < 2) {
                rook_y[BLACK][rook_count[BLACK]] = y;
                rook_x[BLACK][rook_count[BLACK]] = x;
                rook_count[BLACK]++;
            }
        }
    }
    // 120 -> 50 (奖励不应超过一个兵)
    if (rook_count[RED] == 2 && (rook_y[RED][0] == rook_y[RED][1] || rook_x[RED][0] == rook_x[RED][1])) {
        score -= 50;
    }
    if (rook_count[BLACK] == 2 && (rook_y[BLACK][0] == rook_y[BLACK][1] || rook_x[BLACK][0] == rook_x[BLACK][1])) {
        score += 50;
    }
    
    return score;
}

/**
 * @brief 优化: 削弱兵结构奖励
 */
static int get_pawn_structure_score(const BoardState* state) {
    int score = 0;
    for (int y = 0; y < ROWS; ++y) {
        for (int x = 0; x < COLS; ++x) {
            Piece p = state->board[y][x];
            Piece friendly_pawn;
            int mult;

            if (p == r_pawn) { friendly_pawn = r_pawn; mult = -1; }
            else if (p == b_pawn) { friendly_pawn = b_pawn; mult = 1; }
            else continue;

            int left_neighbor = (is_inside(y, x - 1) && state->board[y][x - 1] == friendly_pawn);
            int right_neighbor = (is_inside(y, x + 1) && state->board[y][x + 1] == friendly_pawn);

            // 30 -> 15
            if (left_neighbor) score += 15 * mult;
            if (right_neighbor) score += 15 * mult;
            
            // 18 -> 10
            if (!left_neighbor && !right_neighbor) {
                score -= 10 * mult;
            }
        }
    }
    return score;
}

/**
 * @brief 优化: 削弱残局奖励, 智能化 "裸王惩罚"
 */
static int get_endgame_score(const int piece_counts[15]) {
    int score = 0;
    
    int bp = piece_counts[b_pawn];
    int rp = piece_counts[r_pawn];
    int br = piece_counts[b_chariot];
    int rr = piece_counts[r_chariot];
    int bc = piece_counts[b_cannon];
    int rc = piece_counts[r_cannon];
    int bh = piece_counts[b_horse];
    int rh = piece_counts[r_horse];
    
    // --- 优化: 裸王惩罚 (300 -> 120, 且仅在对方有攻击子时) ---
    int b_shield = piece_counts[b_advisor] + piece_counts[b_elephant];
    int r_shield = piece_counts[r_advisor] + piece_counts[r_elephant];
    
    // 计算双方的攻击子 (车马炮)
    int r_attackers = piece_counts[r_chariot] + piece_counts[r_cannon] + piece_counts[r_horse];
    int b_attackers = piece_counts[b_chariot] + piece_counts[b_cannon] + piece_counts[b_horse];

    // 只有在对方*有*攻击子力时, "裸王" 才是问题, 并且惩罚降低到 120
    if (piece_counts[b_king] == 1 && b_shield == 0 && r_attackers > 0) score -= 120;
    if (piece_counts[r_king] == 1 && r_shield == 0 && b_attackers > 0) score += 120;
    // --- 修正结束 ---

    // --- 优化: 削弱 (减半) 所有其他残局奖励 ---
    if (bp == 1 && br == 1 && (bc+bh)==0 && (rr+rc+rh+rp)==1) score += 70; // 140->70
    if (rp == 1 && rr == 1 && (rc+rh)==0 && (br+bc+bh+bp)==1) score -= 70; // 140->70

    if (bp >= 1 && bc >= 1 && rh == 1 && rr == 0) score += 45; // 90->45
    if (rp >= 1 && rc >= 1 && bh == 1 && br == 0) score -= 45; // 90->45

    if (bc == 2 && rh == 1 && (rr+rp+rc)==0) score += 55; // 110->55
    if (rc == 2 && bh == 1 && (br+bp+bc)==0) score -= 55; // 110->55

    if (br >= 2 && (rr+rh+rc) <= 1) score += 90; // 180->90
    if (rr >= 2 && (br+bh+bc) <= 1) score -= 90; // 180->90

    return score;
}


int evaluate_board(const BoardState* state) {
    int score = 0;
    Move temp_moves[MAX_MOVES];

    int attack_map[2][ROWS][COLS];
    memset(attack_map, 0, sizeof(attack_map));

    int piece_counts[15] = {0};

    // --- 构建攻击地图 ---
    for (int y = 0; y < ROWS; ++y) {
        for (int x = 0; x < COLS; ++x) {
            Piece p = state->board[y][x];
            if (p == EMPTY) continue;
            piece_counts[p]++;
            Side side = get_piece_side(p);
            int num = get_mobility_count(state, y, x, p, temp_moves);
            for (int i = 0; i < num; ++i) {
                Move m = temp_moves[i];
                attack_map[side][m.to_y][m.to_x]++;
            }
        }
    }

    // --- 主体评估 ---
    for (int y = 0; y < ROWS; ++y) {
        for (int x = 0; x < COLS; ++x) {
            Piece p = state->board[y][x];
            if (p == EMPTY) continue;

            Side side = get_piece_side(p);
            int mult = (side == BLACK) ? 1 : -1;

            int base_value = PIECE_VALUES[p];
            int pst_score = get_pst_score(p, y, x);

            // 动态机动性权重
            int num_moves = get_mobility_count(state, y, x, p, temp_moves);
            int mobility_weight = 2;
            if (p == r_chariot || p == b_chariot) mobility_weight = 5;
            else if (p == r_cannon || p == b_cannon) mobility_weight = 4;
            else if (p == r_horse || p == b_horse) mobility_weight = 3;
            else if (p == r_pawn || p == b_pawn) mobility_weight = 1.5;
            int mobility_score = (int)(num_moves * mobility_weight);

            // 过河兵奖励
            if ((p == r_pawn && y >= 5) || (p == b_pawn && y <= 4))
                pst_score += 15 * ((p == b_pawn) ? 1 : -1);

            score += (base_value + pst_score + mobility_score) * mult;
        }
    }

    // --- 简化王安全：仅检测宫区攻击 ---
    int red_safety = 0, black_safety = 0;
    for (int y = 0; y < ROWS; ++y) {
        for (int x = 3; x <= 5; ++x) {
            if (y <= 2) red_safety -= attack_map[BLACK][y][x] * 5;
            if (y >= 7) black_safety += attack_map[RED][y][x] * 5;
        }
    }
    score += red_safety + black_safety;

    // --- 轻度残局奖励 ---
    score += get_endgame_score(piece_counts);

    return score;
}
