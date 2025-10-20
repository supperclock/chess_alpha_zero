#include <stdio.h>
#include <string.h> // for memset
#include "chess.h"

// --- 棋子基础价值 ---
const int PIECE_VALUES[15] = {
    0,         // EMPTY
    MATE_SCORE, 200, 200, 450, 900, 800, 100, // Red
    MATE_SCORE, 200, 200, 450, 900, 800, 100  // Black
};

// --- PST 定义 (预计算) ---
// [阵营: 0=RED, 1=BLACK][y][x]

// 兵卒 PST
const int PST_PAWN[2][ROWS][COLS] = {
    { // Red Pawn (黑方视角, 负分)
        {  0,   0,   0,   0,   0,   0,   0,   0,   0},
        {  0,   0,   0,   0,   0,   0,   0,   0,   0},
        {  0,   0,   0,   0,   0,   0,   0,   0,   0},
        { -5,  -5,  -5, -10, -10, -10,  -5,  -5,  -5},
        { -5,  -5,  -5, -10, -10, -10,  -5,  -5,  -5},
        {-10, -10, -10, -15, -15, -15, -10, -10, -10},
        {-15, -15, -15, -20, -20, -20, -15, -15, -15},
        {-20, -20, -20, -25, -25, -25, -20, -20, -20},
        {-25, -25, -25, -30, -30, -30, -25, -25, -25},
        {-30, -30, -30, -35, -35, -35, -30, -30, -30}
    },
    { // Black Pawn (黑方视角, 正分)
        { 30,  30,  30,  35,  35,  35,  30,  30,  30},
        { 25,  25,  25,  30,  30,  30,  25,  25,  25},
        { 20,  20,  20,  25,  25,  25,  20,  20,  20},
        { 15,  15,  15,  20,  20,  20,  15,  15,  15},
        { 10,  10,  10,  15,  15,  15,  10,  10,  10},
        {  5,   5,   5,  10,  10,  10,   5,   5,   5},
        {  5,   5,   5,  10,  10,  10,   5,   5,   5},
        {  0,   0,   0,   0,   0,   0,   0,   0,   0},
        {  0,   0,   0,   0,   0,   0,   0,   0,   0},
        {  0,   0,   0,   0,   0,   0,   0,   0,   0}
    }
};
// 车 PST
const int PST_CHARIOT[2][ROWS][COLS] = {
    { // Red Chariot
        {-15,-15,-15,-20,-20,-20,-15,-15,-15}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}
    },
    { // Black Chariot
        {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {15,15,15,20,20,20,15,15,15}
    }
};
// 炮 PST
const int PST_CANNON[2][ROWS][COLS] = {
    { // Red Cannon
        {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {-10,-10,-10,-10,-10,-10,-10,-10,-10}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}
    },
    { // Black Cannon
        {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}, {10,10,10,10,10,10,10,10,10}, {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0}
    }
};

// 其他棋子PST（在ai.py中未定义，为0）
const int PST_HORSE[2][ROWS][COLS] = {0};
const int PST_ELEPHANT[2][ROWS][COLS] = {0};
const int PST_ADVISOR[2][ROWS][COLS] = {0};
const int PST_KING[2][ROWS][COLS] = {0};

// --- 辅助函数 ---

// 根据棋子获取PST
static int get_pst_score(Piece p, int y, int x) {
    Side side = get_piece_side(p);
    switch(p) {
        case r_pawn: case b_pawn:     return PST_PAWN[side][y][x];
        case r_chariot: case b_chariot: return PST_CHARIOT[side][y][x];
        case r_cannon: case b_cannon:   return PST_CANNON[side][y][x];
        // 其他棋子PST为0，无需查询
        default: return 0;
    }
}

// 辅助函数：计算机动性
static int get_mobility_count(const BoardState* state, int y, int x, Piece p, Move* move_list) {
    Side side = get_piece_side(p);
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

// 辅助函数：计算王的安全性
static int get_king_safety_score(const BoardState* state, Side side,
                                 int attack_sum[2][ROWS][COLS], const int piece_counts[15]) {
    int ky, kx;
    if (!find_general(state, side, &ky, &kx)) return -MATE_SCORE; // 王没了

    int palace_attack_val = 0;
    int y_start = (side == RED) ? 0 : 7;
    int y_end = (side == RED) ? 3 : 10;
    int* opp_attack_map = (int*)attack_sum[1 - side];

    for (int y = y_start; y < y_end; ++y) {
        for (int x = 3; x <= 5; ++x) {
            palace_attack_val += opp_attack_map[y * COLS + x];
        }
    }

    // 计算防守子力 (从 piece_counts 获取，比 ai.py 遍历棋盘更高效)
    int advisors = (side == RED) ? piece_counts[r_advisor] : piece_counts[b_advisor];
    int elephants = (side == RED) ? piece_counts[r_elephant] : piece_counts[b_elephant];
    int shield_count = advisors + elephants;
    
    int val = (palace_attack_val / 20) * 18 - (4 - shield_count) * 60;
    val += (opp_attack_map[ky * COLS + kx] / 20) * 25;
    
    // (ai.py: count('black',['將'])==1 and count('black',['士','象'])==0)
    // 裸王惩罚
    if (shield_count == 0) {
        val += 300; // 注意：这里是惩罚值，所以是正数 (因为函数返回的是惩罚)
    }
    
    return val;
}

// 辅助函数：计算协同性
static int get_coordination_score(const BoardState* state) {
    int score = 0;
    // --- 炮架马 (ai.py: cannon_horse_bonus) ---
    for (int y = 0; y < ROWS; ++y) {
        for (int x = 0; x < COLS; ++x) {
            Piece p = state->board[y][x];
            Side side;
            Piece friendly_horse;
            int mult;

            if (p == r_cannon) { side = RED; mult = -1; friendly_horse = r_horse; }
            else if (p == b_cannon) { side = BLACK; mult = 1; friendly_horse = b_horse; }
            else continue;

            // 检查4个相邻格子
            const int DIRS[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (int i = 0; i < 4; ++i) {
                int ny = y + DIRS[i][0];
                int nx = x + DIRS[i][1];
                if (is_inside(ny, nx) && state->board[ny][nx] == friendly_horse) {
                    score += 60 * mult;
                }
            }
        }
    }

    // --- 双车同线 (ai.py: rook_bonus) ---
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
    if (rook_count[RED] == 2 && (rook_y[RED][0] == rook_y[RED][1] || rook_x[RED][0] == rook_x[RED][1])) {
        score -= 120; // 红方优势，黑方视角为负
    }
    if (rook_count[BLACK] == 2 && (rook_y[BLACK][0] == rook_y[BLACK][1] || rook_x[BLACK][0] == rook_x[BLACK][1])) {
        score += 120; // 黑方优势，黑方视角为正
    }
    return score;
}

// 辅助函数：计算兵结构
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

            // 检查水平相邻的兵 (ai.py: pawn_structure)
            int left_neighbor = (is_inside(y, x - 1) && state->board[y][x - 1] == friendly_pawn);
            int right_neighbor = (is_inside(y, x + 1) && state->board[y][x + 1] == friendly_pawn);

            if (left_neighbor) score += 30 * mult;
            if (right_neighbor) score += 30 * mult;
            
            // 孤兵惩罚
            if (!left_neighbor && !right_neighbor) {
                score -= 18 * mult;
            }
        }
    }
    return score;
}

// 辅助函数：残局特判
static int get_endgame_score(const int piece_counts[15]) {
    int score = 0;
    
    // 从数组中获取棋子数量，非常高效
    int bp = piece_counts[b_pawn];
    int rp = piece_counts[r_pawn];
    int br = piece_counts[b_chariot];
    int rr = piece_counts[r_chariot];
    int bc = piece_counts[b_cannon];
    int rc = piece_counts[r_cannon];
    int bh = piece_counts[b_horse];
    int rh = piece_counts[r_horse];

    // 翻译 ai.py 中的残局逻辑
    if (bp == 1 && br == 1 && (bc+bh)==0 && (rr+rc+rh+rp)==1) score += 140;
    if (rp == 1 && rr == 1 && (rc+rh)==0 && (br+bc+bh+bp)==1) score -= 140;

    if (bp >= 1 && bc >= 1 && rh == 1 && rr == 0) score += 90;
    if (rp >= 1 && rc >= 1 && bh == 1 && br == 0) score -= 90;

    if (bc == 2 && rh == 1 && (rr+rp+rc)==0) score += 110;
    if (rc == 2 && bh == 1 && (br+bp+bc)==0) score -= 110;

    if (br >= 2 && (rr+rh+rc) <= 1) score += 180;
    if (rr >= 2 && (br+bh+bc) <= 1) score -= 180;
    
    // 裸王惩罚（已移至 get_king_safety_score）

    return score;
}


// --- 主评估函数 ---
int evaluate_board(const BoardState* state) {
    int score = 0;
    int total_material = 0;
    int black_mob = 0;
    int red_mob = 0;
    Move temp_moves[MAX_MOVES]; // 临时空间

    // VVVV --- 优化：新增棋子计数数组 --- VVVV
    int piece_counts[15];
    memset(piece_counts, 0, sizeof(piece_counts));
    // ^^^^ --- 优化：新增棋子计数数组 --- ^^^^

    // 攻击图: [阵营][y][x]
    // 1. 攻击总价值
    int attack_sum[2][ROWS][COLS];
    // 2. 最小攻击者价值
    int min_attacker[2][ROWS][COLS];
    
    memset(attack_sum, 0, sizeof(attack_sum));
    for (int i = 0; i < 2 * ROWS * COLS; ++i) {
        ((int*)min_attacker)[i] = MATE_SCORE; // 初始化为无穷大
    }

    // --- 阶段一：数据收集 (机动性和攻击图) ---
    for (int y = 0; y < ROWS; ++y) {
        for (int x = 0; x < COLS; ++x) {
            Piece p = state->board[y][x];
            if (p == EMPTY) continue;

            Side side = get_piece_side(p);
            int value = PIECE_VALUES[p];
            total_material += value;

            // VVVV --- 优化：统计棋子数量 --- VVVV
            piece_counts[p]++;
            // ^^^^ --- 优化：统计棋子数量 --- ^^^^

            int num_moves = get_mobility_count(state, y, x, p, temp_moves);
            
            if (side == BLACK) black_mob += num_moves;
            else red_mob += num_moves;

            // 更新攻击图
            for (int i = 0; i < num_moves; ++i) {
                Move* m = &temp_moves[i];
                attack_sum[side][m->to_y][m->to_x] += value;
                if (value < min_attacker[side][m->to_y][m->to_x]) {
                    min_attacker[side][m->to_y][m->to_x] = value;
                }
            }
        }
    }
    
    // 游戏阶段 (0.0 - 1.0)
    double phase = (total_material > 16000) ? 1.0 : (total_material / 16000.0);

    // --- 阶段二：计分 ---
    int safety_penalty = 0;
    for (int y = 0; y < ROWS; ++y) {
        for (int x = 0; x < COLS; ++x) {
            Piece p = state->board[y][x];
            if (p == EMPTY) continue;

            Side side = get_piece_side(p);
            int mult = (side == BLACK) ? 1 : -1;
            int base_value = PIECE_VALUES[p];
            int pst_score = get_pst_score(p, y, x);

            // 1. 子力 + 位置分
            score += (base_value + pst_score) * mult;
            
            // 2. 棋子安全 (ai.py: PIECE SAFETY EVALUATION)
            int opp_min_attacker_val = min_attacker[1 - side][y][x];
            if (opp_min_attacker_val < base_value) {
                // (黑方视角) 黑子被攻击，总分降低；红子被攻击，总分升高
                safety_penalty -= (int)(base_value * 0.4) * mult;
            }
        }
    }
    
    score += safety_penalty;
    
    // 3. 机动性
    score += (int)((black_mob - red_mob) * (10 * (0.6 + 0.4 * phase)));

    // 4. 王的安全
    score += get_king_safety_score(state, BLACK, attack_sum, piece_counts);
    score -= get_king_safety_score(state, RED, attack_sum, piece_counts);

    // 5. 协同性 (为保持简洁，已简化，可按需扩展)
    score += get_coordination_score(state);

    // 6. 兵结构 (为保持简洁，已跳过，可按需扩展)
    score += get_pawn_structure_score(state);

    // 7. 残局 (为保持简洁，已跳过，可按需扩展)
    score += get_endgame_score(piece_counts);

    return score;
}