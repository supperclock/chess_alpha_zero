#include <stdio.h>
#include <stdlib.h> // for qsort
#include <time.h>   // for time management
#include "chess.h"

// --- 全局/静态变量 ---
// 用于启发式搜索，如杀手走法。MAX_SEARCH_DEPTH需要足够大。
#define MAX_SEARCH_DEPTH 32
static Move killer_moves[MAX_SEARCH_DEPTH][2];

// 用于时间管理
static clock_t start_time;
static double time_limit;

// --- 前向声明 ---
// 因为pvs_search和quiescence_search相互（或间接）调用，最好提前声明。
int pvs_search(BoardState* state, int depth, int alpha, int beta, int color_multiplier, int ply);
int quiescence_search(BoardState* state, int alpha, int beta, int color_multiplier);

// --- 走法排序辅助函数 ---
// 这是Alpha-Beta剪枝效率的关键。一个好的排序能剪掉更多分支。
int compare_moves(const void* a, const void* b) {
    return ((Move*)b)->score - ((Move*)a)->score;
}

void score_moves(BoardState* state, Move* move_list, int num_moves, int ply) {
    for (int i = 0; i < num_moves; i++) {
        Move* m = &move_list[i];
        int score = 0;
        // 1. MVV-LVA (最有价值被吃子 - 最低价值攻击子)
        if (m->captured != EMPTY) {
            Piece attacker = state->board[m->from_y][m->from_x];
            score += PIECE_VALUES[m->captured] * 10 - PIECE_VALUES[attacker];
        }
        // 2. 杀手走法启发
        if (m->captured == EMPTY) {
            if (killer_moves[ply][0].from_y == m->from_y && killer_moves[ply][0].from_x == m->from_x &&
                killer_moves[ply][0].to_y == m->to_y && killer_moves[ply][0].to_x == m->to_x) {
                score += 9000;
            } else if (killer_moves[ply][1].from_y == m->from_y && killer_moves[ply][1].from_x == m->from_x &&
                       killer_moves[ply][1].to_y == m->to_y && killer_moves[ply][1].to_x == m->to_x) {
                score += 8000;
            }
        }
        m->score = score;
    }
    qsort(move_list, num_moves, sizeof(Move), compare_moves);
}

// --- 静态搜索 ---
int quiescence_search(BoardState* state, int alpha, int beta, int color_multiplier) {
    // 检查时间
    if ((double)(clock() - start_time) / CLOCKS_PER_SEC > time_limit) {
        return 0; // 超时
    }

    // 静态评估分数
    int stand_pat = evaluate_board(state) * color_multiplier;
    if (stand_pat >= beta) {
        return beta;
    }
    if (stand_pat > alpha) {
        alpha = stand_pat;
    }

    Move move_list[MAX_MOVES];
    int num_moves = generate_moves(state, move_list);
    
    // 只考虑吃子走法
    Move capture_moves[MAX_MOVES];
    int capture_count = 0;
    for(int i = 0; i < num_moves; ++i) {
        if (move_list[i].captured != EMPTY) {
            capture_moves[capture_count++] = move_list[i];
        }
    }
    
    score_moves(state, capture_moves, capture_count, 0); // ply=0 for simplicity

    for (int i = 0; i < capture_count; i++) {
        Move current_move = capture_moves[i];
        make_move(state, &current_move);
        int score = -quiescence_search(state, -beta, -alpha, -color_multiplier);
        unmake_move(state, &current_move);

        if (score >= beta) {
            return beta;
        }
        if (score > alpha) {
            alpha = score;
        }
    }
    return alpha;
}

// --- PVS 主搜索函数 ---
int pvs_search(BoardState* state, int depth, int alpha, int beta, int color_multiplier, int ply) {
    // 检查时间
    if ((double)(clock() - start_time) / CLOCKS_PER_SEC > time_limit) {
        return 0; // 超时
    }
    
    if (depth <= 0) {
        return quiescence_search(state, alpha, beta, color_multiplier);
    }

    Move move_list[MAX_MOVES];
    int num_moves = generate_moves(state, move_list);

    if (num_moves == 0) {
        if (is_in_check(state, state->side_to_move)) {
            return -MATE_SCORE + ply; // 被将死，分数和步数有关，越早被将死越差
        }
        return 0; // 逼和
    }

    score_moves(state, move_list, num_moves, ply);

    int first_move = 1;
    for (int i = 0; i < num_moves; i++) {
        Move current_move = move_list[i];
        make_move(state, &current_move);
        
        int score;
        if (first_move) {
            score = -pvs_search(state, depth - 1, -beta, -alpha, -color_multiplier, ply + 1);
            first_move = 0;
        } else {
            // 零窗口搜索
            score = -pvs_search(state, depth - 1, -alpha - 1, -alpha, -color_multiplier, ply + 1);
            // 如果零窗口搜索失败，说明这个走法可能很好，需要用完整窗口重新搜索
            if (score > alpha && score < beta) {
                score = -pvs_search(state, depth - 1, -beta, -alpha, -color_multiplier, ply + 1);
            }
        }
        unmake_move(state, &current_move);

        if (score > alpha) {
            alpha = score;
        }
        
        if (alpha >= beta) {
            // Beta 截断
            if (current_move.captured == EMPTY) { // 只记录非吃子走法
                killer_moves[ply][1] = killer_moves[ply][0];
                killer_moves[ply][0] = current_move;
            }
            return beta;
        }
    }
    return alpha;
}


// --- 根函数 (入口) ---
Move find_best_move(BoardState* state, int max_depth, double time_limit_sec) {
    start_time = clock();
    time_limit = time_limit_sec;
    
    Move best_move = {0};
    int best_score = -MATE_SCORE - 1;
    int color_multiplier = (state->side_to_move == BLACK) ? 1 : -1;

    // 清空杀手走法表
    for (int i = 0; i < MAX_SEARCH_DEPTH; ++i) {
        killer_moves[i][0] = (Move){0};
        killer_moves[i][1] = (Move){0};
    }

    // 迭代加深
    for (int current_depth = 1; current_depth <= max_depth; ++current_depth) {
        Move move_list[MAX_MOVES];
        int num_moves = generate_moves(state, move_list);
        score_moves(state, move_list, num_moves, 0);

        int alpha = -MATE_SCORE - 1;
        int beta = MATE_SCORE + 1;
        
        for (int i = 0; i < num_moves; ++i) {
            Move current_move = move_list[i];
            make_move(state, &current_move);
            int score = -pvs_search(state, current_depth - 1, -beta, -alpha, -color_multiplier, 1);
            unmake_move(state, &current_move);

            // 在根节点，alpha是当前找到的最佳分数
            if (score > best_score) {
                best_score = score;
                best_move = current_move;
                alpha = score;
            }
        }

        double elapsed_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
        printf("Info: depth %d, score %d, bestmove %c%d-%c%d, time %.2fs\n", 
               current_depth, best_score * color_multiplier,
               'a'+best_move.from_x, best_move.from_y,
               'a'+best_move.to_x, best_move.to_y,
               elapsed_time);

        // 如果找到绝杀，或者时间用完，就提前退出
        if (best_score >= MATE_SCORE - MAX_SEARCH_DEPTH) break;
        if (elapsed_time > time_limit / 2) break; // 启发式时间管理
    }

    return best_move;
}