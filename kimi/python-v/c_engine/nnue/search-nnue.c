// search-nnue.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "chess.h"
#include "nnue_core.h" // <--- 1. 包含新的 NNUE 核心头文件

/*
 * ======================================================================
 * * !!! 重要修改说明 !!!
 * * 1. 修改 'chess.h' 中的 'BoardState'
 * * 您 *必须* 将 NNUE 累加器添加到您的 'BoardState' 结构体中:
 *
 * typedef struct {
 * Piece board[ROWS][COLS];
 * Side side_to_move; // <--- 你的 'side_to_move' 变量
 * // ... 其他状态 ...
 * * // --- 新增 NNUE 累加器 ---
 * NnueAccumulator accumulator;
 * * } BoardState;
 * * 2. 替换 'make_move' 和 'unmake_move'
 * * 您*不能*再使用旧的 'make_move' 和 'unmake_move'。
 * 您必须使用下面提供的 *新版本*，它们包含了增量更新逻辑。
 * 您需要将您旧的棋盘更新逻辑 (例如移动棋子、切换走棋方)
 * 集成到下面新的 'make_move' 和 'unmake_move' 函数中。
 * * ======================================================================
 */


// --- 静态变量 (不变) ---
#define MAX_SEARCH_DEPTH 64
static Move killer_moves[MAX_SEARCH_DEPTH][2];
static unsigned long long hash_stack[MAX_SEARCH_DEPTH];

static clock_t start_time;
static double time_limit;

/*
 * 3. PIECE_VALUES 现在 *只* 用于走法排序 (MVV-LVA)
 * 它不再用于 *评估* 局面。
 */
const int PIECE_VALUES[] = {
    0, MATE_SCORE, 200, 200, 450, 900, 800, 100,
    MATE_SCORE, 200, 200, 450, 900, 800, 100
};


// --- 前向声明 (移除了 color_multiplier) ---
int pvs_search(BoardState* state, int depth, int alpha, int beta, int ply);
int quiescence_search(BoardState* state, int alpha, int beta, int ply);
void make_move(BoardState* state, const Move* m);
void unmake_move(BoardState* state, const Move* m);

// --- 走法排序 (不变) ---
int compare_moves(const void* a, const void* b) {
    return ((Move*)b)->score - ((Move*)a)->score;
}

void score_moves(BoardState* state, Move* move_list, int num_moves, int ply) {
    for (int i = 0; i < num_moves; i++) {
        Move* m = &move_list[i];
        int score = 0;
        if (m->captured != EMPTY) {
            Piece attacker = state->board[m->from_y][m->from_x];
            score += PIECE_VALUES[m->captured] * 10 - PIECE_VALUES[attacker];
        }
        if (m->captured == EMPTY) {
            if (memcmp(&killer_moves[ply][0], m, sizeof(Move)) == 0) score += 9000;
            else if (memcmp(&killer_moves[ply][1], m, sizeof(Move)) == 0) score += 8000;
        }
        m->score = score;
    }
    qsort(move_list, num_moves, sizeof(Move), compare_moves);
}

// --- Zobrist/位置哈希 (不变) ---
static unsigned long long compute_simple_hash(const BoardState* s) {
    unsigned long long h = 1469598103934665603ULL;
    for (int y = 0; y < ROWS; ++y) {
        for (int x = 0; x < COLS; ++x) {
            int v = (int)s->board[y][x];
            h ^= (unsigned long long)((v * 239017) ^ (y * 31 + x));
            h *= 1099511628211ULL;
        }
    }
    h ^= (unsigned long long)(s->side_to_move * 0x9e3779b97f4a7c15ULL);
    return h;
}


/*
 * ======================================================================
 * * 4. !!! 新的 make_move 和 unmake_move !!!
 * * 这些函数现在是 NNUE 核心的一部分。它们包装了您的棋盘更新
 * 逻辑, 并处理累加器的增量更新或完全刷新。
 * * ======================================================================
 */

/**
 * @brief 执行一个走法, 并更新 NNUE 累加器
 */
void make_move(BoardState* state, const Move* m) {
    Piece moving_piece = state->board[m->from_y][m->from_x];
    Piece captured_piece = state->board[m->to_y][m->to_x]; // 假设 Move 结构体存储了被吃掉的子
    
    bool king_moved = (moving_piece == r_king || moving_piece == b_king);

    // --- NNUE 增量更新 (Pop) ---
    // 仅在 "将" 没有移动时执行
    if (!king_moved) {
        // 1. 从累加器中移除 "移动的棋子" (在旧位置)
        nnue_pop_piece(state, &state->accumulator, moving_piece, m->from_y, m->from_x);
        
        // 2. 如果有吃子, 从累加器中移除 "被吃的棋子"
        if (captured_piece != EMPTY) {
            nnue_pop_piece(state, &state->accumulator, captured_piece, m->to_y, m->to_x);
        }
    }

    // --- 在此执行您 *旧的* 棋盘更新逻辑 ---
    // (例如: state->board[...], 切换 state->side_to_move)
    //
    // vvvvv 示例逻辑 (请替换为您自己的) vvvvv
    // 移动棋子
    state->board[m->to_y][m->to_x] = state->board[m->from_y][m->from_x];
    state->board[m->from_y][m->from_x] = EMPTY;

    // 交换走棋方
    state->side_to_move = (state->side_to_move == RED) ? BLACK : RED;
    // (您的真实函数还应处理兵的升变、历史记录等)
    // ^^^^^ 示例逻辑 (请替换为您自己的) ^^^^^
    

    // --- NNUE 增量更新 (Push) 或 刷新 ---
    if (king_moved) {
        // "将" 移动了, 累加器失效, 必须 *完全刷新*
        // (这必须在棋盘状态 *更新后* 调用)
        nnue_refresh_accumulator(state, &state->accumulator);
    } else {
        // "将" 未移动, 继续增量更新
        // 3. 将 "移动的棋子" 添加到累加器 (在新位置)
        nnue_push_piece(state, &state->accumulator, moving_piece, m->to_y, m->to_x);
    }
}

/**
 * @brief 撤销一个走法, 并恢复 NNUE 累加器
 */
void unmake_move(BoardState* state, const Move* m) {
    // (注意: 此时 moving_piece 仍在 m->to_y, m->to_x)
    Piece moving_piece = state->board[m->to_y][m->to_x];
    Piece captured_piece = m->captured;
    
    bool king_moved = (moving_piece == r_king || moving_piece == b_king);

    // --- NNUE 增量更新 (Pop) ---
    // (必须在棋盘 *恢复前* 执行)
    if (!king_moved) {
        // 1. 从累加器中移除 "移动的棋子" (在新位置)
        nnue_pop_piece(state, &state->accumulator, moving_piece, m->to_y, m->to_x);
    }
    
    // --- 在此执行您 *旧的* 棋盘撤销逻辑 ---
    // (例如: 恢复 board, 恢复 captured_piece, 切换 side_to_move)
    //
    // vvvvv 示例逻辑 (请替换为您自己的) vvvvv
    state->board[m->from_y][m->from_x] = state->board[m->to_y][m->to_x];
    state->board[m->to_y][m->to_x] = captured_piece; // 把被吃的子放回去
    state->side_to_move = (state->side_to_move == RED) ? BLACK : RED;
    // (您的真实函数还应恢复历史记录等)
    // ^^^^^ 示例逻辑 (请替换为您自己的) ^^^^^

    
    // --- NNUE 增量更新 (Push) 或 刷新 ---
    if (king_moved) {
        // "将" 移动了, 必须 *完全刷新*
        // (这必须在棋盘状态 *恢复后* 调用)
        nnue_refresh_accumulator(state, &state->accumulator);
    } else {
        // "将" 未移动, 继续增量更新
        // 2. 将 "移动的棋子" 添加回累加器 (在旧位置)
        nnue_push_piece(state, &state->accumulator, moving_piece, m->from_y, m->from_x);
        
        // 3. 如果有吃子, 将 "被吃的棋子" 添加回累加器
        if (captured_piece != EMPTY) {
            nnue_push_piece(state, &state->accumulator, captured_piece, m->to_y, m->to_x);
        }
    }
}


/*
 * ======================================================================
 * * 5. 更新 'quiescence_search'
 * * ======================================================================
 */

// 移除了 'color_multiplier'
int quiescence_search(BoardState* state, int alpha, int beta, int ply) {
    if ((double)(clock() - start_time) / CLOCKS_PER_SEC > time_limit) return 0;

    /*
     * 核心变化:
     * - 不再调用 evaluate_board()
     * - 直接从 state->accumulator 获取 NNUE 评估值
     * - nnue_evaluate 已经返回 "当前走棋方" 的分数, 无需 color_multiplier
     */
    int stand_pat = nnue_evaluate(state->side_to_move, &state->accumulator);

    if (stand_pat >= beta) return beta;
    if (stand_pat > alpha) alpha = stand_pat;

    // --- 后续逻辑不变 (只搜索吃子) ---
    Move move_list[MAX_MOVES];
    int num_moves = generate_moves(state, move_list);
    Move capture_moves[MAX_MOVES];
    int capture_count = 0;
    for (int i = 0; i < num_moves; ++i) if (move_list[i].captured != EMPTY) capture_moves[capture_count++] = move_list[i];

    score_moves(state, capture_moves, capture_count, ply);

    for (int i = 0; i < capture_count; ++i) {
        Move current_move = capture_moves[i];
        
        // (调用新的 make_move, 它会自动更新累加器)
        make_move(state, &current_move); 
        if (ply + 1 < MAX_SEARCH_DEPTH) hash_stack[ply + 1] = compute_simple_hash(state);

        // 移除了 color_multiplier
        int score = -quiescence_search(state, -beta, -alpha, ply + 1);
        
        // (调用新的 unmake_move, 它会自动恢复累加器)
        unmake_move(state, &current_move); 
        
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }
    return alpha;
}


/*
 * ======================================================================
 * * 6. 更新 'pvs_search'
 * * ======================================================================
 */

// 移除了 'color_multiplier'
int pvs_search(BoardState* state, int depth, int alpha, int beta, int ply) {
    if ((double)(clock() - start_time) / CLOCKS_PER_SEC > time_limit) return 0;
    
    // 根节点调用静态评估 (qsearch)
    if (depth <= 0) {
        // 移除了 color_multiplier
        return quiescence_search(state, alpha, beta, ply);
    }

    Move move_list[MAX_MOVES];
    int num_moves = generate_moves(state, move_list);
    if (num_moves == 0) {
        if (is_in_check(state, state->side_to_move)) return -MATE_SCORE + ply;
        return 0; // 逼和
    }

    score_moves(state, move_list, num_moves, ply);
    int first_move = 1;
    for (int i = 0; i < num_moves; i++) {
        Move current_move = move_list[i];

        // (调用新的 make_move, 它会自动更新累加器)
        make_move(state, &current_move);
        if (ply < MAX_SEARCH_DEPTH - 1) hash_stack[ply + 1] = compute_simple_hash(state);

        // --- 重复检测 (不变) ---
        int repeated = 0;
        unsigned long long curh = hash_stack[ply + 1];
        for (int k = 0; k <= ply; ++k) if (hash_stack[k] == curh) { repeated = 1; break; }

        int gives_check = 0;
        Side opp = (state->side_to_move == RED) ? BLACK : RED;
        if (is_in_check(state, opp)) gives_check = 1;

        // --- PVS 搜索 (移除了 color_multiplier) ---
        int score;
        if (first_move) {
            score = -pvs_search(state, depth - 1, -beta, -alpha, ply + 1);
            first_move = 0;
        } else {
            score = -pvs_search(state, depth - 1, -alpha - 1, -alpha, ply + 1);
            if (score > alpha && score < beta)
                score = -pvs_search(state, depth - 1, -beta, -alpha, ply + 1);
        }

        // 重复惩罚 (不变)
        if (repeated && current_move.captured == EMPTY && !gives_check) score -= 150;

        // (调用新的 unmake_move, 它会自动恢复累加器)
        unmake_move(state, &current_move);

        if (score > alpha) alpha = score;
        if (alpha >= beta) {
            if (current_move.captured == EMPTY) {
                killer_moves[ply][1] = killer_moves[ply][0];
                killer_moves[ply][0] = current_move;
            }
            return beta;
        }
    }
    return alpha;
}


/*
 * ======================================================================
 * * 7. 更新 'find_best_move'
 * * ======================================================================
 */

Move find_best_move(BoardState* state, int max_depth, double time_limit_sec) {
    start_time = clock();
    time_limit = time_limit_sec;
    Move best_move = {0};
    int best_score = -MATE_SCORE - 1;

    // --- 1. 初始化 NNUE ---
    // (确保 "your_net.nnue" 文件存在)
    if (!nnue_init("your_net.nnue")) {
        fprintf(stderr, "错误: 无法加载 NNUE 网络! 退出。\n");
        return best_move;
    }

    // 移除了 color_multiplier
    
    for (int i = 0; i < MAX_SEARCH_DEPTH; ++i) {
        killer_moves[i][0] = (Move){0};
        killer_moves[i][1] = (Move){0};
        hash_stack[i] = 0ULL;
    }

    hash_stack[0] = compute_simple_hash(state);
    
    // --- 2. 为根局面计算 *初始* 累加器 ---
    nnue_refresh_accumulator(state, &state->accumulator);


    for (int current_depth = 1; current_depth <= max_depth; ++current_depth) {
        Move move_list[MAX_MOVES];
        int num_moves = generate_moves(state, move_list);
        score_moves(state, move_list, num_moves, 0);
        int alpha = -MATE_SCORE - 1;
        int beta = MATE_SCORE + 1;

        for (int i = 0; i < num_moves; ++i) {
            Move current_move = move_list[i];
            
            // (调用新的 make_move, 它会自动更新累加器)
            make_move(state, &current_move);
            hash_stack[1] = compute_simple_hash(state);

            // 移除了 color_multiplier
            int score = -pvs_search(state, current_depth - 1, -beta, -alpha, 1);
            
            // (调用新的 unmake_move, 它会自动恢复累加器)
            unmake_move(state, &current_move);

            if (score > best_score) {
                best_score = score;
                best_move = current_move;
                alpha = score;
            }
        }

        double elapsed_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
        
        // 打印分数 (不再需要 * color_multiplier)
        printf("Info: depth %d, score %d, bestmove %c%d-%c%d, time %.2fs\n",
               current_depth, best_score, // <--- 变化
               'a' + best_move.from_x, best_move.from_y,
               'a' + best_move.to_x, best_move.to_y, elapsed_time);

        if (best_score >= MATE_SCORE - MAX_SEARCH_DEPTH) break;
        if (elapsed_time > time_limit / 2) break;
    }

    // --- 3. 清理 NNUE 内存 ---
    nnue_cleanup();
    
    return best_move;
}