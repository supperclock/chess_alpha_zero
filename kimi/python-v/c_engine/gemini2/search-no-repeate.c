#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "chess.h"

#define MAX_SEARCH_DEPTH 64
static Move killer_moves[MAX_SEARCH_DEPTH][2];
static unsigned long long hash_stack[MAX_SEARCH_DEPTH];

const int PIECE_VALUES[] = {
    0, MATE_SCORE, 200, 200, 450, 900, 800, 100,
    MATE_SCORE, 200, 200, 450, 900, 800, 100
};

static clock_t start_time;
static double time_limit;

int pvs_search(BoardState* state, int depth, int alpha, int beta, int color_multiplier, int ply);
int quiescence_search(BoardState* state, int alpha, int beta, int color_multiplier, int ply);

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

int quiescence_search(BoardState* state, int alpha, int beta, int color_multiplier, int ply) {
    if ((double)(clock() - start_time) / CLOCKS_PER_SEC > time_limit) return 0;
    int stand_pat = evaluate_board(state) * color_multiplier;
    if (stand_pat >= beta) return beta;
    if (stand_pat > alpha) alpha = stand_pat;

    Move move_list[MAX_MOVES];
    int num_moves = generate_moves(state, move_list);
    Move capture_moves[MAX_MOVES];
    int capture_count = 0;
    for (int i = 0; i < num_moves; ++i) if (move_list[i].captured != EMPTY) capture_moves[capture_count++] = move_list[i];

    score_moves(state, capture_moves, capture_count, ply);

    for (int i = 0; i < capture_count; ++i) {
        Move current_move = capture_moves[i];
        make_move(state, &current_move);
        if (ply + 1 < MAX_SEARCH_DEPTH) hash_stack[ply + 1] = compute_simple_hash(state);
        int score = -quiescence_search(state, -beta, -alpha, -color_multiplier, ply + 1);
        unmake_move(state, &current_move);
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }
    return alpha;
}

int pvs_search(BoardState* state, int depth, int alpha, int beta, int color_multiplier, int ply) {
    if ((double)(clock() - start_time) / CLOCKS_PER_SEC > time_limit) return 0;
    if (depth <= 0) return quiescence_search(state, alpha, beta, color_multiplier, ply);

    Move move_list[MAX_MOVES];
    int num_moves = generate_moves(state, move_list);
    if (num_moves == 0) {
        if (is_in_check(state, state->side_to_move)) return -MATE_SCORE + ply;
        return 0;
    }

    score_moves(state, move_list, num_moves, ply);
    int first_move = 1;
    for (int i = 0; i < num_moves; i++) {
        Move current_move = move_list[i];
        make_move(state, &current_move);
        if (ply < MAX_SEARCH_DEPTH - 1) hash_stack[ply + 1] = compute_simple_hash(state);

        int repeated = 0;
        unsigned long long curh = hash_stack[ply + 1];
        for (int k = 0; k <= ply; ++k) if (hash_stack[k] == curh) { repeated = 1; break; }

        int gives_check = 0;
        Side opp = (state->side_to_move == RED) ? BLACK : RED;
        if (is_in_check(state, opp)) gives_check = 1;

        int score;
        if (first_move) {
            score = -pvs_search(state, depth - 1, -beta, -alpha, -color_multiplier, ply + 1);
            first_move = 0;
        } else {
            score = -pvs_search(state, depth - 1, -alpha - 1, -alpha, -color_multiplier, ply + 1);
            if (score > alpha && score < beta)
                score = -pvs_search(state, depth - 1, -beta, -alpha, -color_multiplier, ply + 1);
        }

        // Soft penalty for repeating non-capture, non-check moves
        if (repeated && current_move.captured == EMPTY && !gives_check) score -= 150;

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

Move find_best_move(BoardState* state, int max_depth, double time_limit_sec) {
    start_time = clock();
    time_limit = time_limit_sec;
    Move best_move = {0};
    int best_score = -MATE_SCORE - 1;
    int color_multiplier = (state->side_to_move == BLACK) ? 1 : -1;

    for (int i = 0; i < MAX_SEARCH_DEPTH; ++i) {
        killer_moves[i][0] = (Move){0};
        killer_moves[i][1] = (Move){0};
        hash_stack[i] = 0ULL;
    }

    hash_stack[0] = compute_simple_hash(state);

    for (int current_depth = 1; current_depth <= max_depth; ++current_depth) {
        Move move_list[MAX_MOVES];
        int num_moves = generate_moves(state, move_list);
        score_moves(state, move_list, num_moves, 0);
        int alpha = -MATE_SCORE - 1;
        int beta = MATE_SCORE + 1;

        for (int i = 0; i < num_moves; ++i) {
            Move current_move = move_list[i];
            make_move(state, &current_move);
            hash_stack[1] = compute_simple_hash(state);
            int score = -pvs_search(state, current_depth - 1, -beta, -alpha, -color_multiplier, 1);
            unmake_move(state, &current_move);

            if (score > best_score) {
                best_score = score;
                best_move = current_move;
                alpha = score;
            }
        }

        double elapsed_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
        printf("Info: depth %d, score %d, bestmove %c%d-%c%d, time %.2fs\n",
               current_depth, best_score * color_multiplier,
               'a' + best_move.from_x, best_move.from_y,
               'a' + best_move.to_x, best_move.to_y, elapsed_time);

        if (best_score >= MATE_SCORE - MAX_SEARCH_DEPTH) break;
        if (elapsed_time > time_limit / 2) break;
    }
    return best_move;
}
