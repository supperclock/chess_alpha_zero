// search.c
#define _POSIX_C_SOURCE 199309L
#include "search.h"
#include "movegen.h"
#include "evaluate.h"
#include "board.h"
#include "rules.h"    /* for in_check() */
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#ifdef _WIN32
#include <windows.h>
static LARGE_INTEGER t_start, t_freq;
#else
#include <time.h>
static struct timespec t_start;
#endif

/* ---------- Config ---------- */
#define TT_ENTRIES (1<<20)   /* 1M entries, adjust if needed */
#define MAX_PLY 64
#define SQ_COUNT (ROWS*COLS)
#define INF 1000000000
#define MATE_SCORE 900000
#define NULL_REDUCTION 2
#define MIN_NULL_DEPTH 3

static TTEntry *TT = NULL;
static size_t TT_mask = 0;

/* ---------- killer & history ---------- */
static Move killer[MAX_PLY][2];
static int *history_table = NULL; /* allocated as [pieceType][from*SQ_COUNT+to] flattened */

/* ---------- timing & control ---------- */
static int time_limit_ms = 0;
static volatile int stop_search = 0;

#ifdef _WIN32
static void start_timer(int ms) {
    QueryPerformanceFrequency(&t_freq);
    QueryPerformanceCounter(&t_start);
    time_limit_ms = ms;
    stop_search = 0;
}
static inline int time_exceeded(void) {
    if (time_limit_ms <= 0) return 0;
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    double elapsed = (double)(now.QuadPart - t_start.QuadPart) * 1000.0 / (double)t_freq.QuadPart;
    return (elapsed >= time_limit_ms) || stop_search;
}
#else
static void start_timer(int ms) {
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    time_limit_ms = ms;
    stop_search = 0;
}
static inline int time_exceeded(void) {
    if (time_limit_ms <= 0) return 0;
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    long elapsed_ms = (now.tv_sec - t_start.tv_sec) * 1000L + (now.tv_nsec - t_start.tv_nsec) / 1000000L;
    return elapsed_ms >= time_limit_ms || stop_search;
}
#endif

void search_stop(void) { stop_search = 1; }

/* ---------- helpers ---------- */
static inline size_t tt_index(uint64_t key) { return (size_t)(key & TT_mask); }

void search_init(void) {
    if (TT) { free(TT); TT = NULL; }
    TT = (TTEntry*)calloc(TT_ENTRIES, sizeof(TTEntry));
    TT_mask = TT_ENTRIES - 1;
    memset(killer, 0, sizeof(killer));
    if (history_table) free(history_table);
    /* piece types up to 7 -> allocate 8 * SQ_COUNT * SQ_COUNT ints */
    history_table = (int*)calloc(8 * SQ_COUNT * SQ_COUNT, sizeof(int));
    stop_search = 0;
}

/* store and probe TT */
static void tt_store(uint64_t key, int depth, int value, int flag, int from, int to) {
    size_t idx = tt_index(key);
    TTEntry *e = &TT[idx];
    if (e->key == 0 || e->depth <= depth) {
        e->key = key; e->depth = depth; e->value = value; e->flag = flag;
        e->move_from = from; e->move_to = to;
    }
}

static int tt_probe(uint64_t key, int depth, int alpha, int beta, int *out_value, int *out_move_from, int *out_move_to) {
    size_t idx = tt_index(key);
    TTEntry *e = &TT[idx];
    if (e->key == key) {
        if (e->depth >= depth) {
            if (e->flag == TT_EXACT) { *out_value = e->value; if (out_move_from) *out_move_from = e->move_from; if (out_move_to) *out_move_to = e->move_to; return 1; }
            if (e->flag == TT_LOWER && e->value >= beta) { *out_value = e->value; return 1; }
            if (e->flag == TT_UPPER && e->value <= alpha) { *out_value = e->value; return 1; }
        } else {
            if (out_move_from) *out_move_from = e->move_from;
            if (out_move_to) *out_move_to = e->move_to;
        }
    }
    return 0;
}

static int static_exchange_eval(const Board *b, int x, int y, Side side_to_move) {
    /* 简化版递归SEE：计算该格位置所有潜在攻击者，模拟吃子交换 */
    int attacker_count = 0;
    int attack_side[32];
    int attack_value[32];

    /* 找出所有攻击此格的棋子 */
    for (int yy=0; yy<ROWS; yy++) {
        for (int xx=0; xx<COLS; xx++) {
            Piece p = b->sq[yy][xx];
            if (p.type == PT_NONE) continue;
            if (square_attacked(b, x, y, p.side)) {
                attack_side[attacker_count] = p.side;
                attack_value[attacker_count] = PIECE_VALUES_STD[p.type];
                attacker_count++;
            }
        }
    }

    if (attacker_count == 0) return 0;

    int gain[32];
    int depth = 0;
    int cur_side = side_to_move;
    int captured_val = PIECE_VALUES_STD[b->sq[y][x].type];
    if (captured_val == 0) captured_val = 0;
    gain[0] = captured_val;

    int last_gain = captured_val;

    /* 模拟交替吃子 */
    while (1) {
        int min_attacker = INT_MAX;
        int idx = -1;
        for (int i=0; i<attacker_count; i++) {
            if (attack_side[i] == cur_side && attack_value[i] < min_attacker) {
                min_attacker = attack_value[i];
                idx = i;
            }
        }
        if (idx == -1) break;

        last_gain = PIECE_VALUES_STD[b->sq[y][x].type] - min_attacker;
        cur_side = (cur_side == SIDE_RED) ? SIDE_BLACK : SIDE_RED;
        gain[++depth] = -gain[depth-1] + last_gain;
    }

    /* minimax合成 */
    for (int i = depth - 1; i >= 0; i--) {
        if (-gain[i+1] < gain[i]) gain[i] = -gain[i+1];
    }

    return gain[0];
}


/* ---------- move scoring & ordering ---------- */
static const int PIECE_VALUE_ORDER[8] = {0,900,400,450,250,250,10000,100};

typedef struct { Move m; int score; } ScoredMove;

static inline int sq_index(int y, int x) { return y * COLS + x; }

/* returns score used for ordering */
static int score_move_for_ordering(const Board *b, const Move *m, int tt_from, int tt_to, int is_in_check_flag) {
    Piece src = b->sq[m->fy][m->fx];
    Piece tgt = b->sq[m->ty][m->tx];
    int from_idx = sq_index(m->fy, m->fx);
    int to_idx   = sq_index(m->ty, m->tx);

    if (tgt.type != PT_NONE) {
        /* captures: use simple_exchange_value to judge */
        int see = static_exchange_eval(b, m->tx, m->ty, b->side_to_move);
        /* boost captures but consider risk */
        return 1000000 + see * 100;
    }

    /* TT move */
    if (tt_from == from_idx && tt_to == to_idx) return 500000;

    /* killer */
    for (int k=0;k<2;k++){
        if (killer[0][k].fy == m->fy && killer[0][k].fx == m->fx && killer[0][k].ty == m->ty && killer[0][k].tx == m->tx) {
            return 200000;
        }
    }

    /* if currently in check, prefer moves that resolve check (handled externally by boosting) */
    int piece_idx = (int)src.type;
    int hist = history_table[piece_idx * (SQ_COUNT * SQ_COUNT) + from_idx * SQ_COUNT + to_idx];
    return hist;
}

/* insertion sort for small arrays */
static void sort_scored_moves(ScoredMove *arr, int n) {
    for (int i=1;i<n;i++){
        ScoredMove key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j].score < key.score) {
            arr[j+1] = arr[j];
            j--;
        }
        arr[j+1] = key;
    }
}

/* ---------- Quiescence (captures + checks) ---------- */
int quiescence(Board *b, int alpha, int beta) {
    if (time_exceeded()) { stop_search = 1; return 0; }
    int stand = evaluate_board(b);
    /* convert boardside perspective: evaluate_board returns positive for RED advantage in our evaluate.c
       We want perspective of side_to_move: if side_to_move == RED, return stand, else -stand */
    int color_mult = (b->side_to_move == SIDE_RED) ? 1 : -1;
    int stand_pat = stand * color_mult;

    if (stand_pat >= beta) return beta;
    if (stand_pat > alpha) alpha = stand_pat;

    /* generate captures+checks */
    Move buf[128];
    int total = 0;
    for (int y=0;y<ROWS;y++)
        for (int x=0;x<COLS;x++)
            if (b->sq[y][x].side == b->side_to_move)
                total += gen_moves_for_piece(b, x, y, buf + total, 128 - total);

    ScoredMove scored[128];
    int cnt = 0;
    for (int i=0;i<total;i++){
        Move m = buf[i];
        Piece tgt = b->sq[m.ty][m.tx];
        if (tgt.type != PT_NONE) {
            scored[cnt].m = m;
            scored[cnt].score = PIECE_VALUE_ORDER[tgt.type] * 100 - PIECE_VALUE_ORDER[b->sq[m.fy][m.fx].type];
            cnt++;
        } else {
            /* check giving moves */
            Piece cap;
            make_move(b, &m, &cap);
            int chk = in_check(b, b->side_to_move);
            unmake_move(b, &m, &cap);
            if (chk) {
                scored[cnt].m = m;
                scored[cnt].score = 5000;
                cnt++;
            }
        }
    }
    if (cnt == 0) return alpha;
    sort_scored_moves(scored, cnt);

    for (int i=0;i<cnt;i++) {
        Move m = scored[i].m;
        Piece captured;
        make_move(b, &m, &captured);
        int score = -quiescence(b, -beta, -alpha);
        unmake_move(b, &m, &captured);
        if (stop_search) return 0;
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }
    return alpha;
}
/* ---------- null move (do by toggling side and not moving) ---------- */
static void make_null_move(Board *b) {
    b->side_to_move = (b->side_to_move == SIDE_RED) ? SIDE_BLACK : SIDE_RED;
}
static void unmake_null_move(Board *b) {
    b->side_to_move = (b->side_to_move == SIDE_RED) ? SIDE_BLACK : SIDE_RED;
}
/* ---------- PVS / NegaScout with null-move, killer/history, TT ---------- */
static int pvs(Board *b, int depth, int alpha, int beta, int ply) {
    if (time_exceeded()) { stop_search = 1; return 0; }
    if (depth <= 0) return quiescence(b, alpha, beta);

    uint64_t key = compute_zobrist(b);
    int tt_from=-1, tt_to=-1, tt_val;
    if (tt_probe(key, depth, alpha, beta, &tt_val, &tt_from, &tt_to)) return tt_val;

    int in_check_flag = in_check(b, b->side_to_move);

    /* Null move reduction */
    if (!in_check_flag && depth >= MIN_NULL_DEPTH) {
        make_null_move(b);
        int val = -pvs(b, depth - 1 - NULL_REDUCTION, -beta, -beta + 1, ply+1);
        unmake_null_move(b);
        if (val >= beta) return beta;
    }

    /* generate moves */
    Move moves_buf[256];
    int total = 0;
    for (int y=0;y<ROWS;y++){
        for (int x=0;x<COLS;x++){
            if (b->sq[y][x].side == b->side_to_move) {
                total += gen_moves_for_piece(b, x, y, moves_buf + total, 256 - total);
                if (total >= 256) break;
            }
        }
        if (total >= 256) break;
    }

    if (total == 0) {
        if (in_check_flag) return -MATE_SCORE + ply;
        return 0;
    }

    /* Build scored moves with SEE and ordering */
    ScoredMove scored[256];
    int cnt = 0;
    for (int i=0;i<total;i++) {
        scored[cnt].m = moves_buf[i];
        scored[cnt].score = score_move_for_ordering(b, &moves_buf[i], tt_from, tt_to, in_check_flag);
        cnt++;
    }

    /* If in check, filter to only moves that resolve check */
    if (in_check_flag) {
        int write = 0;
        for (int i=0;i<cnt;i++){
            Move m = scored[i].m;
            Piece cap;
            make_move(b, &m, &cap);
            int still_in_check = in_check(b, b->side_to_move);
            unmake_move(b, &m, &cap);
            if (!still_in_check) {
                /* boost these moves strongly to search first */
                scored[write] = scored[i];
                scored[write].score += 1000000;
                write++;
            }
        }
        if (write == 0) {
            /* no escape moves: at root, caller will interpret; internally treat as mate */
            return -MATE_SCORE + ply;
        }
        cnt = write;
    }

    /* sort */
    sort_scored_moves(scored, cnt);

    int best_val = -INF;
    int best_from = -1, best_to = -1;
    int first = 1;

    for (int i=0;i<cnt;i++){
        Move m = scored[i].m;

        /* Skip obviously losing captures early (SEE) */
        Piece tgt = b->sq[m.ty][m.tx];
        if (tgt.type != PT_NONE) {
            int see = static_exchange_eval(b, m.tx, m.ty, b->side_to_move);
            if (see < -100) continue; /* threshold: avoid big immediate losses */
        }

        Piece captured;
        make_move(b, &m, &captured);

        int ext = 0;
        if (in_check(b, b->side_to_move)) ext = 1; /* extension if gives check to opponent after move */

        int score;
        if (first) {
            score = -pvs(b, depth - 1 + ext, -beta, -alpha, ply + 1);
        } else {
            score = -pvs(b, depth - 1 + ext, -alpha - 1, -alpha, ply + 1);
            if (score > alpha && score < beta) {
                score = -pvs(b, depth - 1 + ext, -beta, -alpha, ply + 1);
            }
        }

        unmake_move(b, &m, &captured);

        if (stop_search) return 0;

        if (score > best_val) { best_val = score; best_from = sq_index(m.fy,m.fx); best_to = sq_index(m.ty,m.tx); }
        if (score > alpha) {
            alpha = score;
            /* history update for non-capture */
            if (captured.type == PT_NONE) {
                int piece_idx = (int)b->sq[m.fy][m.fx].type;
                int idx = piece_idx * (SQ_COUNT * SQ_COUNT) + sq_index(m.fy,m.fx) * SQ_COUNT + sq_index(m.ty,m.tx);
                history_table[idx] += depth * depth;
                if (history_table[idx] < 0) history_table[idx] = INT_MAX/2;
            }
        }
        if (alpha >= beta) {
            /* fail-hard: record killer */
            if (captured.type == PT_NONE && ply < MAX_PLY) {
                killer[ply][1] = killer[ply][0];
                killer[ply][0] = m;
            }
            tt_store(key, depth, best_val, TT_LOWER, best_from, best_to);
            return best_val;
        }
        first = 0;
    }

    tt_store(key, depth, best_val, TT_EXACT, best_from, best_to);
    return best_val;
}

/* ---------- root search with iterative deepening ----------
   Special behavior: if side to move is in_check and there are no escape moves
   we return a Move with fy = -1 to indicate "no rescue move" to caller.
*/
Move search_root(Board *board, int max_depth, int time_ms) {
    stop_search = 0;
    start_timer(time_ms);

    Move best_move; memset(&best_move, 0, sizeof(best_move));
    int best_score = -INF;

    /* generate root moves once */
    Move root_moves[512];
    int total = 0;
    for (int y=0;y<ROWS;y++)
        for (int x=0;x<COLS;x++)
            if (board->sq[y][x].side == board->side_to_move)
                total += gen_moves_for_piece(board, x, y, root_moves + total, 512 - total);

    if (total == 0) {
        /* no moves at all */
        best_move.fy = -1;
        return best_move;
    }

    int in_check_flag = in_check(board, board->side_to_move);

    /* If in check, filter root moves to only those that resolve check.
       If none, return fy=-1 (signal "no rescue move"). */
    Move filtered[512]; int fcnt = 0;
    if (in_check_flag) {
        for (int i=0;i<total;i++){
            Move m = root_moves[i];
            Piece cap;
            make_move(board, &m, &cap);
            int still = in_check(board, board->side_to_move);
            unmake_move(board, &m, &cap);
            if (!still) filtered[fcnt++] = m;
        }
        if (fcnt == 0) {
            best_move.fy = -1; /* no rescue move */
            return best_move;
        }
    } else {
        for (int i=0;i<total;i++) filtered[fcnt++] = root_moves[i];
    }

    /* iterative deepening */
    for (int depth = 1; depth <= max_depth; depth++) {
        if (time_exceeded()) break;

        /* order moves using SEE / history / tt hints */
        ScoredMove scored[512];
        for (int i=0;i<fcnt;i++){
            scored[i].m = filtered[i];
            scored[i].score = score_move_for_ordering(board, &filtered[i], -1, -1, in_check_flag);
            /* boost moves that resolve check if in_check */
            if (in_check_flag) {
                Piece cap;
                make_move(board, &filtered[i], &cap);
                int still = in_check(board, board->side_to_move);
                unmake_move(board, &filtered[i], &cap);
                if (!still) scored[i].score += 1000000;
            }
        }
        sort_scored_moves(scored, fcnt);

        int local_best_score = -INF;
        Move local_best_move = filtered[0];

        for (int i=0;i<fcnt;i++){
            if (time_exceeded()) break;
            Move m = scored[i].m;
            Piece cap;
            make_move(board, &m, &cap);
            int val = -pvs(board, depth-1, -INF, INF, 1);
            unmake_move(board, &m, &cap);
            if (stop_search) break;
            if (val > local_best_score) { local_best_score = val; local_best_move = m; }
        }

        if (!stop_search) {
            best_move = local_best_move;
            best_score = local_best_score;
        }
    }

    best_move.score = best_score;
    return best_move;
}
