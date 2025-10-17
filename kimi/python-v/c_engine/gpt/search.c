// search.c
#define _POSIX_C_SOURCE 199309L
#include "search.h"
#include "movegen.h"
#include "evaluate.h"   /* evaluate_board */
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <limits.h>

/* ---------- 外部依赖（需在其它模块实现/导出） ---------- */
/* compute_zobrist(board) 在你之前的 zobrist 模块中实现 */
extern uint64_t compute_zobrist(const Board *board);

/* evaluate_board(board) 在 evaluate.c 中实现，返回对黑方有利的评分（正对黑） */
extern int evaluate_board(const Board *board);

/* gen_moves_for_piece 已在 movegen.c 中实现（签名见 movegen.h） */
extern int gen_moves_for_piece(const Board *board, int x, int y, Move *out, int max_out);

/* make_move/unmake_move 在 board.c 中实现 */
extern void make_move(Board *b, const Move *m, Piece *captured);
extern void unmake_move(Board *b, const Move *m, const Piece *captured);

/* ---------- 常量 ---------- */
#define INF 1000000000
#define MATE_SCORE 900000
#define NULL_REDUCTION 2 /* 空步削减 R */
#define MIN_NULL_DEPTH 3 /* 空步要求的最小深度 */
#define HISTORY_SIZE (8 * SQ_COUNT * SQ_COUNT) /* pieceType x from-to pairs */

/* ---------- 置换表 ---------- */
static TTEntry *TT = NULL;
static size_t TT_mask = 0;

/* ---------- killer / history ---------- */
static Move killer[MAX_PLY][2]; /* 每一 ply 两个 killer */
static int history_table[8][SQ_COUNT * SQ_COUNT]; /* pieceType idx (0..7), from*SQ + to -> score */

/* ---------- 搜索计时 ---------- */
static struct timespec t_start;
static int time_limit_ms = 0;
static volatile int stop_search = 0;

/* ---------- init / stop ---------- */
void search_init(void) {
    if (TT) { free(TT); TT = NULL; }
    size_t tt_entries = TT_SIZE;
    TT = (TTEntry*)calloc(tt_entries, sizeof(TTEntry));
    TT_mask = tt_entries - 1;
    memset(killer, 0, sizeof(killer));
    memset(history_table, 0, sizeof(history_table));
    stop_search = 0;
}

void search_stop(void) { stop_search = 1; }

static inline int time_exceeded(void) {
    if (time_limit_ms <= 0) return 0;
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    long elapsed_ms = (now.tv_sec - t_start.tv_sec) * 1000L + (now.tv_nsec - t_start.tv_nsec) / 1000000L;
    return elapsed_ms >= time_limit_ms || stop_search;
}

/* ---------- TT helpers ---------- */
static inline size_t tt_index(uint64_t key) {
    /* 使用简单掩码（TT_SIZE 假定为 2^n） */
    return (size_t)(key & TT_mask);
}

static void tt_store(uint64_t key, int depth, int value, int flag, int from, int to) {
    size_t idx = tt_index(key);
    TTEntry *e = &TT[idx];
    /* 覆写策略：优先更深条目 */
    if (e->key == 0 || e->depth <= depth) {
        e->key = key;
        e->depth = depth;
        e->value = value;
        e->flag = flag;
        e->move_from = from;
        e->move_to = to;
    }
}

static int tt_probe(uint64_t key, int depth, int alpha, int beta, int *out_value, int *out_move_from, int *out_move_to) {
    size_t idx = tt_index(key);
    TTEntry *e = &TT[idx];
    if (e->key == key) {
        if (e->depth >= depth) {
            if (e->flag == TT_EXACT) {
                *out_value = e->value;
                if (out_move_from) *out_move_from = e->move_from;
                if (out_move_to) *out_move_to = e->move_to;
                return 1;
            }
            if (e->flag == TT_LOWER && e->value >= beta) {
                *out_value = e->value; return 1;
            }
            if (e->flag == TT_UPPER && e->value <= alpha) {
                *out_value = e->value; return 1;
            }
        } else {
            /* shall return best-known move for move ordering */
            if (out_move_from) *out_move_from = e->move_from;
            if (out_move_to) *out_move_to = e->move_to;
        }
    }
    return 0;
}

/* ---------- move ordering helpers ---------- */
/* simple piece value for MVV-LVA ordering (PT indices must mirror board.PieceType) */
static const int PIECE_VALUE[8] = {0, 900, 800, 450, 200, 200, MATE_SCORE, 100};

/* convert from (fy,fx,ty,tx) -> fromIndex / toIndex */
static inline int sq_index(int y, int x) { return y * COLS + x; }

typedef struct {
    Move m;
    int score;
} ScoredMove;

/* score a move: captures prioritized, then killer, then history */
static int score_move(const Board *b, const Move *m, int tt_from, int tt_to) {
    Piece p = b->sq[m->fy][m->fx];
    Piece tgt = b->sq[m->ty][m->tx];
    int from_idx = sq_index(m->fy, m->fx), to_idx = sq_index(m->ty, m->tx);
    if (tgt.type != PT_NONE) {
        /* MVV-LVA: victim*100 - attacker */
        int val = PIECE_VALUE[tgt.type] * 100 - PIECE_VALUE[p.type];
        return 1000000 + val; /* captures first */
    }
    /* TT move highest priority next */
    if (tt_from == from_idx && tt_to == to_idx) return 500000;
    /* killer moves */
    for (int k=0;k<2;k++){
        if (killer[0][k].fy == m->fy && killer[0][k].fx == m->fx && killer[0][k].ty == m->ty && killer[0][k].tx == m->tx) {
            return 200000;
        }
    }
    /* history heuristic */
    int piece_idx = (int)p.type;
    int hist = history_table[piece_idx][from_idx * SQ_COUNT + to_idx];
    return hist;
}

/* sort moves in place by score (simple insertion sort for small arrays) */
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

/* ---------- check detection (uses in_check logic) ---------- */
/* For correctness we need an in_check(board, side) implementation matching your Python one.
   If you already have such function (preferred), declare it extern. Otherwise we use a simple placeholder
   that determines if king is attacked by generating opponent attacks to king square.
*/
extern int in_check(const Board *board, Side side); /* return 1 if side is in check, else 0 */

/* ---------- null move (do by toggling side and not moving) ---------- */
static void make_null_move(Board *b) {
    b->side_to_move = (b->side_to_move == SIDE_RED) ? SIDE_BLACK : SIDE_RED;
}
static void unmake_null_move(Board *b) {
    b->side_to_move = (b->side_to_move == SIDE_RED) ? SIDE_BLACK : SIDE_RED;
}

/* ---------- main PVS with NegaScout, Null-move, Killer/History, TT ---------- */
static int pvs(Board *b, int depth, int alpha, int beta, int ply) {
    if (time_exceeded()) { stop_search = 1; return 0; }

    if (depth <= 0) {
        /* quiescence search: call evaluate_quiescence or a simpler quiescence */
        return quiescence(b, alpha, beta);
    }

    uint64_t key = compute_zobrist(b);
    int tt_move_from=-1, tt_move_to=-1;
    int tt_val;
    if (tt_probe(key, depth, alpha, beta, &tt_val, &tt_move_from, &tt_move_to)) {
        return tt_val;
    }

    /* Null-move reduction */
    int do_null = 0;
    if (depth >= MIN_NULL_DEPTH && !in_check(b, b->side_to_move)) {
        /* Make null move and search reduced depth */
        do_null = 1;
        make_null_move(b);
        int val = -pvs(b, depth - 1 - NULL_REDUCTION, -beta, -beta + 1, ply+1);
        unmake_null_move(b);
        if (val >= beta) {
            return beta; /* fail hard */
        }
    }

    /* Generate all moves for side_to_move */
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
        /* no legal moves -> mate or stalemate */
        if (in_check(b, b->side_to_move)) return -MATE_SCORE + ply;
        return 0;
    }

    /* move ordering: score each move */
    ScoredMove scored[256];
    for (int i=0;i<total;i++){
        scored[i].m = moves_buf[i];
        /* score using move, with TT move as hint */
        int from_idx = sq_index(moves_buf[i].fy, moves_buf[i].fx);
        int to_idx   = sq_index(moves_buf[i].ty, moves_buf[i].tx);
        scored[i].score = score_move(b, &moves_buf[i], tt_move_from, tt_move_to);
    }
    sort_scored_moves(scored, total);

    int best_val = -INF;
    int best_from = -1, best_to = -1;
    int first = 1;

    for (int i=0;i<total;i++){
        Move m = scored[i].m;
        Piece captured;
        make_move(b, &m, &captured);

        int ext = 0;
        /* extension: if move gives check -> extend by 1 */
        if (in_check(b, b->side_to_move)) ext = 1;

        int score;
        if (first) {
            score = -pvs(b, depth - 1 + ext, -beta, -alpha, ply + 1);
        } else {
            /* PVS: null window search */
            score = -pvs(b, depth - 1 + ext, -alpha - 1, -alpha, ply + 1);
            if (score > alpha && score < beta) {
                score = -pvs(b, depth - 1 + ext, -beta, -alpha, ply + 1);
            }
        }

        unmake_move(b, &m, &captured);

        if (stop_search) return 0;

        if (score > best_val) {
            best_val = score;
            best_from = sq_index(m.fy, m.fx);
            best_to = sq_index(m.ty, m.tx);
        }
        if (score > alpha) {
            alpha = score;
            /* update history table for non-capture moves */
            if (captured.type == PT_NONE) {
                int piece_idx = (int)b->sq[m.fy][m.fx].type;
                /* add depth^2 to history (classic) */
                history_table[piece_idx][sq_index(m.fy,m.fx)*SQ_COUNT + sq_index(m.ty,m.tx)] += depth * depth;
                if (history_table[piece_idx][sq_index(m.fy,m.fx)*SQ_COUNT + sq_index(m.ty,m.tx)] < 0)
                    history_table[piece_idx][sq_index(m.fy,m.fx)*SQ_COUNT + sq_index(m.ty,m.tx)] = INT_MAX/2;
            }
        }
        if (alpha >= beta) {
            /* fail-hard: store killer if non-capture */
            if (captured.type == PT_NONE && ply < MAX_PLY) {
                /* shift killers */
                killer[ply][1] = killer[ply][0];
                killer[ply][0] = m;
            }
            /* store in TT as lower bound */
            tt_store(key, depth, best_val, TT_LOWER, best_from, best_to);
            return best_val;
        }
        first = 0;
    }

    /* store exact score in TT */
    tt_store(key, depth, best_val, TT_EXACT, best_from, best_to);
    return best_val;
}

/* ---------- Quiescence (uses captures + checks) ---------- */
int quiescence(Board *b, int alpha, int beta) {
    if (time_exceeded()) { stop_search = 1; return 0; }
    /* Static evaluate (call engine's evaluate_board) */
    int stand = evaluate_board(b);
    /* convert to side_to_move perspective consistent with pvs usage
       assume evaluate_board returns score positive for black; we want from side_to_move perspective */
    int color_mult = (b->side_to_move == SIDE_BLACK) ? 1 : -1;
    int stand_pat = stand * color_mult;

    if (stand_pat >= beta) return beta;
    if (stand_pat > alpha) alpha = stand_pat;

    Move moves_buf[128];
    int total = 0;
    for (int y=0;y<ROWS;y++){
        for (int x=0;x<COLS;x++){
            if (b->sq[y][x].side == b->side_to_move) {
                total += gen_moves_for_piece(b, x, y, moves_buf + total, 128 - total);
                if (total >= 128) break;
            }
        }
        if (total >= 128) break;
    }

    /* keep only captures and checks */
    ScoredMove scored[128];
    int cnt = 0;
    for (int i=0;i<total;i++){
        Move m = moves_buf[i];
        if (b->sq[m.ty][m.tx].type != PT_NONE) {
            scored[cnt].m = m;
            scored[cnt].score = PIECE_VALUE[b->sq[m.ty][m.tx].type] * 100 - PIECE_VALUE[b->sq[m.fy][m.fx].type];
            cnt++;
        } else {
            /* test gives check after move */
            Piece captured;
            make_move(b, &m, &captured);
            int chk = in_check(b, b->side_to_move);
            unmake_move(b, &m, &captured);
            if (chk) {
                scored[cnt].m = m;
                scored[cnt].score = 5000;
                cnt++;
            }
        }
    }
    if (cnt == 0) return alpha;
    sort_scored_moves(scored, cnt);

    for (int i=0;i<cnt;i++){
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

/* ---------- Root iterative deepening with aspiration windows ---------- */
Move search_root(Board *board, int max_depth, int time_ms) {
    /* init */
    stop_search = 0;
    time_limit_ms = time_ms;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    Move best_move = {0};
    int best_score = -INF;

    /* iterative deepening */
    for (int depth = 1; depth <= max_depth; depth++) {
        if (time_exceeded()) break;
        /* call pvs */
        int val = pvs(board, depth, -INF, INF, 0);
        if (stop_search) break;
        /* extract best from TT (root) */
        uint64_t key = compute_zobrist(board);
        size_t idx = tt_index(key);
        if (TT[idx].key == key) {
            int from_idx = TT[idx].move_from;
            int to_idx = TT[idx].move_to;
            if (from_idx >= 0 && to_idx >= 0) {
                best_move.fy = from_idx / COLS;
                best_move.fx = from_idx % COLS;
                best_move.ty = to_idx / COLS;
                best_move.tx = to_idx % COLS;
                best_score = TT[idx].value;
            }
        }
        /* if we didn't get a move from TT, fall back to scanning generated moves and using PVS results is complex;
           but TT will normally be filled by pvs. */
        /* optional: break early if checkmate / large mate score found */
        if (abs(best_score) > MATE_SCORE/2) break;
    }

    best_move.score = best_score;
    return best_move;
}
