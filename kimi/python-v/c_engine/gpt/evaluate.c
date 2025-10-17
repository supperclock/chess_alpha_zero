// evaluate.c
#include "evaluate.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* ---------- 常量与数据结构 ---------- */

#define MAX_ATTACKERS_PER_SQ 32

/* piece value table (对应 Python PIECE_VALUES)
   这里主观数值应与 Python 保持一致或近似 */
static const int PIECE_VALUE_MAP[] = {
    0,    /* PT_NONE */
    900,  /* PT_ROOK */
    800,  /* PT_CANNON */
    450,  /* PT_HORSE */
    200,  /* PT_ELEPHANT */
    200,  /* PT_ADVISOR */
    1000000, /* PT_GENERAL (mate score) */
    100   /* PT_PAWN */
};

/* PST: piece-square-table
   维度： pieceType (use index PT_*), side (0=red,1=black), ROWS, COLS
   使用静态数组并在 eval_init 初始化 */
static int PST[8][2][ROWS][COLS]; /* 8 because PT_NONE..PT_PAWN index up to 7 */

/* attack info entry */
typedef struct {
    int value; /* attacker piece value */
    Side side; /* SIDE_RED / SIDE_BLACK */
} AttackEntry;

/* per-square attack lists */
static AttackEntry attack_lists[ROWS][COLS][MAX_ATTACKERS_PER_SQ];
static int attack_counts[ROWS][COLS];

/* 小工具 */
static inline int piece_value(PieceType pt) {
    if (pt >= PT_NONE && pt <= PT_PAWN) return PIECE_VALUE_MAP[pt];
    return 0;
}

/* find general on board */
KingPos find_general_pos(const Board *board, Side side) {
    KingPos kp = {-1,-1,0};
    for (int y=0;y<ROWS;y++){
        for (int x=0;x<COLS;x++){
            Piece p = board->sq[y][x];
            if (p.type == PT_GENERAL && p.side == side) {
                kp.x = x; kp.y = y; kp.found = 1;
                return kp;
            }
        }
    }
    kp.found = 0;
    return kp;
}

/* count pieces of certain types for a side */
int count_pieces(const Board *board, Side side, PieceType types[], int ntypes) {
    int cnt = 0;
    for (int y=0;y<ROWS;y++){
        for (int x=0;x<COLS;x++){
            Piece p = board->sq[y][x];
            if (p.side != side) continue;
            for (int i=0;i<ntypes;i++){
                if (p.type == types[i]) { cnt++; break; }
            }
        }
    }
    return cnt;
}

/* ---------- PST & 初始化 ---------- */
void eval_init(void) {
    /* 清零 */
    memset(PST, 0, sizeof(PST));

    /* 简单还原 Python 版 PST 逻辑：对兵、卒、车、炮做少量位置偏好 */
    for (int y=0;y<ROWS;y++){
        for (int x=0;x<COLS;x++){
            /* red pawn (兵) */
            PST[PT_PAWN][0][y][x] = (y - 3) * 5;
            if (3 <= x && x <= 5) PST[PT_PAWN][0][y][x] += 5;
            /* black pawn (卒) */
            PST[PT_PAWN][1][y][x] = (6 - y) * 5;
            if (3 <= x && x <= 5) PST[PT_PAWN][1][y][x] += 5;
            /* rooks slightly prefer center files */
            int val = (3 <= x && x <= 5) ? 15 : 0;
            PST[PT_ROOK][0][y][x] += val;
            PST[PT_ROOK][1][y][x] += val;
        }
    }
    /* cannons: give row bonus similar to Python */
    for (int x=0;x<COLS;x++) {
        PST[PT_CANNON][1][7][x] += 10; /* black cannon */
        PST[PT_CANNON][0][2][x] += 10; /* red cannon */
    }

    /* clear attack lists */
    memset(attack_counts, 0, sizeof(attack_counts));
}

/* ---------- Attack map 构建 ----------
   需要 move generation 接口：gen_moves_for_piece
   对每个棋子调用 gen_moves_for_piece，把它能攻击到的格子记录在 attack_lists 中。
   注：gen_moves_for_piece 应返回 pseudo moves（包括吃子与不吃子）
*/
static void build_attack_map(const Board *board) {
    /* reset */
    for (int y=0;y<ROWS;y++) for (int x=0;x<COLS;x++) attack_counts[y][x]=0;

    Move temp_moves[128];
    for (int y=0;y<ROWS;y++){
        for (int x=0;x<COLS;x++){
            Piece p = board->sq[y][x];
            if (p.type == PT_NONE) continue;
            /* 调用外部 movegen（需要实现） */
            int n = gen_moves_for_piece(board, x, y, temp_moves, 128);
            for (int i=0;i<n;i++){
                Move *m = &temp_moves[i];
                int ty = m->ty, tx = m->tx;
                if (ty<0 || ty>=ROWS || tx<0 || tx>=COLS) continue;
                int idx = attack_counts[ty][tx];
                if (idx >= MAX_ATTACKERS_PER_SQ) continue; /* overflow safety */
                attack_lists[ty][tx][idx].value = piece_value(p.type);
                attack_lists[ty][tx][idx].side  = p.side;
                attack_counts[ty][tx]++;
            }
        }
    }
}

/* ---------- king_safety_score helper ---------- */
static int king_safety_score(const Board *board, Side side) {
    KingPos k = find_general_pos(board, side);
    if (!k.found) {
        return (side == SIDE_BLACK) ? -2000 : 2000;
    }
    int kx = k.x, ky = k.y;
    int palace_attack_val = 0;
    int y_start = (side == SIDE_BLACK) ? 7 : 0;
    int y_end   = (side == SIDE_BLACK) ? 9 : 2;
    for (int yy = y_start; yy <= y_end; yy++) {
        for (int xx = 3; xx <=5; xx++) {
            for (int i=0;i<attack_counts[yy][xx];i++) {
                AttackEntry ae = attack_lists[yy][xx][i];
                if (ae.side != side) palace_attack_val += ae.value;
            }
        }
    }
    /* advisors & elephants count */
    int advisors=0, elephants=0;
    for (int y=0;y<ROWS;y++){
        for (int x=0;x<COLS;x++){
            Piece q = board->sq[y][x];
            if (q.side != side) continue;
            if (q.type == PT_ADVISOR) advisors++;
            if (q.type == PT_ELEPHANT) elephants++;
        }
    }
    int shield = advisors + elephants;
    int val = (palace_attack_val / 20) * 18 - (2 - shield) * 60;

    /* center attack */
    int center_attack_val = 0;
    for (int i=0;i<attack_counts[ky][kx];i++){
        AttackEntry ae = attack_lists[ky][kx][i];
        if (ae.side != side) center_attack_val += ae.value;
    }
    val += (center_attack_val / 20) * 25;
    return (side == SIDE_BLACK) ? val : -val;
}

/* ---------- pawn_structure helper ---------- */
static int pawn_structure(const Board *board, Side side) {
    int bonus = 0;
    for (int y=0;y<ROWS;y++){
        for (int x=0;x<COLS;x++){
            Piece q = board->sq[y][x];
            if (q.side != side || q.type != PT_PAWN) continue;
            /* adjacent pawn bonus */
            for (int dx=-1; dx<=1; dx += 2){
                int nx = x + dx;
                if (nx>=0 && nx < COLS) {
                    Piece r = board->sq[y][nx];
                    if (r.side == side && r.type == PT_PAWN) {
                        bonus += (side==SIDE_BLACK) ? 30 : -30;
                    }
                }
            }
            int left = (x-1>=0 && board->sq[y][x-1].side==side && board->sq[y][x-1].type==PT_PAWN);
            int right= (x+1<COLS && board->sq[y][x+1].side==side && board->sq[y][x+1].type==PT_PAWN);
            if (!left && !right) bonus += (side==SIDE_BLACK) ? -18 : 18; /* Python 里是 -18 if black else -(-18) -> careful */
            /* 注意 Python 里对 sign 的处理：score += pawn_structure('black','卒') 和 += pawn_structure('red','兵') 
               在这里统一返回：对于 black 返回正数（对黑方有利），对于 red 返回负数 */
        }
    }
    return bonus;
}

/* ---------- evaluate_board (主函数) ---------- */
int evaluate_board(const Board *board_state) {
    /* build attack map first */
    build_attack_map(board_state);

    long total_material = 0;
    long black_material = 0;
    long red_material = 0;

    /* 材料统计 */
    for (int y=0;y<ROWS;y++){
        for (int x=0;x<COLS;x++){
            Piece p = board_state->sq[y][x];
            if (p.type == PT_NONE) continue;
            int v = piece_value(p.type);
            total_material += v;
            if (p.side == SIDE_BLACK) black_material += v;
            else if (p.side == SIDE_RED) red_material += v;
        }
    }

    double phase = (double)total_material / 16000.0;
    if (phase > 1.0) phase = 1.0;

    long score = 0;
    int black_mob = 0, red_mob = 0;
    long safety_penalty = 0;

    for (int y=0;y<ROWS;y++){
        for (int x=0;x<COLS;x++){
            Piece p = board_state->sq[y][x];
            if (p.type == PT_NONE) continue;
            int mult = (p.side == SIDE_BLACK) ? 1 : -1;
            int base = piece_value(p.type);
            int piece_val = base;

            /* 过河兵奖励 */
            if (p.type == PT_PAWN) {
                if (p.side == SIDE_BLACK) base += ((4 - y) > 0 ? (4 - y) * 12 : 0);
                else base += ((y - 5) > 0 ? (y - 5) * 12 : 0);
            }

            /* PST */
            int pst_val = PST[p.type][ (p.side==SIDE_BLACK)?1:0 ][y][x];

            /* net threat */
            int net_threat = 0;
            for (int i=0;i<attack_counts[y][x];i++){
                AttackEntry ae = attack_lists[y][x][i];
                if (ae.side == SIDE_BLACK) net_threat += ae.value;
                else if (ae.side == SIDE_RED) net_threat -= ae.value;
            }
            /* 权重 0.08 * mult */
            base += (int)(net_threat * 0.08 * mult);

            /* piece safety: lowest attacker value vs piece_val */
            int min_att = 0;
            int att_found = 0;
            for (int i=0;i<attack_counts[y][x];i++){
                AttackEntry ae = attack_lists[y][x][i];
                if (ae.side != p.side) {
                    if (!att_found || ae.value < min_att) min_att = ae.value, att_found = 1;
                }
            }
            if (att_found && min_att < piece_val) {
                int penalty = (piece_val * 40) / 100; /* 40% */
                safety_penalty += (long)penalty * mult; /* 黑子受罚 -> 减少总分 */
            }

            /* mobility */
            if (p.type != PT_GENERAL) {
                /* use gen_moves_for_piece to count moves */
                Move tmp[128];
                int mcnt = gen_moves_for_piece(board_state, x, y, tmp, 128);
                if (p.side == SIDE_BLACK) black_mob += mcnt; else red_mob += mcnt;
            }

            score += (base + pst_val) * mult;
        }
    }

    score -= safety_penalty;
    score += (int)((black_mob - red_mob) * (10 * (0.6 + 0.4 * phase)));

    /* King safety */
    score += king_safety_score(board_state, SIDE_BLACK);
    score += king_safety_score(board_state, SIDE_RED);

    /* Rook bonus (双车同线) */
    /* find rooks for each side */
    int rooks_black[2][2], rooks_red[2][2];
    int rb = 0, rr = 0;
    for (int y=0;y<ROWS;y++){
        for (int x=0;x<COLS;x++){
            Piece q = board_state->sq[y][x];
            if (q.type == PT_ROOK) {
                if (q.side == SIDE_BLACK) { if (rb<2) { rooks_black[rb][0]=y; rooks_black[rb][1]=x; rb++; } }
                else if (q.side == SIDE_RED) { if (rr<2) { rooks_red[rr][0]=y; rooks_red[rr][1]=x; rr++; } }
            }
        }
    }
    if (rb>=2) {
        if (rooks_black[0][0]==rooks_black[1][0] || rooks_black[0][1]==rooks_black[1][1]) score += 120;
    }
    if (rr>=2) {
        if (rooks_red[0][0]==rooks_red[1][0] || rooks_red[0][1]==rooks_red[1][1]) score -= 120;
    }

    /* cannon-horse bonus */
    auto cannon_horse = [&](Side side)->int {
        int bonus = 0;
        for (int y=0;y<ROWS;y++){
            for (int x=0;x<COLS;x++){
                Piece p = board_state->sq[y][x];
                if (p.side != side) continue;
                if (p.type == PT_CANNON) {
                    const int dirs[4][2] = {{-1,0},{1,0},{0,-1},{0,1}};
                    for (int d=0;d<4;d++){
                        int nx = x + dirs[d][0], ny = y + dirs[d][1];
                        if (nx>=0 && nx<COLS && ny>=0 && ny<ROWS){
                            Piece q = board_state->sq[ny][nx];
                            if (q.side == side && q.type == PT_HORSE) bonus += (side==SIDE_BLACK)?60:-60;
                        }
                    }
                }
            }
        }
        return bonus;
    };
    score += cannon_horse(SIDE_BLACK);
    score += cannon_horse(SIDE_RED);

    /* pawn structure */
    score += pawn_structure(board_state);

    /* 残局特判（保留原 Python 的分支） */
    PieceType br_types[] = {PT_ROOK};
    /* counts we need: bp(r) rp etc */
    int bp = count_pieces(board_state, SIDE_BLACK, (PieceType[]){PT_PAWN}, 1);
    int rp = count_pieces(board_state, SIDE_RED,   (PieceType[]){PT_PAWN}, 1);
    int br = count_pieces(board_state, SIDE_BLACK, (PieceType[]){PT_ROOK}, 1);
    int rr_ = count_pieces(board_state, SIDE_RED,  (PieceType[]){PT_ROOK}, 1);
    /* cannons/horse counts */
    int bc = count_pieces(board_state, SIDE_BLACK, (PieceType[]){PT_CANNON}, 1);
    int rc = count_pieces(board_state, SIDE_RED, (PieceType[]){PT_CANNON}, 1);
    int bh = count_pieces(board_state, SIDE_BLACK, (PieceType[]){PT_HORSE}, 1);
    int rh = count_pieces(board_state, SIDE_RED, (PieceType[]){PT_HORSE}, 1);

    if (bp == 1 && br == 1 && (bc+bh)==0 && (rr_+rc+rh+rp)==1) score += 140;
    if (rp == 1 && rr_ == 1 && (rc+rh)==0 && (br+bc+bh+bp)==1) score -= 140;
    if (bp >= 1 && bc >= 1 && rh == 1 && rr_ == 0) score += 90;
    if (rp >= 1 && rc >= 1 && bh == 1 && br == 0) score -= 90;
    if (bc == 2 && rh == 1 && (rr_+rp+rc)==0) score += 110;
    if (rc == 2 && bh == 1 && (br+bp+bc)==0) score -= 110;
    if (br >= 2 && (rr_+rh+rc) <= 1) score += 180;
    if (rr_ >= 2 && (br+bh+bc) <= 1) score -= 180;
    /* king with no advisors/elephants penalty */
    int black_knight_like = count_pieces(board_state, SIDE_BLACK, (PieceType[]){PT_GENERAL}, 1);
    /* simpler: count of advisor/elephant for side */
    int black_shield = count_pieces(board_state, SIDE_BLACK, (PieceType[]){PT_ADVISOR, PT_ELEPHANT}, 2);
    int red_shield   = count_pieces(board_state, SIDE_RED, (PieceType[]){PT_ADVISOR, PT_ELEPHANT}, 2);
    if (black_shield == 0) score -= 300;
    if (red_shield == 0) score += 300;

    return (int)score;
}
