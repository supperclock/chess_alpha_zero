// movegen.c
#include "movegen.h"
#include <string.h>

/* helper macros */
static inline int inside(int x, int y) {
    return x >= 0 && x < COLS && y >= 0 && y < ROWS;
}

/* append move if space */
static int add_move(Move *out, int max_out, int idx, int fy, int fx, int ty, int tx) {
    if (idx >= max_out) return idx; /* overflow: caller promised buffer large enough */
    out[idx].fy = fy; out[idx].fx = fx; out[idx].ty = ty; out[idx].tx = tx;
    out[idx].score = 0;
    return idx + 1;
}

/* Directions used (dx,dy) */
static const int D4[4][2] = { {1,0}, {-1,0}, {0,1}, {0,-1} };
static const int D4_O[4][2] = { {1,1}, {1,-1}, {-1,1}, {-1,-1} };
static const int H8[8][2]  = { {1,2},{2,1},{-1,2},{-2,1},{1,-2},{2,-1},{-1,-2},{-2,-1} };
static const int E4[4][2]  = { {2,2},{2,-2},{-2,2},{-2,-2} };

/* generate chariot (車) */
static int gen_chariot(const Board *board, int x, int y, Move *out, int max_out) {
    int idx = 0;
    for (int d=0; d<4; d++) {
        int dx = D4[d][0], dy = D4[d][1];
        int nx = x + dx, ny = y + dy;
        while (inside(nx, ny)) {
            Piece t = board->sq[ny][nx];
            if (t.type == PT_NONE) {
                idx = add_move(out, max_out, idx, y, x, ny, nx);
            } else {
                if (t.side != board->sq[y][x].side) {
                    idx = add_move(out, max_out, idx, y, x, ny, nx);
                }
                break;
            }
            nx += dx; ny += dy;
        }
    }
    return idx;
}

/* generate cannon (炮) */
static int gen_cannon(const Board *board, int x, int y, Move *out, int max_out) {
    int idx = 0;
    for (int d=0; d<4; d++) {
        int dx = D4[d][0], dy = D4[d][1];
        int nx = x + dx, ny = y + dy;
        /* non-capture moves while empty */
        while (inside(nx, ny) && board->sq[ny][nx].type == PT_NONE) {
            idx = add_move(out, max_out, idx, y, x, ny, nx);
            nx += dx; ny += dy;
        }
        if (!inside(nx, ny)) continue;
        /* jump over first piece to find capture target */
        nx += dx; ny += dy;
        while (inside(nx, ny)) {
            Piece t = board->sq[ny][nx];
            if (t.type != PT_NONE) {
                if (t.side != board->sq[y][x].side) {
                    idx = add_move(out, max_out, idx, y, x, ny, nx);
                }
                break;
            }
            nx += dx; ny += dy;
        }
    }
    return idx;
}

/* generate horse (馬) with leg blocking */
static int gen_horse(const Board *board, int x, int y, Move *out, int max_out) {
    int idx = 0;
    for (int i=0;i<8;i++){
        int dx = H8[i][0], dy = H8[i][1];
        int nx = x + dx, ny = y + dy;
        if (!inside(nx, ny)) continue;
        int legx, legy;
        if (abs(dx) == 2) {
            legx = x + dx/2; legy = y;
        } else {
            legx = x; legy = y + dy/2;
        }
        if (!inside(legx, legy)) continue;
        if (board->sq[legy][legx].type != PT_NONE) continue; /* leg blocked */
        Piece t = board->sq[ny][nx];
        if (t.type == PT_NONE || t.side != board->sq[y][x].side) {
            idx = add_move(out, max_out, idx, y, x, ny, nx);
        }
    }
    return idx;
}

/* generate soldier (兵/卒) */
static int gen_soldier(const Board *board, int x, int y, Move *out, int max_out) {
    int idx = 0;
    Side side = board->sq[y][x].side;
    int forward = (side == SIDE_RED) ? 1 : -1;
    int ny = y + forward;
    if (inside(x, ny)) {
        Piece t = board->sq[ny][x];
        if (t.type == PT_NONE || t.side != side) {
            idx = add_move(out, max_out, idx, y, x, ny, x);
        }
    }
    int river_crossed = (side == SIDE_RED && y >= 5) || (side == SIDE_BLACK && y <= 4);
    if (river_crossed) {
        for (int dx = -1; dx <= 1; dx += 2) {
            int nx = x + dx;
            if (!inside(nx, y)) continue;
            Piece t = board->sq[y][nx];
            if (t.type == PT_NONE || t.side != side) {
                idx = add_move(out, max_out, idx, y, x, y, nx);
            }
        }
    }
    return idx;
}

/* generate general (帥/將), includes kings-facing capture */
static int gen_general(const Board *board, int x, int y, Move *out, int max_out) {
    int idx = 0;
    Side side = board->sq[y][x].side;
    int palace_x_min = 3, palace_x_max = 5;
    int y_min = (side == SIDE_RED) ? 0 : 7;
    int y_max = (side == SIDE_RED) ? 2 : 9;
    /* orthogonal moves within palace */
    for (int d=0; d<4; d++) {
        int nx = x + D4[d][0], ny = y + D4[d][1];
        if (nx < palace_x_min || nx > palace_x_max || ny < y_min || ny > y_max) continue;
        Piece t = board->sq[ny][nx];
        if (t.type == PT_NONE || t.side != side) {
            idx = add_move(out, max_out, idx, y, x, ny, nx);
        }
    }
    /* kings-facing: find opposing general and if same file and no blocking pieces allow capture */
    Side opp = (side == SIDE_RED) ? SIDE_BLACK : SIDE_RED;
    int kx=-1, ky=-1;
    for (int yy=0; yy<ROWS; yy++) {
        for (int xx=0; xx<COLS; xx++) {
            Piece q = board->sq[yy][xx];
            if (q.type == PT_GENERAL && q.side == opp) { kx = xx; ky = yy; break; }
        }
        if (kx != -1) break;
    }
    if (kx != -1 && kx == x) {
        int blocked = 0;
        int y1 = (y < ky) ? y : ky;
        int y2 = (y < ky) ? ky : y;
        for (int yy = y1 + 1; yy < y2; yy++) {
            if (board->sq[yy][x].type != PT_NONE) { blocked = 1; break; }
        }
        if (!blocked) {
            idx = add_move(out, max_out, idx, y, x, ky, kx);
        }
    }
    return idx;
}

/* generate advisor (仕/士) */
static int gen_advisor(const Board *board, int x, int y, Move *out, int max_out) {
    int idx = 0;
    Side side = board->sq[y][x].side;
    int palace_x_min = 3, palace_x_max = 5;
    int y_min = (side == SIDE_RED) ? 0 : 7;
    int y_max = (side == SIDE_RED) ? 2 : 9;
    for (int d=0; d<4; d++) {
        int nx = x + D4_O[d][0], ny = y + D4_O[d][1];
        if (nx < palace_x_min || nx > palace_x_max || ny < y_min || ny > y_max) continue;
        Piece t = board->sq[ny][nx];
        if (t.type == PT_NONE || t.side != side) {
            idx = add_move(out, max_out, idx, y, x, ny, nx);
        }
    }
    return idx;
}

/* generate elephant (相/象) with elephant-eye and side-specific board half restriction */
static int gen_elephant(const Board *board, int x, int y, Move *out, int max_out) {
    int idx = 0;
    Side side = board->sq[y][x].side;
    for (int d=0; d<4; d++) {
        int dx = E4[d][0], dy = E4[d][1];
        int nx = x + dx, ny = y + dy;
        /* side-specific limit: match Python logic:
           y_limit = 5 if side == 'red' else 4
           require (side == 'red' and ny >= y_limit) or (side == 'black' and ny <= y_limit)
        */
        int y_limit = (side == SIDE_RED) ? 5 : 4;
        if (side == SIDE_RED) {
            if (!(ny >= y_limit)) continue;
        } else {
            if (!(ny <= y_limit)) continue;
        }
        if (!inside(nx, ny)) continue;
        int mx = x + dx/2, my = y + dy/2;
        if (!inside(mx, my)) continue;
        if (board->sq[my][mx].type != PT_NONE) continue; /* elephant eye blocked */
        Piece t = board->sq[ny][nx];
        if (t.type == PT_NONE || t.side != side) {
            idx = add_move(out, max_out, idx, y, x, ny, nx);
        }
    }
    return idx;
}

/* Public entry: generate moves for piece at (x,y) */
int gen_moves_for_piece(const Board *board, int x, int y, Move *out, int max_out) {
    if (!inside(x,y) || max_out <= 0) return 0;
    Piece p = board->sq[y][x];
    if (p.type == PT_NONE) return 0;
    switch (p.type) {
        case PT_ROOK:    return gen_chariot(board, x, y, out, max_out);
        case PT_CANNON:  return gen_cannon(board, x, y, out, max_out);
        case PT_HORSE:   return gen_horse(board, x, y, out, max_out);
        case PT_PAWN:    return gen_soldier(board, x, y, out, max_out);
        case PT_GENERAL: return gen_general(board, x, y, out, max_out);
        case PT_ADVISOR: return gen_advisor(board, x, y, out, max_out);
        case PT_ELEPHANT:return gen_elephant(board, x, y, out, max_out);
        default: return 0;
    }
}
