// rules.c
#include "rules.h"

/* 辅助函数 */
static inline int inside(int x, int y) {
    return x >= 0 && x < COLS && y >= 0 && y < ROWS;
}

/* 找到指定阵营的王位置 */
static int find_general(const Board *b, Side side, int *gx, int *gy) {
    for (int y=0; y<ROWS; y++) {
        for (int x=0; x<COLS; x++) {
            Piece p = b->sq[y][x];
            if (p.type == PT_GENERAL && p.side == side) {
                *gx = x; *gy = y;
                return 1;
            }
        }
    }
    return 0;
}

/* 判断是否被将军 */
int in_check(const Board *b, Side side) {
    int gx, gy;
    if (!find_general(b, side, &gx, &gy)) return 0;

    Side opp = (side == SIDE_RED) ? SIDE_BLACK : SIDE_RED;

    /* --- 敌方车/炮攻击 --- */
    for (int d=0; d<4; d++) {
        int dx = (d==0)-(d==1);
        int dy = (d==2)-(d==3);
        int nx = gx + dx, ny = gy + dy;
        int blocked = 0;
        while (inside(nx, ny)) {
            Piece t = b->sq[ny][nx];
            if (t.type != PT_NONE) {
                if (t.side == opp) {
                    if (!blocked && t.type == PT_ROOK)
                        return 1;
                    if (blocked && t.type == PT_CANNON)
                        return 1;
                }
                blocked++;
            }
            nx += dx; ny += dy;
        }
    }

    /* --- 敌方马攻击 --- */
    const int H8[8][2] = {{1,2},{2,1},{-1,2},{-2,1},{1,-2},{2,-1},{-1,-2},{-2,-1}};
    for (int i=0; i<8; i++) {
        int dx = H8[i][0], dy = H8[i][1];
        int nx = gx + dx, ny = gy + dy;
        if (!inside(nx, ny)) continue;
        int legx = gx + (abs(dx)==2 ? dx/2 : 0);
        int legy = gy + (abs(dx)==1 ? dy/2 : 0);
        if (!inside(legx, legy)) continue;
        if (b->sq[legy][legx].type != PT_NONE) continue;
        Piece t = b->sq[ny][nx];
        if (t.side == opp && t.type == PT_HORSE)
            return 1;
    }

    /* --- 敌方兵攻击 --- */
    for (int dy=-1; dy<=1; dy+=2) {
        int ny = gy + dy;
        if (!inside(gx, ny)) continue;
        Piece t = b->sq[ny][gx];
        if (t.side == opp && t.type == PT_PAWN) {
            if ((t.side == SIDE_RED && dy == -1) || (t.side == SIDE_BLACK && dy == 1))
                return 1; /* 直前攻击 */
        }
    }
    /* 横向攻击（过河后） */
    for (int dx=-1; dx<=1; dx+=2) {
        int nx = gx + dx;
        if (!inside(nx, gy)) continue;
        Piece t = b->sq[gy][nx];
        if (t.side == opp && t.type == PT_PAWN) {
            if ((t.side == SIDE_RED && gy >= 5) || (t.side == SIDE_BLACK && gy <= 4))
                return 1;
        }
    }

    /* --- 敌方相、士攻击 --- */
    const int E4[4][2] = {{2,2},{2,-2},{-2,2},{-2,-2}};
    for (int i=0;i<4;i++){
        int nx = gx + E4[i][0];
        int ny = gy + E4[i][1];
        if (!inside(nx, ny)) continue;
        int mx = gx + E4[i][0]/2, my = gy + E4[i][1]/2;
        if (!inside(mx,my)) continue;
        if (b->sq[my][mx].type != PT_NONE) continue;
        Piece t = b->sq[ny][nx];
        if (t.side == opp && t.type == PT_ELEPHANT) {
            int ylimit = (t.side == SIDE_RED) ? 5 : 4;
            if ((t.side == SIDE_RED && ny >= ylimit) || (t.side == SIDE_BLACK && ny <= ylimit))
                return 1;
        }
    }

    const int D4O[4][2] = {{1,1},{1,-1},{-1,1},{-1,-1}};
    for (int i=0;i<4;i++){
        int nx = gx + D4O[i][0];
        int ny = gy + D4O[i][1];
        if (!inside(nx, ny)) continue;
        int x_min = 3, x_max = 5;
        int y_min = (opp == SIDE_RED) ? 0 : 7;
        int y_max = (opp == SIDE_RED) ? 2 : 9;
        if (nx >= x_min && nx <= x_max && ny >= y_min && ny <= y_max) {
            Piece t = b->sq[ny][nx];
            if (t.side == opp && t.type == PT_ADVISOR)
                return 1;
        }
    }

    /* --- 敌方将面对面 --- */
    for (int y=gy+1; y<ROWS; y++) {
        Piece t = b->sq[y][gx];
        if (t.type == PT_NONE) continue;
        if (t.side == opp && t.type == PT_GENERAL)
            return 1;
        else break;
    }
    for (int y=gy-1; y>=0; y--) {
        Piece t = b->sq[y][gx];
        if (t.type == PT_NONE) continue;
        if (t.side == opp && t.type == PT_GENERAL)
            return 1;
        else break;
    }

    return 0;
}

/* square_attacked: 检查 square (tx,ty) 是否被 attacker_side 攻击
   简化但包含关键攻击形式：车/炮 直线、炮跳跃、马（马脚）、兵（前进/横吃）、相/仕/将 */
int square_attacked(const Board *b, int tx, int ty, Side attacker_side) {
    /* rooks and cannons: straight lines */
    const int D4[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
    for (int d=0; d<4; d++) {
        int dx = D4[d][0], dy = D4[d][1];
        int nx = tx + dx, ny = ty + dy;
        int blocked = 0;
        while (nx>=0 && nx<COLS && ny>=0 && ny<ROWS) {
            Piece p = b->sq[ny][nx];
            if (p.type != PT_NONE) {
                if (p.side == attacker_side) {
                    if (!blocked && p.type == PT_ROOK) return 1;
                    if (blocked && p.type == PT_CANNON) return 1;
                }
                blocked++;
            }
            nx += dx; ny += dy;
        }
    }

    /* horse attacks (consider leg) */
    const int H8[8][2] = {{1,2},{2,1},{-1,2},{-2,1},{1,-2},{2,-1},{-1,-2},{-2,-1}};
    for (int i=0;i<8;i++) {
        int dx = H8[i][0], dy = H8[i][1];
        int sx = tx - dx, sy = ty - dy; /* source square where attacker horse would stand */
        if (sx<0||sx>=COLS||sy<0||sy>=ROWS) continue;
        int legx, legy;
        if (abs(dx) == 2) { legx = sx + dx/2; legy = sy; }
        else { legx = sx; legy = sy + dy/2; }
        if (legx<0||legx>=COLS||legy<0||legy>=ROWS) continue;
        if (b->sq[legy][legx].type != PT_NONE) continue; /* leg blocked */
        Piece p = b->sq[sy][sx];
        if (p.side == attacker_side && p.type == PT_HORSE) return 1;
    }

    /* pawn attacks */
    /* attacker_side's pawn attacks forward (depends on side) */
    if (attacker_side == SIDE_RED) {
        /* red pawns move +y */
        int sy = ty - 1;
        if (sy>=0) {
            if (tx-1>=0) { Piece p = b->sq[sy][tx-1]; if (p.side==attacker_side && p.type==PT_PAWN) return 1; }
            if (tx+1<COLS) { Piece p = b->sq[sy][tx+1]; if (p.side==attacker_side && p.type==PT_PAWN) return 1; }
        }
        /* horizontal after crossing river */
        int sy2 = ty;
        if (sy2 >= 5) { /* red pawn past river can attack sideways */
            if (tx-1>=0) { Piece p = b->sq[sy2][tx-1]; if (p.side==attacker_side && p.type==PT_PAWN) return 1; }
            if (tx+1<COLS) { Piece p = b->sq[sy2][tx+1]; if (p.side==attacker_side && p.type==PT_PAWN) return 1; }
        }
    } else if (attacker_side == SIDE_BLACK) {
        int sy = ty + 1;
        if (sy<ROWS) {
            if (tx-1>=0) { Piece p = b->sq[sy][tx-1]; if (p.side==attacker_side && p.type==PT_PAWN) return 1; }
            if (tx+1<COLS) { Piece p = b->sq[sy][tx+1]; if (p.side==attacker_side && p.type==PT_PAWN) return 1; }
        }
        int sy2 = ty;
        if (sy2 <= 4) {
            if (tx-1>=0) { Piece p = b->sq[sy2][tx-1]; if (p.side==attacker_side && p.type==PT_PAWN) return 1; }
            if (tx+1<COLS) { Piece p = b->sq[sy2][tx+1]; if (p.side==attacker_side && p.type==PT_PAWN) return 1; }
        }
    }

    /* advisor/elephant near palace and diagonals (conservative) */
    const int D4O[4][2] = {{1,1},{1,-1},{-1,1},{-1,-1}};
    for (int i=0;i<4;i++){
        int sx = tx + D4O[i][0], sy = ty + D4O[i][1];
        if (sx<0||sx>=COLS||sy<0||sy>=ROWS) continue;
        Piece p = b->sq[sy][sx];
        if (p.side == attacker_side && (p.type == PT_ADVISOR)) return 1;
    }
    const int E4[4][2] = {{2,2},{2,-2},{-2,2},{-2,-2}};
    for (int i=0;i<4;i++){
        int sx = tx + E4[i][0], sy = ty + E4[i][1];
        if (sx<0||sx>=COLS||sy<0||sy>=ROWS) continue;
        int mx = tx + E4[i][0]/2, my = ty + E4[i][1]/2;
        if (mx<0||mx>=COLS||my<0||my>=ROWS) continue;
        if (b->sq[my][mx].type != PT_NONE) continue;
        Piece p = b->sq[sy][sx];
        if (p.side == attacker_side && p.type == PT_ELEPHANT) return 1;
    }

    /* general facing (opponent general in file with no blocking) */
    for (int y=ty-1;y>=0;y--) {
        Piece p = b->sq[y][tx];
        if (p.type != PT_NONE) {
            if (p.side == attacker_side && p.type == PT_GENERAL) return 1;
            break;
        }
    }
    for (int y=ty+1;y<ROWS;y++) {
        Piece p = b->sq[y][tx];
        if (p.type != PT_NONE) {
            if (p.side == attacker_side && p.type == PT_GENERAL) return 1;
            break;
        }
    }

    return 0;
}