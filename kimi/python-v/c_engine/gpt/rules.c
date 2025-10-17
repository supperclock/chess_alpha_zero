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
