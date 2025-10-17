// board.c
#include "board.h"
#include <string.h>

/* 执行走子 */
void make_move(Board *b, const Move *m, Piece *captured) {
    Piece src = b->sq[m->fy][m->fx];
    *captured = b->sq[m->ty][m->tx];
    b->sq[m->ty][m->tx] = src;
    b->sq[m->fy][m->fx].type = PT_NONE;
    b->sq[m->fy][m->fx].side = SIDE_NONE;
    /* 交换行棋方 */
    b->side_to_move = (b->side_to_move == SIDE_RED) ? SIDE_BLACK : SIDE_RED;
}

/* 撤销走子 */
void unmake_move(Board *b, const Move *m, const Piece *captured) {
    Piece src = b->sq[m->ty][m->tx];
    b->sq[m->fy][m->fx] = src;
    b->sq[m->ty][m->tx] = *captured;
    /* 恢复行棋方 */
    b->side_to_move = (b->side_to_move == SIDE_RED) ? SIDE_BLACK : SIDE_RED;
}
