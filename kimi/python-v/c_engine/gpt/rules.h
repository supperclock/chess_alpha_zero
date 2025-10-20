// rules.h
#ifndef RULES_H
#define RULES_H

#include "board.h"

int in_check(const Board *board, Side side);
/* 判断某方是否攻击 (x,y) */
int square_attacked(const Board *b, int x, int y, Side attacker);

#endif
