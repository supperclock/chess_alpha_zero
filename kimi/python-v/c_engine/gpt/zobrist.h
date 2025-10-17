// zobrist.h
#ifndef ZOBRIST_H
#define ZOBRIST_H

#include "board.h"
#include <stdint.h>

void zobrist_init(void);
uint64_t compute_zobrist(const Board *board);

#endif
