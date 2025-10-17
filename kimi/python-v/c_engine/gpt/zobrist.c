// zobrist.c
#include "zobrist.h"
#include <stdlib.h>
#include <time.h>

static uint64_t ZOBRIST[8][3][ROWS][COLS]; /* [PieceType][Side][y][x] */
static uint64_t ZOBRIST_SIDE = 0;

void zobrist_init(void) {
    srand(123456); /* 固定种子，保证可复现 */
    for (int p=0; p<8; p++) {
        for (int s=0; s<3; s++) {
            for (int y=0; y<ROWS; y++) {
                for (int x=0; x<COLS; x++) {
                    uint64_t hi = (uint64_t)(rand() & 0xFFFF);
                    uint64_t lo = (uint64_t)(rand() & 0xFFFF);
                    uint64_t v  = (hi << 48) ^ (lo << 32) ^ (rand() << 16) ^ rand();
                    ZOBRIST[p][s][y][x] = v;
                }
            }
        }
    }
    uint64_t hi = (uint64_t)(rand() & 0xFFFF);
    uint64_t lo = (uint64_t)(rand() & 0xFFFF);
    ZOBRIST_SIDE = (hi << 48) ^ (lo << 32) ^ (rand() << 16) ^ rand();
}

/* 根据当前局面计算 Zobrist 哈希 */
uint64_t compute_zobrist(const Board *b) {
    uint64_t h = 0;
    for (int y=0; y<ROWS; y++) {
        for (int x=0; x<COLS; x++) {
            Piece p = b->sq[y][x];
            if (p.type != PT_NONE && p.side != SIDE_NONE) {
                h ^= ZOBRIST[p.type][p.side][y][x];
            }
        }
    }
    if (b->side_to_move == SIDE_BLACK)
        h ^= ZOBRIST_SIDE;
    return h;
}
