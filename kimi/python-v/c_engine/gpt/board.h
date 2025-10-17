// board.h （补充）
// board.h
#ifndef BOARD_H
#define BOARD_H

#include <stdint.h>

#define ROWS 10
#define COLS 9

typedef enum {
    SIDE_NONE = 0,
    SIDE_RED  = 1,
    SIDE_BLACK= 2
} Side;

typedef enum {
    PT_NONE = 0,
    PT_ROOK,    // 車
    PT_CANNON,  // 炮
    PT_HORSE,   // 馬
    PT_ELEPHANT,// 相/象
    PT_ADVISOR, // 仕/士
    PT_GENERAL, // 帥/將
    PT_PAWN     // 兵/卒
} PieceType;

/* 简化 Piece 结构：type 和 side 就够了 */
typedef struct {
    PieceType type;
    Side side;
} Piece;

/* Move struct 对应 Python 中 opening_book.Move */
typedef struct {
    int fy, fx;
    int ty, tx;
    int score;
    /* captured pointer 不在这里保存（由 make_move/unmake_move 维护） */
} Move;

/* Board wrapper */
typedef struct {
    Piece sq[ROWS][COLS];
    Side side_to_move;
} Board;

/* Find general position */
typedef struct { int x, y; int found; } KingPos;

/* prototypes provided by other modules (move generation) */
/* gen_moves_for_piece: 根据 (x,y) 返回该子所有 pseudo moves，填到 moves 中，返回数量 */
/* 具体实现由 movegen 模块提供，签名需一致 */
int gen_moves_for_piece(const Board *board, int x, int y, Move *moves_out, int max_moves);


void make_move(Board *b, const Move *m, Piece *captured);
void unmake_move(Board *b, const Move *m, const Piece *captured);

#endif
