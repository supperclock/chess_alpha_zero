// movegen.h
#ifndef MOVEGEN_H
#define MOVEGEN_H

#include "board.h"

/*
 * gen_moves_for_piece:
 *   board  - pointer to current board
 *   x,y    - coordinates of the piece to generate moves for (x: file 0..8, y: rank 0..9)
 *   out    - output array of Move (caller supplies buffer)
 *   max_out- maximum number of entries in out
 * Returns:
 *   number of moves written to out (0..max_out)
 *
 * NOTE: Move.fy/fx filled with source (y,x), Move.ty/tx with target (ty,tx).
 */
int gen_moves_for_piece(const Board *board, int x, int y, Move *out, int max_out);

#endif // MOVEGEN_H
