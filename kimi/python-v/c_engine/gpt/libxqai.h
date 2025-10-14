// libxqai.h
#ifndef LIBXQAI_H
#define LIBXQAI_H

#ifdef __cplusplus
extern "C" {
#endif

// board: int buf[90], row-major (y*9 + x), codes described below
// side: 0 = black to move, 1 = red to move
// out_move: int out_move[4] where out_move = {fy, fx, ty, tx}
// returns 1 on success, 0 on failure
int xq_minimax_root(const int *board90, int side, double time_limit, int *out_move);

// evaluate single position quickly (black perspective positive)
int xq_evaluate(const int *board90);

// initialization (optional)
void xq_init();

#ifdef __cplusplus
}
#endif

#endif
