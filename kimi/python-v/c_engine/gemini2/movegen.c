#include <stddef.h> // for NULL
#include "chess.h"

// --- 辅助宏和函数 ---


// 根据棋子枚举值获取其阵营
Side get_piece_side(Piece p) {
    if (p >= r_king && p <= r_pawn) return RED;
    if (p >= b_king && p <= b_pawn) return BLACK;
    return -1; // 表示空位或错误
}

// --- 各个棋子的走法生成函数 ---
// 这些函数生成的是“伪合法”走法，即只考虑棋子自身的规则，不考虑王是否被将军。
// 它们都接收一个指向 move_list 的指针，并将生成的走法添加到列表中，然后返回生成的走法数量。

int gen_chariot_moves(const BoardState* state, int y, int x, Side side, Move* move_list) {
    int count = 0;
    const int DIRS[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (int i = 0; i < 4; ++i) {
        for (int step = 1; step < 10; ++step) {
            int ny = y + DIRS[i][0] * step;
            int nx = x + DIRS[i][1] * step;
            if (!is_inside(ny, nx)) break;
            
            Piece target = state->board[ny][nx];
            if (target == EMPTY) {
                move_list[count++] = (Move){y, x, ny, nx, EMPTY, 0};
            } else {
                if (get_piece_side(target) != side) {
                    move_list[count++] = (Move){y, x, ny, nx, target, 0};
                }
                break; // 撞到棋子，停止延伸
            }
        }
    }
    return count;
}

int gen_horse_moves(const BoardState* state, int y, int x, Side side, Move* move_list) {
    int count = 0;
    const int H_MOVES[8][2] = {{2, 1}, {2, -1}, {-2, 1}, {-2, -1}, {1, 2}, {1, -2}, {-1, 2}, {-1, -2}};
    const int LEG_MOVES[8][2] = {{1, 0}, {1, 0}, {-1, 0}, {-1, 0}, {0, 1}, {0, -1}, {0, 1}, {0, -1}};
    
    for (int i = 0; i < 8; ++i) {
        // 检查马腿是否被别住
        if (state->board[y + LEG_MOVES[i][0]][x + LEG_MOVES[i][1]] != EMPTY) continue;
        
        int ny = y + H_MOVES[i][0];
        int nx = x + H_MOVES[i][1];
        if (!is_inside(ny, nx)) continue;

        Piece target = state->board[ny][nx];
        if (target == EMPTY || get_piece_side(target) != side) {
            move_list[count++] = (Move){y, x, ny, nx, target, 0};
        }
    }
    return count;
}

int gen_cannon_moves(const BoardState* state, int y, int x, Side side, Move* move_list) {
    int count = 0;
    const int DIRS[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    for (int i = 0; i < 4; ++i) {
        int screen = 0; // 炮架
        for (int step = 1; step < 10; ++step) {
            int ny = y + DIRS[i][0] * step;
            int nx = x + DIRS[i][1] * step;
            if (!is_inside(ny, nx)) break;

            Piece target = state->board[ny][nx];
            if (screen == 0) {
                if (target == EMPTY) {
                    move_list[count++] = (Move){y, x, ny, nx, EMPTY, 0};
                } else {
                    screen = 1; // 遇到了第一个棋子，作为炮架
                }
            } else { // 已经有炮架
                if (target != EMPTY) {
                    if (get_piece_side(target) != side) {
                        move_list[count++] = (Move){y, x, ny, nx, target, 0};
                    }
                    break; // 无论吃不吃，都停止延伸
                }
            }
        }
    }
    return count;
}

int gen_elephant_moves(const BoardState* state, int y, int x, Side side, Move* move_list) {
    int count = 0;
    const int E_MOVES[4][2] = {{2, 2}, {2, -2}, {-2, 2}, {-2, -2}};
    const int EYE_MOVES[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    
    for (int i = 0; i < 4; ++i) {
        // 检查象眼是否被别住
        if (state->board[y + EYE_MOVES[i][0]][x + EYE_MOVES[i][1]] != EMPTY) continue;

        int ny = y + E_MOVES[i][0];
        int nx = x + E_MOVES[i][1];
        if (!is_inside(ny, nx)) continue;

        // 检查是否过河
        if ((side == RED && ny >= 5) || (side == BLACK && ny <= 4)) continue;
        
        Piece target = state->board[ny][nx];
        if (target == EMPTY || get_piece_side(target) != side) {
            move_list[count++] = (Move){y, x, ny, nx, target, 0};
        }
    }
    return count;
}

int gen_advisor_moves(const BoardState* state, int y, int x, Side side, Move* move_list) {
    int count = 0;
    const int A_MOVES[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    
    for (int i = 0; i < 4; ++i) {
        int ny = y + A_MOVES[i][0];
        int nx = x + A_MOVES[i][1];
        if (!is_inside(ny, nx)) continue;

        // 检查是否在九宫内
        if (nx < 3 || nx > 5) continue;
        if (side == RED && ny > 2) continue;
        if (side == BLACK && ny < 7) continue;

        Piece target = state->board[ny][nx];
        if (target == EMPTY || get_piece_side(target) != side) {
            move_list[count++] = (Move){y, x, ny, nx, target, 0};
        }
    }
    return count;
}

int gen_king_moves(const BoardState* state, int y, int x, Side side, Move* move_list) {
    int count = 0;
    const int K_MOVES[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    
    for (int i = 0; i < 4; ++i) {
        int ny = y + K_MOVES[i][0];
        int nx = x + K_MOVES[i][1];
        if (!is_inside(ny, nx)) continue;

        // 检查是否在九宫内
        if (nx < 3 || nx > 5) continue;
        if (side == RED && ny > 2) continue;
        if (side == BLACK && ny < 7) continue;

        Piece target = state->board[ny][nx];
        if (target == EMPTY || get_piece_side(target) != side) {
            move_list[count++] = (Move){y, x, ny, nx, target, 0};
        }
    }

    // 飞将规则
    int opp_ky, opp_kx;
    if (find_general(state, (side == RED) ? BLACK : RED, &opp_ky, &opp_kx)) {
        if (x == opp_kx) {
            int is_clear = 1;
            int start_y = (y < opp_ky) ? y : opp_ky;
            int end_y = (y < opp_ky) ? opp_ky : y;
            for (int i = start_y + 1; i < end_y; ++i) {
                if (state->board[i][x] != EMPTY) {
                    is_clear = 0;
                    break;
                }
            }
            if (is_clear) {
                move_list[count++] = (Move){y, x, opp_ky, opp_kx, state->board[opp_ky][opp_kx], 0};
            }
        }
    }
    return count;
}

int gen_pawn_moves(const BoardState* state, int y, int x, Side side, Move* move_list) {
    int count = 0;
    int forward = (side == RED) ? 1 : -1;

    // 向前走
    int ny = y + forward;
    if (is_inside(ny, x)) {
        Piece target = state->board[ny][x];
        if (target == EMPTY || get_piece_side(target) != side) {
            move_list[count++] = (Move){y, x, ny, x, target, 0};
        }
    }
    
    // 过河后可以横走
    int river_crossed = (side == RED && y >= 5) || (side == BLACK && y <= 4);
    if (river_crossed) {
        for (int dx = -1; dx <= 1; dx += 2) {
            int nx = x + dx;
            if (is_inside(y, nx)) {
                Piece target = state->board[y][nx];
                if (target == EMPTY || get_piece_side(target) != side) {
                    move_list[count++] = (Move){y, x, y, nx, target, 0};
                }
            }
        }
    }
    return count;
}


// --- 主走法生成函数 ---
// 这是暴露给外部的唯一接口。
// 它首先生成所有伪合法走法，然后逐一测试，过滤掉会导致自己王被将军的走法。
int generate_moves(BoardState* state, Move legal_move_list[]) {
    Move pseudo_legal_moves[MAX_MOVES];
    int pseudo_count = 0;
    Side current_side = state->side_to_move;

    // 1. 生成所有伪合法走法
    for (int y = 0; y < ROWS; ++y) {
        for (int x = 0; x < COLS; ++x) {
            Piece p = state->board[y][x];
            if (p != EMPTY && get_piece_side(p) == current_side) {
                switch(p) {
                    case r_king:
                    case b_king:
                        pseudo_count += gen_king_moves(state, y, x, current_side, pseudo_legal_moves + pseudo_count);
                        break;
                    case r_advisor:
                    case b_advisor:
                        pseudo_count += gen_advisor_moves(state, y, x, current_side, pseudo_legal_moves + pseudo_count);
                        break;
                    case r_elephant:
                    case b_elephant:
                        pseudo_count += gen_elephant_moves(state, y, x, current_side, pseudo_legal_moves + pseudo_count);
                        break;
                    case r_horse:
                    case b_horse:
                        pseudo_count += gen_horse_moves(state, y, x, current_side, pseudo_legal_moves + pseudo_count);
                        break;
                    case r_chariot:
                    case b_chariot:
                        pseudo_count += gen_chariot_moves(state, y, x, current_side, pseudo_legal_moves + pseudo_count);
                        break;
                    case r_cannon:
                    case b_cannon:
                        pseudo_count += gen_cannon_moves(state, y, x, current_side, pseudo_legal_moves + pseudo_count);
                        break;
                    case r_pawn:
                    case b_pawn:
                        pseudo_count += gen_pawn_moves(state, y, x, current_side, pseudo_legal_moves + pseudo_count);
                        break;
                    default: break;
                }
            }
        }
    }

    // 2. 过滤伪合法走法，得到合法走法
    int legal_count = 0;
    for (int i = 0; i < pseudo_count; ++i) {
        Move current_move = pseudo_legal_moves[i];
        make_move(state, &current_move);
        // 如果走完这步棋后，自己的王没有被将军，那么这是一步合法走法
        if (!is_in_check(state, current_side)) {
            legal_move_list[legal_count++] = current_move;
        }
        // 无论如何都要撤销走法，以测试下一个伪合法走法
        unmake_move(state, &current_move);
    }
    
    return legal_count;
}