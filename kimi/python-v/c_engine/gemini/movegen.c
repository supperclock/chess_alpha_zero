#include <stddef.h> // for NULL
#include "chess.h"

// --- 辅助宏和函数 ---

// 检查坐标是否在棋盘内
#define is_inside(y, x) (x >= 0 && x < COLS && y >= 0 && y < ROWS)

// 根据棋子枚举值获取其阵营
Side get_piece_side(Piece p) {
    if (p >= r_king && p <= r_pawn) return RED;
    if (p >= b_king && p <= b_pawn) return BLACK;
    return -1; // 表示空位或错误
}

// --- 内部依赖函数声明 ---
// (假设这些函数在 chess.h 中或链接时可用)
extern int find_general(const BoardState* state, Side side, int* y, int* x);
extern void make_move(BoardState* state, Move* move);
extern void unmake_move(BoardState* state, const Move* move);
extern int is_in_check(const BoardState* state, Side side);


// --- MODIFIED: 新增走法生成上下文 ---
//
// 为了满足 "函数不要新增参数" 的要求，
// 我们使用一个 file-scope (static) 的上下文变量。
// generate_moves() 会在开始时设置这个变量，
// 
// 
// 而其它走法生成函数 (兵/仕/相/帅) 会读取它。
typedef struct {
    int forward_dir;              // 兵的 "前进" 方向 (1 或 -1)
    int palace_bottom;            // 九宫 "底" (0 或 7)
    int palace_top;               // 九宫 "顶" (2 或 9)
    int river_cross_line;         // 兵 "过河" 的 Y 坐标 (5 或 4)
    int elephant_home_river_line; // 象 "不过河" 的 Y 坐标 (4 或 5)
} MoveGenContext;

// 定义这个静态上下文变量
static MoveGenContext g_ctx;

// --- 各个棋子的走法生成函数 ---

// (车, 马, 炮 的函数没有改动，因为它们的走法与绝对方向无关)

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

// --- MODIFIED: 不再使用 side 判断方向，而是读取 g_ctx ---
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

        // --- MODIFIED: 使用 g_ctx 检查是否过河 ---
        // g_ctx.elephant_home_river_line 是己方一侧的河界 (y=4 或 y=5)
        // g_ctx.forward_dir 是己方的前进方向 (1 或 -1)
        if (g_ctx.forward_dir == 1) { // 己方在下方 (e.g. Red)
            if (ny > g_ctx.elephant_home_river_line) continue; // 目标点 y > 4 (即 >= 5) 则算过河
        } else { // 己方在上方 (e.g. Black)
            if (ny < g_ctx.elephant_home_river_line) continue; // 目标点 y < 5 (即 <= 4) 则算过河
        }
        
        Piece target = state->board[ny][nx];
        if (target == EMPTY || get_piece_side(target) != side) {
            move_list[count++] = (Move){y, x, ny, nx, target, 0};
        }
    }
    return count;
}

// --- MODIFIED: 不再使用 side 判断九宫，而是读取 g_ctx ---
int gen_advisor_moves(const BoardState* state, int y, int x, Side side, Move* move_list) {
    int count = 0;
    const int A_MOVES[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    
    for (int i = 0; i < 4; ++i) {
        int ny = y + A_MOVES[i][0];
        int nx = x + A_MOVES[i][1];
        if (!is_inside(ny, nx)) continue;

        // --- MODIFIED: 使用 g_ctx 检查是否在九宫内 ---
        if (nx < 3 || nx > 5) continue;
        if (ny < g_ctx.palace_bottom || ny > g_ctx.palace_top) continue; // 动态Y坐标检查

        Piece target = state->board[ny][nx];
        if (target == EMPTY || get_piece_side(target) != side) {
            move_list[count++] = (Move){y, x, ny, nx, target, 0};
        }
    }
    return count;
}

// --- MODIFIED: 不再使用 side 判断九宫，而是读取 g_ctx ---
int gen_king_moves(const BoardState* state, int y, int x, Side side, Move* move_list) {
    int count = 0;
    const int K_MOVES[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    
    for (int i = 0; i < 4; ++i) {
        int ny = y + K_MOVES[i][0];
        int nx = x + K_MOVES[i][1];
        if (!is_inside(ny, nx)) continue;

        // --- MODIFIED: 使用 g_ctx 检查是否在九宫内 ---
        if (nx < 3 || nx > 5) continue;
        if (ny < g_ctx.palace_bottom || ny > g_ctx.palace_top) continue; // 动态Y坐标检查

        Piece target = state->board[ny][nx];
        if (target == EMPTY || get_piece_side(target) != side) {
            move_list[count++] = (Move){y, x, ny, nx, target, 0};
        }
    }

    // 飞将规则 (这段逻辑本身就是相对的，不需要修改)
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

// --- MODIFIED: 不再使用 side 判断方向，而是读取 g_ctx ---
int gen_pawn_moves(const BoardState* state, int y, int x, Side side, Move* move_list) {
    int count = 0;
    // --- MODIFIED: 使用 g_ctx "前进" 方向 ---
    int forward = g_ctx.forward_dir;

    // 向前走
    int ny = y + forward;
    if (is_inside(ny, x)) {
        Piece target = state->board[ny][x];
        if (target == EMPTY || get_piece_side(target) != side) {
            move_list[count++] = (Move){y, x, ny, x, target, 0};
        }
    }
    
    // --- MODIFIED: 使用 g_ctx "过河" 检查 ---
    int river_crossed;
    if (g_ctx.forward_dir == 1) { // 己方在下方, 前进是 +1
        river_crossed = (y >= g_ctx.river_cross_line); // river_cross_line = 5
    } else { // 己方在上方, 前进是 -1
        river_crossed = (y <= g_ctx.river_cross_line); // river_cross_line = 4
    }
    
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
// --- MODIFIED: 增加了 "找王" 和 "设置 g_ctx" 的预处理步骤 ---
int generate_moves(BoardState* state, Move legal_move_list[]) {
    Move pseudo_legal_moves[MAX_MOVES];
    int pseudo_count = 0;
    Side current_side = state->side_to_move;

    // --- NEW LOGIC START ---
    int king_y = -1;
    Piece king_piece = (current_side == RED) ? r_king : b_king;

    // 1. 第一次遍历: 找到我方将/帅以确定方向
    for (int y = 0; y < ROWS; ++y) {
        for (int x = 0; x < COLS; ++x) {
            if (state->board[y][x] == king_piece) {
                king_y = y;
                goto found_king; // 找到王，跳出双重循环
            }
        }
    }

found_king:
    if (king_y == -1) {
        return 0; // 棋盘上找不到王，没有合法走法
    }

    // 2. 根据王的位置，设置 file-scope 的 g_ctx 变量
    if (king_y < 5) { // 判定为下方 (e.g. 传统红方)
        g_ctx.forward_dir              = 1;
        g_ctx.palace_bottom            = 0;
        g_ctx.palace_top               = 2;
        g_ctx.river_cross_line         = 5; // y >= 5 算过河
        g_ctx.elephant_home_river_line = 4; // y <= 4 才算没过河
    } else { // 判定为上方 (e.g. 传统黑方)
        g_ctx.forward_dir              = -1;
        g_ctx.palace_bottom            = 7;
        g_ctx.palace_top               = 9;
        g_ctx.river_cross_line         = 4; // y <= 4 算过河
        g_ctx.elephant_home_river_line = 5; // y >= 5 才算没过河
    }
    // --- NEW LOGIC END ---


    // 3. 第二次遍历: (原)生成所有伪合法走法
    //    (函数调用签名不变，但它们内部会读取 g_ctx)
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

    // 4. (原)过滤伪合法走法，得到合法走法
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