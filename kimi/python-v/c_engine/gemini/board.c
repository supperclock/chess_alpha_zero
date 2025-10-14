#include <stdio.h>
#include "chess.h"

// 1. C 语言版本的棋盘初始布局
// 使用 static const 确保这个数据是只读的，并且只在当前文件内可见。
static const Piece INITIAL_BOARD[ROWS][COLS] = {
    {r_chariot, r_horse, r_elephant, r_advisor, r_king, r_advisor, r_elephant, r_horse, r_chariot},
    {EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY},
    {EMPTY, r_cannon, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, r_cannon, EMPTY},
    {r_pawn, EMPTY, r_pawn, EMPTY, r_pawn, EMPTY, r_pawn, EMPTY, r_pawn},
    {EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY},
    {EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY},
    {b_pawn, EMPTY, b_pawn, EMPTY, b_pawn, EMPTY, b_pawn, EMPTY, b_pawn},
    {EMPTY, b_cannon, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, b_cannon, EMPTY},
    {EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY},
    {b_chariot, b_horse, b_elephant, b_advisor, b_king, b_advisor, b_elephant, b_horse, b_chariot}
};

// 2. 初始化棋盘状态的函数
void init_board_from_initial_setup(BoardState* state) {
    for (int y = 0; y < ROWS; y++) {
        for (int x = 0; x < COLS; x++) {
            state->board[y][x] = INITIAL_BOARD[y][x];
        }
    }
    state->side_to_move = RED; // 红方先行
}

// 3. 执行走法的函数
// 这个函数会修改棋盘状态，并记录被吃的棋子到 move 结构体中。
void make_move(BoardState* state, Move* move) {
    // 记录被吃的棋子
    move->captured = state->board[move->to_y][move->to_x];

    // 移动棋子
    state->board[move->to_y][move->to_x] = state->board[move->from_y][move->from_x];
    state->board[move->from_y][move->from_x] = EMPTY;

    // 交换走棋方
    state->side_to_move = (state->side_to_move == RED) ? BLACK : RED;
}

// 4. 撤销走法的函数 (悔棋)
// 这个函数对于搜索算法至关重要，它能高效地将棋盘恢复到上一步的状态。
void unmake_move(BoardState* state, const Move* move) {
    // 将移动的棋子放回原位
    state->board[move->from_y][move->from_x] = state->board[move->to_y][move->to_x];
    // 将被吃的棋子放回目标位置
    state->board[move->to_y][move->to_x] = move->captured;

    // 交换回原来的走棋方
    state->side_to_move = (state->side_to_move == RED) ? BLACK : RED;
}


// 5. 查找将/帅位置的函数
// C 语言没有多返回值，所以我们使用指针作为输出参数。
// 返回 1 表示找到，0 表示未找到。
int find_general(const BoardState* state, Side side, int* out_y, int* out_x) {
    Piece king_to_find = (side == RED) ? r_king : b_king;
    for (int y = 0; y < ROWS; y++) {
        for (int x = 0; x < COLS; x++) {
            if (state->board[y][x] == king_to_find) {
                *out_y = y; // 将 y 坐标存入 out_y 指向的地址
                *out_x = x; // 将 x 坐标存入 out_x 指向的地址
                return 1; // 找到
            }
        }
    }
    return 0; // 未找到
}


// --- 调试用函数 ---

// 辅助函数：将 Piece 枚举转换为字符用于打印
const char* piece_to_char(Piece p) {
    switch(p) {
        case EMPTY: return ".";
        case r_king: return "K"; case b_king: return "k";
        case r_advisor: return "A"; case b_advisor: return "a";
        case r_elephant: return "E"; case b_elephant: return "e";
        case r_horse: return "H"; case b_horse: return "h";
        case r_chariot: return "R"; case b_chariot: return "r";
        case r_cannon: return "C"; case b_cannon: return "c";
        case r_pawn: return "P"; case b_pawn: return "p";
        default: return "?";
    }
}

// 打印当前棋盘状态
void print_board(const BoardState* state) {
    printf("\n  a b c d e f g h i\n");
    printf(" +-------------------\n");
    for (int y = 0; y < ROWS; y++) {
        printf("%d|", y);
        for (int x = 0; x < COLS; x++) {
            printf("%s ", piece_to_char(state->board[y][x]));
        }
        printf("\n");
    }
    printf("Side to move: %s\n", (state->side_to_move == RED) ? "RED" : "BLACK");
}

/**
 * @brief 检查指定方是否被将军
 *
 * @param state 当前棋盘状态
 * @param side  要检查的一方 (e.g., 检查 RED 是否被将军)
 * @return int  1 表示被将军, 0 表示没有被将军
 */
int is_in_check(const BoardState* state, Side side) {
    int ky, kx;
    // 1. 找到己方将/帅的位置
    if (!find_general(state, side, &ky, &kx)) {
        return 1; // 如果找不到王，视作已经被吃，返回“被将军”状态
    }

    Side opponent_side = (side == RED) ? BLACK : RED;
    Piece opp_pawn = (opponent_side == RED) ? r_pawn : b_pawn;
    Piece opp_chariot = (opponent_side == RED) ? r_chariot : b_chariot;
    Piece opp_cannon = (opponent_side == RED) ? r_cannon : b_cannon;
    Piece opp_horse = (opponent_side == RED) ? r_horse : b_horse;
    Piece opp_king = (opponent_side == RED) ? r_king : b_king;

    // 2. 检查直线攻击 (车、炮、将、兵)
    const int DIRS[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (int i = 0; i < 4; ++i) {
        int screen = 0; // 用于炮的计数器
        for (int step = 1; step < 10; ++step) {
            int ny = ky + DIRS[i][0] * step;
            int nx = kx + DIRS[i][1] * step;
            if (!is_inside(ny, nx)) break;

            Piece p = state->board[ny][nx];
            if (p != EMPTY) {
                if (screen == 0) { // 遇到的第一个棋子
                    if (p == opp_chariot || p == opp_king) return 1; // 被对方车或王直接将军
                    screen = 1; // 作为炮架
                } else { // 隔着一个炮架
                    if (p == opp_cannon) return 1; // 被对方炮将军
                    break; // 被第二个棋子挡住，此方向安全
                }
            }
        }
    }

    // 3. 检查兵/卒的攻击
    int pawn_forward = (side == RED) ? 1 : -1;
    // 检查来自前方的卒
    if (is_inside(ky - pawn_forward, kx) && state->board[ky - pawn_forward][kx] == opp_pawn) return 1;
    // 检查来自侧翼的卒
    if (is_inside(ky, kx - 1) && state->board[ky][kx - 1] == opp_pawn) return 1;
    if (is_inside(ky, kx + 1) && state->board[ky][kx + 1] == opp_pawn) return 1;


    // 4. 检查马的攻击
    const int H_MOVES[8][2] = {{2, 1}, {2, -1}, {-2, 1}, {-2, -1}, {1, 2}, {1, -2}, {-1, 2}, {-1, -2}};
    const int LEG_MOVES[8][2] = {{1, 0}, {1, 0}, {-1, 0}, {-1, 0}, {0, 1}, {0, -1}, {0, 1}, {0, -1}};
    for (int i = 0; i < 8; ++i) {
        // 检查马腿是否被别住
        if (state->board[ky + LEG_MOVES[i][0]][kx + LEG_MOVES[i][1]] != EMPTY) continue;
        
        int ny = ky + H_MOVES[i][0];
        int nx = kx + H_MOVES[i][1];
        if (is_inside(ny, nx) && state->board[ny][nx] == opp_horse) {
            return 1; // 被马将军
        }
    }

    return 0; // 所有检查都通过，没有被将军
}