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