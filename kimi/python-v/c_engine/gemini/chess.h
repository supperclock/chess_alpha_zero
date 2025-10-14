#ifndef CHESS_H
#define CHESS_H

// --- 常量定义 ---
#define ROWS 10
#define COLS 9
#define MATE_SCORE 1000000 // 将死分值
#define MAX_MOVES 256      // 一个局面最多可能的走法数量，用于预分配数组

// --- 枚举类型 (比字符串更高效) ---

// 使用整数表示棋子类型，更高效。
// 0表示空位，红方棋子用小写，黑方用大写，便于区分和计算。
typedef enum {
    EMPTY,
    r_king, r_advisor, r_elephant, r_horse, r_chariot, r_cannon, r_pawn,
    b_king, b_advisor, b_elephant, b_horse, b_chariot, b_cannon, b_pawn
} Piece;

// 阵营
typedef enum {
    RED, BLACK
} Side;

// --- 核心数据结构 ---

// 走法结构体 (Move)
// 存储走法的起点、终点、被吃掉的棋子（用于悔棋）和评分（用于排序）
typedef struct {
    int from_y, from_x;
    int to_y, to_x;
    Piece captured;
    int score;
} Move;

// 棋盘状态结构体 (BoardState)
// 包含一个棋局所需的所有信息
typedef struct {
    Piece board[ROWS][COLS]; // 棋盘布局
    Side side_to_move;       // 当前轮到哪一方走棋
    // 未来可以添加更多信息，如Zobrist键、历史记录等
    // uint64_t zobrist_key;
} BoardState;


// --- 函数原型声明 ---
// 提前声明我们将在其他文件中实现的函数，这样文件之间可以相互调用。

// 来自 board.c
void init_board_from_initial_setup(BoardState* state);
void print_board(const BoardState* state); // 用于调试
void make_move(BoardState* state, Move* move);
void unmake_move(BoardState* state, const Move* move);
int is_in_check(const BoardState* state, Side side);
int find_general(const BoardState* state, Side side, int* out_y, int* out_x);

// 来自 movegen.c
int generate_moves(BoardState* state, Move move_list[]);
// --- VVVV 在下方追加 VVVV ---
int gen_chariot_moves(const BoardState* state, int y, int x, Side side, Move* move_list);
int gen_horse_moves(const BoardState* state, int y, int x, Side side, Move* move_list);
int gen_cannon_moves(const BoardState* state, int y, int x, Side side, Move* move_list);
int gen_elephant_moves(const BoardState* state, int y, int x, Side side, Move* move_list);
int gen_advisor_moves(const BoardState* state, int y, int x, Side side, Move* move_list);
int gen_king_moves(const BoardState* state, int y, int x, Side side, Move* move_list);
int gen_pawn_moves(const BoardState* state, int y, int x, Side side, Move* move_list);


// 来自 evaluate.c
int evaluate_board(const BoardState* state);

// 来自 search.c
Move find_best_move(BoardState* state, int max_depth, double time_limit);




#endif // CHESS_H