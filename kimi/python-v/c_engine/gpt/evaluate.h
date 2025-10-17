// evaluate.h
#ifndef EVALUATE_H
#define EVALUATE_H

#include "board.h"

/* 初始化 PST / PIECE VALUES 等（在程序启动时调用一次） */
void eval_init(void);

/* 返回对黑方有利程度的评估分，越大对黑方越有利；与 Python evaluate_board 保持语义 */
int evaluate_board(const Board *board_state);

/* 辅助函数：在评估里可能需要的工具函数（在这里公开以便测试） */
KingPos find_general_pos(const Board *board, Side side);
int count_pieces(const Board *board, Side side, PieceType types[], int ntypes);

#endif // EVALUATE_H
