#ifndef EVALUATE_H
#define EVALUATE_H

#include "board.h"

/* ============================================================
 * evaluate.h — 棋盘评估模块（C 语言版本）
 * ------------------------------------------------------------
 * 提供：
 *   int evaluate_board(const Board *board_state);
 *
 * 功能：
 *   - 基于子力价值与位置表 (PST)
 *   - 炮马配合加分
 *   - 兵形结构惩罚
 *   - 车连线加分
 *   - 王安全惩罚
 *
 * 注意：
 *   - 所有函数纯计算，不修改棋盘
 *   - 返回值为相对分值（>0 表示红方优势，<0 表示黑方优势）
 * ============================================================ */

/* ---- 常量定义 ---- */
#define BONUS_CANNON_HORSE   60    /* 炮马组合加分 */
#define BONUS_PAWN_STRUCTURE 20    /* 兵形结构惩罚（同列） */
#define BONUS_ROOK_CONNECTION 40   /* 车连线加分 */
#define PENALTY_WEAK_KING    80    /* 王安全惩罚 */

/* ---- 主接口 ---- */
int evaluate_board(const Board *board_state);

#endif /* EVALUATE_H */
