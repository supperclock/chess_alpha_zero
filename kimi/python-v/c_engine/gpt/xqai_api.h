#ifndef XQAI_API_H
#define XQAI_API_H

#include "board.h"
#include "search.h"
#include "evaluate.h"
#include "zobrist.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------
 *  xqai_api.h - 中国象棋引擎外部接口头文件
 * ---------------------------------------------------------------
 *  提供给 Python (ctypes) 的导出函数声明。
 *  全部函数均为纯 C 接口，无需依赖 Python.h。
 * --------------------------------------------------------------- */

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

/* 初始化棋盘结构体 */
EXPORT void init_board(Board *b);

/* 调用 AI 搜索最佳走法
 * 参数：
 *   b_in         输入棋盘
 *   best_move    输出最佳走法
 *   depth        搜索深度
 *   time_limit   时间限制（毫秒）
 */
EXPORT void search_best(const Board *b_in, Move *best_move, int depth, int time_limit);

/* 调用评估函数 */
EXPORT int evaluate_board_c(const Board *b);

#ifdef __cplusplus
}
#endif

#endif /* XQAI_API_H */
