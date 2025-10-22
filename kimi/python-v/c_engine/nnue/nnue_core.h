// nnue_core.h
#pragma once

#include "chess.h"
#include <stdint.h>
#include <stdbool.h>

// --- NNUE 架构参数 (必须与训练好的网络文件匹配!) ---

// 累加器大小 (第一层隐藏层)
#define NNUE_ACCUM_SIZE 256
// 第二层隐藏层大小
#define NNUE_HIDDEN2_SIZE 32

// --- 数据结构 ---

/**
 * @brief NNUE 累加器 (第一层的激活值)
 *
 * 这是 HGP 架构的核心:
 * - white_acc: 从 "红王" 视角看, 棋盘上所有棋子特征的激活总和
 * - black_acc: 从 "黑王" 视角看, 棋盘上所有棋子特征的激活总和
 *
 * 搜索器 (Search) 需要在其状态堆栈中为每个节点存储这个结构。
 */
typedef struct {
    int16_t white_acc[NNUE_ACCUM_SIZE];
    int16_t black_acc[NNUE_ACCUM_SIZE];
} NnueAccumulator;


// --- API 函数 ---

/**
 * @brief 加载 NNUE 网络权重
 * @param file_path 指向 .nnue 权重文件的路径
 * @return 成功返回 true, 失败 (文件未找到, 格式错误等) 返回 false
 */
bool nnue_init(const char* file_path);

/**
 * @brief 释放所有 NNUE 相关的内存
 * (在程序退出时调用)
 */
void nnue_cleanup(void);

/**
 * @brief 从头计算一个局面的累加器 (慢)
 *
 * 仅用于:
 * 1. FEN 串设置新局面
 * 2. "将" (King) 发生移动后, 必须调用此函数刷新 (因为所有特征都变了)
 *
 * @param state 完整的棋盘状态
 * @param acc (出参) 将被填充的累加器
 */
void nnue_refresh_accumulator(const BoardState* state, NnueAccumulator* acc);

/**
 * @brief 增量更新: 从累加器中 "移除" 一个棋子
 *
 * 在 make_move 中, 在棋子移动 *之前* 调用。
 * 必须在 `state` 尚未改变时调用, 因为需要 "将" 的旧位置。
 *
 * @param state 棋盘状态 (移除前)
 * @param acc (入/出参) 要更新的累加器
 * @param p 要移除的棋子 (e.g., r_pawn)
 * @param y, x 棋子的位置
 */
void nnue_pop_piece(const BoardState* state, NnueAccumulator* acc, Piece p, int y, int x);

/**
 * @brief 增量更新: 向累加器中 "添加" 一个棋子
 *
 * 在 make_move 中, 在棋子移动 *之后* 调用。
 * 必须在 `state` *已经* 改变后调用吗? 不, 为了对称性, 
 * 我们假设 `state` 仍然是旧的 (为了获取王的位置)。
 * * ... 让我们重新设计一下 ...
 *
 * 为了简化, pop 和 push 都使用 *调用时* 的 "将" 的位置。
 * 这意味着 `make_move` 必须:
 * 1. pop(old_state, piece, from_y, from_x)
 * 2. pop(old_state, captured_piece, to_y, to_x) // 如果有吃子
 * 3. // ... 更新棋盘 ...
 * 4. push(new_state, piece, to_y, to_x)
 *
 * 如果 "将" 移动了, 1 和 2 使用 old_king_pos, 4 使用 new_king_pos,
 * 这太复杂了。
 *
 * **最终设计 (标准设计):**
 * 当 "将" 移动时, *必须* 调用 `nnue_refresh_accumulator`。
 * 当 "将" *不* 移动时, `state` (王的位置) 不变, 
 * pop 和 push 可以安全调用。
 */
void nnue_push_piece(const BoardState* state, NnueAccumulator* acc, Piece p, int y, int x);


/**
 * @brief 评估函数 (快)
 *
 * 获取已更新的累加器, 运行隐藏层 (e.g., 512 -> 32 -> 1), 
 * 并返回一个分数。
 *
 * @param to_move 当前轮到谁走
 * @param acc 包含当前局面激活值的累加器
 * @return 评估分数 (以兵 (pawn) 为单位, 正分利于红方)。
 */
int nnue_evaluate(Side to_move, const NnueAccumulator* acc);