// nnue_core.c
#include "nnue_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// --- 内部定义 (必须与 .h 文件匹配) ---
#define NNUE_ACCUM_SIZE 256
#define NNUE_HIDDEN2_SIZE 32

// --- 特征维度 (HGP 架构) ---
// (Piece * PieceSquare)
#define NNUE_PIECE_SQUARE_DIM (PIECE_TYPE_NB * (ROWS * COLS)) // 15 * 90 = 1350
// (KingSquare * Piece * PieceSquare)
#define NNUE_FEATURE_DIM ( (ROWS * COLS) * NNUE_PIECE_SQUARE_DIM ) // 90 * 1350 = 121,500

// --- 量化和激活 (必须与训练脚本匹配!) ---
// CReLU (Clipped ReLU) 的上下限
#define CRELU_MIN 0
#define CRELU_MAX 255
// 累加器 -> 隐藏层 1 的量化缩小因子
#define QA_1 64
// 隐藏层 1 -> 隐藏层 2 的量化缩小因子
#define QA_2 64
// 输出层 (W3) 的权重缩放 (评估值 = sum / NNUE_OUTPUT_SCALE)
#define NNUE_OUTPUT_SCALE 16


// --- 全局网络权重 (由 nnue_init 加载) ---
static bool g_initialized = false;

// 第 1 层 (输入 -> 累加器)
// [2][FEATURE_DIM][ACCUM_SIZE] (红/黑视角)
static int16_t* g_W1 = NULL; 
// [2][ACCUM_SIZE] (红/黑偏置)
static int16_t* g_B1 = NULL;

// 第 2 层 (累加器*2 -> 隐藏层2)
// [ACCUM_SIZE * 2][HIDDEN2_SIZE]
static int16_t* g_W2 = NULL;
// [HIDDEN2_SIZE]
static int16_t* g_B2 = NULL;

// 第 3 层 (隐藏层2 -> 输出)
// [HIDDEN2_SIZE][1]
static int16_t* g_W3 = NULL;
// [1]
static int32_t g_B3 = 0;


// --- 内部辅助函数 ---

/**
 * @brief 辅助函数：安全地分配和读取文件块
 */
static void* read_chunk(FILE* f, size_t size) {
    void* ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "NNUE: Failed to allocate %zu bytes\n", size);
        return NULL;
    }
    if (fread(ptr, 1, size, f) != size) {
        fprintf(stderr, "NNUE: Failed to read %zu bytes from file\n", size);
        free(ptr);
        return NULL;
    }
    return ptr;
}

/**
 * @brief 辅助函数：获取 HGP 特征索引
 * 这是特征转换器的核心。
 */
static inline int get_feature_index(int k_y, int k_x, Piece p, int p_y, int p_x) {
    assert(p > EMPTY && p < PIECE_TYPE_NB);
    assert(k_y >= 0 && k_y < ROWS && k_x >= 0 && k_x < COLS);
    assert(p_y >= 0 && p_y < ROWS && p_x >= 0 && p_x < COLS);
    
    int k_sq = k_y * COLS + k_x;
    int p_sq = p_y * COLS + p_x;
    int p_id = (int)p; // 假设 Piece enum (1-14)

    // (KingPos * PieceDim) + (PieceID * SquareDim) + PieceSquare
    int piece_square_idx = (p_id * (ROWS * COLS)) + p_sq;
    int feature_index = (k_sq * NNUE_PIECE_SQUARE_DIM) + piece_square_idx;
    
    assert(feature_index >= 0 && feature_index < NNUE_FEATURE_DIM);
    return feature_index;
}

/**
 * @brief 辅助函数：获取指向 W1 (输入层) 权重的指针
 */
static inline const int16_t* get_w1_weights(Side king_perspective, int feature_index) {
    // king_perspective: 0=RED, 1=BLACK
    size_t offset = ((size_t)king_perspective * NNUE_FEATURE_DIM + feature_index) * NNUE_ACCUM_SIZE;
    return &g_W1[offset];
}

/**
 * @brief 辅助函数：向量加法 (acc += weights)
 */
static inline void add_vec(int16_t* acc, const int16_t* weights) {
    for (int i = 0; i < NNUE_ACCUM_SIZE; ++i) {
        acc[i] += weights[i];
    }
}

/**
 * @brief 辅助函数：向量减法 (acc -= weights)
 */
static inline void sub_vec(int16_t* acc, const int16_t* weights) {
    for (int i = 0; i < NNUE_ACCUM_SIZE; ++i) {
        acc[i] -= weights[i];
    }
}

/**
 * @brief 辅助函数：CReLU 激活
 */
static inline int16_t crelu(int32_t x) {
    if (x < (int32_t)CRELU_MIN) return (int16_t)CRELU_MIN;
    if (x > (int32_t)CRELU_MAX) return (int16_t)CRELU_MAX;
    return (int16_t)x;
}


// --- API 实现 ---

bool nnue_init(const char* file_path) {
    if (g_initialized) {
        fprintf(stderr, "NNUE: Already initialized.\n");
        return true;
    }

    FILE* f = fopen(file_path, "rb");
    if (!f) {
        fprintf(stderr, "NNUE: Failed to open file: %s\n", file_path);
        return false;
    }

    // (TODO: 在此添加 Magic Number/版本检查)
    // uint32_t magic;
    // fread(&magic, sizeof(magic), 1, f);
    // if (magic != 0xABCDEF) { ... }
    
    // 假设文件格式为：
    // 1. B1 (2 * ACCUM_SIZE * sizeof(int16_t))
    // 2. W1 (2 * FEATURE_DIM * ACCUM_SIZE * sizeof(int16_t))
    // 3. B2 (HIDDEN2_SIZE * sizeof(int16_t))
    // 4. W2 (ACCUM_SIZE * 2 * HIDDEN2_SIZE * sizeof(int16_t))
    // 5. B3 (1 * sizeof(int32_t))
    // 6. W3 (HIDDEN2_SIZE * 1 * sizeof(int16_t))

    g_B1 = read_chunk(f, 2 * NNUE_ACCUM_SIZE * sizeof(int16_t));
    if (!g_B1) goto error;
    
    g_W1 = read_chunk(f, (size_t)2 * NNUE_FEATURE_DIM * NNUE_ACCUM_SIZE * sizeof(int16_t));
    if (!g_W1) goto error;

    g_B2 = read_chunk(f, NNUE_HIDDEN2_SIZE * sizeof(int16_t));
    if (!g_B2) goto error;

    g_W2 = read_chunk(f, (size_t)NNUE_ACCUM_SIZE * 2 * NNUE_HIDDEN2_SIZE * sizeof(int16_t));
    if (!g_W2) goto error;

    if (fread(&g_B3, sizeof(int32_t), 1, f) != 1) goto error;
    
    g_W3 = read_chunk(f, NNUE_HIDDEN2_SIZE * sizeof(int16_t));
    if (!g_W3) goto error;

    // (TODO: 检查是否已到文件末尾)
    
    fclose(f);
    g_initialized = true;
    printf("NNUE: Network loaded successfully.\n");
    return true;

error:
    fprintf(stderr, "NNUE: Error loading network file.\n");
    fclose(f);
    nnue_cleanup(); // 释放已分配的部分
    return false;
}

void nnue_cleanup(void) {
    free(g_W1); g_W1 = NULL;
    free(g_B1); g_B1 = NULL;
    free(g_W2); g_W2 = NULL;
    free(g_B2); g_B2 = NULL;
    free(g_W3); g_W3 = NULL;
    g_B3 = 0;
    g_initialized = false;
}

void nnue_refresh_accumulator(const BoardState* state, NnueAccumulator* acc) {
    assert(g_initialized);
    
    int rky, rkx, bky, bkx;
    if (!find_general(state, RED, &rky, &rkx) || !find_general(state, BLACK, &bky, &bkx)) {
        // 局面非法, 但我们不能崩溃
        memset(acc, 0, sizeof(NnueAccumulator)); 
        return;
    }

    // 1. 复制偏置 (Bias)
    memcpy(acc->white_acc, &g_B1[0 * NNUE_ACCUM_SIZE], NNUE_ACCUM_SIZE * sizeof(int16_t));
    memcpy(acc->black_acc, &g_B1[1 * NNUE_ACCUM_SIZE], NNUE_ACCUM_SIZE * sizeof(int16_t));

    // 2. 累加所有棋子的权重
    for (int y = 0; y < ROWS; ++y) {
        for (int x = 0; x < COLS; ++x) {
            Piece p = state->board[y][x];
            if (p == EMPTY) continue;
            
            // 获取此棋子 (p, y, x) 分别在红王和黑王视角下的特征索引
            int white_idx = get_feature_index(rky, rkx, p, y, x);
            int black_idx = get_feature_index(bky, bkx, p, y, x);

            // 累加
            add_vec(acc->white_acc, get_w1_weights(RED, white_idx));
            add_vec(acc->black_acc, get_w1_weights(BLACK, black_idx));
        }
    }
}

void nnue_pop_piece(const BoardState* state, NnueAccumulator* acc, Piece p, int y, int x) {
    assert(g_initialized);
    if (p == EMPTY) return;
    
    int rky, rkx, bky, bkx;
    // (我们假设王没有移动, 所以王的位置是固定的)
    find_general(state, RED, &rky, &rkx);
    find_general(state, BLACK, &bky, &bkx);
    
    int white_idx = get_feature_index(rky, rkx, p, y, x);
    int black_idx = get_feature_index(bky, bkx, p, y, x);

    // 减去
    sub_vec(acc->white_acc, get_w1_weights(RED, white_idx));
    sub_vec(acc->black_acc, get_w1_weights(BLACK, black_idx));
}

void nnue_push_piece(const BoardState* state, NnueAccumulator* acc, Piece p, int y, int x) {
    assert(g_initialized);
    if (p == EMPTY) return;

    int rky, rkx, bky, bkx;
    // (我们假设王没有移动, 所以王的位置是固定的)
    find_general(state, RED, &rky, &rkx);
    find_general(state, BLACK, &bky, &bkx);

    int white_idx = get_feature_index(rky, rkx, p, y, x);
    int black_idx = get_feature_index(bky, bkx, p, y, x);
    
    // 加上
    add_vec(acc->white_acc, get_w1_weights(RED, white_idx));
    add_vec(acc->black_acc, get_w1_weights(BLACK, black_idx));
}

int nnue_evaluate(Side to_move, const NnueAccumulator* acc) {
    if (!g_initialized) {
        // 如果网络未加载, 返回 0 (退化为无评估)
        return 0;
    }

    // --- 第 1 层 -> 第 2 层 (CReLU + 拼接) ---
    
    // 512-dim 输入向量
    int16_t layer2_input[NNUE_ACCUM_SIZE * 2];
    
    // CReLU (L1 激活) 并复制到 layer2_input
    for (int i = 0; i < NNUE_ACCUM_SIZE; ++i) {
        layer2_input[i] = crelu((int32_t)acc->white_acc[i]);
    }
    for (int i = 0; i < NNUE_ACCUM_SIZE; ++i) {
        layer2_input[NNUE_ACCUM_SIZE + i] = crelu((int32_t)acc->black_acc[i]);
    }

    // --- 第 2 层 -> 第 3 层 (隐藏层) ---
    
    // 32-dim 输入向量
    int16_t layer3_input[NNUE_HIDDEN2_SIZE];
    
    for (int i = 0; i < NNUE_HIDDEN2_SIZE; ++i) {
        // (W2 是 [512][32])
        int32_t sum = g_B2[i] * QA_1; // 偏置需要预先缩放

        for (int j = 0; j < NNUE_ACCUM_SIZE * 2; ++j) {
            // (W2 的内存布局是 [j][i])
            size_t w2_idx = (size_t)j * NNUE_HIDDEN2_SIZE + i;
            sum += (int32_t)layer2_input[j] * g_W2[w2_idx];
        }
        
        // 量化 + CReLU
        layer3_input[i] = crelu(sum / QA_1);
    }
    
    // --- 第 3 层 -> 输出 (评估) ---
    
    // (W3 是 [32][1])
    int32_t final_score = g_B3; // g_B3 已经是缩放后的
    
    for (int i = 0; i < NNUE_HIDDEN2_SIZE; ++i) {
        final_score += (int32_t)layer3_input[i] * g_W3[i];
    }
    
    // (最终的缩放因子是 QA_2 * NNUE_OUTPUT_SCALE, 
    //  但 B3 和 W3 通常是预先缩放过的)
    // 假设 final_score 已经是 "Pawn" 单位
    // 并且 W3 和 B3 已经包含了 QA_2 和 OUTPUT_SCALE
    
    // (假设网络总是从 RED 的视角训练的)
    if (to_move == RED) {
        return (int)final_score;
    } else {
        return (int)-final_score;
    }
}