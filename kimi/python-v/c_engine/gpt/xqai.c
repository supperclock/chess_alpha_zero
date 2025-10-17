#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "board.h"
#include "search.h"
#include "zobrist.h"
#include "rules.h"
#include "movegen.h"
#include "evaluate.h"

/* 转换 Python board_state -> C Board */
static int py_to_board(PyObject *py_board, const char *side_str, Board *out) {
    if (!PyList_Check(py_board)) return 0;
    memset(out, 0, sizeof(Board));
    out->side_to_move = strcmp(side_str, "black") == 0 ? SIDE_BLACK : SIDE_RED;

    for (int y = 0; y < ROWS; y++) {
        PyObject *row = PyList_GetItem(py_board, y);
        if (!PyList_Check(row)) continue;
        for (int x = 0; x < COLS; x++) {
            PyObject *cell = PyList_GetItem(row, x);
            if (cell == Py_None || cell == NULL) continue;
            PyObject *type_obj = PyDict_GetItemString(cell, "type");
            PyObject *side_obj = PyDict_GetItemString(cell, "side");
            if (!type_obj || !side_obj) continue;
            const char *t = PyUnicode_AsUTF8(type_obj);
            const char *s = PyUnicode_AsUTF8(side_obj);
            PieceType ptype = PT_NONE;
            if (strstr(t, "车") || strstr(t, "俥")) ptype = PT_ROOK;
            else if (strstr(t, "炮") || strstr(t, "砲")) ptype = PT_CANNON;
            else if (strstr(t, "马") || strstr(t, "傌")) ptype = PT_HORSE;
            else if (strstr(t, "相") || strstr(t, "象")) ptype = PT_ELEPHANT;
            else if (strstr(t, "仕") || strstr(t, "士")) ptype = PT_ADVISOR;
            else if (strstr(t, "帥") || strstr(t, "將")) ptype = PT_GENERAL;
            else if (strstr(t, "兵") || strstr(t, "卒")) ptype = PT_PAWN;
            Side pside = strcmp(s, "black") == 0 ? SIDE_BLACK : SIDE_RED;
            out->sq[y][x] = (Piece){ptype, pside};
        }
    }
    return 1;
}

/* Python 接口函数 */
static PyObject* py_minimax_root(PyObject *self, PyObject *args, PyObject *kwds) {
    PyObject *py_board;
    const char *side_str;
    int time_limit = 3000;
    static char *kwlist[] = {"board_state", "side", "time_limit", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Os|i", kwlist, &py_board, &side_str, &time_limit))
        return NULL;

    Board board;
    if (!py_to_board(py_board, side_str, &board)) {
        PyErr_SetString(PyExc_ValueError, "Invalid board_state format");
        return NULL;
    }

    zobrist_init();
    search_init();

    Move best = search_root(&board, 6, time_limit);

    PyObject *move_dict = Py_BuildValue(
        "{s:{s:i,s:i}, s:{s:i,s:i}, s:i}",
        "from", "y", best.fy, "x", best.fx,
        "to", "y", best.ty, "x", best.tx,
        "score", best.score
    );

    return move_dict;
}

/* 模块方法表 */
static PyMethodDef XqaiMethods[] = {
    {"minimax_root", (PyCFunction)py_minimax_root, METH_VARARGS | METH_KEYWORDS, "Run minimax search on board"},
    {NULL, NULL, 0, NULL}
};

/* 模块定义 */
static struct PyModuleDef xqaimodule = {
    PyModuleDef_HEAD_INIT,
    "xqai",
    "Chinese Chess AI (C backend)",
    -1,
    XqaiMethods
};

/* 初始化函数 */
PyMODINIT_FUNC PyInit_xqai(void) {
    return PyModule_Create(&xqaimodule);
}
