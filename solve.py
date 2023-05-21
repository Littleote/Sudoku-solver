# -*- coding: utf-8 -*-
"""
@author: david
"""

import ctypes as cpp
import numpy as np
from os import path


def _as_c_uint8_array(array):
    """
    Change Numpy array to ctypes uint8 Array
    """
    list_ = list(array)
    return (len(list_) * cpp.c_uint8)(*list_)


def _as_list(c_array):
    """
    Change ctypes uint8 Array to list
    """
    return list(c_array)


RESULT = {
    -1: "NO_SOLUTION",
    0: "ONE_SOLUTION",
    1: "MULTI_SOLUTION",
}
solver = cpp.CDLL(path.join(path.dirname(__file__),
                  'solver/bin/Release/solver.dll'))


def solve(board, shape=(3, 3)):
    """
    Solve board using cpp dll
    """
    c_board = _as_c_uint8_array(board.flatten())
    result = solver.solve(shape[0], shape[1], c_board)
    board = np.array(_as_list(c_board)).reshape(board.shape)
    return board, RESULT[result]


def _main():
    board = np.zeros((9, 9), dtype=int)
    board, result = solve(board)
    board.reshape(9, 9)
    print(board, result)


if __name__ == '__main__':
    _main()
