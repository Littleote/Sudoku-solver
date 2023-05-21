#ifndef EXPORTED_H_INCLUDED
#define EXPORTED_H_INCLUDED

#include "Sudoku.hpp"

Sudoku::Result dll_solve(const std::size_t blockX, const std::size_t blockY, Sudoku::Cell board[]);

#endif // EXPORTED_H_INCLUDED
