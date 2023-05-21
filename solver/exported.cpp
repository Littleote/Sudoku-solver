#include "exported.h"

Sudoku::Result dll_solve(const std::size_t blockX, const std::size_t blockY, Sudoku::Cell board[])
{
    auto sudoku = Sudoku(blockX, blockY);
    const std::size_t side = blockX * blockY;
    Sudoku::Cell* boardBegin = board, * boardEnd = board + side * side;
    Sudoku::Board sBoard(boardBegin, boardEnd);
    sudoku.setOriginalBoard(sBoard);
    auto result = sudoku.solve();
    sBoard = sudoku.getSolvedBoard();
    std::copy(sBoard.begin(), sBoard.end(), boardBegin);
    return result;
}
