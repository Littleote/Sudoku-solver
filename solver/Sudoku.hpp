#ifndef SUDOKU_H
#define SUDOKU_H

#include <cstdint>
#include <vector>
#include <assert.h>

class Sudoku {
    public:

        typedef std::uint8_t Cell;
        typedef std::vector<Cell> Board;

        enum Result : std::int8_t {
                NO_SOLUTION = -1,
                SOLVED = 0,
                ONE_SOLUTION = 0,
                MULTI_SOLUTION = 1,
        };

        Sudoku(const std::size_t x, const std::size_t y);

        inline void setOriginalBoard(const Board & board) {
            mOriginal = board;
            assert(mOriginal.size() == mSize); /// Board size doesn't match with the specified box size
        }
        inline const Board & getOriginalBoard() const noexcept {
            return mOriginal;
        }
        inline const Board & getSolvedBoard() const noexcept {
            return mSolution;
        }

        Result solve();

    private:
        inline Cell& at(Board & board, const std::size_t & x, const std::size_t & y) const {
            return board[x + y * mSide];
        }

        bool isValid(Board & board, const std::size_t & index) const;

        const std::size_t mX, mY, mSide, mSize;

        Board mOriginal;
        Board mSolution;

};

#endif // SUDOKU_H
