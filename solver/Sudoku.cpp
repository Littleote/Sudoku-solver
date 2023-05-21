#include "Sudoku.hpp"

#define EMPTY 0

#include <iostream>

Sudoku::Sudoku(const std::size_t x, const std::size_t y) :
    mX(x),
    mY(y),
    mSide(mX * mY),
    mSize(mSide * mSide),
    mOriginal(0, mSize)
{}

Sudoku::Result Sudoku::solve() {
    /// Prepare necessary variables
    std::size_t index = 0;
    std::uint8_t solutions = 0;
    bool increase = true, searching = true;
    Sudoku::Cell limit = (Sudoku::Cell)mSide;
    Sudoku::Board solved = (mSolution = mOriginal);

    /// Assert initial board is valid
    for(std::size_t i = 0; i < mSize; i++)
        if(mSolution[i] != EMPTY)
            if(!isValid(mSolution, i)) {
                searching = false;
                break;
            }

    /// Search for a valid solution
    while(searching) {
        if(mOriginal[index] == EMPTY) {
            increase = false;
            for(mSolution[index]++; mSolution[index] <= limit; mSolution[index]++)
                if(isValid(mSolution, index)) {
                    increase = true;
                    break;
                }
        }
        if(increase) {
            index++;
            if(index >= mSize) {
                solutions++;
                index--;
                increase = false;
                solved = mSolution;
                if (solutions > 1)
                    searching = false;
            }
        } else {
            mSolution[index] = mOriginal[index];
            if(index == 0)
                searching = false;
            else
                index--;
        }
        #ifdef DEBUG
        std::cout << index << ' ' << (int)mSolution[index] << std::endl;
        #endif // DEBUG
    }

    #ifdef DEBUG
    std::cout << "Solutions: "<< (int)solutions << std::endl;
    std::cout << "Index: "<< (int)index << std::endl;
    std::cout << "Last movement: "<< (increase ? "forwards" : "backwards") << std::endl;
    #endif // DEBUG

    /// Return info of the number of solutions;
    mSolution = solved;
    switch(solutions) {
    case 0:
        return Sudoku::Result::NO_SOLUTION;
    case 1:
        return Sudoku::Result::ONE_SOLUTION;
    default:
        return Sudoku::Result::MULTI_SOLUTION;
    }
}

bool Sudoku::isValid(Board & board, const std::size_t & index) const {
    const std::size_t col = index % mSide, row = index / mSide;
    bool failed = false;

    /// Check column
    for(std::size_t r = 0; r < mSide; r++)
        failed |= (at(board, col, r) == board[index]) && (r != row);
    if(failed)
        return false;

    /// Check row
    for(std::size_t c = 0; c < mSide; c++)
        failed |= (at(board, c, row) == board[index]) && (c != col);
    if(failed)
        return false;

    /// Check box
    const std::size_t boxX = col / mX * mX, boxY = row / mY * mY;
    for(std::size_t r = 0; r < mY; r++)
        for(std::size_t c = 0; c < mX; c++)
            failed |= (at(board, boxX + c, boxY + r) == board[index]) && (boxY + r != row) && (boxX + c != col);
    if(failed)
        return false;
    return true;
}
