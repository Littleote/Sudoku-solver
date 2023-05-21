#include <iostream>

#include "exported.h"

using namespace std;

int main() {
    const int blockX = 3, blockY = 3;
    const size_t side = blockX * blockY, size = side * side;
    unsigned char board [size];
    for(size_t i = 0; i < size; i++)
        board[i] = 0;
    auto result = dll_solve(blockX, blockY, board);
    cout << endl << (int)result << endl;
    for(size_t i = 0; i < size; i++)
        cout << (int)board[i] << (i % side == side - 1 ? '\n' : ' ');
}
