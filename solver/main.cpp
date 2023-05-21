#include "main.h"
#include "exported.h"
#include "Sudoku.hpp"

#include <algorithm>

/// Solve the Sudoku board made of blocks of x by y
int DLL_EXPORT solve(int x, int y, Sudoku::Cell board[])
{
    return (int)dll_solve(x, y, board);
}

#ifdef DEBUG
extern "C" DLL_EXPORT BOOL APIENTRY DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved)
{
    switch (fdwReason)
    {
    case DLL_PROCESS_ATTACH:
        MessageBoxA(0, "Process attached", "Solver", MB_OK | MB_ICONINFORMATION);
        break;

    case DLL_PROCESS_DETACH:
        MessageBoxA(0, "Process detached", "Solver", MB_OK | MB_ICONINFORMATION);
        break;

    case DLL_THREAD_ATTACH:
        MessageBoxA(0, "Thread attached", "Solver", MB_OK | MB_ICONINFORMATION);
        break;

    case DLL_THREAD_DETACH:
        MessageBoxA(0, "Thread detached", "Solver", MB_OK | MB_ICONINFORMATION);
        break;
    }
    return TRUE;
}
#endif
