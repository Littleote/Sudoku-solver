# -*- coding: utf-8 -*-
"""
@author: david
"""

import os
import generate_digits as computer
import load_mnist as human


if __name__ == "__main__":
    try:
        os.mkdir('digits28')
    except FileExistsError:
        pass
    computer._main(28)
    try:
        os.mkdir('digits64')
    except FileExistsError:
        pass
    computer._main(64)
    human._main()
