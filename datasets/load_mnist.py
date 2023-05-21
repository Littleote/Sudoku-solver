# -*- coding: utf-8 -*-
"""
@author: david
"""

from torchvision import datasets


def _main():
    datasets.MNIST("", train=True, download=True)
    datasets.MNIST("", train=False, download=True)


if __name__ == "__main__":
    _main()
