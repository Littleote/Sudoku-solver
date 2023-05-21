# -*- coding: utf-8 -*-
"""
@author: david
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils

CATEGORIES = [
    "initial",
    "solved",
    "invalid",
    "written",
    "none",
]

NOTES = [
    "right",
    "wrong",
    "maybe",
    "none",
]


def display_solution(image, coordinates, values, categories=None, notes=None):
    """
    Display image with the values ovrelaid acording to the categories and notes
    """
    if categories is None:
        categories = np.full((9, 9), CATEGORIES[-1], dtype=object)
    if notes is None:
        notes = np.full((9, 9), CATEGORIES[-1], dtype=object)
    final_shape = [1000, 1000]
    dlt = utils.square_dlt(
        coordinates[[0, 0, -1, -1], [0, -1, 0, -1]], final_shape)
    coords = utils.forward_dlt(dlt, coordinates)
    upper, lower = np.max(coords, axis=(0, 1)), np.min(coords, axis=(0, 1))
    limits = np.array([lower, [upper[0], lower[1]],
                      [lower[0], upper[1]], upper])
    limits = utils.backwards_dlt(dlt, limits)
    dlt = utils.square_dlt(limits, final_shape)
    coords = utils.forward_dlt(dlt, coordinates)
    sudoku = cv2.warpPerspective(image, dlt, final_shape)
    scale = np.sort(sudoku.flatten())
    low, high = scale[1000], scale[-1000]
    sudoku = (sudoku - low) / (high - low)
    sudoku = np.minimum(np.maximum(sudoku, 0), 1)
    sudoku = np.repeat(sudoku, 3).reshape(final_shape + [3])

    plt.figure(figsize=(5, 5))
    plt.axis('off')
    plt.imshow(sudoku)
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    for i in range(9):
        for j in range(9):
            corners = coords[[i, i + 1, i + 1, i], [j, j, j + 1,  j + 1]]
            center = np.mean(corners, axis=0)
            weight = 0.95
            color_shift = 0.5
            corners = weight * corners + (1 - weight) * center
            if categories[i, j] == CATEGORIES[0]:
                if notes[i, j] == NOTES[2]:
                    color = np.array([1, 1, 0])
                    symbol = "?"
                else:
                    color = np.array([0, 0, 0])
                    symbol = ""
                plt.fill(*corners.T, color=color, alpha=0.3, linewidth=2)
                plt.text(*(center + corners[3]) / 2, s=values[i, j],
                         color=color * color_shift, alpha=0.8,
                         ha='center', va='center', fontsize='large')
                plt.text(*(center + corners[0]) / 2, s=symbol,
                         color=color * color_shift, alpha=0.8,
                         ha='center', va='center', fontsize='large')
            elif categories[i, j] == CATEGORIES[1]:
                color = np.array([0, 0, 1])
                plt.fill(*corners.T, color=color, alpha=0.3, linewidth=2)
                plt.text(*center.T, s=values[i, j],
                         color=color * color_shift, alpha=0.8,
                         ha='center', va='center', fontsize='xx-large')
            elif categories[i, j] == CATEGORIES[2]:
                color = np.array([1, 0, 0])
                plt.fill(*corners.T, color=color, alpha=0.3)
                plt.plot(*corners[[0, 1, 2, 3, 1, 3, 0, 2]].T, color=color)
            elif categories[i, j] == CATEGORIES[3]:
                if notes[i, j] == NOTES[0]:
                    color = np.array([0, 1, 0])
                    symbol = "✓"
                elif notes[i, j] == NOTES[1]:
                    color = np.array([1, 0.5, 0])
                    symbol = "X"
                elif notes[i, j] == NOTES[2]:
                    color = np.array([1, 1, 0])
                    symbol = "?"
                else:
                    color = np.array([0, 1, 1])
                    symbol = ""
                plt.fill(*corners.T, color=color, alpha=0.3, linewidth=2)
                plt.text(*(center + corners[3]) / 2, s=values[i, j],
                         color=color * color_shift, alpha=0.8,
                         ha='center', va='center', fontsize='large')
                plt.text(*(center + corners[0]) / 2, s=symbol,
                         color=color * color_shift, alpha=0.8,
                         ha='center', va='center', fontsize='large')
    plt.show()


def _main():
    image = utils.rgb_to_gray(
        utils.byte_to_double(cv2.imread("training_im/im_3.jpg")))
    image, coords = gd.get_coordinates(image)
    if coords is None:
        return
    coords = gd.rotate_coordinates(coords, 1)

    categories = np.full((9, 9), "", dtype=object)
    notes = np.full((9, 9), "", dtype=object)
    cell_solution = np.zeros((9, 9), dtype=object)

    for i, cat in enumerate(CATEGORIES):
        categories[i] = cat
    for i, note in enumerate(NOTES):
        notes[:, i] = note

    display_solution(image, coords, cell_solution, categories, notes)


if __name__ == "__main__":
    import grid_detection as gd

    _main()
