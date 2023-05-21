# -*- coding: utf-8 -*-
"""
@author: david
"""

import numpy as np
import cv2
import utils


def cut_cell(image, corners, target_size):
    """
    Extract the cell at coordinates extreme from image leaving cell of target_size
    """
    corners = corners.reshape((-1, 2))
    dlt = utils.square_dlt(corners, target_size)
    return cv2.warpPerspective(image, dlt, target_size)


def resize_cell(cell, target_size):
    """
    Changes the size of a given cell
    """
    return cv2.resize(cell, target_size)


def binarize_cell(cell):
    """
    Binarizes cell and extracts biggest connected component with some restrictions
    """
    cell = utils.im_as_byte(cell)
    if np.max(cell) - np.min(cell) < 64:
        return cell != cell
    _, binary_cell = cv2.threshold(
        cell, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_cell)
    if num_labels <= 1:
        return binary_cell & False

    middle = np.array(cell.shape)[::-1] // 3
    center = (np.array(cell.shape)[[1, 0]] - 1) / 2
    area = stats[:, cv2.CC_STAT_AREA]

    in_center = np.unique(labels[middle[1]:-middle[1], middle[0]:-middle[0]])
    centered = np.nonzero(
        np.all(np.abs(center - centroids) < middle, axis=-1))[0]
    meaningful = np.nonzero(area >= np.mean(cell.shape) / 2)[0]

    valid = np.zeros(num_labels)
    valid[in_center] += 1
    valid[centered] += 1
    valid[meaningful] += 1
    valid = valid >= 3
    valid[0] = False
    valid = np.nonzero(valid)[0]

    if len(valid) == 0:
        return binary_cell & False

    importance = area / (1 + np.linalg.norm(
        np.tile(center, (num_labels, 1)) - centroids, axis=-1))
    biggest = valid[np.argmax(importance[valid])]
    region = utils.im_as_byte(labels == biggest)
    translation = (center - centroids[biggest, :]).flatten()
    translation = np.array([[1, 0, translation[0]],
                            [0, 1, translation[1]]],
                           dtype=np.float32)
    region = cv2.warpAffine(region, translation, region.shape).astype(bool)
    return region


def gray_cell(cell):
    """
    Centers, inverts and cleans cell according to the biggest
        connected component with some restrictions
    """
    cell = utils.im_as_byte(cell)
    if np.max(cell) - np.min(cell) < 64:
        return cell * 0.0
    _, binary_cell = cv2.threshold(
        cell, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_cell)
    if num_labels <= 1:
        return cell * 0.0

    middle = np.array(cell.shape)[::-1] // 3
    center = (np.array(cell.shape)[[1, 0]] - 1) / 2
    area = stats[:, cv2.CC_STAT_AREA]

    in_center = np.unique(labels[middle[1]:-middle[1], middle[0]:-middle[0]])
    centered = np.nonzero(
        np.all(np.abs(center - centroids) < middle, axis=-1))[0]
    meaningful = np.nonzero(area >= np.mean(cell.shape) / 2)[0]

    valid = np.zeros(num_labels)
    valid[in_center] += 1
    valid[centered] += 1
    valid[meaningful] += 1
    valid = valid >= 3
    valid[0] = False
    valid = np.nonzero(valid)[0]

    if len(valid) == 0:
        return cell * 0.0

    importance = area / (1 + np.linalg.norm(
        np.tile(center, (num_labels, 1)) - centroids, axis=-1))
    biggest = valid[np.argmax(importance[valid])]
    region = utils.im_as_byte(labels == biggest)

    cell = utils.im_as_double(cell)
    translation = (center - centroids[biggest, :]).flatten()
    translation = np.array([[1, 0, translation[0]],
                            [0, 1, translation[1]]],
                           dtype=np.float32)
    extra = np.round(np.array(cell.shape) / 5).astype(int)
    region = cv2.dilate(region, extra).astype(bool)
    eq_low, eq_high = np.min(cell[region]), np.max(cell[region])
    if eq_low + 1e-6 >= eq_high:
        cell[region] = 1
    else:
        cell[region] = 1 - (cell[region] - eq_low) / (eq_high - eq_low)
    cell[~region] = 0
    cell = cv2.warpAffine(cell, translation, cell.shape)
    return cell


def make_cell_array(image, coordinates, cell_size, binary=True):
    """
    Extract all Sudoku cells with size cell_size from an image given the corner coordinates
    """
    dtype = np.uint8 if binary else np.float32
    cells = np.zeros((9, 9, cell_size, cell_size), dtype=dtype)
    for i in range(9):
        for j in range(9):
            cell = cut_cell(image, coordinates[i:i + 2, j:j + 2],
                            (cell_size, cell_size))
            if binary:
                cells[i, j] = binarize_cell(cell)
            else:
                cells[i, j] = gray_cell(cell)
    return cells


def process_cell(cell, cell_size, binary=True):
    """
    Apply all steps to process cell
    """
    if not cell.shape == (cell_size, cell_size):
        cell = resize_cell(cell, (cell_size, cell_size))
    if binary:
        cell = binarize_cell(cell)
    else:
        cell = gray_cell(cell)
    return cell


def _main():
    image = utils.rgb_to_gray(
        utils.byte_to_double(cv2.imread("training_im/im_1.jpg")))
    size_a = 64
    size_b = 64
    size_c = 64
    image, coords = gd.get_coordinates(image)
    coords = np.rot90(coords, 0, (0, 1))
    coords = gd.rotate_coordinates(coords, -1)
    plt.imshow(image, 'gray')
    plt.scatter(coords[..., 0].flatten(), coords[..., 1].flatten(), 3, c='r')
    plt.show()
    
    _, axs = plt.subplots(9, 9, figsize=(20, 20))
    for i in range(9):
        for j in range(0, 3):
            cell = cut_cell(image, coords[i:i + 2, j:j + 2], (size_a, size_a))
            cell = 255 - cell
            empty = np.sum(cell) < 64
            axs[i, j].imshow(cell, 'gray' if empty else None)
            axs[i, j].axis('off')
            axs[i, j].axis('tight')
            axs[i, j].set_aspect('equal')

    for i in range(9):
        for j in range(3, 6):
            cell = cut_cell(image, coords[i:i + 2, j:j + 2], (size_b, size_b))
            cell = gray_cell(cell)
            empty = np.sum(cell) == 0
            axs[i, j].imshow(cell, 'gray' if empty else None)
            axs[i, j].axis('off')
            axs[i, j].axis('tight')
            axs[i, j].set_aspect('equal')

    for i in range(9):
        for j in range(6, 9):
            cell = cut_cell(image, coords[i:i + 2, j:j + 2], (size_c, size_c))
            cell = binarize_cell(cell)
            empty = np.sum(cell) == 0
            axs[i, j].imshow(cell, 'gray' if empty else None)
            axs[i, j].axis('off')
            axs[i, j].axis('tight')
            axs[i, j].set_aspect('equal')
    plt.show()
    
    _, axs = plt.subplots(9, 9, figsize=(20, 20))
    for i in range(9):
        for j in range(9):
            cell = cut_cell(image, coords[i:i + 2, j:j + 2], (size_a, size_a))
            cell = 255 - cell
            empty = np.sum(cell) < 64
            axs[i, j].imshow(cell, 'gray' if empty else None)
            axs[i, j].axis('off')
            axs[i, j].axis('tight')
            axs[i, j].set_aspect('equal')
    plt.show()
    
    _, axs = plt.subplots(9, 9, figsize=(20, 20))
    for i in range(9):
        for j in range(9):
            cell = cut_cell(image, coords[i:i + 2, j:j + 2], (size_b, size_b))
            cell = gray_cell(cell)
            empty = np.sum(cell) == 0
            axs[i, j].imshow(cell, 'gray' if empty else None)
            axs[i, j].axis('off')
            axs[i, j].axis('tight')
            axs[i, j].set_aspect('equal')
    plt.show()
    
    _, axs = plt.subplots(9, 9, figsize=(20, 20))
    for i in range(9):
        for j in range(9):
            cell = cut_cell(image, coords[i:i + 2, j:j + 2], (size_c, size_c))
            cell = binarize_cell(cell)
            empty = np.sum(cell) == 0
            axs[i, j].imshow(cell, 'gray' if empty else None)
            axs[i, j].axis('off')
            axs[i, j].axis('tight')
            axs[i, j].set_aspect('equal')
    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import grid_detection as gd

    _main()
