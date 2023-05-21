# -*- coding: utf-8 -*-
"""
@author: david
"""

import torch
import numpy as np
import cv2

def gray_cell(cell):
    """
    Centers, inverts and cleans cell according to the biggest
        connected component with some restrictions
    """
    binary_cell = cv2.adaptiveThreshold(
        cell, 1, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 2 * cell.shape[0] + 1, 10)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_cell)
    if num_labels <= 1:
        return cell * 0.0
    area = stats[1:, cv2.CC_STAT_AREA]
    center = np.array(cell.shape)[[1, 0]] / 2
    centering = np.linalg.norm(
        np.tile(center, (num_labels - 1, 1)) - centroids[1:], axis=-1)

    biggest = np.argmax(area / (centering + 1)) + 1
    region = labels == biggest
    translation = center - centroids[biggest, :]
    middle = np.array(cell.shape)[::-1] // 3

    if area[biggest - 1] < np.mean(cell.shape) or \
            np.any(np.abs(center - centroids[biggest]) > middle) or \
            np.sum(region[middle[1]:-middle[1], middle[0]:-middle[0]]) == 0:
        biggest = -1
        return cell * 0.0
    translation = np.array([[1, 0, translation[0]],
                            [0, 1, translation[1]]],
                           dtype=np.float32)
    eq_low, eq_high = np.min(cell[region]), np.max(cell[region])
    cell = cell.astype(np.float32)
    if eq_low + 1e-6 >= eq_high:
        cell[region] = 1
    else:
        cell[region] = 1 - (cell[region] - eq_low) / (eq_high - eq_low)
    cell[~region] = 0
    cell = cv2.warpAffine(cell, translation, cell.shape)
    return cell

with torch.no_grad():
    model = torch.jit.load('digit_classifier.pt')
    model.eval()
    
    batch = torch.tensor(np.zeros((9, 1, 28, 28)), dtype=torch.float32)
    for i in range(1, 10):
        image = cv2.imread(f"../datasets/digits28/{i}n80arial.png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        batch[i - 1, 0] = torch.tensor(gray_cell(image))
    
    number = model(batch)
    print(np.max(number.detach().numpy(), axis=-1))
    print(np.argmax(number.detach().numpy(), axis=-1))