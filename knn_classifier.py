# -*- coding: utf-8 -*-
"""
@author: david
"""

import numpy as np
import scipy as sp

CLASSIFICATOR = "classificator"
REGRESSOR = "regressor"


def classification(labels):
    """
    Return most voted value for classification
    """
    mode = sp.stats.mode(labels, axis=-1, keepdims=False)
    return mode[0], mode[1] / labels.shape[-1]


def regression(numbers):
    """
    Return mean value for regression
    """
    return np.mean(numbers), 1


class KNN():
    """
    Classificator or Regressor of numpy arrays
    """

    def __init__(self, neighbours, task=CLASSIFICATOR):
        self.k = neighbours
        self.train_data = None
        self.train_labels = None
        self.fit = False
        if task == CLASSIFICATOR:
            self.vote = classification
        elif task == REGRESSOR:
            self.vote = regression
        else:
            raise ValueError(f"""Invalid task ({task}) passed, accepted tasks are:
{CLASSIFICATOR} and {REGRESSOR}""")

    def train(self, data, labels):
        """
        Save all the data and labels for reference in new cases
        """
        self.train_data = data.reshape(data.shape[0], -1)
        self.train_labels = labels
        self.fit = True

    def predict(self, data, return_probablity=False):
        """
        Assign a label to the data based on all the previous known data
        """
        assert self.fit
        test_data = data.reshape(data.shape[0], -1)
        dist = sp.spatial.distance_matrix(test_data, self.train_data)
        neighbours = np.argsort(dist)[:, :self.k]
        value = self.train_labels[neighbours]
        result, p_value = self.vote(value)
        if return_probablity:
            return result, p_value
        return result


def _main():
    cell_size = 64

    image = utils.rgb_to_gray(
        utils.byte_to_double(cv2.imread("training_im/im_2.jpg")))
    image, coords = gd.get_coordinates(image)
    if coords is None:
        return
    coords = gd.rotate_coordinates(coords, 1)

    neighbours = 7
    knn = KNN(neighbours)
    labels = [[] for i in range(10)]
    digits = [[] for i in range(10)]
    labels[0] = np.zeros(neighbours, dtype=int)
    digits[0] = np.zeros((neighbours, cell_size, cell_size))
    for i in range(1, 10):
        for file in glob.glob(f"datasets/digits{cell_size}/{i}*.png"):
            im_digit = utils.rgb_to_gray(cv2.imread(file))
            digits[i].append(cs.binarize_cell(
                cs.resize_cell(im_digit, (cell_size, cell_size))))
        labels[i] = np.repeat(i, len(digits[i]))
        digits[i] = np.array(digits[i])
    labels = np.concatenate(labels)
    digits = np.concatenate(digits)
    knn.train(digits, labels)

    cells = cs.make_cell_array(image, coords, cell_size)
    cell_values = knn.predict(cells.reshape((81, -1)))
    cell_values = cell_values.reshape((9, 9))
    
    cell_solution, result = solve.solve(cell_values)
    print(result)

    _, axs = plt.subplots(9, 9, figsize=(20, 20))
    for i in range(9):
        for j in range(9):
            cell = cells[i, j]
            axs[i, j].imshow(cell, 'gray' if cell_values[i, j] == 0 else None)
            axs[i, j].axis('off')
            axs[i, j].axis('tight')
            axs[i, j].set_aspect('equal')
            axs[i, j].text(0, cell_size / 2,
                           cell_values[i, j], fontsize=80, c='red')
            if not result == solve.RESULT[-1]:
                axs[i, j].text(cell_size / 2, cell_size / 2,
                               cell_solution[i, j], fontsize=80, c='lime')
    plt.show()


if __name__ == "__main__":
    import utils
    import solve
    import cv2
    import glob
    import matplotlib.pyplot as plt
    import grid_detection as gd
    import cell_segmentation as cs

    _main()
