# -*- coding: utf-8 -*-
"""
@author: david
"""

NEIGHBOURS = 7
KNN_FORMAT = [64, False]
CNN_FORMAT = [28, False]
CLUSTER_FORMAT = [64, False]

MAX_LINE = 0


def info_update(message):
    global MAX_LINE
    MAX_LINE = max(MAX_LINE, len(message))
    print("{info:{width}}".format(info=message, width=MAX_LINE), end='\r')


def from_image(args):
    # Load image
    info_update(f"Loading image '{args.image.name}'...")
    image = utils.im_load_gray(args.image.name)

    # Search grid coordinates
    info_update("Searching for the Sudoku grid...")
    image, coordinates = get_coordinates(image)

    if coordinates is None:
        info_update("No Sudoku found")
        sys.exit()

    # Board information
    cell_values = np.zeros((9, 9), dtype=int)
    info_values = np.full((9, 9), "initial", dtype=object)
    note_values = np.full((9, 9), "none", dtype=object)

    if args.classifier == 'KNN':
        # Segment sudoku cells
        info_update("Segmenting sudoku...")
        cell_size, binarized = KNN_FORMAT
        cell_contents = make_cell_array(
            image, coordinates, cell_size, binarized)
        blank = np.sum(cell_contents, axis=(2, 3)) == 0

        # Load Images to train the k-Nearest Neighbour
        info_update("Loading model information...")
        knn = KNN(NEIGHBOURS)
        with open(f"datasets/digits{cell_size}/data_" +
                  f"{'binarized' if binarized else 'gray'}.npy",
                  "rb") as handler:
            digits = np.load(handler)
            labels = np.load(handler)

        # Search the rotation of the grid
        info_update("Orienting Sudoku...")
        rot_labels = np.zeros(labels.size)
        rot_digits = digits.copy()
        perm = np.random.permutation(labels.size)
        for i in range(1, 4):
            ind = perm[i * labels.size // 4:(i + 1) * labels.size // 4]
            rot_labels[ind] = i
            rot_digits[ind] = np.rot90(rot_digits[ind], k=-i, axes=(1, 2))
        knn.train(rot_digits, rot_labels)
        vote, weight = knn.predict(cell_contents[~blank], True)
        weights = np.zeros(4)
        for i in range(4):
            weights[i] = np.sum(weight[vote == i])
        rot = np.argmax(weights)
        coordinates = np.rot90(coordinates, rot)
        cell_contents = make_cell_array(
            image, coordinates, cell_size, binary=binarized)
        blank = np.sum(cell_contents, axis=(2, 3)) == 0
        num_i, num_j = np.nonzero(~blank)

        # Recognize the values
        info_update("Extracting digits...")
        knn.train(digits, labels)
        info_values[blank] = "solved"
        cell_values[~blank], probability = knn.predict(
            cell_contents[~blank], True)
        low_prob = probability < 0.8
        note_values[num_i[low_prob], num_j[low_prob]] = "maybe"

    elif args.classifier == 'Cluster':
        # Segment sudoku cells
        info_update("Segmenting sudoku...")
        cell_size, binarized = CLUSTER_FORMAT
        cell_contents = make_cell_array(
            image, coordinates, cell_size, binary=binarized)
        blank = np.sum(cell_contents, axis=(2, 3)) == 0

        # Load Classification models and information matrices
        info_update("Loading models...")
        with open("models/cluster.pickle", "rb") as handler:
            mapping = pickle.load(handler)
            clusters = pickle.load(handler)
            diversity = pickle.load(handler)

        # Search the rotation of the grid
        info_update("Orienting Sudoku...")
        values = np.concatenate([np.rot90(cell_contents[~blank], -i, (1, 2))
                                 for i in range(4)])
        values = mapping.transform(values.reshape((-1, cell_size * cell_size)))
        scores = clusters.score_samples(values)
        weights = np.sum(scores.reshape((4, -1)), axis=-1)
        rot = np.argmax(weights)
        coordinates = np.rot90(coordinates, rot)
        cell_contents = np.rot90(cell_contents, rot, (0, 1))
        cell_contents = np.rot90(cell_contents, rot, (2, 3))
        blank = np.sum(cell_contents, axis=(2, 3)) == 0
        num_i, num_j = np.nonzero(~blank)

        # Recognize the values
        info_update("Extracting digits...")
        values = cell_contents[~blank]
        values = mapping.transform(values.reshape((-1, cell_size * cell_size)))
        probabilities = clusters.predict_proba(values)
        probabilities = probabilities @ diversity
        info_values[blank] = "solved"
        cell_values[~blank] = np.argmax(probabilities, axis=-1)
        low_prob = np.max(probabilities, axis=-1) < 0.8
        note_values[num_i[low_prob], num_j[low_prob]] = "maybe"

    elif args.classifier == 'CNN':
        # Segment sudoku cells
        info_update("Segmenting sudoku...")
        cell_size, binarized = CNN_FORMAT
        cell_contents = make_cell_array(
            image, coordinates, cell_size, binary=binarized)
        blank = np.sum(cell_contents, axis=(2, 3)) <= 1e-6

        with torch.no_grad():
            # Load CNN models
            info_update("Loading models...")
            orientation = torch.jit.load('models/orientation_classifier.pt')
            digit = torch.jit.load('models/digit_classifier.pt')

            # Search the rotation of the grid
            info_update("Orienting Sudoku...")
            t_cells = cell_contents[~blank].reshape(-1,
                                                    1, cell_size, cell_size)
            t_cells = torch.tensor(t_cells)
            weights = orientation(t_cells)
            rot = torch.argmax(torch.sum(weights, axis=0))
            coordinates = np.rot90(coordinates, rot)
            cell_contents = np.rot90(cell_contents, rot, (0, 1))
            cell_contents = np.rot90(cell_contents, rot, (2, 3))
            blank = np.sum(cell_contents, axis=(2, 3)) == 0
            num_i, num_j = np.nonzero(~blank)

            # Recognize the values
            info_update("Extracting digits...")
            t_cells = cell_contents[~blank].reshape(-1,
                                                    1, cell_size, cell_size)
            t_cells = torch.tensor(t_cells)
            probabilities = digit(t_cells)
            info_values[blank] = "solved"
            cell_values[~blank] = np.argmax(
                probabilities.detach().numpy(), axis=-1)
            low_prob = np.max(probabilities.detach().numpy(), axis=-1) < 0.8
            note_values[num_i[low_prob], num_j[low_prob]] = "maybe"

    info_update("Solving Sudoku...")
    solution_values, result = solve.solve(cell_values)
    if result == solve.RESULT[-1]:
        info_values[info_values == "solved"] = "invalid"
        note_values[info_values == "written"] = ""
        category = 'no'

    if result == solve.RESULT[0]:
        note_values[info_values == "written"] = "correct"
        note_values[info_values == "written"] = "wrong"
        category = 'the'

    if result == solve.RESULT[-1]:
        note_values[info_values == "written"] = "maybe"
        category = 'a'


    info_update(f"Displaying {category} solution...")
    display.display_solution(
        image, coordinates, solution_values, info_values, note_values)
    info_update("Done")


def from_digits(args):
    if len(args.manual) < 81:
        if len(args.manual) > 0:
            print("Not all digits inputed in the call, reinput them")
        else:
            print("Input number list (0 as blank)")
        nums = []
        valid = [str(i) for i in range(10)]
        while len(nums) < 81:
            nums += [int(n) for n in input() if n in valid]
        nums = nums[:81]
    else:
        nums = args.manual[:81]
    board = np.array(nums, dtype=int).reshape((9, 9))
    solved_board, result = solve.solve(board)
    print(result)
    if not result == solve.RESULT[-1]:
        print("Board:")
        for i in range(9):
            print("{}{}{} {}{}{} {}{}{}".format(*solved_board[i]))
            if i in [2, 5]:
                print()


def start_up(args):
    info_update("Start up of models...")
    if args.classifier == 'KNN':
        cell_size, binarized = KNN_FORMAT
        labels = [[] for i in range(10)]
        digits = [[] for i in range(10)]
        labels[0] = np.zeros(2 * NEIGHBOURS, dtype=int)
        digits[0] = np.concatenate([
            np.zeros((NEIGHBOURS, cell_size, cell_size)),
            np.ones((NEIGHBOURS, cell_size, cell_size))
        ])
        for i in range(1, 10):
            for file in glob.glob(path.join(path.dirname(__file__),
                                            f"datasets/digits{cell_size}/{i}*.png")):
                im_digit = utils.im_load_gray(file, int)
                digits[i].append(process_cell(
                    im_digit, cell_size, binary=binarized))
            labels[i] = np.repeat(i, len(digits[i]))
            digits[i] = np.array(digits[i])
        labels = np.concatenate(labels)
        digits = np.concatenate(digits)
        with open(f"datasets/digits{cell_size}/data_" +
                  f"{'binarized' if binarized else 'gray'}.npy",
                  "wb") as handler:
            np.save(handler, digits, allow_pickle=False)
            np.save(handler, labels, allow_pickle=False)

    elif args.classifier == 'Cluster':
        cell_size, binarized = CLUSTER_FORMAT
        labels = [[] for i in range(10)]
        digits = [[] for i in range(10)]
        labels[0] = np.zeros(NEIGHBOURS, dtype=int)
        digits[0] = np.zeros((NEIGHBOURS, cell_size, cell_size))
        for i in range(1, 10):
            for file in glob.glob(path.join(path.dirname(__file__),
                                            f"datasets/digits{cell_size}/{i}*.png")):
                im_digit = utils.im_load_gray(file, int)
                digits[i].append(process_cell(
                    im_digit, cell_size, binary=binarized))
            labels[i] = np.repeat(i, len(digits[i]))
            digits[i] = np.array(digits[i])
        labels = np.concatenate(labels)
        digits = np.concatenate(digits)
        with open(f"datasets/digits{cell_size}/data_" +
                  f"{'binarized' if binarized else 'gray'}.npy",
                  "wb") as handler:
            np.save(handler, digits, allow_pickle=False)
            np.save(handler, labels, allow_pickle=False)

        mapping = umap.UMAP(n_components=3)
        data = mapping.fit_transform(digits.reshape((-1, cell_size * cell_size)))

        CLUSTERS = 100
        clusters = mixture.GaussianMixture(n_components=CLUSTERS)
        c_labels = clusters.fit_predict(data)

        diversity = np.zeros((CLUSTERS, 10))
        for i in range(CLUSTERS):
            cases = c_labels == i
            if np.any(cases):
                for j in range(10):
                    diversity[i, j] = np.sum(labels[cases] == j)

        used = np.any(diversity, axis=-1)
        diversity[used] = diversity[used] / \
            np.linalg.norm(diversity[used], axis=-1, keepdims=True)

        with open("models/cluster.pickle", "wb") as handler:
            pickle.dump(mapping, handler)
            pickle.dump(clusters, handler)
            pickle.dump(diversity, handler)
            # pickle.dump(written, handler)

    elif args.classifier == 'CNN':
        print("Startup of CNN models is not recommended.")
        print("If necessary, reinstall or run all files in models directory.")


if __name__ == "__main__":
    info_update("Importing dependencies...")
    from os import sys, path
    import argparse
    import utils
    import solve
    import display
    import numpy as np
    from grid_detection import get_coordinates
    from cell_segmentation import make_cell_array, process_cell

    argParser = argparse.ArgumentParser(
        description='Solve sudoku given a photo',
        epilog='Program by David Candela')
    input_method = argParser.add_mutually_exclusive_group(required=True)
    input_method.add_argument("image", nargs='?', type=argparse.FileType(
        'r'), help="Input image file to solve")
    argParser.add_argument("-c", "--classifier", metavar='METHOD', type=str,
                           default='KNN', choices=['KNN', 'CNN', 'Cluster'],
                           help="Classifier for image digit recognition")
    input_method.add_argument("--manual", metavar='N',
                              nargs='*', type=int, help="Input sudoku manually")
    argParser.add_argument("--startup", action="store_true",
                           help="Reset all saved classification methods")

    args = argParser.parse_args(sys.argv[1:])
    if args.classifier == 'KNN':
        from knn_classifier import KNN
    elif args.classifier == 'CNN':
        import torch
    elif args.classifier == 'Cluster':
        import pickle
        if args.startup:
            from sklearn import mixture
            import umap
    if args.startup:
        import glob
        start_up(args)
    if args.image is not None:
        from_image(args)
    elif args.manual is not None:
        from_digits(args)
