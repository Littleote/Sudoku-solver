# -*- coding: utf-8 -*-
"""
@author: david
"""

import numpy as np
import cv2
import utils


def mod_diff(center, omega, mod):
    """
    Calculate the difference of modular continuous variables
    """
    return (omega - center + mod / 2) % mod - mod / 2


def mod_mean(omega, mod):
    """
    Calculate mean value of modular continuous variables
    """
    phi = omega * (2 * np.pi) / mod
    mean_y = np.sum(np.sin(phi))
    mean_x = np.sum(np.cos(phi))
    omega = np.arctan2(mean_y, mean_x) * mod / (2 * np.pi)
    return omega % mod


def cluster(points, cluster_size, max_clusters=None, threshold=0.5, take_mean=True):
    """
    Cluster Hough Line peak points.
        points (entries x (rho, theta, votes)): HoughLine points with oprionally the votes
        cluster_size (rho_area, theta_area): area to cluster
        max_cluster: limit of clusters to return
        threshold: minimum percentage of weight respect to the most voted
        take_mean: return mean of cluter or max of cluster
    """
    point_in_cluster = np.full(points.shape[0], False)
    clusters = []
    cluster_weights = []
    rho, theta = points[:, 0:2].T
    if points.shape[1] > 2:
        weights = points[:, 2]
    else:
        weights = np.ones(points.shape[0])
    order = np.argsort(weights)[::-1]
    for i in order:
        if not point_in_cluster[i]:
            out_of_phase = np.abs(theta[i] - theta) > np.pi / 2
            rho_sign = np.where(out_of_phase, -1, 1)
            close_rho = np.abs(rho[i] - rho_sign * rho) < cluster_size[0]
            theta_diff = mod_diff(theta[i], theta, np.pi)
            close_theta = np.abs(theta_diff) < cluster_size[1]
            close_points = close_rho & close_theta & ~point_in_cluster

            if take_mean:
                total_weight = np.sum(weights[close_points])
                c_theta = np.sum(
                    theta_diff[close_points] * weights[close_points])
                c_theta = (theta[i] + c_theta / total_weight) % np.pi
                c_rho = np.sum(
                    (rho_sign * rho)[close_points] * weights[close_points])
                c_rho = c_rho / total_weight * \
                    (1 if np.abs(theta[i] - c_theta) < np.pi / 2 else -1)

                clusters.append([c_rho, c_theta])
            else:
                clusters.append([rho[i], theta[i]])
            cluster_weights.append(weights[i])
            point_in_cluster[close_points] = True
    order = np.argsort(cluster_weights)[::-1]
    max_weight = cluster_weights[order[0]]
    order = order[:np.sum(cluster_weights > threshold * max_weight)]
    clusters = np.array(clusters)[order]
    return np.array(clusters[:max_clusters])


def get_zones(image, return_stats=True):
    """
    Binarize image and return connected components
    """
    image = utils.im_as_double(image)
    im_min, im_max = np.min(image), np.max(image)
    image = (image - im_min) * (im_max - im_min)
    im_threshold = cv2.adaptiveThreshold(
        utils.double_to_byte(image), 1,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 501, 8)
    im_threshold = cv2.morphologyEx(
        im_threshold, cv2.MORPH_OPEN, np.ones((3, 3)))
    im_threshold = cv2.morphologyEx(
        im_threshold, cv2.MORPH_CLOSE, np.ones((3, 3)))
    if return_stats:
        return cv2.connectedComponentsWithStats(im_threshold)
    return cv2.connectedComponents(im_threshold)


def hough_to_plane(hough_lines):
    """
    Transform a line in the form of (rho, theta) to (a, b, c) x (X, Y, 1)
    """
    coeff_a = np.cos(hough_lines[..., 1])
    coeff_b = np.sin(hough_lines[..., 1])
    coeff_c = -hough_lines[..., 0]
    return np.array([coeff_a, coeff_b, coeff_c]).T


def plane_to_point(planes):
    """
    Return plane to dual point at (x, y, 1)
    """
    return planes[..., 0:2] / planes[..., [2, 2]]


def search_grid_like(im_zones, zone_areas, zone_centroids):
    """
    Search image with connected components for the most grid-like
    Return:
        found: If a grid-like zone has been found
        full_grid: If all the grid lines have been found or only a part
        grid_format: What elements of a grid have been found (zone, hough_lines)
        zone: The zone most likely to be the grid
        (lines): The lines in forma rho, theta of the grid lines
        (direction): The direction the lines follow
            vertical: True if the line is in the vertical group
            horizontal: True if the line is in the horizontal group
    """
    big_zones = np.argsort(zone_areas[1:])[::-1] + 1
    candidate = None
    for i in range(min(10, len(big_zones))):
        i_zone = utils.set_type_byte(im_zones == big_zones[i])
        hough_lines = cv2.HoughLinesWithAccumulator(
            i_zone, 1, 1 * np.pi / 180, 400)
        if hough_lines is not None:
            hough_lines = hough_lines[:, 0, :]
            centroid = zone_centroids[big_zones[i]]
            phase = np.arctan2(*centroid[::-1])
            amplitude = np.linalg.norm(centroid)
            hough_lines[:, 0] -= amplitude * np.cos(hough_lines[:, 1] - phase)
            hough_clusters = cluster(
                hough_lines[:, :], (25, 25 * np.pi / 180), 20, take_mean=False)
            hough_clusters[:, 0] += amplitude * \
                np.cos(hough_clusters[:, 1] - phase)

            num_lines = hough_clusters.shape[0]
            mean_theta = mod_mean(
                hough_clusters[:, 1] + np.pi / 4, np.pi / 2) - np.pi / 4
            direction = mod_diff(mean_theta, hough_clusters[:, 1], np.pi)
            vertical = np.abs(direction) < np.pi / 8
            horizontal = np.abs(direction) > 3 * np.pi / 8

            if num_lines < 8 or \
                    np.count_nonzero(horizontal) < num_lines / 4 or \
                    np.count_nonzero(vertical) < num_lines / 4:
                continue

            if num_lines == 20 and \
                    np.count_nonzero(horizontal) == 10 and \
                    np.count_nonzero(vertical) == 10:
                return {
                    "found": True,
                    "full_grid": True,
                    "grid_format": ['zone', 'hough_lines'],
                    "zone": i_zone,
                    "lines": hough_clusters,
                    "direction": {'vertical': vertical, 'horizontal': horizontal}
                }

            if candidate is None:
                candidate = {
                    "found": True,
                    "full_grid": False,
                    "grid_format": ['zone', 'hough_lines'],
                    "zone": i_zone,
                    "lines": hough_clusters,
                    "direction": {'vertical': vertical, 'horizontal': horizontal}
                }
    if candidate is None:
        return {
            "found": False,
            "full_grid": False,
            "grid_format": ['zone'],
            "zone": (im_zones == big_zones[0])
        }
    return candidate


def poly_approx_grid(im_zone):
    """
    Approximate connected component with four sided polygon
    Return:
        found: If a quadrilater fits the components
        grid_format: What elements of a grid have been found (zone, corners)
        zone: The input zone
        (corners): The corners of the quadrilater in the form (x, y)
    """
    points = np.array(np.nonzero(im_zone)).T
    hull = cv2.convexHull(points)
    perimeter = cv2.arcLength(hull, True)
    polygon = cv2.approxPolyDP(hull, perimeter * 0.01, True)
    if not polygon.shape[0] == 4:
        return {
            "found": False,
            "grid_format": ['zone'],
            "zone": im_zone
        }
    polygon = polygon[[0, 3, 1, 2], 0, ::-1]
    return {
        "found": True,
        "grid_format": ['zone', 'corners'],
        "corners": polygon,
        "zone": im_zone
    }


def add_grid_coords(grid_info):
    """
    Adds the coordinates of the grid based on the current information of the grid
    """
    if 'hough_lines' in grid_info["grid_format"] and grid_info["full_grid"]:
        grid_info["coords"] = np.zeros((10, 10, 2))
        planes = hough_to_plane(grid_info["lines"])
        vertical = grid_info["direction"]['vertical']
        v_planes = planes[vertical]
        v_order = grid_info["lines"][vertical, 0] * \
            np.where(grid_info["lines"][vertical, 1] < np.pi / 2, 1, -1)
        v_order = np.argsort(v_order)
        v_planes = v_planes[v_order]
        horizontal = grid_info["direction"]['horizontal']
        h_planes = planes[horizontal]
        h_order = np.argsort(grid_info["lines"][horizontal, 0])
        h_planes = h_planes[h_order]
        for i in range(10):
            points = np.cross(h_planes[i], v_planes)
            grid_info["coords"][i] = plane_to_point(points)
        grid_info["grid_format"] += ['coordinates']
    elif 'corners' in grid_info["grid_format"]:
        corners = np.array(grid_info["corners"])
        corners = corners.reshape((-1, 2))
        dlt = utils.square_dlt(corners, (9, 9))
        coords = np.stack(np.meshgrid(np.arange(10), np.arange(10)), axis=2)
        grid_info["coords"] = utils.backwards_dlt(dlt, coords)
        grid_info["grid_format"] += ['coordinates']
    elif 'zone' in grid_info["grid_format"]:
        grid_info |= poly_approx_grid(grid_info["zone"])
        if grid_info["found"]:
            grid_info |= add_grid_coords(grid_info)
    else:
        raise ValueError(f"""
Grid info doesen't contain any valid format.
Valid formats are 'hough_lines', 'corners', 'zone'.
Instead it has {', '.join(grid_info['grid_format'])}""")
    return grid_info


def refine_grid(image, grid_info, precentage=0.8):
    """
    Adjusts grid cell corners to the real corners in the image
    """
    size = 100
    interval = int(size * precentage / 2)

    final_shape = [9 * size, 9 * size]
    dlt = utils.square_dlt(
        grid_info["coords"][[0, 0, -1, -1], [0, -1, 0, -1]], final_shape)
    old_coords = utils.forward_dlt(dlt, grid_info["coords"])
    weights = np.zeros((10, 10, 1))
    zone = cv2.warpPerspective(grid_info["zone"], dlt, final_shape)
    image = cv2.warpPerspective(image, dlt, final_shape)
    image = 1 - cv2.GaussianBlur(image, (3, 3), 1)
    eq_low, eq_high = np.min(image), np.max(image)
    image = (image - eq_low) / (eq_high - eq_low)
    image = np.maximum(image, zone)
    image = np.stack([image, image, image], axis=2)
    image = utils.im_as_byte(image)
    regions = utils.im_as_byte(zone < 0.5)
    _, regions, stats, _ = cv2.connectedComponentsWithStats(regions)

    x_centers = np.mean(old_coords[..., 0], axis=0)
    x_centers = (x_centers[:-1] + x_centers[1:]).astype(int) // 2
    y_centers = np.mean(old_coords[..., 1], axis=1)
    y_centers = (y_centers[:-1] + y_centers[1:]).astype(int) // 2

    markers = np.zeros(image.shape[0:2], dtype=np.int32)
    region_id = np.zeros((9, 9), dtype=int)
    for i, center_x in enumerate(x_centers):
        for j, center_y in enumerate(y_centers):
            region_id[j, i] = regions[center_y, center_x]
            if stats[region_id[j, i], cv2.CC_STAT_AREA] < size * size / 2:
                region_id[j, i] = -1
            markers[center_y - interval:center_y + interval,
                    center_x - interval:center_x + interval] = 1 + i + 9 * j

    id_list = region_id.reshape(-1)
    _, ind, counts = np.unique(
        id_list, return_inverse=True, return_counts=True)
    for i, count in enumerate(counts):
        if count > 1:
            id_list[ind == i] = -1
    region_id = id_list.reshape((9, 9))

    if -1 in id_list:
        watershed = cv2.watershed(image, markers)
        watershed = watershed.astype(np.uint8)
        watershed[watershed > 81] = 0
        watershed[zone > 0.9] = 0
        watershed = cv2.morphologyEx(
            watershed, cv2.MORPH_OPEN, np.ones((size // 20, size // 20)), iterations=3)

    new_coords = np.zeros((10, 10, 2))
    for i in range(9):
        i_ind = [i, i, i + 1, i + 1]
        for j in range(9):
            j_ind = [j, j + 1, j + 1, j]
            if region_id[j, i] == -1:
                cell = watershed == 1 + i + 9 * j
            else:
                cell = regions == region_id[j, i]
            grid_y, grid_x = np.nonzero(cell)
            if grid_x.size == 0:
                grid_x = old_coords[j_ind, i_ind, 0]
                grid_y = old_coords[j_ind, i_ind, 1]
            corners = [np.argmin(grid_x + grid_y), np.argmin(grid_x - grid_y),
                       np.argmax(grid_x + grid_y), np.argmax(grid_x - grid_y)]
            new_coords[j_ind, i_ind] += \
                np.array([grid_x[corners], grid_y[corners]]).T
            weights[j_ind, i_ind] += 1

    new_coords /= weights
    new_coords[:, 1:-1] = cv2.GaussianBlur(new_coords, (3, 1), 1)[:, 1:-1]
    new_coords[1:-1, :] = cv2.GaussianBlur(new_coords, (1, 3), 1)[1:-1, :]
    new_coords = utils.backwards_dlt(dlt, new_coords)
    return new_coords


def get_coordinates(image, image_scheme=cv2.COLOR_RGB2GRAY):
    """
    Get coordinates of the Sudoku from an image
    """
    image = utils.im_as_double(image)
    if len(image.shape) >= 3:
        if image.shape[2] == 3:
            image = utils.rgb_to_gray(image, image_scheme)
        else:
            raise ValueError("Unkown image shape")
    size = np.round(image.shape[::-1] / np.mean(image.shape) * 2000)
    size = tuple(size.astype(int))
    image = cv2.resize(image, size)
    image = cv2.GaussianBlur(image, (3, 3), 1)
    _, im_zones, stats, centroids = get_zones(image)
    grid_info = search_grid_like(
        im_zones, stats[:, cv2.CC_STAT_AREA], centroids)
    if not grid_info["found"]:
        return image, None
    grid_info = add_grid_coords(grid_info)
    if not grid_info["found"]:
        return image, None
    return image, refine_grid(image, grid_info)


def rotate_coordinates(grid_coords, rotation):
    """
    Rotate the grid by pi/2 * rotation radians
    """
    return np.rot90(grid_coords, k=rotation)


def _main():
    image = utils.rgb_to_gray(
        utils.byte_to_double(cv2.imread("training_im/noise_2.jpg")))
    image = cv2.resize(
        image, tuple(np.round(image.shape[::-1] / np.mean(image.shape) * 2000).astype(int)))
    image = cv2.GaussianBlur(image, (3, 3), 1)

    _, im_zones, stats, centroids = get_zones(image)
    grid_info = search_grid_like(
        im_zones, stats[:, cv2.CC_STAT_AREA], centroids)
    grid_info = add_grid_coords(grid_info)
    if 'coordinates' in grid_info['grid_format']:
        grid_info['coords'] = refine_grid(image, grid_info)

    plt.figure(figsize=(20, 20))
    plt.axis('off')
    plt.imshow(image, 'gray')
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())

    if 'zone' in grid_info['grid_format']:
        zone = np.stack([1 - grid_info['zone'], 1 - grid_info['zone'],
                         1 - grid_info['zone'], .2 + .5 * grid_info['zone']],
                        axis=2)
        plt.imshow(zone)

    if 'hough_lines' in grid_info['grid_format']:
        for line, (rho, theta) in enumerate(grid_info['lines']):
            if grid_info['direction']['horizontal'][line]:
                color = 'red'
            elif grid_info['direction']['vertical'][line]:
                color = 'lime'
            else:
                color = 'blue'
            plt.axline((rho * np.cos(theta), rho * np.sin(theta)),
                       (rho * np.cos(theta) + np.sin(theta),
                        rho * np.sin(theta) - np.cos(theta)),
                       c=color, linewidth=2)

    if 'corners' in grid_info['grid_format']:
        plt.plot(*grid_info['corners'][[0, 1, 3, 2, 0]].T,
                 linewidth=10, c='yellow')

    if 'coordinates' in grid_info['grid_format']:
        for i, row in enumerate(grid_info['coords']):
            for j, coord in enumerate(row):
                color = (j / 9, i / 9, 1 - j / 18 - i / 18)
                plt.plot(coord[0], coord[1], 'P', markersize=20, markerfacecolor=color,
                         markeredgewidth=2, markeredgecolor='black')

    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    _main()
