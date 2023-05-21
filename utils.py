# -*- coding: utf-8 -*-
"""
@author: david
"""

import numpy as np
import cv2


def rgb_to_gray(img, scheme=cv2.COLOR_BGR2GRAY):
    """
    Switch from Blue Green Red color to Gray scale color
    """
    return cv2.cvtColor(img, scheme)


def set_type_double(img):
    """
    Set image type as floating point
    """
    return img.astype(np.float32)


def set_type_byte(img):
    """
    Set image type as single byte
    """
    return img.astype(np.uint8)


def im_as_double(img):
    """
    Change image type to floating point
    """
    if np.issubdtype(img.dtype, np.integer):
        return byte_to_double(img)
    return set_type_double(img)


def im_as_byte(img):
    """
    Change image type to single byte
    """
    if np.issubdtype(img.dtype, np.integer):
        return set_type_byte(img)
    return double_to_byte(img)


def double_to_byte(img):
    """
    Change from floating point to single byte 
    """
    return set_type_byte(np.round(255.0 * img))


def byte_to_double(img):
    """
    Change from single byte to floating point
    """
    return set_type_double(img) / 255.0


def im_load_gray(img_file, type=float):
    """
    Load image and set it as a gray_scale
    """
    img = cv2.imread(img_file)
    if type == float:
        img = im_as_double(img)
    elif type == int:
        img = im_as_byte(img)
    else:
        raise ValueError("Invalid value type for image")
    if len(img.shape) > 2:
        if img.shape[2] == 3:
            img = rgb_to_gray(img)
        elif img.shape[2] == 1:
            img = img.reshape(img.shape[:2])
    return img


def square_dlt(src, target_size):
    """
    Calculate DLT from src to a square of size target_size
        src: points in form (x, y) ordered (TL, TR, BL, BR).
        target_size: shape (x, y)
    """
    src = src.astype(np.float32)
    dst = np.zeros((4, 2), dtype=np.float32)
    dst[[1, 3], 0] = target_size[0]
    dst[[2, 3], 1] = target_size[1]
    return cv2.getPerspectiveTransform(src, dst)


def inverse_dlt(dlt):
    """
    Calculates the inverse of the DLT
    """
    return np.linalg.inv(dlt)


def forward_dlt(dlt, points):
    """
    Apply DLT to multiple points of shape (..., 2)
    """
    initial_shape = points.shape
    points_3d = points.reshape((-1, 2))
    ones = np.ones((points_3d.shape[0], 1))
    points_3d = np.concatenate([points_3d, ones], axis=1)
    points_3d = points_3d @ dlt.T
    points = points_3d[:, 0:2] / points_3d[:, [2, 2]]
    return points.reshape(initial_shape)


def backwards_dlt(dlt, points):
    """
    Apply inverse DLT to multiple points of shape (..., 2)
    """
    return forward_dlt(inverse_dlt(dlt), points)
