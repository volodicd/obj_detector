#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""3D to 2D projection module for object recognition pipeline"""

from typing import Tuple, Optional
import numpy as np
import cv2
import open3d as o3d


def project_points_to_image(points: np.ndarray,
                            colors: np.ndarray,
                            image_size: Tuple[int, int] = (480, 640)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to 2D image space using the calibrated camera parameters

    Args:
        points: (N,3) array of 3D points
        colors: (N,3) array of RGB colors
        image_size: Output image size (height, width)

    Returns:
        Tuple containing:
        - 2D projected points (N,2)
        - Image array with projected colors
    """
    from camera_params import(fx_rgb, fy_rgb, cx_rgb, cy_rgb, k1_rgb, k2_rgb, k3_rgb, p1_rgb, p2_rgb)

    # Initialize outputs
    points_2d = np.zeros((len(points), 2))
    image = np.zeros((*image_size, 3), dtype=np.uint8)

    # Project points
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Filter points behind camera
    valid_mask = z > 0

    if np.sum(valid_mask) == 0:
        return points_2d, image

    # Normalize coordinates
    x_norm = x[valid_mask] / z[valid_mask]
    y_norm = y[valid_mask] / z[valid_mask]

    # Apply distortion
    r2 = x_norm ** 2 + y_norm ** 2
    r4 = r2 ** 2
    r6 = r2 ** 3

    # Radial distortion
    radial = (1 + k1_rgb * r2 + k2_rgb * r4 + k3_rgb * r6)
    x_dist = x_norm * radial
    y_dist = y_norm * radial

    # Tangential distortion
    x_dist = x_dist + (2 * p1_rgb * x_norm * y_norm + p2_rgb * (r2 + 2 * x_norm ** 2))
    y_dist = y_dist + (p1_rgb * (r2 + 2 * y_norm ** 2) + 2 * p2_rgb * x_norm * y_norm)

    # Project to pixel coordinates
    x_proj = fx_rgb * x_dist + cx_rgb
    y_proj = fy_rgb * y_dist + cy_rgb

    # Store projected points
    points_2d[valid_mask, 0] = x_proj
    points_2d[valid_mask, 1] = y_proj

    # Convert to pixel coordinates
    pixels = points_2d[valid_mask].astype(int)

    # Filter points within image bounds
    in_bounds = ((pixels[:, 0] >= 0) & (pixels[:, 0] < image_size[1]) &
                 (pixels[:, 1] >= 0) & (pixels[:, 1] < image_size[0]))

    pixels = pixels[in_bounds]
    valid_colors = colors[valid_mask][in_bounds]

    # Create image
    if len(pixels) > 0:
        image[pixels[:, 1], pixels[:, 0]] = (valid_colors * 255).astype(np.uint8)

    return points_2d, image


def fill_holes(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Fill holes in projected image using morphological operations

    Args:
        image: Input image
        kernel_size: Size of morphological kernel

    Returns:
        Processed image with holes filled
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Create binary mask
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Create kernel for morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Fill holes using closing operation
    filled = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if len(image.shape) == 3:
        result = image.copy()
        for i in range(3):
            channel = result[:, :, i]
            channel[filled > 0] = cv2.morphologyEx(channel, cv2.MORPH_CLOSE, kernel)[filled > 0]
        return result

    return filled


def project_pointcloud(pcd: o3d.geometry.PointCloud,
                       labels: np.ndarray,
                       image_size: Tuple[int, int] = (480, 640),
                       fill_holes_kernel: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Project entire pointcloud to 2D image space with hole filling

    Args:
        pcd: Input pointcloud
        labels: Cluster labels for each point
        image_size: Output image size
        fill_holes_kernel: Kernel size for hole filling (0 to disable)

    Returns:
        Tuple containing:
        - 2D projected points
        - Original projected image
        - Processed image with holes filled
        - Label image with cluster IDs
    """
    # Convert pointcloud to numpy arrays
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Project points
    points_2d, image = project_points_to_image(points, colors, image_size)

    # Create label image
    label_image = create_label_image(points_2d, labels, image_size)

    # Fill holes in both color and label images
    if fill_holes_kernel > 0:
        filled_image = fill_holes(image, fill_holes_kernel)
        filled_label_image = fill_label_holes(label_image, fill_holes_kernel)
    else:
        filled_image = image.copy()
        filled_label_image = label_image.copy()

    return points_2d, image, filled_image, filled_label_image


def fill_label_holes(label_image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Fill small holes in the label image using morphological operations

    Args:
        label_image: Input label image
        kernel_size: Size of morphological kernel

    Returns:
        Label image with holes filled
    """
    # Create binary mask for each label
    labels = np.unique(label_image)
    labels = labels[labels != -1]  # Exclude background

    filled_image = label_image.copy()
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    for label in labels:
        # Create binary mask for current label
        mask = (label_image == label).astype(np.uint8)

        # Apply morphological closing
        filled_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Update label image with filled regions
        filled_image[filled_mask == 1] = label

    return filled_image


def create_label_image(points_2d: np.ndarray,
                       labels: np.ndarray,
                       image_size: Tuple[int, int] = (480, 640)) -> np.ndarray:
    """
    Create a 2D label image where each pixel stores the cluster ID

    Args:
        points_2d: (N,2) array of projected 2D points
        labels: (N,) array of cluster labels
        image_size: Output image size (height, width)

    Returns:
        2D array where each pixel value is the cluster ID (-1 for background)
    """
    label_image = np.full(image_size, fill_value=-1, dtype=np.int32)

    for i, (u, v) in enumerate(points_2d):
        if labels[i] == -1:  # Skip noise points
            continue

        u_i, v_i = int(round(u)), int(round(v))
        if 0 <= u_i < image_size[1] and 0 <= v_i < image_size[0]:
            label_image[v_i, u_i] = labels[i]

    return label_image