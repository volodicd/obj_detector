#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Helper functions for 3D object recognition pipeline"""

import numpy as np
import cv2
from typing import Dict, List
from pathlib import Path


def visualize_label_image (label_image: np.ndarray) -> np.ndarray:
    """
    Visualize label image with different colors for each cluster

    Args:
        label_image: 2D array of cluster IDs
    Returns:
        Colored visualization of clusters
    """
    # Create colormap (excluding background)
    unique_labels = np.unique (label_image)
    unique_labels = unique_labels[unique_labels != -1]
    n_labels = len (unique_labels)

    # Create colored image
    vis_image = np.zeros ((*label_image.shape, 3), dtype=np.uint8)

    # Use HSV color space for better color distribution
    for i, label in enumerate (unique_labels):
        hue = int (180 * i / max (n_labels, 1))
        color = cv2.cvtColor (np.uint8 ([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
        vis_image[label_image == label] = color

    return vis_image


def visualize_recognition_results(image: np.ndarray,
                                label_image: np.ndarray,
                                classifications: Dict[int, str],
                                draw_boxes: bool = True) -> np.ndarray:
    """
    Visualize object recognition results with labels and optional bounding boxes

    Args:
        image: Original or segmented image
        label_image: 2D array of cluster IDs
        classifications: Dictionary mapping cluster IDs to class names
        draw_boxes: Whether to draw bounding boxes around objects

    Returns:
        Annotated image
    """
    # Create result image without copying - only copy if we need to modify
    result = image if not draw_boxes else image.copy()

    for cluster_id, class_name in classifications.items():
        # Create mask for current cluster - use mask directly without copying
        mask = label_image == cluster_id
        if not np.any(mask):
            continue

        # Find cluster boundaries efficiently using numpy operations
        y_indices, x_indices = np.nonzero(mask)
        if len(y_indices) == 0:
            continue

        # Calculate boundaries directly
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        centroid = ((x_min + x_max) // 2, (y_min + y_max) // 2)

        # Draw bounding box if requested
        if draw_boxes:
            cv2.rectangle(result,
                         (x_min, y_min),
                         (x_max, y_max),
                         (0, 255, 0), 2)

        # Add text label - calculate text size to better position text
        text_size = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = max(centroid[0] - text_size[0] // 2, 0)  # Center text horizontally
        text_y = centroid[1]  # Keep vertical position

        cv2.putText(result,
                   class_name,
                   (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.7,
                   (0, 255, 0),
                   2)

    return result


def create_montage (images: List[np.ndarray],titles: List[str]) -> np.ndarray:
    """
    Create a montage of images for visualization
    """
    n_images = len (images)
    n_cols = min (3, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols

    cell_height = max (img.shape[0] for img in images)
    cell_width = max (img.shape[1] for img in images)

    montage = np.zeros ((cell_height * n_rows, cell_width * n_cols, 3), dtype=np.uint8)

    for idx, (img, title) in enumerate (zip (images, titles)):
        i, j = idx // n_cols, idx % n_cols
        y, x = i * cell_height, j * cell_width

        # Resize if necessary
        if img.shape[:2] != (cell_height, cell_width):
            img = cv2.resize (img, (cell_width, cell_height))

        # Convert grayscale to BGR if needed
        if len (img.shape) == 2:
            img = cv2.cvtColor (img, cv2.COLOR_GRAY2BGR)

        montage[y:y + cell_height, x:x + cell_width] = img

        # Add title
        cv2.putText (montage,
                     title,
                     (x + 10, y + 30),
                     cv2.FONT_HERSHEY_SIMPLEX,
                     1,
                     (255, 255, 255),
                     2)

    return montage


def save_visualization (save_dir: Path,
                        scene_id: str,
                        original_image: np.ndarray,
                        label_image: np.ndarray,
                        result_image: np.ndarray,
                        classifications: Dict[int, str]):
    """Save comprehensive visualization results"""
    # Create visualization montage
    label_viz = visualize_label_image(label_image)

    montage = create_montage (
        [original_image, label_viz, result_image],
        ["Original", "Clusters", "Recognition Results"]
    )

    # Save images
    cv2.imwrite (str (save_dir / f"{scene_id}_montage.png"), montage)
    cv2.imwrite (str (save_dir / f"{scene_id}_labels.png"), label_viz)
    cv2.imwrite (str (save_dir / f"{scene_id}_result.png"), result_image)

    # Save classifications to text file
    with open (save_dir / f"{scene_id}_classifications.txt", 'w') as f:
        for cluster_id, class_name in classifications.items ():
            f.write (f"Cluster {cluster_id}: {class_name}\n")