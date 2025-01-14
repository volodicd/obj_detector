#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Helper functions for 3D object recognition pipeline"""

import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from pathlib import Path

from object_recognition import ObjectRecognizer


def visualize_label_image(label_image: np.ndarray, title: str = "Cluster Labels") -> np.ndarray:
    """
    Visualize label image with different colors for each cluster

    Args:
        label_image: 2D array of cluster IDs
        title: Title for the visualization

    Returns:
        Colored visualization of clusters
    """
    # Create colormap (excluding background)
    unique_labels = np.unique(label_image)
    unique_labels = unique_labels[unique_labels != -1]
    n_labels = len(unique_labels)

    # Create colored image
    vis_image = np.zeros((*label_image.shape, 3), dtype=np.uint8)

    # Use HSV color space for better color distribution
    for i, label in enumerate(unique_labels):
        hue = int(180 * i / max(n_labels, 1))
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
        vis_image[label_image == label] = color

    return vis_image


def visualize_recognition_results (image: np.ndarray,
                                   label_image: np.ndarray,
                                   classifications: Dict[int, str],
                                   recognizer: ObjectRecognizer,
                                   draw_boxes: bool = True) -> np.ndarray:
    """
    Memory-efficient visualization of recognition results
    """
    # Create base result image
    result = image.copy ()

    # Draw bounding boxes and labels
    for cluster_id, class_name in classifications.items ():
        mask = (label_image == cluster_id)
        if not np.any (mask):
            continue

        y_indices, x_indices = np.where (mask)
        if len (y_indices) == 0 or len (x_indices) == 0:
            continue

        x_min, x_max = np.min (x_indices), np.max (x_indices)
        y_min, y_max = np.min (y_indices), np.max (y_indices)
        centroid = (int ((x_min + x_max) / 2), int ((y_min + y_max) / 2))

        if draw_boxes:
            cv2.rectangle (result,
                           (x_min, y_min),
                           (x_max, y_max),
                           (0, 255, 0), 2)

        cv2.putText (result,
                     f"{class_name} (ID: {cluster_id})",
                     (centroid[0] - 20, centroid[1]),
                     cv2.FONT_HERSHEY_SIMPLEX,
                     0.7,
                     (0, 255, 0),
                     2)

    # Instead of storing all matches, show only best match per object
    if hasattr (recognizer, 'scene_keypoints') and hasattr (recognizer, 'best_matches'):
        # Calculate reasonable size for match visualization
        max_height = 200  # Limit height of match visualization
        scale = min (max_height / image.shape[0], 1.0)
        match_size = (int (image.shape[1] * scale), int (image.shape[0] * scale))

        # Create space for matches at bottom of result
        final_height = result.shape[0] + match_size[1]
        final_result = np.zeros ((final_height, result.shape[1], 3), dtype=np.uint8)
        final_result[:result.shape[0]] = result

        # Add single match visualization at bottom
        if recognizer.best_matches:
            # Take first match only to save memory
            class_name = list (recognizer.best_matches.keys ())[0]
            matches = recognizer.best_matches[class_name]

            match_img = recognizer.visualize_matches (
                cv2.resize (image, match_size),
                cv2.resize (image, match_size),
                matches[:100],  # Limit number of displayed matches
                recognizer.scene_keypoints,
                recognizer.scene_keypoints
            )

            final_result[result.shape[0]:] = cv2.resize (match_img, (result.shape[1], match_size[1]))

            # Add label
            cv2.putText (final_result,
                         f"Feature matches for {class_name}",
                         (10, result.shape[0] + 30),
                         cv2.FONT_HERSHEY_SIMPLEX,
                         0.7,
                         (255, 255, 255),
                         2)

        return final_result

    return result


def create_montage(images: List[np.ndarray],
                   titles: List[str],
                   layout: Tuple[int, int] = None) -> np.ndarray:
    """
    Create a montage of images for visualization
    """
    n_images = len(images)
    if layout is None:
        n_cols = min(3, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols
    else:
        n_rows, n_cols = layout

    cell_height = max(img.shape[0] for img in images)
    cell_width = max(img.shape[1] for img in images)

    montage = np.zeros((cell_height * n_rows, cell_width * n_cols, 3), dtype=np.uint8)

    for idx, (img, title) in enumerate(zip(images, titles)):
        i, j = idx // n_cols, idx % n_cols
        y, x = i * cell_height, j * cell_width

        # Resize if necessary
        if img.shape[:2] != (cell_height, cell_width):
            img = cv2.resize(img, (cell_width, cell_height))

        # Convert grayscale to BGR if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        montage[y:y + cell_height, x:x + cell_width] = img

        # Add title
        cv2.putText(montage,
                    title,
                    (x + 10, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2)

    return montage


def save_visualization(save_dir: Path,
                       scene_id: str,
                       original_image: np.ndarray,
                       label_image: np.ndarray,
                       result_image: np.ndarray,
                       classifications: Dict[int, str]):
    """Save comprehensive visualization results"""
    # Create visualization montage
    label_viz = visualize_label_image(label_image)

    montage = create_montage(
        [original_image, label_viz, result_image],
        ["Original", "Clusters", "Recognition Results"]
    )

    # Save images
    cv2.imwrite(str(save_dir / f"{scene_id}_montage.png"), montage)
    cv2.imwrite(str(save_dir / f"{scene_id}_labels.png"), label_viz)
    cv2.imwrite(str(save_dir / f"{scene_id}_result.png"), result_image)

    # Save classifications to text file
    with open(str(save_dir / f"{scene_id}_classifications.txt"), 'w') as f:
        for cluster_id, class_name in classifications.items():
            f.write(f"Cluster {cluster_id}: {class_name}\n")