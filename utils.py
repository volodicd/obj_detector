#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Utility functions for 3D object recognition pipeline"""

from pathlib import Path
import time
import numpy as np
import open3d as o3d
import cv2
from typing import Dict, List, Tuple


def load_pointcloud(filepath: str) -> o3d.geometry.PointCloud:
    """Load point cloud and remove zero points"""
    pcd = o3d.io.read_point_cloud(filepath)

    # Remove origin points (0,0,0)
    points_numpy = np.array(pcd.points)
    indices_origin_points = list(np.where(np.all(points_numpy == [0, 0, 0], axis=1))[0])
    pcd = pcd.select_by_index(indices_origin_points, invert=True)

    return pcd


def load_training_data(training_dir: Path) -> Dict[str, List[o3d.geometry.PointCloud]]:
    """Load training point clouds for each object class"""
    training_data = {}

    for pcd_path in training_dir.glob("**/*.pcd"):
        class_name = pcd_path.parent.name
        if class_name not in training_data:
            training_data[class_name] = []

        pcd = load_pointcloud(str(pcd_path))
        training_data[class_name].append(pcd)

    return training_data


def timing_decorator(func):
    """Decorator to measure function execution time"""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.3f} seconds")
        return result

    return wrapper


def evaluate_recognition(predictions: Dict[str, str],
                         ground_truth: Dict[str, str]) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate object recognition performance

    Args:
        predictions: Dictionary mapping object IDs to predicted classes
        ground_truth: Dictionary mapping object IDs to true classes

    Returns:
        Tuple containing:
        - Overall accuracy
        - Per-class accuracy dictionary
    """
    if not predictions or not ground_truth:
        return 0.0, {}

    correct = 0
    class_correct = {}
    class_total = {}

    for obj_id, true_class in ground_truth.items():
        if true_class not in class_total:
            class_total[true_class] = 0
            class_correct[true_class] = 0

        class_total[true_class] += 1
        if obj_id in predictions and predictions[obj_id] == true_class:
            correct += 1
            class_correct[true_class] += 1

    # Calculate accuracies
    overall_acc = correct / len(ground_truth)
    class_acc = {cls: class_correct[cls] / class_total[cls]
                 for cls in class_total.keys()}

    return overall_acc, class_acc


def save_results(results_dir: Path,
                 scene_id: str,
                 image: np.ndarray,
                 predictions: Dict[str, str]) -> None:
    """Save recognition results"""
    results_dir.mkdir(exist_ok=True)

    # Save annotated image
    cv2.imwrite(str(results_dir / f"{scene_id}_result.png"), image)

    # Save predictions to text file
    with open(str(results_dir / f"{scene_id}_predictions.txt"), 'w') as f:
        for obj_id, pred_class in predictions.items():
            f.write(f"Object {obj_id}: {pred_class}\n")


def visualize_plane_removal(pcd: o3d.geometry.PointCloud,
                            inliers: np.ndarray,
                            plane_eq: np.ndarray) -> None:
    """
    Visualize the results of plane removal

    Args:
        pcd: Original point cloud
        inliers: Array indicating which points are inliers
        plane_eq: Plane equation coefficients [a,b,c,d]
    """
    # Create copy for visualization
    viz_pcd = o3d.geometry.PointCloud(pcd)

    # Color the plane points red and the rest in their original colors
    colors = np.asarray(viz_pcd.colors)
    plane_mask = inliers > 0  # Assuming inliers are boolean or 0/1
    colors[plane_mask] = [1, 0, 0]  # Red for plane points
    viz_pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([viz_pcd])