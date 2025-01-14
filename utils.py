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

