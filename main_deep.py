#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main_deep.py: 3D Object Recognition using a deep-learning model
instead of SIFT-based matching.

Steps:
1) Remove ground plane from the scene
2) Cluster the remaining points
3) For each cluster, project to 2D and do CNN inference
4) Visualize bounding boxes & labels
"""

import open3d as o3d
import numpy as np
from pathlib import Path

from fit_plane import remove_ground_plane
from clustering import cluster_objects
from projection import project_pointcloud
from helper_functions import (
    visualize_recognition_results,
    save_visualization
)
# Import your new CNN recognizer
from deeplearning_recognizer import DeepLearningRecognizer

class DeepLearningPipeline:
    def __init__(self, model_path: str, class_names: list):
        self.recognizer = DeepLearningRecognizer(model_path, class_names)
        self.results_dir = Path("results_deep")
        self.results_dir.mkdir(exist_ok=True)

    def process_scene(self, pcd: o3d.geometry.PointCloud, scene_id: str):
        """Process a single scene with deep-learning approach."""
        print("Removing ground plane...")
        filtered_pcd, plane_model = remove_ground_plane(pcd)

        # Optionally down-sample
        voxel_size = 0.001
        filtered_pcd = filtered_pcd.voxel_down_sample(voxel_size)

        print("Clustering objects...")
        clusters, labels = cluster_objects(filtered_pcd)

        # (Optional) We can also do a 2D projection for the entire scene
        # just to have a background image for bounding box drawing
        print("Projecting entire scene for visualization...")
        _, color_image, processed_image, label_image = project_pointcloud(filtered_pcd, labels)

        # Now for each cluster, we do inference
        classifications = {}
        for cluster_id, _ in enumerate(clusters):
            # Extract sub-cloud for that cluster
            if cluster_id == -1:
                continue  # skip noise
            cluster_pcd = filtered_pcd.select_by_index(
                np.where(labels == cluster_id)[0]
            )
            predicted_class = self.recognizer.predict_cluster(cluster_pcd)
            classifications[cluster_id] = predicted_class

        # Draw bounding boxes + text
        result_image = visualize_recognition_results(
            processed_image,
            label_image,
            classifications
        )

        # Save the results
        save_visualization(
            self.results_dir,
            scene_id,
            processed_image,
            label_image,
            result_image,
            classifications
        )

        return classifications, result_image, label_image


def main():
    # Provide the same class names used during training
    class_names = ['book','cookiebox','cup','ketchup','sugar','sweets','tea']
    model_path = "deeplearn_model.pth"  # The file you exported after training

    pipeline = DeepLearningPipeline(model_path, class_names)

    # Now we do inference for each test pcd
    test_dir = Path("data/test")
    for pcd_file in test_dir.glob("*.pcd"):
        print(f"\nProcessing {pcd_file.name}")
        scene_id = pcd_file.stem
        try:
            scene_pcd = o3d.io.read_point_cloud(str(pcd_file))
            classifications, result_image, label_image = pipeline.process_scene(
                pcd=scene_pcd,
                scene_id=scene_id
            )
            print("Detected objects:")
            for cluster_id, class_name in classifications.items():
                print(f"  Cluster {cluster_id}: {class_name}")
        except Exception as e:
            print(f"Error processing {pcd_file.name}: {str(e)}")

if __name__ == "__main__":
    main()
