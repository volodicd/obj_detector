import open3d as o3d
import numpy as np
from pathlib import Path

from fit_plane import remove_ground_plane
from clustering import cluster_objects
from projection import project_pointcloud
from helper_functions import visualize_recognition_results, save_visualization
from deeplearning_recognizer import DeepLearningRecognizer

class DeepLearningPipeline:
    def __init__(self, model_path: str, class_names: list):
        self.recognizer = DeepLearningRecognizer(model_path, class_names)
        self.results_dir = Path("results_deep")
        self.results_dir.mkdir(exist_ok=True)

    def process_scene(self, pcd: o3d.geometry.PointCloud, scene_id: str):
        print("Removing ground plane...")
        filtered_pcd, plane_model = remove_ground_plane(pcd)
        voxel_size = 0.001
        filtered_pcd = filtered_pcd.voxel_down_sample(voxel_size)
        print("Clustering objects...")
        clusters, labels = cluster_objects(filtered_pcd)
        print("Projecting entire scene for visualization...")
        _, color_image, processed_image, label_image = project_pointcloud(filtered_pcd, labels)

        classifications = {}
        for cluster_id, _ in enumerate(clusters):
            if cluster_id == -1:
                continue  # skip noise
            cluster_pcd = filtered_pcd.select_by_index(
                np.where(labels == cluster_id)[0]
            )
            predicted_class = self.recognizer.predict_cluster(cluster_pcd)
            classifications[cluster_id] = predicted_class

        result_image = visualize_recognition_results(
            processed_image,
            label_image,
            classifications
        )
        save_visualization(
            self.results_dir,
            scene_id,
            processed_image,
            label_image,
            result_image,
        )

        return classifications, result_image, label_image


def main():
    class_names = ['book','cookiebox','cup','ketchup','sugar','sweets','tea']
    model_path = "deeplearn_model.pth"
    pipeline = DeepLearningPipeline(model_path, class_names)

    pcd_file = Path("data/test/image003.pcd")
    scene_id = pcd_file.stem

    try:
        print(f"\nProcessing {pcd_file.name}")
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
