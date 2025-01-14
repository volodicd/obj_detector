from fit_plane import remove_ground_plane
from clustering import cluster_objects
from projection import project_pointcloud
from object_recognition import ObjectRecognizer
import open3d as o3d
from helper_functions import *


class ObjectRecognitionPipeline:
    def __init__(self):
        self.recognizer = ObjectRecognizer()
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)

    def load_training_data(self, training_dir: Path):
        print(f"Loading training data from: {training_dir}")
        for pcd_file in training_dir.glob("*.pcd"):
            filename = pcd_file.name

            class_name = ''.join([c for c in filename if not c.isdigit()]).rstrip('.')
            print(f"Processing class: {class_name}")

            if class_name not in self.recognizer.training_data:
                try:
                    pcd = o3d.io.read_point_cloud(str(pcd_file))

                    fake_labels = -1 * np.ones(len(pcd.points), dtype=int)
                    _, image, processed_image, _ = project_pointcloud(pcd, fake_labels)

                    keypoints, descriptors = self.recognizer.sift.detectAndCompute(processed_image, None)
                    if descriptors is not None:
                        self.recognizer.train(class_name, descriptors)
                except Exception as e:
                    print(f"Error processing {pcd_file.name}: {str(e)}")

    def process_scene(self, pcd: o3d.geometry.PointCloud, scene_id: str):
        """Process a single scene"""
        print("Removing ground plane...")
        filtered_pcd, plane_model = remove_ground_plane(pcd)
        voxel_size = 0.001
        filtered_pcd = filtered_pcd.voxel_down_sample(voxel_size)
        print("Clustering objects...")
        clusters, labels = cluster_objects(filtered_pcd)

        print("Projecting to 2D...")
        points_2d, image, processed_image, label_image = project_pointcloud(filtered_pcd, labels)

        print("Recognizing objects...")
        classifications = self.recognizer.recognize_objects(processed_image, label_image)

        result_image = visualize_recognition_results(
            processed_image,
            label_image,
            classifications
        )

        if self.results_dir is not None:
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
    pipeline = ObjectRecognitionPipeline()

    training_dir = Path("data/training")
    pipeline.load_training_data(training_dir)
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