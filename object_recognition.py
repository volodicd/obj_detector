from typing import Dict, List
import numpy as np
import cv2


class ObjectRecognizer:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
        self.training_data = {}
        self.descriptors_per_class = {}


    def train(self, class_name: str, descriptors: np.ndarray):
        if class_name not in self.training_data:
            self.training_data[class_name] = []
            self.descriptors_per_class[class_name] = 0
        self.training_data[class_name].append(descriptors)
        self.descriptors_per_class[class_name] += len(descriptors)


    def match_features(self, desc1: np.ndarray, desc2: np.ndarray, ratio_threshold: float = 0.7) -> List[cv2.DMatch]:
        if desc1 is None or desc2 is None:
            return []
        # Find 2 nearest matches for each descriptor
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
        return good_matches


    def recognize_objects(self, scene_image: np.ndarray, label_image: np.ndarray) -> Dict[int, str]:
        """
        Recognize objects in the scene using per-cluster voting

        Args:
            scene_image: Projected and processed 2D image
            label_image: 2D array where each pixel value is the cluster ID

        Returns:
            Dictionary mapping cluster IDs to object class names
        """
        # Extract scene features
        scene_keypoints, scene_descriptors = self.sift.detectAndCompute(scene_image, None)
        if scene_descriptors is None:
            return {}

        unique_clusters = np.unique(label_image)
        unique_clusters = unique_clusters[unique_clusters != -1]  # Remove background
        # Initialize voting dictionary
        cluster_votes = {cid: {class_name: 0 for class_name in self.training_data.keys()}
                         for cid in unique_clusters}

        for class_name, train_descriptor_list in self.training_data.items():
            for train_descriptors in train_descriptor_list:
                matches = self.match_features(scene_descriptors, train_descriptors)

                for match in matches:
                    kp = scene_keypoints[match.queryIdx]
                    x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
                    if 0 <= y < label_image.shape[0] and 0 <= x < label_image.shape[1]:
                        cluster_id = label_image[y, x]
                        if cluster_id != -1:  # If keypoint is on a valid cluster
                            cluster_votes[cluster_id][class_name] += 1

        classifications = {}
        for cluster_id in cluster_votes.keys():
            votes = cluster_votes[cluster_id]
            if not any(votes.values()):
                continue
            normalized_votes = normalize_votes(votes, self.descriptors_per_class)
            best_class = max(normalized_votes.items(), key=lambda x: x[1])[0]
            classifications[cluster_id] = best_class

        return classifications


def normalize_votes(votes: Dict[str, int], descriptors_per_class: Dict[str, int]) -> Dict[str, float]:
    """
    Normalize vote counts by number of descriptors per class
    to avoid bias towards classes with more features
    """
    normalized = {}
    for class_name, vote_count in votes.items():
        if class_name in descriptors_per_class and descriptors_per_class[class_name] > 0:
            normalized[class_name] = vote_count / descriptors_per_class[class_name]
        else:
            normalized[class_name] = 0
    return normalized