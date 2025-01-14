from typing import Tuple, List
import numpy as np
import open3d as o3d


def cluster_objects(pcd: o3d.geometry.PointCloud,
                     eps: float = 0.02,
                     min_points: int = 100) -> Tuple[List[o3d.geometry.PointCloud], np.ndarray]:
    """
    Cluster objects in point cloud after ground plane removal.

    Args:
        pcd: Input point cloud
        eps: DBSCAN epsilon parameter (cluster radius)
        min_points: Minimum points for a cluster

    Returns:
        Tuple containing:
        - List of point clouds, one for each detected object
        - Array of labels for each point in the original cloud
    """
    labels = np.array(dbscan(np.asarray(pcd.points), eps=eps, min_samples=min_points))
    unique_labels = np.unique(labels)
    clusters = []
    # Filter out noise points (label -1) and small clusters
    for label in unique_labels:
        if label == -1:  # Skip noise points
            continue
        cluster_points = pcd.select_by_index(np.where(labels == label)[0])
        if len(np.asarray(cluster_points.points)) < min_points:
            continue
        clusters.append(cluster_points)
    return clusters, labels


def dbscan(points: np.ndarray,
           eps: float = 0.1,
           min_samples: int = 50) -> np.ndarray:
    """ Find clusters in the provided data coming from a pointcloud using the DBSCAN algorithm.

    :param points: The (down-sampled) points of the pointcloud to be clustered
    :type points: np.ndarray with shape=(n_points, 3)

    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :type eps: float

    :param min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core
        point. This includes the point itself.
    :type min_samples: float

    :return: Labels array with a different label for each cluster for each point (shape = (n_points,) and dtype=int)
            The label -1 is assigned to points that are considered to be noise.
    :rtype: np.ndarray
    """
    ######################################################
    # Write your own code here
    n_points = len(points)
    labels = np.full(n_points, -2, dtype=int)  # -2 = unvisited
    cluster_id = 0


    for i in range(n_points):
        # If visited (either assigned to cluster or noise) - skip
        if labels[i] != -2:
            continue

        dist_i = np.linalg.norm(points - points[i], axis=1)
        neighbors = np.where(dist_i <= eps)[0]

        if len(neighbors) < min_samples: # if noise
            labels[i] = -1
        else:
            cluster_id += 1
            labels[i] = cluster_id

            queue = list(neighbors)
            idx_ptr = 0
            while idx_ptr < len(queue):
                neighbor_idx = queue[idx_ptr]
                idx_ptr += 1

                # If neighbor is unvisited
                if labels[neighbor_idx] == -2:
                    labels[neighbor_idx] = cluster_id
                    dist_neighbor = np.linalg.norm(points - points[neighbor_idx], axis=1)
                    neighbors_of_neighbor = np.where(dist_neighbor <= eps)[0]

                    # If neighbor is also a core point, addint its neighbors to the queue
                    if len(neighbors_of_neighbor) >= min_samples:
                        queue.extend(neighbors_of_neighbor)

                elif labels[neighbor_idx] == -1:
                    # noise to border escalation
                    labels[neighbor_idx] = cluster_id

    # Replacing leftovers -2 (never visited) with -1 if any remain
    labels[labels == -2] = -1
    return labels