from typing import Tuple
import numpy as np
import math
import open3d as o3d

def remove_ground_plane(pcd, confidence = 0.99, inlier_threshold = 0.01, min_points = 100) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
      Remove the ground plane from a point cloud using RANSAC.
      Returns the point cloud with ground plane removed and the plane coefficients.

      Args:
          pcd: Input point cloud
          confidence: RANSAC confidence level (default: 0.99)
          inlier_threshold: Max distance for inlier points (default: 0.01 meters)
          min_points: Minimum number of points to consider a valid plane

      Returns:
          Tuple containing:
          - Point cloud with ground plane removed
          - Plane coefficients [a, b, c, d] for plane equation ax + by + cz + d = 0
      """
    plane_model, inliers = pcd.segment_plane(distance_threshold=inlier_threshold, ransac_n=3, num_iterations=1000)
    if len(inliers) < min_points: # Not enough points to fit a plane
        plane_model, inliers = fit_plane_ransac(pcd, confidence, inlier_threshold)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    return outlier_cloud, plane_model

def fit_plane_ransac(pcd: o3d.geometry.PointCloud, confidence: float, inlier_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(pcd.points)
    n_points = points.shape[0]

    # imit best plane
    best_plane = np.array([0., 0., 1., 0.])
    best_inliers = np.zeros(n_points, dtype=bool)
    max_inliers_count = 0

    outlier_ratio = 0.3
    p_inliers = 1.0 - outlier_ratio
    p_triplet = max(p_inliers ** 3, 1e-9)
    """
    The pobability of selecting three iniliers points is (lets take our case) is 0.7^3 = 0.343, so we can calculate statistically best 
    number of iterations. Cause probability of failure is 1 - 0.343 = 0.657, probability of failure after n iterations is 0.657^n
    We set 0.99 confidence, then 0.657^n = 0.01, n = log(0.01) / log(0.657) 
    in this line we are solving this equation, but limmiting number of iterations to 5000
    not sure if I should describe it in this way, but I think it's good to know how we get this number
    """
    n_iter = int(np.clip(math.log(1 - confidence) / math.log(1 - p_triplet), 500, 5000))

    for _ in range(n_iter):
        # chosing random 3 points
        idx = np.random.choice(n_points, 3, replace=False)
        p1, p2, p3 = points[idx]
        normal = np.cross(p2 - p1, p3 - p1)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-9:
            continue
        normal /= norm_len
        a,b,c = normal
        d = -np.dot(normal, p1)
        distances = np.abs(np.dot(points, normal) + d)

        inliers = distances < inlier_threshold
        inliers_count = np.sum(inliers)

        # If we found better points (with more inliers) - updating it
        if inliers_count > max_inliers_count:
            best_plane = np.array([a,b,c,d])
            best_inliers = inliers.copy()
            max_inliers_count = inliers_count

        """
        This is the part where we are trying to refine our plane, we are taking all inliers and trying to fit plane to them.
        
        """
        if max_inliers_count >= 3:
            inlier_points = points[best_inliers]
            # Create matrix A for least squares: Ax = b
            # Add column of ones for d parameter
            A = np.hstack([inlier_points, np.ones((inlier_points.shape[0], 1))])
            b = np.zeros(inlier_points.shape[0])
            refined_plane, *_ = np.linalg.lstsq(A, b, rcond=None)
            normal = refined_plane[:3]
            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-9:
                refined_plane[:3] /= norm_len
                # if our normal is pointing down, we should flip it
                if refined_plane[2] < 0:
                    refined_plane = -refined_plane
                best_plane = refined_plane

    return best_plane, best_inliers