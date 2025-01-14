import numpy as np
import cv2
from typing import Dict, List
from pathlib import Path

def visualize_label_image(label_image: np.ndarray) -> np.ndarray:
    """
    Visualize label image with different colors for each cluster

    Args:
        label_image: 2D array of cluster IDs
    Returns:
        Colored visualization of clusters
    """
    unique_labels = np.unique(label_image)
    # -1 represents background pixels
    unique_labels = unique_labels[unique_labels != -1]
    n_labels = len(unique_labels)
    vis_image = np.zeros((*label_image.shape, 3), dtype=np.uint8)
    for i, label in enumerate(unique_labels):
        hue = int(180 * i / max(n_labels, 1))
        # Label 0: hue = 180 * 0/3 = 0   (Red)
        # Label 1: hue = 180 * 1/3 = 60  (Green)
        # Label 2: hue = 180 * 2/3 = 120 (Blue)
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
        vis_image[label_image == label] = color
    return vis_image


def visualize_recognition_results(image: np.ndarray, label_image: np.ndarray,
                                  classifications: Dict[int, str], draw_boxes: bool = True) -> np.ndarray:
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
    # For debugging - will not modify the original image if we are not drawing boxes
    result = image if not draw_boxes else image.copy()
    for cluster_id, class_name in classifications.items():
        # identifying pixels belonging to each object
        mask = label_image == cluster_id
        if not np.any(mask):
            continue
        # getting coordinates of object pixels
        y_indices, x_indices = np.nonzero(mask)
        if len(y_indices) == 0:
            continue
        # find the edge pixels and center for bounding box of the object
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        centroid = ((x_min + x_max) // 2, (y_min + y_max) // 2)
        # draw bounding box and label
        if draw_boxes: cv2.rectangle(result, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        text_size = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = max(centroid[0] - text_size[0] // 2, 0)
        text_y = centroid[1]
        cv2.putText(result, class_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return result


def create_montage(images: List[np.ndarray], titles: List[str]) -> np.ndarray:
    """
    Create a montage grid of images with titles.

    The function arranges multiple images in a grid format with maximum 3 columns.
    Each image is resized to match the largest image dimensions, converted to BGR if grayscale,
    and labeled with its corresponding title.

    Args:
        images: List of images to arrange in grid (can be BGR or grayscale)
        titles: List of titles corresponding to each image

    Returns:
        Combined montage image with all input images arranged in a grid
    """
    n_images = len(images)
    n_cols = min(3, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    # find maximum cell dimensions to make uniform grid
    cell_height = max(img.shape[0] for img in images)
    cell_width = max(img.shape[1] for img in images)
    # creating empty montage array
    montage = np.zeros((cell_height * n_rows, cell_width * n_cols, 3), dtype=np.uint8)

    for idx, (img, title) in enumerate(zip(images, titles)):
        i, j = idx // n_cols, idx % n_cols
        y, x = i * cell_height, j * cell_width
        if img.shape[:2] != (cell_height, cell_width):
            img = cv2.resize(img, (cell_width, cell_height))
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        montage[y:y + cell_height, x:x + cell_width] = img
        cv2.putText(montage, title, (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return montage


def save_visualization(save_dir: Path, scene_id: str, original_image: np.ndarray,
                       label_image: np.ndarray, result_image: np.ndarray):
    label_viz = visualize_label_image(label_image)
    montage = create_montage(
        [original_image, label_viz, result_image],
        ["Original", "Clusters", "Recognition Results"]
    )
    cv2.imwrite(str(save_dir / f"{scene_id}_montage.png"), montage)
    cv2.imwrite(str(save_dir / f"{scene_id}_labels.png"), label_viz)
    cv2.imwrite(str(save_dir / f"{scene_id}_result.png"), result_image)