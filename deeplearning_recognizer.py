import open3d as o3d
import torch
import torchvision.transforms as transforms
import numpy as np
from projection import project_points_to_image


class DeepLearningRecognizer:
    def __init__(self, model_path: str, class_names: list):
        """
        model_path : path to the .pth file with your trained model
        class_names: list of class names in the same order as used during training
        """
        self.device = torch.device("cpu") # enought for inference
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()

        self.class_names = class_names

        # The same transforms you used for inference:
        self.inference_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def predict_cluster(self, pcd: o3d.geometry.PointCloud) -> str:
        """
        Given a single cluster's point cloud, project to 2D and run inference.
        Returns the predicted class name.
        """
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        points_2d, color_image = project_points_to_image(points, colors)
        img_tensor = self.inference_transform(color_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            _, preds = torch.max(outputs, 1)

        predicted_idx = preds[0].item()
        return self.class_names[predicted_idx]
