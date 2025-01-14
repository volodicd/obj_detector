#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3D Object Recognition using a Pre-trained CNN (ResNet50) + Data Augmentation.
We remove the last layer of ResNet, add our own classification head, and train on projected .pcd data.

Steps:
1) Load point cloud data from "data/training" & "data/test".
2) Project the 3D points to 2D color images using your 'project_points_to_image'.
3) Feed the images into a pre-trained ResNet50. We freeze most layers except the new final head.
4) Train & evaluate on your small dataset, using heavy data augmentation to reduce overfitting.
5) Optionally produce confusion matrix & classification report, and attempt single-file predictions.
"""

import open3d as o3d
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms

import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report

# ---------------------------------------------------------------------
# 1) Environment / Device
# ---------------------------------------------------------------------
# If you have MPS on an Apple Silicon device:
if torch.backends.mps.is_available ():
    device = torch.device ("mps")
    print ("Using device: MPS (Apple Silicon)")
elif torch.cuda.is_available ():
    device = torch.device ("cuda")
    print ("Using device: CUDA")
else:
    device = torch.device ("cpu")
    print ("Using device: CPU")

# ---------------------------------------------------------------------
# 2) Import your projection function
#    This function is assumed to return:
#       (points_2d, color_image)
#    'color_image' is np.uint8, shape (H, W, 3).
# ---------------------------------------------------------------------
from projection import project_points_to_image  # Must remain unchanged per instructions


# ---------------------------------------------------------------------
# 3) Custom Dataset
# ---------------------------------------------------------------------
class PointCloudDataset (Dataset):
    """
    Expects .pcd files in data_dir, named e.g.:
       book001.pcd => class 'book'
       cookiebox13.pcd => class 'cookiebox'
    etc.
    - We parse the class by removing digits from the filename.
    - We project the 3D cloud into a 2D color image.
    - Then we apply transforms (including data augmentation).
    - Return (image_tensor, class_idx).
    """

    def __init__ (self, data_dir: str, transform=None):
        self.data_dir = Path (data_dir)
        self.transform = transform
        self.samples = []
        self.classes = set ()

        # Identify all classes
        for pcd_file in self.data_dir.glob ("*.pcd"):
            stem = pcd_file.stem  # e.g., "book003"
            class_name = "".join ([c for c in stem if not c.isdigit ()])
            self.classes.add (class_name)

        self.classes = sorted (list (self.classes))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate (self.classes)}

        # Gather all pcd files & map them to class indices
        for pcd_file in self.data_dir.glob ("*.pcd"):
            stem = pcd_file.stem
            class_name = "".join ([c for c in stem if not c.isdigit ()])
            if class_name in self.class_to_idx:
                class_idx = self.class_to_idx[class_name]
                self.samples.append ((str (pcd_file), class_idx))

    def __len__ (self):
        return len (self.samples)

    def __getitem__ (self, idx):
        pcd_path, class_idx = self.samples[idx]
        # Read point cloud
        pcd = o3d.io.read_point_cloud (pcd_path)
        # Project to 2D
        points_2d, color_image = project_points_to_image (
            np.asarray (pcd.points),
            colors=np.asarray (pcd.colors)
        )
        # color_image => shape(H, W, 3), dtype=uint8

        # Possibly transpose if shape is (3,H,W):
        # if color_image.shape[0] == 3:
        #     color_image = color_image.transpose(1,2,0)

        # Apply transforms
        if self.transform is not None:
            color_image = self.transform (color_image)  # => Tensor

        return color_image, class_idx


# ---------------------------------------------------------------------
# 4) Data Transforms (with heavy augmentation for training)
# ---------------------------------------------------------------------
train_transforms = transforms.Compose ([
    transforms.ToPILImage (),
    # Data augmentation:
    transforms.RandomHorizontalFlip (p=0.5),
    transforms.RandomRotation (15),
    transforms.ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.Resize ((224, 224)),
    transforms.ToTensor (),
    transforms.Normalize ([0.485, 0.456, 0.406],
                          [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose ([
    transforms.ToPILImage (),
    transforms.Resize ((224, 224)),
    transforms.ToTensor (),
    transforms.Normalize ([0.485, 0.456, 0.406],
                          [0.229, 0.224, 0.225])
])

train_dir = "data/training"
val_dir = "data/test"

train_dataset = PointCloudDataset (train_dir, transform=train_transforms)
val_dataset = PointCloudDataset (val_dir, transform=val_transforms)

# DataLoaders (set num_workers=0 on macOS M1 to avoid stalling)
train_loader = DataLoader (train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader (val_dataset, batch_size=8, shuffle=False, num_workers=0)

print (f"Num train samples = {len (train_dataset)}")
print (f"Num val samples   = {len (val_dataset)}")
print (f"Training classes = {train_dataset.classes}")


# ---------------------------------------------------------------------
# 5) Create a Pre-trained Model, removing last layer
# ---------------------------------------------------------------------
def create_model (num_classes):
    # Load pre-trained ResNet50
    model = models.resnet50 (pretrained=True)

    # Freeze all layers
    for param in model.parameters ():
        param.requires_grad = False

    # Remove final layer => replace with a smaller classification head
    in_features = model.fc.in_features  # 2048 for ResNet50
    model.fc = nn.Sequential (
        nn.Linear (in_features, 256),
        nn.ReLU (),
        nn.Dropout (0.3),
        nn.Linear (256, num_classes)
    )
    return model


model = create_model (len (train_dataset.classes)).to (device)

# We train only the last layer by default
criterion = nn.CrossEntropyLoss ()
optimizer = optim.Adam (model.fc.parameters (), lr=0.001)


# ---------------------------------------------------------------------
# 6) Training Loop
# ---------------------------------------------------------------------
def train_model (model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    best_acc = 0.0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range (num_epochs):
        print (f"\nEpoch {epoch}/{num_epochs - 1}")
        print ("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train ()
                dataloader = train_loader
            else:
                model.eval ()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm (dataloader, desc=f"{phase} epoch {epoch}"):
                inputs, labels = inputs.to (device), labels.to (device)
                optimizer.zero_grad ()

                with torch.set_grad_enabled (phase == "train"):
                    outputs = model (inputs)
                    _, preds = torch.max (outputs, 1)
                    loss = criterion (outputs, labels)

                    if phase == "train":
                        loss.backward ()
                        optimizer.step ()

                running_loss += loss.item () * inputs.size (0)
                running_corrects += (preds == labels).sum ().item ()

            epoch_loss = running_loss / len (dataloader.dataset)
            epoch_acc = running_corrects / len (dataloader.dataset)

            if phase == "train":
                history["train_loss"].append (epoch_loss)
                history["train_acc"].append (epoch_acc)
            else:
                history["val_loss"].append (epoch_loss)
                history["val_acc"].append (epoch_acc)

            print (f"{phase} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            # Save best model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save (model.state_dict (), "best_model.pth")

    print (f"\nTraining complete. Best val acc: {best_acc:.4f}")
    return model, history


model, history = train_model (model, train_loader, val_loader, criterion, optimizer, num_epochs=10)


# ---------------------------------------------------------------------
# 7) Evaluate on Validation Set: Confusion Matrix & Report
# ---------------------------------------------------------------------
def evaluate_model (model, loader):
    model.eval ()
    all_preds = []
    all_labels = []

    with torch.no_grad ():
        for inputs, labels in loader:
            inputs, labels = inputs.to (device), labels.to (device)
            outputs = model (inputs)
            _, preds = torch.max (outputs, 1)
            all_preds.extend (preds.cpu ().numpy ())
            all_labels.extend (labels.cpu ().numpy ())

    return np.array (all_labels), np.array (all_preds)


val_labels, val_preds = evaluate_model (model, val_loader)
classes = val_dataset.classes

# Confusion Matrix
cm = confusion_matrix (val_labels, val_preds)
plt.figure (figsize=(6, 5))
sns.heatmap (cm, annot=True, cmap="Blues", xticklabels=classes, yticklabels=classes, fmt='d')
plt.title ("Confusion Matrix (Validation)")
plt.xlabel ("Predicted")
plt.ylabel ("True")
plt.show ()

# Classification Report
report = classification_report (val_labels, val_preds, target_names=classes)
print ("Classification Report:")
print (report)


# ---------------------------------------------------------------------
# 8) Single PCD Prediction
# ---------------------------------------------------------------------
def predict_pointcloud (model, pcd_path: str, transform_fn):
    """
    Predict the class of a single .pcd using the trained model.
    We do the same steps: read pcd => project => transform => forward pass
    """
    model.eval ()
    pcd = o3d.io.read_point_cloud (pcd_path)

    points_2d, color_image = project_points_to_image (
        np.asarray (pcd.points),
        colors=np.asarray (pcd.colors)
    )
    # color_image => shape(H, W, 3), dtype=uint8

    tensor_img = transform_fn (color_image).unsqueeze (0).to (device)
    with torch.no_grad ():
        outputs = model (tensor_img)
        _, preds = torch.max (outputs, 1)
    predicted_idx = preds[0]
    return train_dataset.classes[predicted_idx]


# Example usage
test_pcd = "data/test/image000.pcd"
if Path (test_pcd).exists ():
    pred = predict_pointcloud (model, test_pcd, transform_fn=val_transforms)
    print (f"Predicted class for {test_pcd}: {pred}")
else:
    print (f"No test file found at {test_pcd}")

# ---------------------------------------------------------------------
# 9) (Optional) Plot Training Curves
# ---------------------------------------------------------------------
epochs_range = range (len (history["train_loss"]))

plt.figure (figsize=(12, 5))
plt.subplot (1, 2, 1)
plt.plot (epochs_range, history["train_loss"], label="Train Loss")
plt.plot (epochs_range, history["val_loss"], label="Val Loss")
plt.title ("Loss")
plt.legend ()

plt.subplot (1, 2, 2)
plt.plot (epochs_range, history["train_acc"], label="Train Acc")
plt.plot (epochs_range, history["val_acc"], label="Val Acc")
plt.title ("Accuracy")
plt.legend ()

plt.show ()
