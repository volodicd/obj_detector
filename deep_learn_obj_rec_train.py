import matplotlib
matplotlib.use('Agg')  # Force non-gui (was trained on the cloud)
import matplotlib.pyplot as plt
import seaborn as sns

import open3d as o3d
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms

import numpy as np
import random
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import confusion_matrix, classification_report
from projection import project_points_to_image


# chosing the device (best on Nvidia GPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device: MPS (M proc)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using device: CUDA")
else:
    device = torch.device("cpu")
    print("Using device: CPU")



class PointCloudDataset(Dataset):
    """
    Reads .pcd files from a list of (pcd_path, class_idx).
    Projects to 2D, transforms to tensor.
    Returns (image_tensor, class_idx).
    """
    def __init__(self, samples, class_to_idx, transform=None):
        """
        samples: list of (pcd_path, class_idx)
        class_to_idx: dict mapping class_name -> index
        transform: Torch transform pipeline
        """
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pcd_path, class_idx = self.samples[idx]
        pcd_path = Path(pcd_path)
        pcd = o3d.io.read_point_cloud(str(pcd_path))

        # Project to 2D
        points_2d, color_image = project_points_to_image(
            np.asarray(pcd.points),
            colors=np.asarray(pcd.colors)
        )
        # color_image => uint8 (H, W, 3)
        if self.transform is not None:
            color_image = self.transform(color_image)

        return color_image, class_idx



train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# splitting train/validation 80:20
train_dir = Path("data/training")

all_files = sorted(list(train_dir.glob("*.pcd")))

class_names_set = set()
samples_all = []

for pcd_file in all_files:
    stem = pcd_file.stem
    class_name = "".join(c for c in stem if not c.isdigit())
    class_names_set.add(class_name)
    samples_all.append((str(pcd_file), class_name))

class_names = sorted(list(class_names_set))
class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}

# Convert each (path, class_name) -> (path, class_idx)
samples_mapped = []
for (path_str, cls_n) in samples_all:
    if cls_n in class_to_idx:
        class_idx = class_to_idx[cls_n]
        samples_mapped.append((path_str, class_idx))

random.shuffle(samples_mapped)

train_size = int(0.8 * len(samples_mapped))
train_samples = samples_mapped[:train_size]
val_samples   = samples_mapped[train_size:]

print(f"Split: {len(train_samples)} train samples, {len(val_samples)} val samples.")

train_dataset = PointCloudDataset(train_samples, class_to_idx, transform=train_transforms)
val_dataset   = PointCloudDataset(val_samples,   class_to_idx, transform=val_transforms)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,  num_workers=0) # do not increase num_workers on MacOS(its crashing)
val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False, num_workers=0)

print(f"Train loader: {len(train_loader.dataset)} samples")
print(f"Val loader:   {len(val_loader.dataset)} samples")

# Create pretrained ResNet50 model
def create_model(num_classes: int):
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    # Replace final FC
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model

num_classes = len(class_names)
model = create_model(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    best_acc = 0.0
    history = {
        "train_loss": [],
        "train_acc":  [],
        "val_loss":   [],
        "val_acc":    []
    }

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}/{num_epochs - 1}")
        print("-"*10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloader, desc=f"{phase} epoch {epoch}"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc  = running_corrects / len(dataloader.dataset)

            if phase == "train":
                history["train_loss"].append(epoch_loss)
                history["train_acc"].append(epoch_acc)
            else:
                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(epoch_acc)

            print(f"{phase} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), "best_model.pth")

    print(f"\nTraining complete. Best val acc: {best_acc:.4f}")
    return model, history

model, history = train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs=10 # 10 epochs are enough to get 90% accuracy
)


def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)

val_labels, val_preds = evaluate_model(model, val_loader)

labels_list = list(range(num_classes))  # e.g. [0..(num_classes-1)]
cm = confusion_matrix(val_labels, val_preds, labels=labels_list)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm, annot=True, cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    fmt='d'
)
plt.title("Confusion Matrix (Val - 80/20 Split)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix.png")

report = classification_report(
    val_labels, val_preds,
    labels=labels_list,
    target_names=class_names,
    zero_division=0
)
print("Classification Report:")
print(report)

torch.save(model, "deeplearn_model.pth")

# adding plots for training curves
epochs_range = range(len(history["train_loss"]))

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(epochs_range, history["train_loss"], label="Train Loss")
plt.plot(epochs_range, history["val_loss"],   label="Val Loss")
plt.title("Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs_range, history["train_acc"],  label="Train Acc")
plt.plot(epochs_range, history["val_acc"],    label="Val Acc")
plt.title("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png")
