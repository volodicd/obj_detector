# 3D Object Recognition Pipeline

This project implements a 3D object recognition system with both traditional (SIFT-based) and deep learning approaches. It processes point cloud data to detect and classify objects in scenes.

## Project Structure

```
mv_exercise5/
├── data/                      # Data directory
│   ├── training/             # Training point clouds
│   └── test/                 # Test scenes
├── results/                  # Results from SIFT pipeline
├── results_deep/            # Results from deep learning pipeline
├── .gitignore
├── camera_params.py         # Camera calibration parameters
├── clustering.py            # DBSCAN implementation
├── deep_learn_obj_rec.py    # Deep learning training script
├── deeplearn_model.pth      # Trained model weights
├── deeplearning_recognizer.py # Deep learning inference
├── fit_plane.py             # Ground plane removal
├── helper_functions.py      # Visualization utilities
├── main.py                  # Traditional SIFT pipeline
├── main_deep.py            # Deep learning pipeline
├── mv_ex3_cpu.yml          # Conda environment specification
├── object_recognition.py    # SIFT-based recognition
├── projection.py           # 3D to 2D projection
├── requirements.txt        # Python package requirements
└── setup.sh               # Environment setup script
```

## Quick Start

### Environment Setup

1. Make sure you have Conda installed (Miniconda or Anaconda)

2. Run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```

Alternatively, create the environment manually:
```bash
conda env create -f mv_ex3_cpu.yml
conda activate machinevision
```

### Running the Pipelines

#### Traditional SIFT Pipeline
```bash
python main.py
```

#### Deep Learning Pipeline
1. Train the model (if not using pre-trained):
```bash
python deep_learn_obj_rec_train.py
```

2. Run inference:
```bash
python main_deep.py
```

## Data Organization

Place your point cloud data as follows:
```
data/
├── training/           # Individual object point clouds
│   ├── book001.pcd
│   ├── cup001.pcd
│   └── ...
└── test/              # Scene point clouds for testing
    ├── scene001.pcd
    └── ...
```

## Configuration Files

- `mv_ex3_cpu.yml`: Conda environment specification
- `requirements.txt`: Additional Python package requirements
- `camera_params.py`: Camera calibration parameters

## Output Directories

### SIFT Pipeline (`results/`)
```
results/
├── scene001_montage.png        # Combined visualization
├── scene001_labels.png         # Cluster visualization
├── scene001_result.png         # Recognition results
└── scene001_classifications.txt # Detected objects list
```

### Deep Learning Pipeline (`results_deep/`)
Similar structure as above, with additional training artifacts:
- `confusion_matrix.png`
- `training_curves.png`

## Parameters and Configuration

### Camera Parameters
Located in `camera_params.py`:
- RGB camera intrinsics and distortion
- Depth camera parameters
- Transform between RGB and depth

### Recognition Parameters
1. Ground Plane Removal:
```python
confidence = 0.99
inlier_threshold = 0.01
min_points = 100
```

2. Clustering:
```python
eps = 0.02
min_points = 100
```

3. Deep Learning:
```python
batch_size = 8
num_epochs = 20
learning_rate = 0.001
```

## Troubleshooting

1. Environment Issues:
   - Run `conda env update -f mv_ex3_cpu.yml` to update environment
   - Check CUDA availability for deep learning
   - Verify OpenCV installation with `import cv2`

2. Data Issues:
   - Ensure PCD files are properly formatted
   - Check file permissions in data directories
   - Verify camera parameters match your setup

3. Runtime Issues:
   - Use `voxel_size` parameter for large point clouds
   - Monitor GPU memory usage
   - Check disk space for results

## Dataset Information

Supports recognition of 7 objects:
- Book
- Cookiebox
- Cup
- Ketchup
- Sugar bowl
- Sweets
- Tea
