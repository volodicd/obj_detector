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


### Environment Setup

1. Requires Python 11 and pip.

2. Run the setup script:
# Create and activate virtual environment
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
pip install -r requirements.txt



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

2. Run inference(change scene number in the script):
```bash
python main_deep.py 
```

## Data Organization

Place your data folder in the root as follows:
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

## Parameters and Configuration

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


## Dataset Information

Supports recognition of 7 objects:
- Book
- Cookiebox
- Cup
- Ketchup
- Sugar bowl
- Sweets
- Tea
