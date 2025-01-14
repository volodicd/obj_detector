#!/bin/bash

set -e  # Exit on error

echo "Setting up environment for Machine Vision Exercise 5..."

# Ensure Python 3.11 is installed
chmod +x install_python.sh && ./install_python.sh

# Create and activate virtual environment
echo "Creating virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip and install dependencies
echo "Installing dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

# Create required directories
echo "Creating directories..."
for dir in data/training data/test results results_deep; do
    [[ ! -d $dir ]] && mkdir -p $dir
done

# Verify environment setup
echo "Verifying environment..."
python -c "
import sys, open3d, torch, cv2, numpy as np
print(f'Python {sys.version}')
print(f'OpenCV {cv2.__version__}')
print(f'PyTorch {torch.__version__}')
print(f'Open3D {open3d.__version__}')
print('All key packages verified successfully!')
"

echo -e "\nSetup complete! Quick start:
1. Add training data to: data/training/
2. Add test data to: data/test/
3. Activate environment: source venv/bin/activate
4. Run traditional pipeline: python main.py
   Or deep learning pipeline: python main_deep
