#!/bin/bash

# Exit on error
set -e

echo "Setting up Machine Vision Exercise 5 environment..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.8 or newer."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
python3 -m pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create required directories if they don't exist
echo "Creating project directories..."
mkdir -p data/training
mkdir -p data/test
mkdir -p results
mkdir -p results_deep

# Verify installation
echo "Verifying installation..."
python3 -c "
import sys
import open3d
import torch
import cv2
import numpy as np
print(f'Python {sys.version}')
print(f'OpenCV {cv2.__version__}')
print(f'PyTorch {torch.__version__}')
print(f'Open3D {open3d.__version__}')
print('All key packages verified successfully!')
"

echo "
Setup complete! Environment is ready to use.

Quick start:
1. Place training data in: data/training/
2. Place test data in: data/test/
3. Activate environment: source venv/bin/activate
4. Run traditional pipeline: python main.py
   Or deep learning pipeline: python main_deep.py

For more information, please read the README.md
"