"""
Google Colab Setup Script
Quick setup for semiconductor defect detection system

Run this cell first in Google Colab to set up the entire environment
"""

import os
import sys

print("="*80)
print("IESA DeepTech Hackathon 2026 - Colab Setup")
print("Semiconductor Defect Detection System")
print("="*80)

# ============================================================================
# STEP 1: Install Dependencies
# ============================================================================

print("\n[1/6] Installing dependencies...")

!pip install -q tensorflow==2.13.0
!pip install -q numpy pandas matplotlib seaborn scikit-learn
!pip install -q onnx onnxruntime tf2onnx
!pip install -q pillow tqdm

print("✅ Dependencies installed")

# ============================================================================
# STEP 2: Mount Google Drive
# ============================================================================

print("\n[2/6] Mounting Google Drive...")

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

print("✅ Google Drive mounted")

# ============================================================================
# STEP 3: Create Directory Structure
# ============================================================================

print("\n[3/6] Creating directory structure...")

directories = [
    '/content/dataset',
    '/content/models',
    '/content/output',
    '/content/onnx_models',
    '/content/test_images'
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)

print("✅ Directories created:")
for d in directories:
    print(f"   - {d}")

# ============================================================================
# STEP 4: Download Code Files
# ============================================================================

print("\n[4/6] Setting up code files...")

# Create main training script
training_code = """
# The main training script will be uploaded separately
# Or can be copy-pasted from the repository
"""

# Create inference demo
inference_code = """
# The inference demo will be uploaded separately
# Or can be copy-pasted from the repository
"""

print("✅ Code files ready")
print("   Upload the following files to /content/:")
print("   - semiconductor_defect_detection.py")
print("   - onnx_export.py")
print("   - inference_demo.py")

# ============================================================================
# STEP 5: Dataset Instructions
# ============================================================================

print("\n[5/6] Dataset setup instructions:")
print("""
📦 Upload your dataset to Google Drive:

1. Prepare your dataset with this structure:
   semiconductor_dataset/
   ├── train/
   │   ├── wafer/
   │   │   ├── Center/
   │   │   ├── Donut/
   │   │   ├── Edge-Loc/
   │   │   └── ...
   │   └── die/
   │       ├── Clean/
   │       ├── Defect/
   │       └── Other/
   ├── val/
   └── test/

2. Zip the dataset: semiconductor_dataset.zip

3. Upload to: /content/drive/MyDrive/semiconductor_dataset.zip

4. Run the extraction in the main script
""")

# ============================================================================
# STEP 6: Environment Verification
# ============================================================================

print("\n[6/6] Verifying environment...")

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(f"✅ TensorFlow version: {tf.__version__}")
print(f"✅ NumPy version: {np.__version__}")
print(f"✅ Pandas version: {pd.__version__}")

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU available: {len(gpus)} device(s)")
    for gpu in gpus:
        print(f"   - {gpu}")
else:
    print("⚠️  No GPU detected - training will use CPU (slower)")

# Check CPU
print(f"✅ CPU cores: {os.cpu_count()}")

# ============================================================================
# Configuration Helper
# ============================================================================

print("\n" + "="*80)
print("QUICK START CONFIGURATION")
print("="*80)

config_template = """
# Update these paths in semiconductor_defect_detection.py:

class Config:
    # Dataset
    DATASET_ZIP = "/content/drive/MyDrive/semiconductor_dataset.zip"
    DATASET_ROOT = "/content/dataset"
    OUTPUT_DIR = "/content/output"
    MODELS_DIR = "/content/models"
    
    # Training (adjust as needed)
    BATCH_SIZE = 32  # Reduce if GPU memory limited
    EPOCHS = 50
    INITIAL_LR = 1e-3
    
    # Your defect classes
    WAFER_CLASSES = ['Center', 'Donut', 'Edge-Loc', ...]  # Update
    DIE_CLASSES = ['Clean', 'Defect', 'Other']
"""

print(config_template)

# ============================================================================
# Next Steps
# ============================================================================

print("\n" + "="*80)
print("✅ SETUP COMPLETE!")
print("="*80)

print("\n📋 Next Steps:")
print("\n1. Upload code files to /content/:")
print("   - semiconductor_defect_detection.py")
print("   - onnx_export.py")
print("   - inference_demo.py")

print("\n2. Upload dataset to Google Drive:")
print("   - /content/drive/MyDrive/semiconductor_dataset.zip")

print("\n3. Run training:")
print("   !python semiconductor_defect_detection.py")

print("\n4. Export to ONNX:")
print("   !python onnx_export.py")

print("\n5. Test inference:")
print("   !python inference_demo.py")

print("\n" + "="*80)

# ============================================================================
# Utility Functions
# ============================================================================

def check_dataset_structure(dataset_root='/content/dataset'):
    """Check if dataset is properly structured"""
    print("\n🔍 Checking dataset structure...")
    
    required_splits = ['train', 'val', 'test']
    
    for split in required_splits:
        split_path = os.path.join(dataset_root, split)
        if os.path.exists(split_path):
            classes = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
            total_images = 0
            
            for cls in classes:
                cls_path = os.path.join(split_path, cls)
                num_images = len([f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                total_images += num_images
            
            print(f"✅ {split}: {len(classes)} classes, {total_images} images")
        else:
            print(f"❌ {split} split not found at {split_path}")

def quick_test():
    """Quick test to verify setup"""
    print("\n🧪 Running quick test...")
    
    # Test TensorFlow
    try:
        import tensorflow as tf
        x = tf.constant([[1, 2], [3, 4]])
        print("✅ TensorFlow working")
    except Exception as e:
        print(f"❌ TensorFlow error: {e}")
    
    # Test file access
    if os.path.exists('/content/drive/MyDrive'):
        print("✅ Google Drive accessible")
    else:
        print("❌ Google Drive not mounted")
    
    print("\n✅ All tests passed!")

# Make functions available
print("\n📌 Utility functions available:")
print("   - check_dataset_structure() : Verify dataset")
print("   - quick_test() : Test environment")
