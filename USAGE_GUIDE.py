# Complete Usage Guide
# IESA DeepTech Hackathon 2026 - Semiconductor Defect Detection System

"""
This guide provides step-by-step instructions for using the semiconductor
defect detection system, from setup to deployment.
"""

# ============================================================================
# TABLE OF CONTENTS
# ============================================================================

"""
1. Initial Setup
2. Data Preparation
3. Training Models
4. Evaluation & Testing
5. Model Export
6. Edge Deployment
7. Advanced Usage
8. Troubleshooting
"""

# ============================================================================
# 1. INITIAL SETUP
# ============================================================================

"""
Google Colab Setup:
------------------
"""

# Step 1: Run setup script
# !python colab_setup.py

# Step 2: Upload code files to /content/
# - semiconductor_defect_detection.py
# - onnx_export.py
# - inference_demo.py

# Step 3: Upload dataset to Google Drive
# Path: /content/drive/MyDrive/semiconductor_dataset.zip

"""
Local Setup:
-----------
"""

# git clone https://github.com/yourusername/semiconductor-defect-detection.git
# cd semiconductor-defect-detection
# pip install -r requirements.txt

# ============================================================================
# 2. DATA PREPARATION
# ============================================================================

"""
Dataset Structure Requirements:
------------------------------

semiconductor_dataset/
├── train/
│   ├── wafer/
│   │   ├── Center/           (defect class 1)
│   │   ├── Donut/            (defect class 2)
│   │   ├── Edge-Loc/         (defect class 3)
│   │   ├── Edge-Ring/        (defect class 4)
│   │   ├── Loc/              (defect class 5)
│   │   ├── Near-full/        (defect class 6)
│   │   ├── Random/           (defect class 7)
│   │   ├── Scratch/          (defect class 8)
│   │   └── None/             (no defect)
│   └── die/
│       ├── Clean/            (no defect)
│       ├── Defect/           (has defect)
│       └── Other/            (uncertain/unknown)
├── val/
│   └── [same structure as train]
└── test/
    └── [same structure as train]

Recommended Image Counts:
- Training: 1000+ per class
- Validation: 200+ per class
- Test: 200+ per class
"""

# Example: Verify dataset structure
"""
from semiconductor_defect_detection import DatasetManager, Config

config = Config()
dataset_manager = DatasetManager(config)

# Extract dataset (if zipped)
dataset_manager.extract_dataset()

# Validate structure
dataset_manager.validate_structure()

# Check class distribution
distribution = dataset_manager.get_class_distribution('train')
print(distribution)
"""

# ============================================================================
# 3. TRAINING MODELS
# ============================================================================

"""
Basic Training:
--------------
"""

# Method 1: Run complete pipeline
# !python semiconductor_defect_detection.py

"""
Method 2: Custom training with code
"""

from semiconductor_defect_detection import (
    Config, DatasetManager, DualHeadDefectClassifier,
    TrainingPipeline, InferenceEngine, ModelEvaluator
)

# Initialize
config = Config()
config.create_directories()

# Build models
model_builder = DualHeadDefectClassifier(config)
model_builder.build_models()

# Prepare data
dataset_manager = DatasetManager(config)
training_pipeline = TrainingPipeline(config, model_builder)

# Create data generators (example for wafer)
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=5
)

train_gen_wafer = train_datagen.flow_from_directory(
    '/content/dataset/train/wafer',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

val_gen_wafer = val_datagen.flow_from_directory(
    '/content/dataset/val/wafer',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)
"""

# Compute class weights for imbalanced data
"""
distribution = dataset_manager.get_class_distribution('train')
class_weights = training_pipeline.compute_class_weights(distribution)
"""

# Train wafer classifier
"""
training_pipeline.train_classifier(
    model_builder.wafer_model,
    'wafer',
    train_gen_wafer,
    val_gen_wafer,
    class_weights
)
"""

# Train die classifier (similar process)
"""
# Create die data generators
# ...
training_pipeline.train_classifier(
    model_builder.die_model,
    'die',
    train_gen_die,
    val_gen_die,
    class_weights_die
)
"""

"""
Advanced Training Configuration:
-------------------------------
"""

# Custom configuration
"""
class CustomConfig(Config):
    # Adjust hyperparameters
    BATCH_SIZE = 64  # Increase if GPU has enough memory
    EPOCHS = 100     # More epochs for larger datasets
    INITIAL_LR = 5e-4  # Lower learning rate
    
    # Adjust thresholds
    CONFIDENCE_THRESHOLD = 0.7  # Stricter confidence
    EARLY_EXIT_THRESHOLD = 0.98  # Higher threshold for early exit
    
    # Custom augmentation
    AUGMENTATION_CONFIG = {
        'horizontal_flip': True,
        'vertical_flip': True,
        'rotation_range': 10,  # More rotation
        'brightness_range': [0.9, 1.1],  # Add brightness variation
        'fill_mode': 'reflect'
    }

custom_config = CustomConfig()
"""

# ============================================================================
# 4. EVALUATION & TESTING
# ============================================================================

"""
Model Evaluation:
----------------
"""

# Create inference engine
"""
inference_engine = InferenceEngine(
    config,
    model_builder.wafer_model,
    model_builder.die_model,
    model_builder.stage_router
)

evaluator = ModelEvaluator(config, inference_engine)

# Evaluate on test set
test_datagen = ImageDataGenerator(rescale=1./255)

test_gen_wafer = test_datagen.flow_from_directory(
    '/content/dataset/test/wafer',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

wafer_results = evaluator.evaluate_on_test_set(test_gen_wafer, 'wafer')

# View results
print(f"Accuracy: {wafer_results['accuracy']:.4f}")
print(f"Precision: {wafer_results['precision']:.4f}")
print(f"Recall: {wafer_results['recall']:.4f}")
print(f"F1-Score: {wafer_results['f1_score']:.4f}")
"""

"""
Performance Benchmarking:
------------------------
"""

# CPU-only benchmark
"""
import numpy as np

# Generate test images
test_images = np.random.rand(100, 224, 224, 3).astype(np.float32)

# Benchmark
performance = evaluator.benchmark_edge_performance(test_images, num_runs=100)

print(f"Mean Latency: {performance['cpu_inference']['mean_ms']:.2f} ms")
print(f"Throughput: {performance['cpu_inference']['fps']:.2f} FPS")
print(f"Model Size: {performance['model_sizes_mb']['total']:.2f} MB")
"""

# ============================================================================
# 5. MODEL EXPORT
# ============================================================================

"""
Export to ONNX:
--------------
"""

# Method 1: Run export script
# !python onnx_export.py

"""
Method 2: Programmatic export
"""

from onnx_export import EdgeOptimizer

"""
optimizer = EdgeOptimizer(
    models_dir='/content/models',
    output_dir='/content/onnx_models'
)

# Convert to ONNX
onnx_path = optimizer.convert_to_onnx(
    '/content/models/wafer_final.h5',
    'wafer_classifier'
)

# Quantize for edge deployment
quantized_path = optimizer.quantize_onnx_model(onnx_path)

# Validate ONNX model
test_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
metrics = optimizer.validate_onnx_inference(quantized_path, test_input)
"""

# ============================================================================
# 6. EDGE DEPLOYMENT
# ============================================================================

"""
NXP i.MX RT Deployment:
----------------------

1. Copy ONNX models to device:
   - wafer_classifier_quantized.onnx
   - die_classifier_quantized.onnx
   - stage_router_quantized.onnx

2. Install ONNX Runtime on device:
   pip install onnxruntime

3. Load models:
"""

import onnxruntime as ort
import numpy as np

def load_models():
    """Load ONNX models on edge device"""
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 4
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    wafer_session = ort.InferenceSession(
        "wafer_classifier_quantized.onnx",
        sess_options=session_options
    )
    
    die_session = ort.InferenceSession(
        "die_classifier_quantized.onnx",
        sess_options=session_options
    )
    
    router_session = ort.InferenceSession(
        "stage_router_quantized.onnx",
        sess_options=session_options
    )
    
    return wafer_session, die_session, router_session

def preprocess_image(image_path):
    """Preprocess image for ONNX inference"""
    from PIL import Image
    
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def edge_inference(image_path, wafer_session, die_session, router_session):
    """Complete inference pipeline on edge device"""
    
    # Preprocess
    image = preprocess_image(image_path)
    
    # Stage routing
    stage_probs = router_session.run(None, {"input": image})[0]
    stage = "wafer" if stage_probs[0][0] > 0.5 else "die"
    
    # Classification
    if stage == "wafer":
        result = wafer_session.run(None, {"input": image})[0]
        classes = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 
                  'Loc', 'Near-full', 'Random', 'Scratch', 'None']
    else:
        result = die_session.run(None, {"input": image})[0]
        classes = ['Clean', 'Defect', 'Other']
    
    # Get prediction
    pred_idx = np.argmax(result[0])
    confidence = result[0][pred_idx]
    
    return {
        'stage': stage,
        'class': classes[pred_idx],
        'confidence': float(confidence)
    }

# Usage example
"""
wafer_session, die_session, router_session = load_models()
result = edge_inference('test_image.jpg', wafer_session, die_session, router_session)
print(f"Predicted: {result['class']} ({result['confidence']:.2%})")
"""

# ============================================================================
# 7. ADVANCED USAGE
# ============================================================================

"""
Tile-Based Processing for High-Resolution Images:
------------------------------------------------
"""

from semiconductor_defect_detection import TileProcessor

"""
tile_processor = TileProcessor(tile_size=224, overlap=32)

# Load high-resolution image
import cv2
high_res_image = cv2.imread('high_res_wafer.tiff')

# Generate tiles
tiles_and_positions = tile_processor.generate_tiles(high_res_image)

# Process each tile
predictions = []
for tile, position in tiles_and_positions:
    # Normalize tile
    tile_norm = tile.astype(np.float32) / 255.0
    
    # Predict
    pred = model.predict(np.expand_dims(tile_norm, 0))
    predictions.append(pred[0])

# Aggregate predictions
final_pred = tile_processor.aggregate_predictions(
    predictions,
    [pos for _, pos in tiles_and_positions],
    high_res_image.shape[:2]
)

print(f"Aggregated prediction: {np.argmax(final_pred)}")
"""

"""
Custom Confidence Thresholds:
----------------------------
"""

def adaptive_threshold_prediction(image, model, base_threshold=0.6):
    """
    Adaptive confidence thresholding based on prediction entropy
    """
    probs = model.predict(np.expand_dims(image, 0))[0]
    
    # Calculate entropy
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(len(probs))
    normalized_entropy = entropy / max_entropy
    
    # Adjust threshold based on entropy
    dynamic_threshold = base_threshold * (1 + normalized_entropy)
    
    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx]
    
    if confidence < dynamic_threshold:
        return "Unknown", confidence
    else:
        return pred_idx, confidence

# ============================================================================
# 8. TROUBLESHOOTING
# ============================================================================

"""
Common Issues and Solutions:
---------------------------

1. Out of Memory (GPU):
   - Reduce BATCH_SIZE in Config
   - Use gradient accumulation
   - Enable mixed precision training

2. Low Accuracy:
   - Increase training data
   - Adjust learning rate
   - Try different augmentation strategies
   - Ensure data quality

3. Slow Inference:
   - Use quantized ONNX models
   - Enable multi-threading
   - Reduce image resolution
   - Use early-exit logic

4. Model Not Converging:
   - Check learning rate (try 1e-4 to 1e-2)
   - Verify data preprocessing
   - Check class balance
   - Increase FREEZE_BACKBONE_EPOCHS

5. ONNX Export Fails:
   - Update tf2onnx: pip install --upgrade tf2onnx
   - Check TensorFlow version compatibility
   - Try different opset versions

6. Dataset Issues:
   - Verify directory structure
   - Check image formats (JPG, PNG)
   - Ensure sufficient samples per class
   - Remove corrupted images
"""

# ============================================================================
# MONITORING & LOGGING
# ============================================================================

"""
Training Monitoring:
-------------------
"""

# Enable TensorBoard
"""
from tensorflow.keras.callbacks import TensorBoard
import datetime

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Add to training callbacks
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=[tensorboard_callback, ...]
)

# View in Colab
%load_ext tensorboard
%tensorboard --logdir logs/fit
"""

"""
Production Logging:
------------------
"""

import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('defect_detection.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('DefectDetection')

# Log predictions
"""
def logged_inference(image_path, model):
    logger.info(f"Processing image: {image_path}")
    
    try:
        result = predict(image_path, model)
        logger.info(f"Prediction: {result['class']}, Confidence: {result['confidence']:.4f}")
        return result
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise
"""

print("Usage guide loaded successfully!")
print("Refer to code comments for detailed examples.")
