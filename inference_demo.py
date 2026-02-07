"""
Inference Demo Script
Test the trained semiconductor defect detection models

This script demonstrates:
1. Loading trained models
2. Stage-aware inference
3. Tile-based processing
4. Performance benchmarking
5. Visualization of results
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from typing import List, Dict
import tensorflow as tf
from tensorflow import keras


class DefectDetectionDemo:
    """Demo class for semiconductor defect detection inference"""
    
    def __init__(self, models_dir: str):
        """
        Initialize demo with trained models
        
        Args:
            models_dir: Directory containing trained .h5 models
        """
        self.models_dir = models_dir
        self.wafer_model = None
        self.die_model = None
        self.stage_router = None
        
        # Configuration
        self.img_size = (224, 224)
        self.confidence_threshold = 0.6
        self.early_exit_threshold = 0.95
        
        # Class labels
        self.wafer_classes = [
            'Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 
            'Loc', 'Near-full', 'Random', 'Scratch', 'None'
        ]
        self.die_classes = ['Clean', 'Defect', 'Other']
        
        self.load_models()
    
    def load_models(self):
        """Load all trained models"""
        print("="*80)
        print("Loading Trained Models")
        print("="*80)
        
        try:
            wafer_path = os.path.join(self.models_dir, 'wafer_final.h5')
            if os.path.exists(wafer_path):
                self.wafer_model = keras.models.load_model(wafer_path)
                print(f"✅ Wafer classifier loaded: {wafer_path}")
            else:
                print(f"⚠️  Wafer model not found at {wafer_path}")
            
            die_path = os.path.join(self.models_dir, 'die_final.h5')
            if os.path.exists(die_path):
                self.die_model = keras.models.load_model(die_path)
                print(f"✅ Die classifier loaded: {die_path}")
            else:
                print(f"⚠️  Die model not found at {die_path}")
            
            router_path = os.path.join(self.models_dir, 'stage_router_best.h5')
            if os.path.exists(router_path):
                self.stage_router = keras.models.load_model(router_path)
                print(f"✅ Stage router loaded: {router_path}")
            else:
                print(f"⚠️  Stage router not found at {router_path}")
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for inference
        
        Args:
            image_path: Path to input image
            
        Returns:
            Preprocessed image array
        """
        # Load image
        img = Image.open(image_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize
        img = img.resize(self.img_size)
        
        # Convert to array and normalize
        img_array = np.array(img).astype(np.float32) / 255.0
        
        return img_array
    
    def detect_stage(self, image: np.ndarray) -> str:
        """
        Detect if image is Wafer or Die
        
        Args:
            image: Preprocessed image
            
        Returns:
            'wafer' or 'die'
        """
        if self.stage_router is None:
            # Default to wafer if router not available
            return 'wafer'
        
        # Add batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Predict
        probs = self.stage_router.predict(image, verbose=0)[0]
        stage = 'wafer' if probs[0] > probs[1] else 'die'
        
        return stage, {'wafer': probs[0], 'die': probs[1]}
    
    def classify_defect(self, image: np.ndarray, stage: str) -> Dict:
        """
        Classify defect type
        
        Args:
            image: Preprocessed image
            stage: 'wafer' or 'die'
            
        Returns:
            Classification results dictionary
        """
        # Select model
        if stage == 'wafer':
            if self.wafer_model is None:
                return {'error': 'Wafer model not loaded'}
            model = self.wafer_model
            classes = self.wafer_classes
        else:
            if self.die_model is None:
                return {'error': 'Die model not loaded'}
            model = self.die_model
            classes = self.die_classes
        
        # Add batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Predict
        start_time = time.time()
        probs = model.predict(image, verbose=0)[0]
        inference_time = (time.time() - start_time) * 1000
        
        # Get prediction
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]
        
        # Check early exit
        early_exit = confidence >= self.early_exit_threshold
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            # Route to "Other/Unknown"
            if stage == 'die' and 'Other' in classes:
                pred_idx = classes.index('Other')
            elif stage == 'wafer' and 'None' in classes:
                pred_idx = classes.index('None')
        
        return {
            'predicted_class': classes[pred_idx],
            'confidence': float(confidence),
            'probabilities': {cls: float(prob) for cls, prob in zip(classes, probs)},
            'inference_time_ms': inference_time,
            'early_exit': early_exit
        }
    
    def predict(self, image_path: str, visualize: bool = True) -> Dict:
        """
        Complete prediction pipeline
        
        Args:
            image_path: Path to input image
            visualize: Whether to show visualization
            
        Returns:
            Complete prediction results
        """
        print("\n" + "="*80)
        print(f"Processing: {image_path}")
        print("="*80)
        
        # Preprocess
        image = self.preprocess_image(image_path)
        
        # Stage detection
        stage, stage_probs = self.detect_stage(image)
        print(f"\n🎯 Stage Detection: {stage.upper()}")
        print(f"   Wafer probability: {stage_probs['wafer']:.4f}")
        print(f"   Die probability: {stage_probs['die']:.4f}")
        
        # Classification
        result = self.classify_defect(image, stage)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return result
        
        print(f"\n📊 Classification Results:")
        print(f"   Predicted Class: {result['predicted_class']}")
        print(f"   Confidence: {result['confidence']:.4f}")
        print(f"   Inference Time: {result['inference_time_ms']:.2f} ms")
        print(f"   Early Exit: {'Yes' if result['early_exit'] else 'No'}")
        
        # Visualize
        if visualize:
            self.visualize_prediction(image_path, image, stage, result)
        
        # Complete result
        complete_result = {
            'image_path': image_path,
            'stage': stage,
            'stage_probabilities': stage_probs,
            **result
        }
        
        return complete_result
    
    def visualize_prediction(self, image_path: str, processed_image: np.ndarray, 
                           stage: str, result: Dict):
        """
        Visualize prediction results
        
        Args:
            image_path: Original image path
            processed_image: Preprocessed image
            stage: Detected stage
            result: Classification results
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Original image
        original_img = Image.open(image_path)
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Prediction visualization
        axes[1].imshow(processed_image)
        
        # Add prediction text
        pred_text = f"Stage: {stage.upper()}\n"
        pred_text += f"Class: {result['predicted_class']}\n"
        pred_text += f"Confidence: {result['confidence']:.2%}\n"
        pred_text += f"Latency: {result['inference_time_ms']:.1f} ms"
        
        # Color based on confidence
        if result['confidence'] >= 0.9:
            color = 'green'
        elif result['confidence'] >= 0.6:
            color = 'orange'
        else:
            color = 'red'
        
        axes[1].text(
            10, 30, pred_text,
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.7),
            fontsize=12, color='white', fontweight='bold'
        )
        axes[1].set_title('Prediction', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def batch_predict(self, image_paths: List[str]) -> List[Dict]:
        """
        Predict on multiple images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of prediction results
        """
        results = []
        
        print("\n" + "="*80)
        print(f"Batch Processing: {len(image_paths)} images")
        print("="*80)
        
        for i, path in enumerate(image_paths):
            print(f"\nProcessing {i+1}/{len(image_paths)}: {path}")
            result = self.predict(path, visualize=False)
            results.append(result)
        
        # Summary statistics
        self.print_batch_summary(results)
        
        return results
    
    def print_batch_summary(self, results: List[Dict]):
        """Print summary statistics for batch predictions"""
        print("\n" + "="*80)
        print("BATCH PREDICTION SUMMARY")
        print("="*80)
        
        # Stage distribution
        stages = [r['stage'] for r in results if 'stage' in r]
        print(f"\n📊 Stage Distribution:")
        print(f"   Wafer: {stages.count('wafer')}")
        print(f"   Die: {stages.count('die')}")
        
        # Average metrics
        confidences = [r['confidence'] for r in results if 'confidence' in r]
        latencies = [r['inference_time_ms'] for r in results if 'inference_time_ms' in r]
        
        print(f"\n⚡ Performance Metrics:")
        print(f"   Average Confidence: {np.mean(confidences):.4f}")
        print(f"   Average Latency: {np.mean(latencies):.2f} ms")
        print(f"   Throughput: {1000.0 / np.mean(latencies):.2f} FPS")
        
        # Early exit statistics
        early_exits = [r['early_exit'] for r in results if 'early_exit' in r]
        print(f"\n🚀 Early Exit Statistics:")
        print(f"   Early exits: {sum(early_exits)}/{len(early_exits)} ({100*sum(early_exits)/len(early_exits):.1f}%)")


def main():
    """Main demo execution"""
    
    print("="*80)
    print("SEMICONDUCTOR DEFECT DETECTION - INFERENCE DEMO")
    print("="*80)
    
    # Configuration
    MODELS_DIR = "/content/models"  # Update this path
    
    # Initialize demo
    demo = DefectDetectionDemo(MODELS_DIR)
    
    # Example 1: Single image inference
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Image Inference")
    print("="*80)
    
    # Replace with actual image path
    sample_image = "/content/test_images/wafer_sample.jpg"
    
    if os.path.exists(sample_image):
        result = demo.predict(sample_image, visualize=True)
    else:
        print(f"⚠️  Sample image not found: {sample_image}")
        print("Please update the path to a valid image")
    
    # Example 2: Batch inference
    print("\n" + "="*80)
    print("EXAMPLE 2: Batch Inference")
    print("="*80)
    
    # Replace with actual image directory
    test_dir = "/content/test_images"
    
    if os.path.exists(test_dir):
        image_paths = [
            os.path.join(test_dir, f) 
            for f in os.listdir(test_dir) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ][:10]  # Process first 10 images
        
        if image_paths:
            results = demo.batch_predict(image_paths)
        else:
            print(f"⚠️  No images found in {test_dir}")
    else:
        print(f"⚠️  Test directory not found: {test_dir}")
        print("Please update the path to a valid directory")
    
    # Example 3: Performance benchmarking
    print("\n" + "="*80)
    print("EXAMPLE 3: Performance Benchmarking")
    print("="*80)
    
    # Generate random test images
    print("\nGenerating synthetic test images...")
    num_tests = 100
    test_images = []
    
    for i in range(num_tests):
        # Create random image
        img_array = np.random.rand(224, 224, 3).astype(np.float32)
        test_images.append(img_array)
    
    # Benchmark
    latencies = []
    
    print(f"Running {num_tests} inferences...")
    for i, img in enumerate(test_images):
        if demo.wafer_model is not None:
            start = time.time()
            _ = demo.wafer_model.predict(np.expand_dims(img, 0), verbose=0)
            latencies.append((time.time() - start) * 1000)
        
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{num_tests}")
    
    if latencies:
        print(f"\n📊 Benchmark Results:")
        print(f"   Mean Latency: {np.mean(latencies):.2f} ms")
        print(f"   Std Latency: {np.std(latencies):.2f} ms")
        print(f"   Min Latency: {np.min(latencies):.2f} ms")
        print(f"   Max Latency: {np.max(latencies):.2f} ms")
        print(f"   Throughput: {1000.0 / np.mean(latencies):.2f} FPS")


if __name__ == "__main__":
    main()
