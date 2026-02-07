"""
ONNX Model Export & Optimization Script
For NXP i.MX RT series deployment via eIQ platform

This script handles:
1. Model conversion to ONNX format
2. Quantization for edge optimization
3. Performance validation
4. eIQ deployment preparation
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import onnx
import onnxruntime as ort
from typing import Dict, Tuple
import json
import time


class EdgeOptimizer:
    """Optimize models for edge deployment"""
    
    def __init__(self, models_dir: str, output_dir: str):
        self.models_dir = models_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def convert_to_onnx(self, model_path: str, model_name: str, 
                       input_shape: Tuple = (1, 224, 224, 3)) -> str:
        """
        Convert Keras model to ONNX format
        
        Args:
            model_path: Path to .h5 Keras model
            model_name: Name for output file
            input_shape: Model input shape
            
        Returns:
            Path to ONNX model
        """
        print(f"\n{'='*80}")
        print(f"Converting {model_name} to ONNX")
        print(f"{'='*80}")
        
        # Load Keras model
        model = keras.models.load_model(model_path)
        
        # Convert to ONNX
        try:
            import tf2onnx
        except ImportError:
            print("Installing tf2onnx...")
            os.system("pip install -q tf2onnx")
            import tf2onnx
        
        onnx_path = os.path.join(self.output_dir, f"{model_name}.onnx")
        
        spec = (tf.TensorSpec(input_shape, tf.float32, name="input"),)
        
        model_proto, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=spec,
            opset=13,
            output_path=onnx_path
        )
        
        # Validate
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        
        print(f"✅ ONNX conversion successful")
        print(f"   Output: {onnx_path}")
        print(f"   Size: {file_size_mb:.2f} MB")
        
        return onnx_path
    
    def quantize_onnx_model(self, onnx_path: str) -> str:
        """
        Quantize ONNX model for edge deployment
        
        Args:
            onnx_path: Path to ONNX model
            
        Returns:
            Path to quantized model
        """
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
        except ImportError:
            print("⚠️  ONNX quantization not available")
            return onnx_path
        
        print(f"\n{'='*80}")
        print("Quantizing model for edge deployment")
        print(f"{'='*80}")
        
        quantized_path = onnx_path.replace('.onnx', '_quantized.onnx')
        
        quantize_dynamic(
            onnx_path,
            quantized_path,
            weight_type=QuantType.QUInt8
        )
        
        original_size = os.path.getsize(onnx_path) / (1024 * 1024)
        quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)
        reduction = ((original_size - quantized_size) / original_size) * 100
        
        print(f"✅ Quantization complete")
        print(f"   Original: {original_size:.2f} MB")
        print(f"   Quantized: {quantized_size:.2f} MB")
        print(f"   Reduction: {reduction:.1f}%")
        
        return quantized_path
    
    def validate_onnx_inference(self, onnx_path: str, test_input: np.ndarray) -> Dict:
        """
        Validate ONNX model inference
        
        Args:
            onnx_path: Path to ONNX model
            test_input: Test input array
            
        Returns:
            Performance metrics
        """
        print(f"\n{'='*80}")
        print("Validating ONNX inference")
        print(f"{'='*80}")
        
        # Create inference session
        session = ort.InferenceSession(onnx_path)
        
        # Get input name
        input_name = session.get_inputs()[0].name
        
        # Run inference benchmark
        num_runs = 100
        inference_times = []
        
        for i in range(num_runs):
            start = time.time()
            outputs = session.run(None, {input_name: test_input})
            inference_times.append((time.time() - start) * 1000)
        
        metrics = {
            'mean_latency_ms': float(np.mean(inference_times)),
            'std_latency_ms': float(np.std(inference_times)),
            'min_latency_ms': float(np.min(inference_times)),
            'max_latency_ms': float(np.max(inference_times)),
            'fps': float(1000.0 / np.mean(inference_times))
        }
        
        print(f"✅ ONNX validation complete")
        print(f"   Mean Latency: {metrics['mean_latency_ms']:.2f} ms")
        print(f"   Throughput: {metrics['fps']:.2f} FPS")
        
        return metrics
    
    def generate_deployment_package(self, model_info: Dict):
        """
        Generate deployment package for NXP eIQ
        
        Args:
            model_info: Dictionary containing model metadata
        """
        print(f"\n{'='*80}")
        print("Generating NXP eIQ Deployment Package")
        print(f"{'='*80}")
        
        # Create deployment config
        deployment_config = {
            'platform': 'NXP i.MX RT series',
            'framework': 'eIQ',
            'models': model_info,
            'inference_config': {
                'input_shape': [1, 224, 224, 3],
                'input_type': 'float32',
                'preprocessing': {
                    'normalization': 'divide_by_255',
                    'mean': [0, 0, 0],
                    'std': [1, 1, 1]
                }
            },
            'optimization': {
                'quantization': 'dynamic_uint8',
                'target_latency_ms': 100,
                'target_throughput_fps': 10
            }
        }
        
        config_path = os.path.join(self.output_dir, 'deployment_config.json')
        with open(config_path, 'w') as f:
            json.dump(deployment_config, f, indent=2)
        
        print(f"✅ Deployment config saved: {config_path}")
        
        # Create README
        readme_content = """# NXP eIQ Deployment Package

## Models Included
- Wafer Classifier (ONNX)
- Die Classifier (ONNX)
- Stage Router (ONNX)

## Deployment Steps

### 1. Prerequisites
- NXP i.MX RT series board
- eIQ Toolkit installed
- ONNX Runtime for eIQ

### 2. Model Loading
```python
import onnxruntime as ort

# Load models
wafer_session = ort.InferenceSession("wafer_classifier_quantized.onnx")
die_session = ort.InferenceSession("die_classifier_quantized.onnx")
router_session = ort.InferenceSession("stage_router_quantized.onnx")
```

### 3. Inference Pipeline
```python
# Preprocess image
image = preprocess_image(raw_image)  # Normalize to [0, 1]

# Stage routing
stage_probs = router_session.run(None, {"input": image})[0]
stage = "wafer" if stage_probs[0][0] > 0.5 else "die"

# Classification
if stage == "wafer":
    result = wafer_session.run(None, {"input": image})[0]
else:
    result = die_session.run(None, {"input": image})[0]
```

### 4. Performance Optimization
- Use CPU threads: `ort.SessionOptions().intra_op_num_threads = 4`
- Enable graph optimization: `ort.GraphOptimizationLevel.ORT_ENABLE_ALL`
- Consider model caching for faster startup

## Expected Performance
- Latency: < 100ms per image
- Throughput: > 10 FPS
- Model Size: < 50 MB total

## Support
For deployment issues, refer to NXP eIQ documentation or contact support.
"""
        
        readme_path = os.path.join(self.output_dir, 'DEPLOYMENT_README.md')
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        print(f"✅ Deployment README saved: {readme_path}")


def main():
    """Main export pipeline"""
    
    print("="*80)
    print("ONNX EXPORT & EDGE OPTIMIZATION")
    print("NXP i.MX RT series via eIQ platform")
    print("="*80)
    
    # Configuration
    MODELS_DIR = "/content/models"
    OUTPUT_DIR = "/content/onnx_models"
    
    optimizer = EdgeOptimizer(MODELS_DIR, OUTPUT_DIR)
    
    # Model paths (update these based on your trained models)
    models_to_export = {
        'wafer_classifier': os.path.join(MODELS_DIR, 'wafer_final.h5'),
        'die_classifier': os.path.join(MODELS_DIR, 'die_final.h5'),
        'stage_router': os.path.join(MODELS_DIR, 'stage_router_best.h5')
    }
    
    model_info = {}
    
    # Convert all models
    for model_name, model_path in models_to_export.items():
        if os.path.exists(model_path):
            # Convert to ONNX
            onnx_path = optimizer.convert_to_onnx(model_path, model_name)
            
            # Quantize
            quantized_path = optimizer.quantize_onnx_model(onnx_path)
            
            # Validate
            test_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
            metrics = optimizer.validate_onnx_inference(quantized_path, test_input)
            
            model_info[model_name] = {
                'onnx_path': onnx_path,
                'quantized_path': quantized_path,
                'performance': metrics
            }
        else:
            print(f"⚠️  Model not found: {model_path}")
    
    # Generate deployment package
    if model_info:
        optimizer.generate_deployment_package(model_info)
    
    print("\n" + "="*80)
    print("✅ EXPORT COMPLETE")
    print("="*80)
    print(f"\n📁 Output Directory: {OUTPUT_DIR}")
    print("\nGenerated Files:")
    print("- *.onnx (Original ONNX models)")
    print("- *_quantized.onnx (Quantized models)")
    print("- deployment_config.json (Configuration)")
    print("- DEPLOYMENT_README.md (Deployment guide)")


if __name__ == "__main__":
    main()
