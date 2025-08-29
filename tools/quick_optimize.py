#!/usr/bin/env python3
"""
Quick Start Model Optimization
Dynamically selects and optimizes a model from available .pth files
"""

import sys
import os
import glob
from pathlib import Path

# Add the tools directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.model_optimizer import ModelOptimizer

def list_available_models():
    """List all .pth files in the current directory"""
    pth_files = glob.glob("*.pth")
    return sorted(pth_files)

def get_config_for_model(model_name):
    """Get the appropriate config file for a given model name"""
    config_mapping = {
        'segformer.b0.1024x1024.city.160k': 'local_configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py',
        'segformer.b0.512x1024.city.160k': 'local_configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py',
        'segformer.b0.512x512.ade.160k': 'local_configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py',
        'segformer.b1.1024x1024.city.160k': 'local_configs/segformer/segformer_mit-b1_8xb1-160k_cityscapes-1024x1024.py',
        'segformer.b1.512x512.ade.160k': 'local_configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py',
        'segformer.b2.1024x1024.city.160k': 'local_configs/segformer/segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py',
        'segformer.b2.512x512.ade.160k': 'local_configs/segformer/segformer_mit-b2_8xb2-160k_ade20k-512x512.py',
        'segformer.b3.1024x1024.city.160k': 'local_configs/segformer/segformer_mit-b3_8xb1-160k_cityscapes-1024x1024.py',
        'segformer.b3.512x512.ade.160k': 'local_configs/segformer/segformer_mit-b3_8xb2-160k_ade20k-512x512.py',
        'segformer.b4.1024x1024.city.160k': 'local_configs/segformer/segformer_mit-b4_8xb1-160k_cityscapes-1024x1024.py',
        'segformer.b4.512x512.ade.160k': 'local_configs/segformer/segformer_mit-b4_8xb2-160k_ade20k-512x512.py',
        'segformer.b5.1024x1024.city.160k': 'local_configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py',
        'segformer.b5.640x640.ade.160k': 'local_configs/segformer/segformer_mit-b5_8xb2-160k_ade20k-640x640.py',
        'segformer.b5.512x512.ade.160k': 'local_configs/segformer/segformer_mit-b5_8xb2-160k_ade20k-512x512.py',
    }

    # Extract model base name (remove .pth extension)
    model_base = model_name.replace('.pth', '')

    return config_mapping.get(model_base)

def select_model():
    """Interactive model selection"""
    models = list_available_models()

    if not models:
        print("❌ No .pth files found in the current directory")
        return None, None

    print("🚀 MMSegmentation Model Optimization Quick Start")
    print("=" * 60)
    print(f"📁 Found {len(models)} model(s) in current directory:")
    print()

    for i, model in enumerate(models, 1):
        # Get config info
        config_path = get_config_for_model(model)
        config_status = "✅ Config found" if config_path else "❌ Config missing"

        print(f" {i}. {model} - {config_status}")

    print()
    print("0. Exit")
    print()

    while True:
        try:
            choice = input("Select a model to optimize (1-{}): ".format(len(models)))

            if choice == '0':
                print("👋 Exiting...")
                return None, None

            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(models):
                selected_model = models[choice_idx]
                config_path = get_config_for_model(selected_model)

                if not config_path:
                    print(f"❌ No config file found for {selected_model}")
                    print("   Please ensure the config file exists in local_configs/segformer/")
                    return None, None

                return selected_model, config_path
            else:
                print(f"❌ Invalid choice. Please select 1-{len(models)} or 0 to exit.")

        except ValueError:
            print("❌ Please enter a valid number.")

def quick_optimize():
    """Quick optimization with dynamic model selection"""

    # Select model interactively
    checkpoint_path, config_path = select_model()

    if not checkpoint_path or not config_path:
        return

    device = "cuda:0"

    print(f"\n🎯 Selected Model: {checkpoint_path}")
    print(f"📋 Config: {config_path}")
    print(f"🔧 Device: {device}")
    print("\n" + "="*60)

    # Initialize optimizer
    try:
        optimizer = ModelOptimizer(config_path, checkpoint_path, device)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("Make sure the config and checkpoint paths are correct")
        return

    # Apply most effective optimizations
    print("\n📊 Applying optimizations...")

    # FP16 for immediate speedup
    print("1. Converting to FP16...")
    optimizer.optimize_fp16()

    # INT8 for maximum compression
    print("2. Applying INT8 quantization...")
    optimizer.optimize_quantization_aware()

    # Batch processing optimization
    print("3. Optimizing for batch processing...")
    optimizer.optimize_batch_processing(batch_size=4)

    # ONNX conversion for cross-platform deployment
    print("4. Converting to ONNX...")
    onnx_path = optimizer.convert_to_onnx()
    if onnx_path:
        print("5. Optimizing ONNX model...")
        optimizer.optimize_onnx_model(onnx_path)

    # Benchmark all versions
    print("\n⚡ Benchmarking optimized models...")
    results = optimizer.compare_optimizations(num_runs=5)

    # Show recommendations
    print("\n💡 Recommendations:")
    if 'fp16' in results:
        fp16_speedup = results['original']['avg_inference_time_ms'] / results['fp16']['avg_inference_time_ms']
        print(f"   - FP16: {fp16_speedup:.1f}x speedup")
        print("   - Best for: Real-time applications with GPU")

    if 'int8' in results:
        int8_speedup = results['original']['avg_inference_time_ms'] / results['int8']['avg_inference_time_ms']
        print(f"   - INT8: {int8_speedup:.1f}x speedup")
        print("   - Best for: Edge devices, mobile deployment")

    print("   - ONNX: Cross-platform compatibility")
    print("   - Best for: Deployment on different frameworks (TensorRT, OpenVINO, etc.)")

    print("\n✅ Optimization complete!")
    print("Use the optimized models in your inference pipeline for better performance.")

if __name__ == "__main__":
    quick_optimize()
