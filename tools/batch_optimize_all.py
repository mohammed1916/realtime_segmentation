#!/usr/bin/env python3
"""
Batch Model Optimization Script
Optimizes all .pth models in the root directory automatically
"""

import sys
import os
import glob
from pathlib import Path

# Add the tools directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_optimizer import ModelOptimizer

def get_model_config_mapping():
    """Map model filenames to their corresponding config files"""
    return {
        'segformer.b0.1024x1024.city.160k.pth': 'local_configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py',
        'segformer.b0.512x1024.city.160k.pth': 'local_configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py',
        'segformer.b0.512x512.ade.160k.pth': 'local_configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py',
        'segformer.b1.512x512.ade.160k.pth': 'local_configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py',
        'segformer.b2.1024x1024.city.160k.pth': 'local_configs/segformer/segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py',
        'segformer.b2.512x512.ade.160k.pth': 'local_configs/segformer/segformer_mit-b2_8xb2-160k_ade20k-512x512.py',
        'segformer.b3.1024x1024.city.160k.pth': 'local_configs/segformer/segformer_mit-b3_8xb1-160k_cityscapes-1024x1024.py',
        'segformer.b3.512x512.ade.160k.pth': 'local_configs/segformer/segformer_mit-b3_8xb2-160k_ade20k-512x512.py',
        'segformer.b4.1024x1024.city.160k.pth': 'local_configs/segformer/segformer_mit-b4_8xb1-160k_cityscapes-1024x1024.py',
        'segformer.b4.512x512.ade.160k.pth': 'local_configs/segformer/segformer_mit-b4_8xb2-160k_ade20k-512x512.py',
        'segformer.b5.1024x1024.city.160k.pth': 'local_configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py',
        'segformer.b5.640x640.ade.160k.pth': 'local_configs/segformer/segformer_mit-b5_8xb2-160k_ade20k-640x640.py'
    }

def find_pth_files(root_dir='.'):
    """Find all .pth files in the root directory"""
    pth_pattern = os.path.join(root_dir, '*.pth')
    return glob.glob(pth_pattern)

def optimize_all_models():
    """Optimize all .pth models in the root directory"""
    print("🚀 MMSegmentation Batch Model Optimization")
    print("=" * 60)

    # Get model-config mapping
    model_config_map = get_model_config_mapping()

    # Find all .pth files
    pth_files = find_pth_files()
    print(f"Found {len(pth_files)} .pth files in root directory")

    if not pth_files:
        print("❌ No .pth files found in root directory")
        return

    # Process each model
    successful_optimizations = 0
    failed_optimizations = 0

    for i, pth_file in enumerate(pth_files, 1):
        model_name = os.path.basename(pth_file)
        print(f"\n{'='*60}")
        print(f"📦 Processing Model {i}/{len(pth_files)}: {model_name}")
        print(f"{'='*60}")

        # Find corresponding config
        if model_name not in model_config_map:
            print(f"⚠️  No config found for {model_name}, skipping...")
            failed_optimizations += 1
            continue

        config_path = model_config_map[model_name]
        print(f"📋 Using config: {config_path}")

        try:
            # Initialize optimizer
            optimizer = ModelOptimizer(
                config_path=config_path,
                checkpoint_path=pth_file,
                device='cuda:0',
                output_dir='optimized_models'
            )

            # Apply optimizations
            print("\n🔧 Applying optimizations...")

            # FP16 optimization
            print("1. Converting to FP16...")
            optimizer.optimize_fp16()

            # INT8 optimization
            print("2. Applying INT8 quantization...")
            optimizer.optimize_quantization_aware()

            # Batch processing optimization
            print("3. Optimizing for batch processing...")
            optimizer.optimize_batch_processing(batch_size=4)

            # ONNX conversion
            print("4. Converting to ONNX...")
            onnx_path = optimizer.convert_to_onnx()
            if onnx_path:
                print("5. Optimizing ONNX model...")
                optimizer.optimize_onnx_model(onnx_path)

            # Benchmark all versions
            print("\n⚡ Benchmarking optimized models...")
            results = optimizer.compare_optimizations(num_runs=5)

            # Print quick results
            if 'original' in results and 'fp16' in results:
                fp16_speedup = results['original']['avg_inference_time_ms'] / results['fp16']['avg_inference_time_ms']
                print(f"  FP16 Speedup: {fp16_speedup:.1f}x")
                if 'int8' in results:
                    int8_speedup = results['original']['avg_inference_time_ms'] / results['int8']['avg_inference_time_ms']
                    print(f"  INT8 Speedup: {int8_speedup:.1f}x")
                print("Optimization completed successfully!")
            successful_optimizations += 1
            print(f"✅ Successfully optimized {model_name}")

        except Exception as e:
            print(f"❌ Failed to optimize {model_name}: {str(e)}")
            failed_optimizations += 1
            continue

    # Print final summary
    print(f"\n{'='*60}")
    print("📊 BATCH OPTIMIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total models processed: {len(pth_files)}")
    print(f"✅ Successful optimizations: {successful_optimizations}")
    print(f"❌ Failed optimizations: {failed_optimizations}")

    if successful_optimizations > 0:
        print("📁 Optimized models saved to: optimized_models/")
        print("📋 Each model has its own subdirectory with:")
        print("   - FP16 and INT8 optimized .pth files")
        print("   - ONNX models (.onnx files)")
        print("   - Benchmark results and performance comparisons")
        print("   - Optimization summaries")

    print("\n🎉 Batch optimization complete!")

if __name__ == "__main__":
    optimize_all_models()