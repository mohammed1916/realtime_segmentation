#!/usr/bin/env python3
"""
Batch Benchmarking Script for MMSegmentation Models
Benchmarks all .pth files in the root directory
"""

import sys
import os
import glob
import argparse
from pathlib import Path
import json
import csv
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Add current directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.model_optimizer import ModelOptimizer
import torch
ort = None
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except Exception:
    ort = None
    ONNX_AVAILABLE = False
import numpy as np

def find_pth_files(root_dir):
    """Find all .pth files in the root directory"""
    pth_files = []
    for file in glob.glob(os.path.join(root_dir, "*.pth")):
        pth_files.append(os.path.basename(file))
    return sorted(pth_files)

def find_matching_config(checkpoint_name, config_dir="local_configs"):
    """Find matching config file for a checkpoint"""
    # Common config patterns for SegFormer models
    config_patterns = {
        "segformer.b0": "segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py",
        "segformer.b1": "segformer/segformer_mit-b1_8xb1-160k_cityscapes-1024x1024.py",
        "segformer.b2": "segformer/segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py",
        "segformer.b3": "segformer/segformer_mit-b3_8xb1-160k_cityscapes-1024x1024.py",
        "segformer.b4": "segformer/segformer_mit-b4_8xb1-160k_cityscapes-1024x1024.py",
        "segformer.b5": "segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py"
    }

    for key, config_path in config_patterns.items():
        if key in checkpoint_name:
            full_config_path = os.path.join(config_dir, config_path)
            if os.path.exists(full_config_path):
                return full_config_path

    # Fallback to default config
    default_config = os.path.join(config_dir, "segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py")
    if os.path.exists(default_config):
        return default_config

    return None


def find_latest_variants(root_dir):
    """Scan optimized_models tree and return latest variant files grouped by base model name.

    Returns dict: { base_name: { variant: filepath, ... }, ... }
    Variant keys come from the suffix after last '_' in filename (before extension), e.g. 'fp16', 'onnx', 'onnx_optimized', 'int8'
    """
    variants_by_base = {}

    for file in glob.glob(os.path.join(root_dir, '**'), recursive=True):
        if not os.path.isfile(file):
            continue
        name = os.path.basename(file)
        stem = Path(name).stem

        # Only consider commonly produced variants
        allowed = ('fp16', 'int8', 'onnx_optimized', 'onnx', 'tensorrt', 'engine')

        matched = False
        for a in allowed:
            suffix = '_' + a
            if stem.endswith(suffix):
                base_name = stem[:-len(suffix)]
                variant = a
                matched = True
                break

        if not matched:
            continue
        mtime = os.path.getmtime(file)

        variants_by_base.setdefault(base_name, {})

        # Keep latest file per variant
        prev = variants_by_base[base_name].get(variant)
        if prev is None or mtime > prev[1]:
            variants_by_base[base_name][variant] = (file, mtime)

    # Strip mtimes
    cleaned = {b: {v: p[0] for v, p in mv.items()} for b, mv in variants_by_base.items()}
    return cleaned

def benchmark_single_model(checkpoint_path, config_path, device='cuda:0', output_dir='optimized_models'):
    """Benchmark a single model"""
    print(f"\n🔍 Benchmarking: {os.path.basename(checkpoint_path)}")
    print("-" * 60)

    try:
        # Initialize optimizer (to access init_model and helpers)
        optimizer = ModelOptimizer(config_path, checkpoint_path, device, output_dir)

        # Benchmark original model only (no optimization)
        print("Benchmarking original model...")
        results = optimizer.benchmark_model(optimizer.original_model, num_runs=5)

        # Save results with checkpoint filename
        checkpoint_name = Path(checkpoint_path).stem
        optimizer._save_benchmark_results(results, f"benchmark_{checkpoint_name}_original")

        print(".2f")
        print(".1f")
        print(f"✅ Benchmark completed for {os.path.basename(checkpoint_path)}")

        return results

    except Exception as e:
        print(f"❌ Failed to benchmark {os.path.basename(checkpoint_path)}: {e}")
        return None

def create_batch_summary(all_results, output_dir):
    """Create a summary of all benchmark results"""
    benchmarks_dir = Path(output_dir) / "benchmarks"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save comprehensive results
    summary_file = benchmarks_dir / f"batch_benchmark_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Create CSV summary
    csv_file = benchmarks_dir / f"batch_benchmark_summary_{timestamp}.csv"
    if all_results:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['checkpoint', 'model_size_mb', 'avg_inference_time_ms', 'fps'])
            writer.writeheader()
            for checkpoint, results in all_results.items():
                if results:
                    writer.writerow({
                        'checkpoint': checkpoint,
                        'model_size_mb': results.get('model_size_mb', 0),
                        'avg_inference_time_ms': results.get('avg_inference_time_ms', 0),
                        'fps': results.get('fps', 0)
                    })

    # Create human-readable summary
    txt_file = benchmarks_dir / f"batch_benchmark_summary_{timestamp}.txt"
    with open(txt_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BATCH BENCHMARK SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total models benchmarked: {len(all_results)}\n\n")

        f.write("PERFORMANCE COMPARISON:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Model':<40} {'Size (MB)':<10} {'Time (ms)':<12} {'FPS':<8}\n")
        f.write("-"*80 + "\n")

        for checkpoint, results in all_results.items():
            if results:
                f.write(f"{checkpoint:<40} {results['model_size_mb']:<10.1f} "
                       f"{results['avg_inference_time_ms']:<12.2f} {results['fps']:<8.1f}\n")

        f.write("-"*80 + "\n")
        f.write(f"\nDetailed results saved to: {benchmarks_dir}\n")

    return summary_file


def benchmark_optimized_variants(output_dir, config_dir='local_configs', device='cuda:0', num_runs=5):
    """Benchmark latest optimized variants found under output_dir.

    Supports .pth variants (fp16/int8) by loading state dict into a fresh model instance
    created from the matching config; supports ONNX via ONNX Runtime.
    """
    root = Path(output_dir)
    # Store all benchmark outputs under a top-level `results/benchmarks` directory
    result_root = Path('results')
    benchmarks_dir = result_root / 'benchmarks'
    benchmarks_dir.mkdir(parents=True, exist_ok=True)

    variants = find_latest_variants(output_dir)
    all_results = {}

    for base_name, var_map in sorted(variants.items()):
        print(f"\n🔍 Benchmarking optimized session for: {base_name}")
        results = {}

        # Find matching config
        config_path = find_matching_config(base_name, config_dir)
        if not config_path:
            print(f"⚠️  No matching config found for {base_name}, skipping pth variants (ONNX still may be benchmarkable)")

        # Helper to build a model from config and a state file
        def build_model_from_state(state_path, variant_key):
            from mmseg.apis import init_model
            # Build model without loading weights
            model = init_model(str(config_path), None, device='cpu')

            sd = torch.load(state_path, map_location='cpu')
            sd = sd.get('state_dict', sd) if isinstance(sd, dict) else sd

            # Normalize keys
            new = {}
            for k, v in sd.items():
                nk = k[len('module.'):] if k.startswith('module.') else k
                new[nk] = v

            model_state = model.state_dict()
            filtered = {}
            for k, v in new.items():
                if k in model_state and tuple(v.shape) == tuple(model_state[k].shape):
                    filtered[k] = v

            model.load_state_dict(filtered, strict=False)

            if 'fp16' in variant_key:
                model = model.half()

            try:
                model.to(device)
            except Exception:
                pass

            return model

        # Benchmark PTH variants
        for v, path in var_map.items():
            if path.endswith('.pth') and config_path:
                try:
                    print(f"Benchmarking {v} (.pth): {path}")
                    model = build_model_from_state(path, v)
                    # Create a temporary ModelOptimizer instance to reuse its benchmark helper
                    # Ensure ModelOptimizer saves into the top-level results directory
                    mo = ModelOptimizer(config_path, path, device, str(result_root))
                    metrics = mo.benchmark_model(model, num_runs=num_runs)
                    metrics['model_file'] = path
                    results[v] = metrics
                    # Save individual result
                    mo._save_benchmark_results(metrics, f"benchmark_{base_name}_{v}")
                except Exception as e:
                    print(f"❌ Failed to benchmark {path}: {e}")

        # Benchmark ONNX variants
        for v, path in var_map.items():
            if path.endswith('.onnx') and ONNX_AVAILABLE:
                try:
                    print(f"Benchmarking {v} (ONNX): {path}")
                    sess_options = ort.SessionOptions()
                    sess = ort.InferenceSession(path, sess_options)

                    inp = sess.get_inputs()[0]
                    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
                    dtype = np.float16 if 'fp16' in v else np.float32
                    dummy = np.random.randn(*shape).astype(dtype)

                    # Warmup
                    for _ in range(3):
                        _ = sess.run(None, {inp.name: dummy})

                    # Timed runs
                    import time as _time
                    start = _time.time()
                    for _ in range(num_runs):
                        _ = sess.run(None, {inp.name: dummy})
                    end = _time.time()
                    avg_ms = (end - start) / num_runs * 1000
                    fps = 1000 / avg_ms if avg_ms > 0 else 0
                    size_mb = os.path.getsize(path) / 1024 / 1024

                    metrics = {
                        'avg_inference_time_ms': avg_ms,
                        'fps': fps,
                        'model_size_mb': size_mb,
                        'model_file': path
                    }
                    results[v] = metrics
                    # Save JSON/CSV
                    base_fname = Path(path).stem
                    with open(benchmarks_dir / f"benchmark_{base_name}_{v}.json", 'w') as f:
                        json.dump(metrics, f, indent=2)
                except Exception as e:
                    print(f"❌ Failed to benchmark ONNX {path}: {e}")

        # Store results
        if results:
            all_results[base_name] = results

    # Save batch summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_json = benchmarks_dir / f"optimized_batch_benchmark_summary_{timestamp}.json"
    with open(summary_json, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # CSV summary
    csv_file = benchmarks_dir / f"optimized_batch_benchmark_summary_{timestamp}.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'variant', 'model_size_mb', 'avg_inference_time_ms', 'fps', 'file'])
        for model, vars in all_results.items():
            for v, m in vars.items():
                writer.writerow([model, v, m.get('model_size_mb', 0), m.get('avg_inference_time_ms', 0), m.get('fps', 0), m.get('model_file', '')])

    print(f"\n📊 Optimized batch benchmarking completed. Results saved to: {benchmarks_dir}")
    return summary_json

def main():
    parser = argparse.ArgumentParser(description='Batch Benchmark MMSegmentation Models')
    parser.add_argument('--root-dir', default='.', help='Root directory to scan for .pth files')
    parser.add_argument('--config-dir', default='local_configs', help='Directory containing config files')
    parser.add_argument('--device', default='cuda:0', help='Device for benchmarking')
    parser.add_argument('--output-dir', default='optimized_models', help='Output directory for results')
    parser.add_argument('--num-runs', type=int, default=5, help='Number of runs for benchmarking')
    parser.add_argument('--checkpoint', help='Specific checkpoint file to benchmark (optional)')
    parser.add_argument('--optimized', action='store_true', help='Benchmark latest optimized variants in output-dir')

    args = parser.parse_args()

    print("🚀 MMSegmentation Batch Benchmarking")
    print("=" * 60)

    if args.optimized:
        print("📦 Running optimized-variants benchmark...")
        summary = benchmark_optimized_variants(args.output_dir, args.config_dir, args.device, args.num_runs)
        print(f"✅ Optimized variants summary saved to: {summary}")
        return

    # Find .pth files
    if args.checkpoint:
        pth_files = [args.checkpoint]
        print(f"📁 Benchmarking specific file: {args.checkpoint}")
    else:
        pth_files = find_pth_files(args.root_dir)
        print(f"📁 Found {len(pth_files)} .pth files in {args.root_dir}")

    if not pth_files:
        print("❌ No .pth files found!")
        return

    # Benchmark all models
    all_results = {}

    for pth_file in pth_files:
        checkpoint_path = os.path.join(args.root_dir, pth_file)

        # Find matching config
        config_path = find_matching_config(pth_file, args.config_dir)

        if not config_path:
            print(f"⚠️  No matching config found for {pth_file}, skipping...")
            continue

        print(f"📋 Using config: {config_path}")

        # Benchmark the model
        results = benchmark_single_model(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device=args.device,
            output_dir=args.output_dir
        )

        if results:
            all_results[pth_file] = results

    # Create batch summary
    if all_results:
        summary_file = create_batch_summary(all_results, args.output_dir)
        print("\n📊 Batch benchmarking completed!")
        print(f"📄 Summary saved to: {summary_file}")

        # Print quick summary
        print("\n🎯 QUICK RESULTS SUMMARY:")
        print("-" * 50)
        for checkpoint, results in sorted(all_results.items()):
            if results:
                print(f"{checkpoint:<40} {results['model_size_mb']:<10.1f} "
                      f"{results['avg_inference_time_ms']:<12.2f} {results['fps']:<8.1f}")
    else:
        print("❌ No models were successfully benchmarked!")

if __name__ == "__main__":
    main()
