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

def benchmark_single_model(checkpoint_path, config_path, device='cuda:0', output_dir='optimized_models'):
    """Benchmark a single model"""
    print(f"\n🔍 Benchmarking: {os.path.basename(checkpoint_path)}")
    print("-" * 60)

    try:
        # Initialize optimizer
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
    # ensure output directory exists
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
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

def main():
    parser = argparse.ArgumentParser(description='Batch Benchmark MMSegmentation Models')
    parser.add_argument('--root-dir', default='.', help='Root directory to scan for .pth files')
    parser.add_argument('--config-dir', default='local_configs', help='Directory containing config files')
    parser.add_argument('--device', default='cuda:0', help='Device for benchmarking')
    parser.add_argument('--output-dir', default='optimized_models', help='Output directory for results')
    parser.add_argument('--num-runs', type=int, default=5, help='Number of runs for benchmarking')
    parser.add_argument('--checkpoint', help='Specific checkpoint file to benchmark (optional)')
    parser.add_argument('--eval', action='store_true', help='Also run dataset evaluation (mIoU/accuracy) for each checkpoint (disabled by default)')
    parser.add_argument('--eval-metrics', nargs='+', default=['mIoU'], help='Evaluation metrics to pass to dataset.evaluate when --eval is used')

    args = parser.parse_args()

    print("🚀 MMSegmentation Batch Benchmarking")
    print("=" * 60)

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

        # Optionally evaluate on dataset (single-GPU only). This is guarded so dataset is not loaded by default.
        if args.eval:
                # Import heavy evaluation dependencies lazily and fail gracefully if unavailable
                try:
                    import mmcv
                    import torch

                    # MMDataParallel location has moved across mmcv/mmengine versions. Try common locations
                    MMDataParallel = None
                    try:
                        # preferred new location
                        from mmengine.model import MMDataParallel as _MMDataParallel
                        MMDataParallel = _MMDataParallel
                    except Exception:
                        try:
                            # older mmcv location
                            from mmcv.parallel import MMDataParallel as _MMDataParallel
                            MMDataParallel = _MMDataParallel
                        except Exception:
                            # fallback to torch's DataParallel
                            MMDataParallel = None

                    from mmengine.runner import load_checkpoint
                    from mmseg.apis import single_gpu_test
                    from mmseg.datasets import build_dataset, build_dataloader
                    from mmseg.models import build_segmentor
                except Exception as e:
                    print(f"⚠️ Required packages for evaluation are missing or failed to import: {e}")
                    print("Skipping evaluation for this run. Install mmcv/mmengine/mmseg to enable --eval.")
                    continue

            try:
                print(f"\n📈 Running evaluation for {pth_file} using metrics: {args.eval_metrics}")
                cfg = mmcv.Config.fromfile(config_path)
                cfg.model.pretrained = None
                cfg.data.test.test_mode = True

                dataset = build_dataset(cfg.data.test)
                data_loader = build_dataloader(
                    dataset,
                    samples_per_gpu=1,
                    workers_per_gpu=cfg.data.workers_per_gpu,
                    dist=False,
                    shuffle=False
                )

                cfg.model.train_cfg = None
                model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
                checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
                if 'meta' in checkpoint and checkpoint['meta'] is not None:
                    if 'CLASSES' in checkpoint['meta']:
                        model.CLASSES = checkpoint['meta']['CLASSES']
                    if 'PALETTE' in checkpoint['meta']:
                        model.PALETTE = checkpoint['meta']['PALETTE']

                # Wrap model for single-GPU testing. Use the detected MMDataParallel if available,
                # otherwise fall back to torch.nn.DataParallel which is widely available.
                if MMDataParallel is not None:
                    model = MMDataParallel(model.cuda(), device_ids=[0])
                else:
                    model = torch.nn.DataParallel(model.cuda(), device_ids=[0])
                outputs = single_gpu_test(model, data_loader, show=False, show_dir=None, efficient_test=True)

                # Run dataset evaluation and save results
                eval_kwargs = {}
                results_eval = dataset.evaluate(outputs, args.eval_metrics, **eval_kwargs)
                # Save evaluation results next to benchmark summary
                bench_dir = Path(args.output_dir) / 'benchmarks'
                bench_dir.mkdir(parents=True, exist_ok=True)
                eval_file = bench_dir / f"eval_{Path(pth_file).stem}.json"
                with open(eval_file, 'w') as f:
                    json.dump(results_eval, f, indent=2, default=str)
                print(f"✅ Evaluation results saved to: {eval_file}")
            except Exception as e:
                print(f"⚠️ Evaluation failed for {pth_file}: {e}")

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
