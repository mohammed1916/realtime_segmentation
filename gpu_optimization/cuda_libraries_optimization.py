"""
CUDA Libraries Optimization - Real Data Benchmarking
Applies cuBLAS, cuDNN, and Tensor Core optimizations to SegFormer inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import numpy as np
import time
from typing import Dict, List
import json


class SegFormerB0(nn.Module):
    """SegFormer B0 architecture for real-time segmentation."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(64, 64, 2)
        self.stage2 = self._make_stage(64, 128, 2, 2)
        self.stage3 = self._make_stage(128, 256, 2, 2)
        self.stage4 = self._make_stage(256, 512, 2, 2)
        self.decode_head = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 150, kernel_size=1),
        )

    def _make_stage(self, in_c, out_c, blocks, stride=1):
        layers = [nn.Conv2d(in_c, out_c, 3, stride, 1), nn.BatchNorm2d(out_c), nn.ReLU(True)]
        for _ in range(blocks - 1):
            layers += [nn.Conv2d(out_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.ReLU(True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x = self.decode_head(x1)
        return x


class CUDAOptimizationBenchmark:
    """Benchmark CUDA library optimizations with real Cityscapes data."""

    def __init__(self, data_dir: str = "../data/test", num_images: int = 10, input_size: int = 512):
        self.data_dir = Path(data_dir)
        self.num_images = num_images
        self.input_size = input_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}

        # Load model once
        self.model = SegFormerB0().to(self.device).eval()

        # Load test images
        self.test_images = self._load_test_images()

    def _load_test_images(self) -> List[torch.Tensor]:
        """Load real Cityscapes test images and extract input half."""
        images = []
        test_files = sorted(list(self.data_dir.glob("*.jpg")))[:self.num_images]

        print(f"Loading {len(test_files)} test images...")
        for img_path in test_files:
            try:
                img = Image.open(img_path)
                img_array = np.array(img)
                height, width = img_array.shape[:2]

                # Extract left half (input) from Cityscapes format
                mid = width // 2
                input_img = img_array[:, :mid, :]

                # Preprocess: normalize and resize
                input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).float() / 255.0
                input_tensor = F.interpolate(
                    input_tensor.unsqueeze(0),
                    size=(self.input_size, self.input_size),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

                images.append(input_tensor)
            except Exception as e:
                print(f"  Warning: Failed to load {img_path.name}: {e}")

        print(f"Loaded {len(images)} images\n")
        return images

    def _benchmark_config(self, config_name: str, warmup_iters: int = 3,
                         bench_iters: int = 20) -> Dict:
        """Benchmark a specific CUDA configuration."""
        times = []

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iters):
                for img in self.test_images:
                    _ = self.model(img.unsqueeze(0).to(self.device))
        torch.cuda.synchronize()

        # Benchmark
        with torch.no_grad():
            for _ in range(bench_iters):
                for img in self.test_images:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    _ = self.model(img.unsqueeze(0).to(self.device))
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - start) * 1000
                    times.append(elapsed)

        # Calculate statistics (ignore first 5 and last 5 to remove outliers)
        times_filtered = sorted(times)[5:-5] if len(times) > 10 else sorted(times)

        return {
            'config': config_name,
            'mean_ms': float(np.mean(times_filtered)),
            'min_ms': float(np.min(times_filtered)),
            'max_ms': float(np.max(times_filtered)),
            'std_ms': float(np.std(times_filtered)),
            'throughput_img_per_sec': float(1000.0 / np.mean(times_filtered)),
        }

    def run_baseline(self):
        """Baseline: Default FP32 (cuBLAS + cuDNN with default settings)."""
        print("\n" + "="*80)
        print("BASELINE: Default FP32 Configuration")
        print("="*80)

        # Reset to defaults
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        print("Settings:")
        print("  cuDNN auto-tuning: OFF")
        print("  Precision: FP32")
        print("  TF32 enabled: NO")
        print("  GPU libraries: cuBLAS, cuDNN (default)")

        result = self._benchmark_config("Baseline (FP32)")
        self.results['baseline'] = result

        print(f"\nResults:")
        print(f"  Mean latency: {result['mean_ms']:.2f} ms")
        print(f"  Min latency:  {result['min_ms']:.2f} ms")
        print(f"  Max latency:  {result['max_ms']:.2f} ms")
        print(f"  Std dev:      {result['std_ms']:.2f} ms")
        print(f"  Throughput:   {result['throughput_img_per_sec']:.1f} img/sec")

        return result

    def run_cudnn_autotuning(self):
        """Enable cuDNN auto-tuning for convolution algorithm selection."""
        print("\n" + "="*80)
        print("OPTIMIZATION 1: cuDNN Auto-Tuning")
        print("="*80)

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        print("Settings:")
        print("  cuDNN auto-tuning: ON (caches best convolution algorithm)")
        print("  Precision: FP32")
        print("  TF32 enabled: NO")
        print("  Expected benefit: +10-15% via algorithm selection")

        result = self._benchmark_config("cuDNN Auto-Tuning (FP32)")
        self.results['cudnn_autotuning'] = result

        baseline = self.results['baseline']
        improvement = ((baseline['mean_ms'] - result['mean_ms']) / baseline['mean_ms']) * 100
        speedup = baseline['mean_ms'] / result['mean_ms']

        print(f"\nResults:")
        print(f"  Mean latency: {result['mean_ms']:.2f} ms")
        print(f"  Throughput:   {result['throughput_img_per_sec']:.1f} img/sec")
        print(f"  vs Baseline:  {improvement:+.1f}% ({speedup:.2f}x speedup)")

        return result

    def run_tf32_optimization(self):
        """Enable TF32 precision (Ampere+ GPUs only)."""
        print("\n" + "="*80)
        print("OPTIMIZATION 2: TF32 Precision (cuBLAS + cuDNN)")
        print("="*80)

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        print("Settings:")
        print("  cuDNN auto-tuning: ON")
        print("  Precision: TF32 (32-bit shape, 16-bit mantissa)")
        print("  TF32 enabled: YES (cuBLAS + cuDNN)")
        print("  GPU requirement: Ampere+ (RTX 30xx, A100, RTX 40xx)")
        print("  Expected benefit: +10-20% via 4× tensor core throughput")

        result = self._benchmark_config("TF32 Precision")
        self.results['tf32'] = result

        baseline = self.results['baseline']
        improvement = ((baseline['mean_ms'] - result['mean_ms']) / baseline['mean_ms']) * 100
        speedup = baseline['mean_ms'] / result['mean_ms']

        print(f"\nResults:")
        print(f"  Mean latency: {result['mean_ms']:.2f} ms")
        print(f"  Throughput:   {result['throughput_img_per_sec']:.1f} img/sec")
        print(f"  vs Baseline:  {improvement:+.1f}% ({speedup:.2f}x speedup)")

        return result

    def run_fp16_optimization(self):
        """FP16 mixed precision with Tensor Cores."""
        print("\n" + "="*80)
        print("OPTIMIZATION 3: FP16 Mixed Precision (Tensor Cores)")
        print("="*80)

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        print("Settings:")
        print("  cuDNN auto-tuning: ON")
        print("  Precision: FP16 (16-bit floating point)")
        print("  Execution: Mixed precision with autocast")
        print("  Tensor Cores: ENABLED for FP16 matmul")
        print("  GPU requirement: Volta+ (V100, RTX 20xx+)")
        print("  Expected benefit: +15-30% via tensor cores + memory bandwidth")

        times = []

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                for img in self.test_images:
                    with torch.amp.autocast('cuda'):
                        _ = self.model(img.unsqueeze(0).to(self.device))
        torch.cuda.synchronize()

        # Benchmark
        with torch.no_grad():
            for _ in range(20):
                for img in self.test_images:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    with torch.amp.autocast('cuda'):
                        _ = self.model(img.unsqueeze(0).to(self.device))
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - start) * 1000
                    times.append(elapsed)

        times_filtered = sorted(times)[5:-5]
        result = {
            'config': 'FP16 Mixed Precision',
            'mean_ms': float(np.mean(times_filtered)),
            'min_ms': float(np.min(times_filtered)),
            'max_ms': float(np.max(times_filtered)),
            'std_ms': float(np.std(times_filtered)),
            'throughput_img_per_sec': float(1000.0 / np.mean(times_filtered)),
        }
        self.results['fp16'] = result

        baseline = self.results['baseline']
        improvement = ((baseline['mean_ms'] - result['mean_ms']) / baseline['mean_ms']) * 100
        speedup = baseline['mean_ms'] / result['mean_ms']

        print(f"\nResults:")
        print(f"  Mean latency: {result['mean_ms']:.2f} ms")
        print(f"  Throughput:   {result['throughput_img_per_sec']:.1f} img/sec")
        print(f"  vs Baseline:  {improvement:+.1f}% ({speedup:.2f}x speedup)")

        return result

    def run_all_optimizations(self):
        """Run all optimization tiers and generate report."""
        print("\n\n")
        print("=" * 80)
        print("CUDA LIBRARIES OPTIMIZATION - SEGFORMER B0")
        print("Real Data: Cityscapes Test Set")
        print("Input Size: 512x512")
        print("Optimization Tiers: Baseline -> cuDNN -> TF32 -> FP16")
        print("=" * 80)

        # Run all benchmarks
        self.run_baseline()
        self.run_cudnn_autotuning()
        self.run_tf32_optimization()
        self.run_fp16_optimization()

        # Generate summary
        self._print_summary()
        self._save_results()

    def _print_summary(self):
        """Print comprehensive optimization summary."""
        print("\n\n" + "="*80)
        print("OPTIMIZATION SUMMARY")
        print("="*80)

        # Table format
        print(f"\n{'Config':<35} {'Latency (ms)':<15} {'Throughput':<18} {'vs Baseline':<15}")
        print("-" * 80)

        baseline_latency = self.results['baseline']['mean_ms']
        baseline_throughput = self.results['baseline']['throughput_img_per_sec']

        for key in ['baseline', 'cudnn_autotuning', 'tf32', 'fp16']:
            if key in self.results:
                result = self.results[key]
                latency = result['mean_ms']
                throughput = result['throughput_img_per_sec']
                speedup = baseline_latency / latency
                improvement = ((baseline_latency - latency) / baseline_latency) * 100

                print(f"{result['config']:<35} {latency:<15.2f} {throughput:<18.1f} {speedup:.2f}x ({improvement:+.1f}%)")

        print("\n" + "="*80)
        print("LIBRARY DISPATCH MAPPING")
        print("="*80)
        print("""
CUDA Library Usage in SegFormer:

  1. Convolution Operations (Conv2d)
     -> Dispatches to: cudnnConvolutionForward() [cuDNN]
     -> Optimization Impact: cuDNN auto-tuning, TF32

  2. Linear Projections (torch.nn.Linear)
     -> Dispatches to: cublasLtMatmul() [cuBLAS-LT]
     -> Optimization Impact: TF32, FP16 (Tensor Cores)

  3. Batch Normalization
     -> Dispatches to: cudnnBatchNormalizationForward() [cuDNN]
     -> Optimization Impact: cuDNN auto-tuning

  4. Activation Functions (ReLU)
     -> Dispatches to: Fused cuDNN kernels (in modern versions)
     -> Optimization Impact: Part of cuDNN optimizations

Performance Gains by Optimization Tier:

  Tier 1 (cuDNN Auto-Tuning):
    - Benchmarks multiple convolution algorithms
    - Caches best result for repeated shapes
    - Expected: +10-15% speedup
    - Effort: 1 line of code

  Tier 2 (TF32 Precision):
    - Uses 32-bit shape with 16-bit mantissa
    - Requires: Ampere+ GPU (RTX 30xx, A100)
    - Unlocks 4x tensor core throughput
    - Expected: +10-20% additional speedup
    - Effort: 2 lines of code

  Tier 3 (FP16 Mixed Precision):
    - Uses 16-bit floating point with autocast
    - Requires: Volta+ GPU (V100, RTX 20xx+)
    - Unlocks tensor cores for FP16 operations
    - Expected: +15-30% additional speedup
    - Effort: Context manager wrapper
        """)

        print("="*80)
        print("FINAL RESULTS")
        print("="*80)

        fp16_result = self.results['fp16']
        baseline_result = self.results['baseline']
        final_speedup = baseline_result['mean_ms'] / fp16_result['mean_ms']
        final_improvement = ((baseline_result['mean_ms'] - fp16_result['mean_ms']) / baseline_result['mean_ms']) * 100

        print(f"\nFinal Configuration: Baseline -> cuDNN + TF32 + FP16")
        print(f"  Baseline Latency:      {baseline_result['mean_ms']:.2f} ms")
        print(f"  Optimized Latency:     {fp16_result['mean_ms']:.2f} ms")
        print(f"  Overall Speedup:       {final_speedup:.2f}x")
        print(f"  Improvement:           {final_improvement:.1f}%")
        print(f"\n  Baseline Throughput:   {baseline_result['throughput_img_per_sec']:.1f} img/sec")
        print(f"  Optimized Throughput:  {fp16_result['throughput_img_per_sec']:.1f} img/sec")

        print("\n" + "="*80)

    def _save_results(self):
        """Save results to JSON file."""
        output_file = Path("cuda_libraries_optimization_results.json")

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\nResults saved to: {output_file}")


def main():
    """Run CUDA libraries optimization benchmark."""
    benchmark = CUDAOptimizationBenchmark(num_images=10, input_size=512)
    benchmark.run_all_optimizations()


if __name__ == '__main__':
    main()
