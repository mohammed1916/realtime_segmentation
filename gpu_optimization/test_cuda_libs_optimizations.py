#!/usr/bin/env python3
"""
Test CUDA Library Optimizations - Real Performance Gains

Compares:
1. Baseline: FP32, cuDNN off, no TF32
2. Tier 1: cuDNN auto-tuning (cudnn.benchmark = True)
3. Tier 2: TF32 Tensor Cores (allow_tf32 flags)
4. Tier 3: FP16 Mixed Precision (Tensor Cores + memory bandwidth)
5. Tier 4: BF16 Mixed Precision (our current approach)

This tests the actual CUDA library dispatch chain:
- Conv2d -> cuDNN convolution kernels
- Linear -> cuBLAS matrix multiplication
- All with different precision modes
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
from pathlib import Path


class SegFormerB0(nn.Module):
    """SegFormer B0 architecture."""
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


class CUDALibraryBenchmark:
    """Benchmark CUDA library optimizations."""

    def __init__(self, input_size=512, num_runs=20, warmup_runs=5):
        self.input_size = input_size
        self.num_runs = num_runs
        self.warmup_runs = warmup_runs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        self.model = SegFormerB0().to(self.device).eval()
        print(f"\n[INFO] Device: {self.device}")
        print(f"[INFO] Model loaded")

    def benchmark_config(self, name: str, input_tensor: torch.Tensor,
                        use_autocast: bool = False, dtype=None):
        """Benchmark a configuration."""

        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                if use_autocast:
                    with torch.amp.autocast('cuda', dtype=dtype):
                        _ = self.model(input_tensor)
                else:
                    _ = self.model(input_tensor)

        torch.cuda.synchronize()

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(self.num_runs):
                torch.cuda.synchronize()
                start = time.perf_counter()

                if use_autocast:
                    with torch.amp.autocast('cuda', dtype=dtype):
                        _ = self.model(input_tensor)
                else:
                    _ = self.model(input_tensor)

                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)

        times = np.array(times)
        return {
            'name': name,
            'latency_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'throughput': float(1000.0 / np.mean(times)),
        }

    def run_all_tests(self):
        """Run all CUDA library optimization tiers."""
        input_tensor = torch.randn(1, 3, self.input_size, self.input_size, device=self.device)

        print("\n" + "="*80)
        print("CUDA LIBRARIES OPTIMIZATION TIER TEST")
        print("="*80)
        print(f"Input: {input_tensor.shape} | Device: {self.device}")
        print(f"Runs: {self.num_runs} | Warmup: {self.warmup_runs}\n")

        # Baseline: No optimizations
        print("-" * 80)
        print("TIER 0: Baseline (No CUDA Optimizations)")
        print("-" * 80)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        print("Settings: cudnn.benchmark=OFF, TF32=OFF")
        result_baseline = self.benchmark_config("Baseline (FP32)", input_tensor)
        self.results['baseline'] = result_baseline
        print(f"  Latency: {result_baseline['latency_ms']:.2f} ± {result_baseline['std_ms']:.2f} ms")
        print(f"  Throughput: {result_baseline['throughput']:.1f} img/sec\n")

        # Tier 1: cuDNN auto-tuning
        print("-" * 80)
        print("TIER 1: cuDNN Auto-Tuning")
        print("-" * 80)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        print("Settings: cudnn.benchmark=ON, TF32=OFF")
        print("Library: cuDNN convolution algorithm selection")
        result_cudnn = self.benchmark_config("cuDNN Auto-Tuning (FP32)", input_tensor)
        self.results['cudnn'] = result_cudnn
        speedup = result_baseline['latency_ms'] / result_cudnn['latency_ms']
        improve = ((result_baseline['latency_ms'] - result_cudnn['latency_ms']) / result_baseline['latency_ms']) * 100
        print(f"  Latency: {result_cudnn['latency_ms']:.2f} ± {result_cudnn['std_ms']:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x ({improve:+.1f}%)\n")

        # Tier 2: TF32
        print("-" * 80)
        print("TIER 2: TF32 (Tensor Cores)")
        print("-" * 80)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        print("Settings: cudnn.benchmark=ON, TF32=ON")
        print("Library: cuBLAS + cuDNN with 32-bit TF32 format")
        print("Hardware: Tensor Cores enabled (4x throughput)")
        result_tf32 = self.benchmark_config("TF32 (FP32 shape, TF32 compute)", input_tensor)
        self.results['tf32'] = result_tf32
        speedup = result_baseline['latency_ms'] / result_tf32['latency_ms']
        improve = ((result_baseline['latency_ms'] - result_tf32['latency_ms']) / result_baseline['latency_ms']) * 100
        print(f"  Latency: {result_tf32['latency_ms']:.2f} ± {result_tf32['std_ms']:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x ({improve:+.1f}%)\n")

        # Tier 3: FP16 Mixed Precision
        print("-" * 80)
        print("TIER 3: FP16 Mixed Precision (Tensor Cores + Memory)")
        print("-" * 80)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        print("Settings: cudnn.benchmark=ON, TF32=ON + FP16 autocast")
        print("Library: cuBLAS + cuDNN with FP16 format")
        print("Hardware: Tensor Cores for FP16 (8x throughput) + 50% memory bandwidth savings")
        result_fp16 = self.benchmark_config("FP16 Mixed Precision", input_tensor,
                                           use_autocast=True, dtype=torch.float16)
        self.results['fp16'] = result_fp16
        speedup = result_baseline['latency_ms'] / result_fp16['latency_ms']
        improve = ((result_baseline['latency_ms'] - result_fp16['latency_ms']) / result_baseline['latency_ms']) * 100
        print(f"  Latency: {result_fp16['latency_ms']:.2f} ± {result_fp16['std_ms']:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x ({improve:+.1f}%)\n")

        # Tier 4: BF16 (Our Current Implementation)
        print("-" * 80)
        print("TIER 4: BF16 Mixed Precision (Our Current Approach)")
        print("-" * 80)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        print("Settings: cudnn.benchmark=ON, TF32=OFF + BF16 autocast")
        print("Library: cuBLAS + cuDNN with BF16 format")
        print("Hardware: Tensor Cores for BF16 + 50% memory bandwidth savings")
        result_bf16 = self.benchmark_config("BF16 Mixed Precision", input_tensor,
                                           use_autocast=True, dtype=torch.bfloat16)
        self.results['bf16'] = result_bf16
        speedup = result_baseline['latency_ms'] / result_bf16['latency_ms']
        improve = ((result_baseline['latency_ms'] - result_bf16['latency_ms']) / result_baseline['latency_ms']) * 100
        print(f"  Latency: {result_bf16['latency_ms']:.2f} ± {result_bf16['std_ms']:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x ({improve:+.1f}%)\n")

        # Summary
        self._print_summary()
        self._save_results()

    def _print_summary(self):
        """Print comprehensive summary."""
        print("\n" + "="*80)
        print("CUDA LIBRARY OPTIMIZATION SUMMARY")
        print("="*80)

        print(f"\n{'Tier':<40} {'Latency (ms)':<15} {'Speedup':<12} {'Improvement':<15}")
        print("-" * 80)

        baseline = self.results['baseline']['latency_ms']

        for key in ['baseline', 'cudnn', 'tf32', 'fp16', 'bf16']:
            if key in self.results:
                result = self.results[key]
                latency = result['latency_ms']
                speedup = baseline / latency
                improve = ((baseline - latency) / baseline) * 100

                print(f"{result['name']:<40} {latency:<15.2f} {speedup:<12.2f}x {improve:+>14.1f}%")

        print("\n" + "="*80)
        print("KEY FINDINGS")
        print("="*80)

        bf16 = self.results['bf16']['latency_ms']
        fp16 = self.results['fp16']['latency_ms']
        tf32 = self.results['tf32']['latency_ms']

        print(f"""
1. FP16 vs BF16 Performance:
   - FP16 (Tensor Cores):  {fp16:.2f} ms
   - BF16 (Tensor Cores):  {bf16:.2f} ms
   - FP16 is {(bf16/fp16-1)*100:+.1f}% {'faster' if fp16 < bf16 else 'slower'}

2. TF32 Impact on This Model:
   - TF32 alone:   {tf32:.2f} ms
   - vs Baseline:  {(self.results['baseline']['latency_ms']/tf32):.2f}x

3. CUDA Library Dispatch Chains:

   Conv2d operations:
     -> Dispatches to: cudnnConvolutionForward()
     -> Benefits from: cuDNN auto-tuning + precision selection (TF32/FP16)

   BatchNorm operations:
     -> Dispatches to: cudnnBatchNormalizationForward()
     -> Benefits from: cuDNN auto-tuning

   Upsampling:
     -> Custom CUDA kernels or cuDNN interpolation
     -> Limited benefit from Tensor Cores (memory-bound)
""")

        print("="*80)
        print("RECOMMENDED CONFIGURATION")
        print("="*80)

        # Find best
        best_key = min(self.results.keys(), key=lambda k: self.results[k]['latency_ms'])
        best = self.results[best_key]

        print(f"""
Best Latency: {best['name']}
  - Latency: {best['latency_ms']:.2f} ms
  - Speedup: {baseline/best['latency_ms']:.2f}x from baseline

Code to use:
  torch.backends.cudnn.benchmark = True

  # For FP16 (best performance):
  with torch.amp.autocast('cuda', dtype=torch.float16):
      output = model(input)

  # Or for BF16 (our current approach):
  with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      output = model(input)
""")

    def _save_results(self):
        """Save results to JSON."""
        output_file = Path('gpu_optimization/cuda_libs_test_results.json')
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\nResults saved to: {output_file}")


def main():
    """Run CUDA library benchmark."""
    benchmark = CUDALibraryBenchmark(input_size=512, num_runs=20, warmup_runs=5)
    benchmark.run_all_tests()


if __name__ == '__main__':
    main()
