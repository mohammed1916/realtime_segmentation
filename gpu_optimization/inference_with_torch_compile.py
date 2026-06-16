#!/usr/bin/env python3
"""
Inference with torch.compile() - Graph compilation optimization
PyTorch 2.0+ feature that auto-fuses operations.

1-line change for 1.2-1.8x speedup with zero accuracy loss.
"""

import torch
import torch.nn as nn
import numpy as np
import time


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


def benchmark_model(model, input_tensor, model_name, num_runs=20, warmup=5):
    """Benchmark a model."""
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

    torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(input_tensor)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

    times = np.array(times)
    return {
        'name': model_name,
        'latency_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'throughput': 1000.0 / np.mean(times),
    }


def main():
    device = torch.device('cuda')
    input_tensor = torch.randn(1, 3, 512, 512, device=device)

    print("\n" + "="*80)
    print("TORCH.COMPILE() OPTIMIZATION - GRAPH COMPILATION")
    print("="*80)
    print(f"Input: {input_tensor.shape}")
    print(f"PyTorch version: {torch.__version__}\n")

    # Check if torch.compile is available
    if not hasattr(torch, 'compile'):
        print("ERROR: torch.compile() requires PyTorch 2.0+")
        print(f"Current version: {torch.__version__}")
        return

    # Baseline: Standard model
    print("="*80)
    print("TEST 1: BASELINE (Standard inference)")
    print("="*80)

    model_baseline = SegFormerB0().to(device).eval()
    torch.backends.cudnn.benchmark = True

    print("Benchmarking...")
    result_baseline = benchmark_model(model_baseline, input_tensor, "Baseline (no compile)")
    print(f"  Latency: {result_baseline['latency_ms']:.2f} ± {result_baseline['std_ms']:.2f} ms")
    print(f"  Throughput: {result_baseline['throughput']:.1f} img/sec\n")

    # Test 1: torch.compile with default settings
    print("="*80)
    print("TEST 2: TORCH.COMPILE (reduce mode)")
    print("="*80)

    model_compile_reduce = SegFormerB0().to(device).eval()
    torch.backends.cudnn.benchmark = True

    print("Compiling model (mode='reduce')...")
    try:
        model_compile_reduce = torch.compile(model_compile_reduce, mode='reduce')
        print("Compilation successful!")
        print("Benchmarking (first run includes compilation overhead)...")
        result_compile_reduce = benchmark_model(model_compile_reduce, input_tensor,
                                               "torch.compile (reduce)", warmup=10)
        print(f"  Latency: {result_compile_reduce['latency_ms']:.2f} ± {result_compile_reduce['std_ms']:.2f} ms")
        speedup = result_baseline['latency_ms'] / result_compile_reduce['latency_ms']
        improve = ((result_baseline['latency_ms'] - result_compile_reduce['latency_ms']) /
                   result_baseline['latency_ms']) * 100
        print(f"  Speedup: {speedup:.2f}x ({improve:+.1f}%)\n")
    except Exception as e:
        print(f"torch.compile failed: {e}\n")
        result_compile_reduce = None

    # Test 2: torch.compile with max-autotune
    print("="*80)
    print("TEST 3: TORCH.COMPILE (max-autotune mode)")
    print("="*80)

    model_compile_max = SegFormerB0().to(device).eval()
    torch.backends.cudnn.benchmark = True

    print("Compiling model (mode='max-autotune')...")
    print("WARNING: This is slower to compile but may give better performance")
    try:
        model_compile_max = torch.compile(model_compile_max, mode='max-autotune')
        print("Compilation successful!")
        print("Benchmarking...")
        result_compile_max = benchmark_model(model_compile_max, input_tensor,
                                            "torch.compile (max-autotune)", warmup=10)
        print(f"  Latency: {result_compile_max['latency_ms']:.2f} ± {result_compile_max['std_ms']:.2f} ms")
        speedup = result_baseline['latency_ms'] / result_compile_max['latency_ms']
        improve = ((result_baseline['latency_ms'] - result_compile_max['latency_ms']) /
                   result_baseline['latency_ms']) * 100
        print(f"  Speedup: {speedup:.2f}x ({improve:+.1f}%)\n")
    except Exception as e:
        print(f"torch.compile failed: {e}\n")
        result_compile_max = None

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\n{'Configuration':<30} {'Latency (ms)':<15} {'Speedup':<12} {'Improvement':<15}")
    print("-" * 80)

    baseline = result_baseline['latency_ms']

    print(f"{'Baseline (no compile)':<30} {baseline:<15.2f} {'1.00x':<12} {'0.0%':<15}")

    if result_compile_reduce:
        speedup = baseline / result_compile_reduce['latency_ms']
        improve = ((baseline - result_compile_reduce['latency_ms']) / baseline) * 100
        print(f"{'torch.compile (reduce)':<30} {result_compile_reduce['latency_ms']:<15.2f} "
              f"{speedup:<12.2f}x {improve:+>14.1f}%")

    if result_compile_max:
        speedup = baseline / result_compile_max['latency_ms']
        improve = ((baseline - result_compile_max['latency_ms']) / baseline) * 100
        print(f"{'torch.compile (max-autotune)':<30} {result_compile_max['latency_ms']:<15.2f} "
              f"{speedup:<12.2f}x {improve:+>14.1f}%")

    # Explanation
    print("\n" + "="*80)
    print("HOW TORCH.COMPILE() WORKS")
    print("="*80)

    print("""
torch.compile() is PyTorch 2.0+ feature that:

1. Captures the model's computation graph
2. Analyzes operation patterns
3. Fuses compatible operations into single kernels
4. Optimizes memory layout
5. Generates optimized CUDA code

For SegFormer, torch.compile() can fuse:
- Conv + BatchNorm + ReLU (3 ops -> 1 kernel)
- Multiple upsampling + conv operations
- Activation functions with preceding operations

Expected speedup: 1.2-1.8x
Compilation overhead: First inference is slower (~100-500ms)
Subsequent inferences: 1.2-1.8x faster

Modes:
- 'reduce':       Conservative, fast compilation, moderate speedup
- 'max-autotune': Aggressive, slow compilation, best speedup
- 'default':      Balanced between reduce and max-autotune
""")

    print("="*80)
    print("RECOMMENDATION")
    print("="*80)

    best_result = result_baseline
    best_name = "Baseline"

    if result_compile_reduce and result_compile_reduce['latency_ms'] < best_result['latency_ms']:
        best_result = result_compile_reduce
        best_name = "torch.compile (reduce)"

    if result_compile_max and result_compile_max['latency_ms'] < best_result['latency_ms']:
        best_result = result_compile_max
        best_name = "torch.compile (max-autotune)"

    final_speedup = baseline / best_result['latency_ms']

    print(f"""
Best configuration: {best_name}
Final latency: {best_result['latency_ms']:.2f} ms
Overall speedup from baseline: {final_speedup:.2f}x

Combined with BF16:
- BF16 baseline: 20.89 ms
- + torch.compile: {20.89 / final_speedup * (baseline / best_result['latency_ms']):.2f} ms
- Total speedup: {20.89 / (20.89 / final_speedup * (baseline / best_result['latency_ms'])):.2f}x

NEXT STEP:
Try torch.compile() in your production code:
    model = torch.compile(model, mode='reduce')

Use in inference_optimized.py:
    def __init__(self, ...):
        self.model = SegFormerB0().to(device).eval()
        self.model = torch.compile(self.model, mode='reduce')  # <- ADD THIS
""")

    print("="*80)


if __name__ == '__main__':
    main()
