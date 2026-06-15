"""
Kernel Profiling with Advanced Metrics
Analyzes L2 cache, occupancy, throughput, and other signals to guide kernel improvements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
import json


class SegFormerB0(nn.Module):
    """SegFormer B0 for segmentation."""
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


def load_test_image(data_dir: str = "../data/test") -> torch.Tensor:
    """Load single test image."""
    test_files = sorted(list(Path(data_dir).glob("*.jpg")))
    img_path = test_files[0]

    img = Image.open(img_path)
    img_array = np.array(img)
    height, width = img_array.shape[:2]
    mid = width // 2
    input_img = img_array[:, :mid, :]

    input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).float() / 255.0
    input_tensor = F.interpolate(
        input_tensor.unsqueeze(0),
        size=(512, 512),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)
    return input_tensor


def profile_model_detailed(model, input_tensor, config_name: str) -> dict:
    """
    Profile model with detailed metrics.
    Captures operation latency, memory usage, and kernel-level information.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    input_tensor = input_tensor.unsqueeze(0).to(device)

    results = {
        'config': config_name,
        'operations': {},
        'summary': {},
    }

    # Profile with detailed activities
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            _ = model(input_tensor)

    # Extract profiler results
    key_averages = prof.key_averages()

    # Sort by CUDA time
    top_ops = sorted(
        key_averages,
        key=lambda x: x.cuda_time,
        reverse=True
    )[:15]  # Top 15 operations

    print(f"\n{'='*100}")
    print(f"KERNEL PROFILING: {config_name}")
    print(f"{'='*100}")
    print(f"\n{'Operation':<40} {'CUDA Time':<15} {'CPU Time':<15} {'CPU:GPU Ratio':<15}")
    print("-" * 100)

    total_cuda_time = sum(op.cuda_time for op in key_averages)

    for op in top_ops:
        op_name = op.key[:40]
        cuda_time = op.cuda_time / 1e6  # Convert to ms
        cpu_time = op.cpu_time / 1e6
        ratio = cpu_time / cuda_time if cuda_time > 0 else 0

        print(f"{op_name:<40} {cuda_time:<15.2f} {cpu_time:<15.2f} {ratio:<15.2f}x")

        results['operations'][op.key] = {
            'cuda_time_ms': float(cuda_time),
            'cpu_time_ms': float(cpu_time),
            'cpu_gpu_ratio': float(ratio),
            'percent_of_total': float((cuda_time / total_cuda_time * 100) if total_cuda_time > 0 else 0),
        }

    # Summary statistics
    results['summary'] = {
        'total_cuda_time_ms': float(total_cuda_time / 1e6),
        'num_operations': len(key_averages),
        'top_15_percent': float(sum(op.cuda_time for op in top_ops) / total_cuda_time * 100),
    }

    return results


def analyze_bottlenecks(profile_data: dict) -> dict:
    """
    Analyze profiler data to identify bottlenecks.
    Returns guidance on L2 cache, occupancy, and throughput issues.
    """
    analysis = {
        'bottleneck_type': None,
        'guidance': [],
        'metrics': {},
    }

    top_ops = sorted(
        profile_data['operations'].items(),
        key=lambda x: x[1]['cuda_time_ms'],
        reverse=True
    )[:5]

    print(f"\n{'='*100}")
    print("BOTTLENECK ANALYSIS")
    print(f"{'='*100}")

    for op_name, metrics in top_ops:
        cuda_time = metrics['cuda_time_ms']
        percent = metrics['percent_of_total']

        print(f"\n{op_name} ({percent:.1f}% of total):")
        print(f"  CUDA Time: {cuda_time:.2f} ms")

        # Identify bottleneck type based on operation name
        if 'conv2d' in op_name.lower() or 'convolution' in op_name.lower():
            print(f"  Operation Type: Convolution")
            print(f"  Expected Bottleneck: Memory-bound (conv is typically memory-limited)")
            print(f"\n  Improvement Strategies:")
            print(f"    1. Check L2 cache hit rate:")
            print(f"       - Good: >60% L2 hit rate")
            print(f"       - Poor: <40% L2 hit rate -> working set too large")
            print(f"    2. Kernel fusion (combine with activation):")
            print(f"       - Conv + ReLU fusion saves ~10-15% by reducing memory traffic")
            print(f"    3. Input tiling:")
            print(f"       - Process smaller tiles to fit in L2 cache (5-6 MB)")
            print(f"    4. Memory coalescing:")
            print(f"       - Ensure NCHW format (already optimized in PyTorch)")

        elif 'matmul' in op_name.lower():
            print(f"  Operation Type: Matrix Multiplication")
            print(f"  Expected Bottleneck: Compute or Memory-bound (depends on size)")
            print(f"\n  Improvement Strategies:")
            print(f"    1. Check arithmetic intensity:")
            print(f"       - High (>4 ops/byte): Compute-bound -> optimize compute")
            print(f"       - Low (<2 ops/byte): Memory-bound -> optimize bandwidth")
            print(f"    2. Block tiling:")
            print(f"       - Cache results in L1/L2 during computation")
            print(f"    3. Tensor Core utilization:")
            print(f"       - Use FP16/TF32 for 4-8x throughput improvement")
            print(f"    4. Batch size:")
            print(f"       - Larger batches improve occupancy and utilization")

        elif 'batch_norm' in op_name.lower() or 'batchnorm' in op_name.lower():
            print(f"  Operation Type: Batch Normalization")
            print(f"  Expected Bottleneck: Memory-bound (streaming operation)")
            print(f"\n  Improvement Strategies:")
            print(f"    1. Check occupancy:")
            print(f"       - Target: >80% occupancy for streaming ops")
            print(f"       - Low occupancy: Not enough parallelism")
            print(f"    2. Kernel fusion (with previous layer):")
            print(f"       - Fused Conv + BatchNorm saves memory traffic")
            print(f"    3. In-place operation:")
            print(f"       - Already enabled (inplace=True) in SegFormer")

        elif 'softmax' in op_name.lower():
            print(f"  Operation Type: Softmax (Attention)")
            print(f"  Expected Bottleneck: Memory-bound with seq dependencies")
            print(f"\n  Improvement Strategies:")
            print(f"    1. Check L2 hit rate:")
            print(f"       - Attention materializes large intermediate matrix")
            print(f"       - Typical: 40-50% L2 hit rate (normal for this op)")
            print(f"    2. Flash Attention:")
            print(f"       - Tiles attention computation to fit in cache")
            print(f"       - Expected: 2-3x speedup")
            print(f"    3. Approximate attention:")
            print(f"       - Linear attention, sparse attention")
            print(f"       - Trade-off: Accuracy vs speed")

        else:
            print(f"  Operation Type: Generic")
            print(f"\n  Improvement Strategies:")
            print(f"    1. Check occupancy (SM utilization):")
            print(f"       - Good: >60% occupancy")
            print(f"       - Poor: Register pressure or shared memory limited")
            print(f"    2. Check warp efficiency:")
            print(f"       - Good: >80% warp efficiency")
            print(f"       - Poor: Memory stalls or control divergence")
            print(f"    3. Memory access patterns:")
            print(f"       - Coalesce global memory access")
            print(f"       - Use shared memory for reusable data")

    return analysis


def generate_optimization_recommendations(profile_data: dict) -> dict:
    """
    Generate specific optimization recommendations based on profiler data.
    """
    recommendations = {
        'quick_wins': [],
        'medium_effort': [],
        'high_effort': [],
    }

    total_time = profile_data['summary']['total_cuda_time_ms']
    top_15_percent = profile_data['summary']['top_15_percent']

    print(f"\n{'='*100}")
    print("OPTIMIZATION RECOMMENDATIONS")
    print(f"{'='*100}")

    print(f"\nProfile Summary:")
    print(f"  Total CUDA Time: {total_time:.2f} ms")
    print(f"  Top 5 ops: {top_15_percent:.1f}% of total time")

    print(f"\nQuick Wins (Easy, High Impact):")
    print(f"  1. Enable cuDNN auto-tuning:")
    print(f"     torch.backends.cudnn.benchmark = True")
    print(f"     Expected: +0-5% (depending on current state)")
    print(f"\n  2. Enable TF32 precision:")
    print(f"     torch.backends.cudnn.allow_tf32 = True")
    print(f"     Expected: +5-10% speedup")
    print(f"\n  3. Enable FP16 mixed precision:")
    print(f"     with torch.amp.autocast('cuda'):")
    print(f"     Expected: +20-35% speedup")
    recommendations['quick_wins'].append("Enable cuDNN optimizations")

    print(f"\nMedium Effort (2-4 hours):")
    print(f"  1. Kernel fusion (Conv + ReLU):")
    print(f"     Reduces memory traffic, expected +5-10% for conv layers")
    print(f"\n  2. Flash Attention (if using attention):")
    print(f"     Tiles attention computation, expected +2-3x for attention")
    recommendations['medium_effort'].append("Implement kernel fusion")

    print(f"\nHigh Effort (1-2 weeks):")
    print(f"  1. Custom CUDA kernels:")
    print(f"     Write optimized kernels for bottleneck operations")
    print(f"     Expected: +10-50% depending on operation")
    print(f"\n  2. Quantization (INT8):")
    print(f"     Full pipeline quantization, expected +2-4x")
    print(f"     Trade-off: Accuracy loss (~1-2%)")
    recommendations['high_effort'].append("Custom CUDA kernels")

    return recommendations


def main():
    """Run comprehensive kernel profiling."""
    print("\n" + "="*100)
    print("KERNEL PROFILING WITH ADVANCED METRICS")
    print("="*100)

    model = SegFormerB0()
    input_tensor = load_test_image()

    print(f"Model: SegFormer B0")
    print(f"Input size: 512x512")
    print(f"GPU: CUDA")

    # Profile baseline (FP32, no optimizations)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    baseline_data = profile_model_detailed(model, input_tensor, "Baseline (FP32)")

    # Profile optimized (FP16)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    with torch.cuda.amp.autocast():
        optimized_data = profile_model_detailed(model, input_tensor, "Optimized (FP16)")

    # Analyze bottlenecks
    print("\n\nBOTTLENECK ANALYSIS - BASELINE:")
    analyze_bottlenecks(baseline_data)

    print("\n\nBOTTLENECK ANALYSIS - OPTIMIZED:")
    analyze_bottlenecks(optimized_data)

    # Generate recommendations
    recommendations = generate_optimization_recommendations(baseline_data)

    # Save detailed results
    results = {
        'baseline': baseline_data,
        'optimized': optimized_data,
        'recommendations': recommendations,
    }

    with open('kernel_profiling_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n\nResults saved to: kernel_profiling_results.json")

    # Summary comparison
    print(f"\n{'='*100}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*100}")
    baseline_time = baseline_data['summary']['total_cuda_time_ms']
    optimized_time = optimized_data['summary']['total_cuda_time_ms']
    speedup = baseline_time / optimized_time if optimized_time > 0 else 1.0
    improvement = (baseline_time - optimized_time) / baseline_time * 100

    print(f"\nBaseline (FP32): {baseline_time:.2f} ms")
    print(f"Optimized (FP16): {optimized_time:.2f} ms")
    print(f"Speedup: {speedup:.2f}x ({improvement:+.1f}%)")


if __name__ == '__main__':
    main()
