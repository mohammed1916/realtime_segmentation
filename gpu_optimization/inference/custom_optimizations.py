import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from benchmark_synthetic import SimpleSegFormer, benchmark_model

class OptimizedConv2d(nn.Module):
    """Conv2d with fused activation."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class OptimizedSegFormerV1(SimpleSegFormer):
    """SegFormer with fused Conv+ReLU operations."""
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.stage1 = self._make_optimized_stage(64, 64, num_blocks=2)
        self.stage2 = self._make_optimized_stage(64, 128, num_blocks=2, stride=2)
        self.stage3 = self._make_optimized_stage(128, 256, num_blocks=2, stride=2)
        self.stage4 = self._make_optimized_stage(256, 512, num_blocks=2, stride=2)

        self.decode_head = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 150, kernel_size=1),
        )

    def _make_optimized_stage(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_blocks - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

class OptimizedSegFormerV2(SimpleSegFormer):
    """SegFormer with reduced precision computation."""
    def __init__(self, use_fp16=True):
        super().__init__()
        self.use_fp16 = use_fp16

    def forward(self, x):
        if self.use_fp16:
            with torch.cuda.amp.autocast():
                x = self.stem(x)
                x1 = self.stage1(x)
                x2 = self.stage2(x1)
                x3 = self.stage3(x2)
                x4 = self.stage4(x3)
                x = self.decode_head(x1)
        else:
            x = self.stem(x)
            x1 = self.stage1(x)
            x2 = self.stage2(x1)
            x3 = self.stage3(x2)
            x4 = self.stage4(x3)
            x = self.decode_head(x1)
        return x

def run_optimization():
    """Run baseline vs optimizations."""
    print("OPTIMIZATION EXPERIMENTS")
    print("="*70)

    print("\nV0: Baseline Model")
    baseline_model = SimpleSegFormer()
    baseline_result = benchmark_model(baseline_model, num_iterations=50)
    print(f"  Latency: {baseline_result['avg_ms']:.2f} ms")
    print(f"  Throughput: {baseline_result['throughput']:.1f} img/sec")

    print("\nV1: Optimized Architecture (Fused activations)")
    v1_model = OptimizedSegFormerV1()
    v1_result = benchmark_model(v1_model, num_iterations=50)
    v1_improvement = (baseline_result['avg_ms'] - v1_result['avg_ms']) / baseline_result['avg_ms'] * 100
    print(f"  Latency: {v1_result['avg_ms']:.2f} ms ({v1_improvement:+.2f}%)")
    print(f"  Throughput: {v1_result['throughput']:.1f} img/sec")
    print(f"  Speedup: {baseline_result['avg_ms'] / v1_result['avg_ms']:.2f}x")

    print("\nV2: Mixed Precision (FP16 with autocast)")
    v2_model = OptimizedSegFormerV2(use_fp16=True).cuda()
    v2_result = benchmark_model(v2_model, num_iterations=50)
    v2_improvement = (baseline_result['avg_ms'] - v2_result['avg_ms']) / baseline_result['avg_ms'] * 100
    print(f"  Latency: {v2_result['avg_ms']:.2f} ms ({v2_improvement:+.2f}%)")
    print(f"  Throughput: {v2_result['throughput']:.1f} img/sec")
    print(f"  Speedup: {baseline_result['avg_ms'] / v2_result['avg_ms']:.2f}x")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    results = [
        ("Baseline", baseline_result['avg_ms'], baseline_result['throughput']),
        ("V1 (Architecture)", v1_result['avg_ms'], v1_result['throughput']),
        ("V2 (FP16)", v2_result['avg_ms'], v2_result['throughput']),
    ]

    print(f"\n{'Model':<25} {'Latency (ms)':<15} {'Throughput (img/s)':<20} {'Speedup':<10}")
    print("-"*70)

    for name, latency, throughput in results:
        speedup = baseline_result['avg_ms'] / latency
        print(f"{name:<25} {latency:<15.2f} {throughput:<20.1f} {speedup:<10.2f}x")

if __name__ == '__main__':
    run_optimization()
