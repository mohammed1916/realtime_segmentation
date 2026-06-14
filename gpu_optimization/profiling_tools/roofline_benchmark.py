"""
Roofline Model Analysis for SegFormer Operations.

Measures arithmetic intensity and compares to GPU roofline to identify bottlenecks.

Reference: "Roofline: An Insightful Visual Performance Model for Floating-Point Programs"
(Williams, et al., 2009)

Usage:
    python roofline_benchmark.py --operation attention --device cuda:0
    python roofline_benchmark.py --operation ffn --device cuda:0
    python roofline_benchmark.py --all --device cuda:0
"""

import argparse
import torch
import time
from typing import Tuple
import json
from pathlib import Path


class RooflineAnalyzer:
    """Analyze operations on the roofline model."""

    # GPU Specs (RTX 3090)
    PEAK_FLOPS = 10.0  # TFLOP/s (single precision)
    PEAK_BW = 912.0    # GB/s (HBM bandwidth)
    COMPUTE_ROOF = PEAK_FLOPS * 1e12  # FLOP/s
    MEMORY_ROOF = PEAK_BW * 1e9  # bytes/s

    # GPU Specs (RTX 4090) - Uncomment to use
    # PEAK_FLOPS = 16.2  # TFLOP/s
    # PEAK_BW = 1008.0   # GB/s
    # COMPUTE_ROOF = PEAK_FLOPS * 1e12
    # MEMORY_ROOF = PEAK_BW * 1e9

    # GPU Specs (A100) - Uncomment to use
    # PEAK_FLOPS = 19.5  # TFLOP/s
    # PEAK_BW = 2039.0   # GB/s (HBM)
    # COMPUTE_ROOF = PEAK_FLOPS * 1e12
    # MEMORY_ROOF = PEAK_BW * 1e9

    def __init__(self, device: str = 'cuda:0'):
        self.device = device
        self.gpu_name = torch.cuda.get_device_name(device)

    def measure_matmul(self, M: int, N: int, K: int, num_iterations: int = 100) -> dict:
        """Measure dense matrix multiplication performance."""
        A = torch.randn(M, K, device=self.device, dtype=torch.float32)
        B = torch.randn(K, N, device=self.device, dtype=torch.float32)

        # Warmup
        for _ in range(10):
            C = torch.matmul(A, B)
        torch.cuda.synchronize()

        # Measure
        torch.cuda.reset_peak_memory_stats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_iterations):
            C = torch.matmul(A, B)
        end_event.record()
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event) / num_iterations
        elapsed_s = elapsed_ms / 1000

        # Compute metrics
        flops = 2 * M * N * K  # Standard dense matmul
        achieved_tflops = (flops / elapsed_s) / 1e12

        # Memory transfer
        bytes_transferred = (M * K + K * N + M * N) * 4  # float32 = 4 bytes
        achieved_bw = (bytes_transferred / elapsed_s) / 1e9

        # Arithmetic intensity
        ai = flops / bytes_transferred

        # Roofline classification
        flops_per_byte_roofline = self.PEAK_FLOPS / self.PEAK_BW
        is_compute_bound = ai > flops_per_byte_roofline
        bottleneck = "COMPUTE" if is_compute_bound else "MEMORY"

        return {
            'shape': (M, N, K),
            'flops': flops,
            'elapsed_ms': elapsed_ms,
            'achieved_tflops': achieved_tflops,
            'peak_tflops': self.PEAK_FLOPS,
            'utilization_percent': (achieved_tflops / self.PEAK_FLOPS) * 100,
            'bytes_transferred': bytes_transferred,
            'achieved_bandwidth_gb_s': achieved_bw,
            'peak_bandwidth_gb_s': self.PEAK_BW,
            'bw_utilization_percent': (achieved_bw / self.PEAK_BW) * 100,
            'arithmetic_intensity': ai,
            'bottleneck': bottleneck,
        }

    def benchmark_attention(self, seq_len: int, head_dim: int, num_heads: int,
                           sr_ratio: int = 1) -> dict:
        """Benchmark attention operation (Q @ K^T then attn @ V)."""

        # Shapes
        kv_seq_len = seq_len // sr_ratio
        B = 1

        # Create tensors
        Q = torch.randn(B, seq_len, head_dim, device=self.device, dtype=torch.float32)
        K = torch.randn(B, kv_seq_len, head_dim, device=self.device, dtype=torch.float32)
        V = torch.randn(B, kv_seq_len, head_dim, device=self.device, dtype=torch.float32)

        # Warmup
        for _ in range(5):
            S = torch.matmul(Q, K.transpose(-1, -2))
            attn_out = torch.matmul(S, V)
        torch.cuda.synchronize()

        # Measure
        num_iterations = 50
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_iterations):
            # MatMul 1: Q @ K^T
            S = torch.matmul(Q, K.transpose(-1, -2))  # (B, seq, kv_seq)

            # MatMul 2: attention @ V
            attn_out = torch.matmul(S, V)  # (B, seq, head_dim)
        end_event.record()
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event) / num_iterations
        elapsed_s = elapsed_ms / 1000

        # FLOPs: two matmuls
        flops_qk = 2 * B * seq_len * head_dim * kv_seq_len  # Q @ K^T
        flops_attn = 2 * B * seq_len * kv_seq_len * head_dim  # attn @ V
        total_flops = flops_qk + flops_attn

        # Memory: Read Q, K, V; Write S, output
        bytes_read = (seq_len * head_dim + kv_seq_len * head_dim + kv_seq_len * head_dim) * 4
        bytes_written = (seq_len * kv_seq_len + seq_len * head_dim) * 4  # S and output
        bytes_total = bytes_read + bytes_written

        achieved_tflops = (total_flops / elapsed_s) / 1e12
        achieved_bw = (bytes_total / elapsed_s) / 1e9
        ai = total_flops / bytes_total

        is_compute_bound = ai > (self.PEAK_FLOPS / self.PEAK_BW)
        bottleneck = "COMPUTE" if is_compute_bound else "MEMORY"

        return {
            'operation': 'attention',
            'shapes': {
                'Q': (B, seq_len, head_dim),
                'K': (B, kv_seq_len, head_dim),
                'V': (B, kv_seq_len, head_dim),
            },
            'sr_ratio': sr_ratio,
            'total_flops': total_flops,
            'elapsed_ms': elapsed_ms,
            'achieved_tflops': achieved_tflops,
            'peak_tflops': self.PEAK_FLOPS,
            'utilization_percent': (achieved_tflops / self.PEAK_FLOPS) * 100,
            'bytes_transferred': bytes_total,
            'achieved_bandwidth_gb_s': achieved_bw,
            'peak_bandwidth_gb_s': self.PEAK_BW,
            'bw_utilization_percent': (achieved_bw / self.PEAK_BW) * 100,
            'arithmetic_intensity': ai,
            'bottleneck': bottleneck,
        }

    def benchmark_conv1x1(self, B: int, C_in: int, C_out: int,
                         H: int, W: int) -> dict:
        """Benchmark 1x1 convolution (used in FFN and fusion layers)."""

        x = torch.randn(B, C_in, H, W, device=self.device, dtype=torch.float32)
        conv = torch.nn.Conv2d(C_in, C_out, kernel_size=1, stride=1).cuda()

        # Warmup
        for _ in range(5):
            y = conv(x)
        torch.cuda.synchronize()

        # Measure
        num_iterations = 50
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_iterations):
            y = conv(x)
        end_event.record()
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event) / num_iterations
        elapsed_s = elapsed_ms / 1000

        # FLOPs: (C_in * 1 * 1) * C_out * H * W * 2 (for multiply-add)
        flops = 2 * C_in * C_out * H * W

        # Memory
        bytes_input = B * C_in * H * W * 4
        bytes_output = B * C_out * H * W * 4
        bytes_weights = C_in * C_out * 4
        bytes_total = bytes_input + bytes_output + bytes_weights

        achieved_tflops = (flops / elapsed_s) / 1e12
        achieved_bw = (bytes_total / elapsed_s) / 1e9
        ai = flops / bytes_total

        is_compute_bound = ai > (self.PEAK_FLOPS / self.PEAK_BW)
        bottleneck = "COMPUTE" if is_compute_bound else "MEMORY"

        return {
            'operation': 'conv1x1',
            'shape': (B, C_in, C_out, H, W),
            'total_flops': flops,
            'elapsed_ms': elapsed_ms,
            'achieved_tflops': achieved_tflops,
            'peak_tflops': self.PEAK_FLOPS,
            'utilization_percent': (achieved_tflops / self.PEAK_FLOPS) * 100,
            'bytes_transferred': bytes_total,
            'achieved_bandwidth_gb_s': achieved_bw,
            'peak_bandwidth_gb_s': self.PEAK_BW,
            'bw_utilization_percent': (achieved_bw / self.PEAK_BW) * 100,
            'arithmetic_intensity': ai,
            'bottleneck': bottleneck,
        }

    def print_result(self, result: dict):
        """Print a single benchmark result."""
        print(f"\nOperation: {result.get('operation', 'matmul').upper()}")
        print("-" * 70)

        if 'shape' in result:
            print(f"Shape: {result['shape']}")
        elif 'shapes' in result:
            for name, shape in result['shapes'].items():
                print(f"  {name}: {shape}")
            if 'sr_ratio' in result:
                print(f"  sr_ratio: {result['sr_ratio']}")

        print(f"Time:        {result['elapsed_ms']:.3f} ms")
        print(f"FLOPs:       {result['total_flops']/1e9:.2f} GFLOP")
        print()
        print(f"Compute:")
        print(f"  Achieved:  {result['achieved_tflops']:.2f} TFLOP/s ({result['utilization_percent']:.1f}% of peak)")
        print(f"  Peak:      {result['peak_tflops']:.2f} TFLOP/s")
        print()
        print(f"Memory:")
        print(f"  Achieved:  {result['achieved_bandwidth_gb_s']:.0f} GB/s ({result['bw_utilization_percent']:.1f}% of peak)")
        print(f"  Peak:      {result['peak_bandwidth_gb_s']:.0f} GB/s")
        print()
        print(f"Arithmetic Intensity: {result['arithmetic_intensity']:.3f} ops/byte")
        print(f"Roofline Limit: {self.PEAK_FLOPS / self.PEAK_BW:.3f} ops/byte")
        print(f"Bottleneck: {result['bottleneck']}")

        # Interpretation
        if result['bottleneck'] == 'MEMORY':
            ceiling = result['arithmetic_intensity'] * self.PEAK_BW / 1e12
            print(f"  (Ceiling: {ceiling:.2f} TFLOP/s due to memory bandwidth)")
        else:
            print(f"  (Can only improve with more parallelism or computation)")


def main():
    parser = argparse.ArgumentParser(description="Roofline Analysis for SegFormer")
    parser.add_argument('--operation', type=str, default='attention',
                       choices=['attention', 'conv1x1', 'matmul', 'ffn'],
                       help='Operation to benchmark')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='CUDA device')
    parser.add_argument('--all', action='store_true',
                       help='Run all benchmarks')
    parser.add_argument('--export', type=str, default=None,
                       help='Export results to JSON')

    args = parser.parse_args()

    analyzer = RooflineAnalyzer(args.device)

    print("="*70)
    print(f"ROOFLINE ANALYSIS - {analyzer.gpu_name}")
    print("="*70)
    print(f"Peak Compute: {analyzer.PEAK_FLOPS:.1f} TFLOP/s")
    print(f"Peak Memory:  {analyzer.PEAK_BW:.0f} GB/s")
    print(f"Roofline Limit: {analyzer.PEAK_FLOPS / analyzer.PEAK_BW:.3f} ops/byte")
    print()

    results = []

    # Attention benchmarks
    if args.operation == 'attention' or args.all:
        print("ATTENTION OPERATIONS")
        print("="*70)

        # Stage 1
        result = analyzer.benchmark_attention(seq_len=128*128, head_dim=64, num_heads=1, sr_ratio=8)
        analyzer.print_result(result)
        results.append(result)

        # Stage 2
        result = analyzer.benchmark_attention(seq_len=64*64, head_dim=128, num_heads=2, sr_ratio=4)
        analyzer.print_result(result)
        results.append(result)

        # Stage 3
        result = analyzer.benchmark_attention(seq_len=32*32, head_dim=64, num_heads=5, sr_ratio=2)
        analyzer.print_result(result)
        results.append(result)

        # Stage 4
        result = analyzer.benchmark_attention(seq_len=16*16, head_dim=64, num_heads=8, sr_ratio=1)
        analyzer.print_result(result)
        results.append(result)

    # Conv1x1 benchmarks
    if args.operation == 'conv1x1' or args.all:
        print("\n1×1 CONVOLUTION OPERATIONS (MixFFN)")
        print("="*70)

        # Typical FFN: C→4C and 4C→C
        result = analyzer.benchmark_conv1x1(B=1, C_in=64, C_out=256, H=128, W=128)
        analyzer.print_result(result)
        results.append(result)

        result = analyzer.benchmark_conv1x1(B=1, C_in=128, C_out=512, H=64, W=64)
        analyzer.print_result(result)
        results.append(result)

    # Dense matmul benchmarks
    if args.operation == 'matmul' or args.all:
        print("\nDENSE MATRIX MULTIPLICATION")
        print("="*70)

        # Various sizes to show roofline boundary
        sizes = [
            (512, 512, 512),    # Square, small
            (1024, 1024, 1024), # Square, medium
            (4096, 4096, 1024), # Tall, large
        ]

        for m, n, k in sizes:
            result = analyzer.measure_matmul(m, n, k)
            analyzer.print_result(result)
            results.append(result)

    # Export results
    if args.export:
        output_path = Path(args.export)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results exported to {output_path}")


if __name__ == '__main__':
    main()
