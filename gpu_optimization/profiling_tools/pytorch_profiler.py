"""
PyTorch Profiler for SegFormer - Identify GPU bottlenecks.

Usage:
    python pytorch_profiler.py --model b0 --input-size 512 --batch-size 1
    python pytorch_profiler.py --model b1 --input-size 1024 --batch-size 4
"""

import argparse
import json
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mmseg.models import build_segmentor
from mmengine.config import Config


def load_segformer_model(model_name: str):
    """Load SegFormer model from config."""
    config_path = f"configs/_base_/models/segformer_mit-{model_name}.py"

    try:
        cfg = Config.fromfile(config_path)
        model = build_segmentor(cfg.model)
        model.cuda().eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting fallback...")
        # Fallback: create minimal model structure for testing
        return None


def profile_segformer(model, input_tensor: torch.Tensor, export_path: str = None):
    """Run PyTorch profiler on SegFormer inference."""

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
        on_trace_ready=lambda p: export_trace(p, export_path) if export_path else None,
    ) as prof:
        with record_function("inference"):
            with torch.no_grad():
                _ = model(input_tensor)

    return prof


def export_trace(prof, path: str):
    """Export profiler trace to JSON (Chrome trace format)."""
    prof.export_chrome_trace(path)
    print(f"✓ Chrome trace exported to {path}")


def print_profiler_summary(prof):
    """Print profiler summary in human-readable format."""
    print("\n" + "="*80)
    print("PROFILER SUMMARY")
    print("="*80)

    # Table format
    print("\nKernel Execution Times (CUDA):")
    print("-" * 80)
    table = prof.key_averages(group_by_stack_n=5)

    # Filter to CUDA ops only
    cuda_ops = [(name, metrics) for name, metrics in table.items()
                if metrics.self_cuda_time_total > 0]

    # Sort by CUDA time
    cuda_ops.sort(key=lambda x: x[1].self_cuda_time_total, reverse=True)

    # Print top ops
    print(f"{'Name':<50} {'CUDA (ms)':<12} {'CPU (ms)':<12} {'Count':<8}")
    print("-" * 80)

    total_cuda_time = sum(m[1].self_cuda_time_total for m in cuda_ops) / 1000  # Convert to ms

    for name, metrics in cuda_ops[:30]:  # Top 30
        cuda_ms = metrics.self_cuda_time_total / 1000
        cpu_ms = metrics.self_cpu_time_total / 1000
        count = metrics.count

        if cuda_ms > 0.01:  # Only show ops > 0.01ms
            print(f"{name:<50} {cuda_ms:<12.3f} {cpu_ms:<12.3f} {count:<8}")

    print("-" * 80)
    print(f"{'Total CUDA Time':<50} {total_cuda_time:<12.3f} ms")
    print()

    # Memory stats
    print("Memory Statistics:")
    print("-" * 80)
    try:
        mem_stats = prof.key_averages()
        for idx, (name, metrics) in enumerate(mem_stats.items()):
            if metrics.cpu_memory_usage > 0 or metrics.cuda_memory_usage > 0:
                if idx < 10:  # First 10 memory-using ops
                    cpu_mem_mb = metrics.cpu_memory_usage / 1e6
                    cuda_mem_mb = metrics.cuda_memory_usage / 1e6
                    if cpu_mem_mb > 0 or cuda_mem_mb > 0:
                        print(f"{name:<50} CPU: {cpu_mem_mb:>8.2f} MB  CUDA: {cuda_mem_mb:>8.2f} MB")
    except:
        pass


def generate_csv_report(prof, output_file: str):
    """Generate CSV report of profiler results."""
    import csv

    table = prof.key_averages()

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Operation', 'CUDA Time (ms)', 'CPU Time (ms)', 'Count',
                        'Avg CUDA (ms)', 'Avg CPU (ms)'])

        for name, metrics in table.items():
            if metrics.self_cuda_time_total > 0:
                cuda_ms = metrics.self_cuda_time_total / 1000
                cpu_ms = metrics.self_cpu_time_total / 1000
                count = metrics.count

                writer.writerow([
                    name,
                    f"{cuda_ms:.3f}",
                    f"{cpu_ms:.3f}",
                    count,
                    f"{cuda_ms/count:.3f}" if count > 0 else 0,
                    f"{cpu_ms/count:.3f}" if count > 0 else 0,
                ])

    print(f"✓ CSV report exported to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Profile SegFormer with PyTorch Profiler")
    parser.add_argument('--model', type=str, default='b0',
                       choices=['b0', 'b1', 'b2', 'b3', 'b4', 'b5'],
                       help='SegFormer model variant')
    parser.add_argument('--input-size', type=int, default=512,
                       help='Input image size (square)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size')
    parser.add_argument('--export-trace', type=str, default=None,
                       help='Export chrome trace to file')
    parser.add_argument('--export-csv', type=str, default=None,
                       help='Export CSV report to file')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    parser.add_argument('--num-iterations', type=int, default=1,
                       help='Number of profiling iterations (averaged)')

    args = parser.parse_args()

    print(f"SegFormer PyTorch Profiler")
    print(f"=" * 50)
    print(f"Model: SegFormer-{args.model.upper()}")
    print(f"Input: {args.batch_size}x3x{args.input_size}x{args.input_size}")
    print(f"Device: {args.device}")
    print(f"Iterations: {args.num_iterations}")

    # Check device
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"=" * 50)

    # Try to load model
    model = load_segformer_model(args.model)
    if model is None:
        print("Using dummy model for demonstration...")
        # Create dummy model that mimics SegFormer structure
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 7, 4, 3),
            torch.nn.LayerNorm([64, args.input_size//4, args.input_size//4]),
            torch.nn.Conv2d(64, 256, 3, 1, 1),
            torch.nn.Conv2d(256, 150, 1, 1, 0),  # 150 classes
        ).cuda()

    # Create input tensor
    x = torch.randn(args.batch_size, 3, args.input_size, args.input_size,
                   device=args.device)

    # Warmup
    print("\nWarming up...")
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
    torch.cuda.synchronize()

    # Profile
    print(f"\nProfiling ({args.num_iterations} iterations)...")
    prof = profile_segformer(model, x, export_path=args.export_trace)

    # Print summary
    print_profiler_summary(prof)

    # Export CSV if requested
    if args.export_csv:
        generate_csv_report(prof, args.export_csv)

    # Suggestions
    print("\n" + "="*80)
    print("PROFILING TIPS")
    print("="*80)
    print("""
1. IDENTIFY BOTTLENECKS:
   - Look for kernels with longest self_cuda_time_total
   - Check if they are Attention (GELU, softmax, matmul) or FFN ops

2. MEMORY HIERARCHY:
   - High CUDA time + moderate CPU time = GPU-bound (good scaling)
   - Kernel count varies = CPU scheduling overhead (bad)

3. NEXT STEPS:
   - Use --export-trace to generate Chrome timeline
   - Open trace in chrome://tracing for visual inspection
   - Use Nsight Systems for detailed profiling (kernel by kernel)
   - Use Nsight Compute for register/occupancy analysis

4. SPECIFIC BOTTLENECKS TO LOOK FOR:
   - aten::_scaled_dot_product_attention (low TFLOP/s)
   - aten::matmul (check matrix shapes)
   - aten::conv2d with small kernels (bandwidth-bound)
   - aten::layer_norm (fast, should be <1ms per call)
    """)


if __name__ == '__main__':
    main()
