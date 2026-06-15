import torch
import torch.nn as nn
import torch.profiler as profiler
import numpy as np
import time
from pathlib import Path

class SegFormerCNN(nn.Module):
    """Simple CNN baseline (not true SegFormer, but sufficient for optimization study)."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True),
        )
        self.decode_head = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 150, kernel_size=1),
        )

    def forward(self, x):
        x = self.stem(x)
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        out = self.decode_head(x1)
        return out

class FP16Mixed(nn.Module):
    """FP16 mixed precision with torch.amp."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        with torch.amp.autocast('cuda', dtype=torch.float16):
            return self.model(x)

class ChannelsLastOptimized(nn.Module):
    """Memory format optimization: NCHW -> NHWC."""
    def __init__(self, model):
        super().__init__()
        self.model = model.to(memory_format=torch.channels_last)

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last)
        return self.model(x)

class TF32Optimized(nn.Module):
    """Tensor Core TF32 precision."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def forward(self, x):
        return self.model(x)

def profile_model(model, num_iters=5, device='cuda'):
    """Profile model with PyTorch Profiler (simplified)."""
    model.eval()
    x = torch.randn(1, 3, 512, 512, device=device)

    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
        torch.cuda.synchronize()

        try:
            prof = profiler.profile(
                activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
                record_shapes=False
            )
            prof.__enter__()

            for _ in range(2):
                _ = model(x)

            prof.__exit__(None, None, None)

            kernel_stats = prof.key_averages()
            kernel_count = len(kernel_stats)
            cuda_time_ms = 0
            for evt in kernel_stats:
                if hasattr(evt, 'cuda_time_total'):
                    cuda_time_ms += evt.cuda_time_total
            cuda_time_ms /= 1000.0
        except Exception as e:
            kernel_count = 0
            cuda_time_ms = 0

    return {
        'total_cuda_time': cuda_time_ms,
        'kernel_count': kernel_count,
        'avg_kernel_time': cuda_time_ms / max(kernel_count, 1)
    }

def benchmark_latency(model, num_iters=50, device='cuda'):
    """Measure latency with GPU synchronization."""
    model.eval()
    x = torch.randn(1, 3, 512, 512, device=device)

    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
        torch.cuda.synchronize()

        times = []
        for _ in range(num_iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

    times = sorted(times)[5:-5]
    return {
        'latency_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
    }

def run_optimization_suite():
    """Run GPU optimization suite with profiling."""
    print("\nGPU-Optimized SegFormer: Profiling Suite")
    print("="*90)
    print(f"Device: CUDA")
    print(f"Input: (1, 3, 512, 512)")
    print()

    baseline_model = SegFormerCNN().cuda().eval()

    optimizations = [
        ("Baseline (FP32)", baseline_model),
        ("FP16 Mixed Precision (Tensor Cores)", FP16Mixed(baseline_model)),
        ("Channels-Last Format (cuDNN friendly)", ChannelsLastOptimized(baseline_model)),
        ("TF32 Precision (Tensor Cores)", TF32Optimized(baseline_model)),
    ]

    results = {}
    baseline_latency = None

    print(f"{'Optimization':<40} {'Latency (ms)':<15} {'Speedup':<12} {'Kernel Count':<15}")
    print("-"*90)

    for name, model in optimizations:
        try:
            lat = benchmark_latency(model)
            prof = profile_model(model)
            results[name] = (lat, prof)

            if baseline_latency is None:
                baseline_latency = lat['latency_ms']
                speedup = 1.0
            else:
                speedup = baseline_latency / lat['latency_ms']

            kernel_cnt = prof['kernel_count']
            print(f"{name:<40} {lat['latency_ms']:<15.2f} {speedup:<12.2f}x {kernel_cnt:<15}")

        except Exception as e:
            print(f"{name:<40} ERROR: {str(e)[:40]}")

    print("\n" + "="*90)
    print("Profile Analysis")
    print("="*90)

    for name, (lat, prof) in results.items():
        print(f"\n{name}")
        print(f"  Latency: {lat['latency_ms']:.2f}ms ±{lat['std_ms']:.2f}ms")
        print(f"  Range: {lat['min_ms']:.2f}ms - {lat['max_ms']:.2f}ms")
        print(f"  Kernels launched: {prof['kernel_count']}")
        print(f"  Avg kernel duration: {prof['avg_kernel_time']:.3f}ms")

    print("\n" + "="*90)
    print("GPU Optimization Techniques Demonstrated")
    print("="*90)
    print("""
1. FP16 Mixed Precision (Tensor Core dispatch)
   - Uses torch.amp.autocast('cuda')
   - Reduces memory bandwidth requirement by 2x
   - Tensor Cores accelerate FP16 GEMM operations
   - Expected: 1.3-1.6x speedup

2. Channels-Last Memory Format (cuDNN optimization)
   - Converts NCHW -> NHWC layout
   - Better memory coalescing on GPUs
   - cuDNN may select different (faster) kernels
   - Expected: 5-10% speedup

3. TF32 Precision (Tensor Core precision mode)
   - torch.backends.cuda.matmul.allow_tf32 = True
   - Enables 32-bit GEMM through Tensor Cores
   - Expected: 10-15% improvement for large GEMMs

4. Profiling Infrastructure
   - torch.profiler captures kernel-level timing
   - Tracks kernel count (proxy for launch overhead)
   - Measures actual CUDA time per kernel
   - Enables roofline model analysis
""")

    print("\n" + "="*90)
    print("Next Steps for Production GPU Optimization")
    print("="*90)
    print("""
Level 1: Current optimizations
  [+] FP16 mixed precision
  [+] Memory format tuning
  [+] Precision mode selection (TF32)

Level 2: Advanced (would require custom kernels)
  [ ] Kernel fusion (Conv+BN+ReLU)
  [ ] Operator fusion at graph level
  [ ] Custom CUDA kernels for bottlenecks

Level 3: Deployment
  [ ] TensorRT conversion for production
  [ ] INT8 quantization for edge
  [ ] Batch processing optimization
""")

if __name__ == '__main__':
    run_optimization_suite()
