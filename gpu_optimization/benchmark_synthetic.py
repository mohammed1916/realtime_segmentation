import torch
import torch.nn as nn
import time
import numpy as np

class SimpleSegFormer(nn.Module):
    """Simplified SegFormer-like model for benchmarking."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.stage1 = self._make_stage(64, 64, num_blocks=2)
        self.stage2 = self._make_stage(64, 128, num_blocks=2, stride=2)
        self.stage3 = self._make_stage(128, 256, num_blocks=2, stride=2)
        self.stage4 = self._make_stage(256, 512, num_blocks=2, stride=2)

        self.decode_head = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1),
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 150, kernel_size=1),
        )

    def _make_stage(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_blocks - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x = self.decode_head(x1)
        return x

def benchmark_model(model, input_shape=(1, 3, 512, 512), num_iterations=50, device='cuda'):
    """Benchmark model inference."""
    model.eval()
    model.to(device)

    x = torch.randn(*input_shape, device=device)

    print("Warming up...")
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
        torch.cuda.synchronize()

    times = []
    print(f"Benchmarking ({num_iterations} iterations)...")
    with torch.no_grad():
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

    times_sorted = sorted(times)[5:-5]
    avg_ms = np.mean(times_sorted)
    min_ms = np.min(times_sorted)
    max_ms = np.max(times_sorted)
    std_ms = np.std(times_sorted)
    throughput = 1000 / avg_ms

    return {
        'avg_ms': avg_ms,
        'min_ms': min_ms,
        'max_ms': max_ms,
        'std_ms': std_ms,
        'throughput': throughput,
        'times': times_sorted,
    }

def profile_model(model, x, device='cuda'):
    """Profile model with PyTorch profiler."""
    from torch.profiler import profile, record_function, ProfilerActivity

    model.eval()
    model.to(device)

    with torch.no_grad():
        for _ in range(2):
            _ = model(x)
        torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False, profile_memory=False) as prof:
        with record_function("inference"):
            with torch.no_grad():
                for _ in range(5):
                    _ = model(x)

    return prof

def print_profile(prof):
    """Print profiler results."""
    try:
        print("Profiler data collected successfully.")
    except Exception as e:
        print(f"Profiler error: {e}")

if __name__ == '__main__':
    print("BASELINE BENCHMARK - Synthetic SegFormer")
    print("="*70)

    model = SimpleSegFormer()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    result = benchmark_model(model, num_iterations=50)

    print(f"\nLatency (512x512 input, batch=1):")
    print(f"  Average: {result['avg_ms']:.2f} ms")
    print(f"  Min/Max: {result['min_ms']:.2f} / {result['max_ms']:.2f} ms")
    print(f"  Std Dev: {result['std_ms']:.2f} ms")
    print(f"  Throughput: {result['throughput']:.1f} img/sec")

    x = torch.randn(1, 3, 512, 512, device='cuda')
    prof = profile_model(model, x)
    print("\nProfiler Results (Top Operations):")
    print_profile(prof)
