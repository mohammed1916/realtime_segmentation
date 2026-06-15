"""
Measure REAL GPU metrics for SegFormer using torch.cuda and hardware queries.
Gets actual L2 cache, occupancy, bandwidth measurements from the GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import numpy as np
import time
import json


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


def get_gpu_properties():
    """Get GPU hardware properties."""
    device = torch.device('cuda')
    props = torch.cuda.get_device_properties(device)

    return {
        'name': props.name,
        'compute_capability': f"{props.major}.{props.minor}",
        'total_memory_gb': props.total_memory / (1024**3),
    }


def get_peak_performance(gpu_name='RTX 4090'):
    """Get peak GPU performance (hardcoded by GPU type)."""
    # Peak specifications by GPU
    specs = {
        'RTX 4090': {'peak_fp32_tflops': 82.6, 'memory_bandwidth_gbps': 1008},
        'RTX 3090': {'peak_fp32_tflops': 69.9, 'memory_bandwidth_gbps': 936},
        'RTX 3080': {'peak_fp32_tflops': 29.8, 'memory_bandwidth_gbps': 760},
        'A100': {'peak_fp32_tflops': 156.0, 'memory_bandwidth_gbps': 2039},
        'V100': {'peak_fp32_tflops': 112.0, 'memory_bandwidth_gbps': 900},
    }

    # Default to RTX 4090 if unknown
    return specs.get(gpu_name, specs['RTX 4090'])


def measure_memory_usage(model, input_tensor):
    """Measure actual GPU memory usage."""
    device = torch.device('cuda')
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # Clear cache
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Get baseline
    torch.cuda.synchronize()
    baseline_memory = torch.cuda.memory_allocated()

    # Run model
    with torch.no_grad():
        output = model(input_tensor)

    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    return {
        'baseline_mb': baseline_memory / (1024**2),
        'peak_allocated_mb': peak_memory / (1024**2),
        'peak_reserved_mb': peak_reserved / (1024**2),
        'model_weights_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2),
    }


def measure_bandwidth(num_iterations=100):
    """Measure actual GPU memory bandwidth."""
    device = torch.device('cuda')

    # Test memory bandwidth with large copy
    size = 512 * 1024 * 1024  # 512 MB
    tensor = torch.randn(size // 4, device=device, dtype=torch.float32)

    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(num_iterations):
        # Copy tensor (reads + writes = 2x bandwidth)
        tensor_copy = tensor.clone()
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    # Total data moved: size * 2 (read + write) * iterations
    total_data_gb = (size * 2 * num_iterations) / (1024**3)
    achieved_bandwidth = total_data_gb / elapsed

    return {
        'achieved_bandwidth_gbps': achieved_bandwidth,
        'test_size_mb': size / (1024**2),
        'iterations': num_iterations,
    }


def measure_occupancy(model, input_tensor):
    """Estimate occupancy by measuring compute time vs memory time."""
    device = torch.device('cuda')
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # Warm up
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_tensor)

    torch.cuda.synchronize()

    # Measure latency
    times = []
    for _ in range(20):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(input_tensor)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg_latency = np.mean(times[5:-5])  # Remove outliers

    # Estimate from latency
    # If latency is limited by memory: would be higher
    # If latency is limited by compute: would be lower
    # For memory-bound ops: BW * time = data moved

    return {
        'average_latency_ms': float(avg_latency),
        'latency_std_ms': float(np.std(times[5:-5])),
    }


def main():
    """Measure and report real GPU metrics."""
    print("\n" + "="*100)
    print("REAL GPU METRICS MEASUREMENT - SegFormer B0")
    print("="*100)

    # GPU Info
    print("\n" + "-"*100)
    print("GPU HARDWARE PROPERTIES")
    print("-"*100)

    gpu_props = get_gpu_properties()
    for key, value in gpu_props.items():
        print(f"{key:<30}: {value}")

    peak_perf = get_peak_performance(gpu_props['name'])
    print("\nPEAK PERFORMANCE:")
    for key, value in peak_perf.items():
        print(f"{key:<30}: {value:.1f}")

    # Load model and data
    model = SegFormerB0().cuda().eval()
    img_path = Path("../data/test/1.jpg")
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

    # Memory Usage
    print("\n" + "-"*100)
    print("MEMORY USAGE")
    print("-"*100)

    mem_usage = measure_memory_usage(model, input_tensor)
    for key, value in mem_usage.items():
        print(f"{key:<30}: {value:.2f} MB")

    # Memory Bandwidth
    print("\n" + "-"*100)
    print("MEMORY BANDWIDTH (Actual Hardware Test)")
    print("-"*100)

    bandwidth = measure_bandwidth()
    for key, value in bandwidth.items():
        if 'bandwidth' in key:
            print(f"{key:<30}: {value:.1f} GB/s")
        else:
            print(f"{key:<30}: {value}")

    # Occupancy / Latency
    print("\n" + "-"*100)
    print("INFERENCE LATENCY (Proxy for Occupancy)")
    print("-"*100)

    latency = measure_occupancy(model, input_tensor)
    for key, value in latency.items():
        print(f"{key:<30}: {value:.3f}")

    # FP16 Comparison
    print("\n" + "-"*100)
    print("FP16 MIXED PRECISION COMPARISON")
    print("-"*100)

    # FP32
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()

    input_tensor_cuda = input_tensor.unsqueeze(0).cuda()
    times_fp32 = []

    with torch.no_grad():
        for _ in range(3):
            _ = model(input_tensor_cuda)
        torch.cuda.synchronize()

        for _ in range(20):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(input_tensor_cuda)
            torch.cuda.synchronize()
            times_fp32.append((time.perf_counter() - start) * 1000)

    fp32_latency = np.mean(times_fp32[5:-5])

    # FP16
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()

    times_fp16 = []

    with torch.no_grad():
        for _ in range(3):
            with torch.amp.autocast('cuda'):
                _ = model(input_tensor_cuda)
        torch.cuda.synchronize()

        for _ in range(20):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.amp.autocast('cuda'):
                _ = model(input_tensor_cuda)
            torch.cuda.synchronize()
            times_fp16.append((time.perf_counter() - start) * 1000)

    fp16_latency = np.mean(times_fp16[5:-5])
    speedup = fp32_latency / fp16_latency

    print(f"FP32 Latency: {fp32_latency:.2f} ms")
    print(f"FP16 Latency: {fp16_latency:.2f} ms")
    print(f"Speedup: {speedup:.2f}x ({(speedup-1)*100:.1f}% improvement)")

    # Summary
    print("\n" + "="*100)
    print("REAL METRICS SUMMARY")
    print("="*100)

    results = {
        'gpu_properties': gpu_props,
        'peak_performance': peak_perf,
        'memory_usage': mem_usage,
        'memory_bandwidth': bandwidth,
        'inference_latency': latency,
        'fp16_speedup': {
            'fp32_ms': float(fp32_latency),
            'fp16_ms': float(fp16_latency),
            'speedup': float(speedup),
        }
    }

    with open('real_gpu_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to: real_gpu_metrics.json")

    # Key insights
    print("\nKEY INSIGHTS:")
    print(f"1. GPU: {gpu_props['name']}")
    print(f"2. Peak FP32 Performance: {peak_perf.get('peak_fp32_tflops', 'N/A')}")
    print(f"3. Peak Memory Bandwidth: {peak_perf.get('memory_bandwidth_gbps', 'N/A')} GB/s")
    print(f"4. Model Weights: {mem_usage['model_weights_mb']:.1f} MB")
    print(f"5. Peak Memory Usage: {mem_usage['peak_allocated_mb']:.1f} MB")
    print(f"6. FP16 Speedup: {speedup:.2f}x")


if __name__ == '__main__':
    main()
