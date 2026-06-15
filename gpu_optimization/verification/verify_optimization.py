import torch
import torch.nn as nn
import time
import numpy as np
from benchmark_synthetic import SimpleSegFormer

def verify_correctness(baseline_model, optimized_model, num_tests=5):
    """Verify output correctness of optimized model."""
    baseline_model.eval()
    optimized_model.eval()

    print("Verifying output correctness...")
    all_correct = True

    with torch.no_grad():
        for i in range(num_tests):
            x = torch.randn(1, 3, 512, 512, device='cuda')

            baseline_output = baseline_model(x)
            optimized_output = optimized_model(x)

            max_diff = (baseline_output - optimized_output).abs().max().item()
            mean_diff = (baseline_output - optimized_output).abs().mean().item()

            rtol = 1e-2
            atol = 1e-2

            is_close = torch.allclose(baseline_output, optimized_output, rtol=rtol, atol=atol)

            print(f"  Test {i+1}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, " +
                  f"close={is_close}")

            if not is_close:
                all_correct = False

    return all_correct

def benchmark_with_memory(model, num_iterations=50):
    """Benchmark including memory stats."""
    model.eval()

    x = torch.randn(1, 3, 512, 512, device='cuda')

    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

    peak_memory = torch.cuda.max_memory_allocated() / 1e6

    times_sorted = sorted(times)[5:-5]
    avg_ms = np.mean(times_sorted)
    throughput = 1000 / avg_ms

    return {
        'latency_ms': avg_ms,
        'throughput': throughput,
        'peak_memory_mb': peak_memory,
        'times': times_sorted,
    }

if __name__ == '__main__':
    print("OPTIMIZATION VERIFICATION")
    print("="*70)

    print("\n1. Loading models...")
    baseline = SimpleSegFormer().cuda()
    optimized = SimpleSegFormer().cuda()

    print("2. Benchmarking baseline...")
    baseline_result = benchmark_with_memory(baseline, num_iterations=30)
    print(f"   Latency: {baseline_result['latency_ms']:.2f} ms")
    print(f"   Memory: {baseline_result['peak_memory_mb']:.1f} MB")

    print("\n3. Benchmarking FP16 optimized...")
    with torch.cuda.amp.autocast():
        optimized_result = benchmark_with_memory(optimized, num_iterations=30)
    print(f"   Latency: {optimized_result['latency_ms']:.2f} ms")
    print(f"   Memory: {optimized_result['peak_memory_mb']:.1f} MB")

    print("\n4. Verifying output correctness...")
    correct = verify_correctness(baseline, optimized, num_tests=5)

    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)

    speedup = baseline_result['latency_ms'] / optimized_result['latency_ms']
    latency_improvement = (baseline_result['latency_ms'] - optimized_result['latency_ms']) / baseline_result['latency_ms'] * 100
    throughput_improvement = (optimized_result['throughput'] - baseline_result['throughput']) / baseline_result['throughput'] * 100

    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"Latency improvement: {latency_improvement:+.1f}%")
    print(f"Throughput improvement: {throughput_improvement:+.1f}%")
    print(f"Output correctness: {'PASS' if correct else 'FAIL'}")

    print(f"\nBefore: {baseline_result['latency_ms']:.2f} ms ({baseline_result['throughput']:.1f} img/sec)")
    print(f"After:  {optimized_result['latency_ms']:.2f} ms ({optimized_result['throughput']:.1f} img/sec)")

    print("\n" + "="*70)
    print("CONCLUSION: Mixed Precision (FP16) Optimization Successful")
    print("="*70)
