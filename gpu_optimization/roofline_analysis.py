import torch
import time
import numpy as np

class RooflineAnalysis:
    """Analyze operations against GPU roofline model."""

    def __init__(self, gpu_name="RTX 4060"):
        self.gpu_name = gpu_name
        # RTX 4060 Laptop specs
        self.peak_fp32_tflops = 15.1  # TFLOP/s
        self.peak_bw_gbs = 288  # GB/s (GDDR6)
        self.roofline_knee = self.peak_fp32_tflops / self.peak_bw_gbs  # ops/byte

    def measure_matmul(self, m, n, k, iterations=100):
        """Measure dense matrix multiplication."""
        a = torch.randn(m, k, device='cuda', dtype=torch.float32)
        b = torch.randn(k, n, device='cuda', dtype=torch.float32)

        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(iterations):
            c = torch.matmul(a, b)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        flops = 2 * m * n * k * iterations
        bytes_transferred = (m * k + k * n + m * n) * 4

        tflops = flops / elapsed / 1e12
        bw = bytes_transferred / elapsed / 1e9
        ai = flops / bytes_transferred

        return {
            'tflops': tflops,
            'bw_gbs': bw,
            'ai': ai,
            'peak_util': (tflops / self.peak_fp32_tflops) * 100,
        }

    def measure_conv1x1(self, batch, cin, cout, h, w, iterations=50):
        """Measure 1x1 convolution."""
        x = torch.randn(batch, cin, h, w, device='cuda', dtype=torch.float32)
        conv = torch.nn.Conv2d(cin, cout, kernel_size=1).cuda()

        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            for _ in range(iterations):
                y = conv(x)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        flops = 2 * batch * cin * cout * h * w * iterations
        bytes_accessed = (batch * cin * h * w + batch * cout * h * w + cin * cout) * 4

        tflops = flops / elapsed / 1e12
        bw = bytes_accessed / elapsed / 1e9
        ai = flops / bytes_accessed

        return {
            'tflops': tflops,
            'bw_gbs': bw,
            'ai': ai,
            'peak_util': (tflops / self.peak_fp32_tflops) * 100,
        }

    def print_roofline_summary(self):
        """Print roofline metrics."""
        print("Roofline Analysis")
        print("="*70)
        print(f"GPU: {self.gpu_name}")
        print(f"Peak FP32 TFLOP/s: {self.peak_fp32_tflops}")
        print(f"Peak Bandwidth: {self.peak_bw_gbs} GB/s")
        print(f"Roofline Knee: {self.roofline_knee:.4f} ops/byte")
        print()

        print("Dense MatMul (4096x4096 @ 4096):")
        result = self.measure_matmul(4096, 4096, 4096)
        print(f"  TFLOP/s: {result['tflops']:.2f} ({result['peak_util']:.1f}% of peak)")
        print(f"  Bandwidth: {result['bw_gbs']:.0f} GB/s")
        print(f"  Arithmetic Intensity: {result['ai']:.2f} ops/byte")
        print(f"  Classification: {'COMPUTE-BOUND' if result['ai'] > self.roofline_knee else 'MEMORY-BOUND'}")
        print()

        print("Conv1x1 (1x256x32x32):")
        result = self.measure_conv1x1(1, 256, 256, 32, 32)
        print(f"  TFLOP/s: {result['tflops']:.2f} ({result['peak_util']:.1f}% of peak)")
        print(f"  Bandwidth: {result['bw_gbs']:.0f} GB/s")
        print(f"  Arithmetic Intensity: {result['ai']:.2f} ops/byte")
        print(f"  Classification: {'COMPUTE-BOUND' if result['ai'] > self.roofline_knee else 'MEMORY-BOUND'}")
        print()

        print("Memory Copy (bandwidth peak):")
        x = torch.randn(100000000, device='cuda', dtype=torch.float32)
        torch.cuda.synchronize()
        start = time.perf_counter()
        y = x.clone()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        bw = (x.numel() * 4 * 2) / elapsed / 1e9  # read + write
        print(f"  Bandwidth: {bw:.0f} GB/s ({bw / self.peak_bw_gbs * 100:.1f}% of peak)")
        print()

if __name__ == '__main__':
    roofline = RooflineAnalysis()
    roofline.print_roofline_summary()
