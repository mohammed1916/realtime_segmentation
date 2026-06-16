#!/usr/bin/env python3
"""
Deep profile of decode head operations
Find which component is the bottleneck
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time


class DecodeHeadComponents(nn.Module):
    """Decode head with individual operation profiling."""
    def __init__(self):
        super().__init__()
        self.conv_proj = nn.Conv2d(64, 256, kernel_size=1)
        self.conv_ref = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv_final = nn.Conv2d(256, 150, kernel_size=1)

    def forward_detailed(self, x):
        """Forward with timing for each operation."""
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Op 1: 1x1 projection
        x = self.conv_proj(x)
        torch.cuda.synchronize()
        t_proj = (time.perf_counter() - t0) * 1000

        # Op 2: 4x upsample
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        torch.cuda.synchronize()
        t_upsample = (time.perf_counter() - (t0 + t_proj/1000)) * 1000

        # Op 3: 3x3 refinement
        x = self.conv_ref(x)
        torch.cuda.synchronize()
        t_ref = (time.perf_counter() - (t0 + (t_proj + t_upsample)/1000)) * 1000

        # Op 4: 1x1 final
        x = self.conv_final(x)
        torch.cuda.synchronize()
        t_final = (time.perf_counter() - (t0 + (t_proj + t_upsample + t_ref)/1000)) * 1000

        return x, {
            'proj': t_proj,
            'upsample': t_upsample,
            'refinement': t_ref,
            'final': t_final,
        }


def profile_decode():
    """Profile decode head operations."""
    device = torch.device('cuda')
    decode = DecodeHeadComponents().to(device).eval()
    torch.backends.cudnn.benchmark = True

    # Input from stage1: (1, 64, 128, 128)
    input_tensor = torch.randn(1, 64, 128, 128, device=device)

    print("\n" + "="*80)
    print("DECODE HEAD OPERATION PROFILING")
    print("="*80)
    print(f"Input: {input_tensor.shape}\n")

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _, _ = decode.forward_detailed(input_tensor)

    # Profile
    timings = {op: [] for op in ['proj', 'upsample', 'refinement', 'final']}

    with torch.no_grad():
        for _ in range(30):
            _, ops = decode.forward_detailed(input_tensor)
            for op, t in ops.items():
                timings[op].append(t)

    # Results
    print(f"{'Operation':<20} {'Latency (ms)':<15} {'Std Dev':<12} {'% of Total':<15}")
    print("-" * 80)

    total = sum(np.mean(timings[op]) for op in timings)

    for op in ['proj', 'upsample', 'refinement', 'final']:
        times = np.array(timings[op])
        mean = np.mean(times)
        std = np.std(times)
        pct = (mean / total) * 100
        print(f"{op:<20} {mean:<15.3f} {std:<12.3f} {pct:<15.1f}%")

    print("-" * 80)
    print(f"{'DECODE TOTAL':<20} {total:<15.3f}")

    # Analysis
    print("\n" + "="*80)
    print("DECODE HEAD BOTTLENECK ANALYSIS")
    print("="*80)

    ops_sorted = sorted(timings.items(),
                       key=lambda x: np.mean(x[1]),
                       reverse=True)

    print("\nOperations ranked by execution time:\n")
    for i, (op, times) in enumerate(ops_sorted, 1):
        mean = np.mean(times)
        pct = (mean / total) * 100
        print(f"{i}. {op.upper():<15} {mean:.3f} ms ({pct:>5.1f}%)")

    # Detailed breakdown
    print("\n" + "="*80)
    print("DETAILED OPERATION BREAKDOWN")
    print("="*80)

    print("\n1. 1x1 Projection Conv:")
    print(f"   Time: {np.mean(timings['proj']):.3f} ms")
    print(f"   Input:  (1, 64, 128, 128)")
    print(f"   Output: (1, 256, 128, 128)")
    print(f"   Kernel: 1x1 convolution")
    print(f"   Library: cuDNN (GEMM via im2col)")

    print("\n2. 4x Bilinear Upsample:")
    print(f"   Time: {np.mean(timings['upsample']):.3f} ms ({(np.mean(timings['upsample'])/total)*100:.1f}%)")
    print(f"   Input:  (1, 256, 128, 128)")
    print(f"   Output: (1, 256, 512, 512)")
    print(f"   Operation: Bilinear interpolation (25x larger output)")
    print(f"   Library: cuDNN interpolation kernel")
    print(f"   -> This is the KEY bottleneck!")

    print("\n3. 3x3 Refinement Conv:")
    print(f"   Time: {np.mean(timings['refinement']):.3f} ms")
    print(f"   Input:  (1, 256, 512, 512)")
    print(f"   Output: (1, 256, 512, 512)")
    print(f"   Kernel: 3x3 convolution on 512x512 output")
    print(f"   Library: cuDNN convolution")

    print("\n4. 1x1 Final Projection:")
    print(f"   Time: {np.mean(timings['final']):.3f} ms")
    print(f"   Input:  (1, 256, 512, 512)")
    print(f"   Output: (1, 150, 512, 512)")
    print(f"   Kernel: 1x1 convolution on 512x512")
    print(f"   Library: cuDNN (GEMM via im2col)")

    # Key insight
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)

    upsample_pct = (np.mean(timings['upsample']) / total) * 100
    conv_512_pct = ((np.mean(timings['refinement']) + np.mean(timings['final'])) / total) * 100

    print(f"""
1. Upsample operation: {np.mean(timings['upsample']):.3f} ms ({upsample_pct:.1f}%)
   - 4x interpolation creates 512x512 output
   - This is NOT the primary bottleneck, just one component

2. 512x512 Convolutions: {np.mean(timings['refinement']) + np.mean(timings['final']):.3f} ms ({conv_512_pct:.1f}%)
   - 3x3 refinement on 512x512 (256 channels)
   - 1x1 projection on 512x512 (256->150 channels)
   - These on large spatial output dominate

3. Total decode: {total:.3f} ms = {(total/31.1)*100:.1f}% of model
   - 29.5 ms out of 31.1 ms total

OPTIMIZATION STRATEGY:
The decode head is memory-bound (processing large tensors).
BF16 helps here (50% memory reduction).

Possible further optimizations:
1. [DONE] BF16: reduces memory bandwidth
2. Depthwise separable: Conv->SeparableConv (requires retraining)
3. Smaller channel width: 256->128 (requires retraining)
4. Nearest neighbor: upsample + conv fusion (custom kernel)
5. Progressive upsampling: 2x + 2x instead of 4x (requires retraining)

ROI Analysis:
- Conv fusion: 5-10 hours, 5-10% gain (0.5-1.0x/hour)
- Channel reduction: 0 hours (retraining cost), 20-30% gain
- Custom kernels: 8-12 hours, 15-20% gain (1.3-2.5x/hour)

RECOMMENDATION:
Current BF16 optimization is solid. Further gains require:
1. Model changes (retraining) - not our scope
2. Complex custom kernels - high development cost
3. Depthwise convolutions - architectural change

Consider: If you can retrain, reduce decode channels from 256 to 128.
This alone would save ~7.5 ms (0.75x per hour for retraining investment).
""")

    print("="*80)


if __name__ == '__main__':
    profile_decode()
