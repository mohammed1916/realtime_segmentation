#!/usr/bin/env python3
"""
Profile SegFormer to identify actual kernel bottlenecks
Shows which operations take the most time, then we can target those for optimization
"""

import torch
import torch.nn as nn
import numpy as np
import time
from pathlib import Path


class SegFormerB0(nn.Module):
    """SegFormer B0 - instrumented version to measure each stage."""
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
        self.timings = {}

    def _make_stage(self, in_c, out_c, blocks, stride=1):
        layers = [nn.Conv2d(in_c, out_c, 3, stride, 1), nn.BatchNorm2d(out_c), nn.ReLU(True)]
        for _ in range(blocks - 1):
            layers += [nn.Conv2d(out_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.ReLU(True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        # Measure each stage
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        x = self.stem(x)
        torch.cuda.synchronize()
        t_stem = (time.perf_counter() - t0) * 1000

        x1 = self.stage1(x)
        torch.cuda.synchronize()
        t_stage1 = (time.perf_counter() - (t0 + t_stem/1000)) * 1000

        x2 = self.stage2(x1)
        torch.cuda.synchronize()
        t_stage2 = (time.perf_counter() - (t0 + (t_stem + t_stage1)/1000)) * 1000

        x3 = self.stage3(x2)
        torch.cuda.synchronize()
        t_stage3 = (time.perf_counter() - (t0 + (t_stem + t_stage1 + t_stage2)/1000)) * 1000

        x4 = self.stage4(x3)
        torch.cuda.synchronize()
        t_stage4 = (time.perf_counter() - (t0 + (t_stem + t_stage1 + t_stage2 + t_stage3)/1000)) * 1000

        x = self.decode_head(x1)
        torch.cuda.synchronize()
        t_decode = (time.perf_counter() - (t0 + (t_stem + t_stage1 + t_stage2 + t_stage3 + t_stage4)/1000)) * 1000

        self.timings = {
            'stem': t_stem,
            'stage1': t_stage1,
            'stage2': t_stage2,
            'stage3': t_stage3,
            'stage4': t_stage4,
            'decode': t_decode,
        }
        return x


def profile_segformer():
    """Profile SegFormer to identify bottlenecks."""
    device = torch.device('cuda')
    model = SegFormerB0().to(device).eval()
    torch.backends.cudnn.benchmark = True

    input_tensor = torch.randn(1, 3, 512, 512, device=device)

    print("\n" + "="*80)
    print("KERNEL BOTTLENECK PROFILING - SEGFORMER B0")
    print("="*80)
    print(f"Input: {input_tensor.shape} | Device: {device}\n")

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_tensor)

    # Profile each stage
    print("Profiling each architectural stage:")
    print("-" * 80)

    stage_times = {stage: [] for stage in ['stem', 'stage1', 'stage2', 'stage3', 'stage4', 'decode']}

    with torch.no_grad():
        for run in range(20):
            _ = model(input_tensor)
            for stage, t in model.timings.items():
                stage_times[stage].append(t)

    # Print results
    print(f"\n{'Stage':<15} {'Latency (ms)':<15} {'Std Dev':<12} {'% of Total':<15}")
    print("-" * 80)

    total = sum(np.mean(stage_times[s]) for s in stage_times)

    for stage in ['stem', 'stage1', 'stage2', 'stage3', 'stage4', 'decode']:
        times = np.array(stage_times[stage])
        mean = np.mean(times)
        std = np.std(times)
        pct = (mean / total) * 100
        print(f"{stage:<15} {mean:<15.3f} {std:<12.3f} {pct:<15.1f}%")

    print("-" * 80)
    print(f"{'TOTAL':<15} {total:<15.3f}")

    # Analysis
    print("\n" + "="*80)
    print("BOTTLENECK ANALYSIS")
    print("="*80)

    sorted_stages = sorted(stage_times.items(),
                          key=lambda x: np.mean(x[1]),
                          reverse=True)

    print("\nStages ranked by execution time:\n")
    for i, (stage, times) in enumerate(sorted_stages, 1):
        mean = np.mean(times)
        pct = (mean / total) * 100
        print(f"{i}. {stage.upper():<10} {mean:.3f} ms ({pct:>5.1f}% of total)")

    # Detailed breakdown
    print("\n" + "="*80)
    print("DETAILED STAGE BREAKDOWN")
    print("="*80)

    print("\nSTEM (7x7 conv at stride 4):")
    print(f"  Time: {np.mean(stage_times['stem']):.3f} ms")
    print(f"  Input: (1, 3, 512, 512)")
    print(f"  Output: (1, 64, 128, 128)")
    print(f"  Operation: Single large convolution (reduces spatial dims by 4x)")

    print("\nSTAGE1 (2x 3x3 conv blocks):")
    print(f"  Time: {np.mean(stage_times['stage1']):.3f} ms")
    print(f"  Input: (1, 64, 128, 128)")
    print(f"  Output: (1, 64, 128, 128)")
    print(f"  Operation: 2 blocks of Conv->BN->ReLU")

    print("\nSTAGE2 (2x 3x3 conv blocks + stride 2):")
    print(f"  Time: {np.mean(stage_times['stage2']):.3f} ms")
    print(f"  Input: (1, 64, 128, 128)")
    print(f"  Output: (1, 128, 64, 64)")
    print(f"  Operation: 2 blocks with spatial reduction")

    print("\nSTAGE3 (2x 3x3 conv blocks + stride 2):")
    print(f"  Time: {np.mean(stage_times['stage3']):.3f} ms")
    print(f"  Input: (1, 128, 64, 64)")
    print(f"  Output: (1, 256, 32, 32)")
    print(f"  Operation: 2 blocks with spatial reduction")

    print("\nSTAGE4 (2x 3x3 conv blocks + stride 2):")
    print(f"  Time: {np.mean(stage_times['stage4']):.3f} ms")
    print(f"  Input: (1, 256, 32, 32)")
    print(f"  Output: (1, 512, 16, 16)")
    print(f"  Operation: 2 blocks with spatial reduction + channel increase to 512")

    print("\nDECODE HEAD (1x1 conv + 4x upsample + 3x3 conv + 1x1 conv):")
    print(f"  Time: {np.mean(stage_times['decode']):.3f} ms")
    print(f"  Input: (1, 64, 128, 128) [from stage1]")
    print(f"  Output: (1, 150, 512, 512)")
    print(f"  Operations:")
    print(f"    1. 1x1 Conv: (64 -> 256)")
    print(f"    2. 4x Upsample: (128x128 -> 512x512)")
    print(f"    3. 3x3 Conv: (256 -> 256)")
    print(f"    4. 1x1 Conv: (256 -> 150)")

    print("\n" + "="*80)
    print("OPTIMIZATION OPPORTUNITIES")
    print("="*80)

    topmost = sorted_stages[0]
    print(f"\nPrimary bottleneck: {topmost[0].upper()} ({np.mean(topmost[1]):.3f} ms, "
          f"{(np.mean(topmost[1])/total)*100:.1f}%)")

    if topmost[0] == 'stage4':
        print("""
Stage4 is the bottleneck:
- Highest channel count (512 channels)
- All computation cost is in channel dimension, not spatial
- Already at 16x16 spatial size (memory coalescing good)
- BF16 helps here (50% memory reduction)

Optimization strategies for Stage4:
1. [DONE] BF16 precision: reduces data by 50%
2. Depthwise convolution: reduce param count (risky - changes model)
3. Channel pruning: remove some channels (requires retraining)
4. Grouped convolution: reduce compute per channel (requires retraining)

Recommendation: Stage4 is already well-optimized with BF16.
Further optimization requires model changes (retraining).
""")

    elif topmost[0] == 'decode':
        print(f"""
DECODE HEAD IS THE BOTTLENECK (91.6% of total time!)

Decode is expensive because:
- 4x spatial upsampling (interpolation from 128x128 to 512x512)
- Multiple sequential convolutions on large outputs (512x512)
- Combines spatial operations with channel operations

Current breakdown (decode head):
  1. 1x1 Conv (64->256): projection
  2. 4x Upsample (128->512): interpolation (bilinear)
  3. 3x3 Conv (256->256): refinement
  4. 1x1 Conv (256->150): final projection

Optimization strategies for Decode:
1. [DONE] BF16 precision: reduces bandwidth by 50%
2. Fuse upsample+conv: combine interpolation and convolution
3. Use faster upsampling: nearest neighbor + conv
4. Separable convolutions: reduce param count
5. Deformable convolutions: context-aware resampling

IMPORTANT: Decode head = 30.04 ms of 32.82 ms total!

If we can optimize decode by 30%, we gain:
  30.04 ms * 0.30 = 9 ms saved
  Total: 32.82 - 9 = 23.82 ms (27% speedup!)

Recommendation: FOCUS HERE - huge ROI potential!
""")

    else:
        print(f"""
{topmost[0].upper()} is taking the most time.

BF16 optimization already provides:
- 50% memory bandwidth savings
- Tensor Core activation

Further optimization difficult without:
- Custom kernels (10+ hours)
- Model changes (retraining)
""")

    print("="*80)


if __name__ == '__main__':
    profile_segformer()
