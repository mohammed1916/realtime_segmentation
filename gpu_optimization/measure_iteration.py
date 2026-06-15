#!/usr/bin/env python3
"""
Measure iteration latency and memory for GPU optimization loop.
Simple, repeatable measurement for decision-making.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import time
import json
import argparse
from typing import Dict


class SegFormerB0(nn.Module):
    """SegFormer B0 architecture (simplified)."""
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


def measure_iteration(model: nn.Module, input_size: int = 512, runs: int = 20, warmup: int = 3) -> Dict:
    """Measure latency and memory for one iteration."""
    device = torch.device('cuda')
    model = model.to(device).eval()

    # Create dummy input
    x = torch.randn(1, 3, input_size, input_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Measure latency
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)  # ms

    # Memory
    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB

    times = np.array(times[3:])  # Skip warmup outliers

    return {
        'latency_ms': float(np.mean(times)),
        'latency_std_ms': float(np.std(times)),
        'latency_min_ms': float(np.min(times)),
        'latency_max_ms': float(np.max(times)),
        'peak_memory_mb': float(peak_memory),
        'runs': len(times),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='baseline', choices=['baseline', 'fp16', 'tf32', 'fp16_tf32'])
    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--input-size', type=int, default=512)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"MEASURING: {args.model.upper()}")
    print(f"{'='*80}\n")

    # Create model
    model = SegFormerB0()

    # Apply optimizations
    if args.model == 'tf32':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    elif args.model == 'fp16':
        # No model modification, will use autocast during forward
        def forward_fp16(self, x):
            with torch.amp.autocast('cuda'):
                return SegFormerB0.forward(self, x)
        model.forward = forward_fp16.__get__(model, SegFormerB0)

    elif args.model == 'fp16_tf32':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        def forward_fp16(self, x):
            with torch.amp.autocast('cuda'):
                return SegFormerB0.forward(self, x)
        model.forward = forward_fp16.__get__(model, SegFormerB0)

    # Measure
    print(f"Running {args.runs} iterations (warmup: 3)...")
    results = measure_iteration(model, input_size=args.input_size, runs=args.runs)

    # Print results
    print(f"\nResults:")
    print(f"  Latency:     {results['latency_ms']:.2f} ± {results['latency_std_ms']:.2f} ms")
    print(f"  Min/Max:     {results['latency_min_ms']:.2f} / {results['latency_max_ms']:.2f} ms")
    print(f"  Memory:      {results['peak_memory_mb']:.1f} MB")
    print(f"  Runs:        {results['runs']}")

    # Save if output specified
    if args.output:
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nSaved to: {output_file}")

    return results


if __name__ == '__main__':
    main()
