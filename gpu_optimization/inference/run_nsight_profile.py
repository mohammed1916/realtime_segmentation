"""
Profile SegFormer with Nsight Compute to capture real L2 cache, occupancy,
bandwidth, and warp efficiency metrics.

Usage:
  ncu -o profile_baseline.ncu-rep python run_nsight_profile.py
  ncu -o profile_fp16.ncu-rep python run_nsight_profile.py --fp16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import numpy as np
import argparse


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


def load_test_image():
    """Load real Cityscapes test image."""
    test_files = sorted(list(Path("../data/test").glob("*.jpg")))
    img_path = test_files[0]

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
    return input_tensor


def main(use_fp16=False):
    """Profile SegFormer with Nsight Compute."""
    print("\n" + "="*80)
    print("SegFormer Profiling with Nsight Compute")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Precision: {'FP16 (torch.amp.autocast)' if use_fp16 else 'FP32 (default)'}")
    print(f"  Model: SegFormer B0")
    print(f"  Input: Real Cityscapes test image (512×512)")
    print(f"\nInstructions:")
    print(f"  ncu -o profile_baseline.ncu-rep python run_nsight_profile.py")
    print(f"  ncu -o profile_fp16.ncu-rep python run_nsight_profile.py --fp16")
    print(f"\nThen view results:")
    print(f"  ncu-ui profile_baseline.ncu-rep")
    print(f"  ncu-ui profile_fp16.ncu-rep")
    print(f"\nMetrics to inspect in GUI:")
    print(f"  1. Memory Workload > L2 Cache (Hit Rate, Bandwidth)")
    print(f"  2. Occupancy > SM Occupancy (Register Pressure)")
    print(f"  3. Execution > Warp State (Stall Reasons)")
    print(f"  4. Memory Workload > Memory Access Pattern (Coalescing)")
    print("\n" + "="*80 + "\n")

    # Configure CUDA libs
    if use_fp16:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False

    # Load model and data
    model = SegFormerB0().cuda().eval()
    input_tensor = load_test_image().unsqueeze(0).cuda()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Device: {input_tensor.device}")

    # Warmup
    print(f"\nWarming up...")
    with torch.no_grad():
        for _ in range(3):
            if use_fp16:
                with torch.amp.autocast('cuda'):
                    _ = model(input_tensor)
            else:
                _ = model(input_tensor)

    torch.cuda.synchronize()
    print(f"Warmup complete. Ready for profiling.\n")

    # Main inference (will be profiled by Nsight Compute)
    print(f"Running inference (Nsight Compute is profiling)...")

    with torch.no_grad():
        if use_fp16:
            with torch.amp.autocast('cuda'):
                output = model(input_tensor)
        else:
            output = model(input_tensor)

    torch.cuda.synchronize()
    print(f"Inference complete.")
    print(f"Output shape: {output.shape}")
    print(f"\nProfile saved. View with: ncu-ui profile_*.ncu-rep")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Use FP16 mixed precision')
    args = parser.parse_args()

    main(use_fp16=args.fp16)
