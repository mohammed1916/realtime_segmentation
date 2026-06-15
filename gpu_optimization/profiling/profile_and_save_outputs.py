"""
Profile SegFormer AND save segmentation outputs.
Generates both performance metrics and segmentation visualizations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import numpy as np
import time
from matplotlib import pyplot as plt
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


def process_image(img_path, model, device, use_fp16=False):
    """Process single image and return input, output, latency."""
    img = Image.open(img_path)
    img_array = np.array(img)
    height, width = img_array.shape[:2]

    # Extract input (left half) and ground truth (right half)
    mid = width // 2
    input_img = img_array[:, :mid, :]
    gt_img = img_array[:, mid:, :]

    # Preprocess input
    input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).float() / 255.0
    input_resized = F.interpolate(
        input_tensor.unsqueeze(0),
        size=(512, 512),
        mode='bilinear',
        align_corners=False
    ).to(device)

    # Inference
    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.perf_counter()

        if use_fp16:
            with torch.amp.autocast('cuda'):
                output = model(input_resized)
        else:
            output = model(input_resized)

        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000

    # Get segmentation map (argmax of channels)
    segmentation = output.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    # Resize to original input size
    segmentation_resized = np.array(Image.fromarray(segmentation).resize(
        (input_img.shape[1], input_img.shape[0]), Image.NEAREST
    ))

    return {
        'input': input_img,
        'output': segmentation_resized,
        'gt': gt_img,
        'latency_ms': elapsed,
    }


def visualize_result(input_img, output, gt, output_path):
    """Create visualization with input, prediction, and ground truth."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Input
    axes[0].imshow(input_img)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    # Prediction
    axes[1].imshow(output, cmap='jet')
    axes[1].set_title('Segmentation Prediction')
    axes[1].axis('off')

    # Ground Truth (visualized as segmentation map)
    axes[2].imshow(gt)
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def main():
    """Profile and save segmentation outputs."""
    print("\n" + "="*100)
    print("SEGFORMER PROFILING WITH OUTPUT VISUALIZATION")
    print("="*100)

    # Setup
    device = torch.device('cuda')
    model = SegFormerB0().to(device).eval()
    data_dir = Path("../data/test")
    output_dir = Path("segmentation_outputs")
    output_dir.mkdir(exist_ok=True)

    # Get test images
    test_files = sorted(list(data_dir.glob("*.jpg")))[:5]  # First 5 images
    print(f"\nProcessing {len(test_files)} test images")
    print(f"Output directory: {output_dir}\n")

    results = {
        'fp32': {},
        'fp16': {},
    }

    # FP32 Processing
    print("-" * 100)
    print("FP32 BASELINE PROCESSING")
    print("-" * 100)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False

    fp32_times = []
    for img_path in test_files:
        result = process_image(img_path, model, device, use_fp16=False)
        fp32_times.append(result['latency_ms'])

        # Save visualization
        output_file = output_dir / f"{img_path.stem}_fp32_comparison.png"
        visualize_result(result['input'], result['output'], result['gt'], output_file)

        # Save segmentation map
        seg_file = output_dir / f"{img_path.stem}_fp32_segmentation.npy"
        np.save(seg_file, result['output'])

        results['fp32'][img_path.name] = {
            'latency_ms': result['latency_ms'],
            'visualization': str(output_file),
            'segmentation': str(seg_file),
        }

        print(f"{img_path.name:<30} Latency: {result['latency_ms']:>8.2f} ms")

    avg_fp32 = np.mean(fp32_times)
    print(f"\nFP32 Average Latency: {avg_fp32:.2f} ms")

    # FP16 Processing
    print("\n" + "-" * 100)
    print("FP16 OPTIMIZED PROCESSING")
    print("-" * 100)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True

    fp16_times = []
    for img_path in test_files:
        result = process_image(img_path, model, device, use_fp16=True)
        fp16_times.append(result['latency_ms'])

        # Save visualization
        output_file = output_dir / f"{img_path.stem}_fp16_comparison.png"
        visualize_result(result['input'], result['output'], result['gt'], output_file)

        # Save segmentation map
        seg_file = output_dir / f"{img_path.stem}_fp16_segmentation.npy"
        np.save(seg_file, result['output'])

        results['fp16'][img_path.name] = {
            'latency_ms': result['latency_ms'],
            'visualization': str(output_file),
            'segmentation': str(seg_file),
        }

        print(f"{img_path.name:<30} Latency: {result['latency_ms']:>8.2f} ms")

    avg_fp16 = np.mean(fp16_times)
    speedup = avg_fp32 / avg_fp16

    print(f"\nFP16 Average Latency: {avg_fp16:.2f} ms")

    # Summary
    print("\n" + "="*100)
    print("PERFORMANCE SUMMARY")
    print("="*100)
    print(f"\nFP32 Average:  {avg_fp32:.2f} ms")
    print(f"FP16 Average:  {avg_fp16:.2f} ms")
    print(f"Speedup:       {speedup:.2f}x ({(speedup-1)*100:.1f}% improvement)")
    print(f"\nProcessed:     {len(test_files)} images")
    print(f"Outputs:       {output_dir}/")

    # Save results
    results['summary'] = {
        'num_images': len(test_files),
        'fp32_avg_ms': float(avg_fp32),
        'fp16_avg_ms': float(avg_fp16),
        'speedup': float(speedup),
        'output_directory': str(output_dir),
    }

    with open(output_dir / 'profiling_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir}/profiling_results.json")

    # List output files
    print("\n" + "-" * 100)
    print("OUTPUT FILES GENERATED")
    print("-" * 100)

    png_files = list(output_dir.glob("*.png"))
    npy_files = list(output_dir.glob("*.npy"))

    print(f"\nVisualizations (PNG): {len(png_files)} files")
    for f in sorted(png_files)[:10]:
        print(f"  {f.name}")
    if len(png_files) > 10:
        print(f"  ... and {len(png_files) - 10} more")

    print(f"\nSegmentation Maps (NPY): {len(npy_files)} files")
    for f in sorted(npy_files)[:10]:
        print(f"  {f.name}")
    if len(npy_files) > 10:
        print(f"  ... and {len(npy_files) - 10} more")

    print(f"\nJSON Results: profiling_results.json")

    print("\n" + "="*100)
    print("VISUALIZATION GUIDE")
    print("="*100)
    print("""
Each image generates 2 visualizations:
  1. *_fp32_comparison.png - Side-by-side: Input | FP32 Prediction | Ground Truth
  2. *_fp16_comparison.png - Side-by-side: Input | FP16 Prediction | Ground Truth

Segmentation maps saved as numpy arrays:
  - *_fp32_segmentation.npy
  - *_fp16_segmentation.npy

To compare outputs:
  import numpy as np
  fp32 = np.load('segmentation_outputs/1_fp32_segmentation.npy')
  fp16 = np.load('segmentation_outputs/1_fp16_segmentation.npy')
  difference = np.sum(fp32 != fp16)  # How many pixels differ
  """)


if __name__ == '__main__':
    main()
