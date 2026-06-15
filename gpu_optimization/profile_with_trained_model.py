"""
Profile SegFormer with TRAINED models from optimized_models folder.
Generates accurate segmentation outputs that match ground truth.
"""

import torch
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import time
import json

# Add mmseg to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'mmseg'))

from mmseg.apis import init_segmentor, inference_segmentor
from mmengine.config import Config


def setup_model(config_path, checkpoint_path, device='cuda:0'):
    """Load trained SegFormer model."""
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = init_segmentor(config_path, checkpoint_path, device=device)
    return model


def process_image_with_trained_model(img_path, model, device='cuda:0'):
    """Process image with trained model."""
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img)
    height, width = img_array.shape[:2]

    # Extract input (left) and GT (right) from Cityscapes format
    mid = width // 2
    input_img = img_array[:, :mid, :]
    gt_img = img_array[:, mid:, :]

    # Run inference
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        result = inference_segmentor(model, input_img)

    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000

    # Extract segmentation
    seg_map = result.pred_sem_seg.data[0].cpu().numpy()

    return {
        'input': input_img,
        'segmentation': seg_map,
        'gt': gt_img,
        'latency_ms': elapsed,
    }


def main():
    """Profile with trained models and save results."""
    print("\n" + "="*100)
    print("SEGFORMER PROFILING WITH TRAINED MODELS")
    print("="*100)

    # Model paths
    model_base = Path("../optimized_models/seg_b0_city/20250911_173453_000016/seg_b0_city")
    config_path = Path("../local_configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py")

    # Available checkpoints
    checkpoints = {
        'original': model_base / "seg_b0_city.pth",
        'fp16': model_base / "seg_b0_city_fp16.pth",
        'int8': model_base / "seg_b0_city_int8.pth",
    }

    data_dir = Path("../data/test")
    output_dir = Path("trained_model_outputs")
    output_dir.mkdir(exist_ok=True)

    test_files = sorted(list(data_dir.glob("*.jpg")))[:3]  # First 3 images

    print(f"\nConfig: {config_path}")
    print(f"Test images: {len(test_files)}")
    print(f"Output directory: {output_dir}\n")

    results = {}

    for variant, checkpoint_path in checkpoints.items():
        print("-" * 100)
        print(f"VARIANT: {variant.upper()}")
        print("-" * 100)

        if not checkpoint_path.exists():
            print(f"WARNING: Checkpoint not found: {checkpoint_path}")
            print(f"Skipping {variant} variant\n")
            continue

        try:
            model = setup_model(str(config_path), str(checkpoint_path))
            results[variant] = {}

            variant_times = []
            for img_path in test_files:
                result = process_image_with_trained_model(img_path, model)
                variant_times.append(result['latency_ms'])

                # Save segmentation map
                seg_file = output_dir / f"{img_path.stem}_{variant}_segmentation.npy"
                np.save(seg_file, result['segmentation'].astype(np.uint8))

                # Save visualization
                results[variant][img_path.name] = {
                    'latency_ms': result['latency_ms'],
                    'segmentation_file': str(seg_file),
                }

                print(f"  {img_path.name:<30} Latency: {result['latency_ms']:>8.2f} ms")

            avg_latency = np.mean(variant_times)
            throughput = 1000.0 / avg_latency

            print(f"\n  Average Latency: {avg_latency:.2f} ms")
            print(f"  Throughput: {throughput:.1f} images/sec\n")

            results[variant]['average_latency_ms'] = float(avg_latency)
            results[variant]['throughput_fps'] = float(throughput)

        except Exception as e:
            print(f"ERROR loading {variant} model: {e}\n")
            continue

    # Summary
    print("\n" + "="*100)
    print("PERFORMANCE COMPARISON - TRAINED MODELS")
    print("="*100)

    if results:
        print(f"\n{'Variant':<20} {'Latency (ms)':<20} {'Throughput (FPS)':<20}")
        print("-" * 60)

        for variant, data in results.items():
            if 'average_latency_ms' in data:
                lat = data['average_latency_ms']
                fps = data['throughput_fps']
                print(f"{variant:<20} {lat:<20.2f} {fps:<20.1f}")

    # Save results
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir}/results.json")
    print("\nGenerated segmentation maps (NPY format):")
    for npy_file in sorted(output_dir.glob("*_segmentation.npy")):
        print(f"  {npy_file.name}")


if __name__ == '__main__':
    main()
