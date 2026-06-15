import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from PIL import Image

def load_test_image(image_path):
    """Load image and split into input (left) and ground truth (right)."""
    img = Image.open(image_path)
    img_array = np.array(img)

    height, width = img_array.shape[:2]
    mid = width // 2

    input_img = img_array[:, :mid, :]
    gt_img = img_array[:, mid:, :]

    return input_img, gt_img

def compute_metrics(pred, gt, num_classes=19):
    """Compute IoU and pixel accuracy."""
    pred = pred.argmax(axis=1)
    gt = gt

    iou_list = []
    pixel_correct = 0
    pixel_total = 0

    for class_id in range(num_classes):
        pred_mask = (pred == class_id)
        gt_mask = (gt == class_id)

        intersection = (pred_mask & gt_mask).sum()
        union = (pred_mask | gt_mask).sum()

        if union > 0:
            iou = intersection / union
            iou_list.append(iou)

        pixel_correct += intersection
        pixel_total += gt_mask.sum()

    miou = np.mean(iou_list) if iou_list else 0
    pixel_accuracy = pixel_correct / pixel_total if pixel_total > 0 else 0

    return miou, pixel_accuracy

def run_comparison():
    """Compare baseline ONNX vs optimized versions."""
    print("SegFormer B0 ONNX Model Optimization Comparison")
    print("="*70)

    model_path = Path("../demo/models/segformer.b0.1024x1024.city.160k_onnx.onnx").resolve()
    test_dir = Path("../data/test").resolve()
    test_images = sorted(list(test_dir.glob("*.jpg")))[:3]

    print(f"Model: {model_path.name}")
    print(f"Input shape: (1, 3, 1024, 1024)")
    print(f"Test images: {len(test_images)}\n")

    sess_fp32 = ort.InferenceSession(str(model_path), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    try:
        sess_fp16 = ort.InferenceSession(str(model_path),
                                        providers=['CUDAExecutionProvider'],
                                        sess_options=ort.SessionOptions())
        fp16_available = True
    except:
        fp16_available = False
        print("Note: FP16 not available via ONNX Runtime\n")

    print(f"{'Model':<30} {'Latency (ms)':<15} {'mIoU':<12} {'Speedup':<10}")
    print("-"*70)

    results = {}

    for model_name, sess in [("FP32 (Baseline)", sess_fp32)]:
        latencies = []
        miou_scores = []

        input_name = sess.get_inputs()[0].name

        for img_path in test_images:
            try:
                input_img, gt_img = load_test_image(img_path)

                input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                gt_tensor = torch.from_numpy(gt_img[:, :, 0]).long()

                input_tensor = F.interpolate(input_tensor, size=(1024, 1024), mode='bilinear', align_corners=False)
                gt_tensor = F.interpolate(gt_tensor.unsqueeze(0).unsqueeze(0).float(), size=(256, 256), mode='nearest').long().squeeze()

                input_np = input_tensor.cpu().numpy().astype(np.float32)
                gt_np = gt_tensor.cpu().numpy()

                start = time.perf_counter()
                outputs = sess.run(None, {input_name: input_np})
                elapsed = (time.perf_counter() - start) * 1000

                pred_output = outputs[0]
                miou, _ = compute_metrics(pred_output, gt_np)

                latencies.append(elapsed)
                miou_scores.append(miou)

            except Exception as e:
                print(f"Error: {e}")
                continue

        if latencies:
            avg_latency = np.mean(latencies)
            avg_miou = np.mean(miou_scores)
            results[model_name] = (avg_latency, avg_miou)

            print(f"{model_name:<30} {avg_latency:<15.2f} {avg_miou:<12.4f} {'1.00x':<10}")

    print("\n" + "="*70)
    print("Optimization Recommendations for ONNX")
    print("="*70)

    print("""
Option 1: TensorRT Optimization (Best performance)
   - Convert ONNX to TensorRT engine
   - FP16 precision: Expected 1.5-2.0x speedup
   - Installation: pip install tensorrt
   - Command: trtexec --onnx=model.onnx --fp16 --saveEngine=model.trt

Option 2: ONNX Runtime with GPU optimization
   - Current setup already uses GPU
   - Enable graph optimization with SessionOptions
   - Marginal improvements (<10%)

Option 3: PyTorch conversion + torch.amp
   - Convert ONNX -> PyTorch using onnx2torch
   - Apply torch.amp.autocast('cuda')
   - Expected speedup: 1.5-2.0x
   - Installation: pip install onnx2torch

Option 4: Quantization
   - INT8 quantization
   - Expected speedup: 2-4x
   - Trade: Slight accuracy loss (1-2%)
   - Installation: pip install onnx-simplifier
""")

    print("="*70)
    print("Current Performance (FP32 ONNX)")
    print("="*70)
    if results:
        for model_name, (lat, miou) in results.items():
            print(f"{model_name}: {lat:.2f}ms latency, mIoU={miou:.4f}")

if __name__ == '__main__':
    run_comparison()
