import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
from PIL import Image
import time
import torch
import torch.nn.functional as F

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

def run_onnx_verification():
    """Verify ONNX model with real test data."""
    print("SegFormer B0 ONNX Model Verification")
    print("="*70)

    model_path = Path("../demo/models/segformer.b0.1024x1024.city.160k_onnx.onnx").resolve()

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    print(f"Loading ONNX model: {model_path}\n")

    try:
        sess = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        print(f"Model loaded successfully")
        print(f"Inputs: {[inp.name for inp in sess.get_inputs()]}")
        print(f"Outputs: {[out.name for out in sess.get_outputs()]}\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    test_dir = Path("../data/test").resolve()
    test_images = sorted(list(test_dir.glob("*.jpg")))[:5]

    print(f"Found {len(test_images)} test images\n")
    print(f"{'Image':<10} {'mIoU':<12} {'Accuracy':<12} {'Latency (ms)':<15}")
    print("-"*70)

    latencies = []
    miou_results = []

    for idx, img_path in enumerate(test_images):
        try:
            input_img, gt_img = load_test_image(img_path)

            input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            gt_tensor = torch.from_numpy(gt_img[:, :, 0]).long()

            input_tensor = F.interpolate(input_tensor, size=(1024, 1024), mode='bilinear', align_corners=False)
            gt_tensor = F.interpolate(gt_tensor.unsqueeze(0).unsqueeze(0).float(), size=(256, 256), mode='nearest').long().squeeze()

            input_np = input_tensor.cpu().numpy().astype(np.float32)
            gt_np = gt_tensor.cpu().numpy()

            input_name = sess.get_inputs()[0].name

            start = time.perf_counter()
            outputs = sess.run(None, {input_name: input_np})
            elapsed = (time.perf_counter() - start) * 1000

            pred_output = outputs[0]

            miou, accuracy = compute_metrics(pred_output, gt_np)

            latencies.append(elapsed)
            miou_results.append(miou)

            print(f"{idx+1:<10} {miou:<12.4f} {accuracy:<12.4f} {elapsed:<15.2f}")

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            continue

    if latencies:
        print("\n" + "="*70)
        print("Summary")
        print("="*70)
        print(f"Average Latency:  {np.mean(latencies):.2f} ms")
        print(f"Min/Max Latency:  {np.min(latencies):.2f} / {np.max(latencies):.2f} ms")
        print(f"Average mIoU:      {np.mean(miou_results):.4f}")

        print("\n" + "="*70)
        print("Model Information")
        print("="*70)
        print(f"Model: SegFormer-B0 (Cityscapes, 1024x1024)")
        print(f"Framework: ONNX")
        print(f"Input shape: {sess.get_inputs()[0].shape}")
        print(f"Output shape: {sess.get_outputs()[0].shape}")

        print("\n" + "="*70)
        print("Optimization Opportunity")
        print("="*70)
        print("To optimize this ONNX model:")
        print("1. Convert ONNX -> PyTorch (onnx2torch)")
        print("2. Apply FP16 optimization: torch.amp.autocast('cuda')")
        print("3. Expected speedup: 1.5-2.0x")
        print("4. Or use TensorRT for ONNX direct optimization")

if __name__ == '__main__':
    run_onnx_verification()
