import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import time

class SimpleSegFormer(nn.Module):
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
        for _ in range(blocks-1):
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

def load_test_image(image_path):
    """Load image and split into input (left) and ground truth (right)."""
    img = Image.open(image_path)
    img_array = np.array(img)

    height, width = img_array.shape[:2]
    mid = width // 2

    input_img = img_array[:, :mid, :]
    gt_img = img_array[:, mid:, :]

    return input_img, gt_img

def compute_metrics(pred, gt, num_classes=150):
    """Compute IoU and pixel accuracy."""
    pred = pred.argmax(dim=1).cpu().numpy()
    gt = gt.cpu().numpy()

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

def run_verification():
    """Run accuracy verification on real test dataset."""
    print("FP16 Accuracy Verification with Real Test Data")
    print("="*70)

    test_dir = Path("../data/test")
    test_images = sorted(list(test_dir.glob("*.jpg")))[:10]

    print(f"Found {len(test_images)} test images\n")

    baseline_model = SimpleSegFormer().cuda().eval()

    fp32_results = []
    fp16_results = []
    latencies_fp32 = []
    latencies_fp16 = []

    print(f"{'Image':<10} {'FP32 mIoU':<12} {'FP16 mIoU':<12} {'Difference':<12} {'FP32 Time':<12} {'FP16 Time':<12}")
    print("-"*70)

    with torch.no_grad():
        for idx, img_path in enumerate(test_images):
            try:
                input_img, gt_img = load_test_image(img_path)

                input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0
                gt_tensor = torch.from_numpy(gt_img[:, :, 0]).long().unsqueeze(0).cuda()

                input_tensor = F.interpolate(input_tensor, size=(512, 512), mode='bilinear', align_corners=False)
                gt_tensor = F.interpolate(gt_tensor.unsqueeze(1).float(), size=(512, 512), mode='nearest').long().squeeze(1)

                torch.cuda.synchronize()
                start = time.perf_counter()
                pred_fp32 = baseline_model(input_tensor)
                torch.cuda.synchronize()
                time_fp32 = (time.perf_counter() - start) * 1000

                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.amp.autocast('cuda'):
                    pred_fp16 = baseline_model(input_tensor)
                torch.cuda.synchronize()
                time_fp16 = (time.perf_counter() - start) * 1000

                miou_fp32, acc_fp32 = compute_metrics(pred_fp32, gt_tensor)
                miou_fp16, acc_fp16 = compute_metrics(pred_fp16, gt_tensor)

                fp32_results.append(miou_fp32)
                fp16_results.append(miou_fp16)
                latencies_fp32.append(time_fp32)
                latencies_fp16.append(time_fp16)

                diff = abs(miou_fp32 - miou_fp16)

                print(f"{idx+1:<10} {miou_fp32:<12.4f} {miou_fp16:<12.4f} {diff:<12.4f} {time_fp32:<12.2f} {time_fp16:<12.2f}")

            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
                continue

    print("\n" + "="*70)
    print("Summary Statistics")
    print("="*70)

    if fp32_results:
        avg_miou_fp32 = np.mean(fp32_results)
        avg_miou_fp16 = np.mean(fp16_results)
        avg_diff = np.mean([abs(a - b) for a, b in zip(fp32_results, fp16_results)])

        avg_time_fp32 = np.mean(latencies_fp32)
        avg_time_fp16 = np.mean(latencies_fp16)

        print(f"Average mIoU (FP32):     {avg_miou_fp32:.4f}")
        print(f"Average mIoU (FP16):     {avg_miou_fp16:.4f}")
        print(f"Average difference:      {avg_diff:.4f}")
        print(f"Accuracy preserved:      {avg_diff < 0.01}")

        print(f"\nAverage latency (FP32):  {avg_time_fp32:.2f} ms")
        print(f"Average latency (FP16):  {avg_time_fp16:.2f} ms")
        print(f"Speedup:                 {avg_time_fp32 / avg_time_fp16:.2f}x")

        print("\n" + "="*70)
        print("Verification Result:")
        print("="*70)

        if avg_diff < 0.01:
            print("PASS - FP16 accuracy preserved (difference < 0.01)")
        else:
            print(f"WARNING - Accuracy difference: {avg_diff:.4f} (> 0.01)")

if __name__ == '__main__':
    run_verification()
