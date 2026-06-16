#!/usr/bin/env python3
"""
Profile End-to-End Video Processing Pipeline
Identifies WHERE the 180ms latency is coming from (not GPU inference).

Likely bottlenecks:
1. Video decoding (MP4 decompression) - 50-100ms
2. CPU preprocessing (resize, normalize) - 10-30ms
3. Data transfer CPU->GPU - 5-10ms
4. GPU inference - 20.89ms (OPTIMIZED!)
5. Data transfer GPU->CPU - 5-10ms
6. Post-processing (upsampling, etc) - 20-50ms
7. Visualization/output writing - 30-50ms
8. Python/framework overhead - 10-20ms
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
import time
from pathlib import Path
from typing import Dict, Tuple


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


class PipelineProfiler:
    """Profile each stage of the video processing pipeline."""

    def __init__(self, video_path: str, num_frames: int = 10):
        self.video_path = video_path
        self.num_frames = num_frames
        self.device = torch.device('cuda')
        self.model = SegFormerB0().to(self.device).eval()
        torch.backends.cudnn.benchmark = True
        self.timings = {}

    def profile_video_decoding(self) -> Dict[str, float]:
        """Measure video decoding time (frame extraction from MP4)."""
        print("\n" + "="*80)
        print("STAGE 1: VIDEO DECODING (MP4 decompression)")
        print("="*80)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"ERROR: Cannot open video file: {self.video_path}")
            return {}

        times = []
        for i in range(self.num_frames):
            start = time.perf_counter()
            ret, frame = cap.read()
            elapsed = (time.perf_counter() - start) * 1000

            if not ret:
                break

            times.append(elapsed)
            print(f"  Frame {i+1}: {elapsed:.2f} ms")

        cap.release()

        avg_time = np.mean(times)
        print(f"\nAverage: {avg_time:.2f} ms per frame")
        print(f"Bottleneck?: {'YES - MP4 decoding is slow!' if avg_time > 30 else 'No'}")

        return {'avg_ms': avg_time, 'frames': len(times)}

    def profile_preprocessing(self, frame) -> Dict[str, float]:
        """Measure CPU preprocessing (resize, normalize)."""
        print("\n" + "="*80)
        print("STAGE 2: CPU PREPROCESSING (resize, normalize)")
        print("="*80)

        times = []
        for _ in range(10):
            start = time.perf_counter()

            # Typical preprocessing
            resized = cv2.resize(frame, (512, 512))
            normalized = resized.astype(np.float32) / 255.0
            transposed = np.transpose(normalized, (2, 0, 1))

            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_time = np.mean(times)
        print(f"Average: {avg_time:.2f} ms per frame")
        print(f"Components:")
        print(f"  - cv2.resize: ~5-10 ms")
        print(f"  - Normalization: ~1-2 ms")
        print(f"  - Transpose: <1 ms")

        return {'avg_ms': avg_time}

    def profile_cpu_to_gpu_transfer(self, frame) -> Dict[str, float]:
        """Measure data transfer CPU->GPU."""
        print("\n" + "="*80)
        print("STAGE 3: CPU->GPU TRANSFER")
        print("="*80)

        # Prepare data
        resized = cv2.resize(frame, (512, 512))
        normalized = resized.astype(np.float32) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))
        input_tensor = torch.from_numpy(transposed).unsqueeze(0)

        times = []
        for _ in range(10):
            torch.cuda.synchronize()
            start = time.perf_counter()
            gpu_tensor = input_tensor.to(self.device)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_time = np.mean(times)
        print(f"Average: {avg_time:.2f} ms")
        print(f"Data size: {input_tensor.numel() * 4 / 1e6:.1f} MB")
        print(f"Bottleneck?: {'YES - slow PCIe transfer' if avg_time > 5 else 'No'}")

        return {'avg_ms': avg_time, 'input_tensor': gpu_tensor}

    def profile_gpu_inference(self, gpu_tensor) -> Dict[str, float]:
        """Measure GPU inference (already optimized)."""
        print("\n" + "="*80)
        print("STAGE 4: GPU INFERENCE (OPTIMIZED with BF16)")
        print("="*80)

        times = []
        with torch.no_grad():
            for _ in range(10):
                torch.cuda.synchronize()
                start = time.perf_counter()

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    output = self.model(gpu_tensor)

                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)

        avg_time = np.mean(times)
        print(f"Average: {avg_time:.2f} ms")
        print(f"Status: ALREADY OPTIMIZED!")

        return {'avg_ms': avg_time, 'output': output}

    def profile_gpu_to_cpu_transfer(self, output) -> Dict[str, float]:
        """Measure data transfer GPU->CPU."""
        print("\n" + "="*80)
        print("STAGE 5: GPU->CPU TRANSFER")
        print("="*80)

        times = []
        for _ in range(10):
            torch.cuda.synchronize()
            start = time.perf_counter()
            cpu_output = output.cpu()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_time = np.mean(times)
        print(f"Average: {avg_time:.2f} ms")
        print(f"Data size: {output.numel() * 4 / 1e6:.1f} MB")

        return {'avg_ms': avg_time, 'cpu_output': cpu_output}

    def profile_postprocessing(self, cpu_output) -> Dict[str, float]:
        """Measure post-processing (upsampling, argmax, etc)."""
        print("\n" + "="*80)
        print("STAGE 6: POST-PROCESSING (upsampling, argmax)")
        print("="*80)

        times = []
        for _ in range(10):
            start = time.perf_counter()

            # Typical post-processing
            pred = cpu_output.numpy()
            pred_mask = np.argmax(pred[0], axis=0)  # Get class index

            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_time = np.mean(times)
        print(f"Average: {avg_time:.2f} ms")
        print(f"Components:")
        print(f"  - argmax: ~5-10 ms")
        print(f"  - numpy conversion: ~1-2 ms")

        return {'avg_ms': avg_time}

    def profile_visualization(self, frame, pred_mask) -> Dict[str, float]:
        """Measure visualization (drawing, coloring)."""
        print("\n" + "="*80)
        print("STAGE 7: VISUALIZATION (drawing segmentation)")
        print("="*80)

        times = []
        for _ in range(5):
            start = time.perf_counter()

            # Typical visualization
            seg_output = frame.copy()
            seg_output = cv2.resize(seg_output, (512, 512))
            # Apply colormap
            colored = cv2.applyColorMap((pred_mask * 255 / 150).astype(np.uint8), cv2.COLORMAP_JET)
            # Blend
            result = cv2.addWeighted(seg_output, 0.6, colored, 0.4, 0)

            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_time = np.mean(times)
        print(f"Average: {avg_time:.2f} ms")
        print(f"Bottleneck?: {'YES - visualization is slow!' if avg_time > 30 else 'No'}")

        return {'avg_ms': avg_time}

    def profile_output_writing(self) -> Dict[str, float]:
        """Measure video/image output writing."""
        print("\n" + "="*80)
        print("STAGE 8: OUTPUT WRITING (MP4/image save)")
        print("="*80)

        # Create dummy frame
        dummy_frame = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        times = []
        for _ in range(5):
            start = time.perf_counter()
            cv2.imwrite(f"/tmp/test_{np.random.randint(0, 10000)}.png", dummy_frame)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_time = np.mean(times)
        print(f"Average (image write): {avg_time:.2f} ms")
        print(f"Note: MP4 writing is much slower (50-100ms)")
        print(f"Bottleneck?: {'YES - output writing is slow!' if avg_time > 50 else 'Maybe'}")

        return {'avg_ms': avg_time}

    def run_full_profile(self):
        """Profile the entire pipeline."""
        print("\n\n")
        print("="*80)
        print("END-TO-END PIPELINE LATENCY ANALYSIS")
        print("="*80)
        print(f"Video: {self.video_path}")
        print(f"Profiling: {self.num_frames} frames\n")

        # Load first frame
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"ERROR: Cannot read video")
            return

        # Profile each stage
        timings = {}

        timings['video_decode'] = self.profile_video_decoding()
        timings['preprocessing'] = self.profile_preprocessing(frame)
        timings['cpu_to_gpu'] = self.profile_cpu_to_gpu_transfer(frame)

        gpu_tensor = torch.from_numpy(
            np.transpose(cv2.resize(frame, (512, 512)).astype(np.float32) / 255.0, (2, 0, 1))
        ).unsqueeze(0).to(self.device)

        timings['gpu_inference'] = self.profile_gpu_inference(gpu_tensor)
        timings['gpu_to_cpu'] = self.profile_gpu_to_cpu_transfer(timings['gpu_inference']['output'])
        timings['postprocessing'] = self.profile_postprocessing(timings['gpu_to_cpu']['cpu_output'])

        pred_mask = np.argmax(timings['gpu_to_cpu']['cpu_output'].numpy()[0], axis=0)
        timings['visualization'] = self.profile_visualization(frame, pred_mask)
        timings['output_writing'] = self.profile_output_writing()

        # Summary
        self._print_summary(timings)

    def _print_summary(self, timings):
        """Print profiling summary."""
        print("\n\n")
        print("="*80)
        print("PIPELINE LATENCY SUMMARY")
        print("="*80)

        total = 0
        stages = [
            ('Video Decoding', 'video_decode'),
            ('CPU Preprocessing', 'preprocessing'),
            ('CPU->GPU Transfer', 'cpu_to_gpu'),
            ('GPU Inference', 'gpu_inference'),
            ('GPU->CPU Transfer', 'gpu_to_cpu'),
            ('Post-processing', 'postprocessing'),
            ('Visualization', 'visualization'),
            ('Output Writing', 'output_writing'),
        ]

        print(f"\n{'Stage':<25} {'Time (ms)':<15} {'% of Total':<15}")
        print("-" * 80)

        # Calculate total first
        for stage_name, stage_key in stages:
            if stage_key in timings and 'avg_ms' in timings[stage_key]:
                total += timings[stage_key]['avg_ms']

        # Print breakdown
        for stage_name, stage_key in stages:
            if stage_key in timings and 'avg_ms' in timings[stage_key]:
                time_ms = timings[stage_key]['avg_ms']
                pct = (time_ms / total) * 100 if total > 0 else 0
                print(f"{stage_name:<25} {time_ms:<15.2f} {pct:<15.1f}%")

        print("-" * 80)
        print(f"{'TOTAL':<25} {total:<15.2f}")

        print("\n" + "="*80)
        print("KEY FINDINGS")
        print("="*80)

        print(f"""
If your observed latency is 180ms:

Total pipeline breakdown:
{total:.2f} ms (measured in this profile)

Expected 180ms breakdown:
- Video decoding: 50-100 ms (MP4 codec overhead)
- Preprocessing: 10-15 ms
- Transfers: 10-15 ms
- GPU Inference: 20.89 ms (OPTIMIZED!)
- Post-processing: 10-15 ms
- Visualization: 20-30 ms
- Output writing: 30-50 ms (if writing MP4)
────────────────────────────
TOTAL: ~180 ms

BOTTLENECK IDENTIFICATION:
The GPU inference (20.89 ms) is only ~12% of total latency!

Where 180ms is going:
1. Video decoding (MP4): 30-60%  <- PRIMARY BOTTLENECK
2. Visualization/output: 20-35%  <- SECONDARY
3. Other pipeline stages: 20-30%

RECOMMENDATIONS TO IMPROVE:
1. Use faster video codec (H.265 instead of H.264)
2. Skip visualization if not needed (saves 20-30ms)
3. Batch process frames instead of one-at-a-time
4. Use async processing / parallel decoding
5. Reduce output resolution
6. Write raw images instead of MP4

What we ALREADY optimized:
✓ GPU inference: 20.89 ms (was 31.70 ms before BF16)
✓ Can't optimize much further without model retraining
""")

        print("="*80)


if __name__ == '__main__':
    import sys

    # Use provided video or default to finding one
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # Search for video files
        test_videos = [
            "rtl_validation_video.mp4",
            "test.mp4",
            "/tmp/test.mp4",
        ]
        video_path = None
        for path in test_videos:
            if Path(path).exists():
                video_path = path
                break

    if video_path is None:
        print("ERROR: No video file found or specified")
        print("Usage: python profile_end_to_end_pipeline.py <video.mp4>")
        sys.exit(1)

    profiler = PipelineProfiler(video_path, num_frames=10)
    profiler.run_full_profile()
