#!/usr/bin/env python3
"""
ONNX-based Real-time Video Segmentation with SegFormer B0

This script performs real-time video segmentation using optimized ONNX models
for maximum performance and temporal smoothing for consistent results.

Usage:
    python onnx_video_segmentation_b0.py --video_path videos/stuttgart_00.avi --output_path results/onnx_segmentation.avi
"""

import argparse
import cv2
import numpy as np
import onnxruntime as ort
import time
from pathlib import Path
from typing import Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ONNXSegFormerB0:
    """ONNX-based SegFormer B0 model for real-time segmentation."""

    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Initialize ONNX model.

        Args:
            model_path: Path to ONNX model file
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.device = device

        # Set ONNX providers
        if device == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        # Load ONNX model
        logger.info(f"Loading ONNX model from {model_path}")
        self.session = ort.InferenceSession(model_path, providers=providers)

        # Get model input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        input_shape = self.session.get_inputs()[0].shape
        logger.info(f"Model input shape: {input_shape}")

        # Image normalization (Cityscapes/ImageNet standard)
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

        # Cityscapes class colors (19 classes + background)
        self.palette = np.array([
            [128, 64, 128],   # road
            [244, 35, 232],   # sidewalk
            [70, 70, 70],     # building
            [102, 102, 156],  # wall
            [190, 153, 153],  # fence
            [153, 153, 153],  # pole
            [250, 170, 30],   # traffic light
            [220, 220, 0],    # traffic sign
            [107, 142, 35],   # vegetation
            [152, 251, 152],  # terrain
            [70, 130, 180],   # sky
            [220, 20, 60],    # person
            [255, 0, 0],      # rider
            [0, 0, 142],      # car
            [0, 0, 70],       # truck
            [0, 60, 100],     # bus
            [0, 80, 100],     # train
            [0, 0, 230],      # motorcycle
            [119, 11, 32],    # bicycle
            [0, 0, 0]         # background/unlabeled
        ], dtype=np.uint8)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input.

        Args:
            image: Input image in BGR format

        Returns:
            Preprocessed image tensor
        """
        # Convert BGR to RGB
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to model input size (1024x1024 for B0)
        img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LINEAR)

        # Convert to float32
        img = img.astype(np.float32)

        # Normalize
        img = (img - self.mean) / self.std

        # Transpose to CHW format
        img = img.transpose(2, 0, 1)

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        return img

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Run inference on preprocessed image.

        Args:
            image: Preprocessed image tensor

        Returns:
            Segmentation logits
        """
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: image})

        # Get logits (shape: [1, num_classes, H, W])
        logits = outputs[0][0]  # Remove batch dimension

        return logits

    def postprocess(self, logits: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
        """
        Postprocess logits to get segmentation mask.

        Args:
            logits: Model output logits
            original_size: Original image size (H, W)

        Returns:
            Segmentation mask
        """
        # Get prediction by argmax
        pred = np.argmax(logits, axis=0).astype(np.uint8)

        # Resize back to original size
        pred = cv2.resize(pred, original_size[::-1], interpolation=cv2.INTER_NEAREST)

        return pred

    def visualize(self, image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Visualize segmentation result on original image.

        Args:
            image: Original image
            mask: Segmentation mask
            alpha: Transparency for overlay

        Returns:
            Visualized image with segmentation overlay
        """
        # Create color mask
        color_mask = self.palette[mask]

        # Blend with original image
        overlay = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)

        return overlay

def main():
    """Main function for ONNX video segmentation."""
    parser = argparse.ArgumentParser(description="ONNX-based real-time video segmentation with SegFormer B0")
    parser.add_argument('--video_path', required=True, help='Path to input video file')
    parser.add_argument('--output_path', required=True, help='Path to output video file')
    parser.add_argument('--onnx_model', default='optimized_models/segformer.b0.1024x1024.city.160k/20250829_215706_000001/segformer.b0.1024x1024.city.160k/segformer.b0.1024x1024.city.160k_onnx.onnx',
                        help='Path to ONNX model file')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Inference device')
    parser.add_argument('--smooth_alpha', type=float, default=0.7, help='Temporal smoothing weight (0-1)')
    parser.add_argument('--display', action='store_true', help='Display video during processing')
    parser.add_argument('--fps', type=int, default=-1, help='Output video FPS (-1 to use input FPS)')

    args = parser.parse_args()

    # Initialize ONNX model
    model = ONNXSegFormerB0(args.onnx_model, args.device)

    # Open video
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {args.video_path}")
        return 1

    # Get video properties
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Input video: {input_width}x{input_height} @ {input_fps} FPS, {total_frames} frames")

    # Set output FPS
    output_fps = args.fps if args.fps > 0 else input_fps

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(args.output_path, fourcc, output_fps, (input_width, input_height))

    if not out.isOpened():
        logger.error(f"Could not create output video: {args.output_path}")
        return 1

    # Initialize temporal smoothing
    prev_logits = None
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Preprocess frame
            input_tensor = model.preprocess(frame)

            # Run inference
            logits = model.predict(input_tensor)

            # Apply temporal smoothing
            if prev_logits is not None:
                logits = (1 - args.smooth_alpha) * logits + args.smooth_alpha * prev_logits
            prev_logits = logits.copy()

            # Postprocess to get segmentation mask
            mask = model.postprocess(logits, (input_height, input_width))

            # Visualize result
            result_frame = model.visualize(frame, mask)

            # Write to output video
            out.write(result_frame)

            # Display progress
            if frame_count % 50 == 0:
                elapsed = time.time() - start_time
                fps_current = frame_count / elapsed
                logger.info(f"Processed {frame_count}/{total_frames} frames ({fps_current:.1f} FPS)")

                import pynvml

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                logger.info(f"GPU memory used: {mem_info.used / (1024*1024):.2f} MB / {mem_info.total / (1024*1024):.2f} MB") # type: ignore

            # Display frame if requested
            if args.display:
                cv2.imshow('ONNX SegFormer B0 Segmentation', result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")

    finally:
        # Cleanup
        cap.release()
        out.release()
        if args.display:
            cv2.destroyAllWindows()

        # Final statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        logger.info(f"Processing complete!")
        logger.info(f"Total frames: {frame_count}")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Average FPS: {avg_fps:.2f}")
        logger.info(f"Output saved to: {args.output_path}")

    return 0

if __name__ == "__main__":
    exit(main())
