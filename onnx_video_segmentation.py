#!/usr/bin/env python3
"""
ONNX-based Real-time Video Segmentation with Temporal Smoothing

This script performs real-time semantic segmentation on video using ONNX models
with temporal logit smoothing for consistent results across frames.

Usage:
    python onnx_video_segmentation.py --video_path videos/stuttgart_00.avi \
        --onnx_model optimized_models/segformer.b0.1024x1024.city.160k/20250829_215706_000001/segformer.b0.1024x1024.city.160k/segformer.b0.1024x1024.city.160k_onnx.onnx \
        --output_file results/onnx_cityscapes_segmentation.avi \
        --show --smooth-alpha 0.7
"""

import argparse
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import time
from typing import Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ONNXVideoSegmenter:
    """ONNX-based video segmentation with temporal smoothing."""

    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize the ONNX segmenter.

        Args:
            model_path: Path to ONNX model
            device: Device to run inference on ('cpu', 'cuda')
        """
        self.model_path = model_path
        self.device = device

        # Initialize ONNX session
        providers = ['CPUExecutionProvider'] if device == 'cpu' else ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)

        # Get model input/output names and shapes
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        input_shape = self.session.get_inputs()[0].shape
        logger.info(f"Model input shape: {input_shape}")

        # Assume input shape is [1, 3, H, W]
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]

        # Cityscapes color palette (19 classes)
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
        ], dtype=np.uint8)

        # Image normalization (ImageNet)
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for ONNX model inference.

        Args:
            image: Input image (H, W, C)

        Returns:
            Preprocessed image tensor (1, C, H, W)
        """
        # Convert BGR to RGB
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        img = cv2.resize(img, (self.input_width, self.input_height))

        # Convert to float32
        img = img.astype(np.float32)

        # Normalize
        img = (img - self.mean) / self.std

        # Transpose to CHW
        img = img.transpose(2, 0, 1)

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        return img

    def postprocess_logits(self, logits: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
        """
        Postprocess model output logits.

        Args:
            logits: Model output logits (1, num_classes, H, W)
            original_shape: Original image shape (H, W)

        Returns:
            Processed logits (num_classes, H, W)
        """
        # Remove batch dimension
        logits = logits[0]  # (num_classes, H, W)

        # Upsample to original size if needed
        if logits.shape[1:] != original_shape:
            logits = cv2.resize(logits.transpose(1, 2, 0), original_shape[::-1], interpolation=cv2.INTER_LINEAR)
            logits = logits.transpose(2, 0, 1)

        return logits

    def apply_temporal_smoothing(self, current_logits: np.ndarray,
                               prev_logits: Optional[np.ndarray],
                               alpha: float) -> np.ndarray:
        """
        Apply temporal smoothing to logits.

        Args:
            current_logits: Current frame logits
            prev_logits: Previous frame logits
            alpha: Smoothing factor (0-1)

        Returns:
            Smoothed logits
        """
        if prev_logits is None:
            return current_logits

        # Apply exponential moving average
        smoothed = (1 - alpha) * current_logits + alpha * prev_logits
        return smoothed

    def create_visualization(self, image: np.ndarray, pred_mask: np.ndarray,
                           opacity: float = 0.5) -> np.ndarray:
        """
        Create visualization by overlaying segmentation on original image.

        Args:
            image: Original image
            pred_mask: Predicted segmentation mask
            opacity: Opacity for overlay

        Returns:
            Visualization image
        """
        # Create colored segmentation mask
        colored_mask = self.palette[pred_mask]

        # Resize mask to match image size
        if colored_mask.shape[:2] != image.shape[:2]:
            colored_mask = cv2.resize(colored_mask, (image.shape[1], image.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)

        # Blend with original image
        overlay = cv2.addWeighted(image, 1 - opacity, colored_mask, opacity, 0)
        return overlay

    def process_video(self, video_path: str, output_path: Optional[str] = None,
                     show: bool = False, smooth_alpha: float = 0.6,
                     opacity: float = 0.5) -> None:
        """
        Process video with real-time segmentation.

        Args:
            video_path: Path to input video
            output_path: Path to output video (optional)
            show: Whether to display results
            smooth_alpha: Temporal smoothing factor
            opacity: Overlay opacity
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")

        # Initialize video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            logger.info(f"Output video: {output_path}")

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

                # Preprocess image
                input_tensor = self.preprocess_image(frame)

                # Run inference
                outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
                logits = outputs[0]

                # Postprocess logits
                logits = self.postprocess_logits(logits, (height, width))

                # Apply temporal smoothing
                smoothed_logits = self.apply_temporal_smoothing(logits, prev_logits, smooth_alpha)
                prev_logits = smoothed_logits.copy()

                # Get prediction
                pred_mask = np.argmax(smoothed_logits, axis=0).astype(np.uint8)

                # Create visualization
                vis_image = self.create_visualization(frame, pred_mask, opacity)

                # Display
                if show:
                    cv2.imshow('ONNX Video Segmentation', vis_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Write to output video
                if writer:
                    writer.write(vis_image)

                # Progress logging
                if frame_count % 50 == 0:
                    elapsed = time.time() - start_time
                    fps_current = frame_count / elapsed
                    logger.info(f"Processed {frame_count}/{total_frames} frames ({fps_current:.2f} FPS)")

        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()

            # Final statistics
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time
            logger.info(f"Processing complete: {frame_count} frames in {total_time:.2f}s ({avg_fps:.2f} FPS)")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='ONNX-based Real-time Video Segmentation')
    parser.add_argument('--video_path', required=True, help='Path to input video')
    parser.add_argument('--onnx_model', required=True, help='Path to ONNX model')
    parser.add_argument('--output_file', help='Output video file path')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Inference device')
    parser.add_argument('--show', action='store_true', help='Display results')
    parser.add_argument('--smooth-alpha', type=float, default=0.6, help='Temporal smoothing factor (0-1)')
    parser.add_argument('--opacity', type=float, default=0.5, help='Overlay opacity')

    args = parser.parse_args()

    # Initialize segmenter
    segmenter = ONNXVideoSegmenter(args.onnx_model, args.device)

    # Process video
    segmenter.process_video(
        video_path=args.video_path,
        output_path=args.output_file,
        show=args.show,
        smooth_alpha=args.smooth_alpha,
        opacity=args.opacity
    )


if __name__ == '__main__':
    main()
