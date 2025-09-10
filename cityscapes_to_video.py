#!/usr/bin/env python3
"""
Cityscapes Image Sequence to Video Converter

This script converts Cityscapes image sequences to video files using OpenCV.
It supports converting individual sequences or batch processing multiple sequences.

Usage:
    python cityscapes_to_video.py --input_dir /path/to/sequence --output_dir /path/to/output
    python cityscapes_to_video.py --batch --input_dir /path/to/demoVideo --output_dir /path/to/output

Author: Mohammed Abdullah
"""

import os
import cv2
import glob
import argparse
from pathlib import Path
from typing import List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CityscapesVideoConverter:
    """Converts Cityscapes image sequences to video files."""

    def __init__(self, fps: int = 17, codec: str = 'MJPG'):
        """
        Initialize the video converter.

        Args:
            fps: Frames per second for the output video (default: 17, typical for Cityscapes)
            codec: Video codec to use (default: MJPG)
        """
        self.fps = fps
        self.codec = codec

        # Handle different OpenCV versions
        try:
            self.fourcc = cv2.VideoWriter_fourcc(*codec)
        except AttributeError:
            # For newer OpenCV versions
            self.fourcc = cv2.VideoWriter.fourcc(*codec)

    def get_image_sequence(self, input_dir: str) -> List[str]:
        """
        Get sorted list of image files from the input directory.

        Args:
            input_dir: Path to directory containing image sequence

        Returns:
            Sorted list of image file paths
        """
        # Cityscapes image naming pattern: *_leftImg8bit.png
        pattern = os.path.join(input_dir, "*_leftImg8bit.png")
        image_files = glob.glob(pattern)

        if not image_files:
            raise ValueError(f"No Cityscapes images found in {input_dir}")

        # Sort by frame number (extract numbers from filename)
        def extract_frame_number(filename):
            # Extract the frame number from filename like "stuttgart_00_000000_000123_leftImg8bit.png"
            basename = os.path.basename(filename)
            parts = basename.split('_')
            if len(parts) >= 4:
                try:
                    return int(parts[3])  # The frame number part
                except ValueError:
                    pass
            return 0

        image_files.sort(key=extract_frame_number)
        logger.info(f"Found {len(image_files)} images in sequence")
        return image_files

    def convert_sequence_to_video(self, input_dir: str, output_path: str) -> bool:
        """
        Convert a single image sequence to video.

        Args:
            input_dir: Path to directory containing image sequence
            output_path: Path for output video file

        Returns:
            True if conversion successful, False otherwise
        """
        try:
            # Get sorted image files
            image_files = self.get_image_sequence(input_dir)

            if not image_files:
                logger.error(f"No images found in {input_dir}")
                return False

            # Read first image to get dimensions
            first_image = cv2.imread(image_files[0])
            if first_image is None:
                logger.error(f"Could not read first image: {image_files[0]}")
                return False

            height, width = first_image.shape[:2]
            logger.info(f"Video dimensions: {width}x{height}, {len(image_files)} frames")

            # Create video writer
            out = cv2.VideoWriter(output_path, self.fourcc, self.fps, (width, height))

            if not out.isOpened():
                logger.error(f"Could not create video writer for {output_path}")
                return False

            # Process each image
            for i, image_file in enumerate(image_files):
                if i % 50 == 0:  # Progress logging
                    logger.info(f"Processing frame {i+1}/{len(image_files)}")

                image = cv2.imread(image_file)
                if image is None:
                    logger.warning(f"Could not read image: {image_file}")
                    continue

                out.write(image)

            out.release()
            logger.info(f"Successfully created video: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error converting sequence: {str(e)}")
            return False

    def batch_convert(self, input_dir: str, output_dir: str) -> int:
        """
        Convert multiple sequences in batch mode.

        Args:
            input_dir: Path to directory containing multiple sequences
            output_dir: Path for output directory

        Returns:
            Number of successfully converted sequences
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Find all sequence directories
        sequence_dirs = [d for d in os.listdir(input_dir)
                        if os.path.isdir(os.path.join(input_dir, d))]

        if not sequence_dirs:
            logger.error(f"No sequence directories found in {input_dir}")
            return 0

        logger.info(f"Found {len(sequence_dirs)} sequences to convert")

        success_count = 0
        for sequence_dir in sequence_dirs:
            input_path = os.path.join(input_dir, sequence_dir)
            output_path = os.path.join(output_dir, f"{sequence_dir}.avi")

            logger.info(f"Converting sequence: {sequence_dir}")
            if self.convert_sequence_to_video(input_path, output_path):
                success_count += 1

        logger.info(f"Batch conversion complete: {success_count}/{len(sequence_dirs)} successful")
        return success_count

def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description="Convert Cityscapes image sequences to video")
    parser.add_argument("--input_dir", required=True, help="Input directory containing image sequence(s)")
    parser.add_argument("--output_dir", required=True, help="Output directory for video file(s)")
    parser.add_argument("--output_file", help="Output video filename (for single sequence)")
    parser.add_argument("--fps", type=int, default=17, help="Frames per second (default: 17)")
    parser.add_argument("--codec", default="MJPG", help="Video codec (default: MJPG)")
    parser.add_argument("--batch", action="store_true", help="Batch convert multiple sequences")

    args = parser.parse_args()

    # Create converter
    converter = CityscapesVideoConverter(fps=args.fps, codec=args.codec)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if args.batch:
        # Batch conversion
        success_count = converter.batch_convert(args.input_dir, args.output_dir)
        print(f"Batch conversion completed: {success_count} sequences converted")
    else:
        # Single sequence conversion
        if args.output_file:
            output_path = os.path.join(args.output_dir, args.output_file)
        else:
            # Use input directory name as output filename
            input_name = os.path.basename(os.path.normpath(args.input_dir))
            output_path = os.path.join(args.output_dir, f"{input_name}.avi")

        if converter.convert_sequence_to_video(args.input_dir, output_path):
            print(f"Successfully converted sequence to: {output_path}")
        else:
            print("Conversion failed")
            return 1

    return 0

if __name__ == "__main__":
    exit(main())
