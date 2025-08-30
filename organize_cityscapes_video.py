#!/usr/bin/env python3
"""
Cityscapes Video Data Organizer

This script helps organize Cityscapes video frames into the structure expected
by the TV3S video dataset for training SegFormer with temporal Mamba blocks.

Expected input structure:
- Individual frame images: {city}_{sequence}_{frame}_leftImg8bit.png
- Individual annotation images: {city}_{sequence}_{frame}_gtFine_labelTrainIds.png

Output structure:
dataset/
├── leftImg8bit/
│   ├── train/
│   │   ├── {city}_{sequence}_{frame}_leftImg8bit.png
│   │   └── ...
│   └── val/
│       └── ...
└── gtFine/
    ├── train/
    │   ├── {city}_{sequence}_{frame}_gtFine_labelTrainIds.png
    │   └── ...
    └── val/
        └── ...

Usage:
    python organize_cityscapes_video.py --input_dir /path/to/video/frames --output_dir dataset
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import re

def parse_cityscapes_filename(filename: str) -> Tuple[str, str, str, str]:
    """
    Parse Cityscapes filename to extract components.

    Expected format: {city}_{sequence}_{frame}_{suffix}.png
    Example: stuttgart_00_000000_000001_leftImg8bit.png

    Returns:
        (city, sequence, frame, suffix)
    """
    # Remove extension
    name = filename.replace('.png', '')

    # Split by underscores
    parts = name.split('_')

    if len(parts) < 4:
        raise ValueError(f"Invalid Cityscapes filename format: {filename}")

    city = parts[0]
    sequence = f"{parts[0]}_{parts[1]}"
    frame = parts[2]  # Usually the frame number
    suffix = '_'.join(parts[3:])  # Everything after frame number

    return city, sequence, frame, suffix

def organize_cityscapes_data(input_dir: str, output_dir: str, train_sequences: Optional[List[str]] = None):
    """
    Organize Cityscapes video data into train/val splits.

    Args:
        input_dir: Directory containing Cityscapes video frames
        output_dir: Output directory for organized data
        train_sequences: List of sequence names for training (others go to val)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directories
    img_train_dir = output_path / 'leftImg8bit' / 'train'
    img_val_dir = output_path / 'leftImg8bit' / 'val'
    ann_train_dir = output_path / 'gtFine' / 'train'
    ann_val_dir = output_path / 'gtFine' / 'val'

    for dir_path in [img_train_dir, img_val_dir, ann_train_dir, ann_val_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Find all image and annotation files
    image_files = list(input_path.glob('*_leftImg8bit.png'))
    ann_files = list(input_path.glob('*_gtFine_*.png'))

    print(f"Found {len(image_files)} image files")
    print(f"Found {len(ann_files)} annotation files")

    # Default train sequences if not provided
    if train_sequences is None:
        train_sequences = ['stuttgart_00', 'stuttgart_01', 'stuttgart_02']

    # Organize files
    for img_file in image_files:
        try:
            city, sequence, frame, suffix = parse_cityscapes_filename(img_file.name)

            # Determine if this is train or val based on sequence
            is_train = sequence in train_sequences

            if is_train:
                target_dir = img_train_dir
            else:
                target_dir = img_val_dir

            # Copy image file
            shutil.copy2(img_file, target_dir / img_file.name)
            print(f"Copied image: {img_file.name} -> {target_dir}")

        except ValueError as e:
            print(f"Skipping invalid image file {img_file.name}: {e}")

    for ann_file in ann_files:
        try:
            city, sequence, frame, suffix = parse_cityscapes_filename(ann_file.name)

            # Determine if this is train or val based on sequence
            is_train = sequence in train_sequences

            if is_train:
                target_dir = ann_train_dir
            else:
                target_dir = ann_val_dir

            # Copy annotation file
            shutil.copy2(ann_file, target_dir / ann_file.name)
            print(f"Copied annotation: {ann_file.name} -> {target_dir}")

        except ValueError as e:
            print(f"Skipping invalid annotation file {ann_file.name}: {e}")

    print("\nData organization complete!")
    print(f"Training images: {len(list(img_train_dir.glob('*.png')))}")
    print(f"Validation images: {len(list(img_val_dir.glob('*.png')))}")
    print(f"Training annotations: {len(list(ann_train_dir.glob('*.png')))}")
    print(f"Validation annotations: {len(list(ann_val_dir.glob('*.png')))}")

def main():
    parser = argparse.ArgumentParser(description='Organize Cityscapes video data')
    parser.add_argument('--input_dir', required=True, help='Input directory containing Cityscapes video frames')
    parser.add_argument('--output_dir', default='dataset', help='Output directory for organized data')
    parser.add_argument('--train_sequences', nargs='+',
                       default=['stuttgart_00', 'stuttgart_01', 'stuttgart_02'],
                       help='Sequence names for training set')

    args = parser.parse_args()

    organize_cityscapes_data(args.input_dir, args.output_dir, args.train_sequences)

if __name__ == '__main__':
    main()
