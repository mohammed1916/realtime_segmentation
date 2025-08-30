#!/usr/bin/env python3
"""
Training script for SegFormer on Cityscapes video data.
Uses standard MMSegmentation components without TV3S dependencies.
"""

import os
import sys
import torch
import mmengine
from mmengine.runner import Runner
from mmseg.utils import register_all_modules
# Import custom transforms
from mmseg.transforms.cityscapes_transforms import CityscapesLabelIdToTrainId

def setup_environment():
    """Setup the training environment."""

    # Register MMSegmentation modules
    register_all_modules()

    print("✓ MMSegmentation modules registered successfully")
    return True

def main():
    """Main training function."""
    print("Setting up SegFormer training environment...")

    if not setup_environment():
        print("Failed to setup environment. Exiting.")
        return

    # Configuration file path
    config_file = 'local_configs/segformer/segformer_cityscapes_standalone.py'

    if not os.path.exists(config_file):
        print(f"Configuration file not found: {config_file}")
        return

    print(f"Loading configuration from: {config_file}")

    # Load configuration
    cfg = mmengine.Config.fromfile(config_file)

    print("Configuration loaded successfully!")
    print(f"Model: {cfg.model.type}")
    print(f"Dataset: {cfg.data.train.type}")

    # Create work directory
    work_dir = 'work_dirs/segformer_training'
    os.makedirs(work_dir, exist_ok=True)
    cfg.work_dir = work_dir

    print(f"Work directory: {work_dir}")

    # Initialize the runner
    try:
        runner = Runner.from_cfg(cfg)
        print("✓ Runner initialized successfully")

        # Start training
        print("Starting training...")
        runner.train()

    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
