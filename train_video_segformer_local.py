#!/usr/bin/env python3
"""
Training script for SegFormer with TV3S temporal Mamba blocks on Cityscapes video data.
Uses copied TV3S utilities to avoid import path issues.
"""

import os
import sys
import torch
import mmengine
from mmengine.runner import Runner
from mmseg.utils import register_all_modules

def setup_environment():
    """Setup the training environment and register components."""

    # Register MMSegmentation modules
    register_all_modules()

    # Add current directory to path for local imports
    current_dir = os.path.dirname(__file__)
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    # Import and register TV3S components from copied files
    try:
        from tv3s_utils.models.tv3s_head import TV3SHead_shift_city
        from tv3s_utils.models.encoder_decoder import EncoderDecoder_clips
        print("✓ TV3S model components imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import TV3S model components: {e}")
        return False

    # Try importing dataset components
    try:
        from tv3s_utils.datasets.cityscapes import CityscapesDataset_clips
        print("✓ TV3S dataset components imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import TV3S dataset components: {e}")
        print("Will use standard CityscapesDataset instead")
        return True  # Continue with standard dataset

    return True

def main():
    """Main training function."""
    print("Setting up TV3S training environment...")

    if not setup_environment():
        print("Failed to setup environment. Exiting.")
        return

    # Configuration file path
    config_file = 'local_configs/segformer/segformer_cityscapes_video.py'

    if not os.path.exists(config_file):
        print(f"Configuration file not found: {config_file}")
        return

    print(f"Loading configuration from: {config_file}")

    # Load configuration
    cfg = mmengine.Config.fromfile(config_file)

    # Modify configuration for video training
    cfg.model.type = 'EncoderDecoder_clips'
    cfg.model.decode_head.type = 'TV3SHead_shift_city'

    # Set dataset paths
    cfg.data_root = 'dataset/leftImg8bit_trainvaltest'
    cfg.ann_root = 'dataset/gtFine'

    # Set training parameters
    cfg.train_dataloader.dataset.data_root = cfg.data_root
    cfg.train_dataloader.dataset.ann_root = cfg.ann_root
    cfg.val_dataloader.dataset.data_root = cfg.data_root
    cfg.val_dataloader.dataset.ann_root = cfg.ann_root
    cfg.test_dataloader.dataset.data_root = cfg.data_root
    cfg.test_dataloader.dataset.ann_root = cfg.ann_root

    print("Configuration loaded successfully!")
    print(f"Model: {cfg.model.type}")
    print(f"Decode Head: {cfg.model.decode_head.type}")
    print(f"Data root: {cfg.data_root}")

    # Create work directory
    work_dir = 'work_dirs/tv3s_segformer_training'
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
