#!/usr/bin/env python3
"""
Training script for SegFormer with TV3S temporal Mamba blocks on Cityscapes video data.

This script sets up the environment, registers TV3S components, and starts training.
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

    # Import and register TV3S components
    try:
        # Add TV3S path
        tv3s_path = os.path.join(os.path.dirname(__file__), '..', 'TV3S')
        if tv3s_path not in sys.path:
            sys.path.insert(0, tv3s_path)

        from mmseg.models.decode_heads.tv3s_head import TV3SHead_shift_city
        from mmseg.models.segmentors.encoder_decoder_clips import EncoderDecoder_clips

        # Import TV3S transforms to register them
        import TV3S.utils.datasets.transforms
        import TV3S.utils.datasets.dataset_pipelines

        # Try importing CityscapesDataset_clips
        try:
            from TV3S.utils.datasets.cityscapes import CityscapesDataset_clips
            dataset_imported = True
        except ImportError:
            # Try alternative import
            sys.path.insert(0, os.path.join(tv3s_path, 'utils', 'datasets'))
            from cityscapes import CityscapesDataset_clips
            dataset_imported = True

        if dataset_imported:
            print("✓ TV3S components imported successfully")
        else:
            print("✗ Failed to import CityscapesDataset_clips")
            return False

    except ImportError as e:
        print(f"✗ Failed to import TV3S components: {e}")
        print("Make sure TV3S is properly set up")
        return False

    return True

def main():
    """Main training function."""

    print("Setting up SegFormer + TV3S training environment...")

    if not setup_environment():
        return

    # Configuration file
    config_file = 'local_configs/segformer/segformer_cityscapes_video.py'

    if not os.path.exists(config_file):
        print(f"✗ Configuration file not found: {config_file}")
        print("Please ensure the video configuration file exists")
        return

    # Load configuration
    try:
        cfg = mmengine.Config.fromfile(config_file)
        print("✓ Configuration loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return

    # Update data root if needed (you can modify this path)
    if not os.path.exists(cfg.data_root):
        print(f"⚠ Warning: Data root directory not found: {cfg.data_root}")
        print("Please update the data_root in the configuration file to point to your Cityscapes video data")
        print("Or run: python organize_cityscapes_video.py --input_dir /path/to/your/frames --output_dir dataset")

    # Setup work directory
    work_dir = cfg.get('work_dir', 'work_dirs/segformer_cityscapes_video')
    os.makedirs(work_dir, exist_ok=True)

    # Update configuration
    cfg.work_dir = work_dir

    # Set random seed
    cfg.randomness = dict(seed=42, deterministic=False)

    # Print configuration summary
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION SUMMARY")
    print("="*50)
    print(f"Model: {cfg.model.type}")
    print(f"Backbone: {cfg.model.backbone.type}")
    print(f"Decode Head: {cfg.model.decode_head.type}")
    print(f"Dataset: {cfg.data.train.type}")
    print(f"Data Root: {cfg.data_root}")
    print(f"Batch Size: {cfg.data.samples_per_gpu}")
    print(f"Max Epochs: {cfg.train_cfg.max_epochs}")
    print(f"Work Directory: {cfg.work_dir}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("="*50 + "\n")

    # Create runner
    try:
        runner = Runner.from_cfg(cfg)
        print("✓ Runner created successfully")
    except Exception as e:
        print(f"✗ Failed to create runner: {e}")
        return

    # Start training
    print("Starting training...")
    try:
        runner.train()
        print("✓ Training completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        raise

if __name__ == '__main__':
    main()
