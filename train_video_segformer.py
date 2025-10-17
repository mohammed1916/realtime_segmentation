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
    from mmseg.models.decode_heads.tv3s_head import TV3SHead_shift_city
    from mmseg.models.segmentors.encoder_decoder_clips import EncoderDecoder_clips
    # Import dataset module to register CityscapesDataset_clips
    try:
        import mmseg.datasets.cityscapes_video as _city_ds  # registers dataset classes
        print("✓ Cityscapes video dataset module imported and registered")
    except Exception:
        # not fatal here; the dataset may register elsewhere
        print("⚠ Could not import mmseg.datasets.cityscapes_video; dataset may not be registered yet")

    # Ensure CityscapesDataset_clips is actually registered in the mmseg registry
    try:
        from mmseg.registry import DATASETS as MMSEG_DATASETS
        try:
            # try to resolve the dataset by name
            MMSEG_DATASETS.get('CityscapesDataset_clips')
            print("✓ CityscapesDataset_clips already present in mmseg registry")
        except Exception:
            # import the class and register it explicitly
            try:
                from mmseg.datasets.cityscapes_video import CityscapesDataset_clips
                MMSEG_DATASETS.register_module(module=CityscapesDataset_clips)
                print("✓ Registered CityscapesDataset_clips into mmseg registry")
            except Exception as _e:
                print(f"⚠ Could not import/register CityscapesDataset_clips: {_e}")
    except Exception:
        # If registry import fails, skip explicit registration
        print("⚠ Could not access mmseg registry to verify dataset registration")
    # Import tv3s_utils transforms so custom pipeline transforms (RandomCrop_clips, etc.)
    # are registered into mmseg's TRANSFORMS/PIPELINES before building datasets.
    try:
        import tv3s_utils.utils.datasets.transforms as _tv3s_transforms
        print('✓ Imported tv3s_utils transforms to register custom pipelines')
    except Exception:
        try:
            import tv3s_utils.datasets.transforms as _tv3s_transforms
            print('✓ Imported tv3s_utils.datasets.transforms (alternate path)')
        except Exception as _e:
            print(f"⚠ Could not import tv3s custom transforms: {_e}")

    print("✓ TV3S components imported successfully")
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
        # Remove unsupported keys for EpochBasedTrainLoop in this environment
        try:
            train_cfg = cfg.get('train_cfg', None)
            if train_cfg is not None:
                # handle both dict-like Config and plain dict
                loop_type = train_cfg.get('type', '') if hasattr(train_cfg, 'get') else getattr(train_cfg, 'type', '')
                if loop_type == 'EpochBasedTrainLoop' and 'max_iters' in train_cfg:
                    try:
                        del train_cfg['max_iters']
                    except Exception:
                        # fallback for attribute-style
                        if hasattr(train_cfg, 'max_iters'):
                            delattr(train_cfg, 'max_iters')
                    print("⚑ Removed unsupported 'max_iters' from train_cfg for EpochBasedTrainLoop")
        except Exception as _e:
            print(f"⚠ Could not sanitize train_cfg: {_e}")

        runner = Runner.from_cfg(cfg)
        print("✓ Runner created successfully")
        runner.train()
    except Exception as e:
        print(f"✗ Failed to create runner: {e}")
        return

if __name__ == '__main__':
    main()
