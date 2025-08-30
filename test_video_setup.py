#!/usr/bin/env python3
"""
Test script for SegFormer + TV3S video segmentation setup.

This script verifies that all components are properly configured and can be imported.
"""

import os
import sys
import torch

def test_imports():
    """Test that all required components can be imported."""
    print("Testing imports...")

    # Add TV3S path
    tv3s_path = os.path.join(os.path.dirname(__file__), '..', 'TV3S')
    if tv3s_path not in sys.path:
        sys.path.insert(0, tv3s_path)

    try:
        # Test MMSegmentation imports
        from mmseg.utils import register_all_modules
        print("✓ MMSegmentation imported")

        # Test TV3S components
        from mmseg.models.decode_heads.tv3s_head import TV3SHead_shift_city
        print("✓ TV3SHead_shift_city imported")

        from mmseg.models.segmentors.encoder_decoder_clips import EncoderDecoder_clips
        print("✓ EncoderDecoder_clips imported")

        # Test TV3S dataset import (try multiple approaches)
        dataset_imported = False
        try:
            from TV3S.utils.datasets.cityscapes import CityscapesDataset_clips
            dataset_imported = True
        except ImportError:
            try:
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'TV3S', 'utils', 'datasets'))
                import cityscapes
                dataset_imported = True
            except ImportError:
                pass

        if dataset_imported:
            print("✓ CityscapesDataset_clips imported")
        else:
            print("⚠ CityscapesDataset_clips import failed (will be handled in training script)")
            return False

        # Test mmengine registry
        from mmengine.registry import MODELS
        print("✓ mmengine registry imported")

        return True

    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_configuration():
    """Test that configuration can be loaded."""
    print("\nTesting configuration...")

    config_file = 'local_configs/segformer/segformer_cityscapes_video.py'

    if not os.path.exists(config_file):
        print(f"✗ Configuration file not found: {config_file}")
        return False

    try:
        from mmengine import Config
        cfg = Config.fromfile(config_file)
        print("✓ Configuration loaded successfully")

        # Check key components
        print(f"  - Model type: {cfg.model.type}")
        print(f"  - Dataset type: {cfg.data.train.type}")
        print(f"  - Batch size: {cfg.data.samples_per_gpu}")

        return True

    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_data_organization():
    """Test data organization script."""
    print("\nTesting data organization script...")

    script_file = 'organize_cityscapes_video.py'

    if not os.path.exists(script_file):
        print(f"✗ Data organization script not found: {script_file}")
        return False

    print("✓ Data organization script found")
    print("  Run: python organize_cityscapes_video.py --input_dir /path/to/frames --output_dir dataset")

    return True

def test_environment():
    """Test the training environment."""
    print("\nTesting environment...")

    # Check PyTorch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  - GPU count: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            print(f"  - Current GPU: {torch.cuda.get_device_name()}")

    # Check Python version
    print(f"✓ Python version: {sys.version}")

    return True

def main():
    """Run all tests."""
    print("="*60)
    print("SegFormer + TV3S Video Segmentation Setup Test")
    print("="*60)

    tests = [
        test_environment,
        test_imports,
        test_configuration,
        test_data_organization,
    ]

    results = []
    for test in tests:
        results.append(test())

    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)

    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✓ All tests passed ({passed}/{total})")
        print("\n🎉 Your SegFormer + TV3S setup is ready!")
        print("\nNext steps:")
        print("1. Organize your Cityscapes video data:")
        print("   python organize_cityscapes_video.py --input_dir /path/to/frames --output_dir dataset")
        print("2. Update data_root in segformer_cityscapes_video.py")
        print("3. Start training:")
        print("   python train_video_segformer.py")
    else:
        print(f"⚠ Some tests failed ({passed}/{total})")
        print("\nPlease check the error messages above and fix any issues before training.")

    print("="*60)

if __name__ == '__main__':
    main()
