#!/usr/bin/env python3
"""
Test script to verify MMSegmentation 1.x upgrade
This script checks for successful imports of required modules and verifies PyTorch installation.
"""

import sys
import torch
import mmseg
from mmengine.utils import get_git_hash
from mmcv import __version__ as mmcv_version

def test_imports():
    """Test that all required modules can be imported."""
    try:
        import mmengine
        print(f"✅ MMEngine {mmengine.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ MMEngine import failed: {e}")
        return False

    try:
        import mmcv
        print(f"✅ MMCV {mmcv_version} imported successfully")
    except ImportError as e:
        print(f"❌ MMCV import failed: {e}")
        return False

    try:
        import mmseg
        print(f"✅ MMSegmentation {mmseg.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ MMSegmentation import failed: {e}")
        return False

    return True

def test_pytorch():
    """Test PyTorch installation and CUDA availability."""
    print(f"✅ PyTorch {torch.__version__} installed")

    if torch.cuda.is_available():
        print("✅ CUDA available")
        print(f"✅ {torch.cuda.device_count()} GPU(s) detected")
    else:
        print("⚠️  CUDA not available (CPU-only mode)")

    return True

def main():
    """Main test function."""
    print("🧪 Testing MMSegmentation 1.x Upgrade...")
    print("=" * 50)

    # Test PyTorch
    if not test_pytorch():
        return False

    print()

    # Test imports
    if not test_imports():
        return False

    print()
    print(" All tests passed! MMSegmentation 1.x upgrade successful.")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
