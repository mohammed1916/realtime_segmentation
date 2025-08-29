#!/usr/bin/env python3
"""
Fix all SegFormer config files for MMSegmentation 1.2.2 compatibility
"""

import os
import re
from pathlib import Path

def fix_config_file(config_path):
    """Fix a single config file for MMSegmentation 1.2.2 compatibility"""
    print(f"🔧 Fixing {config_path}...")

    with open(config_path, 'r') as f:
        content = f.read()

    # Fix MultiScaleFlipAug parameters
    # Change img_scale to scale
    content = re.sub(r'img_scale=\(([^)]+)\)', r'scale=(\1)', content)

    # Change flip to allow_flip
    content = re.sub(r'flip=False', r'allow_flip=False', content)
    content = re.sub(r'flip=True', r'allow_flip=True', content)

    # Remove RandomFlip from transforms if allow_flip=False
    if 'allow_flip=False' in content:
        # Remove RandomFlip from the transforms list
        content = re.sub(r'(\s*)"type": "RandomFlip",\s*', r'\1', content)

    # Write back the fixed content
    with open(config_path, 'w') as f:
        f.write(content)

    print(f"✅ Fixed {config_path}")

def fix_all_configs():
    """Fix all SegFormer config files"""
    config_dir = Path("local_configs/segformer")

    if not config_dir.exists():
        print("❌ Config directory not found!")
        return

    print("🚀 Fixing all SegFormer config files for MMSegmentation 1.2.2 compatibility")
    print("=" * 80)

    # Get all config files (excluding already fixed ones)
    config_files = []
    for config_file in config_dir.glob("*.py"):
        if not config_file.name.endswith("_fixed.py"):
            config_files.append(config_file)

    print(f"📁 Found {len(config_files)} config files to fix:")
    for config_file in config_files:
        print(f"   - {config_file.name}")

    print("\n🔧 Applying fixes...")

    for config_file in config_files:
        try:
            fix_config_file(config_file)
        except Exception as e:
            print(f"❌ Failed to fix {config_file}: {e}")

    print("\n✅ All config files have been fixed!")
    print("\n📝 Summary of changes:")
    print("   - Changed 'img_scale' to 'scale' in MultiScaleFlipAug")
    print("   - Changed 'flip' to 'allow_flip' in MultiScaleFlipAug")
    print("   - Removed RandomFlip from transforms when allow_flip=False")

if __name__ == "__main__":
    fix_all_configs()
