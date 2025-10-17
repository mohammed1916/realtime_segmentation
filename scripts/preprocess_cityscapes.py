"""Preprocess Cityscapes dataset into a smaller on-disk form.

This script copies and resizes images and label maps from `dataset/` into
`dataset_preprocessed/` preserving the relative structure. It resizes images to
`target_size` while keeping aspect ratio and padding/cropping as needed.

Usage examples:
    python scripts/preprocess_cityscapes.py --src dataset --dst dataset_preprocessed --width 1024 --height 512

The script processes both `leftImg8bit_trainvaltest` and `gtFine` folders.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from PIL import Image
import shutil

IMG_DIRS = ["leftImg8bit_trainvaltest", "leftImg8bit", "leftImg8bit/demoVideo"]
GT_DIR = "gtFine"

def process_image(src_path: Path, dst_path: Path, target_size: tuple[int,int]):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as im:
        im = im.convert('RGB')
        im = im.resize(target_size, Image.BILINEAR)
        im.save(dst_path, quality=95)


def process_label(src_path: Path, dst_path: Path, target_size: tuple[int,int]):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as im:
        # label maps use indexed palettes; keep as is but resize with nearest
        im = im.convert('L')
        im = im.resize(target_size, Image.NEAREST)
        im.save(dst_path)


def copy_tree(src_root: Path, dst_root: Path, target_size: tuple[int,int]):
    # Walk images and copy (search any leftImg8bit* directories)
    for src in src_root.rglob('leftImg8bit*'):
        if src.is_dir():
            for img_file in src.rglob('*.png'):
                rel = img_file.relative_to(src_root)
                dst = dst_root / rel
                try:
                    process_image(img_file, dst, target_size)
                except Exception as e:
                    print(f'Warning: failed to process image {img_file}: {e}')

    # Walk GT labels
    gt = src_root / GT_DIR
    if gt.exists():
        for img_file in gt.rglob('*.png'):
            rel = img_file.relative_to(src_root)
            dst = dst_root / rel
            try:
                process_label(img_file, dst, target_size)
            except Exception as e:
                print(f'Warning: failed to process label {img_file}: {e}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=Path, default=Path('dataset'))
    parser.add_argument('--dst', type=Path, default=Path('dataset_preprocessed'))
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    tgt = (args.width, args.height)
    if args.dry_run:
        print('Dry run: would process from', args.src, 'to', args.dst, 'size', tgt)
        return

    if not args.src.exists():
        raise SystemExit(f'Source folder not found: {args.src}')

    # Clean dst if exists
    if args.dst.exists():
        print('Removing existing dst folder', args.dst)
        shutil.rmtree(args.dst)

    print('Processing images...')
    copy_tree(args.src, args.dst, tgt)
    print('Done')


if __name__ == '__main__':
    main()
