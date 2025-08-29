# Cityscapes Video Converter

This tool converts Cityscapes image sequences to video files for use with the SegFormer video segmentation pipeline.

## Features

- Convert individual Cityscapes image sequences to video
- Batch convert multiple sequences at once
- Automatic frame sorting by sequence number
- Configurable FPS and video codec
- Progress logging and error handling

## Usage

### Single Sequence Conversion

```bash
python cityscapes_to_video.py \
  --input_dir dataset/leftImg8bit/demoVideo/stuttgart_00 \
  --output_dir videos \
  --output_file stuttgart_00_demo.avi
```

### Batch Conversion

```bash
python cityscapes_to_video.py \
  --batch \
  --input_dir dataset/leftImg8bit/demoVideo \
  --output_dir videos
```

### Options

- `--input_dir`: Path to directory containing image sequence(s)
- `--output_dir`: Output directory for video file(s)
- `--output_file`: Output filename (for single sequence mode)
- `--fps`: Frames per second (default: 17, Cityscapes standard)
- `--codec`: Video codec (default: MJPG)
- `--batch`: Enable batch processing of multiple sequences

## Cityscapes Data Structure

The converter expects Cityscapes image sequences in the following format:

```
dataset/leftImg8bit/demoVideo/
├── stuttgart_00/
│   ├── stuttgart_00_000000_000001_leftImg8bit.png
│   ├── stuttgart_00_000000_000002_leftImg8bit.png
│   └── ...
├── stuttgart_01/
│   └── ...
└── stuttgart_02/
    └── ...
```

## Output

- Videos are saved in AVI format with MJPG codec
- Resolution: 2048x1024 (standard Cityscapes)
- Frame rate: 17 FPS (configurable)

## Integration with SegFormer

The converted videos can be used directly with the video segmentation pipeline:

```python
# Example usage with video_logit_smoothing_final.py
python video_logit_smoothing_final.py --video_path videos/stuttgart_00.avi
```

## Requirements

- OpenCV (cv2)
- Python 3.6+
- Cityscapes dataset with image sequences

## Examples

### Convert all demo sequences

```bash
python cityscapes_to_video.py --batch --input_dir dataset/leftImg8bit/demoVideo --output_dir videos
```

### Convert with custom FPS

```bash
python cityscapes_to_video.py --input_dir dataset/leftImg8bit/demoVideo/stuttgart_00 --output_dir videos --fps 30 --output_file stuttgart_00_30fps.avi
```

## Troubleshooting

- Ensure input directory contains images with Cityscapes naming convention
- Check that output directory exists and is writable
- Verify OpenCV installation: `python -c "import cv2; print(cv2.__version__)"`
