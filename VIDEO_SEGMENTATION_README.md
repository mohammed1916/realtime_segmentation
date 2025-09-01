# SegFormer + TV3S Video Segmentation Training

This repository contains the implementation of SegFormer adapted for video segmentation using TV3S temporal Mamba blocks.

## Overview

The implementation combines:

- **SegFormer**: Transformer-based semantic segmentation model
- **TV3S**: Temporal Video Segmentation with Mamba-SSM blocks for temporal modeling
- **Cityscapes**: Video dataset for urban scene understanding

## Features

- ✅ Temporal video segmentation with Mamba-SSM
- ✅ Cityscapes dataset support with video clips
- ✅ Automatic data organization from video frames
- ✅ Configurable temporal window size
- ✅ MMSegmentation integration

## Quick Start

### 1. Data Preparation

First, organize your Cityscapes video frames using the provided script:

```bash
# Organize your Cityscapes video frames
python organize_cityscapes_video.py \
  --input_dir /path/to/your/cityscapes/frames \
  --output_dir dataset
```

This will create the expected directory structure:

```
dataset/
├── leftImg8bit/
│   ├── train/
│   └── val/
└── gtFine/
    ├── train/
    └── val/
```

### 2. Update Configuration

Edit `local_configs/segformer/segformer_cityscapes_video.py` to update:

- `data_root`: Path to your organized dataset
- Training parameters as needed

### 3. Start Training

```bash
# Train the model
python train_video_segformer.py
```

## Configuration Details

### Model Architecture

The model uses:

- **Backbone**: MixVisionTransformer (SegFormer)
- **Decode Head**: TV3SHead_shift_city (temporal Mamba blocks)
- **Segmentor**: EncoderDecoder_clips (video-compatible)

### Temporal Configuration

The temporal processing is configured via the `dilation` parameter:

- `dilation=[-9, -6, -3]`: Uses 9 frames (current ± 9, ±6, ±3)
- Each training sample loads a temporal clip of frames
- Frames are processed together through the temporal Mamba blocks

### Dataset Configuration

The `CityscapesDataset_clips` class:

- Loads Cityscapes frames with naming: `{city}_{sequence}_{frame}_leftImg8bit.png`
- Creates temporal clips automatically
- Supports both training and validation modes

## File Structure

```
segformer_pytorch/
├── local_configs/segformer/
│   └── segformer_cityscapes_video.py    # Video training config
├── organize_cityscapes_video.py         # Data organization script
├── train_video_segformer.py             # Training script
├── TV3S/                                # TV3S components
│   ├── mmseg/models/
│   │   ├── decode_heads/tv3s_head.py
│   │   └── segmentors/encoder_decoder_clips.py
│   └── utils/datasets/cityscapes.py
└── dataset/                             # Your organized data
    ├── leftImg8bit/
    └── gtFine/
```

## Expected Data Format

### Input Frames

Your Cityscapes video frames should be named as:

```
stuttgart_00_000000_000001_leftImg8bit.png
stuttgart_00_000000_000002_leftImg8bit.png
stuttgart_00_000000_000003_leftImg8bit.png
...
```

### Annotations

Corresponding annotation files:

```
stuttgart_00_000000_000001_gtFine_labelTrainIds.png
stuttgart_00_000000_000002_gtFine_labelTrainIds.png
stuttgart_00_000000_000003_gtFine_labelTrainIds.png
...
```

## Training Parameters

Key parameters you can adjust:

```python
# In segformer_cityscapes_video.py
dilation = [-9, -6, -3]          # Temporal window size
samples_per_gpu = 1              # Batch size (keep small for video)
max_epochs = 160                 # Training epochs
crop_size = (1024, 1024)         # Input resolution
```

## Troubleshooting

### Common Issues

1. **Import Errors**

   - Ensure TV3S is properly installed
   - Check that all dependencies are available
   - Verify Python path includes TV3S directory

2. **Data Not Found**

   - Run the data organization script first
   - Check that file paths in config match your data location
   - Verify Cityscapes naming convention

3. **Memory Issues**

   - Reduce batch size (`samples_per_gpu`)
   - Decrease temporal window (`dilation`)
   - Use smaller crop size

4. **CUDA Errors**

   - Ensure CUDA is properly installed
   - Check GPU memory availability
   - Try CPU training first

### Debug Commands

```bash
# Check data organization
python organize_cityscapes_video.py --input_dir /path/to/frames --output_dir dataset

# Test configuration loading
python -c "from mmengine import Config; cfg = Config.fromfile('local_configs/segformer/segformer_cityscapes_video.py'); print('Config loaded successfully')"

# Check TV3S imports
python -c "import sys; sys.path.append('TV3S'); from mmseg.models.decode_heads.tv3s_head import TV3SHead_shift_city; print('TV3S imported successfully')"
```

## Performance Notes

- **Temporal Window**: Larger windows (more frames) improve temporal consistency but increase memory usage
- **Batch Size**: Keep small (1-2) due to temporal dimension
- **Resolution**: 1024x1024 provides good balance of quality and speed
- **GPU Memory**: Video processing requires more memory than image segmentation

## Citation

If you use this implementation, please cite:

```bibtex
@article{segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  journal={arXiv preprint arXiv:2105.15203},
  year={2021}
}

@article{tv3s,
  title={TV3S: Temporal Video Segmentation with Mamba-SSM},
  author={Your TV3S Authors},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project follows the same license as the original SegFormer and TV3S implementations.
