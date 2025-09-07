# Copyright (c) OpenMMLab. All rights reserved.
from .formatting import PackSegInputs
from .loading import (
    LoadAnnotations, LoadBiomedicalAnnotation,
    LoadBiomedicalData, LoadBiomedicalImageFromFile,
    LoadDepthAnnotation, LoadImageFromNDArray,
    LoadMultipleRSImageFromFile, LoadSingleRSImageFromFile, DefaultFormatBundle_clips
)

# yapf: disable
from .transforms import (
    Albu, BioMedical3DPad, BioMedical3DRandomCrop,
    BioMedical3DRandomFlip, BioMedicalGaussianBlur,
    BioMedicalGaussianNoise, BioMedicalRandomGamma,
    ConcatCDInput, GenerateEdge, RandomCutOut,
    RandomDepthMix, RandomMosaic, RandomRotFlip,
    ResizeShortestEdge, ResizeToMultiple
)

from .transforms_clips import (
    AlignedResize,
    AlignedResize_clips,
    Resize,                     # prefer clips version
    RandomFlip,                 # prefer clips version
    RandomFlip_clips,
    Pad,
    Pad_clips,
    Pad_clips2,
    Normalize,                  # prefer clips version
    Normalize_clips,
    Normalize_clips2,
    Rerange,
    CLAHE,
    RandomCrop,                 # prefer clips version
    RandomCrop_clips,
    CenterCrop,
    RandomRotate,               # prefer clips version
    RGB2Gray,
    AdjustGamma,
    MaillaryHack,
    SegRescale,
    PhotoMetricDistortion,      # prefer clips version
    PhotoMetricDistortion_clips,
    PhotoMetricDistortion_clips2,
)
# yapf: enable

__all__ = [
    # loaders
    'LoadAnnotations', 'LoadBiomedicalAnnotation', 'LoadBiomedicalData',
    'LoadBiomedicalImageFromFile', 'LoadImageFromNDArray',
    'LoadMultipleRSImageFromFile', 'LoadSingleRSImageFromFile',
    'LoadDepthAnnotation',
    # biomedical
    'BioMedical3DRandomCrop', 'BioMedical3DPad', 'BioMedical3DRandomFlip',
    'BioMedicalGaussianNoise', 'BioMedicalGaussianBlur', 'BioMedicalRandomGamma',
    # utils
    'GenerateEdge', 'RandomCutOut', 'RandomMosaic', 'RandomRotFlip',
    'ConcatCDInput', 'RandomDepthMix',
    # resizing
    'AlignedResize', 'AlignedResize_clips', 'Resize', 'ResizeShortestEdge',
    'ResizeToMultiple',
    # padding + normalize
    'Pad', 'Pad_clips', 'Pad_clips2',
    'Normalize', 'Normalize_clips', 'Normalize_clips2',
    # augmentations (clips-preferred)
    'RandomFlip', 'RandomFlip_clips',
    'RandomCrop', 'RandomCrop_clips',
    'RandomRotate',
    'PhotoMetricDistortion', 'PhotoMetricDistortion_clips', 'PhotoMetricDistortion_clips2',
    # misc
    'Rerange', 'CLAHE', 'RGB2Gray', 'AdjustGamma',
    'MaillaryHack', 'SegRescale',
    # packer
    'PackSegInputs', 'Albu', 'DefaultFormatBundle_clips',
]
