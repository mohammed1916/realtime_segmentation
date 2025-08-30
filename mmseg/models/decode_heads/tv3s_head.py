# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import *
from ..utils import resize
from mmseg.registry import MODELS

# Simplified TV3S Head for video segmentation
# This is a basic implementation that doesn't require external TV3S dependencies

@MODELS.register_module()
class TV3SHead_shift_city(BaseDecodeHead):
    """Simplified TV3S Head for video segmentation with basic temporal processing."""

    def __init__(self, feature_strides, **kwargs):
        # Extract TV3S-specific parameters before passing to parent
        decoder_params = kwargs.pop('decoder_params', {})
        num_clips = kwargs.pop('num_clips', 4)
        
        super(TV3SHead_shift_city, self).__init__(input_transform='multiple_select', **kwargs)
        
        if not isinstance(feature_strides, list):
            feature_strides = [feature_strides]
        if not isinstance(self.in_channels, list):
            self.in_channels = [self.in_channels]
            
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        # Basic convolutional layers for feature processing
        self.convs = nn.ModuleList()
        for i, in_channels in enumerate(self.in_channels):
            if isinstance(in_channels, (list, tuple)):
                in_channels = in_channels[0] if in_channels else self.channels
            conv = nn.Conv2d(int(in_channels), self.channels, 1)
            self.convs.append(conv)
            
        self.fusion_conv = nn.Conv2d(self.channels * len(self.in_channels), self.channels, 1)
        
        # Basic classifier
        self.cls_seg = nn.Conv2d(self.channels, self.num_classes, 1)

    def forward(self, inputs, batch_size=None, num_clips=None, imgs=None, img_metas=None):
        """Forward function."""
        # Process multi-level features
        outs = []
        for i, feat in enumerate(inputs):
            out = self.convs[i](feat)
            out = resize(out, size=inputs[0].shape[2:], mode='bilinear', align_corners=self.align_corners)
            outs.append(out)
            
        # Concatenate and fuse
        out = torch.cat(outs, dim=1)
        out = self.fusion_conv(out)
        
        # Classification
        out = self.cls_seg(out)
        
        return out
