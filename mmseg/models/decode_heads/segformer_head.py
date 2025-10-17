# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize


@MODELS.register_module(name='SegFormerHead')
class SegformerHead(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        # Some configs (e.g. original SegFormer configs) include 'feature_strides'
        # which BaseDecodeHead may not accept in this local mmseg version. Remove
        # it to avoid unexpected keyword argument errors when initializing.
        kwargs.pop('feature_strides', None)
        # Older configs may include nested decoder params for SegFormer (e.g.
        # decoder_params=dict(embed_dim=256)). These are not parameters for
        # BaseDecodeHead and should be removed before initialization.
        kwargs.pop('decoder_params', None)
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))

        out = self.cls_seg(out)

        return out

# Also register the head under the class-name spelling 'SegformerHead' so configs
# that use either 'SegFormerHead' or 'SegformerHead' (different capitalisation)
# will both resolve against the registry.
try:
    MODELS.register_module(name='SegformerHead')(SegformerHead)
except Exception:
    # If double-registration fails for any reason, ignore to preserve backward-compat
    pass
