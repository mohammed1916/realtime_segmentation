# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
import torch
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import PixelData

from mmseg.registry import TRANSFORMS
from mmseg.structures import SegDataSample


@TRANSFORMS.register_module()
class PackSegInputs(BaseTransform):
    """Pack the inputs data for clip-based segmentation."""

    def __init__(self,
                 meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        # Alias gt_semantic_seg → gt_seg_map for consistency
        if 'gt_semantic_seg' in results and 'gt_seg_map' not in results:
            results['gt_seg_map'] = results['gt_semantic_seg']

        packed_results = dict()

        # Handle list of frames
        if 'img' in results:
            imgs = results['img']
            if isinstance(imgs, list):
                tensor_list = []
                for img in imgs:
                    # Support numpy arrays (H, W, C) and torch tensors
                    if isinstance(img, np.ndarray):
                        if len(img.shape) < 3:
                            img = np.expand_dims(img, -1)
                        img = img.transpose(2, 0, 1)  # HWC → CHW
                        tensor_list.append(to_tensor(img).contiguous())
                    elif torch.is_tensor(img):
                        # If already CHW, keep as-is; otherwise permute HWC -> CHW
                        if img.dim() == 3 and img.shape[0] in (1, 3):
                            tensor_list.append(img.contiguous())
                        elif img.dim() == 3:
                            tensor_list.append(img.permute(2, 0, 1).contiguous())
                        else:
                            # fallback: convert to numpy then to tensor
                            _img = img.numpy()
                            if _img.ndim < 3:
                                _img = np.expand_dims(_img, -1)
                            _img = _img.transpose(2, 0, 1)
                            tensor_list.append(to_tensor(_img).contiguous())

                # Take the last frame to make it 3D
                img_tensor = tensor_list[-1]  # shape: (C, H, W)
                packed_results['inputs'] = img_tensor
            else:
                img = imgs
                if isinstance(img, np.ndarray):
                    if len(img.shape) < 3:
                        img = np.expand_dims(img, -1)
                    img = img.transpose(2, 0, 1)
                    packed_results['inputs'] = to_tensor(img).contiguous()
                elif torch.is_tensor(img):
                    if img.dim() == 3 and img.shape[0] in (1, 3):
                        packed_results['inputs'] = img.contiguous()
                    elif img.dim() == 3:
                        packed_results['inputs'] = img.permute(2, 0, 1).contiguous()
                    else:
                        _img = img.numpy()
                        if _img.ndim < 3:
                            _img = np.expand_dims(_img, -1)
                        _img = _img.transpose(2, 0, 1)
                        packed_results['inputs'] = to_tensor(_img).contiguous()

        # Build data sample
        data_sample = SegDataSample()
        gt_map = results.get('gt_seg_map', None)
        if gt_map is not None:
            if isinstance(gt_map, list):  # clip-style masks
                gt_map = gt_map[-1]       # usually take last frame’s label
            data = to_tensor(gt_map[None, ...].astype(np.int64))
            data_sample.gt_sem_seg = PixelData(data=data)

        # Meta info
        img_meta = {k: results[k] for k in self.meta_keys if k in results}
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results
