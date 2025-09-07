import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor

@MODELS.register_module()
class EncoderDecoder_clips(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder_clips typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 val_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 pretrained=None,
                 init_cfg=None):
        # Initialize BaseSegmentor with data_preprocessor and init_cfg so
        # mmengine can pass model-level preprocessing configuration.
        super(EncoderDecoder_clips, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        # mmengine.BaseModel.init_weights() expects no arguments in some
        # installed versions, so call without passing `pretrained` to avoid
        # a mismatch. The backbone can still load the pretrained weights
        # explicitly below.
        try:
            super(EncoderDecoder_clips, self).init_weights()
        except TypeError:
            # fallback for versions expecting an argument
            super(EncoderDecoder_clips, self).init_weights(pretrained)
        # Some backbone implementations accept a `pretrained` arg, others do
        # not. Call robustly to support both variants.
        try:
            self.backbone.init_weights(pretrained=pretrained)
        except TypeError:
            self.backbone.init_weights()
        self.decode_head.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self,
                      inputs: Tensor,
                      batch_img_metas: Optional[List[dict]] = None,
                      batch_size: int = 1,
                      num_clips: int = 1) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input.

        Accepts optional ``batch_size`` and ``num_clips`` to support older
        TV3S-style decode heads that expect these parameters during test-time
        forward. If the decode head implements ``forward_test`` we forward
        the extra args, otherwise fall back to the standard ``predict`` call.
        """
        x = self.extract_feat(inputs)
        # Capture original metainfo for one-time debugging if it's malformed
        orig_batch_img_metas = batch_img_metas
        if not getattr(self, '_logged_bad_batch_img_metas', False):
            problematic = False
            try:
                # consider None, non-dict, or missing pad_shape as problematic
                if orig_batch_img_metas is None:
                    problematic = True
                else:
                    for e in orig_batch_img_metas:
                        if e is None or not isinstance(e, dict) or 'pad_shape' not in e:
                            problematic = True
                            break
            except Exception:
                problematic = True
            if problematic:
                # Log the raw problematic object once and raise to make it visible
                print_log(
                    f"ENCODE_DECODE DEBUG: bad batch_img_metas captured: {orig_batch_img_metas}",
                    logger='mmseg',
                    level=logging.WARNING)
                # mark so we only log/assert once per model instance
                self._logged_bad_batch_img_metas = True
                # raise assertion to capture stack and halt (one-time debug)
                raise AssertionError(f"Bad batch_img_metas captured in encode_decode: {orig_batch_img_metas}")
        # Ensure batch_img_metas is present and contains expected keys used
        # by decode heads (pad_shape, ori_shape, etc.). If not provided by
        # the caller, synthesize minimal metadata from the input tensor
        # shape so downstream predict_by_feat() calls don't fail.
        if batch_img_metas is None:
            h, w = inputs.shape[2], inputs.shape[3]
            batch_img_metas = [{
                'pad_shape': (h, w),
                'ori_shape': (h, w),
                'scale_factor': 1.0,
                'flip': False
            }]
        else:
            # Defensive: ensure first element has pad_shape/ori_shape
            try:
                if not batch_img_metas or batch_img_metas[0] is None:
                    h, w = inputs.shape[2], inputs.shape[3]
                    batch_img_metas = [{
                        'pad_shape': (h, w),
                        'ori_shape': (h, w),
                        'scale_factor': 1.0,
                        'flip': False
                    }]
            except Exception:
                # If anything goes wrong, still synthesize safe default
                h, w = inputs.shape[2], inputs.shape[3]
                batch_img_metas = [{
                    'pad_shape': (h, w),
                    'ori_shape': (h, w),
                    'scale_factor': 1.0,
                    'flip': False
                }]
        # Make sure each element of batch_img_metas is a dict with the
        # keys decode heads expect (pad_shape, ori_shape). Replace any
        # None or non-dict entries with safe defaults derived from input
        # spatial shape.
        try:
            h, w = inputs.shape[2], inputs.shape[3]
        except Exception:
            h, w = 0, 0
        sanitized = []
        for entry in batch_img_metas:
            if not isinstance(entry, dict) or entry is None:
                sanitized.append({
                    'pad_shape': (h, w),
                    'ori_shape': (h, w),
                    'scale_factor': 1.0,
                    'flip': False
                })
            else:
                if 'pad_shape' not in entry or entry['pad_shape'] is None:
                    entry['pad_shape'] = (h, w)
                if 'ori_shape' not in entry or entry['ori_shape'] is None:
                    entry['ori_shape'] = (h, w)
                if 'scale_factor' not in entry:
                    entry['scale_factor'] = 1.0
                if 'flip' not in entry:
                    entry['flip'] = False
                sanitized.append(entry)
        batch_img_metas = sanitized

        # If decode head provides a test-time forward that accepts batch_size
        # and num_clips, prefer that to preserve clip-aware logic.
        if hasattr(self.decode_head, 'forward_test'):
            seg_logits = self.decode_head.forward_test(
                x, batch_img_metas, self.test_cfg, batch_size, num_clips)
        else:
            seg_logits = self.decode_head.predict(x, batch_img_metas,
                                                  self.test_cfg)
        # One-time debug: log seg_logits.shape to detect channel/shape issues
        try:
            if not getattr(self, '_logged_seg_logits_shape', False):
                print_log(f'ENCODE_DECODE DEBUG: seg_logits.shape={getattr(seg_logits, "shape", None)}', logger='mmseg')
                self._logged_seg_logits_shape = True
        except Exception:
            pass
        return seg_logits

    # # Used for FLOP calculation
    # def forward(self, img, img_metas=None, batch_size=1, num_clips=1):
    #     """Encode images with backbone and decode into a semantic segmentation
    #     map of the same size as input."""
    #     x = self.extract_feat(img)
    #     # out = self._decode_head_forward_test(x, img_metas, batch_size, num_clips)
    #     # # print(out.shape, img.shape[2:])
    #     # out = resize(
    #     #     input=out,
    #     #     size=img.shape[2:],
    #     #     mode='bilinear',
    #     #     align_corners=self.align_corners)
    #     # return out
    #     return x
    
    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg, batch_size, num_clips, data_samples=None):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        # Prefer decode_head.forward_train if implemented (legacy TV3S
        # heads). Otherwise, fall back to BaseDecodeHead.loss which
        # computes forward + loss_by_feat.
        if hasattr(self.decode_head, 'forward_train'):
            loss_decode = self.decode_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg,
                batch_size, num_clips)
        else:
            # BaseDecodeHead.loss expects batch_data_samples (SegDataSample
            # list). We pass data_samples from the segmentor's loss() call.
            if data_samples is None:
                raise ValueError(
                    'data_samples must be provided to compute decode head loss')
            loss_decode = self.decode_head.loss(x, data_samples, self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas, batch_size, num_clips):
        """Run forward function and calculate loss for decode head in
        inference."""
        # Prefer forward_test if provided; otherwise use BaseDecodeHead.predict
        if hasattr(self.decode_head, 'forward_test'):
            seg_logits = self.decode_head.forward_test(
                x, img_metas, self.test_cfg, batch_size, num_clips)
        else:
            seg_logits = self.decode_head.predict(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        assert img.shape[0]==1
        img = torch.concat((img,img,img,img),0)
        seg_logit = self.encode_decode(img, None)


        return seg_logit

    def forward_train(self, img, img_metas, gt_semantic_seg, data_samples=None):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # Support both clip-based inputs (5D: B, N, C, H, W) and single-frame
        # inputs (4D: B, C, H, W). When single-frame, treat num_clips=1.
        if img.dim() == 5:
            batch_size, num_clips, _, h, w = img.size()
        else:
            batch_size = img.size(0)
            num_clips = 1
            _, _, h, w = img.size()

        img = img.reshape(batch_size * num_clips, -1, h, w)

        x = self.extract_feat(img)

        losses = dict()

        loss_decode = self._decode_head_forward_train(
            x, img_metas, gt_semantic_seg, batch_size, num_clips,
            data_samples=data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale, batch_size, num_clips):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta, batch_size, num_clips)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            # use torch.nn.functional.interpolate instead of undefined resize
            preds = F.interpolate(
                preds,
                size=tuple(img_meta[0]['ori_shape'][:2]),
                mode='bilinear',
                align_corners=self.align_corners)
        return preds

    def whole_inference(self, img, img_meta, rescale, batch_size, num_clips):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta, batch_size, num_clips)
        # print(seg_logit.shape)
        # print(img_meta[0]['ori_shape'][:2])
        if rescale:
            # use torch.nn.functional.interpolate instead of undefined resize
            seg_logit = F.interpolate(
                seg_logit,
                size=tuple(img_meta[0]['ori_shape'][:2]),
                mode='bilinear',
                align_corners=self.align_corners)

        return seg_logit

    def inference(self, img, img_meta, rescale, batch_size, num_clips):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        
        if not type(img_meta) is list:
            # Assumption of img_meta to have len 1
            assert len(img_meta)==1, "Check on the len here!"
            img_meta = img_meta.data[-1]   
            
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale, batch_size, num_clips)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale, batch_size, num_clips)
        # print(seg_logit.shape)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        img=torch.stack(img, dim=1)
        # print(img.shape, img_meta); exit()

        # Same handling as forward_train: support both 5D clip input and 4D
        # single-frame input after stacking.
        if img.dim() == 5:
            batch_size, num_clips, _, h, w = img.size()
        else:
            batch_size = img.size(0)
            num_clips = 1
            _, _, h, w = img.size()

        img = img.reshape(batch_size * num_clips, -1, h, w)
        # exit()
        seg_logit = self.inference(img, img_meta, rescale, batch_size, num_clips)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    # Implement abstract methods required by BaseSegmentor
    def loss(self, inputs, data_samples):
        """Wrapper to compute losses expected by BaseSegmentor API.

        Attempts to extract gt_sem_seg tensors from data_samples and calls
        the existing forward_train implementation.
        """
        # Build img_metas list if available
        # One-time debug: capture inputs shape and data_samples metainfo to
        # diagnose missing dataset metadata or malformed samples. This is
        # deliberately guarded so it only prints once per model instance.
        try:
            if not getattr(self, '_logged_loss_inputs_and_meta', False):
                try:
                    inp_shape = None
                    if isinstance(inputs, torch.Tensor):
                        inp_shape = tuple(inputs.shape)
                    elif isinstance(inputs, (list, tuple)) and len(inputs) > 0 and hasattr(inputs[0], 'shape'):
                        inp_shape = tuple(getattr(inputs[0], 'shape', None))
                    print_log(f'LOSS DEBUG: inputs.shape={inp_shape}', logger='mmseg')
                    if data_samples is None:
                        print_log('LOSS DEBUG: data_samples is None', logger='mmseg')
                    else:
                        try:
                            first_meta = getattr(data_samples[0], 'metainfo', None)
                        except Exception:
                            first_meta = None
                        print_log(f'LOSS DEBUG: data_samples[0].metainfo={first_meta}', logger='mmseg')
                except Exception:
                    pass
                self._logged_loss_inputs_and_meta = True
        except Exception:
            pass

        # Build img_metas list if available
        img_metas = None
        if data_samples is not None:
            try:
                img_metas = [ds.metainfo for ds in data_samples]
            except Exception:
                img_metas = data_samples

        # Extract gt_sem_seg tensors if present
        gt_sem_seg = None
        if data_samples is not None:
            try:
                gt_list = []
                for i, ds in enumerate(data_samples):
                    # Defensive checks to provide clear diagnostics when ground
                    # truth is missing from a data sample.
                    if not hasattr(ds, 'gt_sem_seg') or ds.gt_sem_seg is None or getattr(ds.gt_sem_seg, 'data', None) is None:
                        # Log informative context and raise a clear error.
                        try:
                            info = getattr(ds, 'metainfo', None)
                        except Exception:
                            info = None
                        print_log(f'Missing gt_sem_seg in data_samples[{i}]; metainfo={info}', logger=logging.getLogger())
                        raise ValueError('data_samples must contain gt_sem_seg PixelData; check pipeline outputs (gt_seg_map / gt_semantic_seg)')
                    gt_list.append(ds.gt_sem_seg.data)
                gt_sem_seg = torch.stack(gt_list, dim=0)
            except Exception:
                gt_sem_seg = None

        # Forward data_samples through so decode heads that rely on
        # BaseDecodeHead.loss(batch_data_samples, ...) receive the
        # necessary information (e.g., gt maps, metainfo).
        return self.forward_train(inputs, img_metas, gt_sem_seg,
                                  data_samples=data_samples)

    def predict(self, inputs, data_samples=None):
        """Predict wrapper to satisfy BaseSegmentor API.

        Uses the implemented inference/simple_test logic and postprocesses
        logits into data samples.
        """
        if data_samples is not None:
            try:
                batch_img_metas = [ds.metainfo for ds in data_samples]
            except Exception:
                batch_img_metas = data_samples
        else:
            batch_img_metas = None

        # Try to infer batch_size and num_clips from inputs if possible
        batch_size = None
        num_clips = None
        if isinstance(inputs, torch.Tensor):
            if inputs.dim() == 5:
                batch_size, num_clips = inputs.size(0), inputs.size(1)
            else:
                batch_size = inputs.size(0)

        # Use existing inference function (it handles slide/whole modes)
        seg_logits = self.inference(inputs, batch_img_metas, rescale=True, batch_size=batch_size or 1, num_clips=num_clips or 1)
        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self, inputs, data_samples=None):
        """Low-level forward that returns raw logits/tensors without postprocessing."""
        x = self.extract_feat(inputs)
        # Use decode_head.forward where available
        try:
            return self.decode_head.forward(x)
        except TypeError:
            # Some decode heads expect different signatures; try a generic call
            return self.decode_head(x)
