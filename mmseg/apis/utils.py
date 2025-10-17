# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
from typing import Sequence, Union

import numpy as np
import torch
from mmseg.datasets.pipelines import Compose  # kept for compatibility
from mmseg.datasets import PIPELINES as DATA_PIPELINES
from mmengine.model import BaseModel
# Ensure local mmseg dataset pipeline modules are imported so custom pipeline
# classes (e.g. Collect, DefaultFormatBundle) are registered with the
# TRANSFORMS/PIPELINES registry that mmengine uses.
import mmseg.datasets.pipelines  # noqa: F401

ImageType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]


def _preprare_data(imgs: ImageType, model: BaseModel):

    cfg = model.cfg
    for t in cfg.test_pipeline:
        if t.get('type') == 'LoadAnnotations':
            cfg.test_pipeline.remove(t)

    is_batch = True
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
        is_batch = False

    if isinstance(imgs[0], np.ndarray):
        cfg.test_pipeline[0]['type'] = 'LoadImageFromNDArray'

    # TODO: Consider using the singleton pattern to avoid building
    # a pipeline for each inference
    # Some older configs use 'img_scale' inside 'MultiScaleFlipAug'. Newer
    # mmengine expects 'scale'. Build a plain list of dicts, convert legacy
    # keys, and pass that to Compose to avoid unexpected keyword errors.
    pipeline_transforms = []
    for t in cfg.test_pipeline:
        try:
            td = dict(t)
        except Exception:
            pipeline_transforms.append(t)
            continue

        if td.get('type') == 'MultiScaleFlipAug':
            # Extract inner transforms and adapt them for direct Compose usage.
            inner = td.get('transforms', [])
            # If img_scale present, prefer it; otherwise accept 'scale'
            scale_val = td.pop('img_scale', td.pop('scale', None))
            flip_flag = td.pop('flip', None)

            for it in inner:
                itd = dict(it)
                # Inject scale into Resize if provided by the wrapper
                if itd.get('type') == 'Resize' and scale_val is not None:
                    # older Resize used 'img_scale'; newer expects 'scale'
                    # but many local Resize accept 'img_scale' too — set both
                    itd['scale'] = scale_val
                    itd.pop('img_scale', None)
                if itd.get('type') == 'RandomFlip' and flip_flag is not None:
                    # If wrapper set flip=False, disable flipping deterministically
                    if flip_flag is False:
                        # set probability to 0 to avoid flipping
                        itd['prob'] = 0.0
                pipeline_transforms.append(itd)
        else:
            # convert legacy img_scale if present
            if 'img_scale' in td:
                td['scale'] = td.pop('img_scale')
            pipeline_transforms.append(td)

    # Build pipeline using the repository's PIPELINES registry so custom
    # pipeline modules (Collect, DefaultFormatBundle, etc.) are located.
    built = []
    for item in pipeline_transforms:
        if callable(item):
            built.append(item)
        elif isinstance(item, dict) and 'type' in item:
            ttype = item['type']
            kwargs = {k: v for k, v in item.items() if k != 'type'}
            try:
                built.append(DATA_PIPELINES.build(dict(type=ttype, **kwargs)))
            except Exception:
                # If build fails, fall back to a no-op to keep pipeline shape
                built.append(lambda x: x)
        else:
            built.append(lambda x: x)

    # Ensure final pipeline packs inputs and data_samples in expected keys.
    try:
        from mmseg.datasets.transforms.formatting import PackSegInputs
        built.append(PackSegInputs())
    except Exception:
        # best-effort: if PackSegInputs isn't available, continue
        pass

    def pipeline(results):
        for tr in built:
            results = tr(results)
        return results

    data = defaultdict(list)
    for img in imgs:
        if isinstance(img, np.ndarray):
            data_ = dict(img=img)
        else:
            data_ = dict(img_path=img)
        data_ = pipeline(data_)
        # PackSegInputs (or other pipeline steps) may return tensors directly
        inp = data_.get('inputs', None)
        ds = data_.get('data_samples', None)
        # ensure inputs is a flat list of tensors (one element per image)
        if inp is not None:
            if isinstance(inp, (list, tuple)):
                # if pipeline returned multiple inputs, extend with them
                for x in inp:
                    data['inputs'].append(x)
            else:
                data['inputs'].append(inp)
        # ensure data_samples is a list of SegDataSample objects
        if ds is not None:
            if isinstance(ds, (list, tuple)):
                for x in ds:
                    data['data_samples'].append(x)
            else:
                data['data_samples'].append(ds)

    # If pipeline returned per-image inputs as a list, convert to a
    # single batched torch.Tensor (N, C, H, W) which EncoderDecoder
    # and its slide_inference expect. Prefer cat/stack and move to
    # the model device when possible.
    if len(data.get('inputs', [])) > 0:
        inputs_list = data['inputs']
        torch_inputs = []
        for x in inputs_list:
            # numpy -> torch
            if isinstance(x, np.ndarray):
                t = torch.from_numpy(x)
            else:
                t = x
            # skip non-tensor entries
            if not isinstance(t, torch.Tensor):
                continue
            # if single image without batch dim, ensure it has C,H,W
            if t.dim() == 3:
                # assume (C,H,W)
                pass
            elif t.dim() == 4:
                # already batched per-entry, remove leading batch dim
                t = t.squeeze(0) if t.size(0) == 1 else t
            else:
                # try to make it (C,H,W)
                try:
                    t = t.view(t.shape[-3], t.shape[-2], t.shape[-1])
                except Exception:
                    pass
            torch_inputs.append(t)

        if len(torch_inputs) > 0:
            try:
                batched = torch.stack(torch_inputs, dim=0)
            except Exception:
                try:
                    batched = torch.cat([ti.unsqueeze(0) if ti.dim()==3 else ti for ti in torch_inputs], dim=0)
                except Exception:
                    batched = torch.tensor([ti.cpu().numpy() if isinstance(ti, torch.Tensor) else ti for ti in torch_inputs])

            # move to model device if possible
            try:
                device = next(model.parameters()).device
                batched = batched.to(device)
            except Exception:
                pass

            data['inputs'] = batched

    return data, is_batch
