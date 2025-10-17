"""Register minimal '_clips' transforms as no-ops to satisfy pipelines when
the full TV3S transform shim isn't available.

This file avoids importing the heavy tv3s_utils module and only injects
placeholder transforms into the TRANSFORMS registry if those names are
missing. They return the results dict unchanged so the pipeline can be
constructed and executed without TypeError during transform construction.
"""
from typing import Any, Dict

try:
    from mmseg.registry import TRANSFORMS
except Exception:
    try:
        from mmengine.registry import TRANSFORMS
    except Exception:
        TRANSFORMS = None


class _NoOpClipTransform:
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if results is None:
            return {}
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(noop)"


def _inject_noop_names():
    if TRANSFORMS is None:
        return
    names = [
        'RandomCrop_clips', 'RandomFlip_clips', 'PhotoMetricDistortion_clips',
        'Normalize_clips', 'Pad_clips', 'Pad_clips2', 'DefaultFormatBundle_clips',
        'ImageToTensor_clips', 'Normalize_clips2', 'Normalize_clips2_noop',
        'AlignedResize_clips', 'Resize_clips'
    ]
    moddict = getattr(TRANSFORMS, '_module_dict', None)
    if not isinstance(moddict, dict):
        return
    for name in names:
        if name not in moddict:
            moddict[name] = _NoOpClipTransform


_inject_noop_names()
