"""Lightweight shim for mmseg.ops when compiled ops are unavailable.

This provides a minimal `resize` helper used by TV3S modules. It's not a
drop-in replacement for the full compiled opset, but it allows the code to
import and run basic resizing operations using mmcv fallbacks.
"""
from typing import Any, Tuple

import mmcv

__all__ = ["resize"]


def resize(img: Any,
           scale,
           return_scale: bool = False,
           interpolation: str = 'bilinear',
           keep_ratio: bool = True) -> Any:
    """Resize helper that delegates to mmcv.

    Args:
        img: image array (H,W,C) or similar.
        scale: target scale (tuple) or scalar.
        return_scale: whether to also return the scale factor.
        interpolation: interpolation mode forwarded to mmcv.
        keep_ratio: whether to preserve aspect ratio (uses imrescale).

    Returns:
        If return_scale is False: resized image.
        If return_scale is True: (resized_image, scale) where `scale` is
        whatever mmcv returns (a float or array).
    """
    if keep_ratio:
        return mmcv.imrescale(img, scale, return_scale=return_scale, interpolation=interpolation)
    return mmcv.imresize(img, scale, return_scale=return_scale, interpolation=interpolation)
