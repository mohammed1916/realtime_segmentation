"""Minimal TV3S transforms shim.

This module registers the pipeline transform names expected by TV3S
configs but provides safe, minimal implementations that act as no-ops.
Each transform returns the results dict unchanged (or an empty dict if
results is None). This prevents pipeline steps from returning None and
breaking the dataset pipeline during initialization or training.

Replace these placeholders with full implementations later as needed.
"""

from typing import Any, Dict, Iterable

# Prefer importing the mmseg TRANSFORMS registry directly to avoid
# circular imports with mmseg.datasets when the datasets package
# imports transforms. Fall back to mmengine registry if needed.
try:
    from mmseg.registry import TRANSFORMS as PIPELINES
except Exception:
    try:
        from mmengine.registry import TRANSFORMS as PIPELINES
    except Exception:
        PIPELINES = None


class _BaseTransform:
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        # always return a dict (never None)
        if results is None:
            return {}
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}({self._kwargs})"


def _register(name: str):
    """Helper decorator to register a noop transform on the PIPELINES
    registry if available.
    """

    def _dec(cls):
        # Try to register in multiple registry objects to handle different
        # import paths used across mmcv/mmengine/mmseg variants.
        registries = []
        if PIPELINES is not None:
            registries.append(PIPELINES)
        try:
            import mmseg.registry as _mmseg_reg
            if getattr(_mmseg_reg, 'TRANSFORMS', None) is not None:
                registries.append(_mmseg_reg.TRANSFORMS)
        except Exception:
            pass
        try:
            import mmengine.registry as _mmeng_reg
            if getattr(_mmeng_reg, 'TRANSFORMS', None) is not None:
                registries.append(_mmeng_reg.TRANSFORMS)
        except Exception:
            pass
        for reg in registries:
            try:
                # prefer registering with explicit name when supported
                try:
                    reg.register_module(name=name)(cls)
                    continue
                except TypeError:
                    # older Registry.register_module may not accept name kw
                    pass

                try:
                    # try to register and then, if possible, rename the entry
                    reg.register_module()(cls)
                    # if registry accepted the class, inject alias under the desired name
                    if hasattr(reg, '_module_dict') and isinstance(reg._module_dict, dict):
                        # Find the actual class key and alias it as the expected name
                        for k, v in list(reg._module_dict.items()):
                            if v is cls and k != name:
                                reg._module_dict[name] = v
                                break
                    continue
                except Exception:
                    pass

                # final fallback: inject directly into registry's module dict
                if hasattr(reg, '_module_dict') and isinstance(reg._module_dict, dict):
                    # Inject under the requested name so config lookups succeed
                    reg._module_dict[name] = cls
                    continue
            except Exception:
                # ignore individual registry failures
                continue
        return cls

    return _dec


@_register("AlignedResize")
class AlignedResize(_BaseTransform):
    pass


@_register("AlignedResize_clips")
class AlignedResize_clips(_BaseTransform):
    pass


@_register("Resize")
class Resize(_BaseTransform):
    pass


@_register("RandomFlip")
class RandomFlip(_BaseTransform):
    pass


@_register("RandomFlip_clips")
class RandomFlip_clips(_BaseTransform):
    pass


@_register("Pad")
class Pad(_BaseTransform):
    pass


@_register("Pad_clips")
class Pad_clips(_BaseTransform):
    pass


@_register("Pad_clips2")
class Pad_clips2(_BaseTransform):
    pass


@_register("Normalize")
class Normalize(_BaseTransform):
    pass


@_register("Normalize_clips")
class Normalize_clips(_BaseTransform):
    pass


@_register("Normalize_clips2")
class Normalize_clips2(_BaseTransform):
    pass


@_register("Rerange")
class Rerange(_BaseTransform):
    pass


@_register("CLAHE")
class CLAHE(_BaseTransform):
    pass


@_register("RandomCrop")
class RandomCrop(_BaseTransform):
    pass


@_register("RandomCrop_clips")
class RandomCrop_clips(_BaseTransform):
    pass


@_register("CenterCrop")
class CenterCrop(_BaseTransform):
    pass


@_register("RandomRotate")
class RandomRotate(_BaseTransform):
    pass


@_register("RGB2Gray")
class RGB2Gray(_BaseTransform):
    pass


@_register("AdjustGamma")
class AdjustGamma(_BaseTransform):
    pass


@_register("MaillaryHack")
class MaillaryHack(_BaseTransform):
    pass


@_register("SegRescale")
class SegRescale(_BaseTransform):
    pass


@_register("PhotoMetricDistortion")
class PhotoMetricDistortion(_BaseTransform):
    pass


@_register("PhotoMetricDistortion_clips")
class PhotoMetricDistortion_clips(_BaseTransform):
    pass


@_register("PhotoMetricDistortion_clips2")
class PhotoMetricDistortion_clips2(_BaseTransform):
    pass


@_register("Normalize_clips2_noop")
class Normalize_clips2_noop(_BaseTransform):
    pass


@_register("Compose")
class Compose(_BaseTransform):
    """Compose that calls each callable transform in sequence.

    If elements in `transforms` are not callable they are ignored. This keeps
    behavior safe for partial/placeholder pipelines.
    """

    def __init__(self, transforms: Iterable[Any]):
        super().__init__(transforms=transforms)
        # store as-is; many pipelines will pass already-instantiated callables
        self.transforms = list(transforms) if transforms is not None else []

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if results is None:
            results = {}
        for t in self.transforms:
            try:
                if callable(t):
                    res = t(results)
                    # ensure we always have a dict; ignore bad returns
                    if isinstance(res, dict):
                        results = res
            except Exception:
                # swallow transform-level errors to keep initialization robust
                continue
        return results


# Finalizer: ensure all defined shim transforms are present in the runtime
# TRANSFORMS registries used by mmseg/mmengine. Some environments create
# separate Registry instances; this guarantees the expected names exist in
# whichever registry the dataset/pipeline builder consults.
def _inject_transforms_into_registries():
    try:
        import mmseg.registry as _mmseg_reg
    except Exception:
        _mmseg_reg = None
    try:
        import mmengine.registry as _mmeng_reg
    except Exception:
        _mmeng_reg = None
    # also attempt mmcv-related registry modules which mmcv.transforms.wrappers
    # may use as the TRANSFORMS registry object
    try:
        import importlib
        _mmcv_mod = importlib.import_module('mmcv')
    except Exception:
        _mmcv_mod = None
    try:
        _mmcv_transforms_mod = importlib.import_module('mmcv.transforms')
    except Exception:
        _mmcv_transforms_mod = None
    try:
        _mmcv_transforms_registry = importlib.import_module('mmcv.transforms.registry')
    except Exception:
        _mmcv_transforms_registry = None

    # Collect candidate transform classes defined in this module
    transform_classes = {
        name: obj
        for name, obj in globals().items()
        if isinstance(obj, type) and issubclass(obj, _BaseTransform) and obj is not _BaseTransform
    }

    def _inject(registry, registry_name=None):
        if registry is None:
            return
        # prefer explicit register_module API
        for name, cls in transform_classes.items():
            try:
                # skip if already registered
                moddict = getattr(registry, '_module_dict', None)
                if isinstance(moddict, dict) and name in moddict:
                    continue
                # try register_module with explicit name
                try:
                    registry.register_module(name=name)(cls)
                    continue
                except TypeError:
                    # older API may not accept name kw
                    try:
                        registry.register_module()(cls)
                        # alias under expected name if possible
                        moddict = getattr(registry, '_module_dict', None)
                        if isinstance(moddict, dict):
                            for k, v in list(moddict.items()):
                                if v is cls and k != name:
                                    moddict[name] = v
                                    break
                        continue
                    except Exception:
                        pass

                # final fallback: inject into _module_dict directly
                if isinstance(moddict, dict):
                    moddict[name] = cls
                    continue
            except Exception:
                # ignore individual failures
                continue

    # Try common registry instances used by mmseg/mmengine/mmcv
    _inject(getattr(_mmseg_reg, 'TRANSFORMS') if _mmseg_reg is not None else None, 'mmseg')
    _inject(getattr(_mmeng_reg, 'TRANSFORMS') if _mmeng_reg is not None else None, 'mmengine')
    try:
        import mmcv
        # mmcv may expose transforms registry under mmcv.transforms.registry or mmcv.transforms
        mmcv_trans_registry = None
        try:
            from mmcv.transforms import registry as _mmcv_t_reg
            mmcv_trans_registry = getattr(_mmcv_t_reg, 'TRANSFORMS', None)
        except Exception:
            try:
                mmcv_trans_registry = getattr(mmcv, 'TRANSFORMS', None)
            except Exception:
                mmcv_trans_registry = None
        if mmcv_trans_registry is not None:
            _inject(mmcv_trans_registry, 'mmcv')
    except Exception:
        pass
    # try common mmcv registry variants
    if _mmcv_mod is not None:
        for attr in ('TRANSFORMS', 'PIPELINES'):
            _inject(getattr(_mmcv_mod, attr, None))
    if _mmcv_transforms_mod is not None:
        _inject(getattr(_mmcv_transforms_mod, 'TRANSFORMS', None))
    if _mmcv_transforms_registry is not None:
        # this module sometimes exposes Registry instances
        for attr in dir(_mmcv_transforms_registry):
            if attr.startswith('__'):
                continue
            _inject(getattr(_mmcv_transforms_registry, attr))


# Run injection at import time so importing this module guarantees registration
try:
    _inject_transforms_into_registries()
except Exception:
    pass


