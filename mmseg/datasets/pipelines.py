# Compatibility shim: expose Compose where older code expects it.
try:
    from mmseg.datasets.transforms.transforms import Compose
except Exception:
    try:
        from mmseg.datasets.transforms import Compose
    except Exception:
        # Minimal fallback Compose
        def Compose(transforms):
            def _apply(x):
                for t in transforms:
                    x = t(x)
                return x
            return _apply

__all__ = ['Compose']
