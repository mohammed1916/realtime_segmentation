# Compatibility shim: expose Compose where older code expects it.
try:
    # Prefer mmcv transforms.Compose when available
    from mmcv.transforms import Compose
except Exception:
    try:
        # older mmcv may expose Compose at top-level
        from mmcv import Compose
    except Exception:
        # Safer fallback: build simple callables from pipeline dicts where possible.
        def Compose(pipeline):
            built = []
            for item in pipeline:
                if callable(item):
                    built.append(item)
                elif isinstance(item, dict) and 'type' in item:
                    ttype = item['type']
                    kwargs = {k: v for k, v in item.items() if k != 'type'}
                    # Try to locate transform class in mmseg.datasets.transforms
                    try:
                        mod = __import__('mmseg.datasets.transforms', fromlist=[ttype])
                        cls = getattr(mod, ttype)
                        built.append(cls(**kwargs))
                    except Exception:
                        # Last resort: use a no-op callable to keep pipeline shape
                        built.append(lambda x: x)
                else:
                    built.append(lambda x: x)

            def _apply(results):
                for tr in built:
                    results = tr(results)
                return results

            return _apply

__all__ = ['Compose']
