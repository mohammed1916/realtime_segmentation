"""Compatibility shim: re-export transforms from the tv3s_utils shim so
that worker processes (spawned on Windows) can import the module by name.

This file ensures the transforms live under the import path
`mmseg.datasets.transforms` which multiprocessing child processes can import
when unpickling pipeline objects.
"""
try:
    # Prefer the repo-local tv3s shim which registers the TV3S transforms.
    from tv3s_utils.utils.datasets.transforms import *  # noqa: F401,F403
except Exception as e:  # pragma: no cover - keep simple fallback
    # Provide a helpful error if import fails instead of hiding it.
    raise ImportError(
        "Failed to import tv3s transforms shim. Ensure PYTHONPATH includes the repo root."
    ) from e
