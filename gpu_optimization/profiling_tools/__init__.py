"""
GPU Profiling Tools for SegFormer Performance Analysis
"""

__version__ = "1.0"

try:
    from .pytorch_profiler import profile_segformer, print_profiler_summary
    from .roofline_benchmark import RooflineAnalyzer
except ImportError:
    pass
