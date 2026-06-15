# GPU Optimization Folder Structure

## Overview

The `gpu_optimization/` folder has been reorganized to group files by **topic and purpose** rather than by file type. This makes it easier to navigate, understand dependencies, and maintain related files together.

## Folder Organization

### 📁 **cuda_libs/** - CUDA Library Optimization
Contains all files related to CUDA library integration (cuBLAS, cuDNN, cuSPARSE) and library-level optimizations.

**Files:**
- `CUDA_LIBRARIES_OPTIMIZATION.md` - Comprehensive guide to CUDA library optimizations
- `CUDNN_OPTIMIZATIONS_EXPLAINED.md` - Detailed cuDNN auto-tuning, TF32, and FP16 explanations
- `CUDA_LIBRARIES_IMPLEMENTATION_RESULTS.md` - Implementation results and metrics
- `cuda_libraries_optimization.py` - Main benchmark script (4 optimization tiers: FP32 → FP16 with 1.61x speedup)
- `cuda_libraries_impact.py` - Direct comparison WITH vs WITHOUT CUDA library optimizations (1.57x speedup)
- `cuda_libraries_optimization_results.json` - JSON results from benchmarking

**Why together:** These files form a cohesive unit documenting how CUDA libraries (not just precision) drive optimization.

---

### 📁 **profiling/** - GPU Performance Profiling
Contains profiling scripts, metrics guides, and profiler output for analyzing GPU performance.

**Files:**
- `KERNEL_PROFILING_GUIDE.md` - How to interpret L2 cache, occupancy, warp efficiency
- `PROFILER_METRICS_GUIDE.md` - Comprehensive profiler interpretation guide
- `NSIGHT_COMPUTE_WORKFLOW.md` - Complete Nsight Compute profiling workflow
- `NSIGHT_QUICK_START.md` - Quick Nsight Compute reference
- `REAL_GPU_OPTIMIZATION_SUMMARY.md` - Summary of actual GPU measurements
- `REAL_MEASURED_METRICS.md` - Analysis of theoretical vs actual metrics
- `measure_real_metrics.py` - Measures actual GPU hardware metrics (bandwidth, memory, latency)
- `profile_kernel_metrics.py` - PyTorch profiler-based kernel analysis with bottleneck identification
- `profile_and_save_outputs.py` - Generates segmentation visualizations and profiling results
- `profile_with_timm_model.py` - Profile with timm pretrained models (legacy)
- `profile_with_trained_model.py` - Profile with MMSegmentation trained models (legacy)
- `profiling_summary.py` - Profiling summary generation
- `roofline_analysis.py` - Roofline model analysis for performance characterization
- `kernel_profiling_results.json` - Kernel-level profiling data
- `real_gpu_metrics.json` - Measured GPU hardware metrics

**Why together:** All profiling-related tools, guides, and outputs grouped for easy reference when analyzing GPU performance.

**Note:** `profile_with_timm_model.py` and `profile_with_trained_model.py` are legacy files that couldn't execute due to missing dependencies (timm 0.6.7 lacked SegFormer, mmseg not installed). They document alternative profiling approaches.

---

### 📁 **verification/** - Optimization Verification
Contains scripts and documentation for verifying that optimizations actually work correctly.

**Files:**
- `VERIFICATION_SUMMARY.md` - Complete verification results and status
- `check_model_quality.py` - Analyzes whether a model is trained or randomly initialized
- `verify_optimization.py` - Verifies optimization correctness
- `verify_onnx_model.py` - Verifies ONNX model conversion and correctness
- `verify_with_real_data.py` - Verification using real Cityscapes test images

**Why together:** These tools ensure optimizations don't break model correctness or introduce errors.

---

### 📁 **inference/** - Inference and Deployment
Contains scripts for running inference and preparing models for deployment (ONNX, TensorRT, etc.).

**Files:**
- `TRAINED_MODELS_AVAILABLE.md` - Documentation of trained SegFormer-B0 models with real metrics
- `ONNX_MODEL_OPTIMIZATION.md` - ONNX/TensorRT optimization strategies
- `run_inference.py` - Standard inference runner
- `run_nsight_profile.py` - Inference with Nsight profiling integration
- `benchmark_synthetic.py` - Synthetic benchmarking (legacy)
- `gpu_profiled_optimization.py` - Inference with GPU profiling enabled
- `custom_optimizations.py` - Custom optimization implementation
- `optimize_onnx_tensorrt.py` - ONNX to TensorRT conversion and optimization

**Why together:** All inference execution methods and deployment-related optimizations grouped together.

---

### 📁 **kernels/** - Custom CUDA Kernels (Advanced)
Contains custom CUDA kernel implementations and bindings.

**Files:**
- `fused_relu_bn.cu` - Fused ReLU+BatchNorm CUDA kernel
- `pytorch_binding.py` - PyTorch C++ extension bindings
- `real_gpu_optimizations.py` - Integration layer for real GPU optimizations

**Why together:** Custom kernel implementations that could be integrated into the optimization pipeline.

---

### 📁 **profiling_tools/** - Reusable Profiling Utilities
Contains modular, reusable profiling tools for import into other scripts.

**Files:**
- `__init__.py` - Package initialization
- `pytorch_profiler.py` - PyTorch profiler wrapper and utilities
- `roofline_benchmark.py` - Roofline model benchmarking utilities

**Why together:** Importable utilities for profiling, separate from one-off profiling scripts.

---

### 📁 **results/** - Benchmark Results and Metrics
Contains JSON output files with measured metrics and results summaries.

**Files:**
- `profiling_baseline_b0.json` - Baseline profiling metrics for SegFormer-B0
- `profiling_results.json` - Per-image latency metrics (moved from segmentation_outputs/)
- `README.md` - Results documentation

**Why here:** All quantitative results and metrics collected in one location for easy comparison.

---

### 📁 **segmentation_outputs/** - Generated Segmentation Visualizations
Contains PNG visualizations and NPY arrays comparing FP32 vs FP16 segmentation outputs.

**Files:**
- `*_fp32_comparison.png` - FP32 segmentation visualizations
- `*_fp16_comparison.png` - FP16 segmentation visualizations
- `*_fp32_segmentation.npy` - FP32 raw segmentation arrays
- `*_fp16_segmentation.npy` - FP16 raw segmentation arrays

**Why separate:** Large binary output files (images, arrays) organized separately from code and documentation.

---

### 📁 **Root Level Files**

**Documentation:**
- `README.md` - Main optimization summary and results (shows 1.87x FP16 speedup)
- `QUICKSTART.md` - Quick start guide for optimization
- `RESULTS.md` - High-level results summary

**Config:**
- `.gitignore` - Git ignore rules

---

## Why Was It Disorganized Before?

The folder previously had **37 files in the root level**, mixed together:

### Problems with the Old Structure:
1. **No semantic grouping** - Documentation, scripts, and results all at the same level
2. **Hard to navigate** - No clear indication of what each file does or which files work together
3. **Metrics scattered** - `kernel_profiling_results.json` was randomly in the root instead of with other profiling outputs
4. **Confusing dependencies** - No way to tell which scripts depend on which documentation
5. **Mixed concerns** - Core optimization code sat next to legacy/experimental scripts

### Example Issues:
- `CUDA_LIBRARIES_OPTIMIZATION.md` was separate from `cuda_libraries_optimization.py` and `cuda_libraries_impact.py`
- `kernel_profiling_results.json` was in root (not associated with other profiling outputs)
- `profile_with_timm_model.py` and `profile_with_trained_model.py` (non-functional legacy code) sat alongside working scripts
- Documentation files (35+ .md files) weren't distinguished from code

---

## How to Use This Structure

### For Understanding CUDA Optimizations:
→ Start in `cuda_libs/` directory

### For Profiling and Analyzing Performance:
→ Look in `profiling/` for both guides and tools

### For Verifying Optimizations Work:
→ Check `verification/` directory

### For Running Inference or Deploying:
→ Use scripts and guides in `inference/`

### For Viewing Results:
→ Check `results/` for JSON metrics and `segmentation_outputs/` for visualizations

---

## File Dependencies

```
STRUCTURE:
  cuda_libs/
    ├─ CUDA_LIBRARIES_OPTIMIZATION.md (overview)
    ├─ cuda_libraries_optimization.py (uses → cuDNN, cuBLAS)
    └─ cuda_libraries_impact.py (measures impact)

  profiling/
    ├─ measure_real_metrics.py (produces → real_gpu_metrics.json)
    ├─ profile_kernel_metrics.py (produces → kernel_profiling_results.json)
    └─ *_GUIDE.md (interprets the JSON files)

  verification/
    ├─ check_model_quality.py
    ├─ verify_optimization.py (uses → cuda_libs/)
    └─ verify_with_real_data.py (uses → data/ from root)

  inference/
    ├─ run_inference.py (core inference)
    ├─ optimize_onnx_tensorrt.py (requires ONNX setup)
    └─ TRAINED_MODELS_AVAILABLE.md (documents models in optimized_models/)
```

---

## Key Metrics Locations

| Metric | File | Location |
|--------|------|----------|
| Kernel timing | kernel_profiling_results.json | profiling/ |
| GPU hardware metrics | real_gpu_metrics.json | profiling/ |
| Per-image latencies | profiling_results.json | results/ |
| Optimization results | cuda_libraries_optimization_results.json | cuda_libs/ |
| Segmentation outputs | *.png, *.npy | segmentation_outputs/ |

---

## Maintenance Notes

- **Legacy scripts** in `profiling/` (`profile_with_timm_model.py`, `profile_with_trained_model.py`) could be removed if not needed for reference
- **Custom kernels** in `kernels/` are advanced and may require CUDA toolkit updates
- **Results** are organized by topic in `results/` for easy archival/backup
- **Segmentation outputs** can grow large; consider periodic cleanup of old visualization files

