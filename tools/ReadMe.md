## Use case

- tools/model_optimizer.py
  - Purpose: The core toolkit for optimizing a single model (loads config + checkpoint, converts to FP16/INT8, exports ONNX, optionally creates TensorRT engine, benchmarks, and writes results).
  - Use when: you want to optimize or benchmark one specific checkpoint and control which optimizations to run.
  - Notes: Provides checkpoint meta-preservation, benchmarking utilities, and outputs into `optimized_models/<checkpoint_stem>/<session>/`. Requires mmseg APIs, torch, onnx/onnxruntime, TensorRT (for TRT), and other libs. Some methods may rely on optional backends (onnxruntime, tensorrt).

- tools/batch_optimize_all.py
  - Purpose: Batch wrapper that scans the repo for `.pth` files, maps each checkpoint to a config, creates a timestamped output directory, and calls `ModelOptimizer` to run FP16/INT8/batch/ONNX/TensorRT steps for each model.
  - Use when: you want to run the same set of optimizations for many `.pth` files automatically (all-model optimization).
  - Notes: Contains a mapping dict of checkpoint filename → config path. Ensure mapping matches your file names and `local_configs` exist. Output structure is `optimized_models/<model_name>/<timestamp_xxx>/...`.

- tools/batch_benchmark.py
  - Purpose: Batch benchmarking script that finds `.pth` files and benchmarks the "original" model (and can be adapted to benchmark optimized variants).
  - Use when: you want to produce aggregate benchmark reports (JSON/CSV/txt) without performing optimizations.
  - Notes: Useful after optimization to compare original vs optimized results; expects `ModelOptimizer.benchmark_model` to be available.

- onnx_video_segmentation.py
  - Purpose: Runtime inference demo using an ONNX model with temporal logit smoothing for video – general-purpose and device-agnostic (CPU/CUDA supported).
  - Use when: you already have an ONNX export and want to run live/video segmentation with smoothing and visualization.

- onnx_video_segmentation_b0.py
  - Purpose: Specialized ONNX video demo tuned for SegFormer B0 (assumes 1024x1024 input and a B0 palette/normalization); defaults to CUDA.
  - Use when: you optimized a SegFormer B0 ONNX model and want a slightly higher-performance / model-specific pipeline.

- optimized_models/ (folder)
  - Purpose: Destination layout for optimizer outputs: each model gets its own folder with timestamped runs; contains `.pth` (fp16/int8), `.onnx`, optimized `.onnx`, benchmark CSV/JSON, and an `optimization_summary.txt`.
  - Use when: inspecting results or picking the ONNX/TensorRT artifact to run inference demos.

## Which file to use — quick decision guide

- Single model optimization (one checkpoint)
  - Use `tools/model_optimizer.py`.
  - Example: run the tool specifying config and checkpoint and choose optimizations (fp16, int8, onnx, tensorrt, batch).
  - Why: It's the toolkit implementing conversion, saving, and benchmarking for a single model.

- All-model (batch) optimization
  - Use `tools/batch_optimize_all.py`.
  - Why: It enumerates `.pth` files, finds the right config via the mapping, and calls `ModelOptimizer` for each model while creating timestamped directories.

- Only benchmarking many models (no optimization)
  - Use `tools/batch_benchmark.py`.
  - Why: Designed to run benchmarks across multiple `.pth` files and produce a batch summary (JSON/CSV/TXT).

- Run inference on video with ONNX model
  - If model is general/polymorphic: use `onnx_video_segmentation.py`.
  - If model is SegFormer B0 and you want the optimized B0 pipeline: use `onnx_video_segmentation_b0.py`.

## Practical examples (copyable commands)

- Optimize a single model (example)
```bash
PYTHONPATH=. python tools/model_optimizer.py \
  configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py \
  seg_b0_ade.pth \
  --device cuda:0 \
  --optimize fp16 onnx tensorrt
```

- Run batch optimization (all `.pth` files found)
```bash
PYTHONPATH=. python tools/batch_optimize_all.py
```

- Run batch benchmarking
```bash
PYTHONPATH=. python tools/batch_benchmark.py --root-dir . --num-runs 5
```

- Run ONNX video demo (general)
```bash
PYTHONPATH=. python onnx_video_segmentation.py \
  --video_path videos/input.avi \
  --onnx_model optimized_models/segformer.b0.1024x1024.city.160k/.../segformer..._onnx.onnx \
  --device cuda --show
```

- Run ONNX B0-specific demo
```bash
PYTHONPATH=. python onnx_video_segmentation_b0.py \
  --video_path videos/input.avi \
  --onnx_model optimized_models/.../segformer.b0..._onnx.onnx \
  --device cuda --display
```
