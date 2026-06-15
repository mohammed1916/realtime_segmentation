# GPU Optimization Documentation

Complete documentation for SegFormer FP16+TF32 optimization.

## Quick Links

### If You're Deploying
→ Go back to main folder and read `DEPLOYMENT_GUIDE.md` (5 min)

### If You Want to Understand the Optimization
1. [ACTUAL_OPTIMIZATION_RESULTS.md](ACTUAL_OPTIMIZATION_RESULTS.md) - All 4 iterations with results
2. [OPTIMIZATION_DECISION_LOOP.md](OPTIMIZATION_DECISION_LOOP.md) - How decisions were made
3. [STATUS_FINAL.md](STATUS_FINAL.md) - Executive summary

### If You Want to Replicate
1. [ITERATION_CHECKLIST.md](ITERATION_CHECKLIST.md) - Step-by-step workflow
2. Back to main folder: `measure_iteration.py` - Measurement tool
3. `profiling/iter_*.json` - Measurement data

### If You Want to Learn GPU Optimization Theory
1. [PROFILER_METRICS_GUIDE.md](PROFILER_METRICS_GUIDE.md) - What metrics mean
2. [KERNEL_PROFILING_GUIDE.md](KERNEL_PROFILING_GUIDE.md) - Specific thresholds
3. [REAL_MEASURED_METRICS.md](REAL_MEASURED_METRICS.md) - Hardware baseline

### If You're Preparing for Interviews
1. [ACTUAL_OPTIMIZATION_RESULTS.md](ACTUAL_OPTIMIZATION_RESULTS.md) - Real results
2. [OPTIMIZATION_DECISION_LOOP.md](OPTIMIZATION_DECISION_LOOP.md) - Decision-making framework
3. [STATUS_FINAL.md](STATUS_FINAL.md) - Talking points

---

## File Guide

| File | Purpose | Audience |
|------|---------|----------|
| [ACTUAL_OPTIMIZATION_RESULTS.md](ACTUAL_OPTIMIZATION_RESULTS.md) | Complete optimization results with all iterations | Everyone |
| [OPTIMIZATION_DECISION_LOOP.md](OPTIMIZATION_DECISION_LOOP.md) | Decision-making framework with real examples | Learning, interviews |
| [STATUS_FINAL.md](STATUS_FINAL.md) | Executive summary and key findings | Quick reference |
| [ITERATION_CHECKLIST.md](ITERATION_CHECKLIST.md) | Reproducible workflow for each iteration | Replication |
| [PROFILER_METRICS_GUIDE.md](PROFILER_METRICS_GUIDE.md) | GPU profiler metrics explained | Theory, deep learning |
| [KERNEL_PROFILING_GUIDE.md](KERNEL_PROFILING_GUIDE.md) | Specific metric thresholds for SegFormer | Reference |
| [REAL_MEASURED_METRICS.md](REAL_MEASURED_METRICS.md) | Hardware specs and baseline measurements | Reference |
| [REAL_GPU_OPTIMIZATION_SUMMARY.md](REAL_GPU_OPTIMIZATION_SUMMARY.md) | What works, what doesn't | Reference |
| [INDEX.md](INDEX.md) | Full documentation index | Navigation |
| [QUICKSTART.md](QUICKSTART.md) | Quick start guide for profiling | Getting started |

---

**Most important:** Start with [ACTUAL_OPTIMIZATION_RESULTS.md](ACTUAL_OPTIMIZATION_RESULTS.md), then decide what else you need.
