# Kernel Profiling Guide - Metrics for SegFormer Optimization

## Overview

This guide explains how to interpret GPU kernel profiling metrics and use them to guide optimization decisions for SegFormer inference.

**Key Metrics:**
- L2 Cache Hit Rate
- Memory Bandwidth Utilization
- SM Occupancy (Streaming Multiprocessor)
- Warp Efficiency
- Arithmetic Intensity
- Register Pressure

---

## 1. L2 Cache Hit Rate

### What it is:
```
L2 Hit Rate = L2 Hits / (L2 Hits + L2 Misses)
```
Percentage of memory requests satisfied by L2 cache (vs going to main memory HBM)

### Expected values by operation:

| Operation | Good L2 Hit Rate | Why |
|---|---|---|
| **Conv2d** | 40-60% | Data reuse across tiles, some overflow to HBM |
| **BatchNorm** | 70-80% | Small working set, streaming pattern |
| **Linear/Dense** | 60-80% | Good data reuse with tiling |
| **Attention** | 40-50% | Large intermediate matrix overflows L2 |
| **Upsampling** | 50-70% | Regular memory access pattern |

### For SegFormer:

**Baseline (FP32):**
- Conv layers: ~45-55% L2 hit rate (typical)
- BatchNorm: ~75% L2 hit rate
- Upsampling: ~60% L2 hit rate

**With FP16 (Memory-bound benefit):**
- Conv layers: ~55-65% L2 hit rate (2x less data)
- BatchNorm: ~80-85% L2 hit rate
- Overall: Better cache utilization due to 2× smaller data size

### How to improve L2 hit rate:

**If L2 hit rate < 40%:**
1. **Kernel fusion:** Combine operations to reduce memory round-trips
   ```python
   # Bad: Conv -> ReLU (2 memory loads)
   x = conv(x)
   x = relu(x)
   
   # Better: Fused Conv+ReLU (1 memory load)
   x = fused_conv_relu(x)
   ```

2. **Input tiling:** Process smaller blocks that fit in L2
   ```
   L2 cache size: 5-6 MB
   Working set for 512×512 float32: ~1 GB
   
   Solution: Process 64×64 tiles (fits in L2)
   ```

3. **Data reuse:** Maximize reuse before eviction
   ```python
   # Load once, use multiple times
   # Good for: Convolution with multiple output channels
   # Bad for: Single-pass operations like element-wise add
   ```

---

## 2. Memory Bandwidth Utilization

### What it is:
```
Achieved Bandwidth = (Bytes Read + Bytes Written) / Kernel Time
```
Percentage of peak GPU memory bandwidth being utilized

### Peak bandwidth by GPU:

| GPU | Peak BW | Typical App |
|---|---|---|
| RTX 3090 | 936 GB/s | 350-450 GB/s for conv |
| RTX 4090 | 1008 GB/s | 400-550 GB/s for conv |
| A100 (HBM2) | 2039 GB/s | 800-1200 GB/s for conv |

### For SegFormer:

**Baseline (FP32, 512×512 input):**
- Convolution: ~380 GB/s achieved (41% of peak)
- BatchNorm: ~450 GB/s achieved (48% of peak)
- Upsampling: ~500 GB/s achieved (53% of peak)

**Why not higher?**
- Convolution is memory-bound but has limited arithmetic intensity
- Must wait for data -> can't saturate bandwidth
- Memory latency dominates

**With FP16 (2× less data):**
- Convolution: ~200 GB/s (same operation, but FP16 data)
- Actual time: 2× faster because 2× less data to move

### How to improve bandwidth utilization:

**If achieved < 30% of peak:**
1. **Memory coalescing:** Ensure sequential memory access
   - NCHW format (PyTorch default): Good
   - NHWC format: Bad (poor cache line utilization)

2. **Reduce memory traffic:**
   ```python
   # Bad: Load from HBM 3x (Q, K, V)
   Q = data[:, :, :]
   K = data[:, :, :]
   V = data[:, :, :]
   
   # Better: Load once, reuse
   data = load_once()
   Q, K, V = split(data)
   ```

**If achieved 40-60%:**
- Normal for deep learning (memory-bound is expected)
- Focus on reducing data size (FP16) rather than bandwidth optimization

---

## 3. SM Occupancy

### What it is:
```
Occupancy = (Active Warps per SM / Max Warps per SM) × 100%
```
Percentage of GPU resources being used (warp scheduling)

### Max warps per SM by GPU:

| GPU | Warp Capacity | Max Threads |
|---|---|---|
| RTX 3090 (Ampere) | 48 warps | 1536 threads |
| RTX 4090 (Ada) | 48 warps | 1536 threads |
| A100 (Ampere) | 48 warps | 1536 threads |

### Expected occupancy by operation:

| Operation | Good Occupancy | Why |
|---|---|---|
| **Conv (3×3)** | 60-80% | Register-limited |
| **Conv (1×1)** | 80-95% | Low register count |
| **Attention** | 50-65% | High register count per thread |
| **BatchNorm** | 70-85% | Moderate register usage |

### For SegFormer:

**Baseline (FP32):**
- Stage 1 (64 channels): ~75% occupancy
- Stage 2 (128 channels): ~70% occupancy
- Stage 4 (512 channels): ~65% occupancy (register pressure)

**Why does it decrease?**
- Larger convolutions = more registers per thread
- More registers = fewer warps fit on SM
- Trade-off: Can't improve without changing algorithm

### How to improve occupancy:

**If occupancy < 50%:**
1. **Reduce register count:**
   ```python
   # Bad: Store full result in registers
   result = [r1, r2, r3, r4]
   
   # Better: Recompute when needed (trades registers for compute)
   result_on_demand = compute(x)
   ```

2. **Reduce shared memory usage:**
   ```cuda
   __shared__ float tile[128][128];  // Uses 64KB per block
   // If too much -> reduce tile size
   __shared__ float tile[64][64];    // Uses 16KB per block
   ```

**If occupancy 60-80%:**
- Normal (register or shared memory limited)
- Usually acceptable

**If occupancy > 85%:**
- Excellent (rare, usually only for simple ops)

---

## 4. Warp Efficiency

### What it is:
```
Warp Efficiency = (Non-Stalled Instructions / Total Instructions) × 100%
```
Percentage of time warps are executing (not waiting)

### Stall reasons:

| Stall Reason | % for Conv | % for Attention | Solution |
|---|---|---|---|
| **Memory Dependency** | 50-60% | 60-70% | Pre-load data, pipeline |
| **Memory Throttle** | 15-25% | 10-20% | Reduce data size |
| **Execution Dependency** | 5-10% | 5-10% | Parallelize independent ops |
| **Instruction Cache** | <5% | <5% | Normally not an issue |

### For SegFormer:

**Baseline (FP32):**
- Conv layers: ~70-75% warp efficiency
- Attention (if present): ~65-70% warp efficiency

**With FP16:**
- Conv layers: ~75-80% warp efficiency (less memory stalls)
- Attention: ~70-75% warp efficiency

### How to improve warp efficiency:

**If memory dependency > 60%:**
1. **Prefetch data:**
   ```cuda
   // Load next iteration's data while processing current
   load_data(next_tile);
   process(current_tile);
   ```

2. **Increase data reuse:**
   - More computation per loaded byte
   - Reduces memory waits

**If memory throttle > 25%:**
1. **Reduce data size:**
   - FP16 instead of FP32 (2× reduction)
   - Quantization to INT8 (4× reduction)

2. **Kernel fusion:**
   - Avoid redundant memory loads

---

## 5. Arithmetic Intensity

### What it is:
```
Arithmetic Intensity = Floating Point Operations / Bytes Transferred
```
How much computation per byte of data moved

### Classification:

| Intensity | Classification | Bottleneck |
|---|---|---|
| < 0.5 ops/byte | Extreme memory-bound | Can't compute fast enough |
| 0.5 - 2 ops/byte | Memory-bound | Move data is limiting |
| 2 - 10 ops/byte | Mixed | Depends on GPU |
| > 10 ops/byte | Compute-bound | Computation is limiting |

### For SegFormer:

**Conv2d (3×3 kernel, 64 channels):**
```
Operations: 9 × 64 × 64 × 512 × 512 = ~96 billion ops
Data: Input (3 × 512 × 512) + Weight (3×3×64) + Output (64×512×512) = ~200 MB
Arithmetic Intensity: 96B / 200M = 0.48 ops/byte → MEMORY-BOUND
```

**Conv2d (1×1 kernel, 256 channels):**
```
Operations: 1 × 256 × 256 × 256 × 256 = ~16 billion ops  
Data: Input (256 × 256 × 256) + Weight (1×1×256) + Output = ~32 MB
Arithmetic Intensity: 16B / 32M = 0.5 ops/byte → MEMORY-BOUND
```

**Attention (if present):**
```
Operations: Q@K = seq_len² × dim
Data: Q, K, V, Attention scores
Arithmetic Intensity: 0.77 ops/byte → MEMORY-BOUND
```

### Implication for SegFormer:

**All operations are memory-bound** ✓

→ **Optimization strategy:**
1. Reduce memory bandwidth (FP16, INT8)
2. Improve cache utilization (kernel fusion)
3. Not worth optimizing compute path (won't help)

---

## 6. Roofline Model

Visual way to understand bottlenecks:

```
TFLOP/s
  ^
  | ████████████ Compute Roof (max achievable)
  |           /
  |          /
  |         / Compute-bound
  |        /    region
  |       /
  |______/ Bandwidth Roof = Peak BW × Peak Performance
         \
          \_ Memory-bound region
           \
            \______________> Arithmetic Intensity (ops/byte)
```

### For SegFormer (RTX 3090):

```
Peak Performance: 10 TFLOP/s (FP32)
Peak Bandwidth: 936 GB/s
Bandwidth Roof = min(10, 936 × AI)

For AI = 0.5 ops/byte:
  Achievable TFLOP/s = 936 × 0.5 = 468 GFLOP/s = 0.468 TFLOP/s
  
Current: 1.2 TFLOP/s (above the roof! Limited by other factors)
Realistic: 0.6-0.8 TFLOP/s (typical for conv)

Conclusion: Memory-bound, can't go faster without algorithm change
```

---

## Using Metrics to Guide Kernel Improvements

### Decision Tree:

```
Profiler shows slow kernel
    |
    ├─ High occupancy (>80%) + Low warp efficiency?
    │  └─> Register spilling or shared memory limited
    │      └─> Reduce computation per thread
    │
    ├─ Low occupancy (<50%)?
    │  └─> Register or shared memory limited
    │      └─> Reduce per-thread resource usage
    │
    ├─ Memory-bound (AI < 2)?
    │  ├─> L2 hit rate < 40%?
    │  │   └─> Kernel fusion or tiling
    │  └─> Achieved BW < 30% peak?
    │      └─> Memory coalescing, NCHW format
    │
    └─> Compute-bound (AI > 10)?
        └─> Not typical for SegFormer
            └─> Consider algorithmic change
```

---

## Practical Example: Optimizing Conv Layer

### Initial State (FP32):
```
Latency: 12.5 ms
L2 hit rate: 42%
Occupancy: 65%
Warp efficiency: 72%
Achieved BW: 380 GB/s (41% of peak)
Arithmetic Intensity: 0.48 ops/byte
Classification: MEMORY-BOUND
```

### Analysis:
- **Occupancy is reasonable** (65% is expected for conv)
- **Warp efficiency is low** (72%) → Memory stalls
- **L2 hit rate is low** (42%) → Working set too large
- **Memory-bound confirmed** (AI < 2, achieved BW < 50%)

### Optimization Options (ranked by impact):

1. **FP16 Mixed Precision** (Immediate, +30%)
   - Reduces data size 2×
   - Better L2 hit rate
   - Better warp efficiency
   - No algorithm change
   - **Expected: 12.5 ms → 9 ms**

2. **Kernel Fusion with ReLU** (2-3 hours, +10%)
   - Combine Conv + ReLU kernels
   - Single memory pass instead of two
   - Reduces L2 misses
   - **Expected: 9 ms → 8.1 ms**

3. **Input Tiling** (4-6 hours, +15%)
   - Process 64×64 tiles (fit in L2)
   - Explicit tile-based algorithm
   - Higher L2 hit rate
   - **Expected: 8.1 ms → 7 ms**

4. **Custom Winograd Kernel** (2-4 weeks, +20%)
   - Winograd FFT-based convolution
   - Arithmetic complexity reduction
   - Complex implementation
   - **Expected: 7 ms → 5.6 ms**

---

## Profiling Commands

### PyTorch Profiler (Built-in):
```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    output = model(input)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### Nsight Compute (NVIDIA, detailed):
```bash
# Profile model
ncu -o profile.ncu python run_inference.py

# View results in GUI
ncu-ui profile.ncu
```

### Nsight Systems (Timeline):
```bash
# Capture timeline
nsys profile --trace cuda,osrt python run_inference.py

# View in GUI
nsys-ui qdrep_* / sqlite
```

---

## Summary: Metrics to Priorities

| Metric | Priority | Typical | Target | Action |
|---|---|---|---|---|
| **L2 Hit Rate** | HIGH | 40-60% | >60% | Fusion, tiling |
| **Memory BW** | HIGH | 40-60% peak | Improve with FP16 | Precision selection |
| **Occupancy** | MEDIUM | 60-80% | >70% | Register reduction |
| **Warp Eff** | MEDIUM | 70-80% | >80% | Memory optimization |
| **Arith Intensity** | MEDIUM | <2 | Can't improve | Focus on memory |

**For SegFormer:**
1. Focus on memory metrics (L2, BW) - operations are memory-bound
2. Use FP16 immediately (+30% from memory reduction)
3. Consider kernel fusion next (+10-15%)
4. Custom kernels only if targeting <5ms latency

