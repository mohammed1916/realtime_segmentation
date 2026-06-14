# GPU Profiler Metrics Interpretation Guide

A practical guide to understanding profiler output and connecting metrics to optimization strategies.

---

## Part 1: PyTorch Profiler Metrics

### Metric 1: `self_cuda_time_total`

**Definition:**
```
Total CUDA execution time for this kernel (exclusive of child operations)
Unit: milliseconds
```

**How to Read:**
```
aten::matmul                      self_cuda_time_total: 12.5 ms
aten::softmax                     self_cuda_time_total: 2.1 ms
aten::conv2d                      self_cuda_time_total: 1.3 ms
```

**Interpretation:**
- **Highest values** = Primary bottlenecks
- Sum of all `self_cuda_time_total` ≈ Total inference time
- If top 3 operations account for >70% of time → Good focus area

**For SegFormer Expected Pattern:**
```
Top 3 Operations (should total ~60-70% of time):
1. aten::_scaled_dot_product_attention or matmul  ~15-20 ms (40-50%)
2. aten::conv2d (MixFFN fc1)                      ~4-5 ms   (10-12%)
3. aten::softmax (attention)                      ~2-3 ms   (5-7%)
```

**Action Items:**
- If attention <30% of time → Model is unusually efficient or profiler is not capturing correctly
- If attention >60% of time → Priority optimization target
- If FFN >35% of time → Second optimization target

---

### Metric 2: `cpu_time` vs `cuda_time`

**Definition:**
```
CPU time: Time spent on CPU preparing the kernel (host-side overhead)
CUDA time: Actual execution on GPU
```

**Pattern to Look For:**
```
Operation                     CPU Time    CUDA Time    Ratio
─────────────────────────────────────────────────────────
aten::matmul                  0.08 ms     12.5 ms      1:156
aten::softmax                 0.02 ms     2.1 ms       1:105
aten::conv2d                  0.03 ms     1.3 ms       1:43
aten::layer_norm              0.01 ms     0.8 ms       1:80
```

**Good Ratio:** CPU:CUDA > 1:50

**Why It Matters:**
- If CPU time is high relative to CUDA time → Python overhead, not GPU bottleneck
- Normal for deep learning: GPU is working while CPU prepares next kernel

**Red Flags:**
```
aten::matmul                  2.3 ms      0.1 ms       23:1  ← WRONG!
(means kernel launch overhead > actual compute)
```

**Fix:** Usually indicates:
- Kernel launch failures (run without profiler to check)
- GPU synchronization issues
- Not a common problem with modern PyTorch

---

### Metric 3: `total_time` vs `self_time`

**Definition:**
```
total_time: Including all nested operations
self_time: This operation only (exclusive)

Example:
  TransformerEncoderLayer (total_time: 10 ms)
    ├─ LayerNorm (total_time: 0.5 ms, self_time: 0.5 ms)
    ├─ Attention (total_time: 6 ms, self_time: 6 ms)
    └─ FFN (total_time: 3.5 ms, self_time: 3.5 ms)
```

**For Optimization:**
- Focus on operations with high `self_time` (actual GPU work)
- High `total_time` but low `self_time` = mostly overhead (less critical)

**SegFormer Typical Pattern:**
```
TransformerEncoderLayer
  ├─ norm1 + Attention: 6.2 ms ← Optimize first
  └─ norm2 + FFN: 2.8 ms        ← Optimize second
```

---

### Metric 4: `flops` (Floating Point Operations)

**Definition:**
```
flops: Floating point operations for this kernel
Unit: Operations (not TFLOP/s!)
```

**Calculate Throughput:**
```
TFLOP/s = flops / (self_cuda_time_total × 1e12)

Example:
  aten::matmul: 50e9 flops, 12.5 ms
  TFLOP/s = 50e9 / (12.5e-3 × 1e12) = 4 TFLOP/s

  GPU Peak: 10 TFLOP/s
  Utilization: 4 / 10 = 40%
```

**For SegFormer Expected:**
```
Attention TFLOP/s:  1.0-1.5   (10-15% of peak)  ← Memory-bound, expected
Conv1×1 TFLOP/s:    2.0-3.0   (20-30% of peak)  ← Memory-bound, expected
Dense MM TFLOP/s:   8.0-10.0  (80-100% of peak) ← Compute-bound
```

**Red Flags:**
- Attention showing >5 TFLOP/s on your setup → Check math (may be counting differently)
- Conv showing <1 TFLOP/s → Likely using non-standard precision or there's an issue

---

### Metric 5: `memory_usage` (Profiler Memory Estimate)

**Definition:**
```
CPU memory: Host (CPU-side) memory allocated
CUDA memory: Device (GPU-side) memory allocated
Unit: Bytes
```

**Note:** Profiler estimates are approximate.

**For Accuracy:**
```python
import torch
print(torch.cuda.memory_allocated())      # Actual allocated
print(torch.cuda.max_memory_allocated())  # Peak usage
print(torch.cuda.memory_reserved())       # Reserved (includes free space)
```

**What's Typical for SegFormer-B0 (512×512):**
```
Model weights:      ~50 MB
Activations:        ~300-500 MB
Optimization state: 0 MB (inference)
─────────────────────────
Total:             ~400-600 MB

Typical peak: 2-4 GB (includes caching)
```

**If Memory > 6 GB:**
- Using batch size > 1
- Or using unusually large input
- Or model variant is larger (B3-B5)

---

## Part 2: Nsight Systems Timeline Interpretation

### Understanding the Timeline View

When you open a Nsight Systems trace, you see:

```
Timeline
├─ CPU
│  ├─ Python [===]
│  └─ CUDA Driver [=]
├─ GPU
│  ├─ Kernel Stream
│  │  ├─ patch_embedding_kernel [===] 0.15 ms
│  │  ├─ TransformerEncoderLayer[0] [=======] 2.1 ms
│  │  ├─ TransformerEncoderLayer[1] [=======] 2.1 ms
│  │  ├─ norm1 [=] 0.1 ms
│  │  ├─ attention_kernel [========] 1.8 ms ← Can't parallelize
│  │  ├─ softmax_kernel [=====] 1.2 ms ← Dependency on attention
│  │  └─ ffn_kernels [=======] 1.5 ms
│  └─ Memory Events
│     ├─ DRAM → L2 Cache [========] 2.3 GB/s read
│     └─ L2 → GPU [====] 1.1 GB/s compute
└─ Annotations (if you add NVTX)
```

### Key Observations

1. **Kernel Sequence (Sequential or Parallel?)**

   **Sequential:** Each kernel waits for previous to finish
   ```
   [Kernel A====]
              [Kernel B====]
                      [Kernel C====]
   Total: sum of all
   ```
   
   **Parallel:** (Rare in inference, more common in training)
   ```
   [Kernel A====]
   [Kernel B====]  (started before A finished)
   [Kernel C====]
   Total: max of all
   ```

   **For SegFormer:** Sequential is expected. No parallelization opportunity without algorithmic changes.

2. **Memory Transfer Events**

   Look for sustained DRAM utilization:
   ```
   Timeline: [========== Heavy DRAM ============] (steady memory traffic)
   
   Expected: 350-450 GB/s for attention, 250-350 GB/s for FFN
   ```

   If you see:
   - Idle periods between kernels → Memory latency issues or CPU bottleneck
   - Spikes followed by gaps → Synchronization happening

3. **Kernel Launch Latency**

   ```
   [Kernel finish]
   [Gap]               ← Launch overhead + CPU work
   [Next kernel start]
   ```
   
   Expected gap: <10 µs (usually <1 µs)
   
   If gap > 100 µs: Indicates CPU-side overhead or kernel error

### Creating Custom Nsight Annotations

Add timing markers to your profiling:

```python
import torch
from torch.profiler import profile, record_function

with profile(activities=[...]) as prof:
    with record_function("stage1_backbone"):
        stage1_output = backbone_stage1(x)
    
    with record_function("stage2_backbone"):
        stage2_output = backbone_stage2(stage1_output)
    
    with record_function("decode_head"):
        output = decode_head([stage1_output, stage2_output, ...])
```

This will show in Nsight timeline as labeled sections, making it easy to identify which stage is slowest.

---

## Part 3: Nsight Compute Deep Dive

### Core Metrics to Check

#### 1. SM Occupancy (Streaming Multiprocessor Occupancy)

**Definition:**
```
Occupancy = (Active Warps / Max Warps per SM) × 100%

RTX 3090: 1536 CUDA cores = 48 warps per SM, 130 SMs
Maximum possible occupancy: 100%
```

**What to Expect:**

| Occupancy | Meaning | For Attention | For Conv1×1 |
|-----------|---------|---|---|
| >80% | Excellent | Hard to achieve (register pressure) | Common |
| 60-80% | Good | Expected for attention kernels | Very good |
| 40-60% | OK | Possible, but limited latency hiding | OK |
| <40% | Poor | Likely register spilling or shared mem full | Investigate |

**For SegFormer Attention:**
Expected: 50-65% (because of high register count per thread)

**Why Low Occupancy for Attention?**
```
Registers per thread: 80-100 (high due to local Q, K, V data)
Register file per SM: 65,536 (for RTX 3090)
Max threads per SM: 2048

If using 100 regs/thread:
  Max threads: 65,536 / 100 = 655 threads
  As warps: 655 / 32 = ~20 warps (out of 48)
  Occupancy: 20/48 = 42%
```

**Action:**
- Occupancy >60% for attention = Good
- <50% = Potential register spilling (check "Register Spill" metric)

#### 2. Warp Stall Reasons (Most Important!)

**Definition:**
```
Why warps are NOT executing (stalled)
- Total should equal 100%
```

**Breakdown for SegFormer Attention:**
```
Warp Stall Reasons:
├─ Memory Dependency: 50-60%     ← Waiting for load
├─ Memory Throttle: 15-25%       ← DRAM bottleneck
├─ Execution Dependency: 5-10%   ← Waiting on arithmetic
├─ Instruction Cache: <5%
└─ Other: <5%
```

**Interpretation:**
- **Memory Dependency >50%:** Data isn't arriving fast enough. Normal for bandwidth-bound ops.
- **Memory Throttle >20%:** Hitting DRAM peak bandwidth. Indicates opportunity for optimization.
- **Execution Dependency >10%:** Algorithmic dependencies (like softmax sequential nature). Hard to optimize.

**For Optimization:**
- If Memory Dependency is dominant (>50%) → Focus on memory coalescing, data reuse
- If Memory Throttle is dominant (>25%) → Focus on algorithmic changes (sparsity, low-rank)
- If Execution Dependency >20% → Algorithmic changes required (can't parallelize further)

#### 3. L2 Cache Hit Rate

**Definition:**
```
L2 Hit Rate = L2 Hits / (L2 Hits + L2 Misses)
```

**Memory Hierarchy:**
```
Registers (32KB per thread)
  ↓ Misses go to L1
L1 Cache (128KB per SM)
  ↓ Misses go to L2
L2 Cache (5-6 MB total)
  ↓ Misses go to Main Memory (HBM)
Main Memory (24-48 GB)
```

**Expected Values:**
```
Attention (Q @ K^T):
- Q: ~4 MB (may stay in L2)
- K,V: ~4 MB (shared across threads, high reuse)
- Attention scores: ~8 MB (likely overflow L2)
→ Expected: 40-50% L2 hit rate

Conv1×1:
- Weights: Small, usually in L1/L2
- Activations: Spatial pattern, good reuse
→ Expected: 60-70% L2 hit rate

LayerNorm:
- Input + weight + bias: Small total
- Streamed once per forward
→ Expected: >80% L2 hit rate
```

**Action:**
- L2 hit rate <30% → Working set too large, consider tiling
- L2 hit rate 30-60% → Normal for deep learning
- L2 hit rate >70% → Good data locality

#### 4. Achieved Memory Bandwidth

**Definition:**
```
Achieved BW = (Bytes Read + Bytes Written) / Kernel Time
Unit: GB/s
```

**Peak Bandwidth:**
```
RTX 3090: 936 GB/s (HBM1)
RTX 4090: 1008 GB/s (HBM2)
A100: 2039 GB/s (HBM3)
```

**Expected Achievement Rate:**

| Operation | Expected % Peak | Actual GB/s (RTX 3090) | Why |
|-----------|---|---|---|
| Attention | 40-50% | 350-450 | Data reuse limits utilization |
| Conv1×1 | 50-60% | 450-550 | Better data structure |
| Matmul (large) | 60-80% | 550-700 | Good memory access pattern |
| Memory copy | 90%+ | 850+ | Streaming operation |

**For SegFormer:**
```
Attention achieved: 380 GB/s = 41% of peak
Conv1×1 achieved: 480 GB/s = 51% of peak
```

**Action:**
- If achieved < 30% of peak → Check for serialization, bank conflicts, or misaligned access
- If achieved 40-60% → Expected for deep learning; focus on arithmetic intensity
- If achieved > 70% → Good bandwidth utilization

#### 5. Tensor Core Utilization

**Definition:**
```
TC Util % = (TC FLOPs / All FLOPs) × 100%
```

**For SegFormer:**
```
Attention: 20-35% TC utilization
  Reason: Arithmetic intensity too low (0.77 ops/byte)
  Tensor cores need >1 op/byte to be effective

Conv1×1: 35-50% TC utilization
  Reason: Better structured, but still memory-bound

Dense MatMul: 80-95% TC utilization
  Reason: High arithmetic intensity, perfect for tensor cores
```

**Can We Do Better?**
- If low TC util due to low arithmetic intensity → No easy fix (algorithmic change needed)
- If low TC util due to small matrix sizes → Use libraries (cuBLAS, cuDNN) which auto-tune
- PyTorch already uses optimized kernels; unlikely to improve TC utilization without algorithm change

---

## Part 4: Connecting Metrics to Optimizations

### Case Study: Attention Optimization

**Current State:**
```
Kernel: aten::_scaled_dot_product_attention
CUDA Time: 12.5 ms (50% of total)
Shape: Q(4096, 128) @ K(1024, 128)^T → S(4096, 1024)

Metrics:
├─ TFLOP/s: 1.2 (12% of peak) ← Very low
├─ Achieved BW: 380 GB/s (41% peak)
├─ L2 Hit Rate: 42% ← Intermediate tensors overflow
├─ Occupancy: 58% ← Register-limited
├─ Warp Stalls: 52% memory dependency ← Normal for BW-bound
└─ Stall Reason: Memory Dependency > Memory Throttle
    (Data not arriving fast, not DRAM full)
```

**Optimization: Flash Attention V2**

Flash Attention processes attention in **blocks** that fit in L2 cache:

```
Before (PyTorch attention):
Q @ K^T:     [====] 6 ms   (must materialize full 4096×1024 matrix)
Softmax:     [===] 2 ms    (must wait for all scores)
Attn @ V:    [====] 4 ms   (must wait for all softmax)
─────────────────────────────
Total:       12 ms         (sequential, separate kernel launches)

After (Flash Attention):
Process blocks:
  Block 1: Compute 256 Q rows @ K rows, softmax, gather V → [==] 1.5 ms
  Block 2: Compute 256 Q rows @ K rows, softmax, gather V → [==] 1.5 ms
  ... (8 blocks total)
─────────────────────────────
Total:        5.5 ms        (parallelizable blocks)
```

**Expected Metric Changes:**

| Metric | Before | After | Reason |
|--------|--------|-------|--------|
| Latency | 12.5 ms | 5.5 ms | 2.3× speedup |
| BW Utilization | 41% | 55% | Better coalescing within blocks |
| Occupancy | 58% | 65% | Less register pressure (no full scores matrix) |
| L2 Hit Rate | 42% | 70% | Blocks fit in L2 |
| TC Utilization | 25% | 35% | More structured memory access |
| Warp Efficiency | 72% | 78% | Fewer stalls |

**Overall Impact:**
```
Attention time: 12.5 ms → 5.5 ms (2.3× speedup)
Total inference: 40 ms → 32 ms (20% overall speedup)
```

---

## Part 5: Common Profiling Mistakes

### Mistake 1: Measuring with Small Batch Size

**Problem:**
```
# Profile with batch_size=1
TFLOP/s = 0.8 (seems bad)

# But batch_size=4 achieves:
TFLOP/s = 1.5 (better utilization)
```

**Why:** GPU hides latency better with more independent work.

**Fix:** Profile with representative batch size (or at least batch size ≥2 for attention)

### Mistake 2: Ignoring Kernel Launch Overhead

**Problem:**
```python
for i in range(10):
    y = model(x)  # Includes 10 kernel launches
```

**Actual:** Kernel launch overhead ~500 ns each, 5 µs total (usually negligible)

But if you profile a single-kernel operation:
```python
# Single attention block
y = attention(q, k, v)  # 1 launch + 0.5 ms compute
```

Launch overhead (0.5 µs) seems like 0.1% of compute, but TFLOP/s measurement includes it.

**Fix:** Average over multiple runs, or use `cudaProfilerStart/Stop` to exclude overhead

### Mistake 3: Not Accounting for Memory Initialization

**Problem:**
```python
x = torch.randn(..., device='cuda')  # CUDA initialization overhead ~50 ms first time
y = model(x)  # Profiling includes initialization
```

**Fix:** Warmup first
```python
model(torch.randn_like(x))
torch.cuda.synchronize()
# Now profile
```

### Mistake 4: Synchronization in the Middle of Profiling

**Problem:**
```python
with profile(...):
    y = attention(q, k, v)
    torch.cuda.synchronize()  # Forced sync!
    y = ffn(y)
```

`synchronize()` is necessary for timing, but if you call it multiple times, you're measuring CPU overhead, not GPU.

**Fix:** Let PyTorch handle synchronization automatically, or:
```python
# Manual timing (if needed)
torch.cuda.synchronize()
start = time.perf_counter()

y = attention(q, k, v)

torch.cuda.synchronize()
elapsed = time.perf_counter() - start
```

---

## Summary: The Optimization Decision Tree

```
┌─ Profiler shows operation takes long time
│
├─ Is TFLOP/s < 20% of peak?
│  ├─ YES → Arithmetic intensity too low (memory-bound)
│  │        Solution: Kernel fusion, improve data reuse
│  │        Example: Flash Attention for attention ops
│  │
│  └─ NO → Data parallelism issue
│           Solution: Increase batch size or occupancy
│
├─ Is L2 hit rate < 40%?
│  ├─ YES → Working set too large
│  │        Solution: Tiling, kernel fusion, sparsity
│  │
│  └─ NO → OK (expected for some ops)
│
├─ Is occupancy < 50%?
│  ├─ YES → Register or shared memory limited
│  │        Solution: Reduce registers (recompute vs store), reduce shared mem usage
│  │        Or: Use smaller block size (trades occupancy for launch overhead)
│  │
│  └─ NO → OK (expected for attention)
│
└─ Is warp efficiency < 70%?
   ├─ YES → Memory stalls dominant
   │        Solution: Increase arithmetic intensity, improve coalescing
   │
   └─ NO → OK (expected, minimal room for improvement)
```

---

## Checklist for Complete Analysis

- [ ] PyTorch Profiler: Top 5 operations by time identified
- [ ] Roofline: Arithmetic intensity calculated, bottleneck classified (compute vs memory)
- [ ] Nsight Systems: Timeline shows sequential kernel execution
- [ ] Nsight Compute: L2 hit rate and warp stall breakdown documented
- [ ] Metrics table: Before/after comparison ready for optimization report
- [ ] Optimization prioritization: Top 3 targets identified with expected speedups

**Next:** See GPU_OPTIMIZATION_ROADMAP.md for implementation details.
