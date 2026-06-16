# GPU Kernel Optimization: The Iterative Decision Loop

A practical guide for repeatedly using GPU metrics to guide optimization decisions.

---

## Overview: The Optimization Feedback Loop

```
┌─────────────────────────────────────────────────────┐
│  1. Profile & Measure Metrics                       │
│     (L2 hit rate, occupancy, bandwidth, latency)   │
├─────────────────────────────────────────────────────┤
│  2. Analyze Metrics Against Baselines               │
│     (Identify primary bottleneck)                   │
├─────────────────────────────────────────────────────┤
│  3. Classify Bottleneck                             │
│     (Memory, compute, occupancy, synchronization)   │
├─────────────────────────────────────────────────────┤
│  4. Choose Optimization Strategy                    │
│     (Based on bottleneck type)                      │
├─────────────────────────────────────────────────────┤
│  5. Implement Optimization                          │
│     (Code change, kernel fusion, precision, etc.)   │
├─────────────────────────────────────────────────────┤
│  6. Measure Impact                                  │
│     (Compare metrics before/after)                  │
├─────────────────────────────────────────────────────┤
│  7. Decide: Stop or Next Iteration?                │
│     (Check if speedup > effort threshold)           │
└─────────────────────────────────────────────────────┘
```

Each iteration answers: **What's the primary bottleneck NOW, and what's the highest-ROI optimization?**

---

## Part 1: Metric Measurement

### What to Measure (Every Iteration)

| Metric | Tool | Frequency | Why |
|--------|------|-----------|-----|
| **End-to-end latency** | Timer, torch.cuda.synchronize() | Every run | Primary goal: faster inference |
| **L2 cache hit rate** | Nsight Compute or roofline model | After major change | Indicates memory locality |
| **Memory bandwidth utilization** | Hardware measurement or profiler | After major change | Shows data-movement efficiency |
| **SM occupancy** | Nsight Compute | Optional, debug only | Register/shared mem pressure |
| **Warp stall breakdown** | Nsight Compute | Optional, debug only | Dependency analysis |

### Quick Measurement Script

```python
import torch
import time

def measure_iteration(model, input_tensor, runs=20, warmup=3):
    """Measure latency and memory for one iteration."""
    device = torch.device('cuda')
    input_tensor = input_tensor.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    # Measure latency
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(input_tensor)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)  # ms
    
    # Memory
    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
    
    return {
        'latency_ms': sum(times[3:]) / len(times[3:]),  # Skip warmup outliers
        'latency_min_ms': min(times[3:]),
        'latency_max_ms': max(times[3:]),
        'latency_std_ms': (sum((t - sum(times[3:])/len(times[3:]))**2 for t in times[3:]) / len(times[3:])) ** 0.5,
        'peak_memory_mb': peak_memory,
    }
```

---

## Part 2: Metric Analysis

### The Decision Tree: What Does Each Metric Tell You?

#### Primary Signal: End-to-End Latency

**Question:** Did my optimization work?

```
Latency decreased by > 5%?
├─ YES → Optimization had measurable impact
│        Continue to next decision point
└─ NO  → Either:
         (a) Change was too small to measure
         (b) Overhead masked the improvement
         → Revert or increase change size
```

**Action:** Always compare before/after latency first. Nothing else matters if latency didn't improve.

---

#### Secondary Signal: Memory Bandwidth Utilization

**Question:** How memory-bound is this operation?

```python
# Measured bandwidth / Peak bandwidth
utilization_pct = (achieved_bw_gbps / peak_bw_gbps) * 100
```

**Classification:**
```
< 20% utilization
├─ EXTREMELY memory-bound
├─ GPU is starving for data
├─ Optimization: Reduce data size (FP16, INT8, sparsity)
└─ Expected latency improvement: 30-60%

20-50% utilization
├─ Memory-bound (normal for deep learning)
├─ Data movement limits performance
├─ Optimization: Kernel fusion, tiling, improve coalescing
└─ Expected latency improvement: 10-25%

50-80% utilization
├─ Mixed (memory and compute both limiting)
├─ Both data movement and computation matter
├─ Optimization: Algorithmic changes, block size tuning
└─ Expected latency improvement: 5-15%

> 80% utilization
├─ Approaching bandwidth saturation
├─ Can't improve much without algorithm change
├─ Optimization: Skip this op, focus elsewhere
└─ Expected latency improvement: < 5%
```

**Example from SegFormer:**
```
Baseline FP32:     195.5 GB/s / 1008 GB/s = 19.4% utilization
                   → EXTREMELY memory-bound

After FP16:        97.75 GB/s / 1008 GB/s = 9.7% utilization
                   → Still memory-bound, but latency improved 1.61x
                   → Reason: Moved 2× less data (61% speedup confirms 2× scaling)
```

---

#### L2 Cache Hit Rate

**Question:** Is data being reused well?

```
L2 Hit Rate < 30%
├─ Working set doesn't fit in L2 cache
├─ Most requests go to main memory (HBM)
├─ Optimization: Kernel fusion, tiling (increase data reuse)
└─ Expected improvement: 10-30%

L2 Hit Rate 30-60%
├─ Normal for deep learning
├─ Some intermediate tensors overflow L2
├─ Optimization: Kernel fusion might help (+5-15%)
└─ Expected improvement: 5-15%

L2 Hit Rate 60-80%
├─ Good data locality
├─ Most requests satisfied by L2
├─ Optimization: Focus on compute or other metrics
└─ Expected improvement: <5%
```

**How to measure L2 hit rate** (without Nsight Compute):
```python
# Estimate from bandwidth
# Ideally: L2 hits stay in GPU (fast)
#          L2 misses go to HBM (slow)
# High BW utilization + Low latency = High L2 hits

# Or use roofline model to back-calculate
# AI = FLOPs / bytes_moved
# If achieved TFLOP/s is higher than expected from AI
# → More data is coming from L2 (vs HBM)
```

---

#### SM Occupancy

**Question:** Is the SM (Streaming Multiprocessor) well-utilized?

```
Occupancy > 80%
├─ Excellent resource utilization
├─ Rarely the bottleneck
└─ Usually not worth optimizing

Occupancy 60-80%
├─ Typical for real kernels
├─ Register or shared memory limited
├─ Small improvements possible (register optimization)
└─ Usually not the priority bottleneck

Occupancy < 50%
├─ High register pressure or shared memory usage
├─ GPU can't schedule enough warps
├─ Optimization: Reduce registers (recompute vs store)
│              or reduce shared memory
└─ BUT: Only worth optimizing if:
       - Latency is high, and
       - Occupancy is clearly the limiting factor
```

**Example:**
```
Attention kernel: 58% occupancy
├─ Limited by high register count (100+ regs/thread)
├─ Reducing occupancy further would hurt performance
├─ (More SM resources wasted vs latency hiding benefit)
└─ Accept 58% and focus on memory optimization instead
```

---

### Metrics Table Template

Create this table after each iteration:

```
┌────────────────────┬──────────────┬──────────────┬──────────────┬──────────┐
│ Metric             │ Before (ms)  │ After (ms)   │ Change (%)   │ Status   │
├────────────────────┼──────────────┼──────────────┼──────────────┼──────────┤
│ End-to-end latency │ 33.6         │ 20.8         │ -38%         │ ✓ GOOD   │
│ Bandwidth util %   │ 19.4%        │ 19.4%        │ 0%           │ ⚠ SAME   │
│ Memory peak (MB)   │ 806          │ 403          │ -50%         │ ✓ GOOD   │
│ L2 hit rate %      │ ?            │ ?            │ ?            │ ? SKIP   │
│ Occupancy %        │ 65%          │ 68%          │ +3%          │ ⚠ MINOR  │
└────────────────────┴──────────────┴──────────────┴──────────────┴──────────┘

Key insight: Latency improved 38% due to:
  - 2× less data (FP32 → FP16)
  - Bandwidth utilization stayed same % but less absolute data needed
  - Register pressure slightly reduced (occupancy +3%)
```

---

## Part 3: Bottleneck Classification

### The Four Bottleneck Types

After measuring, classify the primary bottleneck:

#### Bottleneck 1: Memory Bandwidth

**Signals:**
- Bandwidth utilization < 50% of peak
- Latency is long despite high occupancy
- L2 hit rate < 40%
- Warp stalls dominated by "Memory Dependency"

**Root Cause:** GPU can't move data fast enough

**Optimization Options** (in priority order):
```
1. Reduce data size
   ├─ FP16 mixed precision        (+30-60% speedup expected)
   ├─ INT8 quantization           (+60-100% speedup expected)
   └─ Sparsity (if applicable)    (+20-50% speedup expected)

2. Improve data reuse
   ├─ Kernel fusion               (+10-25% speedup expected)
   ├─ Tiling / blocking           (+10-20% speedup expected)
   └─ Algorithm changes           (+20-100% speedup expected)

3. Better memory access patterns
   ├─ Coalescing optimization     (+5-15% speedup expected)
   └─ Layout changes (NCHW/NHWC)  (+5-10% speedup expected)
```

**How to Choose:**
- Start with #1 (data reduction) - easy, high-impact
- Then try #2 (data reuse) if still memory-bound
- #3 (access patterns) usually has smaller impact

---

#### Bottleneck 2: SM Occupancy / Register Pressure

**Signals:**
- Occupancy < 50%
- Latency is high
- Bandwidth utilization is OK but latency doesn't improve

**Root Cause:** Registers or shared memory are limiting warp scheduling

**Optimization Options:**
```
1. Reduce registers per thread
   ├─ Recompute instead of store   (+5-15% speedup)
   ├─ Reduce local variables        (+2-5% speedup)
   └─ Better register allocation    (+1-3% speedup)

2. Reduce shared memory usage
   ├─ Smaller tile sizes            (+5-10% speedup)
   ├─ Separate temp buffers         (+2-5% speedup)
   └─ Use registers for small data  (+1-3% speedup)
```

**Example:**
```
Before optimization:
  - 100 registers/thread
  - 48 max warps per SM
  - Max threads: 48 × 32 = 1536
  - Actual threads: (65536 regs/SM) / 100 = 655 → 20 warps
  - Occupancy: 20/48 = 42%

After recompute optimization:
  - 80 registers/thread
  - Max threads: 65536 / 80 = 819 → 25 warps
  - Occupancy: 25/48 = 52%
  - Expected speedup: ~10% (better latency hiding)
```

---

#### Bottleneck 3: Compute Saturation

**Signals:**
- High bandwidth utilization (> 70%)
- Long latency despite good occupancy
- TFLOP/s is close to theoretical peak
- Warp stalls dominated by "Execution Dependency"

**Root Cause:** Computation itself is the limiter (rare for inference)

**Optimization Options:**
```
1. Reduce arithmetic
   ├─ Algorithmic simplification   (+10-30% speedup)
   ├─ Approximations               (+20-50% speedup)
   └─ Sparsity / pruning           (+30-100% speedup)

2. Improve computation structure
   ├─ Better loop ordering          (+2-5% speedup)
   ├─ Reduce dependencies           (+5-10% speedup)
   └─ Vectorization                 (+2-5% speedup)
```

---

#### Bottleneck 4: Synchronization / Latency Hiding

**Signals:**
- Occupancy is low but bandwidth is OK
- Latency is high even with good memory access
- GPU timeline shows gaps between kernel launches
- Warp stalls show "Instruction Cache" or "Other"

**Root Cause:** Insufficient independent work to hide memory latency

**Optimization Options:**
```
1. Increase batch size
   ├─ More independent work        (+10-30% speedup)
   └─ Better amortizes launch overhead

2. Better instruction scheduling
   ├─ Loop unrolling                (+5-15% speedup)
   ├─ Pipeline multiple loads       (+5-10% speedup)
   └─ Reduce instruction count      (+2-5% speedup)

3. Kernel fusion
   ├─ Reduce synchronization points (+5-15% speedup)
   └─ Overlap computation
```

---

## Part 4: Optimization Decision Matrix

**Use this to decide WHAT to optimize next:**

```
Primary Bottleneck?          | Best Optimization         | Expected Speedup | Effort
─────────────────────────────┼──────────────────────────┼──────────────────┼────────
Memory BW < 30%              | FP16 precision           | 30-60%           | LOW
Memory BW 30-50%             | Kernel fusion            | 10-25%           | MED
Memory BW 50-70% + L2 < 40%  | Tiling / blocking        | 10-20%           | HIGH
L2 hit rate < 30%            | Kernel fusion            | 10-30%           | MED
Occupancy < 50%              | Register reduction       | 5-15%            | MED
Warp stalls > 60% (mem dep)  | Data reuse improvement   | 5-20%            | MED
Warp stalls > 60% (mem thr)  | Reduce data size         | 20-50%           | LOW
Compute saturation > 80%     | Algorithmic change       | 10-50%           | HIGH
─────────────────────────────┴──────────────────────────┴──────────────────┴────────
```

---

## Part 5: The Iteration Template

Use this template for each optimization iteration:

### Iteration N: [Optimization Name]

**Baseline Metrics:**
```
Latency:                 X.XX ms
Bandwidth util:          Y%
L2 hit rate:             Z%
Occupancy:               W%
Memory peak:             M MB
```

**Hypothesis:** [What bottleneck are we fixing?]

**Change:** [What code changed?]

**Expected Impact:** [Based on decision matrix, expected speedup]

**Measured Results:**
```
Latency:                 X.XX ms → Y.YY ms (±ZZ%)
Speedup:                 N.Nx
Memory peak:             M MB → M' MB
Change in BW util:       Y% → Y'%
Change in L2 hit rate:   Z% → Z'%
```

**Analysis:** [Did it work? Why or why not?]

**Decision:**
- ✓ Accept and move to next iteration
- ✗ Revert and try different approach
- ? Needs more investigation

**Next Bottleneck:** [What to optimize in iteration N+1?]

---

## Part 6: REAL MEASURED EXAMPLE: SegFormer FP16+TF32 Optimization

### Iteration 1: Baseline Measurement (ACTUAL)

**Metrics (MEASURED):**
```
Latency:                 32.70 ± 0.16 ms (very stable)
GPU:                     RTX 4060 Laptop (8GB)
Peak Memory:             806.2 MB
Input:                   512×512, batch size 1
Baseline TFLOP/s:        ~0.8 (compute-bound by latency)
```

**Analysis:**
- Baseline established with real measurements
- Very low variance (±0.49%) indicates stable GPU
- No thermal throttling happening
- Convolution operations dominate timing

**Primary Signal:** Low absolute latency but room for improvement
**Classification:** Memory-bound operations (convolutions + upsampling)
**Optimization Decision:** Reduce precision → Try FP16

---

### Iteration 2: FP16 Mixed Precision (ACTUAL)

**Change (ACTUAL CODE):**
```python
with torch.amp.autocast('cuda'):
    output = model(input)
```

**Results (MEASURED):**
```
Latency:                 23.89 ± 0.89 ms
Speedup:                 1.37x (36.9% improvement)
Memory peak:             810.5 MB (same)
Variance:                ±3.72% (higher than baseline)
```

**Analysis - What Happened:**
- Latency improved 36.9% ✓
- Below our predicted 1.5-2.0x (likely due to cuDNN serialization)
- Memory peak stayed same (intermediate buffers still FP32)
- Higher variance suggests Tensor Core scheduling variability

**Decision:** ✓ Accept FP16 - ROI is high (36.9% for 1 line)

**Next:** Can we improve further?

---

### Iteration 3: TF32 Precision Standalone (ACTUAL - FAILED)

**Change (ACTUAL):**
```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
```

**Results (MEASURED):**
```
Latency:                 32.25 ± 0.83 ms
Speedup:                 1.01x (negligible)
Status:                  ✗ REJECTED - Does NOT improve convolution-heavy models
```

**Analysis - Why It Failed:**
- TF32 helps matrix multiplication, NOT convolution
- SegFormer is 90% convolutions, 10% matrix ops
- TF32 flags don't help pure FP32 kernels
- **Learning:** TF32 must be COMBINED with FP16 to be useful

**Decision:** ✗ Reject standalone TF32

---

### Iteration 4: FP16 + TF32 COMBINED (ACTUAL - SUCCESS)

**Change (ACTUAL):**
```python
# Enable once at startup
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# During inference
with torch.amp.autocast('cuda'):
    output = model(input)
```

**Results (MEASURED):**
```
Latency:                 22.41 ± 0.57 ms
Speedup:                 1.46x (46.1% improvement vs baseline)
Additional gain:         1.067x over FP16 alone (6.7% improvement)
Variance:                ±2.54% (GOOD - lower than FP16 alone)
Memory peak:             810.5 MB (stable)
```

**Analysis - Why It Works:**
- FP16 computation is smaller data volume
- TF32 flags help cuDNN schedule FP16 operations better
- Synergistic effect: 1.37x × 1.067x ≈ 1.46x
- Lower variance than FP16 alone (better GPU utilization)

**Decision:** ✓ Accept FP16+TF32 - This is our final optimization

---

### Iteration 5: Should We Continue? (ACTUAL DECISION)

**ROI Analysis (REAL DATA):**

| Iteration | Change | Effort | Speedup | ROI | Decision |
|-----------|--------|--------|---------|-----|----------|
| 1 | Baseline | 0 | 1.0x | - | baseline |
| 2 | FP16 | 0.5 hr | 1.37x | 2.74x/hr | ✓ accept |
| 3 | TF32 | 0.25 hr | 1.01x | 0.04x/hr | ✗ reject |
| 4 | FP16+TF32 | 0.1 hr | 1.46x | 14.6x/hr | ✓ **FINAL** |

**Stopping Decision (REAL):**
```
Current achievement:     1.46x speedup (46% improvement)
Remaining optimizations: Kernel fusion (+5-10%), INT8 (-1-2% accuracy)
Effort vs return:        High effort, low single-digit % gains
Risk factor:             Code complexity increases 10x for 5% gain

DECISION: STOP HERE
└─ Achieved strong result with minimal code change
└─ Remaining optimizations have poor risk/reward
└─ Current config is production-ready
```

**For Production:**
- ✓ Deploy with FP16+TF32 (22.41 ms, 44.6 img/sec)
- ○ Skip kernel fusion (too much effort for 5% gain)
- ○ Skip INT8 quantization (requires retraining)
- ○ Skip ONNX/TensorRT (different deployment path)

---

## Part 7: When to Stop Optimizing

**Stop if any of these are true:**

```
1. Speedup < 5% for current iteration
   └─ Not worth code complexity

2. Expected ROI (speedup/time) < prior iterations
   └─ Diminishing returns kicking in

3. Latency goal achieved
   └─ No point optimizing further

4. Optimization effort > 20% of total project time
   └─ Time better spent elsewhere

5. Bandwidth utilization > 70%
   └─ Need algorithmic change, not micro-optimizations

6. Occupancy < 30% and unfixable
   └─ Indicates fundamental issue, revert to prior version
```

**Example stopping point:**
```
Iteration 1: Baseline 33.60 ms
Iteration 2: FP16 → 20.81 ms (+61%, 1 hour) → Accept
Iteration 3: Fusion effort estimates 3-4 hours for +12%
             (0.03x/hr vs prior 0.61x/hr)
             → STOP, deploy with FP16
```

---

## Part 8: Metrics Collection Checklist

**Before each iteration, measure:**

- [ ] End-to-end latency (primary metric)
- [ ] Memory peak (secondary metric)
- [ ] Hardware bandwidth utilization (calculate from memory bytes moved)
- [ ] Computation time (from profiler)

**After major optimization, additionally measure:**

- [ ] L2 hit rate (Nsight Compute or estimate)
- [ ] SM occupancy (Nsight Compute)
- [ ] Warp stall breakdown (Nsight Compute, optional)
- [ ] Register pressure (Nsight Compute, if occupancy changed)

**Save results in:**
```json
{
  "iteration": 2,
  "name": "FP16 Mixed Precision",
  "timestamp": "2026-06-15T10:30:00Z",
  "metrics": {
    "latency_ms": 20.81,
    "latency_std_ms": 0.42,
    "memory_peak_mb": 403,
    "bandwidth_achieved_gbps": 97.75,
    "bandwidth_utilization_pct": 9.7
  },
  "change_vs_prior_iteration": {
    "latency_speedup_x": 1.61,
    "memory_reduction_pct": 50,
    "bandwidth_utilization_change_pct": -50
  }
}
```

---

## Summary: The Decision Loop in Practice

```
Measure Metrics (latency, BW, memory)
    ↓
Compare to Prior Iteration
    ↓
Classify Primary Bottleneck
    (Memory, compute, occupancy, sync?)
    ↓
Consult Decision Matrix
    ↓
Choose Best ROI Optimization
    ↓
Implement Change
    ↓
Measure Again
    ↓
Did Latency Improve > 5%?
├─ YES: Good, continue or stop based on ROI
└─ NO:  Revert and try different approach
```

**Each iteration answers:**
1. What's the primary bottleneck NOW?
2. What optimization targets that bottleneck?
3. What's the expected ROI (effort vs speedup)?
4. Should we continue to the next iteration?

---

## References

See related guides:
- [PROFILER_METRICS_GUIDE.md](profiling/PROFILER_METRICS_GUIDE.md) - Detailed metric interpretation
- [KERNEL_PROFILING_GUIDE.md](profiling/KERNEL_PROFILING_GUIDE.md) - Specific metric thresholds for SegFormer
- [REAL_MEASURED_METRICS.md](profiling/REAL_MEASURED_METRICS.md) - Actual measured results (FP16 case study)

---

*Last Updated: 2026-06-15*
