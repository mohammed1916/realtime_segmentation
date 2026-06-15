# Nsight Compute Quick Start - Profile SegFormer

## Step 1: Profile Baseline (FP32)

```bash
cd c:\Users\BBBS-AI-01\d\realtime_segmentation\gpu_optimization

ncu -o profile_baseline.ncu-rep python run_nsight_profile.py
```

This will:
- Load SegFormer model
- Warm up GPU
- Run inference (Nsight Compute captures all kernel metrics)
- Save results to `profile_baseline.ncu-rep`

**Time:** ~30-60 seconds

## Step 2: Profile Optimized (FP16)

```bash
ncu -o profile_fp16.ncu-rep python run_nsight_profile.py --fp16
```

This profiles the same model with FP16 mixed precision enabled.

## Step 3: View Results in GUI

```bash
# View baseline
ncu-ui profile_baseline.ncu-rep

# Or FP16
ncu-ui profile_fp16.ncu-rep
```

This opens Nsight Compute GUI with all kernel-level metrics.

---

## What to Look For in the GUI

### 1. L2 Cache Metrics

**Path:** Memory Workload > L2 Cache

**Key Metrics:**
- **L2 Hit Rate (%)** - What percentage of requests hit L2 cache
  - Good: >60%
  - Poor: <40%
  
- **L2 Bandwidth (GB/s)** - How much data flows through L2
  - Compare baseline vs FP16 (should be 2× difference)

- **L2 Read/Write Split** - Ratio of reads to writes
  - Convolution: High read/write ratio (more reads than writes)
  - BatchNorm: Streaming (balanced read/write)

### 2. SM Occupancy

**Path:** Occupancy > SM Occupancy

**Key Metrics:**
- **Occupancy (%)** - Percentage of SM capacity being used
  - Good: >70%
  - Register-limited: 50-70% (expected for conv)
  - Poor: <50%

- **Active Warps/SM** - How many warps running per SM
  - Max: 48 warps (RTX 40xx)
  - Conv 3×3: Usually 30-35 warps (register-limited)

- **Registers per Thread** - How much register space used
  - Conv: 60-100 regs per thread (high)
  - BatchNorm: 20-40 regs per thread (lower)

### 3. Memory Access Pattern

**Path:** Memory Workload > Memory Access Pattern

**Key Metrics:**
- **Global Memory Coalescing** - Efficiency of memory access
  - Good: >90% coalesced
  - NCHW format (PyTorch default): Should be >95%

- **L1 Hit Rate** - Percentage hitting L1 cache
  - Good: 30-50% (L2 is more important)

### 4. Warp Stall Reasons

**Path:** Execution > Warp State

**Key Metrics:**
- **Warp Stall Breakdown:**
  - Memory Dependency (%) - Waiting for memory
    - High (>50%): Normal for memory-bound ops
  - Memory Throttle (%) - Memory bandwidth saturated
    - High (>20%): DRAM bandwidth limit reached
  - Execution Dependency (%) - Waiting for arithmetic
    - Low (<10%): Normal for most ops

- **Warp Efficiency (%)** - Percentage time warps executing
  - Good: >80%
  - Poor: <70%

---

## Expected Results by Operation

### Convolution Kernels (cudnnConvolution)

**Expected Baseline (FP32):**
```
L2 Hit Rate:           40-50%
L2 Bandwidth:          200-250 GB/s
Occupancy:             65-75%
Memory Dependency:     50-60%
Memory Throttle:       15-25%
```

**Expected with FP16:**
```
L2 Hit Rate:           50-60% (better reuse)
L2 Bandwidth:          100-125 GB/s (2× reduction)
Occupancy:             70-80% (less register pressure)
Memory Dependency:     40-50% (less waiting)
Memory Throttle:       10-15% (less DRAM stress)
```

### BatchNorm Kernels (cudnnBatchNorm)

**Expected Baseline (FP32):**
```
L2 Hit Rate:           70-80%
L2 Bandwidth:          200-300 GB/s
Occupancy:             75-85%
Memory Dependency:     30-40%
```

**Expected with FP16:**
```
L2 Hit Rate:           80-85%
L2 Bandwidth:          100-150 GB/s
Occupancy:             80-90%
Memory Dependency:     20-30%
```

---

## Comparing Baseline vs FP16

### In the GUI:

1. **Open both profiles side by side:**
   - File > Open (open `profile_baseline.ncu-rep`)
   - File > Open (open `profile_fp16.ncu-rep` in new window)

2. **Check L2 Bandwidth:**
   - Baseline: ~200-250 GB/s
   - FP16: ~100-125 GB/s
   - Ratio: Should be 2:1 (FP16 uses 2× less bandwidth)

3. **Check Occupancy:**
   - Baseline: 65-75%
   - FP16: 70-80%
   - Should improve with less register pressure

4. **Check Memory Stalls:**
   - Baseline: High memory dependency (50-60%)
   - FP16: Lower memory dependency (40-50%)
   - Should improve because less data to move

---

## Reading the Data

### Example Conv Kernel Analysis

**Baseline (FP32):**
```
Kernel: cudnnConvolution
L2 Hit Rate: 42%          ← Working set partially in L2
L2 Read BW: 210 GB/s      ← 21% of peak 1008 GB/s
Occupancy: 68%            ← Register-limited
Memory Dependency: 55%    ← Mostly waiting for data
Conclusion: MEMORY-BOUND, register pressure limiting occupancy
```

**With FP16:**
```
Kernel: cudnnConvolution
L2 Hit Rate: 52%          ← Better reuse (2× less data)
L2 Read BW: 105 GB/s      ← Still only 10% of peak (2× reduction)
Occupancy: 74%            ← Better (less register pressure)
Memory Dependency: 42%    ← Less stalling (less data to wait for)
Conclusion: Still memory-bound but improved. More data fits in cache.
```

**Key insight:** FP16 works because:
1. 2× less data to move
2. Better L2 hit rate (more fits in cache)
3. Less warp stalling (waiting for less data)
4. Higher occupancy (registers are freed)

---

## Command Reference

### Profile specific metric set:

```bash
# Full metrics (larger output)
ncu --set full -o profile.ncu-rep python run_nsight_profile.py

# Essential metrics only (smaller output, faster)
ncu --set default_metrics -o profile.ncu-rep python run_nsight_profile.py

# Custom metrics
ncu --set sm_memory -o profile.ncu-rep python run_nsight_profile.py
```

### Export results to CSV:

```bash
ncu --import profile_baseline.ncu-rep --csv > metrics.csv
```

### Compare in command line:

```bash
# Show specific section
ncu --import profile_baseline.ncu-rep --section "Memory Workload"
ncu --import profile_fp16.ncu-rep --section "Memory Workload"
```

---

## Interpreting L2 Cache Hit Rate

### What it means:

```
L2 Hit Rate = L2 Hits / (L2 Hits + L2 Misses)

42% hit rate means:
- 42% of memory requests satisfied by L2 (fast, 400 cycles)
- 58% miss L2, go to HBM (slow, 300+ cycles)
```

### Why it matters:

```
Working Set Size: ~1 GB (full forward pass data)
L2 Cache Size: 5 MB (on RTX 4090)
Ratio: 200:1

With 42% hit rate:
- 42% of data accessed from L2 (fast)
- 58% accessed from HBM (slow, limited by bandwidth)

FP16 reduces working set to 500 MB:
- Ratio becomes 100:1
- More data can stay in L2
- Higher hit rate (52% vs 42%)
- Faster access patterns
```

---

## Quick Interpretation Guide

| Metric | Baseline | FP16 | Action |
|---|---|---|---|
| **L2 Hit Rate** | 40-50% | 50-60% | ✓ Improved (less data) |
| **L2 Bandwidth** | 200 GB/s | 100 GB/s | ✓ Improved (2× reduction) |
| **Occupancy** | 65-75% | 70-80% | ✓ Improved (less registers) |
| **Memory Dep** | 50-60% | 40-50% | ✓ Improved (less waiting) |
| **Mem Throttle** | 15-25% | 10-15% | ✓ Improved (less DRAM stress) |

**All metrics should improve with FP16** because we reduce data 2×.

---

## Next Steps After Profiling

1. **Confirm FP16 benefits:**
   - L2 metrics should show 2× less bandwidth
   - Occupancy should improve
   - Memory stalls should decrease

2. **Identify further optimization targets:**
   - If L2 hit rate <40%: Consider kernel fusion
   - If occupancy <60%: Register pressure limiting (algorithmic change needed)
   - If memory throttle >25%: Consider quantization to INT8

3. **Kernel-specific insights:**
   - Conv layers: Most improvement potential
   - BatchNorm: Already well-optimized
   - Upsampling: Good candidates for fusion

---

## Expected Runtime

- Baseline profile: 30-60 seconds
- FP16 profile: 30-60 seconds
- GUI opening: 5-10 seconds
- **Total: ~2 minutes for both profiles**

---

## Troubleshooting

### "ncu not found"
```bash
# Check installation
where ncu

# If not found, add to PATH:
setx PATH "%PATH%;C:\Program Files\NVIDIA\Nsight Compute\"
```

### Large profile file (>500 MB)
```bash
# Use smaller metric set
ncu --set default_metrics -o profile.ncu-rep python run_nsight_profile.py
```

### GUI won't open
```bash
# Try alternative viewer
ncu --import profile_baseline.ncu-rep --section "Memory Workload"
```

---

## Summary

This workflow will:
1. **Measure real L2 cache hit rates** (not theoretical)
2. **Show actual occupancy** (not theoretical)
3. **Profile memory bandwidth usage** (real hardware numbers)
4. **Compare FP32 vs FP16** (side by side)
5. **Guide next optimizations** (based on real data)

**Run it now to see the real metrics!**
