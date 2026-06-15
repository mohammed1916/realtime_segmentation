# GPU Optimization Iteration Checklist

Quick reference for running each optimization iteration.

---

## Pre-Iteration Checklist

### Setup
- [ ] Code baseline is clean (no uncommitted changes)
- [ ] Previous iteration results are saved
- [ ] GPU is idle (no other processes using it)
- [ ] Measurement script is ready
- [ ] Input data is loaded and pinned to device if large

### Baseline Capture (Iteration 1 only)
- [ ] Measure 20+ runs with fresh process
- [ ] Calculate average, min, max, std dev
- [ ] Save to JSON with timestamp
- [ ] Take note of memory peak

---

## During Iteration

### Step 1: Measure Baseline (5 min)

```bash
python measure_iteration.py --output metrics_before.json
```

**Checklist:**
- [ ] Captured ≥20 runs
- [ ] Calculated statistics
- [ ] Output saved
- [ ] Noted latency (ms), memory (MB), variance (ms)

**Record:**
```
Before: _____ ms ± _____ ms (std)
Memory: _____ MB
```

---

### Step 2: Analyze Baseline Metrics (10 min)

**Latency Analysis:**
```
Latency: _____ ms
Is this acceptable for deployment? [Y/N]
```

**Bandwidth Analysis:**
```python
# Calculate utilization
achieved_bw = (bytes_transferred / kernel_time_s) / 1e9  # GB/s
peak_bw = 1008  # RTX 4090; adjust for your GPU
utilization_pct = (achieved_bw / peak_bw) * 100

Achieved bandwidth: _____ GB/s
Peak bandwidth:     _____ GB/s
Utilization:        _____ % → Primary bottleneck: ________
```

**Classification:**
- [ ] Memory Bandwidth (< 50% utilization) → Try data reduction
- [ ] Occupancy (< 50%) → Try register reduction
- [ ] Compute Saturation (> 70% BW util) → Try algorithmic change
- [ ] Synchronization → Try kernel fusion

---

### Step 3: Choose Optimization (5 min)

**Decision Matrix Lookup:**

| If This Matches | Try This | Expected | Effort |
|---|---|---|---|
| BW util < 20% | FP16 precision | +30-60% | LOW |
| BW util 20-50% | Kernel fusion | +10-25% | MED |
| BW util 50-70% + L2 < 40% | Tiling | +10-20% | HIGH |
| Occupancy < 50% | Register reduction | +5-15% | MED |

**Selection:**
- Bottleneck: _______________
- Optimization: ______________
- Expected speedup: _____ %
- Estimated effort: _____ hours

---

### Step 4: Implement Optimization (Variable)

**Code Change:**
```python
# BEFORE:
[paste old code]

# AFTER:
[paste new code]

# Verification:
# - [ ] Code compiles/runs
# - [ ] Produces correct output (spot check)
# - [ ] No correctness regressions
```

**Version Control:**
```bash
git add .
git commit -m "Iteration N: [optimization name]"
```

---

### Step 5: Measure After (5 min)

```bash
python measure_iteration.py --output metrics_after.json
```

**Checklist:**
- [ ] Captured ≥20 runs
- [ ] Calculated statistics
- [ ] Output saved
- [ ] Noted latency (ms), memory (MB), variance (ms)

**Record:**
```
Before: _____ ms ± _____ ms
After:  _____ ms ± _____ ms
Speedup: _____ x (_____ %)
Variance change: _____ ms
```

---

### Step 6: Analyze Results (10 min)

**Impact Assessment:**

| Metric | Before | After | Change | Status |
|--------|--------|-------|--------|--------|
| Latency | ___ ms | ___ ms | ___% | [ ] ✓ [ ] ✗ [ ] ⚠ |
| Memory | ___ MB | ___ MB | ___% | [ ] ✓ [ ] ✗ [ ] ⚠ |
| Variance | ___ ms | ___ ms | ___% | [ ] ✓ [ ] ✗ [ ] ⚠ |

**Success Criteria:**
- [ ] Latency improved > 3% (measurable)
- [ ] Variance is stable (< 5% of latency)
- [ ] No crash or correctness regression
- [ ] Memory is same or better

**ROI Calculation:**

```
Speedup achieved:    _____ %
Time spent:          _____ hours
ROI (speedup/time):  _____ %/hour

Compare to prior iteration:
- This iteration ROI: _____ %/hr
- Prior iteration ROI: _____ %/hr
- Trend: Improving / Flat / Declining ________
```

---

### Step 7: Decision: Continue or Stop? (5 min)

**Checklist (if ALL boxes checked, continue to next iteration):**

- [ ] Latency improved > 5%
- [ ] Speedup/time ROI > prior iteration
- [ ] Bottleneck has changed (new primary bottleneck identified)
- [ ] More optimizations make sense
- [ ] Effort budget remaining

**Checklist (if ANY box checked, STOP):**

- [ ] Speedup < 5%
- [ ] Latency goal achieved (e.g., < 20 ms)
- [ ] ROI < prior iteration (diminishing returns)
- [ ] Bandwidth utilization > 70% (needs algorithm change)
- [ ] Effort budget exhausted

**Decision:** [ ] Continue [ ] Stop

**If STOP:**
- [ ] Document final results
- [ ] Clean up code
- [ ] Create summary report
- [ ] Prepare for deployment

**If CONTINUE:**
- [ ] Update baseline to current metrics
- [ ] Identify new primary bottleneck
- [ ] Go to Step 1

---

## Post-Iteration

### Results Logging

Create `iterations_log.json`:

```json
{
  "iteration_history": [
    {
      "iteration": 1,
      "name": "Baseline FP32",
      "timestamp": "2026-06-15T10:00:00Z",
      "metrics_before": null,
      "metrics_after": {
        "latency_ms": 33.60,
        "memory_mb": 806,
        "bandwidth_gbps": 195.5
      },
      "change": "none (baseline)",
      "speedup_x": 1.0,
      "effort_hours": 0
    },
    {
      "iteration": 2,
      "name": "FP16 Mixed Precision",
      "timestamp": "2026-06-15T10:30:00Z",
      "metrics_before": {
        "latency_ms": 33.60,
        "memory_mb": 806
      },
      "metrics_after": {
        "latency_ms": 20.81,
        "memory_mb": 403,
        "bandwidth_gbps": 97.75
      },
      "change": "torch.amp.autocast('cuda')",
      "speedup_x": 1.61,
      "speedup_pct": 61.5,
      "effort_hours": 0.5,
      "roi_pct_per_hour": 123.0,
      "decision": "accept - high ROI"
    },
    {
      "iteration": 3,
      "name": "Kernel Fusion (Conv + ReLU)",
      "timestamp": "2026-06-15T14:00:00Z",
      "metrics_before": {
        "latency_ms": 20.81,
        "memory_mb": 403
      },
      "metrics_after": {
        "latency_ms": 18.23,
        "memory_mb": 403,
        "bandwidth_gbps": 97.75
      },
      "change": "custom fused kernel",
      "speedup_x": 1.14,
      "speedup_pct": 12.4,
      "effort_hours": 3.0,
      "roi_pct_per_hour": 4.1,
      "decision": "stop - diminishing ROI"
    }
  ],
  "final_summary": {
    "total_speedup_x": 1.84,
    "total_effort_hours": 3.5,
    "total_roi": 0.53,
    "baseline_latency_ms": 33.60,
    "final_latency_ms": 18.23
  }
}
```

---

### Summary Document

Create `ITERATION_SUMMARY.md`:

```markdown
# Optimization Summary

## Iterations Completed

| Iter | Name | Speedup | Effort (hrs) | ROI | Decision |
|------|------|---------|---|---|---|
| 1 | Baseline | 1.0x | 0 | - | baseline |
| 2 | FP16 | 1.61x | 0.5 | 123x/hr | ✓ accept |
| 3 | Fusion | 1.14x | 3.0 | 4x/hr | ✗ stop |

## Total Results

```
Baseline:       33.60 ms
Final:          20.81 ms
Speedup:        1.61x
Total Effort:   0.5 hours
```

## Bottleneck Analysis

| Phase | Bottleneck | Signal |
|-------|---|---|
| Baseline | Memory BW | 19.4% utilization |
| After FP16 | Memory BW | 9.7% utilization (still low) |
| Decision | Stop | Diminishing ROI for further optimizations |

## Key Insights

1. **FP16 is high-impact:** 61% speedup, 1 line of code
2. **Law of diminishing returns:** Each iteration had lower ROI
3. **Memory-bound limit:** Even with 2× data reduction, still BW limited
4. **Deployment ready:** Current 20.81 ms meets requirements

## Next Steps

- [ ] Deploy with FP16 to production
- [ ] Monitor real-world latency
- [ ] Consider INT8 quantization if <15ms needed
- [ ] Revisit if new bottleneck emerges
```

---

## Quick Metrics Reference

### Memory Bandwidth

```python
# Measure actual bandwidth
achieved_bw = (bytes_moved_gb * 1000) / (kernel_time_ms)

# For RTX 4090
peak_bw = 1008  # GB/s

# Utilization
utilization = (achieved_bw / peak_bw) * 100
```

### Latency Calculation

```python
import numpy as np

times = [...]  # list of kernel times in ms
latency = np.mean(times[5:-5])  # Skip warmup/cooldown
std_dev = np.std(times[5:-5])

# Report as: X.XX ± Y.YY ms
```

### Expected Speedups by Optimization Type

```
FP16 precision:              1.5-2.0x (if memory-bound)
Kernel fusion:               1.1-1.3x (10-30% improvement)
Tiling/blocking:             1.1-1.2x (10-20% improvement)
INT8 quantization:           2.0-3.0x (if memory-bound)
Register optimization:       1.05-1.15x (5-15% improvement)
Algorithm change:            1.5-3.0x (major changes only)
```

---

## Troubleshooting

### "Speedup shows 1.0x (no improvement)"

**Possible causes:**
1. Variance is high → Need more runs (>20)
2. Optimization didn't apply → Check code path
3. Compiler optimized both versions the same → Check diff
4. GPU throttled → Run with GPU at fixed frequency

**Fix:**
```python
# Increase runs for better averaging
python measure_iteration.py --runs 50

# Verify code change applied
grep -n "new_optimization" model.py

# Check GPU state
nvidia-smi
```

### "Latency worse than before"

**Possible causes:**
1. Introduced unnecessary overhead → Check critical path
2. Register pressure increased → Check occupancy
3. Memory access pattern degraded → Check coalescing

**Fix:**
```bash
# Revert and try different approach
git revert HEAD
python measure_iteration.py --output metrics_after_revert.json
```

### "Memory usage increased"

**Possible causes:**
1. Caching intermediate results → Check allocation
2. Larger buffers needed → Check algorithm
3. Memory fragmentation → Run multiple times

**Fix:**
```python
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
```

---

## Iteration Workflow (Copy/Paste Ready)

```bash
#!/bin/bash

# Iteration N: [Optimization Name]

echo "=== Step 1: Baseline Measurement ==="
python measure_iteration.py --output metrics_before.json

echo "=== Step 2: Implement Optimization ==="
# [Make code changes]
# [Run verification]

echo "=== Step 3: After Measurement ==="
python measure_iteration.py --output metrics_after.json

echo "=== Step 4: Compare Results ==="
python compare_metrics.py metrics_before.json metrics_after.json

echo "=== Step 5: Decision ==="
echo "ROI analysis:"
echo "  Speedup: ?x"
echo "  Effort:  ? hours"
echo "  ROI:     ?x/hr"
echo ""
echo "Continue? [y/n]"
read -r decision

if [ "$decision" = "y" ]; then
    echo "=== Iteration Complete ==="
    git add .
    git commit -m "Iteration N: [Optimization Name]"
    echo "Ready for next iteration"
else
    echo "=== Final Deployment ==="
    # [Package final model]
fi
```

---

*Last Updated: 2026-06-15*
*Use this checklist for each optimization iteration*
