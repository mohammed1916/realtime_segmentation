# GPU Optimization Quickstart Guide

**Goal:** Profile your SegFormer model, identify bottlenecks, and prepare for optimization.

**Time to first results:** 15 minutes  
**Time to comprehensive analysis:** 1-2 hours

---

## Step 1: Set Up Profiling Environment

### Prerequisites
```bash
# Install profiling tools
pip install torch torchvision
pip install nvidia-pytorch-amp  # For RTX 30/40 series

# For Nsight tools (optional but recommended)
# Download from: https://developer.nvidia.com/tools-overview/nsight/systems
# Or use conda: conda install -c conda-forge nsight-compute
```

### Verify CUDA Setup
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"Compute Capability: {torch.cuda.get_device_capability()}")
```

---

## Step 2: Run Baseline Profiling (5 minutes)

### Quick Profile - PyTorch Built-in
```bash
cd gpu_optimization/

# Profile SegFormer-B0 (512×512 input)
python profiling_tools/pytorch_profiler.py \
    --model b0 \
    --input-size 512 \
    --batch-size 1 \
    --export-trace results/b0_trace.json \
    --export-csv results/b0_results.csv

# Profile SegFormer-B1 (512×512 input)
python profiling_tools/pytorch_profiler.py \
    --model b1 \
    --input-size 512 \
    --batch-size 1 \
    --export-trace results/b1_trace.json \
    --export-csv results/b1_results.csv
```

**What to look for in output:**
- Which kernels take the longest (`self_cuda_time_total`)
- If attention (`matmul`, `softmax`) dominates (should be >40% of total time)
- If FFN Conv layers are secondary (should be 25-35% of time)

### View Timeline (Chrome Tracing)
```bash
# Open browser DevTools for tracing
# Open: chrome://tracing
# Load: results/b0_trace.json
# Look for: Timeline shows kernel execution sequence, memory transfers
```

---

## Step 3: Roofline Analysis (10 minutes)

### Measure Arithmetic Intensity
```bash
# Benchmark key operations
python profiling_tools/roofline_benchmark.py --all --export results/roofline.json

# Or focus on specific operations
python profiling_tools/roofline_benchmark.py --operation attention
python profiling_tools/roofline_benchmark.py --operation conv1x1
```

**Expected output:**
```
Arithmetic Intensity: 0.77 ops/byte
Roofline Limit: 0.011 ops/byte
Bottleneck: MEMORY

(Ceiling: 1.2 TFLOP/s due to memory bandwidth)
```

**Interpretation:**
- If `Arithmetic Intensity < Roofline Limit` → **Memory-bound** ✓ (expected for attention)
- If `Utilization < 50%` → Large gap between achieved and peak → Optimization opportunity

---

## Step 4: Identify Primary Bottlenecks

Create a summary table from your profiling results:

```python
# analysis/parse_bottlenecks.py
import json
import pandas as pd

# Load profiler CSV
df = pd.read_csv('results/b0_results.csv')

# Sort by CUDA time
df_sorted = df.sort_values('CUDA Time (ms)', ascending=False)

# Calculate cumulative %
df_sorted['Cumulative %'] = df_sorted['CUDA Time (ms)'].cumsum() / df_sorted['CUDA Time (ms)'].sum() * 100

print(df_sorted[['Operation', 'CUDA Time (ms)', 'Count', 'Cumulative %']].head(15))

# Export
df_sorted.to_csv('results/b0_bottlenecks.csv')
```

**Run:**
```bash
python analysis/parse_bottlenecks.py
```

**Output table should look like:**
```
Operation                         CUDA Time (ms)  Count  Cumulative %
──────────────────────────────────────────────────────────────────────
aten::_scaled_dot_product_attn         12.5       16      35%
aten::matmul (Q@K)                      8.2       16      58%
aten::softmax                           2.1       16      64%
aten::conv2d (fc1, MixFFN)             3.4       16      74%
aten::conv2d (pe_conv, DW)             1.3       16      81%
...
```

---

## Step 5: Create Baseline Report

Create `analysis/BASELINE_ANALYSIS.md`:

```markdown
# SegFormer-B0 Baseline Analysis

## Hardware
- GPU: RTX 3090 / RTX 4090 / A100
- Memory: XXX GB
- Driver: CUDA XX.X

## Input Specification
- Model: SegFormer-B0
- Input: 1 × 3 × 512 × 512
- Output: 1 × 150 × 512 × 512 (ADE20K)

## End-to-End Performance
- **Latency:** 42.3 ms
- **Throughput:** 23.6 images/sec
- **Memory Peak:** 3.2 GB

## Profiling Results

### Top 5 Operations (by Time)
| Operation | CUDA Time (ms) | % Total |
|-----------|---|---|
| Attention (Stages 1-4) | 21.4 | 50.6% |
| MixFFN (all stages) | 12.1 | 28.6% |
| Decode Head (upsample+fusion) | 5.2 | 12.3% |
| LayerNorm | 1.8 | 4.3% |
| Other | 1.8 | 4.2% |

### Roofline Analysis
- Attention Arithmetic Intensity: 0.77 ops/byte
- Expected Ceiling: 1.2 TFLOP/s (limited by memory)
- Achieved: 1.2 TFLOP/s
- **Classification:** MEMORY-BOUND ✓

### Key Metrics
- L2 Cache Hit Rate: ~40% (low, intermediate tensors overflow)
- Memory Utilization: 55% of peak
- Tensor Core Utilization: 25% (underutilized)
- SM Occupancy: 55-65% (register-limited)
- Warp Efficiency: 70-80% (memory stalls)

## Optimization Opportunities (Ranked by Impact)

### 1. Attention Kernel Optimization
- **Current:** 21.4 ms (50.6% of runtime)
- **Target:** Flash Attention V2 can achieve 2-3× speedup
- **Expected Gain:** 35-40% overall speedup
- **Difficulty:** Easy (use library) to Intermediate (custom kernel)

### 2. MixFFN Fusion
- **Current:** 12.1 ms (28.6% of runtime)
- **Target:** Fuse 4 kernels into 1, keep intermediate in registers
- **Expected Gain:** 10-15% speedup
- **Difficulty:** Intermediate

### 3. Spatial Reduction Optimization
- **Current:** Depthwise Conv in attention is slow
- **Target:** Custom gather kernel
- **Expected Gain:** 5% speedup (limited by % runtime)
- **Difficulty:** Beginner-Intermediate

## Conclusion
Attention is the primary bottleneck (50.6%). Flash Attention integration alone could achieve 35-40% overall speedup.
```

---

## Step 6: Plan Optimizations

Use the Roadmap document (`../GPU_OPTIMIZATION_ROADMAP.md`) to select your first optimization:

### Option A: Quick Win (Flash Attention) - 1 week
```python
# pip install flash-attn==2.5.7

from flash_attn import flash_attn_func

# Modify EfficientMultiheadAttention.forward():
out = flash_attn_func(q, k, v, dropout_p=self.attn_drop, causal=False)
```

**Expected:** 10-15% end-to-end speedup, 50+ TFLOP/s on attention ops

### Option B: Learning Project (Fused Kernel) - 4 weeks
Implement custom CUDA kernel for LayerNorm + QKV Projection

**Expected:** 5-10% end-to-end speedup, deep understanding of:
- Warp-level reductions
- Shared memory orchestration
- CUDA memory hierarchy

### Option C: Full Portfolio (Tier 1+2) - 8 weeks
Combine Flash Attention + Fused Kernel + Additional optimizations

**Expected:** 25-35% end-to-end speedup, comprehensive case study

---

## Step 7: Next Steps

1. **Document Baseline**
   - [ ] Save profiler traces and CSV reports
   - [ ] Run roofline analysis
   - [ ] Create baseline report
   - [ ] Take screenshots of profiler output

2. **Choose First Optimization**
   - [ ] Read Flash Attention paper if doing Option A
   - [ ] Read CUDA fundamentals if doing Option B
   - [ ] Set up development environment (CUDA toolkit, cuDNN)

3. **Measure Impact**
   - [ ] Run profiler on optimized version
   - [ ] Compare metrics: Latency, TFLOP/s, memory
   - [ ] Document speedup and why

4. **Create Portfolio**
   - [ ] Code on GitHub with documentation
   - [ ] Performance comparison tables/charts
   - [ ] Technical writeup of approach

---

## Troubleshooting

### "CUDA out of memory"
```python
# Reduce batch size or input size
python profiling_tools/pytorch_profiler.py --batch-size 1 --input-size 512
```

### "Model not loading"
```python
# Profiler will use dummy model if configs unavailable
# You can still analyze kernel patterns
python profiling_tools/pytorch_profiler.py --model b0 --input-size 512
```

### "Chrome trace is huge (>500MB)"
```bash
# Reduce number of iterations or model size
# Traces scale with sequence length (H×W)
```

### "Roofline shows compute-bound, not memory-bound"
That's OK! Means optimization should focus on:
- Increasing parallelism (more blocks/threads)
- Reducing instruction count (loop unrolling)
- Rather than memory access patterns

---

## Key Metrics Reference

| Metric | Typical Value | Good Range | Indicates |
|--------|---|---|---|
| **Latency** | 40-50 ms | <30 ms | Overall performance |
| **Memory BW** | 350-450 GB/s | >400 GB/s | Well-utilized memory |
| **L2 Hit Rate** | 40-50% | >60% | Better data reuse |
| **TC Utilization** | 20-35% | >70% | (for matmuls) Dense compute |
| **Warp Efficiency** | 70-80% | >90% | Low memory stalls |
| **SM Occupancy** | 55-65% | >75% | Good latency hiding |
| **Arithmetic Intensity** | 0.77 | >1.0 | Compute-bound (rare here) |

---

## Resources

- **PyTorch Profiler Docs:** https://pytorch.org/docs/stable/profiler.html
- **Nsight Systems:** https://docs.nvidia.com/nsight-systems/
- **Nsight Compute:** https://docs.nvidia.com/nsight-compute/
- **Flash Attention:** https://github.com/Dao-AILab/flash-attention
- **CUDA Programming Guide:** https://docs.nvidia.com/cuda/cuda-c-programming-guide/

---

## Checkpoints

- [ ] Week 1: Baseline profiling complete, bottlenecks documented
- [ ] Week 2: First optimization running, speedup measured
- [ ] Week 3-4: Optimization refined and analyzed
- [ ] Week 5: Documentation and portfolio-ready code
- [ ] Week 6: Interview preparation (tradeoffs, lessons learned)

---

**Next:** See `../GPU_OPTIMIZATION_ROADMAP.md` for detailed optimization strategies.
