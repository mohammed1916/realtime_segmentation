# GPU Profiling Commands - Copy/Paste Ready

**All commands ready to run directly in terminal/PowerShell**

---

## Quick Start: Nsight Compute (Exact L1/L2 Metrics)

### Command 1: Profile FP32 Baseline

```bash
ncu --metrics l1tex__throughput,l1tex__average_hit_rate,l2_throughput,l2_hit_rate,sm__throughput \
    -o profile_fp32.ncu \
    python gpu_optimization/measure_iteration.py --model baseline --runs 10
```

### Command 2: Profile FP16 Mixed Precision

```bash
ncu --metrics l1tex__throughput,l1tex__average_hit_rate,l2_throughput,l2_hit_rate,sm__throughput \
    -o profile_fp16.ncu \
    python gpu_optimization/measure_iteration.py --model fp16 --runs 10
```

### Command 3: Profile FP16 + TF32 (Production)

```bash
ncu --metrics l1tex__throughput,l1tex__average_hit_rate,l2_throughput,l2_hit_rate,sm__throughput \
    -o profile_fp16_tf32.ncu \
    python gpu_optimization/measure_iteration.py --model fp16_tf32 --runs 10
```

### View Results in GUI

```bash
ncu-ui profile_fp32.ncu
ncu-ui profile_fp16.ncu
ncu-ui profile_fp16_tf32.ncu
```

---

## PyTorch Profiler Commands

### Command 4: Profile with PyTorch (Quick, No GUI)

```bash
python -c "
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=4, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

model = SimpleModel().cuda().eval()
x = torch.randn(1, 3, 512, 512).cuda()

with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    with torch.no_grad():
        _ = model(x)

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=20))
"
```

### Command 5: FP16 Version

```bash
python -c "
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=4, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

model = SimpleModel().cuda().eval()
x = torch.randn(1, 3, 512, 512).cuda()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            _ = model(x)

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=20))
"
```

---

## Detailed Nsight Compute Commands with All L1/L2 Metrics

### Command 6: Complete L1/L2 Analysis (FP32)

```bash
ncu --set full \
    --metrics \
    l1tex__throughput,\
l1tex__average_hit_rate,\
l1tex__read_hit_rate,\
l1tex__write_hit_rate,\
l2_throughput,\
l2_hit_rate,\
l2__read_throughput,\
l2__write_throughput,\
sm__throughput,\
smsp__throughput \
    -o profile_fp32_full.ncu \
    python gpu_optimization/measure_iteration.py --model baseline --runs 5
```

### Command 7: Complete L1/L2 Analysis (FP16)

```bash
ncu --set full \
    --metrics \
    l1tex__throughput,\
l1tex__average_hit_rate,\
l1tex__read_hit_rate,\
l1tex__write_hit_rate,\
l2_throughput,\
l2_hit_rate,\
l2__read_throughput,\
l2__write_throughput,\
sm__throughput,\
smsp__throughput \
    -o profile_fp16_full.ncu \
    python gpu_optimization/measure_iteration.py --model fp16 --runs 5
```

### Command 8: Complete L1/L2 Analysis (FP16 + TF32)

```bash
ncu --set full \
    --metrics \
    l1tex__throughput,\
l1tex__average_hit_rate,\
l1tex__read_hit_rate,\
l1tex__write_hit_rate,\
l2_throughput,\
l2_hit_rate,\
l2__read_throughput,\
l2__write_throughput,\
sm__throughput,\
smsp__throughput \
    -o profile_fp16_tf32_full.ncu \
    python gpu_optimization/measure_iteration.py --model fp16_tf32 --runs 5
```

---

## Nsight Systems Commands (Timeline View)

### Command 9: Timeline Profiling (FP32)

```bash
nsys profile \
    --output=nsight_fp32 \
    --trace=cuda,osrt \
    python gpu_optimization/measure_iteration.py --model baseline --runs 5
```

### Command 10: Timeline Profiling (FP16)

```bash
nsys profile \
    --output=nsight_fp16 \
    --trace=cuda,osrt \
    python gpu_optimization/measure_iteration.py --model fp16 --runs 5
```

### Command 11: Timeline Profiling (FP16 + TF32)

```bash
nsys profile \
    --output=nsight_fp16_tf32 \
    --trace=cuda,osrt \
    python gpu_optimization/measure_iteration.py --model fp16_tf32 --runs 5
```

### View Timeline Results

```bash
nsys-ui nsight_fp32.nsys-rep
nsys-ui nsight_fp16.nsys-rep
nsys-ui nsight_fp16_tf32.nsys-rep
```

---

## Memory Hierarchy Profiler (Our Custom Script)

### Command 12: Run Memory Hierarchy Profiler

```bash
python gpu_optimization/memory_hierarchy_profiler.py
```

**Output:** `profiling/memory_hierarchy_log.json`
- Includes all commands used
- L1/L2 hit rates
- Latency comparisons
- Detailed profiler output

---

## Batch Profiling Script

### Command 13: Profile All Configurations at Once

```bash
bash -c '
echo "=== Profiling FP32 Baseline ==="
ncu --metrics l1tex__throughput,l1tex__average_hit_rate,l2_throughput,l2_hit_rate \
    -o profile_fp32.ncu \
    python gpu_optimization/measure_iteration.py --model baseline --runs 5

echo "=== Profiling FP16 ==="
ncu --metrics l1tex__throughput,l1tex__average_hit_rate,l2_throughput,l2_hit_rate \
    -o profile_fp16.ncu \
    python gpu_optimization/measure_iteration.py --model fp16 --runs 5

echo "=== Profiling FP16 + TF32 ==="
ncu --metrics l1tex__throughput,l1tex__average_hit_rate,l2_throughput,l2_hit_rate \
    -o profile_fp16_tf32.ncu \
    python gpu_optimization/measure_iteration.py --model fp16_tf32 --runs 5

echo "=== Complete ==="
'
```

---

## Export Results to CSV

### Command 14: Export Nsight Compute Results

```bash
# Export all metrics from .ncu files
ncu --export profile_fp32.ncu -o profile_fp32_metrics.csv
ncu --export profile_fp16.ncu -o profile_fp16_metrics.csv
ncu --export profile_fp16_tf32.ncu -o profile_fp16_tf32_metrics.csv
```

### Command 15: Compare Results (PowerShell)

```powershell
# View CSV comparisons
Write-Host "FP32 Results:"
(Import-Csv profile_fp32_metrics.csv) | Select-Object -First 10

Write-Host "`nFP16 Results:"
(Import-Csv profile_fp16_metrics.csv) | Select-Object -First 10

Write-Host "`nFP16+TF32 Results:"
(Import-Csv profile_fp16_tf32_metrics.csv) | Select-Object -First 10
```

---

## Key Metrics Explained

### Metrics Captured

| Metric | What It Measures | L1 Cache | L2 Cache |
|--------|-----------------|----------|----------|
| `l1tex__throughput` | L1 data moved per second | ✓ | — |
| `l1tex__average_hit_rate` | % of L1 requests satisfied by cache | ✓ | — |
| `l1tex__read_hit_rate` | % of L1 read requests from cache | ✓ | — |
| `l1tex__write_hit_rate` | % of L1 write requests from cache | ✓ | — |
| `l2_throughput` | L2 data moved per second | — | ✓ |
| `l2_hit_rate` | % of L2 requests satisfied by cache | — | ✓ |
| `l2__read_throughput` | L2 read bandwidth | — | ✓ |
| `l2__write_throughput` | L2 write bandwidth | — | ✓ |
| `sm__throughput` | SM instruction throughput | ✓ | ✓ |

---

## Expected Output Format

### From Nsight Compute CLI (Quick Metrics)

```
Metric Results:
  l1tex__throughput: 450.2 GB/s
  l1tex__average_hit_rate: 40.5%
  l2_throughput: 350.1 GB/s
  l2_hit_rate: 30.2%
  sm__throughput: 125.3 GFLOP/s
```

### From PyTorch Profiler

```
Name                          Self CPU %   Self CUDA %   CUDA Time
aten::cudnn_convolution         22.58%       70.04%     34.742ms
aten::batch_norm                 6.05%        1.13%      4.374ms
aten::relu_                      2.80%        0.57%      3.002ms
```

### From Memory Hierarchy Profiler (JSON)

```json
{
  "config": "FP32_Baseline",
  "latency_ms": 33.69,
  "cache_metrics": {
    "l1_hit_rate_estimated_pct": 40.0,
    "l2_hit_rate_estimated_pct": 30.0
  }
}
```

---

## Quick Reference: Which Command to Use?

| Goal | Command | Output |
|------|---------|--------|
| **Quick L1/L2 check** | Command 1-3 | Nsight GUI (.ncu file) |
| **All metrics** | Command 6-8 | Complete metrics (.ncu file) |
| **Timeline view** | Command 9-11 | Timeline visualization |
| **CSV comparison** | Command 14 | Spreadsheet-ready data |
| **All at once** | Command 13 | All profiles in one go |
| **Custom script** | Command 12 | JSON log with commands |

---

## Notes

1. **Nsight Compute** (ncu) requires NVIDIA tools installed
2. **Nsight Systems** (nsys) also requires NVIDIA tools
3. **PyTorch Profiler** (Command 5) works with standard PyTorch
4. All commands are copy-paste ready for PowerShell or Bash
5. Metrics are measured with GPU sync to ensure accuracy
6. Results logged with timestamps for reproducibility

---

*Ready to copy and paste directly into your terminal*
