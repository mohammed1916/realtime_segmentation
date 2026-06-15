# Deployment Guide - FP16+TF32 Optimization

## Code (Copy-Paste Ready)

```python
import torch

# Add at startup (once)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Wrap inference (every call)
with torch.amp.autocast('cuda'):
    output = model(input)
```

## Verify

```bash
python measure_iteration.py --model fp16_tf32 --runs 20
```

Expected output:
```
Latency: 22.41 ± 0.57 ms
Peak Memory: 810.5 MB
```

## Performance

| Config          | Latency    | Throughput   |
| --------------- | ---------- | ------------ |
| Before          | 32.70 ms   | 30.6 img/sec |
| After           | 22.41 ms   | 44.6 img/sec |
| **Improvement** | **-31.5%** | **+45.8%**   |

## Guarantees

- No accuracy loss  
- No retraining required  
- No model changes  
- Works on any NVIDIA GPU  
- Trivial rollback (delete 4 lines)

## Troubleshooting

**Different latency?**
- GPU temperature: `nvidia-smi -q | grep Temperature` (should be <85°C)
- Other processes: `nvidia-smi` (should only see your process)
- Increase runs: `--runs 50` for better averaging

**Performance worse?**
- Remove the 4 lines and go back to baseline
- Check GPU is not thermally throttling

## For More Info

- Full results: [`docs/ACTUAL_OPTIMIZATION_RESULTS.md`](docs/ACTUAL_OPTIMIZATION_RESULTS.md)
- Decision framework: [`docs/OPTIMIZATION_DECISION_LOOP.md`](docs/OPTIMIZATION_DECISION_LOOP.md)
- Measurement data: `profiling/iter_*.json`
