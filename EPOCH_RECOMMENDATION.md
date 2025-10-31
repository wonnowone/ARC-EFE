# How Many Epochs Do You Need? (T4 GPU Guide)

## Step 1: Measure Your Actual Throughput

The benchmark tool will tell you exactly how long each epoch takes on YOUR hardware:

```bash
python measure_throughput.py --device cuda
```

This will:
1. Run 500 batches to measure per-batch time
2. Calculate your actual epoch duration
3. Recommend epoch counts based on time available

**Output will look like:**
```
GPU: Tesla T4
GPU Memory: 15.1 GB

Per-Batch Time Statistics:
  Average:     521.34 ms
  Median:      515.20 ms
  Min:         480.15 ms
  Max:         612.45 ms
  Std Dev:     35.21 ms

Projected Single Epoch Time (3232 samples):
  1686424 seconds
  28.1 minutes
  0.47 hours

RECOMMENDED EPOCH COUNTS
======================================================================
If you have [TIME BUDGET] available:
30 minutes       -> 1 epochs (~0.47 hours)
1 hour           -> 2 epochs (~0.94 hours)
2 hours          -> 4 epochs (~1.88 hours)
4 hours          -> 8 epochs (~3.76 hours)
...
```

## Step 2: Based on T4 Performance

### If Your T4 is Fast (200-300ms/batch):
- **1 epoch**: ~10-15 minutes
- **Recommended: 10 epochs** (~100-150 minutes = 1.5-2.5 hours)
- Good for: Quick validation, iterative development

### If Your T4 is Moderate (400-600ms/batch):
- **1 epoch**: ~20-30 minutes
- **Recommended: 5-8 epochs** (~100-240 minutes = 1.5-4 hours)
- Good for: Standard training with decent convergence

### If Your T4 is Slower (700ms-1s/batch):
- **1 epoch**: ~35-50 minutes
- **Recommended: 2-4 epochs** (~70-200 minutes = 1-3 hours)
- Good for: Limited time, focus on quality

### If Your T4 is Very Slow (>1s/batch):
- **1 epoch**: >50 minutes
- **Recommended: 1-2 epochs** (50-100 minutes)
- Consider: Gradient accumulation, reducing batch validation

## Step 3: Convergence vs Time Tradeoff

### ARC Task Characteristics:
- Highly problem-specific patterns
- Each epoch learns from diverse 3232 examples
- Early epochs show biggest improvements
- Diminishing returns after 5-8 epochs

### Convergence Curve Expectations:

```
Accuracy
  |     ╱╲
  |    ╱  ╲___
  |   ╱       ╲___
  |  ╱            ╲____
  |_╱___________________╲____
    1    3    5    8    10+  epochs

Key insights:
- Epochs 1-3: Steep improvement (biggest gains)
- Epochs 4-8: Moderate improvement
- Epochs 9+: Minimal improvement
```

## Step 4: Recommendations by Time Budget

### If you have **<1 hour**:
```bash
python trainloop_gpu_finetuned.py --epochs 2 --device cuda
# Fast validation of pipeline
# ~20-30 min per epoch = 40-60 min total
```

### If you have **1-2 hours**:
```bash
python trainloop_gpu_finetuned.py --epochs 3 --device cuda
# OR test different hyperparameters
# ~60-90 min total
```

### If you have **2-4 hours**:
```bash
python trainloop_gpu_finetuned.py --epochs 5 --device cuda
# Good baseline training
# Gets into reasonable accuracy
# ~2-3 hours total
```

### If you have **4+ hours**:
```bash
python trainloop_gpu_finetuned.py --epochs 10 --device cuda
# Full training with good convergence
# ~3-5 hours total (depending on T4 speed)
```

## Step 5: Actual Time Formula

Once you know your per-batch time from the benchmark:

```
Epoch Time (minutes) = (per_batch_time_ms / 1000) * 3232 / 60
Total Time = Epoch Time * Number of Epochs
```

**Example calculations:**

| Per-Batch | 1 Epoch | 5 Epochs | 10 Epochs |
|-----------|---------|----------|-----------|
| 200ms     | 10.7 min| 53 min   | 107 min   |
| 300ms     | 16 min  | 80 min   | 160 min   |
| 400ms     | 21 min  | 107 min  | 214 min   |
| 500ms     | 27 min  | 134 min  | 268 min   |
| 600ms     | 32 min  | 161 min  | 322 min   |
| 750ms     | 40 min  | 201 min  | 402 min   |
| 1000ms    | 54 min  | 268 min  | 537 min   |

## Step 6: Quick Decision Tree

```
Is your T4 running faster than expected?
├─ YES (200-400ms/batch)
│  └─ Run: --epochs 10 (1.5-2.5 hours)
│
├─ MODERATE (400-600ms/batch)
│  └─ Run: --epochs 5-8 (1.5-4 hours)
│
├─ SLOWER (600-800ms/batch)
│  └─ Run: --epochs 3-4 (1-2 hours)
│
└─ SLOW (>800ms/batch)
   └─ Run: --epochs 2 (check for issues!)
```

## Step 7: What to Do If Training is Slow

If you measure 1s+ per batch, something might be wrong:

### Possible issues:
1. **Qwen model loading slowly**
   - Solution: Use smaller model: `--model-name Qwen/Qwen1.5-0.5B`

2. **Memory pressure causing slowdown**
   - Check: `nvidia-smi` during training
   - If GPU memory > 12GB, reduce batch size or model size

3. **Feature extraction bottleneck**
   - Solution: Cache features (advanced optimization)

4. **Dataset loading slow**
   - Solution: Ensure training.json is on fast storage (SSD)

5. **CPU bottleneck**
   - Check CPU usage in `nvidia-smi`
   - May need more worker processes

## Step 8: Recommended Command

After running the benchmark, use the suggested command:

```bash
# Run benchmark FIRST
python measure_throughput.py --device cuda

# Then run training with recommended epochs
python trainloop_gpu_finetuned.py --epochs [SUGGESTED] --device cuda
```

## Step 9: Monitor During Training

Watch the progress bar to see if times are consistent:

```
Epoch 0 Training: 25%|██▌       | 807/3232 [06:45<20:15] loss: 0.4521
Epoch 0 Training: 50%|█████     | 1616/3232 [13:30<13:30] loss: 0.4213
Epoch 0 Training: 75%|███████▌  | 2424/3232 [20:15<06:45] loss: 0.4012
```

If times are consistent, your estimate is good. If times get slower, something may be degrading (memory, cache, etc).

## Step 10: Final Recommendation

**For T4 GPU with ARC training:**

| Scenario | Command | Duration |
|----------|---------|----------|
| Quick test | `--epochs 1 --max-batches 100` | 5-10 min |
| Validation | `--epochs 2` | 40-60 min |
| Standard | `--epochs 5` | 100-150 min |
| Full | `--epochs 10` | 180-300 min |

**My suggestion**: Start with `--epochs 5` on T4. That gives:
- Good convergence (most improvement happens in first 5 epochs)
- Reasonable time (2-4 hours)
- Safe baseline to compare against

Then you can extend to 10 epochs if you have time and want to squeeze more accuracy.

## Commands to Run

```bash
# Step 1: Measure your actual speed
python measure_throughput.py --device cuda

# Step 2: Run training with recommended epochs (replace N with number)
python trainloop_gpu_finetuned.py --epochs 5 --device cuda

# Step 3 (optional): If happy with results, run full training
python trainloop_gpu_finetuned.py --epochs 10 --device cuda
```

The benchmark tool will give you the exact answer for your hardware!
