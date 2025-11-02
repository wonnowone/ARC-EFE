# Quick Start: Goal-Oriented Training with RL Agent

## TL;DR

**Old way:** Minimize a loss function (may not solve the problem)
```bash
python trainloop_gpu_finetuned.py --epochs 10
```

**New way:** Explicitly maximize accuracy + size + color + reversibility
```bash
python trainloop_with_rl_agent.py --epochs 10
```

---

## Step 1: Run the Training

```bash
cd "C:\Users\SAMSUNG\OneDrive\바탕 화면\ARC-EFE\ARC-EFE"

# Basic: 10 epochs on full dataset
python trainloop_with_rl_agent.py --epochs 10 --device cuda

# Testing: 5 epochs with 100 batches per epoch
python trainloop_with_rl_agent.py --epochs 5 --device cuda --max_batches 100

# Custom LR: If default doesn't work
python trainloop_with_rl_agent.py --epochs 20 --agent_lr 1e-4 --device cuda
```

---

## Step 2: Watch the Output

### What You'll See

```
======================================================================
GOAL-ORIENTED TRAINING WITH HUMAN RL AGENT
======================================================================

Device: cuda
Epochs: 10
Agent LR: 1e-5
Output: runs/arc_rl_agent_20250101_120000

Loading datasets...
  Train: 3232 | Val: 1920

Creating models...
  [OK] RL Agent initialized - GOAL: Improve accuracy via prompt refinement

Starting goal-oriented training...

[Batch   50] Reward: +0.0456 | Acc_Δ: +0.0234 | Size_Δ: +0.0045 | RLoss: 1.2345 | EFELoss: 7.6543
[Batch  100] Reward: +0.0389 | Acc_Δ: +0.0198 | Size_Δ: +0.0023 | RLoss: 1.1234 | EFELoss: 7.4321
[Batch  150] Reward: +0.0512 | Acc_Δ: +0.0267 | Size_Δ: +0.0089 | RLoss: 1.0123 | EFELoss: 7.2109
```

### Good Signs ✓
- **Acc_Δ:** Positive and not 0.0000
- **Reward:** Averaging positive values
- **Size_Δ:** Increasing if output sizes were mismatched

### Red Flags ❌
- **Acc_Δ:** Always 0.0000 (RL not helping)
- **Reward:** Always negative (RL making things worse)
- **RLoss:** Growing rapidly (RL diverging)

---

## Step 3: Check the Results

After training finishes, you'll have:

```
runs/arc_rl_agent_YYYYMMDD_HHMMSS/
├─ training.log              ← Read this for full details
├─ metrics_goal_oriented.json ← Raw metrics data
└─ agent_best.pt             ← Best model checkpoint
```

### Read the Log

```bash
# Linux/Mac
tail -100 runs/arc_rl_agent_*/training.log

# Windows PowerShell
Get-Content (Get-ChildItem runs/arc_rl_agent_*).FullName -Tail 100 -Path "*training.log"
```

Look for the EPOCH SUMMARY sections:

```
======================================================================
EPOCH 0 SUMMARY (Goal-Oriented Training)
======================================================================
  Average Combined Loss:       3.5678
  Average RL Reward Signal:    +0.0234
  Average RL Loss:             1.2345
  Average EFE Loss:            2.3456

EXPLICIT GOAL PROGRESS (What Actually Matters):
  Accuracy Delta (↑ is good):       +0.0456  ← Primary metric!
  Size Match Delta (↑ is good):     +0.0123
  Color Agreement Delta (↑):        +0.0089
  Reversibility Delta (↑):          +0.0012
======================================================================

[Epoch 0] Val Accuracy: 0.0367 (71/1920)
[Epoch 0] RL Accuracy Delta: +0.0245
[Epoch 0] Time: 456.32s
```

### Good Results
- Accuracy Delta: **+0.04 to +0.10** per epoch (4-10% improvement)
- RL Reward: **+0.02 to +0.08** (positive means improvements found)
- Size Delta: **+0.01 to +0.05** if applicable
- Val Accuracy: Should increase over epochs

### Concerning Results
- Accuracy Delta: **0.0000** (RL not helping)
- RL Reward: **negative** (RL making things worse)
- RLoss: **exploding** (1.0 → 10.0 → 100.0)

---

## Step 4: Analyze Metrics

Extract and plot the metrics:

```python
import json
import matplotlib.pyplot as plt

# Load metrics
with open("runs/arc_rl_agent_YYYYMMDD_HHMMSS/metrics_goal_oriented.json") as f:
    metrics = json.load(f)

# Plot accuracy delta (primary goal)
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.plot(metrics["accuracy_delta"])
plt.title("Accuracy Delta (Should increase)")
plt.ylabel("Improvement %")
plt.xlabel("Batch")

# Plot size delta
plt.subplot(2, 2, 2)
plt.plot(metrics["size_delta"])
plt.title("Size Match Delta")
plt.ylabel("Improvement")
plt.xlabel("Batch")

# Plot RL reward
plt.subplot(2, 2, 3)
plt.plot(metrics["rl_reward"])
plt.title("RL Reward Signal (Should be positive)")
plt.ylabel("Reward")
plt.xlabel("Batch")

# Plot combined loss
plt.subplot(2, 2, 4)
plt.plot(metrics["total_loss"])
plt.title("Combined Loss (Should decrease)")
plt.ylabel("Loss")
plt.xlabel("Batch")

plt.tight_layout()
plt.savefig("metrics_analysis.png", dpi=100)
plt.show()
```

---

## Common Issues & Fixes

### Issue 1: Accuracy Delta is 0.0000
**Symptom:**
```
[Batch 100] Reward: +0.0001 | Acc_Δ: +0.0000 | Size_Δ: +0.0000
```

**Cause:** RL agent isn't finding improvements

**Fix:**
```bash
# Try higher RL learning rate
python trainloop_with_rl_agent.py --epochs 10 --agent_lr 1e-4

# Or more entropy (more exploration)
# Edit policy_refined.py line ~306:
# PolicyRefinedConfig(rl_entropy_coef=0.05)  # Increase from 0.01
```

### Issue 2: Training is Very Slow
**Symptom:**
```
Each batch takes 10+ seconds
```

**Cause:** Two forward passes (with and without RL) + RL computation

**Fix:**
```bash
# Limit batches for faster iteration
python trainloop_with_rl_agent.py --epochs 3 --max_batches 100

# Or reduce after first epoch works
# Most learning happens early, so 5 epochs might be enough
```

### Issue 3: CUDA Out of Memory
**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Cause:** RL agent + Qwen + Agent too large

**Fix:**
```bash
# Use CPU (slower but memory safe)
python trainloop_with_rl_agent.py --device cpu --epochs 5

# Or reduce batch logging (edit line 278)
# Change: if batch_idx % 50 == 0:
# To:     if batch_idx % 100 == 0:
```

### Issue 4: Loss Goes Up But Accuracy Delta Positive
**Symptom:**
```
Combined Loss: 8.5 → 9.2 (increasing)
Accuracy Delta: +0.045 (positive)
```

**Cause:** This is actually NORMAL! Loss is changing to accommodate goal optimization

**What to do:**
Monitor accuracy delta, not loss. This is the whole point!

---

## Comparing Standard vs Goal-Oriented

After running both, compare:

```bash
# Run standard training
python trainloop_gpu_finetuned.py --epochs 10 > standard_output.txt 2>&1 &

# Run goal-oriented training (in another terminal)
python trainloop_with_rl_agent.py --epochs 10 > rl_output.txt 2>&1 &

# Wait for both to finish, then compare
grep "Val Accuracy:" standard_output.txt
grep "Accuracy Delta:" rl_output.txt
```

Expected comparison:
```
Standard:       Epochs 0-9: 2% → 3% (1% total improvement, unclear progress)
Goal-Oriented:  Epochs 0-9: each epoch +4-6% improvements, clear progress
```

---

## Customization

### Change Primary Goal Weight

If accuracy is not the main goal, adjust weights in `policy_refined.py`:

```python
# Line ~306
@dataclass
class PolicyRefinedConfig:
    reward_acc_weight: float = 1.0      # Main goal (change this)
    reward_size_weight: float = 0.5     # Secondary
    reward_color_weight: float = 0.5    # Secondary
    reward_rev_weight: float = 0.5      # Secondary
```

### Adjust Goal/Loss Balance

In `trainloop_with_rl_agent.py` line ~280:

```python
# Default: 70% EFE loss, 30% goal reward
combined_loss = 0.7 * efe_loss + 0.3 * (-reward)

# To focus more on goals:
combined_loss = 0.5 * efe_loss + 0.5 * (-reward)

# To keep more EFE structure:
combined_loss = 0.9 * efe_loss + 0.1 * (-reward)
```

### Adjust RL Learning Rate

```bash
# Default is 1e-5, but RL learns at 5e-5 (from policy_refined.py)
# If RL isn't helping, try:
python trainloop_with_rl_agent.py --agent_lr 1e-4 --epochs 10

# Edit rl_lr in policy_refined.py if needed:
# Line ~305: rl_lr: float = 5e-5  # Increase to 1e-4
```

---

## What Each Metric Means

| Metric | Good Value | What It Means |
|--------|-----------|---|
| **Accuracy Delta** | +0.04 to +0.10 | Percentage of cells that became correct (main goal) |
| **Size Delta** | +0.01 to +0.05 | Output dimensions moved closer to target |
| **Color Delta** | +0.01 to +0.05 | Color distribution became more similar to target |
| **Reversibility Delta** | +0.001 to +0.01 | Backward solver can reconstruct input better |
| **RL Reward** | +0.02 to +0.10 | Combined improvement signal from all 4 metrics |
| **Combined Loss** | Decreasing | But secondary to metrics above! |

---

## Full Command Reference

```bash
# Minimal (quick test)
python trainloop_with_rl_agent.py --epochs 3 --max_batches 50

# Standard (default)
python trainloop_with_rl_agent.py --epochs 10 --device cuda

# Extended (full training)
python trainloop_with_rl_agent.py --epochs 20 --agent_lr 1e-4 --device cuda

# Debugging (slow, verbose)
python trainloop_with_rl_agent.py --epochs 5 --max_batches 100 --device cpu

# Custom seed
python trainloop_with_rl_agent.py --epochs 10 --seed 123 --device cuda
```

---

## Next Steps

1. **Run it:** `python trainloop_with_rl_agent.py --epochs 10`
2. **Monitor:** Watch for positive accuracy deltas in the logs
3. **Analyze:** Compare accuracy delta progress over epochs
4. **Iterate:** Adjust learning rates or goal weights if needed
5. **Deploy:** Use `agent_best.pt` from output directory

---

## Support

If something's not working:

1. Check `training.log` for error messages
2. Look for "RED FLAG" patterns above
3. Try reducing `max_batches` for faster iteration
4. Check `GOAL_ORIENTED_TRAINING.md` for detailed explanations
5. Compare `COMPARISON_STANDARD_VS_RL.md` to understand differences

---

## Key Insight

> Training is no longer about "making loss go down."
> It's about "solving the problem: maximize accuracy, size match, color match, reversibility."
>
> Loss decreases as a consequence of solving the problem, not as the goal itself.

Let's see what real problem-solving looks like! 🚀
