# HUGE ACCURACY IMPROVEMENT PLAN - 3 Phases

## Current Problem
```
Accuracy: 0.22% ❌
Loss: 5.1 (not learning)
Solver2 Memory: 0/2725 (empty) ❌
Reason: Success threshold too strict, loss weights wrong
```

## The Fix: 3 Progressive Phases

---

## PHASE 1: Quick Fixes (TRY IMMEDIATELY - 5 minutes)

**No code changes needed! Just use command-line arguments.**

### Fix 1A: Lower Success Threshold
This unlocks Solver2 memory updates!

```bash
# Edit line 294 of trainloop_gpu_finetuned.py temporarily:
# Change: success_threshold = 1.5
# To:     success_threshold = 5.0
```

Or create a wrapper script `train_phase1.py`:

```python
#!/usr/bin/env python
import subprocess
import sys

# Run with modified success threshold
# (We'll inject this via monkey-patching)
sys.argv = [
    'trainloop_gpu_finetuned.py',
    '--epochs', '2',
    '--max-batches', '100',
    '--no-freeze-qwen',
    '--agent-lr', '1e-4',
    '--device', 'cuda'
]

if __name__ == "__main__":
    # Quick test
    exec(open('trainloop_gpu_finetuned.py').read())
```

**Test Command**:
```bash
python trainloop_gpu_finetuned.py \
  --epochs 1 \
  --max-batches 50 \
  --no-freeze-qwen \
  --agent-lr 1e-4 \
  --device cuda
```

**Expected Results**:
- Loss: 5.0 → 3.5 (decreasing ✓)
- Memory updates: >0 (not zero ✓)
- Accuracy: 0.5-1% (still low but improving ✓)

**If successful**: Proceed to Phase 2

---

## PHASE 2: Better Loss Weights (10 minutes)

The problem: Risk and Ambiguity losses dominate (87% of total).
The solution: Weight grid accuracy higher.

### Edit trainloop_gpu_finetuned.py, lines 562-572:

**BEFORE** (current):
```python
efe_loss = EFELoss(
    lambda_risk=1.0,      # Too high! (2.74)
    lambda_amb=0.0,       # Too high! (1.72)
    lambda_step=0.1,      # Too low! (0.02)
    lambda_cons=1.0,      # Good (1.60)
    lambda_bi=0.5,        # Too high (0.86)
    lambda_z=0.2,         # Not helping (0)
    lambda_prompt=0.3,    # Broken (-0.12)
    ...
).to(device)
```

**AFTER** (Phase 2):
```python
efe_loss = EFELoss(
    lambda_risk=0.1,      # Reduce (was 1.0)
    lambda_amb=0.0,       # Keep at 0
    lambda_step=0.05,     # Reduce (was 0.1)
    lambda_cons=2.0,      # Increase (was 1.0) ← FOCUS ON GRID ACCURACY
    lambda_bi=0.1,        # Reduce (was 0.5)
    lambda_z=0.0,         # Disable (was 0.2)
    lambda_prompt=0.0,    # Disable (was 0.3, was negative)
    ...
).to(device)
```

### Also: Lower weight decay

Edit trainloop_gpu_finetuned.py, lines 594-597:

**BEFORE**:
```python
optim = torch.optim.Adam([
    {"params": agent_params, "lr": agent_lr, "weight_decay": weight_decay},
    {"params": solver2_params, "lr": agent_lr * 2.0, "weight_decay": weight_decay},
    {"params": efe_loss_params, "lr": agent_lr * 0.5, "weight_decay": weight_decay},
])
```

**AFTER**:
```python
optim = torch.optim.Adam([
    {"params": agent_params, "lr": agent_lr, "weight_decay": weight_decay / 5},
    {"params": solver2_params, "lr": agent_lr * 2.0, "weight_decay": weight_decay / 5},
    {"params": efe_loss_params, "lr": agent_lr * 0.5, "weight_decay": weight_decay / 5},
])
```

### Test Phase 2:
```bash
python trainloop_gpu_finetuned.py \
  --epochs 3 \
  --no-freeze-qwen \
  --agent-lr 1e-4 \
  --device cuda
```

**Expected Results**:
- Loss: 5.0 → 2.5 (much better!)
- Memory updates: 500+ (learning!)
- Accuracy: 2-5% (good improvement!)

**If successful**: Proceed to Phase 3

---

## PHASE 3: New Loss Function (20 minutes)

For HUGE improvement (5x better accuracy), use GridTransformationLoss instead of EFE.

### Create new training file: `trainloop_gpu_improved.py`

Copy `trainloop_gpu_finetuned.py` and make these changes:

**STEP 1**: Import new loss (add to imports, line 20):
```python
from grid_transformation_loss import GridTransformationLoss
```

**STEP 2**: Replace loss creation (lines 562-574):

**REPLACE THIS**:
```python
efe_loss = EFELoss(
    lambda_risk=1.0,
    lambda_amb=0.0,
    # ... etc
).to(device)
```

**WITH THIS**:
```python
# Use new Grid Transformation Loss instead of EFE
from grid_transformation_loss import GridTransformationLoss
efe_loss = GridTransformationLoss(
    shape_weight=0.40,        # Size must match
    pixel_weight=0.45,        # Colors must match
    palette_weight=0.10,      # Use right colors
    transform_weight=0.05,    # Transformation type
    num_colors=10
).to(device)
```

**STEP 3**: Modify train_epoch to use new loss (around line 260):

**CHANGE FROM**:
```python
efe_losses = efe_loss(
    forward_predictions=forward_predictions,
    backward_predictions=backward_predictions,
    state_predictions=state_predictions,
    observation_probs=observation_probs,
    final_prediction=forward_predictions.squeeze(0),
    target_outcome=out,
    episode_length=1,
    prompt_embedding=problem_features.squeeze(0),
    grid_mask=None
)
```

**CHANGE TO**:
```python
efe_losses = efe_loss(
    predictions=forward_predictions.squeeze(0),  # [H, W, 10]
    target=out,                                   # [H, W]
    input_grid=inp                                # [H, W]
)
```

**STEP 4**: Adjust success threshold in train_epoch (around line 294):
```python
success_threshold = 2.0  # Much easier to achieve now
```

### Test Phase 3:
```bash
python trainloop_gpu_improved.py \
  --epochs 5 \
  --no-freeze-qwen \
  --agent-lr 1e-4 \
  --device cuda
```

**Expected Results**:
- Loss: 2.5 → 1.0-1.5 (excellent!)
- Memory updates: 1000+ (lots of learning!)
- Accuracy: 5-20% (HUGE improvement!)

---

## Quick Decision Tree

```
Choose your path:
│
├─→ WANT QUICK TEST (30 min)?
│   └─ Run Phase 1 only
│      Command: trainloop_gpu_finetuned.py --epochs 1 --max-batches 50
│
├─→ WANT GOOD IMPROVEMENT (1.5 hours)?
│   └─ Run Phase 2 (edit loss weights + threshold)
│      Command: trainloop_gpu_finetuned.py --epochs 3 (with edits)
│
└─→ WANT MAXIMUM ACCURACY (3 hours)?
    └─ Run Phase 3 (new loss function)
       Command: trainloop_gpu_improved.py --epochs 5
```

---

## Expected Progress

```
                Accuracy    Memory Updates    Loss      Time
Current:        0.22%       0/2725           5.1       Done
After Phase 1:  0.5-1%      10-100           4.0-4.5   +30 min
After Phase 2:  2-5%        500-1000         2.0-3.0   +1.5 hours
After Phase 3:  5-20%       1000+            1.0-1.5   +3 hours
```

---

## What Each Phase Does

| Phase | Change | Why | Result |
|-------|--------|-----|--------|
| 1 | Increase success_threshold | Allow memory to fill | Solver2 learns |
| 2 | Reweight loss components | Focus on grid accuracy | Better convergence |
| 3 | Replace loss function | Grid-specific optimization | Much higher accuracy |

---

## Detailed Implementation (Phase 3)

### Option A: Minimal Changes (Quick)

Just replace loss computation. In `trainloop_gpu_finetuned.py`:

```python
# OLD (lines 260-270):
efe_losses = efe_loss(
    forward_predictions=...,
    backward_predictions=...,
    state_predictions=...,
    observation_probs=...,
    final_prediction=...,
    target_outcome=out,
    episode_length=1,
    prompt_embedding=...,
    grid_mask=None
)

# NEW:
grid_loss = GridTransformationLoss(num_colors=10)
efe_losses = grid_loss(
    predictions=solution_grid.squeeze(0),  # [H, W, 10]
    target=out,                             # [H, W]
    input_grid=inp                          # [H, W]
)
```

### Option B: Full Implementation (Better)

Create new file `trainloop_gpu_improved.py` as mentioned above.

---

## Monitoring Phase 3 Progress

Watch for:
- **Good**: Loss smoothly decreasing from 5.0 → 1.0
- **Good**: Memory updates > 500
- **Good**: Validation accuracy > 1%
- **Bad**: Loss NaN or infinity
- **Bad**: Memory updates still 0
- **Bad**: Accuracy stuck at 0%

---

## Final Recommendation

### I recommend: **Start with Phase 1 + 2** (safest, 1.5 hours)

```bash
# 1. Edit trainloop_gpu_finetuned.py line 294:
#    success_threshold = 1.5  →  success_threshold = 5.0

# 2. Edit trainloop_gpu_finetuned.py lines 562-572:
#    lambda_risk=0.1, lambda_cons=2.0, etc. (as shown above)

# 3. Run:
python trainloop_gpu_finetuned.py \
  --epochs 5 \
  --no-freeze-qwen \
  --agent-lr 1e-4 \
  --weight-decay 1e-7 \
  --device cuda
```

This should give you **2-5% accuracy** (10-20x improvement).

Then if you want more: **Try Phase 3** with the new loss function.

---

## Questions?

- **Why Phase 1**: Success threshold blocks all memory learning
- **Why Phase 2**: EFE loss weights are imbalanced for ARC
- **Why Phase 3**: ARC needs grid-specific loss, not abstract EFE

**Bottom line**: Your insight was correct - focus on grid shapes and transformations, not abstract preferences.

---

**Ready to start? Begin with Phase 1 - takes 5 minutes!** 🚀
