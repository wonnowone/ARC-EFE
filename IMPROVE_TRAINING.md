# Training Performance Analysis & Improvements

## Current Results (DIAGNOSIS)

```
Training Accuracy: 0.22% (2/913 grids correct)
Loss: 5.1014 (very high - not converging)
Memory bank updates: 0/2725 (0% - not learning!)
```

**Problem**: Model is **not learning at all**. It's basically random guessing.

## Root Cause Analysis

### 🔴 Critical Issues (Why it's not working)

1. **Qwen is FROZEN** ❌
   - Frozen at line 437-440 of trainloop_gpu_finetuned.py
   - Cannot adapt to ARC task specifics
   - Extracts generic features, not task-specific ones

2. **Solver2 Memory Empty** ❌
   - Memory updates: 0/2725 (success_threshold=1.5 too strict)
   - No solutions are good enough to store
   - Memory bank provides no learned experience

3. **Loss Components Imbalanced** ❌
   - Risk: 2.74 (should be ~0.1-0.5)
   - Ambiguity: 1.72 (should be ~0.0-0.5)
   - Prompt Consistency: -0.124 (negative?! should be positive)

4. **Learning Rate Too Low** ❌
   - Default: 1e-5 (very conservative)
   - May need 1e-4 to 1e-3 for meaningful learning

5. **Loss Function Weights** ❌
   - Current: risk=1.0, cons=1.0, bi=0.5, step=0.1
   - May be fighting each other

## Quick Fixes (Try These First)

### Fix #1: Enable Qwen Fine-tuning (Most Important!)

**Change this:**
```bash
python trainloop_gpu_finetuned.py --epochs 5 --device cuda
```

**To this:**
```bash
python trainloop_gpu_finetuned.py --epochs 5 --no-freeze-qwen --device cuda
```

This unfreezes Qwen, allowing it to learn task-specific features.

**Expected improvement**: 0.22% → 2-5%

---

### Fix #2: Increase Learning Rate

```bash
python trainloop_gpu_finetuned.py \
  --epochs 5 \
  --agent-lr 1e-4 \
  --no-freeze-qwen \
  --device cuda
```

Increases learning rate 10x (1e-5 → 1e-4).

**Expected improvement**: Another 2-3% accuracy boost

---

### Fix #3: Reduce Success Threshold (Let Solver2 Learn)

Edit `trainloop_gpu_finetuned.py`, line 294:

**Change from:**
```python
success_threshold = 1.5  # EFE loss threshold for "success"
```

**Change to:**
```python
success_threshold = 3.0  # More lenient - allow learning
```

This allows Solver2 to store solutions even if not perfect, building memory.

**Expected improvement**: Memory bank populates (0 → 100s of solutions)

---

## Progressive Improvement Strategy

### Phase 1: Debug & Validate (30 min)
```bash
python trainloop_gpu_finetuned.py \
  --epochs 1 \
  --max-batches 50 \
  --no-freeze-qwen \
  --agent-lr 1e-4 \
  --device cuda
```

Check:
- Does loss decrease? (should go from 5.0 → 4.0)
- Does memory bank update? (should have >0 updates)
- Does validation improve?

### Phase 2: Full Training (2-3 hours)
```bash
python trainloop_gpu_finetuned.py \
  --epochs 5 \
  --no-freeze-qwen \
  --agent-lr 1e-4 \
  --weight-decay 1e-5 \
  --device cuda
```

### Phase 3: Hyperparameter Tuning (if needed)
Try different combinations:
```bash
# Conservative (stable, slower)
--agent-lr 5e-5 --weight-decay 1e-6

# Aggressive (faster learning, riskier)
--agent-lr 5e-4 --weight-decay 1e-4

# Medium (balanced)
--agent-lr 1e-4 --weight-decay 1e-5
```

## Detailed Fixes (Edit File)

### Edit 1: Enable Qwen Fine-tuning

**File**: `trainloop_gpu_finetuned.py`

**Line 437-440, Change from:**
```python
if freeze_qwen:
    for param in qwen.parameters():
        param.requires_grad = False
    logger.log("  Qwen loaded (FROZEN - stable training)")
```

**Change to:**
```python
if freeze_qwen:
    for param in qwen.parameters():
        param.requires_grad = False
    logger.log("  Qwen loaded (FROZEN - stable training)")
else:
    for param in qwen.parameters():
        param.requires_grad = True
    logger.log("  Qwen loaded (TRAINABLE - better learning)")
```

Actually, the code already supports `--no-freeze-qwen`. Just use the command line argument!

### Edit 2: Adjust Success Threshold

**File**: `trainloop_gpu_finetuned.py`

**Line 294, Change from:**
```python
success_threshold = 1.5  # EFE loss threshold for "success"
```

**Change to:**
```python
success_threshold = 3.5  # More lenient for memory learning
```

### Edit 3: Reduce Weight Decay

**File**: `trainloop_gpu_finetuned.py`

**Line 495-498, Change from:**
```python
{"params": agent_params, "lr": agent_lr, "weight_decay": weight_decay},
{"params": solver2_params, "lr": agent_lr * 2.0, "weight_decay": weight_decay},
{"params": efe_loss_params, "lr": agent_lr * 0.5, "weight_decay": weight_decay},
```

**Change to:**
```python
{"params": agent_params, "lr": agent_lr, "weight_decay": weight_decay / 10},
{"params": solver2_params, "lr": agent_lr * 2.0, "weight_decay": weight_decay / 10},
{"params": efe_loss_params, "lr": agent_lr * 0.5, "weight_decay": weight_decay / 10},
```

## Recommended Command (Best Chance of Success)

```bash
python trainloop_gpu_finetuned.py \
  --epochs 10 \
  --no-freeze-qwen \
  --agent-lr 2e-4 \
  --weight-decay 1e-6 \
  --device cuda
```

**Expected results**:
- Epoch 1: Loss ~5.0 → 3.5
- Epoch 5: Loss ~2.0-3.0
- Epoch 10: Loss ~1.5-2.0
- Val Accuracy: 1-5%

**Training time**: ~4-5 hours for 10 epochs

## What Each Parameter Does

| Parameter | Current | Try | Effect |
|-----------|---------|-----|--------|
| `--no-freeze-qwen` | Not used | Add | Qwen learns task-specific features (CRITICAL) |
| `--agent-lr` | 1e-5 | 1e-4 to 5e-4 | Higher = faster learning but riskier |
| `--weight-decay` | 1e-6 | 5e-7 to 1e-5 | Lower = less regularization, more overfitting |
| `--epochs` | 5 | 10 | More epochs = more learning |

## Monitoring During Training

Watch for these signs of improvement:

✅ **Good signs**:
- Loss decreasing each epoch (5.0 → 4.5 → 4.0)
- Memory bank updating (>0 updates)
- Val accuracy > 0.1% (>1 grid perfect)
- Validation accuracy trending upward

❌ **Bad signs**:
- Loss stuck/increasing
- Memory bank still 0 updates
- Val accuracy stays 0%
- NaN or inf in loss

## If Still Not Working

### Step 1: Verify Data Loading
```bash
python -c "
from dataset_arc import ARCDataset
ds = ARCDataset('training.json', split='train')
print(f'Loaded {len(ds)} samples')
batch = ds[0]
print(f'Input shape: {batch[\"input\"].shape}')
print(f'Output shape: {batch[\"output\"].shape}')
"
```

### Step 2: Check Loss Function
```bash
python -c "
import torch
from loss_function import EFELoss
loss_fn = EFELoss()
# Should not error
print('Loss function OK')
"
```

### Step 3: Simple Training Test
```bash
python trainloop_gpu_finetuned.py \
  --epochs 1 \
  --max-batches 10 \
  --no-freeze-qwen \
  --agent-lr 1e-3 \
  --device cuda
```

Very high LR + few batches should show fast loss decrease.

## Expected Timeline

| Step | Time | Accuracy |
|------|------|----------|
| Fix 1 (unfreeze Qwen) | 2-3 hours | 1-3% |
| Fix 2 (increase LR) | 2-3 hours | 2-5% |
| Fix 3 (lower threshold) | Variable | 3-8% |
| Fine-tuning | 4-6 hours | 5-15% |

## Realistic Expectations

For ARC task with current architecture:
- **Epoch 1**: 0-2% accuracy (model warming up)
- **Epoch 5**: 2-5% accuracy (starting to learn)
- **Epoch 10**: 5-10% accuracy (decent learning)
- **After fine-tuning**: 10-20% (with more epochs/data)

Note: ARC is **extremely hard** (humans get ~85%, SOTA ~30%), so 10% is reasonable.

## Next Steps

1. **Run this immediately**:
```bash
python trainloop_gpu_finetuned.py \
  --epochs 2 \
  --max-batches 100 \
  --no-freeze-qwen \
  --agent-lr 1e-4 \
  --device cuda
```

2. **Check results**: Look at loss and accuracy trends

3. **If improving**: Scale to full training (5-10 epochs)

4. **If not**: Try higher LR (1e-3) or check for data issues

---

**TL;DR**: Your main issue is **Qwen is frozen**. Run:
```bash
python trainloop_gpu_finetuned.py --epochs 5 --no-freeze-qwen --agent-lr 1e-4 --device cuda
```

This should improve accuracy from 0.22% to 2-5% or higher. 🚀
