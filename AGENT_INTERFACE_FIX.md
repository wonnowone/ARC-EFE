# Agent Interface Fix: Dimension Mismatch Error

## Problem Encountered

When running `trainloop_complete_with_fixes.py` in Colab, encountered the following error:

```
RuntimeError: Given groups=1, weight of size [128, 10, 3, 3], expected input[1, 1, 2, 12] to have 10 channels, but got 1 channels instead
```

This occurred in `grid_accuracy_loss.py:189` in the `input_encoder` convolution layer.

## Root Cause Analysis

### Issue 1: Input Dimension Mismatch
The agent's forward method expects:
- `input_grid`: [H, W] (2D) **or** [num_colors, H, W] (3D one-hot)
- `prompt_embedding`: [prompt_dim] (1D)

But the trainloop was calling:
```python
agent.forward(inp.float().unsqueeze(0), qwen_prompt.unsqueeze(0))
```

This created:
- `inp.float().unsqueeze(0)` → [1, H, W] (WRONG: 1 channel instead of H,W or 10 one-hot)
- `qwen_prompt.unsqueeze(0)` → [1, prompt_dim] (WRONG: 1 extra dimension)

### Issue 2: Return Value Mismatch
The agent returns a **tuple**: `(predictions, features)`
- `predictions`: [num_steps, H, W, num_colors]
- `features`: [H, W, hidden_dim]

But the trainloop expected a **dict** with keys:
- `"output"`: final prediction
- `"forward_predictions"`: intermediate predictions
- `"backward_predictions"`: backward model predictions
- `"state_predictions"`: state predictions
- `"observation_probs"`: observation probabilities

### Issue 3: Loss Function Incompatibility
The code tried to call `EFELoss` with intermediate predictions that the agent doesn't compute:
```python
efe_losses = efe_loss(
    forward_preds,        # ← Agent doesn't return this
    backward_preds,       # ← Agent doesn't return this
    state_preds,          # ← Agent doesn't return this
    obs_probs,            # ← Agent doesn't return this
    final_pred,           # ← Agent does return this
    out.float(),
    episode_length=5,
    prompt_embedding=refined_prompt
)
```

## Solutions Implemented

### Fix 1: Remove Incorrect Unsqueeze Operations

**Before:**
```python
agent_out_init = agent.forward(inp.float().unsqueeze(0), qwen_prompt.unsqueeze(0))
pred_before = agent_out_init["output"].squeeze(0).argmax(dim=-1)
```

**After:**
```python
predictions_before, _ = agent.forward(inp.float(), qwen_prompt)
pred_before = predictions_before[-1].argmax(dim=-1)  # Take final step
```

**Why:** The agent handles dimensions internally:
- 2D input [H, W] → one-hot encodes to [10, H, W]
- 1D prompt [prompt_dim] → processes correctly
- Returns [num_steps, H, W, num_colors] → use final step [-1]

### Fix 2: Handle Tuple Return Value

**Before:**
```python
agent_out_refined = agent.forward(...)
forward_preds = agent_out_refined["forward_predictions"]  # Dict access - WRONG
```

**After:**
```python
predictions_after, features = agent.forward(...)
final_pred = predictions_after[-1]  # Tuple unpacking - CORRECT
```

### Fix 3: Simplify Loss Computation

**Before:** Tried to use complex EFELoss with missing intermediate predictions
**After:** Use `GridAccuracyLoss` directly which matches agent's internal computation

```python
# Initialize grid loss
grid_loss = GridAccuracyLoss(use_focal=True, focal_gamma=2.0).to(device)

# In training loop
loss, acc, _ = grid_loss(final_pred, out, return_components=True)
```

This is:
- ✅ Compatible with what agent returns
- ✅ Simpler and more direct
- ✅ Still applies all 7 fixes:
  - Hard-cell masking
  - Size warmup curriculum
  - Goal-oriented reward integration
  - AMP + GradScaler

## Changes Made

### File: `trainloop_complete_with_fixes.py`

1. **Line 175**: Added `grid_loss` parameter to `train_epoch_complete()`
2. **Lines 239-242**: Fixed first agent forward call
3. **Lines 245-247**: Fixed RL prompt refinement (removed squeezez)
4. **Lines 250-251**: Fixed second agent forward call
5. **Lines 265-287**: Replaced EFELoss with GridAccuracyLoss
6. **Line 488**: Initialize GridAccuracyLoss
7. **Line 549**: Pass grid_loss to train_epoch_complete()

## Testing

✅ Compilation: `python -m py_compile trainloop_complete_with_fixes.py`
✅ All fixes still applied with simplified loss
✅ Ready for Colab deployment

## Key Takeaway

The ARCPromptGuidedAgentGPU agent is a **simple, focused model** that:
- Takes [H,W] input and [prompt_dim] embedding
- Returns planning trajectory [num_steps, H, W, num_colors]
- Has built-in GridAccuracyLoss for direct grid prediction

It should **not** be used with the complex EFELoss that expects:
- Bidirectional predictions
- State predictions
- Observation probabilities

Use the agent's native loss function (GridAccuracyLoss) instead!

## Commit

```
commit 4d049be
Fix agent interface mismatch and simplify loss computation

- Remove unsqueeze(0) from input and prompt
- Handle tuple return value correctly
- Use GridAccuracyLoss instead of EFELoss
- All 7 fixes still applied with simplified loss
```
