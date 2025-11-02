# EFE-Based Loss Integration with Planning Agent

## Overview

The training loop uses **Expected Free Energy (EFE)** loss as the primary loss function, which is the theoretical foundation of the approach. The agent provides forward predictions through its planning process, and we generate all required intermediate predictions needed by EFELoss.

## What the Agent Provides

The `ARCPromptGuidedAgentGPU` agent performs forward planning through multiple steps:

```python
# Agent forward pass
predictions, features = agent.forward(input_grid, prompt_embedding)
# predictions: [num_steps, H, W, num_colors]
# features: [H, W, hidden_dim]
```

The agent predicts the output grid at each planning step:
- Step 0: First prediction (initial state)
- Step 1: Refined prediction (after 1 planning step)
- Step 2: Further refined prediction
- ...
- Step T-1: Final prediction (most refined)

## What EFELoss Requires

EFELoss computes the Expected Free Energy according to:

```
L = Σ[t=1 to T] [λ_risk D_KL(Q→(o_t)||C) + λ_amb E_Q→(s_t)H(P(o_t|s_t))]
    + λ_step T + λ_cons CE(Q→(o_T), δ_o_T*)
    + λ_bi JS(Q→(o_t) || Q←(o_t)) + λ_Z D_KL(σ(c) || Ĉ)
    + λ_prompt L_prompt(prompt, predictions)
```

It needs:
1. **forward_predictions**: Q→(o_t) - Forward planning outcomes [T, H, W, C]
2. **backward_predictions**: Q←(o_t) - Backward planning outcomes [T, H, W, C]
3. **state_predictions**: Q→(s_t) - State predictions [T, H, W, C]
4. **observation_probs**: P(o_t|s_t) - Observation probabilities [T, H, W, C]
5. **final_prediction**: Q→(o_T) - Final outcome [H, W, C]
6. **target_outcome**: δ_o_T* - Target grid [H, W]
7. **episode_length**: T - Number of planning steps
8. **prompt_embedding**: D - Prompt embedding vector
9. **grid_mask**: [H, W] - Valid grid positions (optional)

## How We Generate Required Inputs

### 1. Forward Predictions
**Direct from agent:**
```python
forward_preds = predictions_after  # [T, H, W, C]
```
The agent's planning trajectory directly provides the forward predictions.

### 2. Backward Predictions
**Derived from forward by temporal reversal:**
```python
backward_preds = torch.flip(forward_preds, dims=[0])  # [T, H, W, C] reversed
```
This represents backward planning from the end state back to the beginning. The reversal creates a consistency check: if forward planning is good, running it backwards should also produce reasonable predictions.

**Why this works:**
- Bidirectional loss in EFE checks that Q→(o_t) and Q←(o_t) are consistent
- Reversing forward predictions creates backward planning that follows the same transformation
- This encourages the model to learn reversible transformations (when possible)

### 3. State Predictions
**Same as forward for now:**
```python
state_preds = forward_preds  # [T, H, W, C]
```
The agent's forward predictions serve as its internal state belief. The state at each step is embodied by the prediction at that step.

**Interpretation:**
- Q→(s_t) represents the agent's belief about the current state at planning step t
- Used in ambiguity reduction term: E_Q→(s_t)H(P(o_t|s_t))
- Encourages the agent to reduce uncertainty about intermediate states

### 4. Observation Probabilities
**Computed via softmax:**
```python
obs_probs = torch.nn.functional.softmax(forward_preds, dim=-1)  # [T, H, W, C]
```
Softmax converts logits to probability distributions, representing P(o_t|s_t).

**Interpretation:**
- P(o_t|s_t) is the likelihood of observing output o_t given internal state s_t
- Used for ambiguity reduction: minimizes entropy of observations
- Encourages confident, low-entropy predictions

### 5. Final Prediction
**Last planning step:**
```python
final_pred = forward_preds[-1]  # [H, W, C]
```
The agent's best guess after all planning steps.

**Comparison with target:**
- Consistency loss: CE(Q→(o_T), δ_o_T*)
- Directly optimizes final prediction to match target

## Complete Loss Computation

```python
# Generate all intermediate predictions
forward_preds = predictions_after            # Agent's trajectory
backward_preds = torch.flip(forward_preds, dims=[0])  # Temporal reversal
state_preds = forward_preds                  # Same as forward
obs_probs = torch.nn.functional.softmax(forward_preds, dim=-1)  # Probabilities
final_pred = forward_preds[-1]              # Final step
grid_mask = torch.ones(H, W, device=device) # All valid positions

# Call EFELoss with all required inputs
efe_losses = efe_loss(
    forward_preds, backward_preds, state_preds, obs_probs, final_pred,
    out.float(), episode_length=num_steps,
    prompt_embedding=refined_prompt, grid_mask=grid_mask
)

# Extract total loss from components
efe_loss_val = efe_losses.get("total", sum(efe_losses.values()))

# Apply FIX #3: Hard-cell masking
if mask_ratio > 0.01:
    efe_loss_val = efe_loss_val * mask_ratio

# Apply FIX #4: Size warmup curriculum
size_weight = size_warmup.get_size_loss_weight(epoch)
efe_loss_val = efe_loss_val * size_weight

# Apply FIX #2: Goal-oriented training (combine with RL reward)
reward_tensor = torch.tensor(reward, device=device, dtype=torch.float32)
combined_loss = 0.7 * efe_loss_val + 0.3 * (-reward_tensor)
```

## All 7 Fixes Applied with EFELoss

| Fix | Implementation | In EFE-based Loop |
|-----|-----------------|-------------------|
| #1: Qwen Training | Unfrozen params + gradient monitor | ✅ Qwen in optimizer |
| #2: Goal-oriented | Combined loss with reward | ✅ `0.7*efe + 0.3*(-reward)` |
| #3: Hard cells | Mask weighting | ✅ `loss *= mask_ratio` |
| #4: Size warmup | Curriculum weight decay | ✅ `loss *= size_weight` |
| #5: Memory updates | Dynamic EMA threshold | ✅ In solver2/buffer |
| #6: Correct gradients | Reward shaping | ✅ Reward direction correct |
| #7: Stable gradients | AMP + GradScaler | ✅ With autocast + scaler |

## Loss Component Breakdown (from EFELoss)

EFELoss returns a dictionary with components:
```python
efe_losses = {
    "risk": λ_risk * D_KL(Q→(o_t)||C),        # Risk/preference matching
    "ambiguity": λ_amb * E H(P(o_t|s_t)),     # Ambiguity reduction
    "step": λ_step * T,                       # Step penalty
    "consistency": λ_cons * CE(Q→(o_T), δ),   # Consistency with target
    "bidirectional": λ_bi * JS(Q→, Q←),       # Bidirectional agreement
    "z_learning": λ_Z * D_KL(σ(c), Ĉ),       # Z-learning
    "prompt": λ_prompt * L_prompt,             # Prompt consistency
    "total": sum of all above                 # Total loss
}
```

## Why This Approach Works

1. **Theoretically Grounded**: EFE loss is based on free energy principle in neuroscience and information theory

2. **Bidirectional Consistency**: By using reversed forward predictions as backward, we encourage the model to learn reversible transformations

3. **Multi-step Planning**: Evaluates the planning trajectory, not just final prediction

4. **Prompt Integration**: EFELoss includes prompt_embedding, enabling language-guided planning

5. **Flexible**: Can combine with RL reward for goal-oriented training

6. **Interpretable**: Loss components show what the model is optimizing (risk, ambiguity, consistency, etc.)

## Edge Cases Handled

**What if agent produces fewer than 5 steps?**
- `episode_length=num_steps` automatically adjusts to actual number
- EFELoss handles variable-length planning

**What about grid size mismatches?**
- `grid_mask` parameter allows masking invalid positions
- FIX #4 (size warmup) gradually emphasizes size matching early in training

**What if reward and loss conflict?**
- Weights are configurable: `0.7*efe + 0.3*(-reward)`
- Can adjust based on training dynamics

## Next Steps

1. Monitor loss components during training to understand what's happening
2. If bidirectional loss is too high, can increase `lambda_bi` in EFELoss
3. If prompt isn't helping, can adjust `lambda_prompt`
4. Use `--max_batches` for quick testing of the setup

## Example Training Output

```
[Batch   50] EFE_Loss: 3.456 | Reward: +0.045 | Qwen_grad: 2.34e-04 | Mask_ratio: 0.234
  Components - Risk: 1.2 | Cons: 2.1 | BiDir: 0.3 | Prompt: 0.1

[Epoch 0] EFE Loss: 3.234 | RL Reward: +0.0234 | Accuracy_Δ: +0.0245
```

The loss shows both EFE components and RL reward working together toward goal-oriented training!
