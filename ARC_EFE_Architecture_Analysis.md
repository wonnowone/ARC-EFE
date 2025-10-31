# ARC-EFE CODEBASE ANALYSIS - QUICK REFERENCE

## 1. SOLVER1 vs SOLVER2

### Solver1: ContextualSolver
- **Type**: Fast episodic memory
- **Memory Size**: 50-item circular buffer
- **Update Rule**: Surprise-gated (gradient magnitude × novelty)
- **Retrieval**: Top-K attention with temporal decay
- **Adaptation**: 3-step inner loop with rapid learning
- **Use**: Real-time reasoning, learns within session

### Solver2: PermanentSolver  
- **Type**: Persistent clustered memory
- **Memory Size**: 10,000 problem-solution pairs
- **Clustering**: DBSCAN (eps=0.3, cosine metric)
- **Retrieval**: Cosine similarity + cluster boost + objective filter
- **Features**: 9 movement features + problem features
- **Use**: Long-term learning across many problems

---

## 2. REVTHINK ORCHESTRATOR

**Purpose**: Adaptive critique and revision gate

**revthink_score()** combines 8 components:
- risk (0.2 weight)
- consistency (0.2)
- bidirectional (0.2)
- ambiguity (0.1)
- critique_consistency (0.15)
- tta_consistency (0.1)
- solver_likelihood (-0.15)

**maybe_revise()** logic:
- If score > tau (0.45): Call Qwen to revise
- Else: Return {apply: False}

---

## 3. FEATURES: 3-LEVEL HIERARCHY

**Level 1**: operators.yaml
- vertical_sym_score, axis_color_runlen, hole_fill_ratio

**Level 2**: transformation_features
- size_change_ratio, colors_added/removed, density_change, spatial_correlation

**Level 3**: movement_features (9, DETERMINISTIC ORDER)
- size_change, color_diversity_in, color_diversity_out, shape_preserved
- pixel_change_ratio, color_preservation, spatial_correlation
- input_symmetry, output_symmetry

**CRITICAL**: Features use L2 normalization + cosine metric for DBSCAN

---

## 4. EFE LOSS FUNCTION (7 Components)

```
L = λ_risk·D_KL(Q→||C) + λ_amb·E[H(P)]
  + λ_step·T + λ_cons·CE(Q_T, target)
  + λ_bi·JS(Q→||Q←) + λ_Z·D_KL(σ(c)||Ĉ)
  + λ_prompt·L_prompt
```

| Component | λ | Purpose |
|-----------|---|---------|
| Risk | 1.0 | Preference matching |
| Ambiguity | 0.0 | (disabled) |
| Step | 0.1 | Encourage short plans |
| Consistency | 1.0 | Supervised learning |
| Bi-directional | 0.5 | Forward-backward agreement |
| Z-anchoring | 0.2 | Prevent preference drift |
| Prompt | 0.3 | Language alignment |

---

## 5. WHAT IS "SO"?

NOT "Stable Optimization"

**State-Outcome Planning Pair** (Active Inference):
- S = State (internal representation)
- O = Outcome (predicted grid)
- Q→(s_t) = Forward state predictions
- Q→(o_t) = Forward outcome predictions
- P(o_t|s_t) = Likelihood of observation given state

---

## 6. FEATURE → LOSS FLOW

```
operators.yaml → feature_registry → transformation_features
→ compose_prompt_from_features → qwen → hybrid_embedding
→ ARCPromptGuidedAgent → predictions → EFELoss → backprop
```

**Feature Usage**:
1. Prompt → Risk Loss (initialize color preferences)
2. Movement Features → Solver2 Retrieval (cosine similarity)
3. Objective Type → Weighting (1.0x same, 0.7x different)
4. Operators → RevThink Decisions (revision triggers)

**IMPORTANT**: Features guide (initialization, retrieval, decisions)
- NO λ weights for features in final loss
- Final loss is purely EFE-based

---

## 7. TRAINING PIPELINE

**File**: trainloop_gpu_finetuned.py

Steps:
1. Extract features (operators.yaml + computed)
2. Generate hybrid embedding (Qwen LLM)
3. Forward pass (ARCPromptGuidedAgent)
4. Compute loss (GridAccuracyLoss or EFELoss)
5. Backward pass + gradient clipping
6. Optimizer step

**Two Loss Options**:
- **GridAccuracyLoss**: Simple cross-entropy (direct supervision)
- **EFELoss**: 7 components (full framework)

---

## 8. QUICK SUMMARY

| Aspect | Details |
|--------|---------|
| **Solver1** | 50-item episodic, fast adaptation |
| **Solver2** | 10K persistent, DBSCAN clustered |
| **RevThink** | 8-component revision gate |
| **Features** | 3 levels (operators + transform + movement) |
| **Loss** | 7-component EFE framework |
| **Qwen** | Fine-tuned LLM for semantic guidance |
| **Training** | Both agent and Qwen weights updated |
| **Capability** | Few-shot + persistent learning + critique |

---

## 9. KEY RECOMMENDATIONS

**For Training**:
- Start with GridAccuracyLoss
- Always include prompt_embedding
- Use Solver2 for > 100 problems
- Monitor revthink_score (target < 0.45)

**For Features**:
- Keep operators.yaml enabled
- Maintain deterministic ordering (CRITICAL)
- L2 normalization for cosine metric
- No λ weights (guidance only)

**For Loss**:
- λ_risk = 1.0 (primary signal)
- λ_cons = 1.0 (supervision)
- λ_bi = 0.5 (symmetry)
- λ_prompt = 0.3 (language)

---

## 10. CAPABILITIES

✓ Few-shot adaptation (Solver1 context)
✓ Persistent learning (Solver2 memory)
✓ Self-critique (RevThink orchestration)
✓ Semantic understanding (Qwen hybrid)
✓ Multi-objective planning (EFE framework)
✓ Variable grid sizes (lazy initialization)
✓ GPU fine-tuning (mixed precision)

