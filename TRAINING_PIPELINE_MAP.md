# ARC-EFE Training Pipeline - Complete Map

## 📋 ENTRY POINTS (Start Here)

### 1. **run_main.py** - Full Pipeline Orchestrator
**Purpose**: End-to-end execution (train + test + report)
**Key Classes**:
- `PipelineConfig` - Master configuration for entire pipeline

**How to Use**:
```python
python run_main.py
```

**Output**: `runs/arc_full_run_YYYYMMDD_HHMMSS/`

---

### 2. **run_gpu_finetuned.py** - GPU-Optimized Training
**Purpose**: GPU-accelerated training with Qwen fine-tuning
**Entry Function**: `trainloop_gpu_finetuned.main()`

**How to Use**:
```python
python run_gpu_finetuned.py \
    --epochs 10 \
    --agent_lr 1e-5 \
    --freeze_qwen False  # NEW: Qwen now trainable!
```

**Output**: `runs/arc_gpu_finetuned_YYYYMMDD_HHMMSS/`

---

### 3. **run_cpu_simple.py** - CPU-Only Training
**Purpose**: Lightweight CPU training for development/testing
**Entry Function**: `trainloop.train_one_epoch()`

**How to Use**:
```python
python run_cpu_simple.py --epochs 5 --no_gpu
```

**Output**: `runs/arc_cpu_YYYYMMDD_HHMMSS/`

---

## 🔄 TRAINING PIPELINE FLOW

```
┌─────────────────────────────────────────────────────────────┐
│              run_main.py / run_gpu_finetuned.py             │
│              (ENTRY POINT - Choose one)                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │   train_sequence.py         │
        │  (Complete training loop)   │
        └────────────┬────────────────┘
                     │
        ┌────────────┴───────────────────┐
        ▼                                 ▼
   ┌──────────────┐            ┌─────────────────┐
   │  trainloop.py│            │test_sequence.py │
   │   (per epoch)│            │  (evaluation)   │
   └──────┬───────┘            └─────────────────┘
          │
   ┌──────┴──────────────────────────────────┐
   ▼                                          ▼
┌──────────────────┐          ┌─────────────────────────────┐
│  Load Data       │          │  Initialize Models          │
│  - ARCDataset    │          │  - QwenHybridPrompt         │
│  - DataLoader    │          │  - ARCPromptGuidedAgent     │
└──────────────────┘          │  - EFELoss                  │
                              │  - PermanentSolver          │
                              │  - RevThinkOrchestrator     │
                              │  - TestTimeAdaptationSystem │
                              └─────────────────────────────┘
                                         │
                                         ▼
                              ┌────────────────────────┐
                              │  Training Loop         │
                              │  (per batch)           │
                              │ 1. Extract features   │
                              │ 2. Forward/Backward   │
                              │ 3. Compute loss       │
                              │ 4. Backward pass      │
                              │ 5. Update weights     │
                              │ 6. RevThink revision  │
                              │ 7. TTA evaluation     │
                              └────────────────────────┘
                                         │
                                         ▼
                              ┌────────────────────────┐
                              │  Logging & Checkpoints │
                              │ - train_log.jsonl     │
                              │ - metrics.jsonl       │
                              │ - model checkpoints   │
                              └────────────────────────┘
```

---

## 📦 ALL TRAINING FILES

### Core Training Files (EXECUTION)

| File | Purpose | Entry Point | Key Functions |
|------|---------|------------|-----------------|
| **run_main.py** | Full pipeline orchestrator | `main()` | Entry point, `PipelineConfig` |
| **run_gpu_finetuned.py** | GPU training wrapper | `main()` from trainloop_gpu_finetuned | GPU setup, device management |
| **run_cpu_simple.py** | CPU training wrapper | `main()` from trainloop | Minimal memory variant |
| **train_sequence.py** | Complete training pipeline | `train_sequence()` | `TrainingConfig`, `TrainingLogger`, epoch loop |
| **trainloop.py** | Single epoch training | `train_one_epoch()` | Batch-level training |
| **trainloop_gpu_finetuned.py** | Full GPU training loop | `main()` | Enhanced logging, checkpointing |
| **test_sequence.py** | Testing/evaluation | `test_sequence()`, `batch_test_checkpoints()` | Evaluation on test set |

---

### Model Architecture Files (MODELS)

| File | Purpose | Key Classes | Parameters |
|------|---------|------------|------------|
| **loss_function.py** | EFE Loss + Agent | `EFELoss`, `EFELossConfig`, `ARCPromptGuidedAgent` | 11 loss weights + EMA decay |
| **qwen_hybrid_prompt.py** | Qwen LLM integration | `QwenHybridPrompt`, `QwenCfg` | Model, dtype, max_tokens, temperature |
| **revthink_orchestrator.py** | Loss-based prompt revision | `RevThinkOrchestrator`, `RevThinkCfg` | tau, alpha, beta, gamma, eta |
| **tta.py** | Test-time adaptation | `TestTimeAdaptationSystem`, `SurpriseBasedMemory`, `MetaAdapter` | Memory size, adaptation steps, LR |
| **solver1.py** | Contextual memory solver | `ContextualMemoryBank`, `ContextualSolver` | Context size, surprise threshold, adaptation steps |
| **solver2.py** | Permanent memory solver | `PermanentMemoryBank`, `PermanentSolver` | Feature dim, max memories, DBSCAN epsilon |

---

### Data & Feature Files

| File | Purpose | Key Classes | Parameters |
|------|---------|------------|------------|
| **dataset_arc.py** | ARC dataset loader | `ARCDataset` | Path, split (train/test) |
| **feature_extraction.py** | Transform feature extraction | `extract_transformation_features()` | 15-dim feature vector |
| **feature_registry.py** | Feature operators registry | `FeatureRegistry`, `apply_operator_config()` | operators.yaml config |
| **configs/operators.yaml** | Feature operator configuration | YAML config file | Built-in operators |

---

### Utility/Analysis Files

| File | Purpose |
|------|---------|
| **grid_accuracy_loss.py** | Alternative loss (grid accuracy based) |
| **grid_transformation_loss.py** | Alternative loss (transformation focused) |
| **priority_efe_loss.py** | Alternative loss (priority-based) |
| **measure_throughput.py** | Performance benchmarking |

---

## ⚙️ CONFIGURATION LOCATIONS

### 1. **HARDCODED PARAMETERS** (in training scripts)

#### train_sequence.py (Lines 23-29)
```python
TTA_EVAL_INTERVAL = 50        # Evaluate TTA every 50 batches
CHECKPOINT_INTERVAL = 100     # Save checkpoint every 100 batches
GRAD_CLIP_NORM = 1.0
LEARNING_RATE = 1e-3
EPOCHS = 5
BATCH_SIZE = 1
```

#### trainloop_gpu_finetuned.py main() (Lines 505-509)
```python
def main(epochs=10,
         agent_lr=1e-5,
         qwen_lr=None,
         weight_decay=1e-6,
         grad_accum_steps=1,
         grad_clip=1.0,
         warmup_steps=100,
         max_batches_per_epoch=None,
         val_frequency=1,
         skip_test=False,
         device="cuda",
         model_name=None,
         seed=42,
         save_frequency=1,
         freeze_qwen=True):  # ← Control Qwen training!
```

---

### 2. **CONFIG CLASSES** (Dataclass-based)

#### EFELossConfig (loss_function.py:25-88)
```python
@dataclass
class EFELossConfig:
    # Loss weights (9 parameters)
    lambda_risk: float = 1.0
    lambda_amb: float = 0.0
    lambda_step: float = 0.1
    lambda_cons: float = 1.0          # Grid matching
    lambda_bi: float = 0.5            # Bidirectional
    lambda_z: float = 0.2             # Z-learning
    lambda_prompt: float = 0.3        # Prompt influence
    lambda_grid_norm: float = 0.1     # Grid fairness
    lambda_reversibility: float = 0.4 # Reversibility check

    # Grid parameters (3 parameters)
    max_grid_size: int = 30
    num_colors: int = 10
    prompt_dim: int = 256

    # Advanced parameters (2 parameters)
    inference_first_threshold: float = 0.7
    ema_decay: float = 0.99

    # Pre-configured profiles
    @staticmethod
    def aggressive_grid_matching() -> 'EFELossConfig'
    @staticmethod
    def reversibility_focus() -> 'EFELossConfig'
    @staticmethod
    def balanced() -> 'EFELossConfig'
```

#### QwenCfg (qwen_hybrid_prompt.py:224-240)
```python
class QwenCfg:
    model_name: str = "Qwen/Qwen2.5-1.8B"
    dtype: str = "float16"
    max_new_tokens: int = 96
    temperature: float = 0.0           # Deterministic
    top_p: float = 0.9
    cache_dir: str = ".cache/hf"
```

#### RevThinkCfg (revthink_orchestrator.py:7-13)
```python
@dataclass
class RevThinkCfg:
    tau: float = 0.45              # Revision trigger threshold
    alpha: float = 2.0             # Gate sharpness
    beta: float = 0.3              # Gate bias
    gamma: float = 0.5             # Lambda prompt boost
    eta: float = 0.2               # Z-anchoring blend
    mask_weight: float = 0.5
```

#### PipelineConfig (run_main.py:18-48)
```python
class PipelineConfig:
    # Output
    base_output = "runs"
    run_name = f"arc_full_run_{timestamp}"

    # Training (6 parameters)
    train_epochs = 5
    train_batch_size = 1
    train_lr = 1e-3
    train_grad_clip = 1.0
    train_tta_eval_interval = 50
    train_checkpoint_interval = 100

    # Testing (3 parameters)
    test_on_train_split = False
    test_on_test_split = True
    use_tta = True
    tta_steps = 5

    # Reporting
    generate_report = True
```

---

### 3. **ENVIRONMENT VARIABLES**

#### trainloop_gpu_finetuned.py (Line 546)
```python
data_dir = os.getenv("ARC_DATA_DIR", ".")
# Set: export ARC_DATA_DIR=/path/to/data
```

---

### 4. **YAML CONFIGURATION FILE** (External Config)

#### configs/operators.yaml
```yaml
# Feature extraction operator configuration
# Define custom operators for feature registry
```

---

## 📊 PARAMETER SUMMARY TABLE

### All Configurable Parameters (Total: 50+)

| Category | Parameter | File | Default | Purpose |
|----------|-----------|------|---------|---------|
| **LOSS WEIGHTS** | lambda_risk | loss_function.py | 1.0 | Risk/preference matching |
| | lambda_amb | loss_function.py | 0.0 | Ambiguity reduction |
| | lambda_step | loss_function.py | 0.1 | Step penalty |
| | lambda_cons | loss_function.py | 1.0 | Consistency (GRID MATCH) |
| | lambda_bi | loss_function.py | 0.5 | Bidirectional agreement |
| | lambda_z | loss_function.py | 0.2 | Z-learning anchor |
| | lambda_prompt | loss_function.py | 0.3 | Prompt influence |
| | lambda_grid_norm | loss_function.py | 0.1 | Grid size fairness |
| | lambda_reversibility | loss_function.py | 0.4 | Reversibility check |
| **TRAINING** | epochs | trainloop_gpu_finetuned.py | 10 | Total epochs |
| | agent_lr | trainloop_gpu_finetuned.py | 1e-5 | Agent learning rate |
| | qwen_lr | trainloop_gpu_finetuned.py | None | Qwen learning rate |
| | weight_decay | trainloop_gpu_finetuned.py | 1e-6 | L2 regularization |
| | grad_clip | trainloop_gpu_finetuned.py | 1.0 | Gradient clipping norm |
| | warmup_steps | trainloop_gpu_finetuned.py | 100 | LR warmup steps |
| | grad_accum_steps | trainloop_gpu_finetuned.py | 1 | Gradient accumulation |
| | freeze_qwen | trainloop_gpu_finetuned.py | True | Qwen training (NEW!) |
| **CHECKPOINTING** | save_frequency | trainloop_gpu_finetuned.py | 1 | Save every N epochs |
| | checkpoint_interval | train_sequence.py | 100 | Save every N batches |
| **EVALUATION** | val_frequency | trainloop_gpu_finetuned.py | 1 | Eval every N epochs |
| | max_batches_per_epoch | trainloop_gpu_finetuned.py | None | Limit batches/epoch |
| **QWEN** | model_name | qwen_hybrid_prompt.py | Qwen2.5-1.8B | Model ID |
| | dtype | qwen_hybrid_prompt.py | float16 | Model precision |
| | temperature | qwen_hybrid_prompt.py | 0.0 | LLM temperature |
| | top_p | qwen_hybrid_prompt.py | 0.9 | Nucleus sampling |
| | max_new_tokens | qwen_hybrid_prompt.py | 96 | Max output length |
| **REVTHINK** | tau | revthink_orchestrator.py | 0.45 | Revision threshold |
| | alpha | revthink_orchestrator.py | 2.0 | Gate sharpness |
| | beta | revthink_orchestrator.py | 0.3 | Gate bias |
| | gamma | revthink_orchestrator.py | 0.5 | Prompt boost |
| | eta | revthink_orchestrator.py | 0.2 | Z-anchor blend |
| **TTA** | memory_size | tta.py | 1000 | Memory bank size |
| | surprise_threshold | tta.py | 0.65 | TTA write threshold |
| | adaptation_steps | tta.py | 5 | TTA adaptation steps |
| | adaptation_lr | tta.py | 1e-3 | TTA learning rate |
| **SOLVER1** | context_size | solver1.py | 50 | Context memory size |
| | surprise_threshold | solver1.py | 0.2 | Store threshold |
| | temporal_decay | solver1.py | 0.9 | Memory decay rate |
| **SOLVER2** | max_memories | solver2.py | 10000 | Permanent memory limit |
| | dbscan_eps | solver2.py | 0.3 | Clustering threshold |
| **GRID** | max_grid_size | loss_function.py | 30 | Max size for padding |
| | num_colors | loss_function.py | 10 | Color palette size |
| **ADVANCED** | inference_first_threshold | loss_function.py | 0.7 | Confidence threshold |
| | ema_decay | loss_function.py | 0.99 | EMA smoothing |

---

## 🎯 HOW TO CHANGE CONFIGURATION

### Method 1: Command Line Arguments (Recommended)
```bash
python run_gpu_finetuned.py \
    --epochs 20 \
    --agent_lr 5e-5 \
    --freeze_qwen False \
    --weight_decay 1e-5
```

### Method 2: Edit trainloop_gpu_finetuned.py
Lines 505-509: Edit default values in `main()` function
```python
def main(epochs=20,              # Change from 10 to 20
         agent_lr=5e-5,          # Change from 1e-5 to 5e-5
         freeze_qwen=False, ...): # NEW: Unfreeze Qwen!
```

### Method 3: Use EFELossConfig Profiles
```python
from loss_function import EFELossConfig, EFELoss

# For aggressive grid matching
cfg = EFELossConfig.aggressive_grid_matching()
efe_loss = EFELoss(**cfg.to_dict())

# For reversibility focus
cfg = EFELossConfig.reversibility_focus()
efe_loss = EFELoss(**cfg.to_dict())
```

### Method 4: Modify Config Classes Directly
Edit the dataclass default values in the respective files:
- Loss weights: `loss_function.py:25-88`
- Qwen params: `qwen_hybrid_prompt.py:224-240`
- RevThink params: `revthink_orchestrator.py:7-13`

---

## 📁 OUTPUT DIRECTORY STRUCTURE

```
runs/arc_gpu_finetuned_YYYYMMDD_HHMMSS/
├── train_log.jsonl              # Training metrics (per batch)
├── eval_log.jsonl               # Evaluation metrics (per epoch)
├── tta_log.jsonl                # TTA evaluation metrics
├── metrics.jsonl                # Summary metrics
├── checkpoints.jsonl            # Checkpoint metadata
├── config.json                  # Training configuration (saved)
├── checkpoints/
│   ├── model_epoch_0.pt         # Agent state dict
│   ├── model_epoch_1.pt
│   └── ...
├── qwen_checkpoints/
│   ├── qwen_epoch_0.pt          # Qwen state dict
│   └── ...
├── metrics_plot.png             # Loss curves visualization
└── training_log.txt             # Human-readable log
```

---

## 💡 QUICK START GUIDE

### To Train with NEW Settings:

```bash
cd C:\Users\SAMSUNG\OneDrive\바탕 화면\ARC-EFE\ARC-EFE

# 1. Use aggressive grid matching
python run_gpu_finetuned.py \
    --epochs 20 \
    --agent_lr 1e-4 \
    --freeze_qwen False \
    --weight_decay 1e-5
```

### To Use Custom Loss Config:

```python
from loss_function import EFELossConfig, EFELoss

# Create custom config
cfg = EFELossConfig()
cfg.lambda_cons = 2.0      # Double grid matching weight
cfg.lambda_bi = 0.3        # Reduce bidirectional
cfg.freeze_qwen = False    # NEW: Train Qwen

efe_loss = EFELoss(**cfg.to_dict())
```

### To Monitor Training:

```bash
# Watch metrics in real-time
tail -f runs/arc_gpu_finetuned_LATEST/train_log.jsonl | jq

# Plot results
python -m tensorboard --logdir=runs/
```

---

## ✅ ALL CONFIGURATION SOURCES SUMMARIZED

| Source | Type | Number | Location | Modifiable |
|--------|------|--------|----------|------------|
| Hardcoded in scripts | Constants | 6 | train_sequence.py | Edit file |
| Function defaults | Function args | 14 | trainloop_gpu_finetuned.py:main() | CLI args |
| Dataclasses | Config objects | 11 | Multiple files | Edit or instantiate |
| Environment vars | OS vars | 1 | trainloop_gpu_finetuned.py:546 | Set env var |
| YAML file | External config | Custom | configs/operators.yaml | Edit file |
| **TOTAL** | | **32+** | | |

---

**All files, configurations, and parameters are now mapped and accessible!** 🚀
