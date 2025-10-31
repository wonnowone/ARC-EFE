import os
import sys
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader
from typing import Dict, Optional

from dataset_arc import ARCDataset
from qwen_hybrid_prompt import QwenHybridPrompt, QwenCfg
from revthink_orchestrator import RevThinkOrchestrator, RevThinkCfg
from feature_registry import FeatureRegistry, apply_operator_config
from feature_extraction import extract_transformation_features, classify_transformation_type
from grid_accuracy_loss import GridAccuracyLoss, ARCPromptGuidedAgentGPU
from solver2 import PermanentSolver
from loss_function import EFELoss
import torch.nn.functional as F


def seed_all(seed=42):
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pack_transform_record(inp, out):
    """Pack input/output into transformation record"""
    # Squeeze batch dimension if present
    if inp.dim() == 3:
        inp = inp.squeeze(0)
    if out is not None and out.dim() == 3:
        out = out.squeeze(0)

    in_grid = inp.cpu().tolist()
    out_grid = out.cpu().tolist() if out is not None else None
    if out_grid is None:
        out_grid = [[0 for _ in row] for row in in_grid]

    feats = extract_transformation_features(in_grid, out_grid)
    feats["transformation_type"] = classify_transformation_type(feats)
    return feats


class TrainingLogger:
    """Log training progress with detailed metrics"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.log_file = os.path.join(output_dir, "training.log")
        os.makedirs(output_dir, exist_ok=True)

    def log(self, message: str):
        """Log message to file and stdout"""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')

    def log_epoch(self, epoch: int, loss: float, val_acc: float, val_perfect: int, total_samples: int):
        """Log epoch summary"""
        message = (f"[Epoch {epoch}] Loss: {loss:.6f} | "
                  f"Val Accuracy: {val_acc:.4f} ({val_perfect}/{total_samples} perfect)")
        self.log(message)


def train_epoch(agent, qwen, solver2, efe_loss, train_loader, optimizer, device, feat_reg, epoch, logger, max_batches=None):
    """
    Train one epoch with Solver2 + EFE Loss (full Active Inference framework).

    Data flow:
      Raw Grid [H,W] → Qwen → problem_features [256]
                           → Solver2 (memory retrieval + solution generation)
                           → EFE Loss (7 components)
                           → Backward pass
    """
    agent.train()
    qwen.eval()  # Qwen is frozen, set to eval mode
    solver2.train()
    efe_loss.train()

    epoch_loss = 0.0
    loss_components_sum = {
        'risk': 0.0, 'ambiguity': 0.0, 'efe': 0.0,
        'step_penalty': 0.0, 'consistency': 0.0,
        'bidirectional': 0.0, 'z_anchoring': 0.0, 'prompt_consistency': 0.0
    }
    batches_processed = 0
    total_batches = len(train_loader)
    memory_updates = 0

    for batch_idx, batch in enumerate(train_loader):
        if max_batches and batch_idx >= max_batches:
            break

        # Progress reporting every 50 batches
        if (batch_idx + 1) % 50 == 0 or batch_idx == total_batches - 1:
            logger.log(f"  Processing batch {batch_idx + 1}/{total_batches}...")

        # ========== LOAD DATA ==========
        # inp: [1, H, W] or [H, W]
        # out: [1, H, W] or [H, W]
        inp = batch["input"].to(device)
        out = batch["output"].to(device) if batch["output"] is not None else None

        if out is None:
            continue

        # Squeeze batch dimension to [H, W]
        inp = inp.squeeze(0) if inp.dim() == 3 else inp  # [H, W]
        out = out.squeeze(0) if out.dim() == 3 else out  # [H, W]

        # Handle size mismatch (pad output to match input dimensions)
        if inp.shape != out.shape:
            if inp.shape[0] > out.shape[0] and inp.shape[1] > out.shape[1]:
                pad_h = inp.shape[0] - out.shape[0]
                pad_w = inp.shape[1] - out.shape[1]
                out = F.pad(out, (0, pad_w, 0, pad_h), mode='constant', value=0)  # [H, W]
            elif inp.shape[0] < out.shape[0] or inp.shape[1] < out.shape[1]:
                continue  # Skip if output is larger (shouldn't happen)
            else:
                continue

        H, W = inp.shape  # Get spatial dimensions

        # ========== FEATURE EXTRACTION ==========
        # Extract transformation features (operators + computed features)
        tr = pack_transform_record(inp.unsqueeze(0), out.unsqueeze(0))  # Dict
        tr = apply_operator_config(tr, inp.unsqueeze(0), out.unsqueeze(0), feat_reg)  # Dict

        # ========== GET PROBLEM FEATURES FROM QWEN ==========
        # Qwen produces: hybrid_embedding [256] (combines text + control vectors)
        with torch.no_grad():  # Qwen is frozen
            pack = qwen(tr, inp.unsqueeze(0), out.unsqueeze(0), control_weight=0.5)
            problem_features = pack["hybrid_embedding"]  # [1, 256] or [256]
            prompt_emb_text = pack.get("prompt_text", "")

        # Ensure problem_features is [1, 256] for Solver2
        if problem_features.dim() == 1:
            problem_features = problem_features.unsqueeze(0)  # [1, 256]

        # ========== SOLVER2: MEMORY RETRIEVAL + SOLUTION GENERATION ==========
        # Solver2 retrieves similar problems from memory and generates solution
        solver2_output = solver2(
            problem_features=problem_features,      # [1, 256]
            input_grid=inp.unsqueeze(0),           # [1, H, W]
            target_shape=(H, W)
        )

        # Extract predictions from Solver2
        solution_grid = solver2_output['solution_grid']  # [1, H, W, 10] (logits)
        memory_guidance = solver2_output['memory_guidance']  # [1, 512] (retrieved features)
        confidence = solver2_output['confidence']  # [1] (solution confidence)

        # Reshape for EFE Loss: [1, H, W, 10] → [1, H, W, 10] (add time dimension)
        # EFE expects [T, H, W, C] where T=1 for single planning step
        forward_predictions = solution_grid  # [1, H, W, 10]
        backward_predictions = solution_grid.clone()  # [1, H, W, 10] (simplified: use forward as backward)
        state_predictions = solution_grid.clone()  # [1, H, W, 10]
        observation_probs = F.softmax(solution_grid, dim=-1)  # [1, H, W, 10]

        # ========== EFE LOSS COMPUTATION (7 COMPONENTS) ==========
        # Compute full Expected Free Energy loss
        efe_losses = efe_loss(
            forward_predictions=forward_predictions,  # [T=1, H, W, 10]
            backward_predictions=backward_predictions,  # [T=1, H, W, 10]
            state_predictions=state_predictions,  # [T=1, H, W, 10]
            observation_probs=observation_probs,  # [T=1, H, W, 10]
            final_prediction=forward_predictions.squeeze(0),  # [H, W, 10] (remove time dim)
            target_outcome=out,  # [H, W] (target grid)
            episode_length=1,  # Single planning step
            prompt_embedding=problem_features.squeeze(0),  # [256]
            grid_mask=None
        )

        # Total loss from all 7 components
        total_loss = efe_losses['total']  # Scalar

        # ========== BACKWARD PASS ==========
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(solver2.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(efe_loss.parameters(), max_norm=1.0)

        optimizer.step()

        # ========== MEMORY UPDATE (IF SUCCESSFUL) ==========
        # Update Solver2 memory with successful solutions
        success_threshold = 1.5  # EFE loss threshold for "success"
        if total_loss.item() < success_threshold:
            try:
                solver2.update_memory(
                    problem_features=problem_features.squeeze(0),  # [256]
                    solution_features=memory_guidance.squeeze(0),  # [512]
                    input_grid=inp,  # [H, W]
                    output_grid=out,  # [H, W]
                    success=True,
                    metadata={
                        'epoch': epoch,
                        'batch': batch_idx,
                        'loss': total_loss.item()
                    }
                )
                memory_updates += 1
            except Exception as e:
                pass  # Silently fail if memory update fails

        # ========== ACCUMULATE LOSSES ==========
        epoch_loss += total_loss.item()
        for key in loss_components_sum:
            if key in efe_losses:
                loss_components_sum[key] += efe_losses[key].item() if torch.is_tensor(efe_losses[key]) else efe_losses[key]

        batches_processed += 1

    # ========== EPOCH SUMMARY ==========
    avg_loss = epoch_loss / max(batches_processed, 1)
    avg_components = {k: v / max(batches_processed, 1) for k, v in loss_components_sum.items()}

    # Log component breakdown
    logger.log(f"    Loss components (avg):")
    logger.log(f"      Risk: {avg_components['risk']:.6f} | Ambiguity: {avg_components['ambiguity']:.6f}")
    logger.log(f"      Step Penalty: {avg_components['step_penalty']:.6f} | Consistency: {avg_components['consistency']:.6f}")
    logger.log(f"      Bidirectional: {avg_components['bidirectional']:.6f} | Z-anchoring: {avg_components['z_anchoring']:.6f}")
    logger.log(f"      Prompt Consistency: {avg_components['prompt_consistency']:.6f}")
    logger.log(f"    Memory bank updates: {memory_updates}/{batches_processed}")
    logger.log(f"    Solver2 memory size: {len(solver2.memory_bank.memories)}")

    return avg_loss


def evaluate(agent, qwen, solver2, efe_loss, eval_loader, device, feat_reg, max_batches=None, binary_accuracy=True):
    """
    Evaluate on validation/test set using Solver2 + EFE Loss.

    Args:
        binary_accuracy: If True, accuracy is 1.0 only if entire grid perfect, else 0.0

    Returns:
        (avg_accuracy, perfect_count, total_samples)
    """
    agent.eval()
    qwen.eval()
    solver2.eval()
    efe_loss.eval()

    acc_sum = 0.0
    perfect_count = 0
    total_samples = 0
    eval_loss = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            if max_batches and batch_idx >= max_batches:
                break

            # ========== LOAD DATA ==========
            # inp: [1, H, W] or [H, W]
            # out: [1, H, W] or [H, W]
            inp = batch["input"].to(device)
            out = batch["output"].to(device) if batch["output"] is not None else None

            if out is None:
                continue

            # Squeeze batch dimension to [H, W]
            inp = inp.squeeze(0) if inp.dim() == 3 else inp  # [H, W]
            out = out.squeeze(0) if out.dim() == 3 else out  # [H, W]

            out_h_true, out_w_true = out.shape  # Save original output size

            # Handle size mismatch (pad output to match input)
            if inp.shape != out.shape:
                if inp.shape[0] > out.shape[0] and inp.shape[1] > out.shape[1]:
                    pad_h = inp.shape[0] - out.shape[0]
                    pad_w = inp.shape[1] - out.shape[1]
                    out = F.pad(out, (0, pad_w, 0, pad_h), mode='constant', value=0)  # [H, W]
                else:
                    continue  # Skip mismatched samples

            H, W = inp.shape  # Padded dimensions

            # ========== FEATURE EXTRACTION ==========
            tr = pack_transform_record(inp.unsqueeze(0), out.unsqueeze(0))  # Dict
            tr = apply_operator_config(tr, inp.unsqueeze(0), out.unsqueeze(0), feat_reg)  # Dict

            # ========== GET PROBLEM FEATURES FROM QWEN ==========
            pack = qwen(tr, inp.unsqueeze(0), out.unsqueeze(0), control_weight=0.5)
            problem_features = pack["hybrid_embedding"]  # [1, 256] or [256]

            # Ensure [1, 256] for Solver2
            if problem_features.dim() == 1:
                problem_features = problem_features.unsqueeze(0)  # [1, 256]

            # ========== SOLVER2: INFERENCE ==========
            # Use Solver2 for solution generation (with memory retrieval)
            solver2_output = solver2(
                problem_features=problem_features,  # [1, 256]
                input_grid=inp.unsqueeze(0),       # [1, H, W]
                target_shape=(H, W)
            )

            # Extract predicted grid
            solution_grid = solver2_output['solution_grid']  # [1, H, W, 10] (logits)

            # Convert logits to class predictions: [1, H, W, 10] → [H, W]
            final_pred = solution_grid.squeeze(0).argmax(dim=-1)  # [H, W]

            # Crop to true output size (remove padding)
            if final_pred.shape != (out_h_true, out_w_true):
                final_pred = final_pred[:out_h_true, :out_w_true]  # [H_true, W_true]

            # ========== CALCULATE ACCURACY ==========
            # Compare predictions with true output
            if binary_accuracy:
                # Binary: 1.0 if entire grid perfect, else 0.0
                is_perfect = (final_pred == out[:out_h_true, :out_w_true]).all().item()
                acc = 1.0 if is_perfect else 0.0
                if is_perfect:
                    perfect_count += 1
            else:
                # Per-cell: partial credit
                acc = (final_pred == out[:out_h_true, :out_w_true]).float().mean().item()

            acc_sum += acc
            total_samples += 1

    avg_acc = acc_sum / max(total_samples, 1)
    return avg_acc, perfect_count, total_samples


def make_qwen_finetunable(device="cuda", model_name=None):
    """
    Create Qwen model with fine-tuning enabled (NOT frozen).
    """
    import os


    name = model_name or os.getenv("QWEN_MODEL", "Qwen/Qwen1.5-0.5B")
    candidates = [
        name,
        "Qwen/Qwen1.5-0.5B",
        "Qwen/Qwen1.5-0.5B-Chat",
        "Qwen/Qwen2.5-0.5B-Instruct",
    ]

    last_err = None
    for cand in candidates:
        try:
            qcfg = QwenCfg(
                model_name=cand,
                dtype="float16",
                temperature=0.0,
                use_qwen=True
            )
            qwen = QwenHybridPrompt(
                prompt_dim=256,
                numeric_in_dim=15,
                fuse="mean",
                qwen=qcfg
            ).to(device)
            print(f"[Qwen] Loaded model: {cand}")
            return qwen
        except Exception as e:
            last_err = e
            print(f"[Qwen] Failed to load {cand}: {e}")

    raise RuntimeError(f"Failed to load any Qwen model. Last error: {last_err}")



def main(epochs=10, agent_lr=1e-5, qwen_lr=None, weight_decay=1e-6,
         grad_accum_steps=1, grad_clip=1.0, warmup_steps=100,
         max_batches_per_epoch=None, val_frequency=1, skip_test=False,
         device="cuda", model_name=None, seed=42, save_frequency=1,
         freeze_qwen=True):
    """
    Main training loop for GPU with Qwen fine-tuning.
    """
    print("\n" + "="*70)
    print("GPU TRAINING WITH QWEN FINE-TUNING".center(70))
    print("="*70 + "\n")

    seed_all(seed)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"runs/arc_gpu_finetuned_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    logger = TrainingLogger(output_dir)

    logger.log(f"SYSTEM CONFIG:")
    logger.log(f"  Device: {device}")
    logger.log(f"  Seed: {seed}")
    logger.log(f"\nTRAINING HYPERPARAMETERS:")
    logger.log(f"  Epochs: {epochs}")
    logger.log(f"  Agent LR: {agent_lr}")
    logger.log(f"  Weight Decay: {weight_decay}")
    logger.log(f"  Gradient Accumulation Steps: {grad_accum_steps}")
    logger.log(f"  Gradient Clip Norm: {grad_clip}")
    logger.log(f"  Warmup Steps: {warmup_steps}")
    logger.log(f"  Validation Frequency: every {val_frequency} epochs")
    logger.log(f"  Save Frequency: every {save_frequency} epochs")
    logger.log(f"\nMODEL CONFIG:")
    logger.log(f"  Qwen Model: {model_name or 'Qwen/Qwen1.5-0.5B (lightweight, Colab-compatible)'}")
    logger.log(f"  Qwen fine-tuning: {'DISABLED (FROZEN - stable)' if freeze_qwen else 'ENABLED (NOT frozen - unstable)'}")
    logger.log(f"\nDATA CONFIG:")
    logger.log(f"  Max batches per epoch: {'UNLIMITED (full dataset)' if not max_batches_per_epoch else max_batches_per_epoch}")
    logger.log(f"  Loss function: Grid Accuracy Based Loss")
    logger.log(f"  Validation: Binary Accuracy (strict ARC evaluation)")
    logger.log(f"\nOutput: {output_dir}\n")

    
    data_dir = os.getenv("ARC_DATA_DIR", ".")
    train_path = os.path.join(data_dir, "training.json")
    logger.log("Loading datasets...")
    train_ds = ARCDataset(train_path, split="train")
    test_ds  = ARCDataset(train_path, split="test")
    val_ds   = ARCDataset(train_path, split="test")  

    train_ld = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_ld   = DataLoader(val_ds,  batch_size=1, shuffle=False)
    test_ld  = DataLoader(test_ds, batch_size=1, shuffle=False)

    logger.log(f"  Train: {len(train_ds)} samples")
    logger.log(f"  Val:   {len(val_ds)} samples")
    logger.log(f"  Test:  {len(test_ds)} samples\n")

    # ----- Qwen  -----
    logger.log("Creating model components...")
    qwen = make_qwen_finetunable(device=device, model_name=model_name)

    # Freeze Qwen if requested (stable training)
    if freeze_qwen:
        for param in qwen.parameters():
            param.requires_grad = False
        logger.log("  Qwen loaded (FROZEN - stable training)")
        qwen_trainable_params = 0
    else:
        logger.log("  Qwen loaded (TRAINABLE - unstable, not recommended)")
        qwen_trainable_params = sum(p.numel() for p in qwen.parameters() if p.requires_grad)

    # ----- Agent  -----
    max_planning_steps = 5  # Maximum planning steps
    agent = ARCPromptGuidedAgentGPU(
        max_grid_size=30,
        num_colors=10,
        hidden_dim=256,
        prompt_dim=256,
        max_steps=max_planning_steps
    ).to(device)
    logger.log(" Agent created with grid accuracy loss")

    # ----- Solver2 (Memory-based learning) -----
    solver2 = PermanentSolver(
        input_dim=256,        # Problem feature dimension (from Qwen)
        hidden_dim=512,       # Internal attention/fusion dimension
        max_grid_size=30,
        num_colors=10
    ).to(device)
    logger.log(" Solver2 created with persistent memory bank")

    # ----- EFE Loss Function -----
    efe_loss = EFELoss(
        lambda_risk=1.0,
        lambda_amb=0.0,       # Disabled (not needed for grid prediction)
        lambda_step=0.1,      # Penalize long inference chains
        lambda_cons=1.0,      # Target consistency (main loss)
        lambda_bi=0.5,        # Bi-directional agreement
        lambda_z=0.2,         # Z-learning anchoring
        lambda_prompt=0.3,    # Prompt consistency
        max_grid_size=30,
        num_colors=10,
        prompt_dim=256
    )
    logger.log(" EFE Loss function created with 7 components")

    # ----- Optimizer -----
    agent_params = [p for p in agent.parameters() if p.requires_grad]
    solver2_params = [p for p in solver2.parameters() if p.requires_grad]
    efe_loss_params = [p for p in efe_loss.parameters() if p.requires_grad]

    all_trainable_params = agent_params + solver2_params + efe_loss_params

    logger.log(f"  Agent parameters:   {sum(p.numel() for p in agent.parameters()):,} (trainable)")
    logger.log(f"  Solver2 parameters: {sum(p.numel() for p in solver2.parameters()):,} (trainable)")
    logger.log(f"  EFELoss parameters: {sum(p.numel() for p in efe_loss.parameters()):,} (trainable)")
    logger.log(f"  Qwen parameters:    {sum(p.numel() for p in qwen.parameters()):,} ({'frozen' if freeze_qwen else 'trainable'})")
    logger.log(f"  Total trainable:    {sum(p.numel() for p in all_trainable_params) + qwen_trainable_params:,}")

    optim = torch.optim.Adam([
        {"params": agent_params, "lr": agent_lr, "weight_decay": weight_decay},
        {"params": solver2_params, "lr": agent_lr * 2.0, "weight_decay": weight_decay},  # Slightly higher LR for memory
        {"params": efe_loss_params, "lr": agent_lr * 0.5, "weight_decay": weight_decay},  # Lower LR for loss
    ])
    logger.log(f"  Optimizer created (Adam, agent_lr={agent_lr})\n")

    feat_reg = FeatureRegistry()

    best_acc = -1
    best_ckpt = os.path.join(output_dir, "agent_best.pt")
    qwen_ckpt = os.path.join(output_dir, "qwen_best.pt")

    logger.log("Starting training...\n")
    logger.log("="*70)

    for epoch in range(epochs):
        logger.log(f"\n[Epoch {epoch}/{epochs-1}]")

        avg_loss = train_epoch(agent, qwen, solver2, efe_loss, train_ld, optim, device, feat_reg,
                               epoch, logger, max_batches_per_epoch)
        logger.log(f"  Average Loss: {avg_loss:.6f}")

        if not skip_test:
            val_acc, val_perfect, val_total = evaluate(
                agent, qwen, solver2, efe_loss, val_ld, device, feat_reg,
                max_batches=None, binary_accuracy=True
            )
            logger.log(f"  Val Accuracy: {val_acc:.4f} ({val_perfect}/{val_total} grids perfect)")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(agent.state_dict(), best_ckpt)
                torch.save(qwen.state_dict(),  qwen_ckpt)
                logger.log(f"  ✓ Best checkpoint saved!")

    logger.log("\n" + "="*70)
    logger.log("FINAL TEST EVALUATION (Strict Binary Accuracy)")
    logger.log("="*70)

    test_acc, test_perfect, test_total = evaluate(
        agent, qwen, solver2, efe_loss, test_ld, device, feat_reg, max_batches=None, binary_accuracy=True
    )
    logger.log(f"\nTest Accuracy (Binary): {test_acc:.4f} ({test_perfect}/{test_total} grids perfect)")

    logger.log("\n" + "="*70)
    logger.log("TRAINING COMPLETED".center(70))
    logger.log("="*70 + "\n")

    logger.log(f"Output directory: {output_dir}")
    logger.log(f"Agent checkpoint: {best_ckpt}")
    logger.log(f"Qwen checkpoint: {qwen_ckpt}")
    logger.log(f"Final test accuracy: {test_acc:.4f}\n")

    return output_dir, best_acc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="GPU Training with Qwen Fine-Tuning and Grid Accuracy Loss",
        epilog="""
KEY FEATURES:
  - Qwen fine-tuning: ENABLED (learns task-specific embeddings)
  - Loss function: Grid accuracy based (1 - accuracy)
  - Validation: Binary accuracy (entire grid correct or not)
  - Optimization: Lower learning rate for LM stability

GPU REQUIREMENTS:
  - NVIDIA GPU with 8GB+ VRAM (Colab T4/A100, RTX 3060, RTX 3080, etc)
  - Qwen1.5-0.5B: ~2GB model, fits easily in Colab
  - CUDA toolkit installed
  - Qwen requires HuggingFace authentication

        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # ===== TRAINING HYPERPARAMETERS =====
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of epochs (default: 5). ")

    parser.add_argument("--agent-lr", type=float, default=1e-5,
                       help="Agent learning rate (default: 1e-5). Reduced for stability.")

    parser.add_argument("--qwen-lr", type=float, default=None,
                       help="Qwen LM learning rate (default: None, Qwen is frozen). Ignored if freeze_qwen=True.")

    parser.add_argument("--weight-decay", type=float, default=1e-6,
                       help="Weight decay for regularization (default: 1e-6). Reduced for stability.")

    parser.add_argument("--grad-accum-steps", type=int, default=1,
                       help="Gradient accumulation steps (default: 1). ")

    parser.add_argument("--grad-clip", type=float, default=1.0,
                       help="Gradient clipping norm (default: 1.0). Prevents instability during fine-tuning")

    parser.add_argument("--warmup-steps", type=int, default=100,
                       help="Warmup steps for learning rate (default: 100). Stabilizes training start")

    # ===== DATA PARAMETERS =====
    parser.add_argument("--max-batches", type=int, default=None,
                       help="Max batches per epoch (default: None = full dataset). Use for quick debugging")

    parser.add_argument("--val-frequency", type=int, default=1,
                       help="Validate every N epochs (default: 1). Can skip to speed up training")

    # ===== MODEL SELECTION =====
    parser.add_argument("--model-name",
                       type=str,
                       default="Qwen/Qwen1.5-0.5B",
                       help=" (default: Qwen1.5-0.5B)")

    # ===== SYSTEM PARAMETERS =====
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device: cuda or cpu (default: cuda)")

    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")

    parser.add_argument("--skip-test", action="store_true",
                       help="Skip final test evaluation")

    parser.add_argument("--no-freeze-qwen", action="store_false", dest="freeze_qwen",
                       help="Train Qwen (default: frozen for stability). Not recommended due to instability.")

    parser.add_argument("--save-frequency", type=int, default=1,
                       help="Save checkpoint every N epochs (default: 1)")


    args = parser.parse_args()


    try:
        output_dir, final_acc = main(
            epochs=args.epochs,
            agent_lr=args.agent_lr,
            qwen_lr=args.qwen_lr,
            weight_decay=args.weight_decay,
            grad_accum_steps=args.grad_accum_steps,
            grad_clip=args.grad_clip,
            warmup_steps=args.warmup_steps,
            max_batches_per_epoch=args.max_batches,
            val_frequency=args.val_frequency,
            skip_test=args.skip_test,
            device=args.device,
            model_name=args.model_name,
            seed=args.seed,
            save_frequency=args.save_frequency,
            freeze_qwen=args.freeze_qwen
        )
        print(f"\n✓ Training completed! Results: {output_dir}")
        print(f"  Best accuracy: {final_acc:.4f}")
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
