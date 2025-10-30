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


def train_epoch(agent, qwen, train_loader, optimizer, device, feat_reg, epoch, logger, max_batches=None):
    """
    Train one epoch with Qwen fine-tuning enabled.

    Returns:
        avg_loss: Average loss for epoch
    """
    agent.train()
    qwen.train()  

    epoch_loss = 0.0
    batches_processed = 0
    total_batches = len(train_loader)

    for batch_idx, batch in enumerate(train_loader):
        if max_batches and batch_idx >= max_batches:
            break

        # Progress every 50 batches
        if (batch_idx + 1) % 50 == 0 or batch_idx == total_batches - 1:
            print(f"  Processing batch {batch_idx + 1}/{total_batches}...")

        inp = batch["input"].to(device)
        out = batch["output"].to(device) if batch["output"] is not None else None

        if out is None:
            continue

        # Squeeze batch dimension
        inp = inp.squeeze(0) if inp.dim() == 3 else inp
        out = out.squeeze(0) if out.dim() == 3 else out

        # Handle size mismatch
        if inp.shape != out.shape:
            if inp.shape[0] > out.shape[0] and inp.shape[1] > out.shape[1]:
                # Pad output
                pad_h = inp.shape[0] - out.shape[0]
                pad_w = inp.shape[1] - out.shape[1]
                out = torch.nn.functional.pad(out, (0, pad_w, 0, pad_h), mode='constant', value=0)
            elif inp.shape[0] < out.shape[0] or inp.shape[1] < out.shape[1]:
                continue
            else:
                continue

        # Feature extraction
        tr = pack_transform_record(inp.unsqueeze(0), out.unsqueeze(0))
        tr = apply_operator_config(tr, inp.unsqueeze(0), out.unsqueeze(0), feat_reg)

        # Get hybrid embedding from Qwen (no no_grad to enable fine-tuning)
        pack = qwen(tr, inp.unsqueeze(0), out.unsqueeze(0), control_weight=0.5)
        prompt_emb = pack["hybrid_embedding"].squeeze(0)

        # Training step
        optimizer.zero_grad()

        # Forward pass
        predictions, _ = agent.forward(inp, prompt_emb, num_steps=3)

        # Calculate loss
        final_pred = predictions[-1]  
        loss_val = agent.loss_fn(final_pred, out, return_components=False)

        # Backward pass
        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)

        for param in qwen.parameters():
            if param.grad is not None:
                torch.nn.utils.clip_grad_norm_(param, max_norm=1.0)

        optimizer.step()

        loss_item = loss_val.item()
        epoch_loss += loss_item
        batches_processed += 1

    avg_loss = epoch_loss / max(batches_processed, 1)
    return avg_loss


def evaluate(agent, qwen, eval_loader, device, feat_reg, max_batches=None, binary_accuracy=True):
    """
    Evaluate on validation/test set with binary accuracy 

    Args:
        binary_accuracy: If True, accuracy is 1.0 only if entire grid perfect, else 0.0

    Returns:
        (avg_accuracy, perfect_count)
    """
    agent.eval()
    qwen.eval()

    acc_sum = 0.0
    perfect_count = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            if max_batches and batch_idx >= max_batches:
                break

            inp = batch["input"].to(device)
            out = batch["output"].to(device) if batch["output"] is not None else None

            if out is None:
                continue

            # Squeeze batch dimension
            inp = inp.squeeze(0) if inp.dim() == 3 else inp
            out = out.squeeze(0) if out.dim() == 3 else out

            out_h_true, out_w_true = out.shape

            # Handle size mismatch
            if inp.shape != out.shape:
                if inp.shape[0] > out.shape[0] and inp.shape[1] > out.shape[1]:
                    pad_h = inp.shape[0] - out.shape[0]
                    pad_w = inp.shape[1] - out.shape[1]
                    out = torch.nn.functional.pad(out, (0, pad_w, 0, pad_h), mode='constant', value=0)
                else:
                    continue

            # Feature extraction
            tr = pack_transform_record(inp.unsqueeze(0), out.unsqueeze(0))
            tr = apply_operator_config(tr, inp.unsqueeze(0), out.unsqueeze(0), feat_reg)

            # Generate prompt with Qwen
            pack = qwen(tr, inp.unsqueeze(0), out.unsqueeze(0), control_weight=0.5)
            prompt_emb = pack["hybrid_embedding"].squeeze(0)

            # Forward pass
            predictions, _ = agent.forward(inp, prompt_emb, num_steps=5)
            final_pred = predictions[-1].argmax(dim=-1)

            # Crop to true output size if padding was added
            if final_pred.shape != (out_h_true, out_w_true):
                final_pred = final_pred[:out_h_true, :out_w_true]

            # Calculate accuracy
            if binary_accuracy:
                # Binary: 1.0 only if entire grid perfect, else 0.0
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



def main(epochs=10, agent_lr=5e-4, qwen_lr=1e-4, weight_decay=1e-5,
         grad_accum_steps=1, grad_clip=1.0, warmup_steps=100,
         max_batches_per_epoch=None, val_frequency=1, skip_test=False,
         device="cuda", model_name=None, seed=42, save_frequency=1):
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
    logger.log(f"  Agent LR: {agent_lr} | Qwen LR: {qwen_lr}")
    logger.log(f"  Weight Decay: {weight_decay}")
    logger.log(f"  Gradient Accumulation Steps: {grad_accum_steps}")
    logger.log(f"  Gradient Clip Norm: {grad_clip}")
    logger.log(f"  Warmup Steps: {warmup_steps}")
    logger.log(f"  Validation Frequency: every {val_frequency} epochs")
    logger.log(f"  Save Frequency: every {save_frequency} epochs")
    logger.log(f"\nMODEL CONFIG:")
    logger.log(f"  Qwen Model: {model_name or 'Qwen/Qwen1.5-0.5B (lightweight, Colab-compatible)'}")
    logger.log(f"  Qwen fine-tuning: ENABLED (NOT frozen)")
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
    logger.log("  Qwen loaded (TRAINABLE - NOT FROZEN)")

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

    # ----- Optimizer -----
    agent_params = list(agent.parameters())
    qwen_params  = list(qwen.parameters())
    total_params = agent_params + qwen_params

    logger.log(f"  Agent parameters: {sum(p.numel() for p in agent_params):,}")
    logger.log(f"  Qwen parameters:  {sum(p.numel() for p in qwen_params):,}")
    logger.log(f"  Total trainable:  {sum(p.numel() for p in total_params):,}")

    optim = torch.optim.Adam([
        {"params": agent_params, "lr": agent_lr, "weight_decay": weight_decay},
        {"params": qwen_params,  "lr": qwen_lr, "weight_decay": weight_decay * 0.1},   # Lower weight decay for LM
    ])
    logger.log(f"  Optimizer created (Adam with specified learning rates)\n")

    feat_reg = FeatureRegistry()

    best_acc = -1
    best_ckpt = os.path.join(output_dir, "agent_best.pt")
    qwen_ckpt = os.path.join(output_dir, "qwen_best.pt")

    logger.log("Starting training...\n")
    logger.log("="*70)

    for epoch in range(epochs):
        logger.log(f"\n[Epoch {epoch}/{epochs-1}]")

        avg_loss = train_epoch(agent, qwen, train_ld, optim, device, feat_reg,
                               epoch, logger, max_batches_per_epoch)
        logger.log(f"  Average Loss: {avg_loss:.6f}")

        if not skip_test:
            val_acc, val_perfect, val_total = evaluate(
                agent, qwen, val_ld, device, feat_reg,
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
        agent, qwen, test_ld, device, feat_reg, max_batches=None, binary_accuracy=True
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

    parser.add_argument("--agent-lr", type=float, default=5e-4,
                       help="Agent learning rate (default: 5e-4). ")

    parser.add_argument("--qwen-lr", type=float, default=1e-4,
                       help="Qwen LM learning rate (default: 1e-4). ")

    parser.add_argument("--weight-decay", type=float, default=1e-5,
                       help="Weight decay for regularization (default: 1e-5). Prevents overfitting on small ARC dataset")

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
            save_frequency=args.save_frequency
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
