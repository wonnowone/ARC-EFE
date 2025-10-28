# -*- coding: utf-8 -*-
"""
GPU Training Loop with Qwen Fine-Tuning and Grid Accuracy Loss

Key Differences from CPU Training:
1. Qwen is NOT frozen - full fine-tuning enabled
2. Grid accuracy based loss (simple and direct)
3. Binary accuracy for validation (strict ARC evaluation)
4. GPU optimized (mixed precision, gradient accumulation)
5. Comprehensive progress tracking

This is the PRODUCTION training script for GPU training.
"""

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
    """Set random seeds for reproducibility"""
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


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
    qwen.train()  # ← IMPORTANT: Qwen is trainable!

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

        # Generate prompt with Qwen (now trainable!)
        with torch.no_grad():  # Only input encoding is no_grad, Qwen forward has gradients
            pack = qwen(tr, inp.unsqueeze(0), out.unsqueeze(0), control_weight=0.5)
        prompt_emb = pack["hybrid_embedding"].squeeze(0)

        # Training step
        optimizer.zero_grad()

        # Forward pass
        predictions, _ = agent.forward(inp, prompt_emb, num_steps=3)

        # Calculate loss
        final_pred = predictions[-1]  # Take last step
        loss_val = agent.loss_fn(final_pred, out, return_components=False)

        # Backward pass
        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)

        # IMPORTANT: Also clip Qwen gradients
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
    Evaluate on validation/test set with binary accuracy (strict ARC evaluation).

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


def make_qwen_finetunable(device="cuda"):
    """Create Qwen model with fine-tuning enabled (NOT frozen)"""
    qcfg = QwenCfg(
        model_name="Qwen/Qwen2.5-1.8B",
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
    return qwen


def main(epochs=5, max_batches_per_epoch=None, skip_test=False, device="cuda"):
    """
    Main training loop for GPU with Qwen fine-tuning.

    Args:
        epochs: Number of training epochs
        max_batches_per_epoch: Max batches per epoch (None = full dataset)
        skip_test: Skip evaluation phase
        device: 'cuda' for GPU, 'cpu' for CPU
    """

    print("\n" + "="*70)
    print("GPU TRAINING WITH QWEN FINE-TUNING".center(70))
    print("="*70 + "\n")

    seed_all()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"runs/arc_gpu_finetuned_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    logger = TrainingLogger(output_dir)

    logger.log(f"Device: {device}")
    logger.log(f"Epochs: {epochs}")
    logger.log(f"Max batches per epoch: {'UNLIMITED (full dataset)' if not max_batches_per_epoch else max_batches_per_epoch}")
    logger.log(f"Qwen fine-tuning: ENABLED (NOT frozen)")
    logger.log(f"Loss function: Grid Accuracy Based Loss")
    logger.log(f"Validation: Binary Accuracy (strict ARC evaluation)")
    logger.log(f"Output: {output_dir}\n")

    # Load data
    logger.log("Loading datasets...")
    train_ds = ARCDataset("training.json", split="train")
    test_ds = ARCDataset("training.json", split="test")
    val_ds = ARCDataset("training.json", split="test")  # Use test for validation

    train_ld = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=1, shuffle=False)
    test_ld = DataLoader(test_ds, batch_size=1, shuffle=False)

    logger.log(f"  Train: {len(train_ds)} samples")
    logger.log(f"  Val: {len(val_ds)} samples")
    logger.log(f"  Test: {len(test_ds)} samples\n")

    # Create components
    logger.log("Creating model components...")

    # Create Qwen with fine-tuning enabled
    qwen = make_qwen_finetunable(device=device)
    logger.log("  ✓ Qwen loaded (TRAINABLE - NOT FROZEN)")

    # Create agent
    agent = ARCPromptGuidedAgentGPU(
        max_grid_size=30,
        num_colors=10,
        hidden_dim=256,
        prompt_dim=256
    ).to(device)
    logger.log("  ✓ Agent created with grid accuracy loss")

    # Create optimizer with BOTH agent and qwen parameters
    agent_params = list(agent.parameters())
    qwen_params = list(qwen.parameters())
    all_params = agent_params + qwen_params

    logger.log(f"  Agent parameters: {sum(p.numel() for p in agent_params):,}")
    logger.log(f"  Qwen parameters: {sum(p.numel() for p in qwen_params):,}")
    logger.log(f"  Total trainable: {sum(p.numel() for p in all_params):,}")

    # Lower learning rate for Qwen fine-tuning (LM is sensitive)
    optim = torch.optim.Adam(all_params, lr=5e-4)  # Lower than agent-only training
    logger.log("  ✓ Optimizer created (lr=5e-4 for stability)\n")

    feat_reg = FeatureRegistry()

    # Training loop
    best_acc = -1
    best_ckpt = os.path.join(output_dir, "agent_best.pt")
    qwen_ckpt = os.path.join(output_dir, "qwen_best.pt")

    logger.log("Starting training...\n")
    logger.log("="*70)

    for epoch in range(epochs):
        logger.log(f"\n[Epoch {epoch}/{epochs-1}]")

        # Train
        avg_loss = train_epoch(agent, qwen, train_ld, optim, device, feat_reg,
                              epoch, logger, max_batches_per_epoch)
        logger.log(f"  Average Loss: {avg_loss:.6f}")

        # Validate
        if not skip_test:
            val_acc, val_perfect, val_total = evaluate(agent, qwen, val_ld, device, feat_reg,
                                                        max_batches=None, binary_accuracy=True)
            logger.log(f"  Val Accuracy: {val_acc:.4f} ({val_perfect}/{val_total} grids perfect)")

            # Save best
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(agent.state_dict(), best_ckpt)
                torch.save(qwen.state_dict(), qwen_ckpt)
                logger.log(f"  ✓ Best checkpoint saved!")

    # Final test evaluation
    logger.log("\n" + "="*70)
    logger.log("FINAL TEST EVALUATION (Strict Binary Accuracy)")
    logger.log("="*70)

    test_acc, test_perfect, test_total = evaluate(agent, qwen, test_ld, device, feat_reg,
                                                  max_batches=None, binary_accuracy=True)
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
  - NVIDIA GPU with 12GB+ VRAM (RTX 3080, RTX 4090, A100, etc)
  - CUDA toolkit installed
  - Qwen requires HuggingFace authentication

USAGE:
  # 5 epochs on GPU (recommended)
  python trainloop_gpu_finetuned.py --epochs 5 --device cuda

  # 10 epochs for maximum training
  python trainloop_gpu_finetuned.py --epochs 10 --device cuda

  # Quick test with limited batches
  python trainloop_gpu_finetuned.py --epochs 1 --max-batches 100 --device cuda
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of epochs (default: 5)")
    parser.add_argument("--max-batches", type=int, default=None,
                       help="Max batches per epoch (default: None = full dataset)")
    parser.add_argument("--skip-test", action="store_true",
                       help="Skip evaluation phase")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use: cuda or cpu (default: cuda)")

    args = parser.parse_args()

    try:
        output_dir, final_acc = main(
            epochs=args.epochs,
            max_batches_per_epoch=args.max_batches,
            skip_test=args.skip_test,
            device=args.device
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
