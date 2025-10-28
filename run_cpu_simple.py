# -*- coding: utf-8 -*-
"""
Simplified CPU Training Script - No Qwen Authentication Required

This script trains without requiring Qwen model (uses dummy prompts instead).
Perfect for CPU-only systems without HuggingFace authentication.
"""

import os
import sys
import torch
from datetime import datetime
from torch.utils.data import DataLoader

from dataset_arc import ARCDataset
from revthink_orchestrator import RevThinkOrchestrator, RevThinkCfg
from loss_function import ARCPromptGuidedAgent
from feature_registry import FeatureRegistry, apply_operator_config
from feature_extraction import extract_transformation_features, classify_transformation_type


def seed_all(seed=42):
    """Set random seeds"""
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


def train_epoch(agent, loader, optim, device, feat_reg, epoch, max_batches=None):
    """Train one epoch"""
    agent.train()
    epoch_loss = 0.0
    batches_processed = 0
    skipped = 0
    total_batches = len(loader)

    for batch_idx, batch in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break

        # Progress every 50 batches or at end
        if (batch_idx + 1) % 50 == 0 or batch_idx == total_batches - 1:
            print(f"  Processing batch {batch_idx + 1}/{total_batches}...")

        inp = batch["input"].to(device)
        out = batch["output"].to(device) if batch["output"] is not None else None

        if out is None:
            continue

        # Squeeze batch dimension
        inp = inp.squeeze(0) if inp.dim() == 3 else inp
        out = out.squeeze(0) if out.dim() == 3 else out

        # Skip if sizes don't match (agent requires same-size input/output)
        if inp.shape != out.shape:
            skipped += 1
            # Try to resize output to match input
            if inp.shape[0] > out.shape[0] and inp.shape[1] > out.shape[1]:
                # Pad output
                pad_h = inp.shape[0] - out.shape[0]
                pad_w = inp.shape[1] - out.shape[1]
                out = torch.nn.functional.pad(out, (0, pad_w, 0, pad_h), mode='constant', value=0)
            elif inp.shape[0] < out.shape[0] or inp.shape[1] < out.shape[1]:
                # Output is larger, skip this sample
                continue
            else:
                continue

        # Feature extraction
        tr = pack_transform_record(inp.unsqueeze(0), out.unsqueeze(0))
        tr = apply_operator_config(tr, inp.unsqueeze(0), out.unsqueeze(0), feat_reg)

        # Create simple prompt embedding (no Qwen needed)
        prompt_emb = torch.randn(256).to(device)
        prompt_text = f"Transform grid {inp.shape}"

        # Training step
        optim.zero_grad()
        losses = agent.train_episode(
            initial_state=inp.squeeze(0),
            target_state=out.squeeze(0),
            prompt_text=prompt_text,
            prompt_embedding=prompt_emb,
            num_steps=3
        )

        # Backward
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
        optim.step()

        loss_val = losses["total"].item()
        epoch_loss += loss_val
        batches_processed += 1

    avg_loss = epoch_loss / max(batches_processed, 1)
    return avg_loss


def evaluate(agent, loader, device, feat_reg, max_batches=None, binary_accuracy=False, predict_size=False):
    """
    Evaluate on test set

    Args:
        binary_accuracy: If True, accuracy is 1.0 only if entire grid correct, else 0.0
                        If False, accuracy is per-cell (partial credit)
        predict_size: If True, try to predict output grid size (advanced)
    """
    agent.eval()
    acc_sum, n = 0.0, 0
    perfect_count = 0  # For binary accuracy
    total_batches = len(loader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
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
                    # Pad output to match input (current approach)
                    pad_h = inp.shape[0] - out.shape[0]
                    pad_w = inp.shape[1] - out.shape[1]
                    out = torch.nn.functional.pad(out, (0, pad_w, 0, pad_h), mode='constant', value=0)
                else:
                    continue

            tr = pack_transform_record(inp.unsqueeze(0), out.unsqueeze(0))
            tr = apply_operator_config(tr, inp, out, feat_reg)

            # Simple prompt
            prompt_emb = torch.randn(256).to(device)

            # Forward planning
            preds, _ = agent.forward_planning(inp.squeeze(0), prompt_emb, num_steps=5)

            final = preds[-1].argmax(dim=-1)

            # Crop to true output size if padding was added
            if final.shape != (out_h_true, out_w_true):
                final = final[:out_h_true, :out_w_true]

            # Calculate accuracy
            if binary_accuracy:
                # Binary: 1.0 only if entire grid perfect, else 0.0
                is_perfect = (final == out[:out_h_true, :out_w_true]).all().item()
                acc = 1.0 if is_perfect else 0.0
                if is_perfect:
                    perfect_count += 1
            else:
                # Per-cell: partial credit
                acc = (final == out[:out_h_true, :out_w_true]).float().mean().item()

            acc_sum += acc
            n += 1

    if binary_accuracy:
        return acc_sum / max(n, 1), perfect_count  # Return accuracy and perfect count
    else:
        return acc_sum / max(n, 1)


def main(epochs=3, max_batches_per_epoch=None, skip_test=False, binary_accuracy=False):
    """
    Simple training loop

    Args:
        epochs: Number of training epochs
        max_batches_per_epoch: Max batches per epoch (for quick testing)
        skip_test: Skip evaluation phase
        binary_accuracy: If True, accuracy is 1.0 only if entire grid perfect, else 0.0
    """

    print("\n" + "="*70)
    print("SIMPLE CPU TRAINING (No Qwen Auth Required)".center(70))
    print("="*70 + "\n")

    seed_all()
    device = "cpu"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"runs/arc_simple_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    if max_batches_per_epoch:
        print(f"Max batches per epoch: {max_batches_per_epoch}")
    else:
        print(f"Max batches per epoch: UNLIMITED (full dataset)")
    print(f"Output: {output_dir}\n")

    # Load data
    print("Loading datasets...")
    train_ds = ARCDataset("training.json", split="train")
    test_ds = ARCDataset("training.json", split="test")
    train_ld = DataLoader(train_ds, batch_size=1, shuffle=True)
    test_ld = DataLoader(test_ds, batch_size=1, shuffle=False)
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Test: {len(test_ds)} samples\n")

    # Create components
    print("Creating model components...")
    agent = ARCPromptGuidedAgent(
        max_grid_size=30,
        num_colors=10,
        hidden_dim=256,
        prompt_dim=256
    ).to(device)
    revthink = RevThinkOrchestrator(qwen=None, cfg=RevThinkCfg())  # No Qwen
    feat_reg = FeatureRegistry()
    optim = torch.optim.Adam(agent.parameters(), lr=1e-3)
    print("  Done!\n")

    # Training
    best_acc = -1
    best_ckpt = os.path.join(output_dir, "agent_best.pt")

    print("Starting training...\n")
    print("="*70)

    for epoch in range(epochs):
        print(f"\n[Epoch {epoch}/{epochs-1}]")

        # Train
        avg_loss = train_epoch(agent, train_ld, optim, device, feat_reg, epoch, max_batches_per_epoch)
        print(f"  Average Loss: {avg_loss:.6f}")

        # Evaluate
        if not skip_test:
            result = evaluate(agent, test_ld, device, feat_reg, max_batches=None, binary_accuracy=binary_accuracy)

            if binary_accuracy:
                avg_acc, perfect_count = result
                print(f"  Test Accuracy: {avg_acc:.4f} ({perfect_count}/{min(1076, len(test_ld))} grids perfect)")
            else:
                avg_acc = result
                print(f"  Test Accuracy: {avg_acc:.4f} (on {min(1076, len(test_ld))} test samples, per-cell)")

            # Save best
            if avg_acc > best_acc:
                best_acc = avg_acc
                torch.save(agent.state_dict(), best_ckpt)
                print(f"  Best checkpoint saved!")

    print("\n" + "="*70)
    print("TRAINING COMPLETED".center(70))
    print("="*70 + "\n")

    print(f"Output directory: {output_dir}")
    print(f"Best model: {best_ckpt}")
    print(f"Final accuracy: {best_acc:.4f}\n")

    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Simple CPU Training - Full Dataset",
        epilog="""
ACCURACY METRIC:
  - Accuracy = percentage of correctly predicted cells in the output grid
  - Example: 95/100 correct cells = 0.95 accuracy
  - Ranges from 0.0 (all wrong) to 1.0 (perfect)

EPOCH LIMITS:
  - Default: 3 epochs (~45 minutes)
  - Use --epochs to set custom limit
  - Larger epochs = longer training, potentially better model
  - No hard limit in code (you control via --epochs)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of epochs (default: 3). No hard limit, set as needed")
    parser.add_argument("--max-batches", type=int, default=None,
                       help="Max batches per epoch (default: None = full 3232 samples). Use for quick testing")
    parser.add_argument("--skip-test", action="store_true",
                       help="Skip evaluation phase (faster, but no accuracy metric)")
    parser.add_argument("--binary-accuracy", action="store_true",
                       help="Use binary accuracy (1.0 only if entire grid perfect, else 0.0) instead of per-cell")

    args = parser.parse_args()

    try:
        output_dir = main(
            epochs=args.epochs,
            max_batches_per_epoch=args.max_batches,
            skip_test=args.skip_test,
            binary_accuracy=args.binary_accuracy
        )
        print(f"Training completed! Results: {output_dir}")
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
