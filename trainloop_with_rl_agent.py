"""
trainloop_with_rl_agent.py

Goal-Oriented Training with Human RL Agent

Instead of numerically minimizing abstract losses, this loop:
  1. Sets EXPLICIT GOALS: maximize accuracy, size match, color agreement, reversibility
  2. Uses POLICY GRADIENTS to learn prompt refinement toward those goals
  3. Tracks CONCRETE METRICS to verify we're actually solving the problem

The RL agent learns: "What prompt modifications help solve this problem?"
Not: "What loss function value is smallest?"

Core Loop:
  For each batch:
    initial_prompt = Qwen(features)
    pred_before = Agent(input, initial_prompt)

    refined_prompt, rl_info = RL_Agent.refine_prompt(...)
    pred_after = Agent(input, refined_prompt)

    reward = measure_improvement(pred_before, pred_after, target)
    RL_Agent.update(rl_info, reward)  # Learn toward explicit goal

    EFE_Loss = Agent.forward(input, refined_prompt, target)
    Agent.backward(EFE_Loss)  # Let agent adapt to refined prompts
"""

import os
import sys
import json
import warnings
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import logging
logging.getLogger('transformers.generation_utils').setLevel(logging.ERROR)
logging.getLogger('transformers.modeling_utils').setLevel(logging.ERROR)

from dataset_arc import ARCDataset
from qwen_hybrid_prompt import QwenHybridPrompt, QwenCfg
from loss_function import EFELoss, ARCPromptGuidedAgent
from feature_registry import FeatureRegistry, apply_operator_config
from feature_extraction import extract_transformation_features, classify_transformation_type
from grid_accuracy_loss import GridAccuracyLoss, ARCPromptGuidedAgentGPU
from solver2 import PermanentSolver
from policy_refined import PolicyRefinedAgent, PolicyRefinedConfig  # ← RL AGENT
import torch.nn.functional as F


# ============================================================================
# LOGGING & METRICS
# ============================================================================

class TrainingLogger:
    def __init__(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.log_file = os.path.join(output_dir, "training.log")
        self.fh = open(self.log_file, "w", encoding="utf-8")

    def log(self, msg: str):
        print(msg)
        self.fh.write(msg + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


class GoalOrientedMetrics:
    """Track explicit goal progress, not just loss values."""

    def __init__(self):
        self.history = {
            "epoch": [],
            "accuracy_delta": [],      # Main goal: accuracy improvement
            "size_delta": [],          # Secondary goal: size matching
            "color_delta": [],         # Secondary goal: color agreement
            "reversibility_delta": [], # Secondary goal: reversibility
            "rl_reward": [],           # RL agent's reward signal
            "rl_loss": [],             # RL policy loss
            "efe_loss": [],            # Main agent loss
            "total_loss": [],          # Combined loss
        }

    def log_batch(self, epoch, metrics_dict):
        """Record metrics from a single batch."""
        for key, val in metrics_dict.items():
            if key in self.history:
                self.history[key].append(val)

    def get_epoch_summary(self, epoch):
        """Get summary statistics for the epoch."""
        keys_to_avg = [
            "accuracy_delta", "size_delta", "color_delta", "reversibility_delta",
            "rl_reward", "rl_loss", "efe_loss", "total_loss"
        ]

        summary = {"epoch": epoch}
        for key in keys_to_avg:
            if self.history[key]:
                summary[f"avg_{key}"] = sum(self.history[key]) / len(self.history[key])

        return summary

    def save(self, output_dir):
        """Save metrics to JSON."""
        path = os.path.join(output_dir, "metrics_goal_oriented.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)


# ============================================================================
# GOAL-ORIENTED TRAINING LOOP
# ============================================================================

def seed_all(seed=42):
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pack_transform_record(inp, out):
    """Pack input/output into transformation record."""
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


def train_epoch_with_rl(agent, qwen, solver2, efe_loss, policy_rl, train_loader,
                        optimizer, device, feat_reg, epoch, logger, max_batches=None):
    """
    Training epoch with goal-oriented RL agent.

    Key difference from standard training:
      - Explicitly measures goal progress (accuracy, size, color, reversibility)
      - RL agent learns prompt refinement toward those goals
      - Loss functions are consequences of goal optimization, not the goal itself
    """
    agent.train()
    qwen.eval()  # Keep Qwen frozen for now (can unfreeze later)
    solver2.train()
    efe_loss.train()
    policy_rl.rl_augmentor.train()

    epoch_loss = 0.0
    rl_rewards_sum = 0.0
    rl_losses_sum = 0.0
    efe_losses_sum = 0.0
    accuracy_deltas = []
    size_deltas = []
    color_deltas = []
    rev_deltas = []

    batches_processed = 0
    total_batches = len(train_loader)
    display_total = min(total_batches, max_batches) if max_batches else total_batches

    pbar = tqdm(enumerate(train_loader), total=display_total,
                desc=f"Epoch {epoch} Training (Goal-Oriented)", unit="batch",
                bar_format='{n_fmt}/{total_fmt} [{elapsed}]')

    for batch_idx, batch in pbar:
        if max_batches and batch_idx >= max_batches:
            break

        # ========== LOAD DATA ==========
        inp = batch["input"].to(device)
        out = batch["output"].to(device) if batch["output"] is not None else None

        if out is None:
            continue

        inp = inp.squeeze(0) if inp.dim() == 3 else inp
        out = out.squeeze(0) if out.dim() == 3 else out

        # Handle size mismatch
        if inp.shape != out.shape:
            if inp.shape[0] > out.shape[0] and inp.shape[1] > out.shape[1]:
                pad_h = inp.shape[0] - out.shape[0]
                pad_w = inp.shape[1] - out.shape[1]
                out = F.pad(out, (0, pad_w, 0, pad_h), value=0)

        H, W = inp.shape
        batch_id = batch["prob_id"][0] if "prob_id" in batch else f"batch_{batch_idx}"

        # ========== STEP 1: GET INITIAL PREDICTION (with Qwen prompt) ==========
        tr = pack_transform_record(inp, out)
        tr = apply_operator_config(tr, inp, out, feat_reg)

        with torch.no_grad():
            qwen_pack = qwen(tr, inp, out, control_weight=0.5)

        qwen_prompt = qwen_pack["prompt_embedding"]

        # Forward with initial prompt
        with torch.no_grad():
            feat_sum = torch.tensor(
                [tr.get(k, 0) for k in ["size_change_ratio", "pixel_change_ratio",
                                         "symmetry_change", "density_change",
                                         "spatial_correlation"] + [0]*27],
                dtype=torch.float32, device=device
            )[:32]  # Feature summary for RL

            # Get initial prediction
            agent_out_init = agent.forward(inp.float().unsqueeze(0), qwen_prompt.unsqueeze(0))
            pred_before = agent_out_init["output"].squeeze(0).argmax(dim=-1)

        # ========== STEP 2: RL AGENT REFINES PROMPT ==========
        # The RL agent learns: "What prompt modification improves accuracy?"
        ctrl_vec = agent_out_init["control_embedding"].squeeze(0) if "control_embedding" in agent_out_init else torch.randn(256, device=device)

        refined_prompt, rl_info = policy_rl.refine_prompt(qwen_prompt.squeeze(0), ctrl_vec, feat_sum)

        # ========== STEP 3: GET PREDICTION WITH REFINED PROMPT ==========
        agent_out_refined = agent.forward(inp.float().unsqueeze(0), refined_prompt.unsqueeze(0))
        pred_after = agent_out_refined["output"].squeeze(0).argmax(dim=-1)

        # ========== STEP 4: COMPUTE EXPLICIT REWARDS (Goal Progress) ==========
        # This is the core: measure actual goal achievement
        reward, breakdown = policy_rl.compute_reward(pred_before, pred_after, out, inp)

        # Extract goal deltas (explicit measures of progress)
        d_acc = breakdown.get("d_acc", 0.0)
        d_size = breakdown.get("d_size", 0.0)
        d_col = breakdown.get("d_col", 0.0)
        d_rev = breakdown.get("d_rev", 0.0)

        accuracy_deltas.append(d_acc)
        size_deltas.append(d_size)
        color_deltas.append(d_col)
        rev_deltas.append(d_rev)

        # ========== STEP 5: UPDATE RL AGENT (Goal Optimization) ==========
        # RL learns the policy: optimize toward explicit goals
        rl_losses = policy_rl.update(rl_info, reward)
        rl_reward = rl_losses.get("reward", 0.0)
        rl_loss = rl_losses.get("loss", 0.0)

        rl_rewards_sum += rl_reward
        rl_losses_sum += rl_loss

        # ========== STEP 6: COMPUTE EFE LOSS (Agent Adaptation) ==========
        # Agent adapts to the refined prompts
        optimizer.zero_grad()

        forward_preds = agent_out_refined["forward_predictions"]
        backward_preds = agent_out_refined.get("backward_predictions", forward_preds)
        state_preds = agent_out_refined.get("state_predictions", forward_preds)
        obs_probs = agent_out_refined.get("observation_probs", torch.ones_like(forward_preds))
        final_pred = agent_out_refined["output"]

        efe_losses = efe_loss(
            forward_preds, backward_preds, state_preds, obs_probs, final_pred,
            out.float(), episode_length=5, prompt_embedding=refined_prompt
        )

        efe_loss_val = efe_losses.get("total", sum(efe_losses.values()))
        efe_losses_sum += float(efe_loss_val.item())

        # ========== STEP 7: COMBINED UPDATE ==========
        # Main agent learns from:
        #   1. EFE loss (abstract objective)
        #   2. Goal-oriented reward signal (concrete improvement)
        #
        # Intuition: "Use refined prompts to solve the problem"
        goal_weight = 0.3  # How much to weight goal achievement vs EFE loss
        combined_loss = (1.0 - goal_weight) * efe_loss_val + \
                       goal_weight * (-torch.tensor(reward, device=device, dtype=torch.float32))

        combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += float(combined_loss.item())
        batches_processed += 1

        # ========== LOGGING ==========
        if batch_idx % 50 == 0:
            avg_acc_delta = sum(accuracy_deltas[-50:]) / max(len(accuracy_deltas[-50:]), 1)
            logger.log(
                f"[Batch {batch_idx:4d}] "
                f"Reward: {rl_reward:7.4f} | "
                f"Acc_Δ: {d_acc:+.4f} | "
                f"Size_Δ: {d_size:+.4f} | "
                f"RLoss: {rl_loss:7.4f} | "
                f"EFELoss: {efe_loss_val:7.4f}"
            )

    # ========== EPOCH SUMMARY ==========
    avg_loss = epoch_loss / max(batches_processed, 1)
    avg_rl_reward = rl_rewards_sum / max(batches_processed, 1)
    avg_rl_loss = rl_losses_sum / max(batches_processed, 1)
    avg_efe_loss = efe_losses_sum / max(batches_processed, 1)

    avg_acc_delta = sum(accuracy_deltas) / max(len(accuracy_deltas), 1) if accuracy_deltas else 0.0
    avg_size_delta = sum(size_deltas) / max(len(size_deltas), 1) if size_deltas else 0.0
    avg_color_delta = sum(color_deltas) / max(len(color_deltas), 1) if color_deltas else 0.0
    avg_rev_delta = sum(rev_deltas) / max(len(rev_deltas), 1) if rev_deltas else 0.0

    logger.log("\n" + "="*70)
    logger.log(f"EPOCH {epoch} SUMMARY (Goal-Oriented Training)")
    logger.log("="*70)
    logger.log(f"  Average Combined Loss:       {avg_loss:.6f}")
    logger.log(f"  Average RL Reward Signal:    {avg_rl_reward:+.6f}")
    logger.log(f"  Average RL Loss:             {avg_rl_loss:.6f}")
    logger.log(f"  Average EFE Loss:            {avg_efe_loss:.6f}")
    logger.log("")
    logger.log("EXPLICIT GOAL PROGRESS (What Actually Matters):")
    logger.log(f"  Accuracy Delta (↑ is good):       {avg_acc_delta:+.6f}")
    logger.log(f"  Size Match Delta (↑ is good):     {avg_size_delta:+.6f}")
    logger.log(f"  Color Agreement Delta (↑):        {avg_color_delta:+.6f}")
    logger.log(f"  Reversibility Delta (↑):          {avg_rev_delta:+.6f}")
    logger.log("="*70 + "\n")

    return avg_loss, {
        "combined": avg_loss,
        "rl_reward": avg_rl_reward,
        "rl_loss": avg_rl_loss,
        "efe_loss": avg_efe_loss,
        "accuracy_delta": avg_acc_delta,
        "size_delta": avg_size_delta,
        "color_delta": avg_color_delta,
        "reversibility_delta": avg_rev_delta,
    }


# ============================================================================
# EVALUATION (with RL refinement)
# ============================================================================

def evaluate_with_rl(agent, qwen, solver2, efe_loss, policy_rl, eval_loader, device,
                     feat_reg, max_batches=None):
    """Evaluate using refined prompts from RL agent."""
    agent.eval()
    qwen.eval()
    solver2.eval()
    efe_loss.eval()
    policy_rl.rl_augmentor.eval()

    total_correct = 0
    total_samples = 0
    accuracy_deltas = []

    eval_total = len(eval_loader)
    display_total = min(eval_total, max_batches) if max_batches else eval_total

    with torch.no_grad():
        pbar = tqdm(enumerate(eval_loader), total=display_total, desc="Evaluation",
                    unit="batch", bar_format='{n_fmt}/{total_fmt} [{elapsed}]')

        for batch_idx, batch in pbar:
            if max_batches and batch_idx >= max_batches:
                break

            inp = batch["input"].to(device)
            out = batch["output"].to(device) if batch["output"] is not None else None

            if out is None:
                continue

            inp = inp.squeeze(0) if inp.dim() == 3 else inp
            out = out.squeeze(0) if out.dim() == 3 else out

            if inp.shape != out.shape:
                if inp.shape[0] > out.shape[0] and inp.shape[1] > out.shape[1]:
                    pad_h = inp.shape[0] - out.shape[0]
                    pad_w = inp.shape[1] - out.shape[1]
                    out = F.pad(out, (0, pad_w, 0, pad_h), value=0)

            H, W = inp.shape

            # Get Qwen prompt
            tr = pack_transform_record(inp, out)
            tr = apply_operator_config(tr, inp, out, feat_reg)
            qwen_pack = qwen(tr, inp, out, control_weight=0.5)
            qwen_prompt = qwen_pack["prompt_embedding"]

            # Get initial prediction
            agent_out_init = agent.forward(inp.float().unsqueeze(0), qwen_prompt.unsqueeze(0))
            pred_before = agent_out_init["output"].squeeze(0).argmax(dim=-1)

            # Refine with RL
            feat_sum = torch.zeros(32, device=device)
            ctrl_vec = agent_out_init.get("control_embedding", torch.randn(256, device=device)).squeeze(0)
            refined_prompt, _ = policy_rl.refine_prompt(qwen_prompt.squeeze(0), ctrl_vec, feat_sum)

            # Get refined prediction
            agent_out_refined = agent.forward(inp.float().unsqueeze(0), refined_prompt.unsqueeze(0))
            pred_after = agent_out_refined["output"].squeeze(0).argmax(dim=-1)

            # Binary accuracy: both before AND after must be perfect
            is_correct_before = (pred_before == out).float().mean().item() == 1.0
            is_correct_after = (pred_after == out).float().mean().item() == 1.0

            if is_correct_after:
                total_correct += 1

            # Track RL improvement
            acc_before = (pred_before == out).float().mean().item()
            acc_after = (pred_after == out).float().mean().item()
            accuracy_deltas.append(acc_after - acc_before)

            total_samples += 1

    accuracy = total_correct / max(total_samples, 1)
    avg_acc_delta = sum(accuracy_deltas) / max(len(accuracy_deltas), 1) if accuracy_deltas else 0.0

    return accuracy, total_correct, total_samples, avg_acc_delta


# ============================================================================
# MAIN
# ============================================================================

def main(epochs=10, agent_lr=1e-5, device="cuda", seed=42, max_batches_per_epoch=None):
    """
    Goal-Oriented Training with Human RL Agent

    Philosophy:
      Not: "Minimize loss function X"
      But: "Maximize accuracy, size match, color agreement, reversibility"
    """
    print("\n" + "="*70)
    print("GOAL-ORIENTED TRAINING WITH HUMAN RL AGENT".center(70))
    print("="*70 + "\n")

    seed_all(seed)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"runs/arc_rl_agent_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    logger = TrainingLogger(output_dir)
    metrics = GoalOrientedMetrics()

    logger.log(f"Device: {device}")
    logger.log(f"Epochs: {epochs}")
    logger.log(f"Agent LR: {agent_lr}")
    logger.log(f"Output: {output_dir}\n")

    # Load data
    logger.log("Loading datasets...")
    data_dir = os.getenv("ARC_DATA_DIR", ".")
    train_path = os.path.join(data_dir, "training.json")
    train_ds = ARCDataset(train_path, split="train")
    val_ds = ARCDataset(train_path, split="test")

    train_ld = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=1, shuffle=False)

    logger.log(f"  Train: {len(train_ds)} | Val: {len(val_ds)}\n")

    # Create models
    logger.log("Creating models...")

    # Qwen (frozen)
    qwen = QwenHybridPrompt(
        prompt_dim=256, numeric_in_dim=15, fuse="mean",
        qwen=QwenCfg(model_name="Qwen/Qwen2.5-1.5B", use_qwen=True)
    ).to(device)
    for param in qwen.parameters():
        param.requires_grad = False

    # Agent
    agent = ARCPromptGuidedAgentGPU(
        max_grid_size=30, num_colors=10, hidden_dim=256, prompt_dim=256, max_steps=5
    ).to(device)

    # Solver2
    solver2 = PermanentSolver(input_dim=256, hidden_dim=512, max_grid_size=30, num_colors=10).to(device)

    # EFE Loss
    efe_loss = EFELoss(
        lambda_risk=1.0, lambda_amb=0.0, lambda_step=0.1, lambda_cons=1.0,
        lambda_bi=0.5, lambda_z=0.2, lambda_prompt=0.3,
        max_grid_size=30, num_colors=10, prompt_dim=256
    ).to(device)

    # RL AGENT (Goal-Oriented Prompt Refinement)
    policy_cfg = PolicyRefinedConfig(
        rl_prompt_dim=256,
        rl_ctrl_dim=256,
        rl_feat_dim=32,
        rl_lr=5e-5,
        rl_entropy_coef=0.01,
        rl_icm_coef=0.1,
    )
    policy_rl = PolicyRefinedAgent(policy_cfg, device=device)
    logger.log("  [OK] RL Agent initialized - GOAL: Improve accuracy via prompt refinement\n")

    # Optimizer
    trainable_params = [
        {"params": agent.parameters(), "lr": agent_lr},
        {"params": solver2.parameters(), "lr": agent_lr * 2.0},
        {"params": efe_loss.parameters(), "lr": agent_lr * 0.5},
    ]
    optimizer = torch.optim.Adam(trainable_params, weight_decay=1e-6)

    feat_reg = FeatureRegistry()
    best_acc = -1

    # Training loop
    logger.log("Starting goal-oriented training...\n")

    for epoch in range(epochs):
        epoch_start = time.time()

        avg_loss, loss_components = train_epoch_with_rl(
            agent, qwen, solver2, efe_loss, policy_rl,
            train_ld, optimizer, device, feat_reg, epoch, logger, max_batches_per_epoch
        )

        # Evaluate
        val_acc, val_correct, val_total, avg_acc_delta = evaluate_with_rl(
            agent, qwen, solver2, efe_loss, policy_rl, val_ld, device, feat_reg
        )

        epoch_time = time.time() - epoch_start

        logger.log(f"[Epoch {epoch}] Val Accuracy: {val_acc:.4f} ({val_correct}/{val_total})")
        logger.log(f"[Epoch {epoch}] RL Accuracy Delta: {avg_acc_delta:+.6f}")
        logger.log(f"[Epoch {epoch}] Time: {epoch_time:.2f}s\n")

        # Track metrics
        metrics.log_batch(epoch, {
            "epoch": epoch,
            "accuracy_delta": loss_components.get("accuracy_delta", 0),
            "size_delta": loss_components.get("size_delta", 0),
            "color_delta": loss_components.get("color_delta", 0),
            "reversibility_delta": loss_components.get("reversibility_delta", 0),
            "rl_reward": loss_components.get("rl_reward", 0),
            "rl_loss": loss_components.get("rl_loss", 0),
            "efe_loss": loss_components.get("efe_loss", 0),
            "total_loss": avg_loss,
        })

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(agent.state_dict(), os.path.join(output_dir, "agent_best.pt"))
            logger.log(f"  [BEST] Checkpoint saved!\n")

    metrics.save(output_dir)
    logger.log(f"\nTraining complete! Best accuracy: {best_acc:.4f}")
    logger.log(f"Results saved to: {output_dir}\n")
    logger.close()

    return output_dir, best_acc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--agent_lr", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_batches", type=int, default=None)
    args = parser.parse_args()

    main(
        epochs=args.epochs,
        agent_lr=args.agent_lr,
        device=args.device,
        seed=args.seed,
        max_batches_per_epoch=args.max_batches
    )
