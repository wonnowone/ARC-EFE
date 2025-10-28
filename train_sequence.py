# -*- coding: utf-8 -*-
"""
Training Sequence for ARC Challenge Solver
Complete training pipeline with checkpointing, logging, and monitoring
"""

import torch
import os
import time
import json
from datetime import datetime
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

from dataset_arc import ARCDataset
from qwen_hybrid_prompt import QwenHybridPrompt, QwenCfg
from revthink_orchestrator import RevThinkOrchestrator, RevThinkCfg
from loss_function import ARCPromptGuidedAgent
from tta import TestTimeAdaptationSystem
from feature_registry import FeatureRegistry, apply_operator_config
from feature_extraction import extract_transformation_features, classify_transformation_type

# Configuration
TTA_EVAL_INTERVAL = 50  # Evaluate TTA every N batches (0 to disable)
CHECKPOINT_INTERVAL = 100  # Save checkpoint every N batches
GRAD_CLIP_NORM = 1.0
LEARNING_RATE = 1e-3
EPOCHS = 5
BATCH_SIZE = 1


class TrainingConfig:
    """Training configuration"""
    def __init__(self):
        self.epochs = EPOCHS
        self.batch_size = BATCH_SIZE
        self.learning_rate = LEARNING_RATE
        self.grad_clip_norm = GRAD_CLIP_NORM
        self.tta_eval_interval = TTA_EVAL_INTERVAL
        self.checkpoint_interval = CHECKPOINT_INTERVAL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_dir = f"runs/arc_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


class TrainingLogger:
    """Comprehensive training logger"""
    def __init__(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.start_time = time.time()

        # Log files
        self.train_log = open(os.path.join(output_dir, "train_log.jsonl"), "w", encoding="utf-8")
        self.eval_log = open(os.path.join(output_dir, "eval_log.jsonl"), "w", encoding="utf-8")
        self.tta_log = open(os.path.join(output_dir, "tta_log.jsonl"), "w", encoding="utf-8")
        self.checkpoint_log = open(os.path.join(output_dir, "checkpoints.jsonl"), "w", encoding="utf-8")
        self.metric_log = open(os.path.join(output_dir, "metrics.jsonl"), "w", encoding="utf-8")

        # Statistics
        self.train_losses = []
        self.eval_accs = []
        self.batch_count = 0
        self.epoch_count = 0

    def log_train(self, epoch: int, batch_idx: int, prob_id: str, losses: Dict[str, Any]):
        """Log training batch"""
        record = {
            "timestamp": time.time(),
            "epoch": epoch,
            "batch": batch_idx,
            "prob_id": prob_id,
        }
        for k, v in losses.items():
            if k == "prompt_text":
                continue
            record[k] = float(v.item() if hasattr(v, "item") else v)

        self.train_log.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.train_log.flush()

        if "total" in losses:
            self.train_losses.append(float(losses["total"].item()))

    def log_eval(self, epoch: int, prob_id: str, acc: float):
        """Log evaluation result"""
        record = {
            "timestamp": time.time(),
            "epoch": epoch,
            "prob_id": prob_id,
            "accuracy": acc
        }
        self.eval_log.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.eval_log.flush()
        self.eval_accs.append(acc)

    def log_tta(self, epoch: int, batch_idx: int, tta_results: Dict[str, Any]):
        """Log TTA evaluation"""
        record = {
            "timestamp": time.time(),
            "epoch": epoch,
            "batch": batch_idx,
            "selected_solver": tta_results.get("selected_solver"),
            "final_surprise": tta_results.get("final_surprise"),
            "memory_size": tta_results.get("memory_size"),
            "avg_adaptation_loss": sum(tta_results.get("adaptation_losses", [0])) / max(len(tta_results.get("adaptation_losses", [1])), 1)
        }
        self.tta_log.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.tta_log.flush()

    def log_checkpoint(self, epoch: int, batch_idx: int, path: str, is_best: bool = False):
        """Log checkpoint"""
        record = {
            "timestamp": time.time(),
            "epoch": epoch,
            "batch": batch_idx,
            "path": path,
            "is_best": is_best,
            "elapsed_seconds": time.time() - self.start_time
        }
        self.checkpoint_log.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.checkpoint_log.flush()

    def log_metric(self, name: str, epoch: int, value: float):
        """Log metric"""
        record = {
            "timestamp": time.time(),
            "name": name,
            "epoch": epoch,
            "value": value
        }
        self.metric_log.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.metric_log.flush()

    def print_summary(self):
        """Print training summary"""
        elapsed = time.time() - self.start_time
        hours = elapsed / 3600

        avg_train_loss = sum(self.train_losses) / len(self.train_losses) if self.train_losses else 0
        avg_eval_acc = sum(self.eval_accs) / len(self.eval_accs) if self.eval_accs else 0
        best_eval_acc = max(self.eval_accs) if self.eval_accs else 0

        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Output Directory: {self.output_dir}")
        print(f"Total Time: {hours:.2f} hours ({elapsed:.1f} seconds)")
        print(f"Batches Processed: {self.batch_count}")
        print(f"Epochs Completed: {self.epoch_count}")
        print(f"Average Train Loss: {avg_train_loss:.6f}")
        print(f"Average Eval Accuracy: {avg_eval_acc:.4f}")
        print(f"Best Eval Accuracy: {best_eval_acc:.4f}")
        print("="*60 + "\n")

    def close(self):
        """Close all log files"""
        self.train_log.close()
        self.eval_log.close()
        self.tta_log.close()
        self.checkpoint_log.close()
        self.metric_log.close()


def seed_all(seed: int = 42):
    """Set random seeds"""
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def make_agent(config: TrainingConfig) -> ARCPromptGuidedAgent:
    """Create agent"""
    agent = ARCPromptGuidedAgent(
        max_grid_size=30,
        num_colors=10,
        hidden_dim=256,
        prompt_dim=256
    ).to(config.device)
    return agent


def make_qwen(config: TrainingConfig) -> QwenHybridPrompt:
    """Create Qwen prompt generator"""
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
    ).to(config.device)
    return qwen


def pack_transform_record(inp: torch.Tensor, out: Optional[torch.Tensor]) -> Dict[str, Any]:
    """Pack input/output into transformation record"""
    # Squeeze batch dimension if present (input comes from batch loader)
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


def train_one_epoch(
    agent: ARCPromptGuidedAgent,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    config: TrainingConfig,
    qwen: QwenHybridPrompt,
    revthink: RevThinkOrchestrator,
    logger: TrainingLogger,
    feat_reg: FeatureRegistry,
    tta_system: Optional[TestTimeAdaptationSystem] = None,
    epoch: int = 0
) -> float:
    """Train one epoch"""
    agent.train()
    qwen.train()

    original_lambda_prompt = agent.efe_loss.lambda_prompt
    epoch_losses = []

    for batch_idx, batch in enumerate(loader):
        inp = batch["input"].to(config.device)
        out = batch["output"].to(config.device) if batch["output"] is not None else None
        prob_id = batch["prob_id"]

        # Feature extraction
        tr = pack_transform_record(inp, out)
        tr = apply_operator_config(tr, inp, out, feat_reg)

        # Hybrid prompt generation
        with torch.no_grad():
            pack = qwen(tr, inp, out, control_weight=0.5)
        prompt_text = pack["prompt_text"]
        prompt_emb = pack["hybrid_embedding"].squeeze(0)

        # Train episode
        optim.zero_grad()
        losses = agent.train_episode(
            initial_state=inp,
            target_state=out,
            prompt_text=prompt_text,
            prompt_embedding=prompt_emb,
            num_steps=3
        )

        # RevThink intervention
        losses_num = {k: (v.item() if hasattr(v, "item") else v)
                     for k, v in losses.items() if k != "prompt_text"}
        issue = revthink.maybe_revise(tr, inp, out, losses_num)

        if issue["apply"]:
            g = issue["gate"]
            new_emb = (1 - g) * prompt_emb + g * issue["hybrid_embedding"].squeeze(0)
            agent.efe_loss.lambda_prompt = original_lambda_prompt * (1 + revthink.cfg.gamma * g)
            losses = agent.train_episode(inp, out, issue["prompt_text"], new_emb, num_steps=3)
        else:
            agent.efe_loss.lambda_prompt = original_lambda_prompt

        # Backward pass
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=config.grad_clip_norm)
        optim.step()

        # Logging
        logger.log_train(epoch, batch_idx, prob_id, losses)
        epoch_losses.append(float(losses["total"].item()))
        logger.batch_count += 1

        # Periodic TTA evaluation
        if config.tta_eval_interval > 0 and tta_system is not None and batch_idx % config.tta_eval_interval == 0:
            try:
                tta_results = tta_system.test_time_adapt(inp, out, prompt_text, prompt_emb)
                logger.log_tta(epoch, batch_idx, tta_results)
            except Exception as e:
                print(f"[Warning] TTA eval failed at epoch {epoch} batch {batch_idx}: {str(e)[:50]}")

        # Periodic checkpoint
        if config.checkpoint_interval > 0 and batch_idx % config.checkpoint_interval == 0 and batch_idx > 0:
            ckpt_path = os.path.join(logger.output_dir, f"agent_epoch{epoch}_batch{batch_idx}.pt")
            torch.save(agent.state_dict(), ckpt_path)
            logger.log_checkpoint(epoch, batch_idx, ckpt_path, is_best=False)
            print(f"  [Checkpoint] Saved to {ckpt_path}")

    avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
    return avg_epoch_loss


def evaluate(
    agent: ARCPromptGuidedAgent,
    loader: DataLoader,
    config: TrainingConfig,
    qwen: QwenHybridPrompt,
    logger: TrainingLogger,
    feat_reg: FeatureRegistry,
    epoch: int = 0
) -> float:
    """Evaluate on test set"""
    agent.eval()
    qwen.eval()

    acc_sum, n = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            inp = batch["input"].to(config.device)
            out = batch["output"].to(config.device) if batch["output"] is not None else None
            prob_id = batch["prob_id"]

            tr = pack_transform_record(inp, out)
            tr = apply_operator_config(tr, inp, out, feat_reg)
            pack = qwen(tr, inp, out, control_weight=0.5)

            preds, _ = agent.forward_planning(
                inp,
                pack["hybrid_embedding"].squeeze(0),
                num_steps=5
            )

            final = preds[-1].argmax(dim=-1)

            if out is not None:
                acc = (final == out).float().mean().item()
                acc_sum += acc
                n += 1
                logger.log_eval(epoch, prob_id, acc)

    avg_acc = acc_sum / max(n, 1)
    return avg_acc


def train_sequence(
    config: Optional[TrainingConfig] = None,
    checkpoint_path: Optional[str] = None
) -> str:
    """
    Complete training sequence

    Args:
        config: Training configuration (uses defaults if None)
        checkpoint_path: Optional checkpoint to resume from

    Returns:
        Path to output directory
    """
    if config is None:
        config = TrainingConfig()

    print("\n" + "="*60)
    print("ARC TRAINING SEQUENCE STARTED")
    print("="*60)
    print(f"Device: {config.device}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Output Directory: {config.output_dir}")
    print("="*60 + "\n")

    seed_all()

    # Create logger
    logger = TrainingLogger(config.output_dir)

    # Load datasets
    print("Loading datasets...")
    train_ds = ARCDataset("training.json", split="train")
    test_ds = ARCDataset("training.json", split="test")
    train_ld = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    test_ld = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Test: {len(test_ds)} samples\n")

    # Create components
    print("Creating model components...")
    agent = make_agent(config)
    qwen = make_qwen(config)
    revthink = RevThinkOrchestrator(qwen=qwen, cfg=RevThinkCfg())
    feat_reg = FeatureRegistry()
    tta_system = TestTimeAdaptationSystem(
        base_agent=agent,
        memory_size=500,
        num_solvers=3,
        adaptation_steps=3,
        adaptation_lr=1e-3
    )

    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        agent.load_state_dict(torch.load(checkpoint_path, map_location=config.device))

    # Optimizer
    optim = torch.optim.Adam(agent.parameters(), lr=config.learning_rate)

    # Training loop
    best_acc = -1
    best_ckpt = os.path.join(logger.output_dir, "agent_best.pt")

    print("Starting training...\n")

    for epoch in range(config.epochs):
        print(f"[Epoch {epoch}/{config.epochs - 1}]")

        # Train
        avg_train_loss = train_one_epoch(
            agent, train_ld, optim, config, qwen, revthink, logger, feat_reg, tta_system, epoch
        )
        print(f"  Train Loss: {avg_train_loss:.6f}")

        # Evaluate
        avg_test_acc = evaluate(agent, test_ld, config, qwen, logger, feat_reg, epoch)
        print(f"  Test Acc: {avg_test_acc:.4f}")

        # Log metrics
        logger.log_metric("epoch_train_loss", epoch, avg_train_loss)
        logger.log_metric("epoch_test_acc", epoch, avg_test_acc)

        # Save best checkpoint
        if avg_test_acc > best_acc:
            best_acc = avg_test_acc
            torch.save(agent.state_dict(), best_ckpt)
            logger.log_checkpoint(epoch, 0, best_ckpt, is_best=True)
            print(f"  Best accuracy! Saved to {best_ckpt}")

        print()
        logger.epoch_count += 1

    # Summary
    logger.print_summary()
    logger.close()

    return logger.output_dir


if __name__ == "__main__":
    output_dir = train_sequence()
    print(f"Training completed! Results saved to: {output_dir}")
