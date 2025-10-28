# -*- coding: utf-8 -*-
"""
Test Sequence for ARC Challenge Solver
Complete evaluation and analysis pipeline for trained models
"""

import torch
import os
import json
import time
from datetime import datetime
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple, Optional

from dataset_arc import ARCDataset
from qwen_hybrid_prompt import QwenHybridPrompt, QwenCfg
from loss_function import ARCPromptGuidedAgent
from tta import TestTimeAdaptationSystem
from feature_registry import FeatureRegistry, apply_operator_config
from feature_extraction import extract_transformation_features, classify_transformation_type


class TestConfig:
    """Test configuration"""
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_dir = f"runs/arc_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.batch_size = 1
        self.use_tta = True
        self.num_tta_steps = 5


class TestLogger:
    """Test result logger"""
    def __init__(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.start_time = time.time()

        self.result_log = open(os.path.join(output_dir, "test_results.jsonl"), "w", encoding="utf-8")
        self.tta_log = open(os.path.join(output_dir, "tta_results.jsonl"), "w", encoding="utf-8")
        self.summary_file = os.path.join(output_dir, "test_summary.json")

        self.results = {
            "test_samples": [],
            "tta_samples": [],
            "statistics": {}
        }

    def log_test_sample(self, prob_id: str, accuracy: float, pred_shape: Tuple, target_shape: Tuple):
        """Log test sample result"""
        record = {
            "timestamp": time.time(),
            "prob_id": prob_id,
            "accuracy": accuracy,
            "pred_shape": pred_shape,
            "target_shape": target_shape
        }
        self.result_log.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.result_log.flush()
        self.results["test_samples"].append(record)

    def log_tta_sample(self, prob_id: str, tta_results: Dict[str, Any]):
        """Log TTA sample result"""
        record = {
            "timestamp": time.time(),
            "prob_id": prob_id,
            "selected_solver": tta_results.get("selected_solver"),
            "final_surprise": tta_results.get("final_surprise"),
            "memory_size": tta_results.get("memory_size"),
            "adaptation_steps": len(tta_results.get("adaptation_losses", [])),
            "avg_adaptation_loss": sum(tta_results.get("adaptation_losses", [0])) / max(len(tta_results.get("adaptation_losses", [1])), 1)
        }
        self.tta_log.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.tta_log.flush()
        self.results["tta_samples"].append(record)

    def compute_statistics(self):
        """Compute test statistics"""
        if not self.results["test_samples"]:
            return

        accs = [s["accuracy"] for s in self.results["test_samples"]]

        stats = {
            "total_samples": len(accs),
            "mean_accuracy": sum(accs) / len(accs),
            "max_accuracy": max(accs),
            "min_accuracy": min(accs),
            "std_accuracy": (sum((x - sum(accs)/len(accs))**2 for x in accs) / len(accs)) ** 0.5 if len(accs) > 1 else 0,
            "samples_perfect": sum(1 for x in accs if x == 1.0),
            "samples_partial": sum(1 for x in accs if 0 < x < 1.0),
            "samples_zero": sum(1 for x in accs if x == 0.0),
        }

        if self.results["tta_samples"]:
            surprises = [s["final_surprise"] for s in self.results["tta_samples"]]
            stats["tta_mean_surprise"] = sum(surprises) / len(surprises)
            stats["tta_max_surprise"] = max(surprises)
            stats["tta_min_surprise"] = min(surprises)

        self.results["statistics"] = stats
        return stats

    def print_summary(self):
        """Print test summary"""
        elapsed = time.time() - self.start_time

        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Checkpoint: {self.results.get('checkpoint', 'N/A')}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Total Time: {elapsed:.2f} seconds")

        stats = self.results.get("statistics", {})
        if stats:
            print(f"\nAccuracy Statistics:")
            print(f"  Total Samples: {stats.get('total_samples', 0)}")
            print(f"  Mean Accuracy: {stats.get('mean_accuracy', 0):.4f}")
            print(f"  Max Accuracy: {stats.get('max_accuracy', 0):.4f}")
            print(f"  Min Accuracy: {stats.get('min_accuracy', 0):.4f}")
            print(f"  Std Deviation: {stats.get('std_accuracy', 0):.6f}")
            print(f"\nResult Distribution:")
            print(f"  Perfect (1.0): {stats.get('samples_perfect', 0)}")
            print(f"  Partial (0<acc<1): {stats.get('samples_partial', 0)}")
            print(f"  Zero (0.0): {stats.get('samples_zero', 0)}")

            if "tta_mean_surprise" in stats:
                print(f"\nTTA Statistics:")
                print(f"  Mean Surprise: {stats.get('tta_mean_surprise', 0):.4f}")
                print(f"  Max Surprise: {stats.get('tta_max_surprise', 0):.4f}")
                print(f"  Min Surprise: {stats.get('tta_min_surprise', 0):.4f}")

        print("="*60 + "\n")

    def save_summary(self, checkpoint_path: str):
        """Save summary to JSON"""
        self.results["checkpoint"] = checkpoint_path
        self.results["timestamp"] = datetime.now().isoformat()

        with open(self.summary_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

    def close(self):
        """Close log files"""
        self.result_log.close()
        self.tta_log.close()


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


def make_qwen(device: str) -> QwenHybridPrompt:
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
    ).to(device)
    return qwen


def test_sequence(
    checkpoint_path: str,
    config: Optional[TestConfig] = None,
    split: str = "test"
) -> str:
    """
    Complete test sequence

    Args:
        checkpoint_path: Path to trained model checkpoint
        config: Test configuration (uses defaults if None)
        split: "test" or "train" split to evaluate

    Returns:
        Path to output directory
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if config is None:
        config = TestConfig(checkpoint_path)

    print("\n" + "="*60)
    print("ARC TEST SEQUENCE STARTED")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Split: {split}")
    print(f"Device: {config.device}")
    print(f"Use TTA: {config.use_tta}")
    print(f"Output Directory: {config.output_dir}")
    print("="*60 + "\n")

    # Create logger
    logger = TestLogger(config.output_dir)

    # Load dataset
    print(f"Loading {split} dataset...")
    test_ds = ARCDataset("training.json", split=split)
    test_ld = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)
    print(f"  {split.capitalize()}: {len(test_ds)} samples\n")

    # Create agent
    print("Creating agent...")
    agent = ARCPromptGuidedAgent(
        max_grid_size=30,
        num_colors=10,
        hidden_dim=256,
        prompt_dim=256
    ).to(config.device)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    agent.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
    agent.eval()

    # Create Qwen and TTA system
    print("Creating Qwen and TTA system...")
    qwen = make_qwen(config.device)
    qwen.eval()

    tta_system = None
    if config.use_tta:
        tta_system = TestTimeAdaptationSystem(
            base_agent=agent,
            memory_size=500,
            num_solvers=3,
            adaptation_steps=config.num_tta_steps,
            adaptation_lr=1e-3
        )

    feat_reg = FeatureRegistry()

    print("Starting evaluation...\n")

    # Test loop
    total_acc = 0.0
    num_samples = 0
    num_tta_tested = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_ld):
            prob_id = batch["prob_id"]
            inp = batch["input"].to(config.device)
            out = batch["output"].to(config.device) if batch["output"] is not None else None

            if out is None:
                continue

            # Standard prediction
            tr = pack_transform_record(inp, out)
            tr = apply_operator_config(tr, inp, out, feat_reg)
            pack = qwen(tr, inp, out, control_weight=0.5)

            preds, _ = agent.forward_planning(
                inp,
                pack["hybrid_embedding"].squeeze(0),
                num_steps=5
            )

            final = preds[-1].argmax(dim=-1)
            acc = (final == out).float().mean().item()

            logger.log_test_sample(prob_id, acc, tuple(final.shape), tuple(out.shape))
            total_acc += acc
            num_samples += 1

            # Optional: TTA evaluation on every Nth sample
            if config.use_tta and tta_system is not None and batch_idx % 5 == 0:
                try:
                    prompt_emb = pack["hybrid_embedding"].squeeze(0)
                    tta_results = tta_system.test_time_adapt(
                        inp, out, pack.get("prompt_text", ""), prompt_emb
                    )
                    logger.log_tta_sample(prob_id, tta_results)
                    num_tta_tested += 1
                except Exception as e:
                    print(f"[Warning] TTA failed for {prob_id}: {str(e)[:50]}")

            # Progress
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} samples...")

    print()

    # Compute and log statistics
    logger.compute_statistics()
    logger.print_summary()
    logger.save_summary(checkpoint_path)
    logger.close()

    return logger.output_dir


def batch_test_checkpoints(checkpoint_dir: str, split: str = "test"):
    """
    Test all checkpoints in a directory

    Args:
        checkpoint_dir: Directory containing checkpoint files
        split: Data split to evaluate on
    """
    print(f"\nTesting all checkpoints in {checkpoint_dir}")
    print("="*60)

    checkpoints = []
    for fname in os.listdir(checkpoint_dir):
        if fname.endswith(".pt"):
            checkpoints.append(os.path.join(checkpoint_dir, fname))

    checkpoints.sort()

    results = {}

    for ckpt_path in checkpoints:
        print(f"\nTesting: {os.path.basename(ckpt_path)}")
        try:
            output_dir = test_sequence(ckpt_path, split=split)
            results[os.path.basename(ckpt_path)] = output_dir
            print(f"  Results saved to: {output_dir}")
        except Exception as e:
            print(f"  Failed: {str(e)[:100]}")

    print("\n" + "="*60)
    print("BATCH TEST COMPLETED")
    print("="*60)
    for ckpt, output_dir in results.items():
        print(f"{ckpt}: {output_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        checkpoint = sys.argv[1]
        output_dir = test_sequence(checkpoint)
        print(f"\nTest completed! Results saved to: {output_dir}")
    else:
        print("Usage: python test_sequence.py <checkpoint_path>")
        print("Example: python test_sequence.py runs/arc_train_20250101_120000/agent_best.pt")
