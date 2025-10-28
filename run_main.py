# -*- coding: utf-8 -*-
"""
Main Orchestrator for ARC Training Pipeline
Complete end-to-end execution with training, testing, and reporting
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

from train_sequence import train_sequence, TrainingConfig
from test_sequence import test_sequence, TestConfig, batch_test_checkpoints


class PipelineConfig:
    """Complete pipeline configuration"""
    def __init__(self):
        self.base_output = "runs"
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_name = f"arc_full_run_{self.timestamp}"
        self.run_dir = os.path.join(self.base_output, self.run_name)

        # Training config
        self.train_epochs = 5
        self.train_batch_size = 1
        self.train_lr = 1e-3
        self.train_grad_clip = 1.0
        self.train_tta_eval_interval = 50
        self.train_checkpoint_interval = 100

        # Testing config
        self.test_on_train_split = False
        self.test_on_test_split = True
        self.use_tta = True
        self.tta_steps = 5

        # Reporting
        self.generate_report = True

    def save(self):
        """Save configuration to file"""
        os.makedirs(self.run_dir, exist_ok=True)
        config_file = os.path.join(self.run_dir, "pipeline_config.json")
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(vars(self), f, indent=2, ensure_ascii=False)
        return config_file


def create_training_config(pipeline_cfg: PipelineConfig) -> TrainingConfig:
    """Create training config from pipeline config"""
    cfg = TrainingConfig()
    cfg.epochs = pipeline_cfg.train_epochs
    cfg.batch_size = pipeline_cfg.train_batch_size
    cfg.learning_rate = pipeline_cfg.train_lr
    cfg.grad_clip_norm = pipeline_cfg.train_grad_clip
    cfg.tta_eval_interval = pipeline_cfg.train_tta_eval_interval
    cfg.checkpoint_interval = pipeline_cfg.train_checkpoint_interval
    cfg.output_dir = os.path.join(pipeline_cfg.run_dir, "training")
    return cfg


def create_testing_config(pipeline_cfg: PipelineConfig, checkpoint_path: str) -> TestConfig:
    """Create testing config from pipeline config"""
    cfg = TestConfig(checkpoint_path)
    cfg.use_tta = pipeline_cfg.use_tta
    cfg.num_tta_steps = pipeline_cfg.tta_steps
    cfg.output_dir = os.path.join(pipeline_cfg.run_dir, "testing")
    return cfg


def print_header(title: str):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {title}".center(70))
    print("="*70 + "\n")


def print_section(title: str):
    """Print formatted section"""
    print("\n" + "-"*70)
    print(f"  {title}")
    print("-"*70 + "\n")


def generate_final_report(pipeline_cfg: PipelineConfig, train_dir: str, test_dirs: dict):
    """Generate final pipeline report"""
    report_path = os.path.join(pipeline_cfg.run_dir, "FINAL_REPORT.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# ARC Training Pipeline Final Report\n\n")

        f.write(f"**Generated:** {datetime.now().isoformat()}\n")
        f.write(f"**Run Name:** {pipeline_cfg.run_name}\n")
        f.write(f"**Base Directory:** {pipeline_cfg.run_dir}\n\n")

        # Configuration
        f.write("## Configuration\n\n")
        f.write("### Training\n")
        f.write(f"- Epochs: {pipeline_cfg.train_epochs}\n")
        f.write(f"- Batch Size: {pipeline_cfg.train_batch_size}\n")
        f.write(f"- Learning Rate: {pipeline_cfg.train_lr}\n")
        f.write(f"- Gradient Clip: {pipeline_cfg.train_grad_clip}\n")
        f.write(f"- TTA Eval Interval: {pipeline_cfg.train_tta_eval_interval} batches\n")
        f.write(f"- Checkpoint Interval: {pipeline_cfg.train_checkpoint_interval} batches\n\n")

        f.write("### Testing\n")
        f.write(f"- Use TTA: {pipeline_cfg.use_tta}\n")
        f.write(f"- TTA Steps: {pipeline_cfg.tta_steps}\n")
        f.write(f"- Test on Train Split: {pipeline_cfg.test_on_train_split}\n")
        f.write(f"- Test on Test Split: {pipeline_cfg.test_on_test_split}\n\n")

        # Results
        f.write("## Results\n\n")

        f.write("### Training\n")
        f.write(f"- Output Directory: `{train_dir}`\n")
        f.write(f"- Logs: `train_log.jsonl`, `eval_log.jsonl`, `metrics.jsonl`\n")
        f.write(f"- Checkpoints: `agent_*.pt`\n\n")

        f.write("### Testing\n")
        for split, test_dir in test_dirs.items():
            f.write(f"- **{split.upper()}**: `{test_dir}`\n")
            summary_file = os.path.join(test_dir, "test_summary.json")
            if os.path.exists(summary_file):
                with open(summary_file, "r", encoding="utf-8") as sf:
                    summary = json.load(sf)
                    stats = summary.get("statistics", {})
                    f.write(f"  - Samples: {stats.get('total_samples', 'N/A')}\n")
                    f.write(f"  - Mean Accuracy: {stats.get('mean_accuracy', 0):.4f}\n")
                    f.write(f"  - Max Accuracy: {stats.get('max_accuracy', 0):.4f}\n")
                    f.write(f"  - Min Accuracy: {stats.get('min_accuracy', 0):.4f}\n")

        f.write("\n## Files\n\n")
        f.write("```\n")
        f.write(f"{pipeline_cfg.run_dir}/\n")
        f.write("├── pipeline_config.json\n")
        f.write("├── FINAL_REPORT.md\n")
        f.write("├── training/\n")
        f.write("│   ├── train_log.jsonl\n")
        f.write("│   ├── eval_log.jsonl\n")
        f.write("│   ├── tta_log.jsonl\n")
        f.write("│   ├── metrics.jsonl\n")
        f.write("│   ├── checkpoints.jsonl\n")
        f.write("│   ├── agent_best.pt\n")
        f.write("│   └── agent_epoch*_batch*.pt\n")
        if test_dirs:
            f.write("└── testing/\n")
            f.write("    ├── test/\n")
            f.write("    │   ├── test_results.jsonl\n")
            f.write("    │   ├── tta_results.jsonl\n")
            f.write("    │   └── test_summary.json\n")
            if pipeline_cfg.test_on_train_split:
                f.write("    └── train/\n")
                f.write("        ├── test_results.jsonl\n")
                f.write("        ├── tta_results.jsonl\n")
                f.write("        └── test_summary.json\n")
        f.write("```\n\n")

        f.write("## Next Steps\n\n")
        f.write("1. Review training logs: `training/train_log.jsonl`\n")
        f.write("2. Check test results: `testing/test/test_summary.json`\n")
        f.write("3. Load best model: `torch.load('training/agent_best.pt')`\n")
        f.write("4. Analyze TTA performance: `testing/test/tta_results.jsonl`\n\n")

        f.write("---\n\n")
        f.write(f"*Report generated at {datetime.now().isoformat()}*\n")

    return report_path


def main(args: argparse.Namespace):
    """Run complete pipeline"""

    print_header("ARC TRAINING PIPELINE - FULL RUN")

    # Create pipeline config
    pipeline_cfg = PipelineConfig()

    if args.epochs:
        pipeline_cfg.train_epochs = args.epochs
    if args.learning_rate:
        pipeline_cfg.train_lr = args.learning_rate
    if args.skip_train:
        pipeline_cfg.train_epochs = 0
    if args.skip_test:
        pipeline_cfg.test_on_test_split = False

    print(f"Run Name: {pipeline_cfg.run_name}")
    print(f"Output Directory: {pipeline_cfg.run_dir}")
    print()

    # Save configuration
    config_file = pipeline_cfg.save()
    print(f"Configuration saved to: {config_file}\n")

    train_dir = None
    test_dirs = {}

    # TRAINING PHASE
    if pipeline_cfg.train_epochs > 0 and not args.skip_train:
        print_header("PHASE 1: TRAINING")

        try:
            train_cfg = create_training_config(pipeline_cfg)
            train_dir = train_sequence(train_cfg)
            print(f"\nTraining completed!")
            print(f"Results saved to: {train_dir}")

            # Find best checkpoint
            best_ckpt = os.path.join(train_dir, "agent_best.pt")
            if os.path.exists(best_ckpt):
                print(f"Best checkpoint: {best_ckpt}")
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            if args.continue_on_error:
                print("Continuing to testing phase...")
            else:
                print("Stopping pipeline.")
                return

    # TESTING PHASE
    if not args.skip_test and train_dir:
        print_header("PHASE 2: TESTING")

        best_ckpt = os.path.join(train_dir, "agent_best.pt")

        if not os.path.exists(best_ckpt):
            print(f"Error: Best checkpoint not found at {best_ckpt}")
            return

        # Test on test split
        if pipeline_cfg.test_on_test_split:
            print_section("Testing on TEST split")
            try:
                test_cfg = TestConfig(best_ckpt)
                test_cfg.output_dir = os.path.join(pipeline_cfg.run_dir, "testing", "test")
                test_cfg.use_tta = pipeline_cfg.use_tta
                test_cfg.num_tta_steps = pipeline_cfg.tta_steps
                test_dir = test_sequence(best_ckpt, test_cfg, split="test")
                test_dirs["test"] = test_dir
                print(f"Test results saved to: {test_dir}")
            except Exception as e:
                print(f"Error during test evaluation: {str(e)}")
                if not args.continue_on_error:
                    return

        # Test on train split (optional)
        if pipeline_cfg.test_on_train_split:
            print_section("Testing on TRAIN split")
            try:
                test_cfg = TestConfig(best_ckpt)
                test_cfg.output_dir = os.path.join(pipeline_cfg.run_dir, "testing", "train")
                test_cfg.use_tta = pipeline_cfg.use_tta
                test_cfg.num_tta_steps = pipeline_cfg.tta_steps
                test_dir = test_sequence(best_ckpt, test_cfg, split="train")
                test_dirs["train"] = test_dir
                print(f"Train results saved to: {test_dir}")
            except Exception as e:
                print(f"Error during train evaluation: {str(e)}")
                if not args.continue_on_error:
                    return

    # REPORTING PHASE
    if pipeline_cfg.generate_report and train_dir:
        print_header("PHASE 3: REPORTING")

        try:
            report_path = generate_final_report(pipeline_cfg, train_dir, test_dirs)
            print(f"Final report generated: {report_path}")
        except Exception as e:
            print(f"Error generating report: {str(e)}")

    # SUMMARY
    print_header("PIPELINE COMPLETED")

    print(f"Run Directory: {pipeline_cfg.run_dir}")
    print(f"Training: {'Completed' if train_dir else 'Skipped'}")
    print(f"Testing: {'Completed' if test_dirs else 'Skipped'}")
    print()

    print("Key Files:")
    if train_dir:
        print(f"  - Training logs: {train_dir}/train_log.jsonl")
        print(f"  - Best model: {train_dir}/agent_best.pt")
    if test_dirs.get("test"):
        print(f"  - Test results: {test_dirs['test']}/test_summary.json")
    print(f"  - Final report: {pipeline_cfg.run_dir}/FINAL_REPORT.md")
    print()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="ARC Training Pipeline - Complete End-to-End Execution"
    )

    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs (default: 5)"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None,
        help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip training phase"
    )
    parser.add_argument(
        "--skip-test", action="store_true",
        help="Skip testing phase"
    )
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help="Continue to next phase on error"
    )
    parser.add_argument(
        "--no-report", action="store_true",
        help="Skip final report generation"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
