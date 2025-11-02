"""
model_persistence.py

Robust model saving and loading system with:
  - Frequent checkpointing (every N batches)
  - Best model tracking
  - Automatic cleanup (old checkpoints)
  - Resume from checkpoint
  - Metadata tracking (epoch, batch, metrics)
  - Cloud backup support (Google Drive)
"""

import os
import json
import shutil
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import torch


class ModelCheckpoint:
    """Metadata for a single checkpoint."""

    def __init__(self, epoch: int, batch: int, metrics: Dict[str, float]):
        self.epoch = epoch
        self.batch = batch
        self.metrics = metrics
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "batch": self.batch,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        ckpt = cls(data["epoch"], data["batch"], data["metrics"])
        ckpt.timestamp = data["timestamp"]
        return ckpt


class ModelPersistence:
    """
    Robust model saving with cloud backup support.

    Features:
      - Frequent local saves (every N batches)
      - Best model tracking (by metric)
      - Resume from checkpoint
      - Automatic cleanup (keep last K checkpoints)
      - Google Drive backup (optional)
      - Metadata tracking
    """

    def __init__(self,
                 output_dir: str,
                 max_checkpoints: int = 5,
                 save_every_n_batches: int = 50,
                 backup_to_drive: bool = False,
                 drive_folder_id: Optional[str] = None):
        """
        Args:
            output_dir: Where to save checkpoints
            max_checkpoints: Keep only last K checkpoints
            save_every_n_batches: Save frequency
            backup_to_drive: Enable Google Drive backup
            drive_folder_id: Google Drive folder ID
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_checkpoints = max_checkpoints
        self.save_every_n_batches = save_every_n_batches
        self.backup_to_drive = backup_to_drive
        self.drive_folder_id = drive_folder_id

        # Paths
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.metadata_file = self.output_dir / "checkpoint_metadata.json"
        self.best_model_file = self.output_dir / "best_model.pt"
        self.best_metadata_file = self.output_dir / "best_metadata.json"

        # Tracking
        self.checkpoints: Dict[int, ModelCheckpoint] = {}  # checkpoint_id -> metadata
        self.best_metric_value: float = -float('inf')
        self.best_checkpoint_id: Optional[int] = None

        # Load existing metadata
        self._load_metadata()

    def should_save_checkpoint(self, batch_idx: int) -> bool:
        """Check if we should save at this batch."""
        return (batch_idx + 1) % self.save_every_n_batches == 0

    def save_checkpoint(self,
                       agent: torch.nn.Module,
                       qwen: torch.nn.Module,
                       solver2: torch.nn.Module,
                       efe_loss: torch.nn.Module,
                       policy_rl: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       batch: int,
                       metrics: Dict[str, float]) -> int:
        """
        Save checkpoint with metadata.

        Returns:
            checkpoint_id
        """
        checkpoint_id = len(self.checkpoints)

        # Prepare checkpoint
        checkpoint = {
            "epoch": epoch,
            "batch": batch,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "agent": agent.state_dict(),
            "qwen": qwen.state_dict(),
            "solver2": solver2.state_dict(),
            "efe_loss": efe_loss.state_dict(),
            "policy_rl": policy_rl.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        # Save to disk
        ckpt_path = self.checkpoint_dir / f"checkpoint_{checkpoint_id:05d}.pt"
        torch.save(checkpoint, ckpt_path)

        # Track metadata
        self.checkpoints[checkpoint_id] = ModelCheckpoint(epoch, batch, metrics)

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        # Cloud backup
        if self.backup_to_drive:
            self._backup_to_drive(ckpt_path)

        return checkpoint_id

    def save_best_model(self,
                       agent: torch.nn.Module,
                       qwen: torch.nn.Module,
                       solver2: torch.nn.Module,
                       efe_loss: torch.nn.Module,
                       policy_rl: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       batch: int,
                       metrics: Dict[str, float],
                       metric_name: str = "accuracy_delta",
                       is_better_fn=None) -> bool:
        """
        Save best model if current metrics are better.

        Args:
            metric_name: Which metric to track
            is_better_fn: Function to determine if better (default: higher is better)

        Returns:
            True if saved (new best), False otherwise
        """
        if is_better_fn is None:
            is_better_fn = lambda new, old: new > old

        metric_value = metrics.get(metric_name, 0.0)

        if is_better_fn(metric_value, self.best_metric_value):
            self.best_metric_value = metric_value

            # Save best model
            best_checkpoint = {
                "epoch": epoch,
                "batch": batch,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
                "agent": agent.state_dict(),
                "qwen": qwen.state_dict(),
                "solver2": solver2.state_dict(),
                "efe_loss": efe_loss.state_dict(),
                "policy_rl": policy_rl.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            torch.save(best_checkpoint, self.best_model_file)

            # Save metadata
            with open(self.best_metadata_file, "w") as f:
                json.dump({
                    "epoch": epoch,
                    "batch": batch,
                    "metrics": metrics,
                    "metric_tracked": metric_name,
                    "timestamp": datetime.now().isoformat(),
                }, f, indent=2)

            # Cloud backup
            if self.backup_to_drive:
                self._backup_to_drive(self.best_model_file)

            return True

        return False

    def load_checkpoint(self,
                       checkpoint_id: int,
                       agent: torch.nn.Module,
                       qwen: torch.nn.Module,
                       solver2: torch.nn.Module,
                       efe_loss: torch.nn.Module,
                       policy_rl: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       device: str = "cuda") -> Dict[str, Any]:
        """Load checkpoint and restore state."""

        ckpt_path = self.checkpoint_dir / f"checkpoint_{checkpoint_id:05d}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint {checkpoint_id} not found at {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=device)

        # Restore state dicts
        agent.load_state_dict(checkpoint["agent"])
        qwen.load_state_dict(checkpoint["qwen"])
        solver2.load_state_dict(checkpoint["solver2"])
        efe_loss.load_state_dict(checkpoint["efe_loss"])
        policy_rl.load_state_dict(checkpoint["policy_rl"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        # Return metadata for resuming
        return {
            "epoch": checkpoint["epoch"],
            "batch": checkpoint["batch"],
            "metrics": checkpoint["metrics"],
        }

    def load_best_model(self,
                       agent: torch.nn.Module,
                       qwen: torch.nn.Module,
                       solver2: torch.nn.Module,
                       efe_loss: torch.nn.Module,
                       policy_rl: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       device: str = "cuda") -> Dict[str, Any]:
        """Load best model."""

        if not self.best_model_file.exists():
            raise FileNotFoundError("Best model checkpoint not found")

        checkpoint = torch.load(self.best_model_file, map_location=device)

        # Restore
        agent.load_state_dict(checkpoint["agent"])
        qwen.load_state_dict(checkpoint["qwen"])
        solver2.load_state_dict(checkpoint["solver2"])
        efe_loss.load_state_dict(checkpoint["efe_loss"])
        policy_rl.load_state_dict(checkpoint["policy_rl"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        return {
            "epoch": checkpoint["epoch"],
            "batch": checkpoint["batch"],
            "metrics": checkpoint["metrics"],
        }

    def get_resume_info(self) -> Optional[Dict[str, Any]]:
        """Get info to resume from last checkpoint."""
        if not self.checkpoints:
            return None

        last_id = max(self.checkpoints.keys())
        ckpt = self.checkpoints[last_id]

        return {
            "checkpoint_id": last_id,
            "epoch": ckpt.epoch,
            "batch": ckpt.batch,
            "metrics": ckpt.metrics,
        }

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keep last K."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return

        # Sort by checkpoint ID
        checkpoint_ids = sorted(self.checkpoints.keys())

        # Remove oldest
        num_to_remove = len(checkpoint_ids) - self.max_checkpoints
        for checkpoint_id in checkpoint_ids[:num_to_remove]:
            ckpt_path = self.checkpoint_dir / f"checkpoint_{checkpoint_id:05d}.pt"
            if ckpt_path.exists():
                ckpt_path.unlink()
            del self.checkpoints[checkpoint_id]

    def _load_metadata(self):
        """Load existing metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                data = json.load(f)
                self.checkpoints = {
                    int(k): ModelCheckpoint.from_dict(v)
                    for k, v in data.items()
                }

        if self.best_metadata_file.exists():
            with open(self.best_metadata_file, "r") as f:
                data = json.load(f)
                self.best_metric_value = data["metrics"].get("accuracy_delta", -float('inf'))

    def _save_metadata(self):
        """Save metadata to disk."""
        data = {
            str(k): v.to_dict()
            for k, v in self.checkpoints.items()
        }

        with open(self.metadata_file, "w") as f:
            json.dump(data, f, indent=2)

    def _backup_to_drive(self, file_path: Path):
        """Upload checkpoint to Google Drive."""
        try:
            from google.colab import auth
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload

            auth.authenticate_user()
            drive_service = build('drive', 'v3')

            file_metadata = {
                'name': file_path.name,
                'parents': [self.drive_folder_id] if self.drive_folder_id else []
            }

            media = MediaFileUpload(str(file_path), resumable=True)
            drive_service.files().create(body=file_metadata, media_body=media).execute()

        except Exception as e:
            print(f"Warning: Could not backup to Drive: {e}")

    def save_metadata(self):
        """Explicitly save metadata."""
        self._save_metadata()


class TrainingState:
    """Track training state for resuming."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.state_file = self.output_dir / "training_state.json"

    def save(self,
            epoch: int,
            batch: int,
            total_batches: int,
            checkpoint_id: int):
        """Save current training state."""
        state = {
            "epoch": epoch,
            "batch": batch,
            "total_batches": total_batches,
            "checkpoint_id": checkpoint_id,
            "timestamp": datetime.now().isoformat(),
        }

        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def load(self) -> Optional[Dict[str, Any]]:
        """Load saved training state."""
        if not self.state_file.exists():
            return None

        with open(self.state_file, "r") as f:
            return json.load(f)


# Utility function for Colab
def setup_drive_backup(output_dir: str) -> Optional[str]:
    """
    Setup Google Drive backup for Colab.

    Returns:
        folder_id if successful, None otherwise
    """
    try:
        from google.colab import auth, drive

        auth.authenticate_user()
        drive.mount('/content/drive')

        # Create backup folder
        backup_path = Path('/content/drive/MyDrive/ARC-EFE-Backups')
        backup_path.mkdir(parents=True, exist_ok=True)

        print(f"Drive backup enabled: {backup_path}")
        return str(backup_path)

    except Exception as e:
        print(f"Note: Google Drive backup not available: {e}")
        return None
