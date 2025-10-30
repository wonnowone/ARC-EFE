# -*- coding: utf-8 -*-
"""
Grid Accuracy Based Loss Function for ARC Challenge

Simple and Direct Approach:
- Training Loss: Grid accuracy (percentage of correct cells)
- Validation: Binary accuracy (entire grid correct or not)
- Test: Binary accuracy (strict ARC evaluation)

This approach directly optimizes for what matters: correct grid predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class GridAccuracyLoss(nn.Module):
    """
    Grid Accuracy Based Loss Function

    Loss = 1 - (correct_cells / total_cells)

    This directly optimizes for grid correctness.
    Simple, interpretable, and aligns with ARC evaluation.
    """

    def __init__(self, use_focal: bool = False, focal_gamma: float = 2.0):
        """
        Args:
            use_focal: If True, apply focal loss weighting (harder samples weighted more)
            focal_gamma: Focal loss exponent (higher = focus on hard cases)
        """
        super().__init__()
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma

    def forward(self,
                predictions: torch.Tensor,  # [H, W, num_colors] logits
                targets: torch.Tensor,      # [H, W] target grid
                return_components: bool = False) -> torch.Tensor:
        """
        Calculate grid accuracy based loss.

        Args:
            predictions: [H, W, num_colors] logits from agent
            targets: [H, W] target grid values (0-9)
            return_components: If True, return (loss, accuracy, perfect)

        Returns:
            loss: Scalar tensor (1 - accuracy)
            Or tuple (loss, accuracy, is_perfect) if return_components=True
        """

        # Ensure correct shapes
        if predictions.dim() == 3:
            pred_classes = predictions.argmax(dim=-1)  # [H, W]
        else:
            pred_classes = predictions

        # Handle size mismatch (pad smaller outputs)
        if pred_classes.shape != targets.shape:
            if pred_classes.shape[0] < targets.shape[0]:
                pad_h = targets.shape[0] - pred_classes.shape[0]
                pred_classes = F.pad(pred_classes, (0, 0, 0, pad_h), value=0)
            if pred_classes.shape[1] < targets.shape[1]:
                pad_w = targets.shape[1] - pred_classes.shape[1]
                pred_classes = F.pad(pred_classes, (0, pad_w), value=0)

        # Crop to target size
        pred_classes = pred_classes[:targets.shape[0], :targets.shape[1]]

        # Calculate per-cell accuracy
        correct_cells = (pred_classes == targets).float()  # [H, W]
        accuracy = correct_cells.mean()  # Scalar: [0, 1]

        # Basic loss: 1 - accuracy
        loss = 1.0 - accuracy

        # Apply focal weighting if enabled
        if self.use_focal:
            # Focus on hard samples (low accuracy gets higher weight)
            focal_weight = (1.0 - accuracy) ** self.focal_gamma
            loss = loss * focal_weight

        if return_components:
            is_perfect = (correct_cells.sum() == correct_cells.numel())
            return loss, accuracy.item(), is_perfect.item()
        else:
            return loss


class ARCPromptGuidedAgentGPU(nn.Module):
    """
    ARC Prompt Guided Agent - GPU optimized version with grid accuracy loss

    Simplified from the original EFE version.
    Focus: Direct grid prediction with prompt guidance.
    """

    def __init__(self,
                 max_grid_size: int = 30,
                 num_colors: int = 10,
                 hidden_dim: int = 256,
                 prompt_dim: int = 256,
                 max_steps: int = 5):
        """
        Args:
            max_grid_size: Maximum grid size for ARC
            num_colors: Number of color classes (0-9)
            hidden_dim: Hidden dimension for internal networks
            prompt_dim: Dimension of prompt embeddings
            max_steps: Maximum number of planning steps (dynamically creates heads)
        """
        super().__init__()

        self.max_grid_size = max_grid_size
        self.num_colors = num_colors
        self.hidden_dim = hidden_dim
        self.prompt_dim = prompt_dim
        self.max_steps = max_steps

        # Grid encoder: encodes input grid to feature map
        self.input_encoder = nn.Sequential(
            nn.Conv2d(num_colors, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Prompt processor: encodes prompt embedding
        self.prompt_processor = nn.Sequential(
            nn.Linear(prompt_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Feature fusion: combine grid and prompt features
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Prediction heads for planning steps (dynamically created based on max_steps)
        self.planning_steps = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, num_colors, kernel_size=1)
            )
            for _ in range(max_steps)
        ])

        # Loss function
        self.loss_fn = GridAccuracyLoss(use_focal=True, focal_gamma=2.0)

    def forward(self, input_grid: torch.Tensor,
                prompt_embedding: torch.Tensor,
                num_steps: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through agent.

        Args:
            input_grid: [H, W] or [num_colors, H, W] input grid
            prompt_embedding: [prompt_dim] prompt embedding
            num_steps: Number of planning steps

        Returns:
            predictions: [num_steps, H, W, num_colors] planning trajectory
            features: [H, W, hidden_dim] intermediate features
        """

        # Handle input grid dimensions
        if input_grid.dim() == 2:
            # One-hot encode [H, W] → [num_colors, H, W]
            one_hot = torch.zeros(self.num_colors, *input_grid.shape,
                                 device=input_grid.device, dtype=input_grid.dtype)
            one_hot.scatter_(0, input_grid.unsqueeze(0), 1.0)
            input_grid = one_hot
        elif input_grid.dim() == 3:
            if input_grid.shape[0] != self.num_colors:
                pass
            else:
                pass

        # Encode input grid
        # Convert to float for convolution (input_grid is typically long/int type)
        grid_input = input_grid.float() if input_grid.dtype != torch.float32 else input_grid
        grid_features = self.input_encoder(grid_input.unsqueeze(0))  # [1, hidden, H, W]
        grid_features = grid_features.squeeze(0)  # [hidden, H, W]
        grid_features = grid_features.permute(1, 2, 0)  # [H, W, hidden]

        # Process prompt
        prompt_features = self.prompt_processor(prompt_embedding)  # [hidden]
        prompt_features = prompt_features.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden]
        prompt_features = prompt_features.expand(grid_features.shape[0],
                                                 grid_features.shape[1], -1)  # [H, W, hidden]

        # Fuse features
        combined = torch.cat([grid_features.unsqueeze(0),
                             prompt_features.unsqueeze(0)], dim=-1)  # [1, H, W, 2*hidden]
        combined = combined.permute(0, 3, 1, 2)  # [1, 2*hidden, H, W]
        fused = self.fusion(combined)  # [1, hidden, H, W]
        fused = fused.squeeze(0)  # [hidden, H, W]

        # Planning steps
        predictions = []
        state = fused
        for step_head in self.planning_steps[:num_steps]:
            pred = step_head(state.unsqueeze(0)).squeeze(0)  # [num_colors, H, W]
            predictions.append(pred)

        predictions = torch.stack(predictions)  # [num_steps, num_colors, H, W]
        predictions = predictions.permute(0, 2, 3, 1)  # [num_steps, H, W, num_colors]

        return predictions, fused

    def train_step(self,
                  input_grid: torch.Tensor,
                  target_grid: torch.Tensor,
                  prompt_embedding: torch.Tensor,
                  num_steps: int = 3) -> Dict:
        """
        Single training step.

        Args:
            input_grid: [H, W] input grid
            target_grid: [H, W] target grid
            prompt_embedding: [prompt_dim] prompt
            num_steps: Number of planning steps

        Returns:
            Dictionary with loss, accuracy, and per-step breakdown
        """
        # Forward pass
        predictions, _ = self.forward(input_grid, prompt_embedding, num_steps)

        # Calculate loss for each step
        losses = []
        accuracies = []
        for step_pred in predictions:
            loss, acc, _ = self.loss_fn(step_pred, target_grid, return_components=True)
            losses.append(loss)
            accuracies.append(acc)

        # Final loss is average of all steps (encourages early convergence)
        total_loss = torch.stack(losses).mean()

        return {
            'total': total_loss,
            'losses': losses,
            'accuracies': accuracies,
            'final_accuracy': accuracies[-1]
        }

    def eval_step(self,
                 input_grid: torch.Tensor,
                 target_grid: torch.Tensor,
                 prompt_embedding: torch.Tensor,
                 num_steps: int = 5,
                 binary_accuracy: bool = True) -> Dict:
        """
        Single evaluation step (no gradients).

        Args:
            input_grid: [H, W] input grid
            target_grid: [H, W] target grid
            prompt_embedding: [prompt_dim] prompt
            num_steps: Number of planning steps
            binary_accuracy: If True, use binary accuracy (strict ARC evaluation)

        Returns:
            Dictionary with accuracy metrics
        """
        with torch.no_grad():
            predictions, _ = self.forward(input_grid, prompt_embedding, num_steps)

            # Take final prediction
            final_pred = predictions[-1].argmax(dim=-1)  # [H, W]

            # Crop to target size if needed
            if final_pred.shape != target_grid.shape:
                final_pred = final_pred[:target_grid.shape[0], :target_grid.shape[1]]

            if binary_accuracy:
                # Binary: 1.0 only if entire grid perfect, else 0.0
                is_perfect = (final_pred == target_grid).all().item()
                acc = 1.0 if is_perfect else 0.0
                return {
                    'accuracy': acc,
                    'is_perfect': is_perfect,
                    'total_cells': target_grid.numel(),
                    'correct_cells': (final_pred == target_grid).sum().item()
                }
            else:
                # Per-cell accuracy
                correct = (final_pred == target_grid).float().mean().item()
                is_perfect = (final_pred == target_grid).all().item()
                return {
                    'accuracy': correct,
                    'is_perfect': is_perfect,
                    'total_cells': target_grid.numel(),
                    'correct_cells': (final_pred == target_grid).sum().item()
                }
