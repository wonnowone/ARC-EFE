"""
Grid Transformation Loss for ARC Challenge
Optimized for grid-level accuracy instead of abstract EFE concepts
Focuses on: Shape, Pixel Accuracy, Color Palette, Transformation Type
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class GridTransformationLoss(nn.Module):
    """
    Loss optimized for ARC grid transformation tasks.

    Key insight: ARC is about transforming grids correctly.
    Focus on:
    1. Output has correct shape (H×W)
    2. Output has correct colors (pixel accuracy)
    3. Output uses correct color palette
    4. Transformation type is preserved
    """

    def __init__(self,
                 shape_weight: float = 0.50,
                 pixel_weight: float = 0.35,
                 palette_weight: float = 0.10,
                 transform_weight: float = 0.05,
                 num_colors: int = 10):
        super().__init__()

        self.shape_weight = shape_weight
        self.pixel_weight = pixel_weight
        self.palette_weight = palette_weight
        self.transform_weight = transform_weight
        self.num_colors = num_colors

    def forward(self,
                predictions: torch.Tensor,  # [H, W, C] logits
                target: torch.Tensor,       # [H, W] class indices
                input_grid: torch.Tensor,   # [H, W] for transformation analysis
                ) -> Dict[str, torch.Tensor]:
        """
        Compute grid transformation loss.

        Args:
            predictions: [H, W, C] - predicted logits for each color
            target: [H, W] - target color indices (0-9)
            input_grid: [H, W] - input grid for transformation analysis

        Returns:
            Dictionary with loss components
        """

        losses = {}

        # 1. SHAPE LOSS - Ensure output dimensions match (or are valid transformation)
        shape_loss = self._compute_shape_loss(predictions, target)
        losses['shape'] = shape_loss

        # 2. PIXEL ACCURACY LOSS - Ensure correct colors at each position
        pixel_loss = self._compute_pixel_loss(predictions, target)
        losses['pixel'] = pixel_loss

        # 3. COLOR PALETTE LOSS - Use correct color set
        palette_loss = self._compute_palette_loss(predictions, target)
        losses['palette'] = palette_loss

        # 4. TRANSFORMATION LOSS - Reward correct transformation type
        transform_loss = self._compute_transformation_loss(
            predictions, target, input_grid
        )
        losses['transformation'] = transform_loss

        # Total weighted loss
        total_loss = (
            self.shape_weight * shape_loss +
            self.pixel_weight * pixel_loss +
            self.palette_weight * palette_loss +
            self.transform_weight * transform_loss
        )

        losses['total'] = total_loss

        return losses

    def _compute_shape_loss(self,
                           predictions: torch.Tensor,
                           target: torch.Tensor) -> torch.Tensor:
        """
        Penalize incorrect output shape.

        Assumption: Target shape is correct, prediction should match.
        If shapes don't match, penalize proportionally.
        """
        pred_h, pred_w = predictions.shape[:2]
        target_h, target_w = target.shape[:2]

        # Euclidean distance of shapes
        shape_error = torch.sqrt(
            torch.tensor((pred_h - target_h) ** 2 + (pred_w - target_w) ** 2).float()
        )

        # Normalize by maximum grid size (30)
        normalized_error = shape_error / 30.0

        # Clamp to [0, 1]
        loss = torch.clamp(normalized_error, 0.0, 1.0)

        return loss

    def _compute_pixel_loss(self,
                           predictions: torch.Tensor,
                           target: torch.Tensor) -> torch.Tensor:
        """
        Cross-entropy loss for pixel-level color accuracy.

        Handle shape mismatch by cropping or padding.
        """
        pred_h, pred_w, num_classes = predictions.shape
        target_h, target_w = target.shape

        # Handle shape mismatch
        if pred_h != target_h or pred_w != target_w:
            min_h = min(pred_h, target_h)
            min_w = min(pred_w, target_w)
            predictions = predictions[:min_h, :min_w, :]
            target = target[:min_h, :min_w]

        # Reshape for cross-entropy: [H*W, C] and [H*W]
        pred_flat = predictions.reshape(-1, num_classes)
        target_flat = target.reshape(-1)

        # Cross-entropy loss
        loss = F.cross_entropy(pred_flat, target_flat)

        return loss

    def _compute_palette_loss(self,
                             predictions: torch.Tensor,
                             target: torch.Tensor) -> torch.Tensor:
        """
        Ensure predicted output uses correct color palette.

        Reward: Using colors that appear in target
        Penalize: Using colors that don't appear in target
        """
        # Get predicted colors (most likely class per pixel)
        pred_colors = predictions.argmax(dim=-1)  # [H, W]

        # Get unique colors in target
        target_colors = torch.unique(target)
        pred_unique = torch.unique(pred_colors)

        # Count colors in target
        target_color_set = set(target_colors.cpu().numpy())
        pred_color_set = set(pred_unique.cpu().numpy())

        # Penalize for extra colors (not in target)
        extra_colors = pred_color_set - target_color_set
        extra_penalty = len(extra_colors) / self.num_colors

        # Penalize for missing colors (in target but not in pred)
        missing_colors = target_color_set - pred_color_set
        missing_penalty = len(missing_colors) / self.num_colors

        loss = (extra_penalty + missing_penalty) / 2.0

        return torch.tensor(loss, device=predictions.device, dtype=predictions.dtype)

    def _compute_transformation_loss(self,
                                    predictions: torch.Tensor,
                                    target: torch.Tensor,
                                    input_grid: torch.Tensor) -> torch.Tensor:
        """
        Detect and reward correct transformation type.

        Transformations: scaling, rotation, reflection, tiling, color_mapping
        """

        # Detect transformation type
        transform_type = self._detect_transformation(
            input_grid, target
        )

        # Check if prediction matches that transformation
        pred_colors = predictions.argmax(dim=-1)
        pred_type = self._detect_transformation(
            input_grid, pred_colors
        )

        # Loss: 0 if types match, 1 if different
        type_match = 0.0 if transform_type == pred_type else 1.0

        return torch.tensor(type_match, device=predictions.device, dtype=predictions.dtype)

    def _detect_transformation(self, input_grid: torch.Tensor,
                              output_grid: torch.Tensor) -> str:
        """
        Detect transformation type between input and output.

        Returns: 'scaling', 'rotation', 'reflection', 'tiling', 'color_mapping', 'unknown'
        """

        in_h, in_w = input_grid.shape
        out_h, out_w = output_grid.shape

        # Scaling
        if (out_h / in_h) == (out_w / in_w) and out_h != in_h:
            return 'scaling'

        # Size-preserving transformation
        if in_h == out_h and in_w == out_w:
            # Check for color mapping
            unique_in = len(torch.unique(input_grid))
            unique_out = len(torch.unique(output_grid))
            if unique_in == unique_out:
                return 'color_mapping'
            else:
                return 'filtering'  # Some colors removed/changed

        # Reflection or Rotation (same dimensions after transform)
        if (in_h == out_w and in_w == out_h):
            return 'reflection'

        # Tiling (expansion without scaling uniformly)
        if out_h > in_h and out_w > in_w:
            return 'tiling'

        return 'unknown'


class CombinedLoss(nn.Module):
    """
    Combines GridTransformationLoss with EFE for hybrid approach.
    """

    def __init__(self,
                 transformation_loss: GridTransformationLoss,
                 efe_loss: 'EFELoss',
                 transformation_weight: float = 0.7,
                 efe_weight: float = 0.3):
        super().__init__()

        self.transformation_loss = transformation_loss
        self.efe_loss = efe_loss
        self.transformation_weight = transformation_weight
        self.efe_weight = efe_weight

    def forward(self,
                predictions: torch.Tensor,
                target: torch.Tensor,
                input_grid: torch.Tensor,
                **efe_kwargs) -> Dict:
        """
        Compute combined loss.
        """

        # Transformation loss
        transform_losses = self.transformation_loss(
            predictions, target, input_grid
        )

        # EFE loss
        efe_losses = self.efe_loss(**efe_kwargs)

        # Combined
        total = (
            self.transformation_weight * transform_losses['total'] +
            self.efe_weight * efe_losses['total']
        )

        return {
            'total': total,
            'transformation': transform_losses['total'],
            'efe': efe_losses['total'],
            **transform_losses,
            **efe_losses
        }


if __name__ == "__main__":
    # Test the loss function
    print("Testing GridTransformationLoss...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    loss_fn = GridTransformationLoss(num_colors=10)

    # Create test data
    H, W, C = 10, 10, 10
    predictions = torch.randn(H, W, C, device=device)
    target = torch.randint(0, 10, (H, W), device=device)
    input_grid = torch.randint(0, 10, (H, W), device=device)

    # Compute loss
    losses = loss_fn(predictions, target, input_grid)

    print(f"\nLoss Components:")
    for key, val in losses.items():
        if torch.is_tensor(val):
            print(f"  {key}: {val.item():.6f}")
        else:
            print(f"  {key}: {val}")

    print("\n✓ GridTransformationLoss works!")
