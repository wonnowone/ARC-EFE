"""
Priority-Based EFE Loss with Auto-Prompt Updates

Insight: Use EFE as a learning framework, not strict matching.
Strategy:
  1. Priority 1: Grid size matching (reduce risk of total failure)
  2. Priority 2: Remove matched grids (reduce ambiguity)
  3. Priority 3: Z-learning for stability
  4. Priority 4: Future plan consistency (with updated prompts)
  5. Priority 5: Bidirectional learning (for coverage)

Additionally:
  - Auto-update prompts based on learned patterns
  - Iteratively remove solved cases (curriculum learning)
  - Integrate Z-learning for model confidence
  - Use bidirectional + future consistency for unmatched cases
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List


class AutoPromptLearner(nn.Module):
    """
    Learns to update prompts based on problem characteristics and success patterns.

    Maps: (input_grid, current_prompt, success_signal) → updated_prompt
    """

    def __init__(self, prompt_dim: int = 256, hidden_dim: int = 512):
        super().__init__()

        # Feature extraction from problem
        self.problem_encoder = nn.Sequential(
            nn.Linear(10, 64),  # 10 colors
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        # Prompt updater network
        # Input: prompt (256) + problem_features (128) + success_signal (3) = 387
        self.prompt_updater = nn.Sequential(
            nn.Linear(prompt_dim + 128 + 3, hidden_dim),  # prompt + problem_features + success_signal
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prompt_dim)
        )

        self.prompt_dim = prompt_dim

    def forward(self,
                prompt: torch.Tensor,              # [D] current prompt embedding
                input_grid: torch.Tensor,         # [H, W] input grid
                loss_signal: torch.Tensor,        # [1] scalar loss (lower = better)
                grid_size_match: torch.Tensor     # [1] grid size match quality (0-1)
                ) -> Tuple[torch.Tensor, Dict]:
        """
        Update prompt based on learning signal.

        Args:
            prompt: Current prompt embedding [D]
            input_grid: Input grid [H, W]
            loss_signal: Loss value (for learning direction)
            grid_size_match: How well output grid size matched target

        Returns:
            updated_prompt: New prompt [D]
            metadata: Info about update
        """

        # Extract color distribution features
        unique_colors = torch.unique(input_grid)
        color_histogram = torch.zeros(10, device=input_grid.device)
        for c in unique_colors:
            color_histogram[c] = (input_grid == c).float().mean()

        # Encode problem
        problem_features = self.problem_encoder(color_histogram)  # [128]

        # Create success signal
        success_signal = torch.cat([
            loss_signal.unsqueeze(0),  # Loss (lower = better)
            grid_size_match.unsqueeze(0),  # Size match
            torch.tensor([problem_features.norm().item()], device=input_grid.device)  # Complexity
        ], dim=0)  # [3]

        # Update prompt
        update_input = torch.cat([
            prompt,
            problem_features,
            success_signal
        ], dim=0)  # [D + 128 + 3]

        prompt_delta = self.prompt_updater(update_input)  # [D]

        # Apply update (residual connection)
        updated_prompt = prompt + 0.1 * prompt_delta  # Small update steps

        # Normalize
        updated_prompt = F.normalize(updated_prompt, p=2, dim=0) * prompt.norm()

        return updated_prompt, {
            'delta_norm': prompt_delta.norm().item(),
            'color_histogram': color_histogram,
            'problem_complexity': problem_features.norm().item()
        }


class PriorityEFELoss(nn.Module):
    """
    Priority-based EFE loss focusing on:
    1. Grid size matching (primary)
    2. Removing matched grids (ambiguity reduction)
    3. Z-learning for stability
    4. Future consistency (with updated prompts)
    5. Bidirectional learning (coverage)
    """

    def __init__(self,
                 prompt_dim: int = 256,
                 num_colors: int = 10,
                 max_grid_size: int = 30):
        super().__init__()

        # Auto-prompt learner
        self.prompt_learner = AutoPromptLearner(prompt_dim, hidden_dim=512)

        # Priority weights - can be scheduled during training
        self.w_grid_size = 1.5      # Priority 1: Grid size (prevent total failure)
        self.w_ambiguity = 0.8      # Priority 2: Remove ambiguity (matched grids)
        self.w_z_learning = 0.5     # Priority 3: Z-stability
        self.w_future = 1.0         # Priority 4: Future consistency
        self.w_bidirectional = 0.7  # Priority 5: Bidirectional coverage

        # EMA for Z-learning
        self.register_buffer('ema_success_rate', torch.tensor(0.5))
        self.ema_decay = 0.99

        # Matched grid tracking
        self.matched_threshold = 0.95  # Accuracy threshold for "matched"
        self.matched_grids_ids = set()  # Track which problems are solved

        self.num_colors = num_colors
        self.max_grid_size = max_grid_size

    def forward(self,
                forward_predictions: torch.Tensor,      # [H, W, C] logits
                backward_predictions: torch.Tensor,     # [H, W, C] logits
                target_outcome: torch.Tensor,           # [H, W] target
                input_grid: torch.Tensor,               # [H, W] input
                prompt_embedding: torch.Tensor,         # [D] prompt
                grid_id: Optional[int] = None,          # Track which grid this is
                ) -> Dict[str, torch.Tensor]:
        """
        Compute priority-based EFE loss.
        """

        losses = {}
        H, W = forward_predictions.shape[:2]

        # ============================================================
        # PRIORITY 1: GRID SIZE MATCHING (Prevent Total Failure)
        # ============================================================
        target_h, target_w = target_outcome.shape
        size_error = torch.tensor(
            abs(H - target_h) + abs(W - target_w),
            dtype=torch.float32,
            device=forward_predictions.device
        ) / self.max_grid_size
        size_loss = torch.clamp(size_error, 0, 1)
        losses['grid_size'] = size_loss

        # ============================================================
        # PRIORITY 2: REMOVE MATCHED GRIDS (Reduce Ambiguity)
        # ============================================================
        # Check if this grid is already "matched"
        pred_colors = forward_predictions.argmax(dim=-1)  # [H, W]

        # Crop to same size for comparison
        if H >= target_h and W >= target_w:
            pred_crop = pred_colors[:target_h, :target_w]
        else:
            # Pad prediction
            pad_h = max(0, target_h - H)
            pad_w = max(0, target_w - W)
            pred_crop = F.pad(pred_colors, (0, pad_w, 0, pad_h), value=0)

        accuracy = (pred_crop == target_outcome).float().mean()
        is_matched = accuracy > self.matched_threshold

        # If matched, reduce its weight in loss (don't overfit)
        if is_matched and grid_id is not None:
            self.matched_grids_ids.add(grid_id)
            matched_weight = 0.1  # Low weight for already-solved problems
        else:
            matched_weight = 1.0

        # Ambiguity loss (entropy over unmatched cases)
        if not is_matched:
            pred_probs = F.softmax(forward_predictions, dim=-1)
            entropy = -(pred_probs * torch.log(pred_probs + 1e-8)).sum(dim=-1).mean()
            ambiguity_loss = entropy  # High entropy = ambiguous
        else:
            ambiguity_loss = torch.tensor(0.0, device=forward_predictions.device)

        losses['ambiguity'] = ambiguity_loss

        # ============================================================
        # PRIORITY 3: Z-LEARNING STABILITY
        # ============================================================
        # Update EMA success rate
        with torch.no_grad():
            success_signal = 1.0 if is_matched else 0.0
            self.ema_success_rate.mul_(self.ema_decay).add_(
                success_signal * (1 - self.ema_decay)
            )

        # Z-anchoring: confidence should reflect actual success
        confidence_loss = torch.abs(accuracy - self.ema_success_rate)
        losses['z_anchoring'] = confidence_loss

        # ============================================================
        # PRIORITY 4: FUTURE PLAN CONSISTENCY (With Prompt Updates)
        # ============================================================
        # Pixel-level cross-entropy (actual correctness)
        pred_flat = forward_predictions.reshape(-1, self.num_colors)

        # Handle size mismatch
        if pred_crop.shape != target_outcome.shape:
            target_flat = target_outcome.reshape(-1)
            pred_flat = pred_flat[:target_flat.shape[0]]
        else:
            target_flat = target_outcome.reshape(-1)

        consistency_loss = F.cross_entropy(pred_flat, target_flat)
        losses['consistency'] = consistency_loss

        # ============================================================
        # PRIORITY 5: BIDIRECTIONAL LEARNING (Coverage)
        # ============================================================
        # Ensure forward and backward agree
        js_divergence = self._js_divergence(
            F.softmax(forward_predictions, dim=-1),
            F.softmax(backward_predictions, dim=-1)
        )
        losses['bidirectional'] = js_divergence

        # ============================================================
        # AUTO-PROMPT UPDATE
        # ============================================================
        updated_prompt, prompt_meta = self.prompt_learner(
            prompt_embedding,
            input_grid,
            torch.tensor(consistency_loss.item()),
            torch.tensor(1.0 - size_loss.item())  # Size match quality
        )
        losses['prompt_update'] = prompt_meta

        # ============================================================
        # TOTAL LOSS (Priority-weighted)
        # ============================================================
        total_loss = (
            self.w_grid_size * size_loss +
            self.w_ambiguity * ambiguity_loss * matched_weight +
            self.w_z_learning * losses['z_anchoring'] +
            self.w_future * consistency_loss +
            self.w_bidirectional * js_divergence
        )

        losses['total'] = total_loss
        losses['accuracy'] = accuracy
        losses['is_matched'] = torch.tensor(float(is_matched))
        losses['matched_weight'] = torch.tensor(matched_weight)
        losses['updated_prompt'] = updated_prompt

        return losses

    def _js_divergence(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Jensen-Shannon divergence between p and q"""
        m = 0.5 * (p + q)
        kl_pm = F.kl_div(torch.log(m + 1e-8), p, reduction='none').sum()
        kl_qm = F.kl_div(torch.log(m + 1e-8), q, reduction='none').sum()
        return 0.5 * (kl_pm + kl_qm)

    def set_schedule(self, epoch: int, total_epochs: int):
        """
        Adjust loss weights based on training stage.

        Early epochs: Focus on size + ambiguity (easy wins)
        Late epochs: Focus on consistency + bidirectional (hard cases)
        """
        progress = epoch / total_epochs

        if progress < 0.3:
            # Early: Focus on not failing completely
            self.w_grid_size = 2.0
            self.w_ambiguity = 1.0
            self.w_future = 0.5
            self.w_bidirectional = 0.3
        elif progress < 0.7:
            # Middle: Balance all objectives
            self.w_grid_size = 1.5
            self.w_ambiguity = 0.8
            self.w_future = 1.0
            self.w_bidirectional = 0.7
        else:
            # Late: Fine-tune consistency and bidirectional
            self.w_grid_size = 1.0
            self.w_ambiguity = 0.5
            self.w_future = 1.5
            self.w_bidirectional = 1.0

    def get_matched_count(self) -> int:
        """How many grids have been matched so far"""
        return len(self.matched_grids_ids)

    def get_ema_success_rate(self) -> float:
        """Current EMA success rate for Z-learning"""
        return self.ema_success_rate.item()


if __name__ == "__main__":
    print("Testing PriorityEFELoss with AutoPromptLearner...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    loss_fn = PriorityEFELoss(prompt_dim=256, num_colors=10)

    # Test data
    H, W, C = 10, 10, 10
    forward_pred = torch.randn(H, W, C, device=device)
    backward_pred = torch.randn(H, W, C, device=device)
    target = torch.randint(0, 10, (H, W), device=device)
    input_grid = torch.randint(0, 10, (H, W), device=device)
    prompt = torch.randn(256, device=device)

    # Compute loss
    losses = loss_fn(
        forward_predictions=forward_pred,
        backward_predictions=backward_pred,
        target_outcome=target,
        input_grid=input_grid,
        prompt_embedding=prompt,
        grid_id=0
    )

    print("\nLoss Components:")
    for key, val in losses.items():
        if torch.is_tensor(val):
            if val.dim() == 0:
                print(f"  {key}: {val.item():.6f}")
        elif isinstance(val, dict):
            print(f"  {key}: {val}")
        else:
            print(f"  {key}: {val}")

    print(f"\nMatched grids: {loss_fn.get_matched_count()}")
    print(f"EMA success rate: {loss_fn.get_ema_success_rate():.4f}")

    print("\n[OK] PriorityEFELoss works!")
