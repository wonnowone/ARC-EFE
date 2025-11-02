"""
policy_refined.py

Integrated RL + Reward Shaping Prototype for Efficient ARC Training.

Combines:
  - HumanRLAugmentor: Learns to refine prompts via policy gradient
  - reward_shaping: Decomposes reward into interpretable signals (accuracy, size, color, reversibility)

Usage:
  policy = PolicyRefinedAgent(config)
  refined_prompt, rl_info = policy.refine_prompt(prompt, ctrl_vec, feat_summary)
  reward, reward_breakdown = policy.compute_reward(pred_before, pred_after, target, inp)
  losses = policy.update(rl_info, reward, advantage)
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as TF


# ============================================================================
# 1) EMBEDDED HUMAN RL AUGMENTOR (from human_rl_agent.py)
# ============================================================================

def _mlp(in_dim, hidden, out_dim, act=nn.ReLU):
    """Helper to create 3-layer MLP."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden), act(),
        nn.Linear(hidden, hidden), act(),
        nn.Linear(hidden, out_dim)
    )


def cosine_sim(a, b, eps=1e-8):
    """Compute cosine similarity between two vectors."""
    a = TF.normalize(a, dim=-1)
    b = TF.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)


@dataclass
class HumanRLConfig:
    prompt_dim: int = 256
    ctrl_dim: int = 256
    feat_dim: int = 32          # summary features (size, color, density, etc.)
    hidden: int = 512
    delta_scale: float = 0.2     # Δprompt magnitude limit (tanh * scale)
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    icm_coef: float = 0.1
    lr: float = 5e-5
    grad_clip: float = 1.0


class HumanRLAugmentor(nn.Module):
    """
    Non-invasive RL module for prompt refinement.

    Input: prompt_emb [D], ctrl_vec [D], feat_summary [F]
    Action: Δprompt [D], alpha ∈ (0,1)
    Value: V(s)
    ICM: Intrinsic reward (novelty)
    """
    def __init__(self, cfg: HumanRLConfig, device="cuda"):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)

        in_dim = cfg.prompt_dim + cfg.ctrl_dim + cfg.feat_dim

        # Policy: continuous Δprompt (Gaussian) + mixing coefficient alpha (sigmoid)
        self.pi_mean = _mlp(in_dim, cfg.hidden, cfg.prompt_dim)
        self.pi_logstd = nn.Parameter(torch.zeros(cfg.prompt_dim))
        self.alpha_head = _mlp(in_dim, cfg.hidden, 1)

        # Value function
        self.v = _mlp(in_dim, cfg.hidden, 1)

        # ICM: phi(s) = projection, phi_pred([s,a]) = prediction
        self.phi = _mlp(in_dim, cfg.hidden, cfg.hidden)
        self.phi_pred = _mlp(in_dim + cfg.prompt_dim, cfg.hidden, cfg.hidden)

        self.to(self.device)
        self.opt = torch.optim.Adam(self.parameters(), lr=cfg.lr)

    def forward(self, prompt_emb, ctrl_vec, feat_summary):
        """
        Args:
            prompt_emb: [B, D] or [D]
            ctrl_vec: [B, D] or [D]
            feat_summary: [B, F] or [F]

        Returns:
            dict with: delta, alpha, value, logp, entropy, x
        """
        # Ensure batch dimension
        if prompt_emb.dim() == 1:
            prompt_emb = prompt_emb.unsqueeze(0)
        if ctrl_vec.dim() == 1:
            ctrl_vec = ctrl_vec.unsqueeze(0)
        if feat_summary.dim() == 1:
            feat_summary = feat_summary.unsqueeze(0)

        x = torch.cat([prompt_emb, ctrl_vec, feat_summary], dim=-1)  # [B, in_dim]
        mu = self.pi_mean(x)                                         # [B, D]
        logstd = self.pi_logstd.expand_as(mu)                        # [B, D]
        std = logstd.exp()

        # Reparameterize
        eps = torch.randn_like(mu)
        delta = mu + std * eps                                       # [B, D]
        delta = torch.tanh(delta) * self.cfg.delta_scale

        alpha_logits = self.alpha_head(x)                            # [B, 1]
        alpha = torch.sigmoid(alpha_logits)                          # [B, 1]

        value = self.v(x).squeeze(-1)                                # [B]

        # Log probability
        logp_delta = -0.5 * (((delta - mu) / (std + 1e-8))**2 + 2*logstd + math.log(2*math.pi)).sum(dim=-1)
        logp_alpha = -TF.softplus(-alpha_logits).sum(dim=-1) - TF.softplus(alpha_logits).sum(dim=-1)
        logp = logp_delta + logp_alpha

        # Entropy
        ent_delta = (0.5 * (1.0 + math.log(2*math.pi)) + logstd).sum(dim=-1)
        p = alpha.clamp(1e-6, 1-1e-6)
        ent_alpha = -(p*torch.log(p) + (1-p)*torch.log(1-p)).sum(dim=-1)
        entropy = ent_delta + ent_alpha

        return {
            "delta": delta,
            "alpha": alpha,
            "value": value,
            "logp": logp,
            "entropy": entropy,
            "x": x
        }

    @torch.no_grad()
    def apply(self, prompt_emb, ctrl_vec, feat_summary):
        """
        Apply refinement to get new prompt.

        Returns:
            (new_prompt, rl_outputs)
        """
        out = self.forward(prompt_emb, ctrl_vec, feat_summary)
        delta, alpha = out["delta"], out["alpha"]

        # Ensure prompt_emb is batched
        if prompt_emb.dim() == 1:
            prompt_emb = prompt_emb.unsqueeze(0)

        # Hybrid: (1-α)*original + α*(original+Δ)
        new_prompt = (1.0 - alpha) * prompt_emb + alpha * (prompt_emb + delta)
        return new_prompt.squeeze(0), out

    def icm_intrinsic(self, s, a, s_next):
        """
        Compute ICM loss and intrinsic reward.

        Args:
            s: state [B, in_dim]
            a: action [B, D]
            s_next: next state [B, in_dim]

        Returns:
            (icm_loss, intrinsic_reward)
        """
        phi_s = self.phi(s)
        phi_next = self.phi(s_next)
        pred_next = self.phi_pred(torch.cat([s, a], dim=-1))
        icm_loss = TF.mse_loss(pred_next, phi_next, reduction="none").mean(dim=-1)
        intrinsic = icm_loss.detach()
        return icm_loss, intrinsic

    def update(self, logp, value, entropy, returns, advantages, icm_loss):
        """Update RL module with loss."""
        policy_loss = -(logp * advantages.detach()).mean()
        value_loss = TF.mse_loss(value, returns.detach())
        ent_bonus = entropy.mean()

        loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * ent_bonus \
               + self.cfg.icm_coef * icm_loss.mean()

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.cfg.grad_clip)
        self.opt.step()

        return {
            "loss": float(loss.item()),
            "policy": float(policy_loss.item()),
            "value": float(value_loss.item()),
            "entropy": float(ent_bonus.item()),
            "icm": float(icm_loss.mean().item())
        }


# ============================================================================
# 2) EMBEDDED REWARD SHAPING (from reward_shaping.py)
# ============================================================================

@torch.no_grad()
def per_cell_accuracy(pred_hw, tgt_hw):
    """Per-cell accuracy: fraction of matching cells.

    Handles size mismatches by resizing prediction to match target.
    Uses nearest-neighbor resize to preserve discrete color values.
    """
    # Handle size mismatch by resizing prediction to target size
    if pred_hw.shape != tgt_hw.shape:
        # Resize prediction to match target dimensions
        # Use nearest-neighbor interpolation to preserve discrete values
        pred_resized = torch.nn.functional.interpolate(
            pred_hw.float().unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
            size=tgt_hw.shape,
            mode='nearest'
        ).squeeze(0).squeeze(0)  # [H, W]
        pred_resized = pred_resized.round().long()  # Convert back to discrete
        pred_hw = pred_resized

    # Now shapes match, compute accuracy
    correct = (pred_hw == tgt_hw).float().mean()
    return correct


@torch.no_grad()
def size_gain(pred_hw, tgt_hw):
    """Size match: penalize dimension differences."""
    Hp, Wp = pred_hw.shape
    Ht, Wt = tgt_hw.shape
    return 1.0 - (abs(Hp - Ht)/(Ht+1e-6) + abs(Wp - Wt)/(Wt+1e-6)) * 0.5


@torch.no_grad()
def color_agreement(pred_hw, tgt_hw, num_colors=10):
    """Color histogram similarity.

    Handles size mismatches by resizing prediction to match target.
    Compares color distributions regardless of spatial layout.
    """
    # Handle size mismatch (same as per_cell_accuracy)
    if pred_hw.shape != tgt_hw.shape:
        pred_resized = torch.nn.functional.interpolate(
            pred_hw.float().unsqueeze(0).unsqueeze(0),
            size=tgt_hw.shape,
            mode='nearest'
        ).squeeze(0).squeeze(0)
        pred_resized = pred_resized.round().long()
        pred_hw = pred_resized

    # Compute color histograms
    pred_hist = torch.stack([(pred_hw==c).float().mean() for c in range(num_colors)])
    tgt_hist  = torch.stack([(tgt_hw==c).float().mean()  for c in range(num_colors)])
    return 1.0 - TF.l1_loss(pred_hist, tgt_hist, reduction="mean")


@torch.no_grad()
def reversible_gain(pred1, inp_hw, solver_back=None):
    """Reversibility: can backward model reconstruct input?

    Handles size mismatches by resizing reconstruction to match input.
    """
    if solver_back is None:
        return torch.tensor(0.5)

    recon = solver_back(pred1)

    # Handle size mismatch between reconstruction and input
    if recon.shape != inp_hw.shape:
        recon_resized = torch.nn.functional.interpolate(
            recon.float().unsqueeze(0).unsqueeze(0),
            size=inp_hw.shape,
            mode='nearest'
        ).squeeze(0).squeeze(0)
        recon_resized = recon_resized.round().long()
        recon = recon_resized

    sim = (recon == inp_hw).float().mean()
    return sim


@torch.no_grad()
def shaping_reward(before, after, tgt, inp, num_colors=10, solver_back=None):
    """
    Compute shaped reward: decomposed into 4 metrics.

    Args:
        before/after: [H,W] predicted grids
        tgt: [H,W] target grid
        inp: [H,W] input grid
        num_colors: number of colors (for histogram)
        solver_back: optional backward solver for reversibility check

    Returns:
        (total_reward, breakdown_dict)
    """
    acc_b = per_cell_accuracy(before, tgt)
    acc_a = per_cell_accuracy(after,  tgt)

    sgain_b = size_gain(before, tgt)
    sgain_a = size_gain(after, tgt)

    cgain_b = color_agreement(before, tgt, num_colors)
    cgain_a = color_agreement(after, tgt, num_colors)

    rvgain_b = reversible_gain(before, inp, solver_back)
    rvgain_a = reversible_gain(after, inp, solver_back)

    # Deltas (improvements) - convert to tensors for clamping
    d_acc  = torch.clamp(torch.tensor(acc_a  - acc_b, dtype=torch.float32), -1, 1)
    d_size = torch.clamp(torch.tensor(sgain_a - sgain_b, dtype=torch.float32), -1, 1)
    d_col  = torch.clamp(torch.tensor(cgain_a - cgain_b, dtype=torch.float32), -1, 1)
    d_rev  = torch.clamp(torch.tensor(rvgain_a - rvgain_b, dtype=torch.float32), -1, 1)

    # Weighted sum
    reward = 1.0*d_acc.item() + 0.5*d_size.item() + 0.5*d_col.item() + 0.5*d_rev.item()

    return reward, {
        "acc_before": float(acc_b), "acc_after": float(acc_a), "d_acc": float(d_acc.item()),
        "size_before": float(sgain_b), "size_after": float(sgain_a), "d_size": float(d_size.item()),
        "color_before": float(cgain_b), "color_after": float(cgain_a), "d_col": float(d_col.item()),
        "rev_before": float(rvgain_b), "rev_after": float(rvgain_a), "d_rev": float(d_rev.item())
    }


# ============================================================================
# 3) POLICY REFINED AGENT (Integration)
# ============================================================================

@dataclass
class PolicyRefinedConfig:
    """Configuration for integrated PolicyRefinedAgent."""
    # RL config
    rl_prompt_dim: int = 256
    rl_ctrl_dim: int = 256
    rl_feat_dim: int = 32
    rl_hidden: int = 512
    rl_delta_scale: float = 0.2
    rl_entropy_coef: float = 0.01
    rl_value_coef: float = 0.5
    rl_icm_coef: float = 0.1
    rl_lr: float = 5e-5
    rl_grad_clip: float = 1.0

    # Reward shaping config
    reward_acc_weight: float = 1.0
    reward_size_weight: float = 0.5
    reward_color_weight: float = 0.5
    reward_rev_weight: float = 0.5
    num_colors: int = 10

    # Integration config
    rl_loss_weight: float = 0.3     # Balance between RL loss and reward signal
    reward_normalization: str = "tanh"  # "tanh" or "linear"


class PolicyRefinedAgent(nn.Module):
    """
    Integrated RL + Reward Shaping agent for efficient prompt refinement.

    Combines:
      1. HumanRLAugmentor: Learns prompt modifications
      2. Reward Shaping: Provides interpretable multi-signal rewards
      3. Integration: Unified update with advantage estimation
    """

    def __init__(self, cfg: PolicyRefinedConfig = None, device="cuda"):
        super().__init__()
        self.cfg = cfg or PolicyRefinedConfig()
        self.device = torch.device(device)

        # Initialize RL augmentor
        rl_cfg = HumanRLConfig(
            prompt_dim=self.cfg.rl_prompt_dim,
            ctrl_dim=self.cfg.rl_ctrl_dim,
            feat_dim=self.cfg.rl_feat_dim,
            hidden=self.cfg.rl_hidden,
            delta_scale=self.cfg.rl_delta_scale,
            entropy_coef=self.cfg.rl_entropy_coef,
            value_coef=self.cfg.rl_value_coef,
            icm_coef=self.cfg.rl_icm_coef,
            lr=self.cfg.rl_lr,
            grad_clip=self.cfg.rl_grad_clip,
        )
        self.rl_augmentor = HumanRLAugmentor(rl_cfg, device=device)

        # Metrics tracking
        self.metrics_history = {
            "reward": [], "rl_loss": [], "accuracy": [], "size_gain": [],
            "color_agreement": [], "reversibility": []
        }

    def refine_prompt(self, prompt_emb, ctrl_vec, feat_summary):
        """
        Refine prompt using RL policy.

        Args:
            prompt_emb: [D] prompt embedding
            ctrl_vec: [D] control vector
            feat_summary: [F] feature summary

        Returns:
            (refined_prompt, rl_info)
        """
        refined, rl_info = self.rl_augmentor.apply(prompt_emb, ctrl_vec, feat_summary)
        return refined, rl_info

    def compute_reward(self, pred_before, pred_after, target, input_grid, solver_back=None):
        """
        Compute shaped reward comparing before/after predictions.

        Args:
            pred_before: [H,W] predicted grid before refinement
            pred_after: [H,W] predicted grid after refinement
            target: [H,W] target grid
            input_grid: [H,W] input grid
            solver_back: optional backward solver for reversibility

        Returns:
            (total_reward, breakdown_dict)
        """
        reward, breakdown = shaping_reward(
            pred_before, pred_after, target, input_grid,
            num_colors=self.cfg.num_colors,
            solver_back=solver_back
        )
        return reward, breakdown

    def normalize_reward(self, reward):
        """Normalize reward to [-1, 1] range for stability."""
        if self.cfg.reward_normalization == "tanh":
            return torch.tanh(torch.tensor(reward, device=self.device, dtype=torch.float32))
        else:  # linear
            return torch.clamp(torch.tensor(reward, device=self.device, dtype=torch.float32), -1, 1)

    def estimate_advantage(self, reward, value):
        """
        Estimate advantage: A(s,a) = R - V(s).

        Args:
            reward: scalar or tensor
            value: scalar or tensor from value function

        Returns:
            advantage tensor
        """
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor(reward, device=self.device, dtype=torch.float32)
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, device=self.device, dtype=torch.float32)

        advantage = reward - value
        return advantage

    def update(self, rl_info, reward, solver_back=None):
        """
        Unified update combining RL loss and reward signal.

        Args:
            rl_info: dict from rl_augmentor.forward()
            reward: float, shaped reward from shaping_reward()
            solver_back: optional backward solver

        Returns:
            losses_dict
        """
        # Normalize reward
        reward_norm = self.normalize_reward(reward)

        # Estimate advantage
        value = rl_info["value"].squeeze() if rl_info["value"].dim() > 0 else rl_info["value"]
        advantage = self.estimate_advantage(reward_norm, value.detach())

        # Prepare inputs for RL update
        logp = rl_info["logp"]
        entropy = rl_info["entropy"]

        # Dummy returns (could be GAE in full version)
        returns = reward_norm.expand_as(logp) if logp.dim() > 0 else reward_norm
        if advantage.dim() == 0:
            advantage = advantage.unsqueeze(0)
        if logp.dim() == 0:
            logp = logp.unsqueeze(0)
        if entropy.dim() == 0:
            entropy = entropy.unsqueeze(0)
        if returns.dim() == 0:
            returns = returns.unsqueeze(0)

        # ICM loss (intrinsic reward)
        # For prototype, use dummy s_next (in production, would be actual next state)
        s = rl_info["x"]
        a = rl_info["delta"]
        icm_loss, _ = self.rl_augmentor.icm_intrinsic(s, a, s)  # dummy s_next=s

        # RL update
        rl_losses = self.rl_augmentor.update(logp, rl_info["value"], entropy, returns, advantage, icm_loss)

        return {
            "reward": float(reward),
            "reward_norm": float(reward_norm.item()),
            "advantage": float(advantage.mean().item()),
            **rl_losses
        }

    def log_metrics(self, epoch, losses_dict, breakdown_dict):
        """Track and log metrics."""
        self.metrics_history["reward"].append(losses_dict.get("reward", 0))
        self.metrics_history["rl_loss"].append(losses_dict.get("loss", 0))
        self.metrics_history["accuracy"].append(breakdown_dict.get("d_acc", 0))
        self.metrics_history["size_gain"].append(breakdown_dict.get("d_size", 0))
        self.metrics_history["color_agreement"].append(breakdown_dict.get("d_col", 0))
        self.metrics_history["reversibility"].append(breakdown_dict.get("d_rev", 0))

    def get_metrics_summary(self):
        """Get summary of tracked metrics."""
        if not self.metrics_history["reward"]:
            return {}

        return {
            "avg_reward": sum(self.metrics_history["reward"]) / len(self.metrics_history["reward"]),
            "avg_rl_loss": sum(self.metrics_history["rl_loss"]) / len(self.metrics_history["rl_loss"]),
            "avg_accuracy_delta": sum(self.metrics_history["accuracy"]) / len(self.metrics_history["accuracy"]),
            "avg_size_delta": sum(self.metrics_history["size_gain"]) / len(self.metrics_history["size_gain"]),
            "avg_color_delta": sum(self.metrics_history["color_agreement"]) / len(self.metrics_history["color_agreement"]),
            "avg_rev_delta": sum(self.metrics_history["reversibility"]) / len(self.metrics_history["reversibility"]),
        }


# ============================================================================
# 4) STANDALONE USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("POLICY REFINED PROTOTYPE - Mock Training Loop")
    print("="*70 + "\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Initialize policy
    cfg = PolicyRefinedConfig()
    policy = PolicyRefinedAgent(cfg, device=device)
    print("[OK] PolicyRefinedAgent initialized\n")

    # Mock data
    batch_size = 1
    H, W = 15, 15
    D = 256
    F = 32

    # Random tensors
    prompt_emb = torch.randn(D, device=device)
    ctrl_vec = torch.randn(D, device=device)
    feat_summary = torch.randn(F, device=device)

    pred_before = torch.randint(0, 10, (H, W), device=device)
    pred_after = torch.randint(0, 10, (H, W), device=device)
    target = torch.randint(0, 10, (H, W), device=device)
    input_grid = torch.randint(0, 10, (H, W), device=device)

    print(f"Input shapes:")
    print(f"  prompt_emb: {prompt_emb.shape}")
    print(f"  ctrl_vec: {ctrl_vec.shape}")
    print(f"  feat_summary: {feat_summary.shape}")
    print(f"  pred grids: {pred_before.shape}\n")

    # Step 1: Refine prompt
    print("Step 1: Refining prompt with RL policy...")
    refined_prompt, rl_info = policy.refine_prompt(prompt_emb, ctrl_vec, feat_summary)
    print(f"  [OK] Refined prompt shape: {refined_prompt.shape}")
    print(f"  RL outputs: {[k for k in rl_info.keys()]}\n")

    # Step 2: Compute reward
    print("Step 2: Computing shaped reward...")
    reward, breakdown = policy.compute_reward(pred_before, pred_after, target, input_grid)
    print(f"  Reward: {reward:.4f}")
    print(f"  Breakdown:")
    for key, val in breakdown.items():
        if isinstance(val, float):
            print(f"    {key}: {val:.4f}")
    print()

    # Step 3: Update
    print("Step 3: Updating RL policy with reward signal...")
    losses = policy.update(rl_info, reward)
    print(f"  Update losses:")
    for key, val in losses.items():
        print(f"    {key}: {val:.6f}")
    print()

    # Step 4: Log metrics
    print("Step 4: Logging metrics...")
    policy.log_metrics(0, losses, breakdown)
    summary = policy.get_metrics_summary()
    print(f"  Summary (after 1 update):")
    for key, val in summary.items():
        print(f"    {key}: {val:.6f}")
    print()

    print("="*70)
    print("PROTOTYPE COMPLETE - Ready for integration into trainloop_gpu_finetuned.py")
    print("="*70 + "\n")
