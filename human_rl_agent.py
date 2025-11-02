import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def _mlp(in_dim, hidden, out_dim, act=nn.ReLU):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), act(),
        nn.Linear(hidden, hidden), act(),
        nn.Linear(hidden, out_dim)
    )

def cosine_sim(a, b, eps=1e-8):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)

@dataclass
class HumanRLConfig:
    prompt_dim: int = 256
    ctrl_dim: int = 256
    feat_dim: int = 32          # 간단 요약 피처(사이즈/색/밀도 등)
    hidden: int = 512
    delta_scale: float = 0.2     # Δprompt 크기 제한 (tanh * scale)
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    icm_coef: float = 0.1
    lr: float = 5e-5
    grad_clip: float = 1.0

class HumanRLAugmentor(nn.Module):
    """
    비침투적 RL 모듈:
      입력: prompt_emb [D], ctrl_vec [D], feat_summary [F]
      행동: Δprompt [D], alpha ∈ (0,1)
      가치: V(s)
      ICM: intrinsic reward(새로움) 유도
    """
    def __init__(self, cfg: HumanRLConfig, device="cuda"):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)

        in_dim = cfg.prompt_dim + cfg.ctrl_dim + cfg.feat_dim

        # 정책: 연속 Δprompt (Gaussian) + 혼합계수 alpha (Bernoulli-sigmoid trick)
        self.pi_mean = _mlp(in_dim, cfg.hidden, cfg.prompt_dim)
        self.pi_logstd = nn.Parameter(torch.zeros(cfg.prompt_dim))  # learnable global logstd
        self.alpha_head = _mlp(in_dim, cfg.hidden, 1)               # alpha ∈ (0,1)

        # function value
        self.v = _mlp(in_dim, cfg.hidden, 1)

        # ICM: phi(s)=proj(state), phi(s')
        self.phi = _mlp(in_dim, cfg.hidden, cfg.hidden)
        self.phi_pred = _mlp(in_dim + cfg.prompt_dim, cfg.hidden, cfg.hidden)  # (state, action) -> phi(s')

        self.to(self.device)
        self.opt = torch.optim.Adam(self.parameters(), lr=cfg.lr)

    def forward(self, prompt_emb, ctrl_vec, feat_summary):
        x = torch.cat([prompt_emb, ctrl_vec, feat_summary], dim=-1)  # [B, in_dim]
        mu = self.pi_mean(x)                                         # [B, D]
        logstd = self.pi_logstd.expand_as(mu)                        # [B, D]
        std = logstd.exp()
        # reparameterize
        eps = torch.randn_like(mu)
        delta = mu + std * eps                                       # [B, D]
        delta = torch.tanh(delta) * self.cfg.delta_scale

        alpha_logits = self.alpha_head(x)                            # [B,1]
        alpha = torch.sigmoid(alpha_logits)                          # [B,1], mix weight

        value = self.v(x).squeeze(-1)                                # [B]

        # log prob (diag Gaussian for delta) + Bernoulli for alpha via logistic
        logp_delta = -0.5 * (((delta - mu) / (std + 1e-8))**2 + 2*logstd + math.log(2*math.pi)).sum(dim=-1)  # [B]
        # treat alpha as continuous w/ logit trick; entropy compute 별도
        logp_alpha = -F.softplus(-alpha_logits).sum(dim=-1) - F.softplus(alpha_logits).sum(dim=-1) + 0  # surrogate
        logp = logp_delta + logp_alpha

        # entropy (delta gaussian + alpha logistic)
        ent_delta = (0.5 * (1.0 + math.log(2*math.pi)) + logstd).sum(dim=-1)    # [B]
        # logistic entropy approximation
        p = alpha.clamp(1e-6, 1-1e-6)
        ent_alpha = -(p*torch.log(p) + (1-p)*torch.log(1-p)).sum(dim=-1)         # [B]
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
        out = self.forward(prompt_emb, ctrl_vec, feat_summary)
        delta, alpha = out["delta"], out["alpha"]
        # 하이브리드 임베딩 생성: (1-α)*prompt + α*(prompt+Δ)
        new_prompt = (1.0 - alpha) * prompt_emb + alpha * (prompt_emb + delta)
        return new_prompt, out  # out에는 logp/value/entropy 포함

    def icm_intrinsic(self, s, a, s_next):
        # s = phi(state), s_next_target = phi(state_next), s_next_pred = phi_pred([s, a])
        phi_s = self.phi(s)
        phi_next = self.phi(s_next)
        pred_next = self.phi_pred(torch.cat([s, a], dim=-1))
        icm_loss = F.mse_loss(pred_next, phi_next, reduction="none").mean(dim=-1)  # [B]
        intrinsic = icm_loss.detach()
        return icm_loss, intrinsic

    def update(self, logp, value, entropy, returns, advantages, icm_loss):
        # returns / advantages: [B]
        policy_loss = -(logp * advantages.detach()).mean()
        value_loss = F.mse_loss(value, returns.detach())
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
