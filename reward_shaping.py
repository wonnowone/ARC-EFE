# reward_shaping.py
import torch
import torch.nn.functional as F

@torch.no_grad()
def per_cell_accuracy(pred_hw, tgt_hw):
    # pred, tgt: [H,W] (int colors)
    correct = (pred_hw == tgt_hw).float().mean()
    return correct

@torch.no_grad()
def size_gain(pred_hw, tgt_hw):
    Hp, Wp = pred_hw.shape
    Ht, Wt = tgt_hw.shape
    return 1.0 - (abs(Hp - Ht)/(Ht+1e-6) + abs(Wp - Wt)/(Wt+1e-6)) * 0.5

@torch.no_grad()
def color_agreement(pred_hw, tgt_hw, num_colors=10):
    # compare normalized histograms
    pred_hist = torch.stack([(pred_hw==c).float().mean() for c in range(num_colors)])
    tgt_hist  = torch.stack([(tgt_hw==c).float().mean()  for c in range(num_colors)])
    return 1.0 - F.l1_loss(pred_hist, tgt_hist, reduction="mean")

@torch.no_grad()
def reversible_gain(pred1, inp_hw, solver_back=None):
    # option: if backward solver provided, check reconstruction
    if solver_back is None:
        return torch.tensor(0.5)  # neutral
    recon = solver_back(pred1)   # should produce input-like grid
    sim = (recon == inp_hw).float().mean()
    return sim

@torch.no_grad()
def shaping_reward(before, after, tgt, inp, num_colors=10, solver_back=None):
    """
    before/after: [H,W] ints
    tgt:          [H,W]
    inp:          [H,W]
    """
    acc_b = per_cell_accuracy(before, tgt)
    acc_a = per_cell_accuracy(after,  tgt)

    sgain_b = size_gain(before, tgt);  sgain_a = size_gain(after, tgt)
    cgain_b = color_agreement(before, tgt, num_colors);  cgain_a = color_agreement(after, tgt, num_colors)
    rvgain_b = reversible_gain(before, inp, solver_back); rvgain_a = reversible_gain(after, inp, solver_back)

    # After - Before
    d_acc  = (acc_a  - acc_b).clamp(-1,1)
    d_size = (sgain_a- sgain_b).clamp(-1,1)
    d_col  = (cgain_a- cgain_b).clamp(-1,1)
    d_rev  = (rvgain_a- rvgain_b).clamp(-1,1)

    # Weighted sum
    reward = 1.0*d_acc + 0.5*d_size + 0.5*d_col + 0.5*d_rev
    return reward.item(), {
        "acc_before": float(acc_b), "acc_after": float(acc_a),
        "size_before": float(sgain_b), "size_after": float(sgain_a),
        "color_before": float(cgain_b), "color_after": float(cgain_a),
        "rev_before": float(rvgain_b), "rev_after": float(rvgain_a)
    }
