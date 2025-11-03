"""
Quick test to verify the tensor dimension mismatch is fixed.
This test isolates the EFE loss computation that was failing.
"""

import torch
import torch.nn as nn
from loss_function import EFELoss

def test_tensor_dimension_fix():
    """Test that EFE loss works with target.long() instead of target.float()"""
    print("Testing tensor dimension fix for EFE loss...")
    print("=" * 60)

    # Initialize EFE loss with standard parameters
    efe_loss = EFELoss(
        lambda_risk=1.0, lambda_amb=0.0, lambda_step=0.1, lambda_cons=1.0,
        lambda_bi=0.5, lambda_z=0.2, lambda_prompt=0.3,
        max_grid_size=30, num_colors=10, prompt_dim=256
    )

    # Create sample tensors
    H, W, C = 6, 6, 10
    T = 3  # time steps
    device = 'cpu'

    # Input tensors
    forward_preds = torch.randn(T, H, W, C, device=device)
    backward_preds = torch.randn(T, H, W, C, device=device)
    state_preds = torch.randn(T, H, W, C, device=device)
    obs_probs = torch.softmax(torch.randn(T, H, W, C, device=device), dim=-1)
    final_pred = torch.randn(H, W, C, device=device)

    # Target with LONG dtype (the fix)
    target = torch.randint(0, C, (H, W), device=device).long()

    prompt_embedding = torch.randn(256, device=device)
    grid_mask = torch.ones(H, W, device=device)

    print(f"Forward predictions shape: {forward_preds.shape}")
    print(f"Final prediction shape: {final_pred.shape}")
    print(f"Target shape: {target.shape}, dtype: {target.dtype}")
    print()

    try:
        # This should now work without tensor dimension mismatch
        losses = efe_loss(
            forward_predictions=forward_preds,
            backward_predictions=backward_preds,
            state_predictions=state_preds,
            observation_probs=obs_probs,
            final_prediction=final_pred,
            target_outcome=target,
            episode_length=T,
            prompt_embedding=prompt_embedding,
            grid_mask=grid_mask
        )

        print("[OK] EFE Loss computation SUCCESSFUL!")
        print()
        print("Loss components:")
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key:20s}: {value.item():.6f}")

        print()
        print("=" * 60)
        print("TEST PASSED: Tensor dimension fix is working correctly!")
        print("=" * 60)
        return True

    except RuntimeError as e:
        print(f"[ERROR] {e}")
        print()
        print("TEST FAILED: The tensor dimension mismatch still exists")
        return False

if __name__ == "__main__":
    success = test_tensor_dimension_fix()
    exit(0 if success else 1)
