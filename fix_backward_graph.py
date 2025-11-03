"""
Fix the double-backward issue by separating RL updates from main training
"""

with open('fixed.py', 'r', encoding='utf-8') as f:
    content = f.read()

# The issue: policy_rl.update() is called BEFORE the main loss backward
# This can cause graph conflicts. Solution: move it to AFTER

# Find the lines around policy_rl.update()
import re

# Pattern to find and move the RL update
pattern = r'''        # FIX #3: Add reward debugging and scaling
        acc_before = \(pred_before == out\)\.float\(\)\.mean\(\)\.item\(\)
        acc_after = \(pred_after == out\)\.float\(\)\.mean\(\)\.item\(\)
        raw_reward = reward
        reward = \(acc_after - acc_before\) \* 5\.0  # Scale reward signal

        # ========== STEP 5: UPDATE RL AGENT ==========
        rl_losses = policy_rl\.update\(rl_info, reward\)
        rl_reward = rl_losses\.get\("reward", 0\.0\)
        rl_rewards_sum \+= rl_reward'''

replacement = r'''        # FIX #3: Add reward debugging and scaling
        acc_before = (pred_before == out).float().mean().item()
        acc_after = (pred_after == out).float().mean().item()
        raw_reward = reward
        reward = (acc_after - acc_before) * 5.0  # Scale reward signal

        # (RL update moved to AFTER main backward to avoid graph conflicts)'''

if re.search(pattern, content):
    print("[OK] Found RL update section to relocate")
    content = re.sub(pattern, replacement, content)
    
    # Now add the RL update AFTER the scaler.step()
    # Find the scaler.step() line and add RL update after it
    step_pattern = r'(scaler\.step\(optimizer\)\s+scaler\.update\(\))'
    step_replacement = r'''\1

        # FIX: RL update moved here (after main backward) to avoid graph conflicts
        rl_losses = policy_rl.update(rl_info, reward)
        rl_reward = rl_losses.get("reward", 0.0)
        rl_rewards_sum += rl_reward'''
    
    if re.search(step_pattern, content):
        print("[OK] Moving RL update to after scaler.step()")
        content = re.sub(step_pattern, step_replacement, content)
    else:
        print("[WARN] Could not find scaler.step() location")
else:
    print("[WARN] Pattern not found, RL update may already be moved")

with open('fixed.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("[COMPLETE] Graph separation applied")
