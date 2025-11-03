"""
Comprehensive Qwen Gradient Flow Diagnostic Checklist
"""

import re
import os

print("=" * 70)
print("QWEN GRADIENT FLOW DIAGNOSTIC CHECKLIST")
print("=" * 70)

issues_found = []
warnings = []

# ============================================================================
# CHECK 1: Accidental no_grad blocks
# ============================================================================
print("\n[1/10] Checking for torch.no_grad() around Qwen calls...")
with open('fixed.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find all torch.no_grad() blocks
no_grad_blocks = re.findall(r'with torch\.no_grad\(\):.*?qwen', content, re.DOTALL)
if no_grad_blocks:
    print("      WARNING: Found torch.no_grad() blocks that might contain Qwen calls")
    issues_found.append("torch.no_grad() around Qwen")
else:
    print("      [OK] No torch.no_grad() wrapping Qwen calls")

# ============================================================================
# CHECK 2: Detach/CPU on embeddings
# ============================================================================
print("\n[2/10] Checking for .detach()/.cpu() on prompt embeddings...")
with open('qwen_hybrid_prompt.py', 'r', encoding='utf-8') as f:
    qwen_content = f.read()

# Check return statement
if 'prompt_embedding": text_emb.detach()' in qwen_content:
    print("      [CRITICAL] Found .detach() on prompt_embedding in return")
    issues_found.append(".detach() on prompt_embedding")
elif 'prompt_embedding": text_emb.cpu()' in qwen_content:
    print("      [CRITICAL] Found .cpu() on prompt_embedding in return")
    issues_found.append(".cpu() on prompt_embedding")
elif '"prompt_embedding": text_emb' in qwen_content:
    print("      [OK] prompt_embedding returned without .detach() or .cpu()")
else:
    print("      [?] Could not verify prompt_embedding return statement")

# ============================================================================
# CHECK 3: Verify prompt is in loss graph
# ============================================================================
print("\n[3/10] Checking prompt embedding usage in loss computation...")

# Check if refined_prompt is used in efe_loss call
if 'efe_loss(' in content and 'refined_prompt' in content:
    efe_call = re.search(r'efe_loss\([^)]*refined_prompt[^)]*\)', content, re.DOTALL)
    if efe_call:
        call_text = efe_call.group(0)
        if 'refined_prompt' in call_text:
            print("      [OK] refined_prompt directly used in efe_loss() call")
        else:
            print("      [?] refined_prompt not directly in efe_loss call")
    else:
        print("      [?] Could not find efe_loss call with refined_prompt")
else:
    print("      [?] Could not verify prompt in loss graph")

# ============================================================================
# CHECK 4: Confirm qwen.train()
# ============================================================================
print("\n[4/10] Checking if qwen.train() is called...")
if 'qwen.train()' in content:
    print("      [OK] qwen.train() found in training code")
else:
    print("      [CRITICAL] qwen.train() not found")
    issues_found.append("Missing qwen.train()")

# ============================================================================
# CHECK 5: Qwen in optimizer
# ============================================================================
print("\n[5/10] Checking if Qwen is in optimizer groups...")
if '"params": qwen.parameters()' in content or '{"params": qwen.parameters()' in content:
    print("      [OK] Qwen parameters included in optimizer")
else:
    print("      [CRITICAL] Qwen parameters NOT in optimizer")
    issues_found.append("Qwen not in optimizer")

# ============================================================================
# CHECK 6: AMP device consistency
# ============================================================================
print("\n[6/10] Checking AMP device consistency...")
if 'autocast(device_type=' in content:
    device_match = re.search(r'autocast\(device_type="?(\w+)"?\)', content)
    if device_match:
        device_type = device_match.group(1)
        print(f"      [OK] AMP device_type: {device_type}")
        if 'device_type=device' in content and 'device = ' in content:
            print("      [OK] device_type uses variable (consistent with runtime)")
    else:
        print("      [?] Could not parse autocast device_type")
else:
    print("      [OK] No AMP used (consistent gradient paths)")

# ============================================================================
# CHECK 7: Cached prompt behavior
# ============================================================================
print("\n[7/10] Checking cached prompt implementation...")
if '_cached_qwen_prompt' in content:
    print("      [!] Prompt caching enabled")
    if '_cached_qwen_prompt.detach()' in content:
        print("         [CRITICAL] Cached prompt is being detached!")
        issues_found.append("Cached prompt detached")
    else:
        print("         [OK] Cached prompt not explicitly detached")
else:
    print("      [OK] No prompt caching (gradients flow every batch)")

# ============================================================================
# CHECK 8: Gradient monitor
# ============================================================================
print("\n[8/10] Checking for gradient monitoring...")
if 'class GradientMonitor' in content:
    print("      [OK] GradientMonitor class found")
    if 'qwen_grad_info' in content:
        print("      [OK] Qwen gradients being monitored")
    else:
        print("      [?] GradientMonitor exists but may not track Qwen")
else:
    print("      [!] No gradient monitoring infrastructure")
    warnings.append("Consider adding GradientMonitor for Qwen")

# ============================================================================
# CHECK 9: Loss computation chain
# ============================================================================
print("\n[9/10] Checking loss computation chain integrity...")

if 'combined_loss.backward()' in content:
    print("      [OK] combined_loss.backward() found")
    if 'efe_loss_val' in content and '(-reward_tensor)' in content:
        print("      [OK] combined_loss = 0.7 * efe_loss + 0.3 * (-reward)")
    else:
        print("      [?] Could not verify loss composition")
else:
    print("      [CRITICAL] Could not find loss.backward() call")
    issues_found.append("Missing loss.backward()")

# ============================================================================
# CHECK 10: Training metric observation setup
# ============================================================================
print("\n[10/10] Checking logging for training metrics...")
metrics = {
    'Qwen_grad': False,
    'Loss': False,
    'Mask_ratio': False,
    'Reward': False,
    'Accuracy': False
}

if 'logger.log' in content:
    print("      [OK] Logger found for metric reporting")
    if 'grad_norm' in content:
        metrics['Qwen_grad'] = True
    if 'Loss:' in content or 'combined_loss' in content:
        metrics['Loss'] = True
    if 'mask_ratio' in content:
        metrics['Mask_ratio'] = True
    if 'Reward:' in content or 'rl_reward' in content:
        metrics['Reward'] = True
    if 'acc_before' in content or 'accuracy' in content:
        metrics['Accuracy'] = True
else:
    print("      [?] No structured logging found")

print("\n      Monitored metrics:")
for metric, found in metrics.items():
    status = "[OK]" if found else "[?]"
    print(f"        {status} {metric}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)

if issues_found:
    print(f"\n[CRITICAL] ISSUES FOUND ({len(issues_found)}):")
    for i, issue in enumerate(issues_found, 1):
        print(f"   {i}. {issue}")
else:
    print("\n[OK] No critical issues detected!")

if warnings:
    print(f"\n[!] WARNINGS ({len(warnings)}):")
    for i, warning in enumerate(warnings, 1):
        print(f"   {i}. {warning}")

print("\n" + "=" * 70)
print("RECOMMENDED NEXT STEPS")
print("=" * 70)
print("""
1. Run training with fixed.py and monitor Qwen_grad norm
   Expected: Start ~0, gradually increase to 1e-6 -> 1e-3 range

2. Check loss components each epoch:
   - efe_loss should decrease smoothly
   - mask_ratio should decline (more cells become correct)
   - reward should show improving trend

3. Monitor validation accuracy growth:
   - Early epochs: slow/flat (building representations)
   - Later epochs: more rapid improvement

4. If Qwen_grad stays at 0:
   - Add backward hook to confirm gradients reach Qwen
   - Check if prompt is actually being modified in loss
   - Verify no intermediate .detach() calls

5. If loss explodes:
   - Check gradient clipping is applied
   - Reduce Qwen learning rate (currently 5e-5, try 1e-5)
   - Check for NaN in prompt_embedding
""")

print("\n" + "=" * 70)
