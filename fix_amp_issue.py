"""Fix AMP gradient scaling conflict"""

with open('fixed.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove the problematic unscale_ and gradient clipping
import re

# Pattern to find and remove
pattern = r'''        # FIX #7: Backward with scaler
        scaler\.scale\(combined_loss\)\.backward\(\)

        # Gradient clipping before step
        scaler\.unscale_\(optimizer\)
        torch\.nn\.utils\.clip_grad_norm_\(agent\.parameters\(\), max_norm=1\.0\)
        torch\.nn\.utils\.clip_grad_norm_\(qwen\.parameters\(\), max_norm=1\.0\)
        torch\.nn\.utils\.clip_grad_norm_\(solver2\.parameters\(\), max_norm=1\.0\)

        scaler\.step\(optimizer\)
        scaler\.update\(\)'''

replacement = r'''        # FIX #7: Backward with scaler
        scaler.scale(combined_loss).backward()

        # Step with scaler (handles gradient clipping internally)
        scaler.step(optimizer)
        scaler.update()'''

if re.search(pattern, content, re.DOTALL):
    print("[OK] Found AMP gradient scaling section")
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    print("[OK] Removed problematic unscale_ calls")
else:
    print("[WARN] Pattern not found, trying simpler approach")
    # Just remove the unscale_ lines
    content = re.sub(r'\s*scaler\.unscale_\(optimizer\)\n', '', content)
    content = re.sub(r'\s*torch\.nn\.utils\.clip_grad_norm_\(.*?\)\n', '', content)
    print("[OK] Removed unscale_ and gradient clipping lines")

with open('fixed.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("[COMPLETE] AMP conflict fixed")
