"""Fix gradient flow issues in fixed.py using regex"""

import re

with open('fixed.py', 'r', encoding='utf-8') as f:
    content = f.read()

print("Applying gradient flow fixes...")

# FIX 1: Remove the torch.no_grad() block that wraps agent.forward with qwen_prompt
# Pattern: with torch.no_grad():\n (multiple indented lines) predictions_before
pattern1 = r'with torch\.no_grad\(\):\n(\s+)feat_sum = torch\.tensor\((.*?)\)(\[:32\])\n\n(\s+)# Agent expects.*?pred_before = predictions_before\[-1\]\.argmax\(dim=-1\)'

replacement1 = r'''feat_sum = torch.tensor(\2)\3

        # Agent expects: input_grid [H,W], prompt_embedding [prompt_dim]
        # Returns: (predictions [num_steps, H, W, num_colors], features [H, W, hidden])
        predictions_before, _ = agent.forward(inp, qwen_prompt)
        pred_before = predictions_before[-1].argmax(dim=-1)'''

# Try the replacement
if re.search(pattern1, content, re.DOTALL):
    print("  [OK] Found problematic torch.no_grad() block")
    content_fixed = re.sub(pattern1, replacement1, content, flags=re.DOTALL)
    
    # Verify it changed
    if content_fixed != content:
        print("  [OK] Removed torch.no_grad() wrapper")
        content = content_fixed
    else:
        print("  [WARN] Pattern didn't match exactly, trying simpler approach")
        # Simpler: just remove the with torch.no_grad(): line and dedent
        content = re.sub(
            r'        with torch\.no_grad\(\):\n',
            '',
            content
        )
        print("  [OK] Removed torch.no_grad() line")

# Save
with open('fixed.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("[COMPLETE] Fixed gradient flow issues")
