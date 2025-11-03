"""Fix gradient flow issues in fixed.py"""

with open('fixed.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find and fix Issue 1: torch.no_grad() at line 232
# We need to move feat_sum creation outside, but keep agent.forward() outside
output_lines = []
i = 0
fixed_issue1 = False

while i < len(lines):
    line = lines[i]
    
    # Check if this is the problematic torch.no_grad() block
    if i == 231 and 'with torch.no_grad():' in line:  # Line 232 in 1-indexed
        print("[FIX 1] Removing torch.no_grad() block that wraps qwen_prompt usage...")
        # Skip the 'with torch.no_grad():' line
        i += 1
        # Dedent the next lines and add them directly
        while i < len(lines) and (lines[i].startswith('            ') or lines[i].strip() == ''):
            # Remove 4 spaces of indentation
            if lines[i].startswith('            '):
                output_lines.append(lines[i][4:])
            else:
                output_lines.append(lines[i])
            i += 1
        fixed_issue1 = True
        continue
    
    output_lines.append(line)
    i += 1

print(f"  {'[OK]' if fixed_issue1 else '[FAIL]'} torch.no_grad() removal")

# Write back
with open('fixed.py', 'w', encoding='utf-8') as f:
    f.writelines(output_lines)

print("[COMPLETE] Fixed gradient flow issues")
