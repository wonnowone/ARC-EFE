"""Fix FP16 gradient unscaling error"""

with open('fixed.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find and remove the autocast context
output_lines = []
i = 0
in_autocast = False
autocast_indent = 0

while i < len(lines):
    line = lines[i]
    
    # Check if this is the autocast line
    if 'with autocast(device_type=device):' in line:
        print("[OK] Found autocast context")
        in_autocast = True
        autocast_indent = len(line) - len(line.lstrip())
        # Skip this line
        i += 1
        # Remove 4 spaces of indentation from following lines
        while i < len(lines):
            next_line = lines[i]
            # Check if we're still in the autocast block
            if next_line.strip() == '' or next_line.startswith(' ' * (autocast_indent + 4)):
                # This line is inside autocast, dedent it
                if next_line.startswith(' ' * (autocast_indent + 4)):
                    output_lines.append(next_line[4:])  # Remove 4 spaces
                else:
                    output_lines.append(next_line)  # Empty line
                i += 1
            else:
                # We've exited the autocast block
                in_autocast = False
                break
        continue
    
    output_lines.append(line)
    i += 1

with open('fixed.py', 'w', encoding='utf-8') as f:
    f.writelines(output_lines)

print("[COMPLETE] Removed autocast context (FP16 conflict)")
