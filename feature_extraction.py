import json
import csv
from collections import Counter, deque
import numpy as np

def merge_challenge_solution_data():
    """Merge training challenges and solutions into a single training.json file."""
    print("Loading challenge and solution files...")
    
    with open('../arc-agi_training_challenges.json', 'r') as f:
        challenges = json.load(f)
    
    with open('../arc-agi_training_solutions.json', 'r') as f:
        solutions = json.load(f)
    
    merged_training = {}
    
    for challenge_id in challenges:
        challenge_data = challenges[challenge_id]
        solution_data = solutions.get(challenge_id, [])
        
        # Merge train data
        merged_training[challenge_id] = {
            'train': challenge_data['train'],
            'test': []
        }
        
        # Merge test data with solutions
        for i, test_example in enumerate(challenge_data['test']):
            test_with_solution = {
                'input': test_example['input'],
                'output': solution_data[i] if i < len(solution_data) else []
            }
            merged_training[challenge_id]['test'].append(test_with_solution)
    
    # Save merged training data
    with open('training.json', 'w') as f:
        json.dump(merged_training, f, indent=2)
    
    print(f"Merged training data saved to training.json ({len(merged_training)} challenges)")
    return merged_training

def get_color_stats(grid):
    """Get color statistics for background classification."""
    if not grid or not grid[0]:
        return 0, 0, 0, 0
    
    flat_grid = [cell for row in grid for cell in row]
    color_counts = Counter(flat_grid)
    total_cells = len(flat_grid)
    
    if len(color_counts) < 2:
        return color_counts.most_common(1)[0][1], 0, 100, 0
    
    sorted_counts = sorted(color_counts.values(), reverse=True)
    dominant_count = sorted_counts[0]
    second_count = sorted_counts[1]
    dominant_pct = (dominant_count / total_cells) * 100
    second_pct = (second_count / total_cells) * 100
    
    return dominant_count, second_count, dominant_pct, second_pct

def get_color_map(grid):
    """Get color mapping for the grid."""
    if not grid or not grid[0]:
        return {}
    
    flat_grid = [cell for row in grid for cell in row]
    color_counts = Counter(flat_grid)
    return dict(color_counts)

def classify_background(grid):
    """Classify if grid has background based on dominant color threshold."""
    dom_count, sec_count, _, _ = get_color_stats(grid)
    return "yes" if dom_count >= 1.5 * sec_count else "no"

def find_lines(grid, color):
    """Find horizontal, vertical, and diagonal lines of specific color."""
    if not grid or not grid[0]:
        return [], [], []
    
    height, width = len(grid), len(grid[0])
    horizontal_lines = []
    vertical_lines = []
    diagonal_lines = []
    
    # Find horizontal lines (full width)
    for row in range(height):
        if all(grid[row][col] == color for col in range(width)):
            horizontal_lines.append(f"row_{row}")
    
    # Find vertical lines (full height)
    for col in range(width):
        if all(grid[row][col] == color for row in range(height)):
            vertical_lines.append(f"col_{col}")
    
    # Find diagonal lines (top-left to bottom-right)
    if height == width:  # Only for square grids
        if all(grid[i][i] == color for i in range(height)):
            diagonal_lines.append("main_diagonal")
        if all(grid[i][height-1-i] == color for i in range(height)):
            diagonal_lines.append("anti_diagonal")
    
    return horizontal_lines, vertical_lines, diagonal_lines

def flood_fill_shape(grid, start_row, start_col, visited):
    """Extract shape using flood fill algorithm."""
    if not grid or not grid[0]:
        return []
    
    height, width = len(grid), len(grid[0])
    target_color = grid[start_row][start_col]
    shape_pixels = []
    queue = deque([(start_row, start_col)])
    
    while queue:
        row, col = queue.popleft()
        
        if (row < 0 or row >= height or col < 0 or col >= width or 
            visited[row][col] or grid[row][col] != target_color):
            continue
        
        visited[row][col] = True
        shape_pixels.append((row, col))
        
        # Check 4-connected neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            queue.append((row + dr, col + dc))
    
    return shape_pixels

def extract_shapes(grid):
    """Extract all shapes from the grid."""
    if not grid or not grid[0]:
        return []
    
    height, width = len(grid), len(grid[0])
    visited = [[False] * width for _ in range(height)]
    shapes = []
    
    for row in range(height):
        for col in range(width):
            if not visited[row][col]:
                shape_pixels = flood_fill_shape(grid, row, col, visited)
                if shape_pixels:
                    color = grid[row][col]
                    
                    # Calculate boundaries
                    rows = [p[0] for p in shape_pixels]
                    cols = [p[1] for p in shape_pixels]
                    min_row, max_row = min(rows), max(rows)
                    min_col, max_col = min(cols), max(cols)
                    
                    # Calculate relative positions within bounding box
                    relative_positions = []
                    for r, c in shape_pixels:
                        rel_row = r - min_row
                        rel_col = c - min_col
                        relative_positions.append(f"({rel_row},{rel_col})")
                    
                    # Calculate bounding box dimensions
                    bounding_width = max_col - min_col + 1
                    bounding_height = max_row - min_row + 1
                    
                    shapes.append({
                        'color': color,
                        'boundaries': f"({min_row},{min_col})-({max_row},{max_col})",
                        'relative_locations': ";".join(relative_positions),
                        'pixel_count': len(shape_pixels),
                        'bounding_width': bounding_width,
                        'bounding_height': bounding_height
                    })
    
    return shapes

def extract_transformation_features(input_grid, output_grid):
    """
    Extract features that describe the transformation from input to output.
    Returns a dictionary of transformation characteristics.
    """
    if not input_grid or not input_grid[0] or not output_grid or not output_grid[0]:
        return {}

    features = {}

    # Grid size features
    in_h, in_w = len(input_grid), len(input_grid[0])
    out_h, out_w = len(output_grid), len(output_grid[0])

    features['input_height'] = in_h
    features['input_width'] = in_w
    features['output_height'] = out_h
    features['output_width'] = out_w
    features['size_change_ratio'] = (out_h * out_w) / (in_h * in_w) if (in_h * in_w) > 0 else 1.0
    features['size_preserved'] = 1.0 if (in_h == out_h and in_w == out_w) else 0.0

    # Color features
    input_colors = set(cell for row in input_grid for cell in row)
    output_colors = set(cell for row in output_grid for cell in row)

    features['input_color_count'] = len(input_colors)
    features['output_color_count'] = len(output_colors)
    features['colors_added'] = len(output_colors - input_colors)
    features['colors_removed'] = len(input_colors - output_colors)
    features['color_preservation'] = len(input_colors & output_colors) / len(input_colors | output_colors) if len(input_colors | output_colors) > 0 else 0.0

    # Pixel change analysis (only if same size)
    if in_h == out_h and in_w == out_w:
        changed_pixels = sum(1 for i in range(in_h) for j in range(in_w)
                           if input_grid[i][j] != output_grid[i][j])
        total_pixels = in_h * in_w
        features['pixel_change_ratio'] = changed_pixels / total_pixels if total_pixels > 0 else 0.0

        # Spatial correlation
        input_flat = np.array([input_grid[i][j] for i in range(in_h) for j in range(in_w)], dtype=float)
        output_flat = np.array([output_grid[i][j] for i in range(in_h) for j in range(in_w)], dtype=float)

        if len(input_flat) > 1 and input_flat.std() > 0 and output_flat.std() > 0:
            correlation = np.corrcoef(input_flat, output_flat)[0, 1]
            features['spatial_correlation'] = float(correlation) if not np.isnan(correlation) else 0.0
        else:
            features['spatial_correlation'] = 0.0
    else:
        features['pixel_change_ratio'] = 1.0
        features['spatial_correlation'] = 0.0

    # Symmetry analysis
    features['input_h_symmetry'] = 1.0 if is_symmetric_horizontal(input_grid) else 0.0
    features['input_v_symmetry'] = 1.0 if is_symmetric_vertical(input_grid) else 0.0
    features['output_h_symmetry'] = 1.0 if is_symmetric_horizontal(output_grid) else 0.0
    features['output_v_symmetry'] = 1.0 if is_symmetric_vertical(output_grid) else 0.0
    features['symmetry_change'] = abs((features['input_h_symmetry'] + features['input_v_symmetry']) -
                                      (features['output_h_symmetry'] + features['output_v_symmetry']))

    # Pattern density
    input_non_bg = sum(1 for row in input_grid for cell in row if cell != get_background_color(input_grid))
    output_non_bg = sum(1 for row in output_grid for cell in row if cell != get_background_color(output_grid))

    features['input_density'] = input_non_bg / (in_h * in_w) if (in_h * in_w) > 0 else 0.0
    features['output_density'] = output_non_bg / (out_h * out_w) if (out_h * out_w) > 0 else 0.0
    features['density_change'] = features['output_density'] - features['input_density']

    return features

def is_symmetric_horizontal(grid):
    """Check if grid is symmetric horizontally (left-right mirror)"""
    if not grid or not grid[0]:
        return False
    height = len(grid)
    width = len(grid[0])
    for i in range(height):
        for j in range(width // 2):
            if grid[i][j] != grid[i][width - 1 - j]:
                return False
    return True

def is_symmetric_vertical(grid):
    """Check if grid is symmetric vertically (top-bottom mirror)"""
    if not grid or not grid[0]:
        return False
    height = len(grid)
    for i in range(height // 2):
        if grid[i] != grid[height - 1 - i]:
            return False
    return True

def get_background_color(grid):
    """Get the most common color (likely background)"""
    if not grid or not grid[0]:
        return 0
    flat = [cell for row in grid for cell in row]
    return Counter(flat).most_common(1)[0][0]

def classify_transformation_type(transform_features):
    """
    Classify the type of transformation based on extracted features.
    Returns a string describing the transformation category.
    """
    if not transform_features:
        return "unknown"

    # Extract key features
    size_change = transform_features.get('size_change_ratio', 1.0)
    size_preserved = transform_features.get('size_preserved', 0.0)
    pixel_change_ratio = transform_features.get('pixel_change_ratio', 1.0)
    color_preservation = transform_features.get('color_preservation', 1.0)
    spatial_correlation = transform_features.get('spatial_correlation', 0.0)
    symmetry_change = transform_features.get('symmetry_change', 0.0)
    density_change = transform_features.get('density_change', 0.0)
    colors_added = transform_features.get('colors_added', 0)
    colors_removed = transform_features.get('colors_removed', 0)

    # Classification logic (order matters - more specific first)

    # 1. Size changes
    if size_change < 0.9:
        return "shrinking"
    elif size_change > 1.1:
        return "expanding"

    # 2. Shape not preserved
    if size_preserved == 0.0:
        if size_change == 1.0:
            return "reshaping"
        else:
            return "size_transformation"

    # 3. Minimal changes (copy-like)
    if pixel_change_ratio < 0.05:
        return "copy_or_minimal"

    # 4. Color-based transformations
    if color_preservation < 0.3:
        if colors_added > 0 and colors_removed > 0:
            return "recoloring"
        elif colors_added > 0:
            return "color_addition"
        elif colors_removed > 0:
            return "color_removal"

    # 5. Symmetry operations
    if symmetry_change > 0.5:
        return "symmetry_operation"

    # 6. Spatial transformations with high pixel change but high correlation
    if pixel_change_ratio > 0.5 and spatial_correlation > 0.5:
        return "spatial_shift"

    # 7. Pattern completion/filling
    if density_change > 0.2:
        return "pattern_completion"
    elif density_change < -0.2:
        return "pattern_removal"

    # 8. Reconstruction (low correlation, high change)
    if spatial_correlation < 0.3 and pixel_change_ratio > 0.5:
        return "reconstruction"

    # 9. Filtering/selection
    if pixel_change_ratio > 0.3 and density_change < -0.1:
        return "filtering"

    # 10. Pattern transformation (moderate changes)
    if 0.2 < pixel_change_ratio < 0.8:
        return "pattern_transformation"

    # Default
    return "general_transformation"

def process_grid(prob_id, grid_type, grid_num, grid):
    """Process a single grid and extract all features."""
    if not grid or not grid[0]:
        return []
    
    height, width = len(grid), len(grid[0])
    grid_size = f"{height}x{width}"
    background = classify_background(grid)
    color_map = get_color_map(grid)
    
    features = []
    
    for color in color_map.keys():
        horizontal_lines, vertical_lines, diagonal_lines = find_lines(grid, color)
        
        # Get line locations as strings
        h_lines = ";".join(horizontal_lines) if horizontal_lines else ""
        v_lines = ";".join(vertical_lines) if vertical_lines else ""
        d_lines = ";".join(diagonal_lines) if diagonal_lines else ""
        
        features.append({
            'prob_id': prob_id,
            'grid_label': f"{grid_type}_{grid_num}",
            'grid_size': grid_size,
            'background': background,
            'color': color,
            'horizontal_lines': h_lines,
            'vertical_lines': v_lines,
            'diagonal_lines': d_lines,
            'shapes': ""  # Will be filled by shape extraction
        })
    
    # Extract shapes
    shapes = extract_shapes(grid)
    
    # Group shapes by color
    shape_by_color = {}
    for shape in shapes:
        color = shape['color']
        if color not in shape_by_color:
            shape_by_color[color] = []
        shape_by_color[color].append(shape)
    
    # Add shape information to features
    for feature in features:
        color = feature['color']
        if color in shape_by_color:
            shape_info = []
            for shape in shape_by_color[color]:
                shape_str = f"boundary:{shape['boundaries']}|location:{shape['relative_locations']}|pixels:{shape['pixel_count']}|bounds:{shape['bounding_width']}x{shape['bounding_height']}"
                shape_info.append(shape_str)
            feature['shapes'] = ";;".join(shape_info)
    
    return features

def extract_all_features():
    """Extract features from all training data."""
    print("Loading training data...")
    with open('training.json', 'r') as f:
        training_data = json.load(f)

    all_grid_features = []
    all_transformation_features = []

    for prob_id in training_data:
        challenge_data = training_data[prob_id]

        # Process training examples
        for i, example in enumerate(challenge_data['train']):
            input_features = process_grid(prob_id, 'train_input', i, example['input'])
            output_features = process_grid(prob_id, 'train_output', i, example['output'])
            all_grid_features.extend(input_features)
            all_grid_features.extend(output_features)

            # Extract transformation features
            transform_features = extract_transformation_features(example['input'], example['output'])
            transform_type = classify_transformation_type(transform_features)

            # Create transformation record
            transform_record = {
                'prob_id': prob_id,
                'example_type': 'train',
                'example_num': i,
                'transformation_type': transform_type,
                **transform_features  # Unpack all transformation features
            }
            all_transformation_features.append(transform_record)

        # Process test examples
        for i, example in enumerate(challenge_data['test']):
            input_features = process_grid(prob_id, 'test_input', i, example['input'])
            output_features = process_grid(prob_id, 'test_output', i, example['output'])
            all_grid_features.extend(input_features)
            all_grid_features.extend(output_features)

            # Extract transformation features
            transform_features = extract_transformation_features(example['input'], example['output'])
            transform_type = classify_transformation_type(transform_features)

            # Create transformation record
            transform_record = {
                'prob_id': prob_id,
                'example_type': 'test',
                'example_num': i,
                'transformation_type': transform_type,
                **transform_features
            }
            all_transformation_features.append(transform_record)

    return all_grid_features, all_transformation_features

def create_prompt_from_transformation(prob_id, transformation_record, grid_features=None):
    """
    Create a natural language prompt/objective from transformation features.
    This will be used to guide the EFE agent.
    """
    transform_type = transformation_record.get('transformation_type', 'unknown')

    # Base objective from transformation type
    type_to_objective = {
        'shrinking': 'Reduce the grid size while preserving key patterns',
        'expanding': 'Expand the grid size by extending or repeating patterns',
        'reshaping': 'Transform the grid shape while maintaining pattern elements',
        'size_transformation': 'Change both size and structure of the pattern',
        'copy_or_minimal': 'Copy or make minimal adjustments to the input',
        'recoloring': 'Change colors while preserving spatial structure',
        'color_addition': 'Add new colors to the existing pattern',
        'color_removal': 'Remove specific colors from the pattern',
        'symmetry_operation': 'Apply symmetry transformations (mirror, rotate)',
        'spatial_shift': 'Shift or translate pattern elements spatially',
        'pattern_completion': 'Complete or fill in missing pattern elements',
        'pattern_removal': 'Remove or filter out specific pattern elements',
        'reconstruction': 'Reconstruct the pattern using transformation rules',
        'filtering': 'Filter and select specific pattern elements',
        'pattern_transformation': 'Transform the pattern structure systematically',
        'general_transformation': 'Apply systematic transformation to the pattern'
    }

    base_objective = type_to_objective.get(transform_type, 'Analyze and transform the input pattern')

    # Add specific guidance based on features
    guidance_parts = [base_objective]

    # Size guidance
    size_change = transformation_record.get('size_change_ratio', 1.0)
    if size_change < 0.9:
        guidance_parts.append(f"Target size is {size_change:.1%} of input")
    elif size_change > 1.1:
        guidance_parts.append(f"Target size is {size_change:.1%} of input")

    # Color guidance
    colors_added = transformation_record.get('colors_added', 0)
    colors_removed = transformation_record.get('colors_removed', 0)
    if colors_added > 0:
        guidance_parts.append(f"Introduce {colors_added} new color(s)")
    if colors_removed > 0:
        guidance_parts.append(f"Remove {colors_removed} color(s)")

    # Symmetry guidance
    symmetry_change = transformation_record.get('symmetry_change', 0.0)
    if symmetry_change > 0.5:
        guidance_parts.append("Pay attention to symmetry changes")

    # Density guidance
    density_change = transformation_record.get('density_change', 0.0)
    if density_change > 0.2:
        guidance_parts.append("Pattern becomes denser")
    elif density_change < -0.2:
        guidance_parts.append("Pattern becomes sparser")

    # Spatial correlation
    spatial_corr = transformation_record.get('spatial_correlation', 0.0)
    if spatial_corr > 0.7:
        guidance_parts.append("Maintain strong spatial relationships")
    elif spatial_corr < 0.3:
        guidance_parts.append("Significant spatial reorganization required")

    # Construct final prompt
    prompt = ". ".join(guidance_parts) + "."

    return {
        'prob_id': prob_id,
        'transformation_type': transform_type,
        'prompt': prompt,
        'size_change_ratio': size_change,
        'complexity_score': calculate_complexity_score(transformation_record)
    }

def calculate_complexity_score(transformation_record):
    """
    Calculate a complexity score for the transformation (0-10 scale).
    Higher scores indicate more complex transformations.
    """
    score = 0.0

    # Size change complexity
    size_change = transformation_record.get('size_change_ratio', 1.0)
    if abs(size_change - 1.0) > 0.1:
        score += 2.0

    # Pixel change complexity
    pixel_change = transformation_record.get('pixel_change_ratio', 0.0)
    score += pixel_change * 3.0

    # Color complexity
    colors_added = transformation_record.get('colors_added', 0)
    colors_removed = transformation_record.get('colors_removed', 0)
    score += (colors_added + colors_removed) * 0.5

    # Spatial correlation (low = more complex)
    spatial_corr = transformation_record.get('spatial_correlation', 0.5)
    score += (1.0 - spatial_corr) * 2.0

    # Symmetry changes
    symmetry_change = transformation_record.get('symmetry_change', 0.0)
    score += symmetry_change * 1.5

    # Cap at 10
    return min(score, 10.0)

def save_to_csv(features, filename='arc_features.csv'):
    """Save extracted features to CSV file."""
    if not features:
        print("No features to save")
        return

    # Get fieldnames from first record
    fieldnames = list(features[0].keys())

    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(features)

        print(f"Features saved to {filename} ({len(features)} rows)")
    except PermissionError:
        # If locked, try backup filename
        backup_filename = filename.replace('.csv', '_backup.csv')
        with open(backup_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(features)

        print(f"Features saved to {backup_filename} ({len(features)} rows)")

def main():
    """Main function to run feature extraction."""
    print("Starting feature extraction process...")

    # Step 1: Merge data
    merged_data = merge_challenge_solution_data()

    # Step 2: Extract all features (grid-level and transformation-level)
    print("Extracting features...")
    grid_features, transformation_features = extract_all_features()

    # Step 3: Save grid-level features
    print("\nSaving grid-level features...")
    save_to_csv(grid_features, 'arc_grid_features.csv')

    # Step 4: Save transformation-level features
    print("\nSaving transformation-level features...")
    save_to_csv(transformation_features, 'arc_transformation_features.csv')

    # Step 5: Generate prompts for EFE system
    print("\nGenerating natural language prompts for EFE...")
    prompts = []
    for transform_record in transformation_features:
        prob_id = transform_record['prob_id']
        prompt_info = create_prompt_from_transformation(prob_id, transform_record)
        prompts.append(prompt_info)

    save_to_csv(prompts, 'arc_efe_prompts.csv')

    # Step 6: Print summary statistics
    print("\n" + "="*60)
    print("Feature extraction completed!")
    print(f"Grid-level features: {len(grid_features)} rows")
    print(f"Transformation features: {len(transformation_features)} rows")
    print(f"EFE prompts generated: {len(prompts)} rows")

    # Transformation type distribution
    transform_types = {}
    for t in transformation_features:
        ttype = t['transformation_type']
        transform_types[ttype] = transform_types.get(ttype, 0) + 1

    print(f"\nTransformation type distribution:")
    for ttype, count in sorted(transform_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ttype}: {count}")

    print(f"\nFiles created:")
    print(f"  - arc_grid_features.csv (per-grid color/shape/line features)")
    print(f"  - arc_transformation_features.csv (per-example transformation classification)")
    print(f"  - arc_efe_prompts.csv (natural language prompts for EFE agent)")
    print("="*60)

if __name__ == "__main__":
    main()