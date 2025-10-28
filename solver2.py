"""
Permanent Memory Solver with DBSCAN-Style Problem Classification
Implements long-term memory storage and retrieval with clustering-based problem classification.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict, deque
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import json
import pickle
from typing import Dict, List, Tuple, Optional, Any

class ProblemObjectiveExtractor:
    """Extracts objectives and movement patterns from ARC problems."""
    
    def __init__(self):
        self.movement_patterns = {
            'translation': ['shift', 'move', 'translate'],
            'rotation': ['rotate', 'turn', 'pivot'],
            'reflection': ['mirror', 'flip', 'reflect'],
            'scaling': ['resize', 'scale', 'expand', 'shrink'],
            'completion': ['fill', 'complete', 'extend'],
            'filtering': ['remove', 'filter', 'select'],
            'grouping': ['group', 'cluster', 'organize'],
            'pattern': ['repeat', 'pattern', 'sequence']
        }
    
    def extract_movement_features(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> Dict[str, float]:
        """
        Extract movement pattern features from input-output pair.
        Features are extracted in a deterministic order for consistency.
        """
        features = {}

        # Basic grid properties
        features['size_change'] = float(torch.numel(output_grid) / torch.numel(input_grid))
        features['color_diversity_in'] = float(len(torch.unique(input_grid)))
        features['color_diversity_out'] = float(len(torch.unique(output_grid)))

        # Shape preservation
        if input_grid.shape == output_grid.shape:
            features['shape_preserved'] = 1.0
            features['pixel_change_ratio'] = float(torch.sum(input_grid != output_grid)) / torch.numel(input_grid)
        else:
            features['shape_preserved'] = 0.0
            features['pixel_change_ratio'] = 1.0

        # Color preservation
        input_colors = set(torch.unique(input_grid).tolist())
        output_colors = set(torch.unique(output_grid).tolist())
        total_colors = input_colors.union(output_colors)
        features['color_preservation'] = (
            len(input_colors.intersection(output_colors)) / len(total_colors)
            if len(total_colors) > 0 else 0.0
        )

        # Spatial correlation (if same shape) - with safe NaN handling
        if input_grid.shape == output_grid.shape:
            flat_in = input_grid.flatten().float()
            flat_out = output_grid.flatten().float()

            # Check variance before corrcoef to avoid NaN
            var_in = float(torch.var(flat_in))
            var_out = float(torch.var(flat_out))

            if var_in > 1e-8 and var_out > 1e-8:
                try:
                    corr_matrix = torch.corrcoef(torch.stack([flat_in, flat_out]))
                    corr_val = float(corr_matrix[0, 1])
                    features['spatial_correlation'] = corr_val if not np.isnan(corr_val) else 0.0
                except:
                    features['spatial_correlation'] = 0.0
            else:
                features['spatial_correlation'] = 0.0
        else:
            features['spatial_correlation'] = 0.0

        # Symmetry detection
        features['input_symmetry'] = self._detect_symmetry(input_grid)
        features['output_symmetry'] = self._detect_symmetry(output_grid)

        return features
    
    def _detect_symmetry(self, grid: torch.Tensor) -> float:
        """
        Detect symmetry in grid (horizontal, vertical, rotational).
        Checks for 90°, 180°, and 270° rotational symmetry if grid is square.
        """
        symmetry_score = 0.0

        # Horizontal symmetry (flip top-bottom)
        if torch.equal(grid, torch.flip(grid, [0])):
            symmetry_score += 0.25

        # Vertical symmetry (flip left-right)
        if torch.equal(grid, torch.flip(grid, [1])):
            symmetry_score += 0.25

        # Rotational symmetry (only for square grids)
        if grid.shape[0] == grid.shape[1]:
            # 90° rotational symmetry
            if torch.equal(grid, torch.rot90(grid, k=1)):
                symmetry_score += 0.25

            # 180° rotational symmetry
            elif torch.equal(grid, torch.rot90(grid, k=2)):
                symmetry_score += 0.20

            # 270° rotational symmetry
            elif torch.equal(grid, torch.rot90(grid, k=3)):
                symmetry_score += 0.25

        return min(symmetry_score, 1.0)  # Cap at 1.0

    def classify_objective(self, features: Dict[str, float]) -> str:
        """
        Classify problem objective based on extracted features.
        Uses tolerance-based comparisons instead of exact equality.
        """
        # Use tolerance for float comparisons (1e-6)
        TOLERANCE = 1e-6

        # Size change classification
        if abs(features['size_change'] - 1.0) > TOLERANCE:
            return 'scaling'
        # Shape transformation
        elif abs(features['shape_preserved'] - 0.0) < TOLERANCE:
            return 'transformation'
        # Minimal pixel changes
        elif features['pixel_change_ratio'] < 0.1:
            return 'minor_edit'
        # Color-focused transformations
        elif features['color_preservation'] < 0.5:
            return 'recoloring'
        # Complex spatial reorganization
        elif features['spatial_correlation'] < 0.3:
            return 'reconstruction'
        # Symmetry-based operations
        elif abs(features['input_symmetry'] - features['output_symmetry']) > 0.3:
            return 'symmetry_operation'
        # Default category
        else:
            return 'pattern_completion'

class PermanentMemoryBank:
    """
    Long-term memory storage with DBSCAN clustering for problem classification.
    Features are normalized for cosine similarity and stored efficiently.
    """

    # Fixed feature key order for deterministic construction
    FEATURE_KEYS = [
        'size_change', 'color_diversity_in', 'color_diversity_out',
        'shape_preserved', 'pixel_change_ratio', 'color_preservation',
        'spatial_correlation', 'input_symmetry', 'output_symmetry'
    ]

    def __init__(self, feature_dim: int = 256, max_memories: int = 10000):
        self.feature_dim = feature_dim
        self.max_memories = max_memories

        # Memory storage
        self.memories = []  # List of memory dictionaries
        self.feature_vectors = []  # Normalized feature vectors (numpy arrays on CPU)
        self.objectives = []  # Problem objectives
        self.success_rates = []  # Success tracking

        # Clustering with cosine metric
        self.clusterer = DBSCAN(eps=0.3, min_samples=3, metric='cosine')
        self.cluster_labels_ = None
        self.cluster_centers = {}
        self.cluster_objectives = {}
        self.clustering_dirty = True  # Flag to track if clustering needs update

        # Noise tracking
        self.noise_ratio = 0.0

        # Problem classifier
        self.objective_extractor = ProblemObjectiveExtractor()
        
    def store_memory(self,
                    problem_features: torch.Tensor,
                    solution_features: torch.Tensor,
                    input_grid: torch.Tensor,
                    output_grid: torch.Tensor,
                    success: bool,
                    metadata: Dict[str, Any] = None):
        """
        Store a problem-solution pair in permanent memory.
        Features are normalized for cosine similarity and tensors stored on CPU.
        """
        # Extract movement features and objective
        movement_features = self.objective_extractor.extract_movement_features(input_grid, output_grid)
        objective = self.objective_extractor.classify_objective(movement_features)

        # Create memory entry (store tensors on CPU to save memory)
        memory = {
            'problem_features': problem_features.cpu().clone().detach(),
            'solution_features': solution_features.cpu().clone().detach(),
            'input_grid': input_grid.cpu().clone().detach(),
            'output_grid': output_grid.cpu().clone().detach(),
            'movement_features': movement_features,
            'objective': objective,
            'success': success,
            'timestamp': len(self.memories),
            'metadata': metadata or {}
        }

        # Combine features in deterministic order for clustering
        # Uses fixed FEATURE_KEYS to ensure consistent ordering
        movement_vec = torch.tensor(
            [movement_features.get(key, 0.0) for key in self.FEATURE_KEYS],
            dtype=torch.float32
        )

        combined_features = torch.cat([
            problem_features.cpu().flatten(),
            movement_vec
        ])

        # Normalize for cosine similarity (L2 normalization)
        combined_features_np = combined_features.numpy()
        norm = np.linalg.norm(combined_features_np)
        if norm > 1e-8:
            combined_features_np = combined_features_np / norm

        self.memories.append(memory)
        self.feature_vectors.append(combined_features_np)
        self.objectives.append(objective)
        self.success_rates.append(float(success))

        # Mark clustering as dirty
        self.clustering_dirty = True

        # Maintain memory limit
        if len(self.memories) > self.max_memories:
            self._prune_memories()

        # Update clustering more frequently for smaller datasets
        cluster_update_freq = max(10, 50 - len(self.memories) // 100)
        if len(self.memories) % cluster_update_freq == 0:
            self._update_clusters()
    
    def retrieve_similar_problems(self,
                                 problem_features: torch.Tensor,
                                 input_grid: torch.Tensor,
                                 k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve similar problems from memory using clustering.
        Uses input-grid-only features to avoid output bias.
        Weights results by similarity, success rate, and cluster priors.
        """
        if len(self.memories) == 0:
            return []

        # Extract input-only movement features (use input grid twice to avoid output bias)
        # This keeps features like pixel_change_ratio neutral (always 0 when shapes match)
        current_movement = self.objective_extractor.extract_movement_features(
            input_grid, input_grid.clone()  # Use input as proxy, not zeros
        )
        current_objective = self.objective_extractor.classify_objective(current_movement)

        # Combine features in deterministic order
        movement_vec = torch.tensor(
            [current_movement.get(key, 0.0) for key in self.FEATURE_KEYS],
            dtype=torch.float32
        )

        combined_features = torch.cat([
            problem_features.cpu().flatten(),
            movement_vec
        ]).numpy().reshape(1, -1)

        # Normalize query features (same preprocessing as stored features)
        query_norm = np.linalg.norm(combined_features)
        if query_norm > 1e-8:
            combined_features = combined_features / query_norm

        # Ensure clustering is up-to-date
        if self.clustering_dirty and len(self.memories) >= 3:
            self._update_clusters()

        # Find similarities using cosine distance
        similarities = cosine_similarity(combined_features, self.feature_vectors)[0]

        # Apply objective and cluster priors
        if self.cluster_labels_ is not None:
            # Boost similarity for memories in same cluster
            current_cluster = self.cluster_labels_[-1] if len(self.cluster_labels_) > 0 else -1
            cluster_boost = np.array([
                1.5 if (self.cluster_labels_[i] == current_cluster and current_cluster != -1) else 1.0
                for i in range(len(self.cluster_labels_))
            ])
            similarities = similarities * cluster_boost

        # Apply objective filter with soft weighting
        objective_weight = np.array([
            1.0 if obj == current_objective else 0.7
            for obj in self.objectives
        ])
        similarities = similarities * objective_weight

        # Weight by success rate (successful memories prioritized)
        success_weight = np.array(self.success_rates) * 0.5 + 0.5  # Range [0.5, 1.0]
        similarities = similarities * success_weight

        # Get top-k candidates
        top_indices = np.argsort(similarities)[-k:][::-1]

        # Build result with softmax-weighted confidence
        valid_similarities = []
        similar_memories = []

        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum threshold
                memory = self.memories[idx].copy()
                memory['similarity'] = float(similarities[idx])
                similar_memories.append(memory)
                valid_similarities.append(similarities[idx])

        # Apply softmax weighting to confidence scores
        if valid_similarities:
            valid_similarities = np.array(valid_similarities)
            softmax_weights = np.exp(valid_similarities) / np.sum(np.exp(valid_similarities))
            for i, mem in enumerate(similar_memories):
                mem['confidence_weight'] = float(softmax_weights[i])

        return similar_memories
    
    def get_cluster_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about problem clusters.
        Includes noise ratio for monitoring clustering health.
        """
        if self.cluster_labels_ is None or len(self.memories) == 0:
            return {
                'total_clusters': 0,
                'noise_points': 0,
                'noise_ratio': 0.0,
                'cluster_sizes': {},
                'cluster_success_rates': {},
                'objective_distribution': defaultdict(int)
            }

        stats = {
            'total_clusters': len(set(self.cluster_labels_)) - (1 if -1 in self.cluster_labels_ else 0),
            'noise_points': sum(1 for label in self.cluster_labels_ if label == -1),
            'cluster_sizes': {},
            'cluster_success_rates': {},
            'cluster_objective_counts': defaultdict(lambda: defaultdict(int)),
            'objective_distribution': defaultdict(int)
        }

        # Calculate noise ratio
        stats['noise_ratio'] = stats['noise_points'] / max(len(self.memories), 1)

        # Calculate cluster statistics
        for label in set(self.cluster_labels_):
            if label == -1:
                continue

            cluster_indices = [i for i, l in enumerate(self.cluster_labels_) if l == label]
            stats['cluster_sizes'][label] = len(cluster_indices)

            # Success rate for this cluster
            cluster_successes = [self.success_rates[i] for i in cluster_indices]
            stats['cluster_success_rates'][label] = float(np.mean(cluster_successes))

            # Objective distribution within cluster
            for idx in cluster_indices:
                obj = self.objectives[idx]
                stats['cluster_objective_counts'][label][obj] += 1

        # Overall objective distribution
        for obj in self.objectives:
            stats['objective_distribution'][obj] += 1

        return stats
    
    def _update_clusters(self):
        """
        Update DBSCAN clustering with current memories.
        Caches cluster_labels_ and computes cluster statistics.
        """
        if len(self.feature_vectors) < 3:
            return

        try:
            # Perform clustering
            self.cluster_labels_ = self.clusterer.fit_predict(self.feature_vectors)

            # Update cluster centers and objectives
            self.cluster_centers = {}
            self.cluster_objectives = {}

            for label in set(self.cluster_labels_):
                if label == -1:  # Noise points
                    continue

                cluster_indices = [i for i, l in enumerate(self.cluster_labels_) if l == label]

                # Calculate cluster center
                cluster_features = [self.feature_vectors[i] for i in cluster_indices]
                self.cluster_centers[label] = np.mean(cluster_features, axis=0)

                # Determine dominant objective
                cluster_objectives = [self.objectives[i] for i in cluster_indices]
                most_common_obj = max(set(cluster_objectives), key=cluster_objectives.count)
                self.cluster_objectives[label] = most_common_obj

            # Mark clustering as clean
            self.clustering_dirty = False

            # Update noise ratio
            if len(self.cluster_labels_) > 0:
                self.noise_ratio = np.sum(self.cluster_labels_ == -1) / len(self.cluster_labels_)

        except Exception as e:
            # If clustering fails, mark as dirty for retry later
            self.clustering_dirty = True
    
    def _prune_memories(self):
        """Remove least useful memories to maintain size limit."""
        # Keep successful memories and recent memories
        keep_indices = []
        
        # Always keep successful memories
        for i, success in enumerate(self.success_rates):
            if success > 0.5:
                keep_indices.append(i)
        
        # Keep recent memories
        recent_start = max(0, len(self.memories) - self.max_memories // 2)
        for i in range(recent_start, len(self.memories)):
            if i not in keep_indices:
                keep_indices.append(i)
        
        # Randomly keep some older memories for diversity
        remaining_slots = self.max_memories - len(keep_indices)
        if remaining_slots > 0:
            older_indices = [i for i in range(recent_start) if i not in keep_indices]
            if older_indices:
                np.random.shuffle(older_indices)
                keep_indices.extend(older_indices[:remaining_slots])
        
        # Sort indices to maintain order
        keep_indices = sorted(set(keep_indices))
        
        # Prune all lists
        self.memories = [self.memories[i] for i in keep_indices]
        self.feature_vectors = [self.feature_vectors[i] for i in keep_indices]
        self.objectives = [self.objectives[i] for i in keep_indices]
        self.success_rates = [self.success_rates[i] for i in keep_indices]
    
    def save_memory_bank(self, filepath: str):
        """Save memory bank to disk."""
        data = {
            'memories': self.memories,
            'feature_vectors': self.feature_vectors,
            'objectives': self.objectives,
            'success_rates': self.success_rates,
            'cluster_centers': self.cluster_centers,
            'cluster_objectives': self.cluster_objectives
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load_memory_bank(self, filepath: str):
        """Load memory bank from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.memories = data['memories']
        self.feature_vectors = data['feature_vectors']
        self.objectives = data['objectives']
        self.success_rates = data['success_rates']
        self.cluster_centers = data.get('cluster_centers', {})
        self.cluster_objectives = data.get('cluster_objectives', {})

class PermanentSolver(nn.Module):
    """Solver with permanent memory and DBSCAN-style problem classification."""
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 512, max_grid_size: int = 30, num_colors: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_grid_size = max_grid_size
        self.num_colors = num_colors

        # Memory system
        self.memory_bank = PermanentMemoryBank(feature_dim=input_dim)

        # Core solver network
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Memory-guided adaptation
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Solution generator - outputs at max size, can be cropped to target
        self.solution_generator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_grid_size * max_grid_size * num_colors)
        )
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self,
                problem_features: torch.Tensor,
                input_grid: torch.Tensor,
                target_shape: Tuple[int, int] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with permanent memory guidance."""

        batch_size = problem_features.shape[0]
        device = problem_features.device

        # Encode problem features
        encoded_features = self.feature_encoder(problem_features)

        # Retrieve similar problems from memory as sequences (not averaged)
        memory_sequences = []
        memory_weights_list = []
        confidences = []

        for i in range(batch_size):
            similar_memories = self.memory_bank.retrieve_similar_problems(
                problem_features[i], input_grid[i], k=5
            )

            if similar_memories:
                # Use top-k memories as a sequence instead of averaging
                memory_features_list = []
                memory_weights = []

                for mem in similar_memories:
                    feat = self.feature_encoder(mem['solution_features'].to(device))
                    memory_features_list.append(feat)
                    # Weight by similarity and success, with confidence weight
                    weight = mem['similarity'] * (mem['success'] if isinstance(mem['success'], float) else float(mem['success']))
                    weight *= mem.get('confidence_weight', 1.0)
                    memory_weights.append(weight)

                # Stack memories as sequence [k, hidden_dim]
                memory_sequence = torch.stack(memory_features_list)
                memory_sequences.append(memory_sequence)

                # Normalize weights
                weight_sum = sum(memory_weights)
                if weight_sum > 0:
                    memory_weights = [w / weight_sum for w in memory_weights]
                memory_weights_list.append(torch.tensor(memory_weights, device=device))
                confidences.append(np.mean(memory_weights))
            else:
                # Create empty sequence: [0, hidden_dim]
                memory_sequences.append(torch.zeros(0, self.hidden_dim, device=device))
                memory_weights_list.append(torch.zeros(0, device=device))
                confidences.append(0.1)

        memory_confidence = torch.tensor(confidences, device=device)

        # Apply memory attention with sequences
        attended_features_list = []
        all_attention_weights = []

        for i in range(batch_size):
            query = encoded_features[i].unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]

            if memory_sequences[i].shape[0] > 0:
                # Use top-k sequence with attention
                key_values = memory_sequences[i].unsqueeze(0)  # [1, k, hidden_dim]

                attended, attn_weights = self.memory_attention(
                    query,
                    key_values,
                    key_values
                )
                attended_features_list.append(attended.squeeze(0).squeeze(0))
                all_attention_weights.append(attn_weights.squeeze(0))
            else:
                # No memories: use encoded feature as-is
                attended_features_list.append(encoded_features[i])
                all_attention_weights.append(torch.zeros(1, 0, device=device))

        attended_features = torch.stack(attended_features_list)

        # Combine with original features
        combined_features = torch.cat([encoded_features, attended_features], dim=-1)

        # Generate solution
        solution_logits = self.solution_generator(combined_features)

        # Estimate confidence
        solution_confidence = self.confidence_estimator(combined_features).squeeze(-1)

        # Adjust confidence by memory confidence
        final_confidence = solution_confidence * (0.5 + 0.5 * memory_confidence)

        # Reshape solution to grid format
        # First reshape to max size, then crop if needed
        solution_grid_full = solution_logits.view(
            batch_size, self.max_grid_size, self.max_grid_size, self.num_colors
        )

        if target_shape is None:
            target_shape = (self.max_grid_size, self.max_grid_size)

        # Crop to target shape if different from max size
        h, w = target_shape
        if h <= self.max_grid_size and w <= self.max_grid_size:
            solution_grid = solution_grid_full[:, :h, :w, :]
        else:
            # Target larger than max - pad with zeros (shouldn't normally happen)
            solution_grid = solution_grid_full

        return {
            'solution_grid': solution_grid,
            'confidence': final_confidence,
            'memory_guidance': torch.stack(attended_features_list),
            'attention_weights': all_attention_weights,
            'memory_confidence': memory_confidence
        }
    
    def update_memory(self,
                     problem_features: torch.Tensor,
                     solution_features: torch.Tensor,
                     input_grid: torch.Tensor,
                     output_grid: torch.Tensor,
                     success: bool,
                     metadata: Dict[str, Any] = None):
        """Update permanent memory with new problem-solution pair."""
        
        for i in range(problem_features.shape[0]):
            self.memory_bank.store_memory(
                problem_features[i],
                solution_features[i] if solution_features is not None else problem_features[i],
                input_grid[i],
                output_grid[i],
                success,
                metadata
            )
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory bank statistics."""
        return self.memory_bank.get_cluster_statistics()
    
    def save_solver(self, filepath: str):
        """Save solver state including memory bank."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'memory_bank_data': {
                'memories': self.memory_bank.memories,
                'feature_vectors': self.memory_bank.feature_vectors,
                'objectives': self.memory_bank.objectives,
                'success_rates': self.memory_bank.success_rates
            }
        }, filepath)
    
    def load_solver(self, filepath: str):
        """Load solver state including memory bank."""
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore memory bank
        memory_data = checkpoint['memory_bank_data']
        self.memory_bank.memories = memory_data['memories']
        self.memory_bank.feature_vectors = memory_data['feature_vectors']
        self.memory_bank.objectives = memory_data['objectives']
        self.memory_bank.success_rates = memory_data['success_rates']

def test_deterministic_features():
    """Test that feature vectors are deterministic across runs."""
    extractor = ProblemObjectiveExtractor()

    grid1 = torch.randint(0, 5, (5, 5))
    grid2 = grid1.clone()

    feat1 = extractor.extract_movement_features(grid1, grid1)
    feat2 = extractor.extract_movement_features(grid2, grid2)

    # Check deterministic construction using FEATURE_KEYS
    vec1 = [feat1.get(k) for k in PermanentMemoryBank.FEATURE_KEYS]
    vec2 = [feat2.get(k) for k in PermanentMemoryBank.FEATURE_KEYS]

    assert np.allclose(vec1, vec2), "Feature vectors should be identical"
    print("[OK] Deterministic features test passed")


def test_no_nan_in_features():
    """Test that constant grids produce valid features without NaN."""
    extractor = ProblemObjectiveExtractor()

    const_grid = torch.ones(5, 5, dtype=torch.long)
    features = extractor.extract_movement_features(const_grid, const_grid)

    for key, val in features.items():
        assert not np.isnan(val), f"{key} is NaN"
        assert not np.isinf(val), f"{key} is Inf"
        assert isinstance(val, float), f"{key} is not float"

    print("[OK] No NaN in features test passed")


def test_clustering_stability():
    """Test that DBSCAN clustering produces valid labels."""
    memory_bank = PermanentMemoryBank(feature_dim=256)

    # Store multiple memories (clustering updates every 10 items)
    for i in range(15):
        feats = torch.randn(256)
        grid = torch.randint(0, 5, (5, 5))
        success = np.random.random() > 0.3
        memory_bank.store_memory(feats, feats, grid, grid, success=success)

    # Manually force clustering if not done
    if memory_bank.cluster_labels_ is None:
        memory_bank._update_clusters()

    # Check clustering
    assert memory_bank.cluster_labels_ is not None, "Clustering should be done"
    assert len(memory_bank.cluster_labels_) == 15, "Should have labels for all memories"
    assert memory_bank.noise_ratio >= 0.0 and memory_bank.noise_ratio <= 1.0, "Noise ratio out of bounds"

    print("[OK] Clustering stability test passed")


def test_retrieval_similarity():
    """Test that retrieved memories have high similarity to themselves."""
    memory_bank = PermanentMemoryBank(feature_dim=256)

    # Store a memory
    base_grid = torch.randint(0, 5, (5, 5))
    feats = torch.randn(256)
    memory_bank.store_memory(feats, feats, base_grid, base_grid, success=True)

    # Retrieve with same grid
    retrieved = memory_bank.retrieve_similar_problems(feats, base_grid, k=1)

    assert len(retrieved) > 0, "Should retrieve at least one memory"
    assert retrieved[0]['similarity'] > 0.9, f"Self-similarity should be very high, got {retrieved[0]['similarity']}"

    print("[OK] Retrieval similarity test passed")


def test_cosine_normalization():
    """Test that normalized features have unit norm."""
    memory_bank = PermanentMemoryBank(feature_dim=256)

    feats = torch.randn(256)
    grid = torch.randint(0, 5, (5, 5))
    memory_bank.store_memory(feats, feats, grid, grid, success=True)

    # Check stored feature normalization
    stored_feat = memory_bank.feature_vectors[0]
    norm = np.linalg.norm(stored_feat)

    assert np.isclose(norm, 1.0, atol=1e-6), f"Norm should be ~1.0, got {norm}"

    print("[OK] Cosine normalization test passed")


def test_solution_shape_cropping():
    """Test that solution grids are properly cropped to target shape."""
    solver = PermanentSolver(input_dim=256, hidden_dim=256, max_grid_size=30)

    problem_features = torch.randn(2, 256)
    input_grids = torch.randint(0, 10, (2, 10, 10))

    # Test with smaller target shape
    results = solver(problem_features, input_grids, target_shape=(10, 10))
    assert results['solution_grid'].shape == (2, 10, 10, 10), f"Expected (2, 10, 10, 10), got {results['solution_grid'].shape}"

    # Test with max shape
    results = solver(problem_features, input_grids, target_shape=(30, 30))
    assert results['solution_grid'].shape == (2, 30, 30, 10), f"Expected (2, 30, 30, 10), got {results['solution_grid'].shape}"

    # Test with None (uses max)
    results = solver(problem_features, input_grids, target_shape=None)
    assert results['solution_grid'].shape == (2, 30, 30, 10), f"Expected (2, 30, 30, 10), got {results['solution_grid'].shape}"

    print("[OK] Solution shape cropping test passed")


def test_attention_with_sequences():
    """Test that attention operates on memory sequences correctly."""
    solver = PermanentSolver(input_dim=256, hidden_dim=256, max_grid_size=30)

    # Store some memories first
    for i in range(10):
        feats = torch.randn(256)
        grid = torch.randint(0, 5, (5, 5))
        solver.update_memory(feats.unsqueeze(0), feats.unsqueeze(0), grid.unsqueeze(0), grid.unsqueeze(0), success=True)

    # Test forward pass with populated memory
    problem_features = torch.randn(2, 256)
    input_grids = torch.randint(0, 5, (2, 5, 5))

    results = solver(problem_features, input_grids, target_shape=(5, 5))

    # Check that attention weights are lists (per-sample)
    assert isinstance(results['attention_weights'], list), "Attention weights should be a list"
    assert len(results['attention_weights']) == 2, "Should have attention weights for each sample"

    # Check confidence is computed
    assert results['confidence'].shape == (2,), f"Expected confidence shape (2,), got {results['confidence'].shape}"

    print("[OK] Attention with sequences test passed")


def test_permanent_solver():
    """Test the permanent solver implementation."""

    # Create solver
    solver = PermanentSolver(input_dim=256, hidden_dim=512)

    # Generate test data
    batch_size = 4
    problem_features = torch.randn(batch_size, 256)
    input_grids = torch.randint(0, 10, (batch_size, 10, 10))
    target_grids = torch.randint(0, 10, (batch_size, 10, 10))

    print("\nTesting Permanent Solver with DBSCAN Classification...")

    # Test forward pass
    results = solver(problem_features, input_grids, target_shape=(10, 10))

    print(f"Solution grid shape: {results['solution_grid'].shape}")
    print(f"Confidence shape: {results['confidence'].shape}")
    print(f"Memory confidence: {results['memory_confidence']}")

    # Test memory updates
    for i in range(10):
        success = np.random.random() > 0.3
        solver.update_memory(
            problem_features,
            problem_features,  # Using same as solution for test
            input_grids,
            target_grids,
            success,
            {'test_iteration': i}
        )

    # Check memory statistics
    stats = solver.get_memory_statistics()
    print(f"\nMemory Statistics:")
    print(f"Total memories: {len(solver.memory_bank.memories)}")
    print(f"Objective distribution: {dict(stats.get('objective_distribution', {}))}")

    print("Permanent solver test completed successfully!")

if __name__ == "__main__":
    print("Running comprehensive solver2 unit tests...\n")
    print("=" * 60)

    try:
        test_deterministic_features()
        test_no_nan_in_features()
        test_clustering_stability()
        test_retrieval_similarity()
        test_cosine_normalization()
        test_solution_shape_cropping()
        test_attention_with_sequences()
        test_permanent_solver()

        print("=" * 60)
        print("\n[SUCCESS] All unit tests passed!")

    except AssertionError as e:
        print(f"\n[FAILED] Test assertion failed: {e}")
        raise
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        raise