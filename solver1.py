# -*- coding: utf-8 -*-
"""
Solver 1: Contextual Memory with Surprise-Based Updates
- Fast, episodic memory for immediate context
- Surprise-gated memory writes based on gradient magnitude
- Temporal decay for contextual relevance
- Dynamic attention-based retrieval
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque, defaultdict
import math
import hashlib

from loss_function import ARCPromptGuidedAgent
from tta import SurpriseBasedMemory, MetaAdapter

class ContextualMemoryBank(nn.Module):
    """
    Fast contextual memory with surprise-based updates and pattern storage.
    Stores recent problem-solving episodes with temporal decay.
    NEW: Enhanced with pattern classification and surprise-gated memory writes.
    """

    def __init__(self,
                 context_size: int = 50,  # Smaller than permanent memory
                 feature_dim: int = 512,
                 surprise_threshold: float = 0.2,
                 temporal_decay: float = 0.9,
                 attention_heads: int = 8):
        super().__init__()

        self.context_size = context_size
        self.feature_dim = feature_dim
        self.surprise_threshold = surprise_threshold
        self.temporal_decay = temporal_decay
        self.attention_heads = attention_heads

        # Contextual memory buffers
        self.register_buffer('context_keys', torch.zeros(context_size, feature_dim))
        self.register_buffer('context_values', torch.zeros(context_size, feature_dim))
        self.register_buffer('context_timestamps', torch.zeros(context_size))
        self.register_buffer('context_surprise', torch.zeros(context_size))
        self.register_buffer('context_success', torch.zeros(context_size))  # Track success rate
        self.register_buffer('context_occupied', torch.zeros(context_size, dtype=torch.bool))
        self.register_buffer('write_pointer', torch.tensor(0, dtype=torch.long))

        # NEW: Pattern storage for known solutions
        self.known_patterns = {}  # Maps pattern hash to solution
        self.pattern_frequencies = defaultdict(int)  # Track pattern frequency
        self.pattern_success_rates = defaultdict(float)  # Track success rates per pattern

        # NEW: Pattern grouping and clustering
        self.pattern_groups = defaultdict(list)  # Group similar patterns by transformation type
        self.pattern_embedding_cache = {}  # Cache pattern embeddings for similarity
        
        # Memory encoding/decoding
        self.key_encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.value_encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Multi-head attention for retrieval
        self.retrieval_attention = nn.MultiheadAttention(
            feature_dim, attention_heads, batch_first=True
        )
        
        # Surprise computation network
        self.surprise_net = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),  # concat query + memory
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def _compute_pattern_hash(self, grid: torch.Tensor) -> str:
        """Compute deterministic hash of grid pattern."""
        grid_np = grid.cpu().numpy().flatten().tobytes()
        return hashlib.md5(grid_np).hexdigest()

    def store_known_pattern(self, input_grid: torch.Tensor, output_grid: torch.Tensor, success: bool):
        """
        Store a known pattern (input -> output mapping) for future retrieval.
        NEW: Also groups pattern with similar ones for surprise mitigation.
        """
        pattern_hash = self._compute_pattern_hash(input_grid)

        self.known_patterns[pattern_hash] = {
            'input': input_grid.cpu().clone().detach(),
            'output': output_grid.cpu().clone().detach(),
            'success': bool(success),
            'timestamp': len(self.known_patterns)
        }

        # Update frequency and success tracking
        self.pattern_frequencies[pattern_hash] += 1
        if success:
            old_rate = self.pattern_success_rates.get(pattern_hash, 0.0)
            freq = self.pattern_frequencies[pattern_hash]
            # Exponential moving average update
            new_rate = (old_rate * (freq - 1) + float(success)) / freq
            self.pattern_success_rates[pattern_hash] = new_rate

        # NEW: Group pattern with similar patterns
        self._group_pattern(pattern_hash, input_grid, output_grid)

    def retrieve_known_pattern(self, input_grid: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Retrieve exact match for known pattern, prioritizing high-success ones.
        Returns output grid if found and successful, None otherwise.
        """
        pattern_hash = self._compute_pattern_hash(input_grid)

        if pattern_hash in self.known_patterns:
            pattern_info = self.known_patterns[pattern_hash]
            if pattern_info['success']:
                return pattern_info['output'].to(input_grid.device)

        return None

    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics on stored patterns."""
        return {
            'total_patterns': len(self.known_patterns),
            'total_retrievals': sum(self.pattern_frequencies.values()),
            'avg_success_rate': np.mean(list(self.pattern_success_rates.values())) if self.pattern_success_rates else 0.0,
            'num_groups': len(self.pattern_groups),
            'frequent_patterns': sorted(
                [(k, self.pattern_frequencies[k]) for k in self.pattern_frequencies.keys()],
                key=lambda x: x[1], reverse=True
            )[:5]
        }

    def _compute_pattern_similarity(self, grid1: torch.Tensor, grid2: torch.Tensor) -> float:
        """
        Compute similarity between two grids for pattern grouping.
        NEW: Uses shape, color distribution, and transformation properties.
        """
        if grid1.shape != grid2.shape:
            return 0.0

        # Shape similarity (exact match)
        shape_sim = 1.0 if grid1.shape == grid2.shape else 0.5

        # Color distribution similarity
        colors1 = set(grid1.flatten().tolist())
        colors2 = set(grid2.flatten().tolist())
        intersection = len(colors1.intersection(colors2))
        union = len(colors1.union(colors2))
        color_sim = intersection / (union + 1e-8)

        # Structural similarity (do neighboring cells match?)
        if grid1.numel() > 1:
            diff = (grid1 != grid2).float()
            # Penalize scattered differences
            local_diff = F.max_pool2d(diff.unsqueeze(0).unsqueeze(0),
                                     kernel_size=3, stride=1, padding=1)
            structure_sim = 1.0 - local_diff.mean().item()
        else:
            structure_sim = 1.0

        # Weighted combination
        total_sim = 0.5 * shape_sim + 0.3 * color_sim + 0.2 * structure_sim
        return float(total_sim)

    def _group_pattern(self, pattern_hash: str, input_grid: torch.Tensor, output_grid: torch.Tensor):
        """
        Group pattern with similar patterns for surprise mitigation.
        NEW: Clusters patterns by transformation similarity.
        """
        # Find most similar existing group
        best_group_key = None
        best_similarity = 0.5  # Threshold for grouping

        for group_key in list(self.pattern_groups.keys()):
            if group_key in self.known_patterns:
                group_input = self.known_patterns[group_key]['input']
                similarity = self._compute_pattern_similarity(input_grid, group_input)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_group_key = group_key

        # Add to best group or create new group
        if best_group_key is None:
            # Create new group with this pattern as representative
            self.pattern_groups[pattern_hash] = [pattern_hash]
        else:
            # Add to existing group
            self.pattern_groups[best_group_key].append(pattern_hash)

    def retrieve_similar_patterns(self, input_grid: torch.Tensor, k: int = 3) -> List[Dict]:
        """
        NEW: Retrieve similar patterns from the same group for surprise mitigation.
        When facing a similar pattern, the model can use prior successful solutions.
        """
        similar_patterns = []

        # Find best matching group
        best_group_key = None
        best_similarity = 0.0

        for group_key in self.pattern_groups.keys():
            if group_key in self.known_patterns:
                group_input = self.known_patterns[group_key]['input']
                similarity = self._compute_pattern_similarity(input_grid, group_input)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_group_key = group_key

        # Retrieve top-k from best group
        if best_group_key is not None:
            group_patterns = self.pattern_groups[best_group_key][:k]

            for pattern_hash in group_patterns:
                if pattern_hash in self.known_patterns:
                    pattern_info = self.known_patterns[pattern_hash]
                    similar_patterns.append({
                        'input': pattern_info['input'],
                        'output': pattern_info['output'],
                        'similarity': best_similarity,
                        'success_rate': self.pattern_success_rates.get(pattern_hash, 0.0)
                    })

        return similar_patterns

    def compute_contextual_surprise(self,
                                   query_features: torch.Tensor,
                                   gradient_magnitude: torch.Tensor,
                                   input_grid: Optional[torch.Tensor] = None,
                                   output_grid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute surprise score based on novelty, gradient magnitude, and pattern recognition.
        NEW: Detects large input-output differences as HIGH SURPRISE - requires focused work.

        Args:
            query_features: [B, D] - Current problem features
            gradient_magnitude: [B] - Gradient magnitude from loss (should be detached)
            input_grid: [H, W] - Optional input grid for pattern-based surprise
            output_grid: [H, W] - Optional output grid to detect LARGE DIFFERENCES

        Returns:
            surprise: [B] - Contextual surprise scores [0, 1]
        """
        # NEW: Compute grid difference surprise (large changes = high surprise)
        grid_difference_surprise = torch.tensor(0.0, device=gradient_magnitude.device)
        if input_grid is not None and output_grid is not None:
            # Compute pixel-level difference ratio
            pixel_diff = (input_grid != output_grid).float().sum()
            total_pixels = input_grid.numel()
            diff_ratio = pixel_diff / (total_pixels + 1e-8)

            # Large differences (>30% of pixels) = HIGH SURPRISE
            # Small differences (<10% of pixels) = LOW SURPRISE
            if diff_ratio > 0.3:
                grid_difference_surprise = 0.9  # High surprise - needs focused work
            elif diff_ratio > 0.1:
                grid_difference_surprise = 0.6  # Medium surprise
            else:
                grid_difference_surprise = 0.2  # Low surprise - mostly preserved

        if not self.context_occupied.any():
            # No context yet, everything is surprising (but check patterns first)
            base_surprise = torch.ones_like(gradient_magnitude) * 0.8

            # NEW: Increase surprise significantly if large input-output differences
            if grid_difference_surprise > 0.5:
                base_surprise = base_surprise + (grid_difference_surprise * 0.3)  # Boost surprise

            # NEW: Reduce surprise if this is a known pattern
            if input_grid is not None:
                known_output = self.retrieve_known_pattern(input_grid)
                if known_output is not None:
                    # Known pattern with high success = low surprise
                    base_surprise = base_surprise * 0.3

            return torch.clamp(base_surprise, 0, 1.0)

        # Get active memories with temporal weighting
        active_mask = self.context_occupied
        active_keys = self.context_keys[active_mask]  # [M, D]
        active_timestamps = self.context_timestamps[active_mask]  # [M]

        # Ensure all tensors on same device
        device = query_features.device
        active_keys = active_keys.to(device)
        active_timestamps = active_timestamps.to(device)

        # Compute temporal weights (more recent = higher weight)
        current_time = torch.max(active_timestamps) + 1
        temporal_weights = torch.exp(-self.temporal_decay * (current_time - active_timestamps))

        # Compute novelty for each query
        novelty_scores = []
        pattern_surprises = []  # NEW: Track pattern-based surprises

        for i, query in enumerate(query_features):
            # Distance to all memories
            distances = torch.norm(query.unsqueeze(0) - active_keys, dim=1)  # [M]

            # Normalize distances to [0, 1]
            max_dist = distances.max()
            if max_dist > 1e-8:
                distances = distances / (max_dist + 1e-8)

            # Weighted by temporal relevance
            weighted_distances = distances * temporal_weights

            # Minimum weighted distance = novelty (closer = less novel)
            # Invert so that larger distance = more novel
            min_weighted = torch.min(weighted_distances)
            novelty = 1.0 - min_weighted if len(weighted_distances) > 0 else torch.tensor(1.0, device=device)
            novelty_scores.append(novelty)

            # NEW: Pattern-based surprise (known patterns have lower surprise)
            pattern_surprise = 0.7  # Default: somewhat surprising
            if input_grid is not None and i == 0:  # Assuming single sample for pattern check
                known_output = self.retrieve_known_pattern(input_grid)
                if known_output is not None:
                    # Known pattern with proven success = low surprise
                    pattern_success_rate = self.pattern_success_rates.get(
                        self._compute_pattern_hash(input_grid), 0.0
                    )
                    pattern_surprise = (1.0 - pattern_success_rate) * 0.3  # Range: [0, 0.3]

            pattern_surprises.append(pattern_surprise)

        novelty_tensor = torch.stack(novelty_scores)
        pattern_surprise_tensor = torch.tensor(pattern_surprises, device=device)

        # Combine novelty with gradient magnitude with clamping
        # Ensure gradient_magnitude is detached (shouldn't propagate through memory)
        grad_mag_detached = gradient_magnitude.detach() if gradient_magnitude.requires_grad else gradient_magnitude
        grad_mag_normalized = torch.clamp(grad_mag_detached, 0, 1.0)  # Normalize gradients

        # NEW: Combine novelty-based, pattern-based, and grid-difference surprises
        surprise = grad_mag_normalized * novelty_tensor
        # Weight: 50% novelty*gradient, 25% pattern knowledge, 25% grid difference
        surprise = (0.5 * surprise + 0.25 * pattern_surprise_tensor +
                   0.25 * grid_difference_surprise)  # Grid difference boost
        surprise = torch.clamp(surprise, 0, 1.0)  # Surprise in [0, 1]

        return surprise
    
    def should_store(self, surprise: torch.Tensor) -> torch.Tensor:
        """Determine if examples should be stored in contextual memory."""
        return surprise > self.surprise_threshold
    
    def write_context(self, 
                     features: torch.Tensor,
                     surprise: torch.Tensor,
                     success_rate: torch.Tensor,
                     timestamp: Optional[torch.Tensor] = None):
        """
        Write to contextual memory with circular buffer.
        
        Args:
            features: [B, D] - Features to store
            surprise: [B] - Surprise scores
            success_rate: [B] - Success rate of the episode
            timestamp: [B] - Optional timestamps
        """
        store_mask = self.should_store(surprise)
        
        if not store_mask.any():
            return
        
        features_to_store = features[store_mask]
        surprise_to_store = surprise[store_mask]
        success_to_store = success_rate[store_mask]
        
        if timestamp is not None:
            timestamp_to_store = timestamp[store_mask]
        else:
            timestamp_to_store = torch.arange(
                len(features_to_store), 
                dtype=torch.float32, 
                device=features.device
            )
        
        for feat, surp, succ, ts in zip(features_to_store, surprise_to_store, success_to_store, timestamp_to_store):
            self._write_single_context(feat, surp, succ, ts)
    
    def _write_single_context(self,
                             feature: torch.Tensor,
                             surprise: float,
                             success_rate: float,
                             timestamp: float):
        """Write single item to contextual memory with circular buffer."""
        # Use circular buffer - overwrite oldest entry
        write_idx = self.write_pointer.item()

        # Ensure feature is on correct device
        device = self.context_keys.device
        feature = feature.to(device)

        # Encode and store
        with torch.no_grad():
            encoded_key = self.key_encoder(feature)
            encoded_value = self.value_encoder(feature)

        self.context_keys[write_idx] = encoded_key
        self.context_values[write_idx] = encoded_value
        self.context_surprise[write_idx] = surprise
        self.context_success[write_idx] = success_rate
        self.context_timestamps[write_idx] = timestamp
        self.context_occupied[write_idx] = True

        # Update write pointer IN-PLACE (circular)
        # Important: Use add_ instead of reassignment to avoid creating new tensor object
        self.write_pointer.add_(1)
        self.write_pointer.mul_((self.write_pointer < self.context_size).int())  # Reset if >= size
        # Alternative: self.write_pointer.copy_((self.write_pointer + 1) % self.context_size)
        # But above is more efficient with in-place ops
    
    def retrieve_context(self, 
                        query_features: torch.Tensor,
                        top_k: int = 5) -> Dict[str, torch.Tensor]:
        """
        Retrieve relevant context using attention mechanism.
        
        Args:
            query_features: [B, D] - Query features
            top_k: Number of top memories to retrieve
            
        Returns:
            retrieved_info: Dict with retrieved memories and metadata
        """
        if not self.context_occupied.any():
            return {
                'values': torch.zeros_like(query_features),
                'attention_weights': torch.zeros(query_features.shape[0], 1),
                'success_rates': torch.zeros(query_features.shape[0]),
                'surprise_scores': torch.zeros(query_features.shape[0])
            }
        
        # Get active memories
        active_mask = self.context_occupied
        active_keys = self.context_keys[active_mask]  # [M, D]
        active_values = self.context_values[active_mask]  # [M, D]
        active_success = self.context_success[active_mask]  # [M]
        active_surprise = self.context_surprise[active_mask]  # [M]
        active_timestamps = self.context_timestamps[active_mask]  # [M]
        
        # Apply temporal decay to attention
        current_time = torch.max(active_timestamps) + 1
        temporal_weights = torch.exp(-self.temporal_decay * (current_time - active_timestamps))
        
        # Prepare for attention
        query = query_features.unsqueeze(1)  # [B, 1, D]
        keys = active_keys.unsqueeze(0).expand(query_features.shape[0], -1, -1)  # [B, M, D]
        values = active_values.unsqueeze(0).expand(query_features.shape[0], -1, -1)  # [B, M, D]
        
        # Compute attention logits manually for better control
        # Standard dot-product attention: scores = Q @ K^T / sqrt(d)
        scale_factor = math.sqrt(active_keys.shape[-1])
        attention_logits = torch.bmm(query, keys.transpose(-2, -1)) / scale_factor  # [B, 1, M]
        attention_logits = attention_logits.squeeze(1)  # [B, M]

        # Add success-aware bias to attention (high success = more likely to retrieve)
        # Normalize success rates to [0, 1] range for bias
        success_bias = (active_success / (active_success.max() + 1e-8)).unsqueeze(0)  # [1, M]
        attention_logits = attention_logits + 0.5 * success_bias  # Bias strength = 0.5

        # Add temporal bias
        temporal_bias = torch.log(temporal_weights + 1e-8).unsqueeze(0)  # [1, M]
        attention_logits = attention_logits + temporal_bias

        # Apply softmax to get attention weights
        adjusted_attention = F.softmax(attention_logits, dim=-1)  # [B, M]

        # Recompose values with adjusted attention weights
        retrieved_values = torch.bmm(adjusted_attention.unsqueeze(1), values).squeeze(1)  # [B, D]

        # Compute weighted success and surprise
        weighted_success = torch.sum(adjusted_attention * active_success.unsqueeze(0), dim=-1)
        weighted_surprise = torch.sum(adjusted_attention * active_surprise.unsqueeze(0), dim=-1)
        
        return {
            'values': retrieved_values,
            'attention_weights': adjusted_attention,
            'success_rates': weighted_success,
            'surprise_scores': weighted_surprise,
            'memory_size': active_mask.sum().item()
        }
    
    def decay_memories(self):
        """Apply temporal decay to stored memories."""
        if self.context_occupied.any():
            # Decay surprise scores over time
            self.context_surprise *= self.temporal_decay
            self.context_success *= (1 - (1 - self.temporal_decay) * 0.1)  # Slower decay for success


class ContextualSolver(nn.Module):
    """
    Solver 1: Fast contextual reasoning with surprise-based memory updates.
    Optimized for immediate problem-solving context and rapid adaptation.
    """
    
    def __init__(self,
                 base_agent: ARCPromptGuidedAgent,
                 context_memory_size: int = 50,
                 adaptation_steps: int = 3,
                 learning_rate: float = 2e-3):
        super().__init__()
        
        self.base_agent = base_agent
        self.adaptation_steps = adaptation_steps
        self.learning_rate = learning_rate
        
        # Contextual memory bank
        self.contextual_memory = ContextualMemoryBank(
            context_size=context_memory_size,
            feature_dim=base_agent.hidden_dim,
            surprise_threshold=0.2,
            temporal_decay=0.9
        )
        
        # Fast adaptation modules
        self.rapid_adapters = nn.ModuleDict({
            'context_fusion': nn.Sequential(
                nn.Linear(base_agent.hidden_dim * 2, base_agent.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ),
            'planning_adapter': MetaAdapter(base_agent.hidden_dim),
            'critique_adapter': MetaAdapter(base_agent.hidden_dim)
        })
        
        # Context-aware objective predictor
        self.objective_predictor = nn.Sequential(
            nn.Linear(base_agent.hidden_dim + base_agent.prompt_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),  # Objective embedding
            nn.Tanh()
        )
        
    def extract_problem_context(self,
                               input_grid: torch.Tensor,
                               target_grid: torch.Tensor,
                               prompt_embedding: torch.Tensor) -> torch.Tensor:
        """Extract contextual features for memory operations."""
        # Grid-level features
        input_features = self._extract_grid_features(input_grid)
        target_features = self._extract_grid_features(target_grid)
        
        # Combine with prompt
        grid_context = torch.cat([input_features, target_features], dim=0)
        
        # Fuse with prompt embedding
        prompt_context = prompt_embedding[:self.base_agent.hidden_dim]  # Truncate if needed
        
        # Create contextual representation
        context_features = torch.cat([grid_context, prompt_context], dim=0)
        
        return context_features
    
    def _extract_grid_features(self, grid: torch.Tensor) -> torch.Tensor:
        """Extract features from a single grid."""
        H, W = grid.shape
        
        # One-hot encoding
        grid_onehot = F.one_hot(grid, num_classes=self.base_agent.num_colors).float()
        
        # Statistical features
        color_dist = grid_onehot.mean(dim=[0, 1])  # Color distribution
        color_var = grid_onehot.var(dim=[0, 1])    # Color variance
        
        # Spatial features
        spatial_features = torch.tensor([
            H, W,  # Grid size
            torch.unique(grid).numel(),  # Number of unique colors
            (grid == 0).float().mean(),  # Background ratio
        ], device=grid.device)
        
        # Combine features
        features = torch.cat([color_dist, color_var, spatial_features])
        
        # Pad/truncate to match hidden_dim
        if len(features) < self.base_agent.hidden_dim:
            padding = torch.zeros(self.base_agent.hidden_dim - len(features), device=grid.device)
            features = torch.cat([features, padding])
        else:
            features = features[:self.base_agent.hidden_dim]
        
        return features
    
    def solve_with_context(self,
                          input_grid: torch.Tensor,
                          target_grid: torch.Tensor,
                          prompt_text: str,
                          prompt_embedding: torch.Tensor) -> Dict[str, Any]:
        """
        Solve ARC problem using contextual memory and rapid adaptation.
        
        Args:
            input_grid: [H, W] - Input grid
            target_grid: [H, W] - Target grid (for training/evaluation)
            prompt_text: Natural language problem description
            prompt_embedding: [D] - Encoded prompt
            
        Returns:
            solution_results: Dict with predictions, context info, and adaptation metrics
        """
        results = {}
        
        # Extract problem context
        context_features = self.extract_problem_context(input_grid, target_grid, prompt_embedding)
        context_features.requires_grad_(True)
        
        # Retrieve relevant context
        context_info = self.contextual_memory.retrieve_context(context_features.unsqueeze(0))
        retrieved_context = context_info['values'].squeeze(0)  # [D]
        
        results['context_retrieved'] = context_info['memory_size']
        results['context_success_rate'] = context_info['success_rates'].item()
        results['context_surprise'] = context_info['surprise_scores'].item()
        
        # Fuse current problem with retrieved context
        fused_features = self.rapid_adapters['context_fusion'](
            torch.cat([context_features, retrieved_context], dim=0)
        )
        
        # Predict objective embedding from context
        objective_embedding = self.objective_predictor(
            torch.cat([fused_features, prompt_embedding[:self.base_agent.hidden_dim]], dim=0)
        )
        
        # Rapid adaptation loop
        adaptation_losses = []
        success_rate = 0.0
        
        for step in range(self.adaptation_steps):
            # Context-guided forward planning
            forward_preds, attended_features, critique_scores = self._context_guided_planning(
                input_grid, prompt_embedding, fused_features, step
            )

            # Compute adaptation loss
            step_loss = self._compute_contextual_loss(
                forward_preds, critique_scores, target_grid, objective_embedding
            )

            adaptation_losses.append(step_loss.item())

            # Rapid parameter adaptation with gradient clipping
            self._rapid_adapt(step_loss)
            
            # Evaluate success (simple heuristic)
            if len(forward_preds) > 0:
                final_pred = forward_preds[-1].argmax(dim=-1)
                success_rate = (final_pred == target_grid).float().mean().item()
            
            results[f'step_{step}_loss'] = step_loss.item()
            results[f'step_{step}_success'] = success_rate
        
        # Final prediction
        final_forward_preds, final_critique = self._context_guided_planning(
            input_grid, prompt_embedding, fused_features, self.adaptation_steps
        )
        
        final_prediction = final_forward_preds[-1].argmax(dim=-1) if len(final_forward_preds) > 0 else input_grid
        final_success = (final_prediction == target_grid).float().mean().item()
        
        # Compute surprise for memory update
        if len(adaptation_losses) > 0:
            final_loss = torch.tensor(adaptation_losses[-1], requires_grad=True)
            gradients = torch.autograd.grad(
                final_loss, context_features, 
                retain_graph=True, create_graph=False, allow_unused=True
            )[0]
            
            if gradients is not None:
                gradient_magnitude = torch.norm(gradients)
            else:
                gradient_magnitude = torch.tensor(0.1)
            
            surprise = self.contextual_memory.compute_contextual_surprise(
                context_features.unsqueeze(0), gradient_magnitude.unsqueeze(0)
            )
        else:
            surprise = torch.tensor([0.5])
        
        # Update contextual memory
        self.contextual_memory.write_context(
            context_features.unsqueeze(0),
            surprise,
            torch.tensor([final_success]),
            torch.tensor([float(self.contextual_memory.write_pointer.item())])
        )

        # NEW: Store known pattern for future retrieval
        self.contextual_memory.store_known_pattern(
            input_grid, final_prediction, final_success > 0.8
        )

        # Apply temporal decay
        self.contextual_memory.decay_memories()
        
        results.update({
            'final_prediction': final_prediction,
            'final_success_rate': final_success,
            'final_surprise': surprise.item(),
            'adaptation_losses': adaptation_losses,
            'objective_embedding': objective_embedding,
            'prompt_text': prompt_text
        })
        
        return results
    
    def _context_guided_planning(self,
                                input_grid: torch.Tensor,
                                prompt_embedding: torch.Tensor,
                                context_features: torch.Tensor,
                                step: int) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Context-guided forward planning with adaptation.
        Applies adapters in feature space (before prediction heads) for proper dimensionality.
        """
        # Apply context adaptation to prompt
        adapted_prompt = prompt_embedding + 0.1 * context_features[:len(prompt_embedding)]

        # Get forward predictions with context influence
        # Note: We need access to attended_features before the final prediction head
        # This requires slight modification to base_agent.forward_planning
        forward_preds, attended_features, critique_scores = self.base_agent.forward_planning_with_features(
            input_grid, adapted_prompt, num_steps=3
        )

        # Apply planning adapter to attended features (in feature space [hidden_dim, H, W])
        if len(attended_features) > 0:
            adapted_features = []
            for feat in attended_features:  # feat is [hidden_dim, H, W]
                # Reshape for adapter: [hidden_dim, H*W]
                feat_reshaped = feat.view(feat.shape[0], -1).transpose(0, 1)  # [H*W, hidden_dim]
                adapted_feat = self.rapid_adapters['planning_adapter'](feat_reshaped)  # [H*W, hidden_dim]
                adapted_feat = adapted_feat.transpose(0, 1).view_as(feat)  # Back to [hidden_dim, H, W]
                adapted_features.append(adapted_feat)
            attended_features = adapted_features

        # Apply critique adapter to critique scores [3] -> [3]
        if critique_scores:
            adapted_critique = []
            for score in critique_scores:
                # Critique scores are [3], expand to match hidden_dim, adapt, then extract relevant dims
                score_expanded = F.pad(score, (0, self.base_agent.hidden_dim - len(score)), value=0)
                adapted_score = self.rapid_adapters['critique_adapter'](score_expanded)
                # Keep only the first 3 dimensions
                adapted_critique.append(adapted_score[:3])
            critique_scores = adapted_critique

        return forward_preds, attended_features, critique_scores
    
    def _compute_contextual_loss(self,
                                forward_preds: torch.Tensor,
                                critique_scores: List[torch.Tensor],
                                target_grid: torch.Tensor,
                                objective_embedding: torch.Tensor) -> torch.Tensor:
        """Compute loss for contextual adaptation."""
        total_loss = torch.tensor(0.0, device=forward_preds.device)
        
        # Prediction accuracy
        if len(forward_preds) > 0:
            final_pred = forward_preds[-1]
            target_onehot = F.one_hot(target_grid, num_classes=final_pred.shape[-1]).float()
            prediction_loss = F.cross_entropy(
                final_pred.view(-1, final_pred.shape[-1]),
                target_grid.view(-1)
            )
            total_loss += prediction_loss
        
        # Critique coherence
        if critique_scores:
            critique_tensor = torch.stack(critique_scores)
            coherence_loss = critique_tensor.var(dim=0).mean()
            total_loss += 0.1 * coherence_loss
        
        # Objective alignment (encourage consistency with predicted objective)
        # Use temperature scaling to sharpen predictions for better alignment signal
        if len(forward_preds) > 0:
            temperature = 0.7
            # Apply temperature to final predictions for sharper features
            final_pred_soft = F.softmax(forward_preds[-1] / temperature, dim=-1)
            pred_features = final_pred_soft.mean(dim=[0, 1])  # [C] - spatially averaged, then temporally

            # Expand objective embedding to match color dimension
            objective_expanded = objective_embedding[:pred_features.shape[-1]]
            if len(objective_expanded) < pred_features.shape[-1]:
                padding = torch.zeros(pred_features.shape[-1] - len(objective_expanded),
                                    device=objective_expanded.device)
                objective_expanded = torch.cat([objective_expanded, padding])

            # Compute alignment loss with softened similarity for stability
            alignment_sim = F.cosine_similarity(
                pred_features.unsqueeze(0), objective_expanded.unsqueeze(0), dim=-1
            )
            alignment_loss = 1 - alignment_sim
            total_loss += 0.2 * alignment_loss
        
        return total_loss
    
    def _rapid_adapt(self, loss: torch.Tensor):
        """
        Rapid adaptation of contextual parameters with gradient clipping.
        Ensures stable inner-loop updates.
        """
        # Get adapter parameters
        adapter_params = []
        for adapter_name, adapter in self.rapid_adapters.items():
            adapter_params.extend(adapter.parameters())

        # Also adapt objective predictor
        adapter_params.extend(self.objective_predictor.parameters())

        # Compute gradients
        gradients = torch.autograd.grad(
            loss, adapter_params,
            retain_graph=True, create_graph=False, allow_unused=True
        )

        # Apply rapid adaptation with gradient clipping for stability
        with torch.no_grad():
            for param, grad in zip(adapter_params, gradients):
                if grad is not None:
                    # Clip gradient to prevent extreme updates
                    grad_clipped = torch.clamp(grad, -1.0, 1.0)
                    param -= self.learning_rate * grad_clipped


def test_contextual_solver():
    """Test the contextual solver with sample problems."""
    print("Testing Contextual Solver (exp_solver1)...")
    
    # Create base agent
    base_agent = ARCPromptGuidedAgent(
        max_grid_size=10,
        num_colors=5,
        hidden_dim=256,
        prompt_dim=512
    )
    
    # Create contextual solver
    solver = ContextualSolver(
        base_agent=base_agent,
        context_memory_size=20,
        adaptation_steps=3,
        learning_rate=1e-3
    )
    
    # Test problems with progressive difficulty
    problems = [
        (torch.randint(0, 3, (3, 3)), torch.randint(0, 3, (3, 3)), "Simple copy task"),
        (torch.randint(0, 3, (3, 3)), torch.randint(0, 3, (3, 3)), "Another copy task"),  # Similar context
        (torch.randint(0, 4, (4, 4)), torch.randint(0, 4, (4, 4)), "Larger grid copy"),
        (torch.randint(0, 2, (5, 5)), torch.randint(0, 2, (5, 5)), "Binary pattern"),
        (torch.randint(0, 3, (3, 3)), torch.randint(0, 3, (3, 3)), "Back to simple")  # Should retrieve context
    ]
    
    # Simple prompt embedding
    def create_prompt_embedding(text: str) -> torch.Tensor:
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        torch.manual_seed(hash_val % 1000)
        return torch.randn(512)
    
    # Solve problems sequentially to build context
    for i, (input_grid, target_grid, prompt_text) in enumerate(problems):
        print(f"\nProblem {i+1}: {prompt_text}")
        print(f"Input shape: {input_grid.shape}")
        
        prompt_embedding = create_prompt_embedding(prompt_text)
        
        # Solve with contextual memory
        results = solver.solve_with_context(
            input_grid, target_grid, prompt_text, prompt_embedding
        )
        
        print(f"Context retrieved: {results['context_retrieved']} memories")
        print(f"Context success rate: {results['context_success_rate']:.3f}")
        print(f"Final success: {results['final_success_rate']:.3f}")
        print(f"Final surprise: {results['final_surprise']:.3f}")
        print(f"Adaptation losses: {[f'{loss:.4f}' for loss in results['adaptation_losses']]}")
    
    print("\nContextual Solver test completed!")
    print("Memory should show increasing retrieval and context-aware adaptation.")


if __name__ == "__main__":
    test_contextual_solver()