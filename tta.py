# -*- coding: utf-8 -*-
"""
Test-time Adaptation Loop for ARC Challenge with Meta-Learning
Based on:
- Expected Free Energy framework
- Titans: Learning to Memorize at Test Time
- Meta-learned adapters with surprise-based memory gating

Key Components:
1. Small adapters/router logits for solver selection
2. Likelihood heads P(o|s) adaptation  
3. Memory module with surprise-based gating
4. Gradient magnitude controls memory writes
5. Decay mechanism for capacity management
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import math

from loss_function import ARCPromptGuidedAgent, EFELoss

class SurpriseBasedMemory(nn.Module):
    """
    Memory module that prioritizes "surprising" cases based on gradient magnitude.
    Implements Titans' approach: surprise gates memory writes, decay manages capacity.
    """
    
    def __init__(self,
                 memory_size: int = 1000,
                 feature_dim: int = 256,
                 surprise_threshold: float = 0.65,  # Adjusted for tanh-normalized surprise (0-1 range)
                 decay_rate: float = 0.95):
        super().__init__()
        
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.surprise_threshold = surprise_threshold
        self.decay_rate = decay_rate
        
        # Memory bank: [memory_size, feature_dim]
        self.register_buffer('memory_bank', torch.zeros(memory_size, feature_dim))
        self.register_buffer('memory_keys', torch.zeros(memory_size, feature_dim))
        self.register_buffer('memory_values', torch.zeros(memory_size, feature_dim))
        self.register_buffer('memory_ages', torch.zeros(memory_size))
        self.register_buffer('memory_surprise', torch.zeros(memory_size))
        self.register_buffer('memory_occupied', torch.zeros(memory_size, dtype=torch.bool))
        
        # Memory write/read mechanisms
        self.key_encoder = nn.Linear(feature_dim, feature_dim)
        self.value_encoder = nn.Linear(feature_dim, feature_dim)
        
        # Ensure attention dimension is divisible by num_heads
        attention_dim = ((feature_dim + 7) // 8) * 8  # Round up to nearest multiple of 8
        self.feature_projector = nn.Linear(feature_dim, attention_dim)
        self.attention = nn.MultiheadAttention(attention_dim, 8, batch_first=True)
        self.output_projector = nn.Linear(attention_dim, feature_dim)
        
    def compute_surprise(self, features: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
        """
        Compute surprise based on gradient magnitude and feature novelty.

        Args:
            features: [B, D] - Input features
            gradients: [B, D] - Gradients w.r.t. features

        Returns:
            surprise: [B] - Surprise scores
        """
        # Normalize gradients
        grad_normalized = gradients / (gradients.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        grad_magnitude = grad_normalized.norm(dim=-1)

        # Feature novelty component (distance to closest memory)
        # Encode queries to match the space of memory keys
        query_encoded = self.key_encoder(features)  # [B, D]

        if self.memory_occupied.any():
            occupied_memory = self.memory_keys[self.memory_occupied]  # [M, D]
            distances = torch.cdist(query_encoded, occupied_memory)  # [B, M]
            min_distances = distances.min(dim=-1)[0]  # [B]
        else:
            min_distances = torch.ones(features.shape[0], device=features.device)

        # Normalize distances for scale invariance
        min_distances = min_distances / (min_distances.mean() + 1e-8)

        # Combined surprise score: normalized gradient magnitude * normalized distance
        surprise = (grad_magnitude * min_distances).clamp(max=5.0).tanh()
        return surprise
    
    def should_memorize(self, surprise: torch.Tensor) -> torch.Tensor:
        """
        Determine which examples should be memorized based on surprise.
        
        Args:
            surprise: [B] - Surprise scores
            
        Returns:
            mask: [B] - Boolean mask for memorization
        """
        return surprise > self.surprise_threshold
    
    def write_memory(self, features: torch.Tensor, surprise: torch.Tensor):
        """
        Write surprising examples to memory with capacity management.
        
        Args:
            features: [B, D] - Features to potentially memorize
            surprise: [B] - Surprise scores
        """
        memorize_mask = self.should_memorize(surprise)
        
        if not memorize_mask.any():
            return
        
        features_to_store = features[memorize_mask]
        surprise_to_store = surprise[memorize_mask]
        
        for feat, surp in zip(features_to_store, surprise_to_store):
            self._write_single_memory(feat, surp)
    
    def _write_single_memory(self, feature: torch.Tensor, surprise: float):
        """Write a single feature to memory."""
        # Find available slot or evict based on age and surprise
        if not self.memory_occupied.all():
            # Find first empty slot
            slot_idx = (~self.memory_occupied).nonzero(as_tuple=True)[0][0]
        else:
            # Evict least surprising and oldest memory
            decay_factor = torch.pow(
                torch.full_like(self.memory_ages, self.decay_rate),
                self.memory_ages
            )
            eviction_scores = self.memory_surprise * decay_factor
            slot_idx = eviction_scores.argmin().item()
        
        # Write to memory
        self.memory_keys[slot_idx] = self.key_encoder(feature)
        self.memory_values[slot_idx] = self.value_encoder(feature)
        self.memory_ages[slot_idx] = 0
        self.memory_surprise[slot_idx] = surprise
        self.memory_occupied[slot_idx] = True
    
    def read_memory(self, query_features: torch.Tensor) -> torch.Tensor:
        """
        Read from memory using attention mechanism.
        
        Args:
            query_features: [B, D] - Query features
            
        Returns:
            retrieved: [B, D] - Retrieved memory content
        """
        if not self.memory_occupied.any():
            return torch.zeros_like(query_features)
        
        # Get occupied memory
        occupied_keys = self.memory_keys[self.memory_occupied]  # [M, D]
        occupied_values = self.memory_values[self.memory_occupied]  # [M, D]
        
        # Project to attention dimension
        query_proj = self.feature_projector(query_features).unsqueeze(1)  # [B, 1, attention_dim]
        keys_proj = self.feature_projector(occupied_keys).unsqueeze(0).expand(query_features.shape[0], -1, -1)  # [B, M, attention_dim]
        values_proj = self.feature_projector(occupied_values).unsqueeze(0).expand(query_features.shape[0], -1, -1)  # [B, M, attention_dim]
        
        # Attention-based retrieval
        retrieved_proj, _ = self.attention(query_proj, keys_proj, values_proj)  # [B, 1, attention_dim]
        
        # Project back to original dimension
        retrieved = self.output_projector(retrieved_proj.squeeze(1))  # [B, D]
        return retrieved
    
    def update_ages(self):
        """Update memory ages and apply decay."""
        self.memory_ages += 1
        # Apply decay to surprise scores
        self.memory_surprise *= self.decay_rate


class MetaAdapter(nn.Module):
    """
    Small adapter module for meta-learning test-time adaptation.
    """
    
    def __init__(self, input_dim: int, bottleneck_dim: int = 64):
        super().__init__()
        
        self.bottleneck_dim = bottleneck_dim
        
        # Down-projection
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        
        # Up-projection  
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)
        
        # Initialize to near-identity
        nn.init.zeros_(self.down_proj.weight)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adapter transformation."""
        return x + self.up_proj(F.relu(self.down_proj(x)))


class SolverRouter(nn.Module):
    """
    Router for selecting among multiple solvers based on problem characteristics.
    """
    
    def __init__(self, 
                 feature_dim: int,
                 num_solvers: int = 4,
                 router_dim: int = 128):
        super().__init__()
        
        self.num_solvers = num_solvers
        
        # Problem encoder
        self.problem_encoder = nn.Sequential(
            nn.Linear(feature_dim, router_dim),
            nn.ReLU(),
            nn.Linear(router_dim, router_dim),
            nn.ReLU()
        )
        
        # Router logits
        self.router_logits = nn.Linear(router_dim, num_solvers)
        
        # Adapters for router weights
        self.router_adapter = MetaAdapter(router_dim)
        
    def forward(self, problem_features: torch.Tensor) -> torch.Tensor:
        """
        Route to appropriate solver.
        
        Args:
            problem_features: [B, D] - Problem representation
            
        Returns:
            routing_weights: [B, num_solvers] - Soft routing weights
        """
        encoded = self.problem_encoder(problem_features)
        adapted = self.router_adapter(encoded)
        logits = self.router_logits(adapted)
        return F.softmax(logits, dim=-1)


class AdaptiveLikelihoodHead(nn.Module):
    """
    Adaptive likelihood head P(o|s) that can be adapted at test time.
    """
    
    def __init__(self, 
                 state_dim: int,
                 obs_dim: int,
                 hidden_dim: int = 256):
        super().__init__()
        
        # Base likelihood model
        self.base_model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
        
        # Adapters for each layer
        self.adapters = nn.ModuleList([
            MetaAdapter(hidden_dim),
            MetaAdapter(hidden_dim)
        ])
    
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute P(o|s) with adaptation.
        
        Args:
            states: [B, state_dim] - State representations
            
        Returns:
            likelihoods: [B, obs_dim] - Observation likelihoods
        """
        x = states
        
        # First layer + adapter
        x = F.relu(self.base_model[0](x))
        x = self.adapters[0](x)
        
        # Second layer + adapter  
        x = F.relu(self.base_model[2](x))
        x = self.adapters[1](x)
        
        # Output layer
        x = self.base_model[4](x)
        
        return F.softmax(x, dim=-1)


class TestTimeAdaptationSystem(nn.Module):
    """
    Complete test-time adaptation system for ARC challenge.
    Combines meta-learning, surprise-based memory, and adaptive components.
    """
    
    def __init__(self,
                 base_agent: ARCPromptGuidedAgent,
                 memory_size: int = 1000,
                 num_solvers: int = 4,
                 adaptation_steps: int = 5,
                 adaptation_lr: float = 1e-3):
        super().__init__()
        
        self.base_agent = base_agent
        self.adaptation_steps = adaptation_steps
        self.adaptation_lr = adaptation_lr
        
        # Calculate actual feature dimension based on problem features
        # num_colors * 4 (input/target stats) + hidden_dim (prompt) + 2 (grid size)
        actual_feature_dim = base_agent.num_colors * 4 + base_agent.hidden_dim + 2
        
        # Surprise-based memory
        self.memory = SurpriseBasedMemory(
            memory_size=memory_size,
            feature_dim=actual_feature_dim
        )
        
        # Solver router
        self.router = SolverRouter(
            feature_dim=actual_feature_dim,  # same as memory feature dim
            num_solvers=num_solvers
        )
        
        # Adaptive likelihood heads
        self.likelihood_heads = nn.ModuleList([
            AdaptiveLikelihoodHead(
                state_dim=base_agent.hidden_dim,
                obs_dim=base_agent.num_colors
            ) for _ in range(num_solvers)
        ])
        
        # Meta-learned adapters for base agent
        self.agent_adapters = nn.ModuleDict({
            'forward_adapter': MetaAdapter(base_agent.hidden_dim),
            'backward_adapter': MetaAdapter(base_agent.hidden_dim),
            'prompt_adapter': MetaAdapter(base_agent.prompt_dim)
        })
    
    def extract_problem_features(self, 
                                initial_state: torch.Tensor,
                                target_state: torch.Tensor,
                                prompt_embedding: torch.Tensor) -> torch.Tensor:
        """Extract features representing the problem for routing and memory."""
        H, W = initial_state.shape
        
        # Grid statistics
        input_onehot = F.one_hot(initial_state, num_classes=self.base_agent.num_colors).float()
        target_onehot = F.one_hot(target_state, num_classes=self.base_agent.num_colors).float()
        
        input_stats = torch.cat([
            input_onehot.mean(dim=[0, 1]),  # Color distribution
            input_onehot.std(dim=[0, 1])    # Color variance
        ])
        
        target_stats = torch.cat([
            target_onehot.mean(dim=[0, 1]),
            target_onehot.std(dim=[0, 1])
        ])
        
        # Combine with prompt features (first hidden_dim elements)
        prompt_features = prompt_embedding[:self.base_agent.hidden_dim]
        
        # Concatenate all features
        problem_features = torch.cat([
            input_stats,
            target_stats,
            prompt_features,
            torch.tensor([H, W], dtype=torch.float32, device=initial_state.device)
        ])
        
        return problem_features
    
    def compute_surprise_gradients(self,
                                 problem_features: torch.Tensor,
                                 loss: torch.Tensor) -> torch.Tensor:
        """Compute gradients for surprise estimation."""
        if problem_features.requires_grad:
            gradients = torch.autograd.grad(
                loss, problem_features,
                retain_graph=True,
                create_graph=False,
                only_inputs=True,
                allow_unused=True
            )[0]
        else:
            # If no gradients available, use random as fallback
            gradients = torch.randn_like(problem_features) * 0.1
        
        if gradients is None:
            gradients = torch.randn_like(problem_features) * 0.1
            
        return gradients
    
    def test_time_adapt(self,
                       initial_state: torch.Tensor,
                       target_state: torch.Tensor,
                       prompt_text: str,
                       prompt_embedding: torch.Tensor,
                       support_examples: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None) -> Dict[str, torch.Tensor]:
        """
        Perform test-time adaptation on a new problem.
        
        Args:
            initial_state: [H, W] - Input grid
            target_state: [H, W] - Target grid (for evaluation only)
            prompt_text: Natural language description
            prompt_embedding: [D] - Encoded prompt
            support_examples: Optional few-shot examples for adaptation
            
        Returns:
            adaptation_results: Dictionary with predictions and adaptation info
        """
        results = {}
        
        # Extract problem features
        problem_features = self.extract_problem_features(
            initial_state, target_state, prompt_embedding
        )
        problem_features.requires_grad_(True)
        
        # Route to appropriate solver
        routing_weights = self.router(problem_features.unsqueeze(0))  # [1, num_solvers]
        selected_solver = routing_weights.argmax(dim=-1).item()
        results['selected_solver'] = selected_solver
        results['routing_weights'] = routing_weights
        
        # Read from memory
        memory_content = self.memory.read_memory(problem_features.unsqueeze(0))  # [1, D]
        results['memory_retrieved'] = memory_content.norm().item()
        
        # Adaptation loop
        adaptation_losses = []

        for step in range(self.adaptation_steps):
            # Apply prompt adapter before forward planning
            adapted_prompt = self.agent_adapters['prompt_adapter'](prompt_embedding[:self.base_agent.prompt_dim])

            # Forward pass with base agent
            forward_preds, critique_scores = self.base_agent.forward_planning(
                initial_state, adapted_prompt, num_steps=3
            )

            # Compute adaptation loss (unsupervised)
            step_loss = self._compute_adaptation_loss(
                forward_preds, critique_scores, selected_solver, problem_features
            )

            # Log loss scalar (detached)
            adaptation_losses.append(step_loss.item())

            # Compute gradients for surprise BEFORE adapting (while graph is intact)
            gradients = self.compute_surprise_gradients(problem_features, step_loss)
            surprise = self.memory.compute_surprise(
                problem_features.unsqueeze(0), gradients.unsqueeze(0)
            )

            # Adapt parameters (this modifies parameters in-place)
            # Must happen AFTER computing surprise, so graph is already detached
            self._adapt_parameters(step_loss)

            results[f'adaptation_loss_step_{step}'] = step_loss.item()
            results[f'surprise_step_{step}'] = surprise.item()

        # Final prediction with adapted prompt (using updated parameters)
        adapted_prompt_final = self.agent_adapters['prompt_adapter'](prompt_embedding[:self.base_agent.prompt_dim])
        final_preds, final_critique = self.base_agent.forward_planning(
            initial_state, adapted_prompt_final, num_steps=5
        )

        # Compute final loss with CURRENT adapted parameters (not from stored tensors)
        # This avoids graph issues from in-place parameter modifications
        final_loss = self._compute_adaptation_loss(
            final_preds, final_critique, selected_solver, problem_features
        )

        # Write to memory if surprising enough
        final_gradients = self.compute_surprise_gradients(problem_features, final_loss)
        final_surprise = self.memory.compute_surprise(
            problem_features.unsqueeze(0), final_gradients.unsqueeze(0)
        )

        self.memory.write_memory(problem_features.unsqueeze(0), final_surprise)
        self.memory.update_ages()
        
        results.update({
            'final_predictions': final_preds,
            'final_critique': final_critique,
            'final_surprise': final_surprise.item(),
            'adaptation_losses': adaptation_losses,
            'memory_size': self.memory.memory_occupied.sum().item()
        })
        
        return results
    
    def _compute_adaptation_loss(self,
                                predictions: torch.Tensor,
                                critique_scores: List[torch.Tensor],
                                solver_idx: int,
                                problem_features: torch.Tensor) -> torch.Tensor:
        """Compute unsupervised adaptation loss."""
        # Consistency loss: predictions should be self-consistent
        consistency_loss = torch.tensor(0.0, device=predictions.device)
        for t in range(len(predictions) - 1):
            consistency_loss += F.mse_loss(predictions[t], predictions[t+1])

        # Critique coherence: critique scores should be coherent
        critique_coherence = torch.tensor(0.0, device=predictions.device)
        if critique_scores:
            scores_tensor = torch.stack(critique_scores)  # [T, 3]
            coherence_loss = scores_tensor.var(dim=0).mean()  # Low variance = more coherent
            critique_coherence = coherence_loss

        # Solver-specific likelihood - use problem features instead of prediction features
        # Create state representation from problem features
        state_features = problem_features[:self.base_agent.hidden_dim].unsqueeze(0)  # [1, hidden_dim]
        solver_likelihood = self.likelihood_heads[solver_idx](state_features).mean()

        # Memory alignment loss: encourage predicted features to align with retrieved memory
        mem_retrieved = self.memory.read_memory(problem_features.unsqueeze(0)).squeeze(0).detach()  # [D]
        if mem_retrieved.norm() > 1e-6:  # Only if memory has content
            mem_align = 1.0 - F.cosine_similarity(problem_features, mem_retrieved, dim=0)  # Higher => misaligned
        else:
            mem_align = torch.tensor(0.0, device=problem_features.device)

        total_loss = consistency_loss + 0.1 * critique_coherence - 0.1 * solver_likelihood + 0.05 * mem_align
        return total_loss
    
    def _adapt_parameters(self, loss: torch.Tensor):
        """Adapt the meta-learned parameters (router adapter, agent adapters, and likelihood heads only)."""
        adapter_params = []

        # Only adapt router adapter, not entire router
        for p in self.router.router_adapter.parameters():
            adapter_params.append(p)

        # Agent adapters
        for m in self.agent_adapters.values():
            adapter_params += list(m.parameters())

        # Likelihood heads
        for head in self.likelihood_heads:
            adapter_params += list(head.parameters())

        if not adapter_params:
            return

        # Compute gradients
        gradients = torch.autograd.grad(
            loss, adapter_params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True
        )

        # Apply gradient-based adaptation with clipping for stability
        with torch.no_grad():
            for param, grad in zip(adapter_params, gradients):
                if grad is not None:
                    param -= self.adaptation_lr * grad

        # Clip gradients for safety
        torch.nn.utils.clip_grad_norm_(adapter_params, 1.0)


def create_test_adaptation_system():
    """Create a test-time adaptation system for experimentation."""
    # Create base agent
    base_agent = ARCPromptGuidedAgent(
        max_grid_size=10,
        num_colors=10,
        hidden_dim=256,
        prompt_dim=768
    )
    
    # Create adaptation system
    adaptation_system = TestTimeAdaptationSystem(
        base_agent=base_agent,
        memory_size=500,
        num_solvers=3,
        adaptation_steps=5,
        adaptation_lr=1e-3
    )
    
    return adaptation_system


def test_adaptation_system():
    """Test the adaptation system with sample problems."""
    print("Testing Test-Time Adaptation System...")
    
    # Create system
    system = create_test_adaptation_system()
    
    # Create sample problems
    problems = [
        (torch.randint(0, 3, (3, 3)), torch.randint(0, 3, (3, 3)), "Copy input pattern"),
        (torch.randint(0, 3, (4, 4)), torch.randint(0, 3, (4, 4)), "Mirror the input"),
        (torch.randint(0, 3, (5, 5)), torch.randint(0, 3, (5, 5)), "Fill empty spaces")
    ]
    
    # Simple prompt embedding
    def create_prompt_embedding(text):
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        torch.manual_seed(hash_val % 1000)
        return torch.randn(768)
    
    # Test adaptation on each problem
    for i, (input_grid, target_grid, prompt_text) in enumerate(problems):
        print(f"\nProblem {i+1}: {prompt_text}")
        print(f"Input shape: {input_grid.shape}")
        
        prompt_embedding = create_prompt_embedding(prompt_text)
        
        # Perform test-time adaptation
        results = system.test_time_adapt(
            input_grid, target_grid, prompt_text, prompt_embedding
        )
        
        print(f"Selected solver: {results['selected_solver']}")
        print(f"Final surprise: {results['final_surprise']:.4f}")
        print(f"Memory size: {results['memory_size']}")
        print(f"Adaptation losses: {[f'{loss:.4f}' for loss in results['adaptation_losses']]}")
    
    print("\nTest-time adaptation system test completed!")


if __name__ == "__main__":
    test_adaptation_system()