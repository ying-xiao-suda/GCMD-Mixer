"""
MiniMind-PPO Implementation with Proper Forward Propagation

This module implements a Proximal Policy Optimization (PPO) agent with proper
forward propagation for training. It addresses the key issues mentioned in the
problem statement:

1. Adds complete forward propagation in PPO updates
2. Implements proper policy ratio calculation
3. Fixes value function loss calculation
4. Removes random gradient injection
5. Improves action selection with real probabilities from model logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class MiniMindPPO(nn.Module):
    """
    MiniMind-PPO implementation with proper forward propagation.
    
    This class implements a PPO agent that addresses the missing forward
    propagation issues in training and provides proper policy ratio calculation.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_seq_length: int = 512,
        dropout: float = 0.1,
        clip_epsilon: float = 0.2,
        vf_coef: float = 0.5,
        entropy_coef: float = 0.01,
        learning_rate: float = 3e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super(MiniMindPPO, self).__init__()
        
        # Model configuration
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        self.dropout = dropout
        self.device = device
        
        # PPO hyperparameters
        self.clip_epsilon = clip_epsilon
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        
        # Model components
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_seq_length, hidden_size)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Output heads
        self.policy_head = nn.Linear(hidden_size, vocab_size)  # For action probabilities
        self.value_head = nn.Linear(hidden_size, 1)  # For value estimation
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        self.to(device)
        
    def _create_attention_mask(self, seq_length: int) -> torch.Tensor:
        """Create causal attention mask for autoregressive generation."""
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask.to(self.device)
    
    def _forward_for_training(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass specifically designed for training.
        
        Returns:
            logits: Raw logits from policy head [batch_size, seq_len, vocab_size]
            values: Value estimates [batch_size, seq_len, 1]
            hidden_states: Hidden representations [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_length = input_ids.shape
        
        # Token embeddings
        token_embeddings = self.embedding(input_ids)
        
        # Position embeddings
        position_ids = torch.arange(seq_length, device=self.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.pos_embedding(position_ids)
        
        # Combine embeddings
        hidden_states = token_embeddings + position_embeddings
        hidden_states = self.dropout_layer(hidden_states)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = self._create_attention_mask(seq_length)
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, src_mask=attention_mask)
        
        # Apply layer normalization
        hidden_states = self.layer_norm(hidden_states)
        
        # Get policy logits and values
        logits = self.policy_head(hidden_states)
        values = self.value_head(hidden_states)
        
        return logits, values, hidden_states
    
    def _batch_forward_pass(
        self, 
        batch_input_ids: torch.Tensor, 
        batch_attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Perform batch forward pass for multiple sequences.
        
        Args:
            batch_input_ids: [batch_size, seq_len]
            batch_attention_mask: Optional attention mask
            
        Returns:
            Dictionary containing logits, values, and hidden states
        """
        logits, values, hidden_states = self._forward_for_training(
            batch_input_ids, batch_attention_mask
        )
        
        return {
            'logits': logits,
            'values': values,
            'hidden_states': hidden_states
        }
    
    def _logits_to_action_probs(
        self, 
        logits: torch.Tensor, 
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert logits to action probabilities and log probabilities.
        
        Args:
            logits: Raw logits [batch_size, seq_len, vocab_size]
            temperature: Temperature for softmax scaling
            
        Returns:
            probs: Action probabilities [batch_size, seq_len, vocab_size]
            log_probs: Log probabilities [batch_size, seq_len, vocab_size]
        """
        # Apply temperature scaling
        scaled_logits = logits / temperature
        
        # Convert to probabilities
        probs = F.softmax(scaled_logits, dim=-1)
        log_probs = F.log_softmax(scaled_logits, dim=-1)
        
        return probs, log_probs
    
    def _extract_value_from_response(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Extract value estimates from hidden states.
        
        Args:
            hidden_states: Hidden representations [batch_size, seq_len, hidden_size]
            
        Returns:
            values: Value estimates [batch_size, seq_len]
        """
        values = self.value_head(hidden_states).squeeze(-1)
        return values
    
    def select_action(
        self, 
        input_ids: torch.Tensor, 
        deterministic: bool = False, 
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Select actions using the current policy.
        
        Args:
            input_ids: Input token sequences [batch_size, seq_len]
            deterministic: Whether to use deterministic action selection
            temperature: Temperature for sampling
            
        Returns:
            Dictionary containing actions, log_probs, values, and probs
        """
        with torch.no_grad():
            # Forward pass
            logits, values, _ = self._forward_for_training(input_ids)
            
            # Convert to probabilities
            probs, log_probs = self._logits_to_action_probs(logits, temperature)
            
            if deterministic:
                # Select most likely actions
                actions = torch.argmax(probs, dim=-1)
            else:
                # Sample from distribution
                actions = torch.multinomial(probs.view(-1, self.vocab_size), num_samples=1)
                actions = actions.view(input_ids.shape[0], input_ids.shape[1])
            
            # Get log probabilities of selected actions
            action_log_probs = torch.gather(log_probs, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
            
            return {
                'actions': actions,
                'log_probs': action_log_probs,
                'values': values.squeeze(-1),
                'probs': probs
            }
    
    def compute_policy_ratio(
        self, 
        old_log_probs: torch.Tensor, 
        new_log_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute policy ratio for PPO update.
        
        Args:
            old_log_probs: Log probabilities from old policy
            new_log_probs: Log probabilities from current policy
            
        Returns:
            ratio: Policy ratio (new_probs / old_probs)
        """
        ratio = torch.exp(new_log_probs - old_log_probs)
        return ratio
    
    def ppo_update(
        self,
        input_ids: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Perform PPO update with proper forward propagation.
        
        Args:
            input_ids: Input sequences [batch_size, seq_len]
            actions: Selected actions [batch_size, seq_len]
            old_log_probs: Log probabilities from old policy [batch_size, seq_len]
            old_values: Value estimates from old policy [batch_size, seq_len]
            returns: Computed returns [batch_size, seq_len]
            advantages: Computed advantages [batch_size, seq_len]
            attention_mask: Optional attention mask
            
        Returns:
            Dictionary containing loss components
        """
        # Forward pass with current policy
        logits, values, _ = self._forward_for_training(input_ids, attention_mask)
        
        # Convert logits to probabilities
        _, log_probs = self._logits_to_action_probs(logits)
        
        # Get log probabilities of taken actions
        action_log_probs = torch.gather(
            log_probs, dim=-1, index=actions.unsqueeze(-1)
        ).squeeze(-1)
        
        # Compute policy ratio
        ratio = self.compute_policy_ratio(old_log_probs, action_log_probs)
        
        # Compute policy loss with clipping
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(
            ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
        )
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
        
        # Compute value loss
        values = values.squeeze(-1)
        value_loss_unclipped = (values - returns) ** 2
        values_clipped = old_values + torch.clamp(
            values - old_values, -self.clip_epsilon, self.clip_epsilon
        )
        value_loss_clipped = (values_clipped - returns) ** 2
        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
        
        # Compute entropy loss
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        entropy_loss = -self.entropy_coef * entropy
        
        # Total loss
        total_loss = policy_loss + self.vf_coef * value_loss + entropy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
        
        # Update parameters
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'entropy': entropy.item(),
            'policy_ratio_mean': ratio.mean().item(),
            'policy_ratio_std': ratio.std().item()
        }
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Standard forward pass for compatibility."""
        logits, values, hidden_states = self._forward_for_training(input_ids, attention_mask)
        return {
            'logits': logits,
            'values': values,
            'hidden_states': hidden_states
        }
    
    def generate(
        self, 
        input_ids: torch.Tensor, 
        max_length: int = 50, 
        temperature: float = 1.0,
        do_sample: bool = True
    ) -> torch.Tensor:
        """
        Generate sequences using the current policy.
        
        Args:
            input_ids: Input sequences [batch_size, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            do_sample: Whether to sample or use greedy decoding
            
        Returns:
            generated_ids: Generated sequences [batch_size, seq_len + max_length]
        """
        self.eval()
        batch_size, seq_len = input_ids.shape
        generated_ids = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                outputs = self.select_action(
                    generated_ids, 
                    deterministic=not do_sample, 
                    temperature=temperature
                )
                
                # Get next token
                next_token = outputs['actions'][:, -1].unsqueeze(1)
                
                # Append to sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # Check for sequence length limit
                if generated_ids.shape[1] >= self.max_seq_length:
                    break
        
        self.train()
        return generated_ids


class PPOTrainer:
    """
    Trainer class for MiniMind-PPO with proper training procedures.
    """
    
    def __init__(
        self,
        model: MiniMindPPO,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        ppo_epochs: int = 4,
        num_mini_batches: int = 4,
        max_grad_norm: float = 0.5
    ):
        self.model = model
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_epochs = ppo_epochs
        self.num_mini_batches = num_mini_batches
        self.max_grad_norm = max_grad_norm
    
    def compute_gae(
        self, 
        rewards: torch.Tensor, 
        values: torch.Tensor, 
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Rewards [batch_size, seq_len]
            values: Value estimates [batch_size, seq_len]
            dones: Done flags [batch_size, seq_len]
            
        Returns:
            returns: Computed returns
            advantages: Computed advantages
        """
        batch_size, seq_len = rewards.shape
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_value = 0
                next_done = 1
            else:
                next_value = values[:, t + 1]
                next_done = dones[:, t + 1]
            
            delta = rewards[:, t] + self.gamma * next_value * (1 - next_done) - values[:, t]
            gae = delta + self.gamma * self.gae_lambda * (1 - next_done) * gae
            advantages[:, t] = gae
            returns[:, t] = advantages[:, t] + values[:, t]
        
        return returns, advantages
    
    def train_step(
        self,
        input_ids: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        old_log_probs: torch.Tensor,
        old_values: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform a complete PPO training step.
        
        Args:
            input_ids: Input sequences
            actions: Taken actions
            rewards: Received rewards
            dones: Episode termination flags
            old_log_probs: Log probabilities from old policy
            old_values: Value estimates from old policy
            
        Returns:
            Training statistics
        """
        # Compute returns and advantages
        returns, advantages = self.compute_gae(rewards, old_values, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_stats = {
            'total_loss': 0.0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy_loss': 0.0,
            'entropy': 0.0,
            'policy_ratio_mean': 0.0,
            'policy_ratio_std': 0.0
        }
        
        # Multiple PPO epochs
        for epoch in range(self.ppo_epochs):
            # Shuffle data
            batch_size = input_ids.shape[0]
            indices = torch.randperm(batch_size)
            
            # Mini-batch training
            mini_batch_size = batch_size // self.num_mini_batches
            
            for i in range(self.num_mini_batches):
                start_idx = i * mini_batch_size
                end_idx = (i + 1) * mini_batch_size
                mb_indices = indices[start_idx:end_idx]
                
                # Get mini-batch data
                mb_input_ids = input_ids[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_old_values = old_values[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]
                
                # PPO update
                stats = self.model.ppo_update(
                    mb_input_ids,
                    mb_actions,
                    mb_old_log_probs,
                    mb_old_values,
                    mb_returns,
                    mb_advantages
                )
                
                # Accumulate statistics
                for key in total_stats:
                    total_stats[key] += stats[key]
        
        # Average statistics
        num_updates = self.ppo_epochs * self.num_mini_batches
        for key in total_stats:
            total_stats[key] /= num_updates
        
        return total_stats


# Example usage and testing functions
def create_dummy_data(batch_size: int = 4, seq_len: int = 20, vocab_size: int = 1000):
    """Create dummy data for testing."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    actions = torch.randint(0, vocab_size, (batch_size, seq_len))
    rewards = torch.randn(batch_size, seq_len)
    dones = torch.zeros(batch_size, seq_len)
    old_log_probs = torch.randn(batch_size, seq_len)
    old_values = torch.randn(batch_size, seq_len)
    
    return input_ids, actions, rewards, dones, old_log_probs, old_values


def test_minimind_ppo():
    """Test the MiniMind-PPO implementation."""
    print("Testing MiniMind-PPO implementation...")
    
    # Create model
    model = MiniMindPPO(
        vocab_size=1000,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        max_seq_length=128
    )
    
    # Create trainer
    trainer = PPOTrainer(model)
    
    # Create dummy data
    input_ids, actions, rewards, dones, old_log_probs, old_values = create_dummy_data()
    
    # Test forward pass
    print("Testing forward pass...")
    outputs = model.forward(input_ids)
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Values shape: {outputs['values'].shape}")
    
    # Test action selection
    print("Testing action selection...")
    action_outputs = model.select_action(input_ids)
    print(f"Actions shape: {action_outputs['actions'].shape}")
    print(f"Log probs shape: {action_outputs['log_probs'].shape}")
    
    # Test PPO update
    print("Testing PPO update...")
    stats = trainer.train_step(input_ids, actions, rewards, dones, old_log_probs, old_values)
    print(f"Training stats: {stats}")
    
    # Test generation
    print("Testing generation...")
    generated = model.generate(input_ids[:1], max_length=10)
    print(f"Generated shape: {generated.shape}")
    
    print("All tests passed!")


if __name__ == "__main__":
    test_minimind_ppo()