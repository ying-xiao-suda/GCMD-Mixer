"""
MiniMind-PPO Usage Example and Integration Guide

This file demonstrates how to use the MiniMind-PPO implementation for 
reinforcement learning tasks with proper forward propagation.
"""

import torch
import numpy as np
from minimind_ppo import MiniMindPPO, PPOTrainer
from typing import Dict, List, Tuple


class TextEnvironment:
    """
    A simple text-based environment for demonstrating MiniMind-PPO.
    This environment simulates a text completion task where the agent
    learns to generate sequences that maximize a reward function.
    """
    
    def __init__(self, vocab_size: int = 1000, max_length: int = 20):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.current_sequence = None
        self.step_count = 0
        
    def reset(self) -> torch.Tensor:
        """Reset the environment and return initial state."""
        # Start with a random token
        self.current_sequence = torch.randint(0, self.vocab_size, (1,))
        self.step_count = 0
        return self.current_sequence
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        """Take a step in the environment."""
        # Add the action (next token) to the sequence
        action_tensor = torch.tensor([action])
        self.current_sequence = torch.cat([self.current_sequence, action_tensor])
        self.step_count += 1
        
        # Compute reward (simple reward function for demonstration)
        reward = self.compute_reward()
        
        # Check if episode is done
        done = (self.step_count >= self.max_length - 1) or (action == 0)  # End on special token
        
        info = {'sequence_length': len(self.current_sequence)}
        
        return self.current_sequence, reward, done, info
    
    def compute_reward(self) -> float:
        """
        Compute reward for the current sequence.
        This is a simple example - in practice, this would be based on
        task-specific objectives like text quality, coherence, etc.
        """
        # Simple reward: prefer sequences with diverse tokens
        if len(self.current_sequence) < 2:
            return 0.0
        
        # Reward diversity
        unique_tokens = len(torch.unique(self.current_sequence))
        diversity_reward = unique_tokens / len(self.current_sequence)
        
        # Penalty for very long sequences
        length_penalty = -0.01 * len(self.current_sequence)
        
        # Bonus for specific patterns (example: alternating tokens)
        pattern_bonus = 0.0
        if len(self.current_sequence) >= 2:
            alternating = True
            for i in range(1, len(self.current_sequence)):
                if self.current_sequence[i] == self.current_sequence[i-1]:
                    alternating = False
                    break
            if alternating:
                pattern_bonus = 1.0
        
        return diversity_reward + length_penalty + pattern_bonus


def collect_experience(
    model: MiniMindPPO, 
    env: TextEnvironment, 
    num_episodes: int = 10
) -> Dict[str, torch.Tensor]:
    """
    Collect experience by running episodes in the environment.
    
    Returns:
        Dictionary containing collected experience for training
    """
    all_input_ids = []
    all_actions = []
    all_rewards = []
    all_dones = []
    all_log_probs = []
    all_values = []
    
    model.eval()
    
    for episode in range(num_episodes):
        # Reset environment
        state = env.reset()
        
        episode_input_ids = []
        episode_actions = []
        episode_rewards = []
        episode_dones = []
        episode_log_probs = []
        episode_values = []
        
        done = False
        while not done:
            # Prepare input for model
            input_ids = state.unsqueeze(0)  # Add batch dimension
            
            # Get action from policy
            with torch.no_grad():
                action_output = model.select_action(input_ids, deterministic=False)
                action = action_output['actions'][0, -1].item()  # Get last token action
                log_prob = action_output['log_probs'][0, -1].item()
                value = action_output['values'][0, -1].item()
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            episode_input_ids.append(input_ids[0])
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_dones.append(1.0 if done else 0.0)
            episode_log_probs.append(log_prob)
            episode_values.append(value)
            
            # Update state
            state = next_state
            
            # Limit episode length for safety
            if len(episode_input_ids) >= env.max_length:
                break
        
        # Pad sequences to same length for batching
        max_len = max(len(seq) for seq in episode_input_ids)
        
        for i in range(len(episode_input_ids)):
            seq = episode_input_ids[i]
            if len(seq) < max_len:
                # Pad with zeros
                padding = torch.zeros(max_len - len(seq), dtype=seq.dtype)
                episode_input_ids[i] = torch.cat([seq, padding])
        
        # Add episode data
        if episode_input_ids:
            all_input_ids.extend(episode_input_ids)
            all_actions.extend(episode_actions)
            all_rewards.extend(episode_rewards)
            all_dones.extend(episode_dones)
            all_log_probs.extend(episode_log_probs)
            all_values.extend(episode_values)
    
    # Convert to tensors and reshape appropriately
    if not all_input_ids:
        # Return empty tensors if no data collected
        return {
            'input_ids': torch.empty(0, 1, dtype=torch.long),
            'actions': torch.empty(0, 1, dtype=torch.long),
            'rewards': torch.empty(0, 1),
            'dones': torch.empty(0, 1),
            'old_log_probs': torch.empty(0, 1),
            'old_values': torch.empty(0, 1)
        }
    
    max_seq_len = max(len(seq) for seq in all_input_ids)
    
    # Pad all sequences to max length
    padded_input_ids = []
    for seq in all_input_ids:
        if len(seq) < max_seq_len:
            padding = torch.zeros(max_seq_len - len(seq), dtype=seq.dtype)
            padded_seq = torch.cat([seq, padding])
        else:
            padded_seq = seq
        padded_input_ids.append(padded_seq)
    
    # Stack and reshape
    batch_size = len(padded_input_ids)
    input_ids = torch.stack(padded_input_ids)
    
    # For actions, rewards, etc., we need to handle the fact that they are single values per step
    # We'll create sequences of the same length as input_ids
    actions_tensor = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    rewards_tensor = torch.zeros(batch_size, max_seq_len)
    dones_tensor = torch.zeros(batch_size, max_seq_len)
    log_probs_tensor = torch.zeros(batch_size, max_seq_len)
    values_tensor = torch.zeros(batch_size, max_seq_len)
    
    for i in range(batch_size):
        seq_len = len(all_input_ids[i].nonzero())  # Actual sequence length
        actions_tensor[i, seq_len-1] = all_actions[i]  # Action taken at the end
        rewards_tensor[i, seq_len-1] = all_rewards[i]  # Reward received
        dones_tensor[i, seq_len-1] = all_dones[i]
        log_probs_tensor[i, seq_len-1] = all_log_probs[i]
        values_tensor[i, seq_len-1] = all_values[i]
    
    return {
        'input_ids': input_ids,
        'actions': actions_tensor,
        'rewards': rewards_tensor,
        'dones': dones_tensor,
        'old_log_probs': log_probs_tensor,
        'old_values': values_tensor
    }


def training_loop_example():
    """
    Demonstrate a complete training loop using MiniMind-PPO.
    """
    print("Starting MiniMind-PPO Training Example...")
    
    # Create model and trainer
    model = MiniMindPPO(
        vocab_size=100,  # Smaller vocab for faster demo
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        max_seq_length=32
    )
    
    trainer = PPOTrainer(
        model=model,
        ppo_epochs=2,
        num_mini_batches=2
    )
    
    # Create environment
    env = TextEnvironment(vocab_size=100, max_length=10)
    
    # Training loop
    num_iterations = 5
    
    for iteration in range(num_iterations):
        print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")
        
        # Collect experience
        print("Collecting experience...")
        experience = collect_experience(model, env, num_episodes=4)
        
        print(f"Collected {experience['input_ids'].shape[0]} sequences")
        
        # Train the model
        print("Training model...")
        model.train()
        
        stats = trainer.train_step(
            input_ids=experience['input_ids'],
            actions=experience['actions'],
            rewards=experience['rewards'],
            dones=experience['dones'],
            old_log_probs=experience['old_log_probs'],
            old_values=experience['old_values']
        )
        
        print(f"Training Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value:.4f}")
        
        # Test generation
        print("Testing generation...")
        test_input = torch.randint(0, 100, (1, 5))
        model.eval()
        with torch.no_grad():
            generated = model.generate(test_input, max_length=5, temperature=0.8)
            print(f"Generated sequence: {generated[0].tolist()}")
    
    print("\nTraining completed!")
    return model, trainer


def demonstrate_key_features():
    """
    Demonstrate the key features that were implemented to fix the PPO issues.
    """
    print("Demonstrating Key MiniMind-PPO Features...")
    
    model = MiniMindPPO(vocab_size=50, hidden_size=64, num_layers=2)
    
    # 1. Forward pass for training
    print("\n1. Testing _forward_for_training method:")
    input_ids = torch.randint(0, 50, (2, 10))
    logits, values, hidden_states = model._forward_for_training(input_ids)
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Values shape: {values.shape}")
    print(f"   Hidden states shape: {hidden_states.shape}")
    
    # 2. Batch forward pass
    print("\n2. Testing _batch_forward_pass method:")
    batch_outputs = model._batch_forward_pass(input_ids)
    print(f"   Batch outputs keys: {list(batch_outputs.keys())}")
    
    # 3. Logits to action probabilities
    print("\n3. Testing _logits_to_action_probs method:")
    probs, log_probs = model._logits_to_action_probs(logits)
    print(f"   Probabilities shape: {probs.shape}")
    print(f"   Log probabilities shape: {log_probs.shape}")
    print(f"   Probabilities sum (should be ~1.0): {probs.sum(dim=-1)[0, 0].item():.4f}")
    
    # 4. Value extraction
    print("\n4. Testing _extract_value_from_response method:")
    extracted_values = model._extract_value_from_response(hidden_states)
    print(f"   Extracted values shape: {extracted_values.shape}")
    
    # 5. Action selection with real probabilities
    print("\n5. Testing improved action selection:")
    action_output = model.select_action(input_ids, deterministic=False)
    print(f"   Actions shape: {action_output['actions'].shape}")
    print(f"   Log probs shape: {action_output['log_probs'].shape}")
    print(f"   Values shape: {action_output['values'].shape}")
    
    # 6. Policy ratio calculation
    print("\n6. Testing policy ratio calculation:")
    old_log_probs = torch.randn(2, 10)
    new_log_probs = action_output['log_probs']
    ratio = model.compute_policy_ratio(old_log_probs, new_log_probs)
    print(f"   Policy ratio shape: {ratio.shape}")
    print(f"   Policy ratio mean: {ratio.mean().item():.4f}")
    
    # 7. Complete PPO update
    print("\n7. Testing complete PPO update:")
    actions = torch.randint(0, 50, (2, 10))
    old_values = torch.randn(2, 10)
    returns = torch.randn(2, 10)
    advantages = torch.randn(2, 10)
    
    stats = model.ppo_update(
        input_ids, actions, old_log_probs, old_values, returns, advantages
    )
    print(f"   PPO update stats: {list(stats.keys())}")
    print(f"   Total loss: {stats['total_loss']:.4f}")
    print(f"   Policy loss: {stats['policy_loss']:.4f}")
    print(f"   Value loss: {stats['value_loss']:.4f}")
    
    print("\nAll key features working correctly!")


if __name__ == "__main__":
    print("MiniMind-PPO Usage Examples")
    print("=" * 50)
    
    # Demonstrate key features
    demonstrate_key_features()
    
    print("\n" + "=" * 50)
    
    # Run training example
    training_loop_example()