"""
Multi-Agent PPO Trainer using Ray RLlib
=========================================
Implements Multi-Agent PPO (MAPPO) training for the trading environment
with specialized policies for each agent type.

Key features:
- Centralized critic with decentralized actors (CTDE)
- Role-specific policy networks
- Risk-aware reward shaping
- Memory-bounded training for 8GB RAM constraint
- AMD ROCm GPU support when available

The Risk Manager agent learns to penalize the Alpha Generator
for taking excessive drawdowns, creating a natural risk-control mechanism.
"""

import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.logger import pretty_print
from typing import Dict, Optional, Any
import os
import warnings

warnings.filterwarnings('ignore')

# Import our custom environment
from marl_environment import MultiAgentTradingEnv, make_marl_trading_env


def create_policy_specs(n_assets: int = 10) -> Dict:
    """
    Create policy specifications for each agent type.
    
    Each agent has a tailored neural network architecture:
    - Alpha Generator: Larger network for pattern recognition
    - Risk Manager: Conservative architecture focused on tail risks
    - Executor: Fast response network for execution timing
    """
    
    # Observation dimension (market + portfolio features)
    obs_dim = 15 + 20 + 10  # market + positions + portfolio
    
    policy_specs = {
        # Alpha Generator - focuses on finding profitable signals
        "alpha_generator": {
            "observation_space": tune.sample_from(lambda spec: 
                tuple([obs_dim])),
            "action_space": tune.sample_from(lambda spec: 
                tuple([n_assets])),
            "config": {
                "fcnet_hiddens": [256, 128, 64],
                "fcnet_activation": "relu",
                "use_gae": True,
                "lambda": 0.95,
                "kl_coeff": 0.2,
                "grad_clip": 0.5,
            }
        },
        
        # Risk Manager - focuses on preventing catastrophic losses
        "risk_manager": {
            "observation_space": tune.sample_from(lambda spec: 
                tuple([obs_dim])),
            "action_space": tune.sample_from(lambda spec: 
                tuple([n_assets])),
            "config": {
                "fcnet_hiddens": [128, 64, 32],
                "fcnet_activation": "tanh",  # More conservative
                "use_gae": True,
                "lambda": 0.9,
                "kl_coeff": 0.3,  # Higher KL penalty for stability
                "grad_clip": 0.3,
            }
        },
        
        # Executor - focuses on minimizing slippage
        "executor": {
            "observation_space": tune.sample_from(lambda spec: 
                tuple([obs_dim])),
            "action_space": tune.sample_from(lambda spec: 
                tuple([5])),
            "config": {
                "fcnet_hiddens": [128, 64],
                "fcnet_activation": "relu",
                "use_gae": True,
                "lambda": 0.95,
                "kl_coeff": 0.15,
                "grad_clip": 0.5,
            }
        },
    }
    
    return policy_specs


def create_shared_critic_config() -> Dict:
    """
    Configuration for the shared value function (critic).
    
    In CTDE, all agents share a common value function that
    estimates the joint value of the multi-agent state.
    """
    return {
        "vf_share_layers": True,  # Share critic across agents
        "vf_loss_coeff": 1.0,
        "value_function_architecture": {
            "fcnet_hiddens": [256, 128, 64],
            "fcnet_activation": "relu",
        },
    }


class MultiAgentTrainer:
    """
    Main trainer class for Multi-Agent PPO.
    
    Handles:
    - Ray initialization and resource management
    - Training loop with periodic evaluation
    - Checkpointing and model persistence
    - Memory monitoring for 8GB constraint
    """
    
    def __init__(self, 
                 n_assets: int = 10,
                 num_cpus: int = 4,
                 num_gpus: float = 0.5,
                 memory_gb: float = 6.0,
                 checkpoint_dir: str = "./checkpoints"):
        """
        Initialize the trainer with resource constraints.
        
        Args:
            n_assets: Number of assets in the environment
            num_cpus: CPU cores for training
            num_gpus: GPUs for training (fraction OK)
            memory_gb: Max memory usage (keep under 8GB total)
            checkpoint_dir: Directory for saving checkpoints
        """
        self.n_assets = n_assets
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.memory_limit_gb = memory_gb
        self.checkpoint_dir = checkpoint_dir
        
        # Ensure checkpoint directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Ray may already be initialized
        if not ray.is_initialized():
            ray.init(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                _memory=int(memory_gb * 1e9),
                object_store_memory=int(memory_gb * 0.3 * 1e9),
                include_dashboard=False,
            )
        
        self.policy_specs = create_policy_specs(n_assets)
        self.critic_config = create_shared_critic_config()
        
        # Training metrics
        self.training_history = []
        self.best_reward = -np.inf
        
    def create_ppo_config(self, 
                          env_creator,
                          train_batch_size: int = 4000,
                          rollout_fragment_length: int = 200,
                          lr: float = 3e-4) -> PPOConfig:
        """
        Create Ray PPO configuration for multi-agent training.
        """
        config = (
            PPOConfig()
            
            # Environment settings
            .environment(env_creator)
            
            # Multi-agent settings
            .multi_agent(
                policies=self.policy_specs.keys(),
                policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: agent_id,
                policies_to_train=list(self.policy_specs.keys()),
            )
            
            # Model settings
            .framework("torch")
            .training(
                model={
                    "fcnet_hiddens": [256, 128, 64],
                    "fcnet_activation": "relu",
                },
                **self.critic_config
            )
            
            # PPO hyperparameters
            .rollouts(
                rollout_fragment_length=rollout_fragment_length,
                batch_mode="truncate_episodes",
            )
            .training(
                train_batch_size=train_batch_size,
                grad_clip=0.5,
                vf_loss_coeff=1.0,
                entropy_coeff=0.01,
                kl_coeff=0.2,
                clip_param=0.2,
                gae_lambda=0.95,
                gamma=0.99,
            )
            
            # Optimizer settings
            .optimizer(
                lr=lr,
            )
            
            # Resource allocation
            .resources(
                num_learners=1,
                num_workers=max(1, self.num_cpus - 1),
                num_gpus_per_learner=self.num_gpus,
                num_gpus_per_worker=0.0,
            )
            
            # Evaluation settings
            .evaluation(
                evaluation_interval=5,
                evaluation_duration=10,
                evaluation_config={"explore": False},
            )
        )
        
        return config
    
    def train(self,
              total_timesteps: int = 10_000_000,
              checkpoint_frequency: int = 50,
              verbose: bool = True) -> PPO:
        """
        Run training loop.
        
        Args:
            total_timesteps: Total training timesteps
            checkpoint_frequency: Save checkpoint every N iterations
            verbose: Print progress
            
        Returns:
            Trained PPO algorithm instance
        """
        # Create environment creator
        def env_creator(config):
            return make_marl_trading_env(n_assets=self.n_assets)
        
        # Build config
        config = self.create_ppo_config(env_creator)
        
        # Build algorithm
        algo = config.build()
        
        if verbose:
            print("=" * 60)
            print("Starting Multi-Agent PPO Training")
            print(f"Assets: {self.n_assets}")
            print(f"Total Timesteps: {total_timesteps:,}")
            print(f"Memory Limit: {self.memory_limit_gb} GB")
            print("=" * 60)
        
        # Training loop
        iterations = total_timesteps // config.rollouts['rollout_fragment_length']
        
        for iteration in range(iterations):
            try:
                # Train one iteration
                result = algo.train()
                
                # Store metrics
                self.training_history.append(result)
                
                # Extract rewards per agent
                policy_rewards = {}
                for policy_id in self.policy_specs.keys():
                    key = f"{policy_id}/policy_reward_mean"
                    if key in result:
                        policy_rewards[policy_id] = result[key]
                
                # Check for improvement
                total_reward = sum(policy_rewards.values())
                if total_reward > self.best_reward:
                    self.best_reward = total_reward
                    
                    # Save best model
                    best_path = os.path.join(self.checkpoint_dir, "best_model")
                    algo.save(best_path)
                
                # Periodic checkpointing
                if (iteration + 1) % checkpoint_frequency == 0:
                    checkpoint_path = os.path.join(
                        self.checkpoint_dir, 
                        f"checkpoint_{iteration}"
                    )
                    algo.save(checkpoint_path)
                    
                    if verbose:
                        print(f"\nCheckpoint saved at iteration {iteration}")
                
                # Progress reporting
                if verbose and (iteration + 1) % 10 == 0:
                    print(f"\nIteration {iteration + 1}/{iterations}")
                    print(f"  Total Reward: {total_reward:.4f}")
                    for agent, reward in policy_rewards.items():
                        print(f"  {agent}: {reward:.4f}")
                    
                    # Memory check
                    self._check_memory_usage()
                
            except Exception as e:
                print(f"Training error at iteration {iteration}: {e}")
                continue
        
        # Final save
        final_path = os.path.join(self.checkpoint_dir, "final_model")
        algo.save(final_path)
        
        if verbose:
            print("\n" + "=" * 60)
            print("Training Complete!")
            print(f"Best Reward: {self.best_reward:.4f}")
            print(f"Final Checkpoint: {final_path}")
            print("=" * 60)
        
        return algo
    
    def _check_memory_usage(self):
        """Monitor and report memory usage."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_gb = process.memory_info().rss / 1e9
            
            if memory_gb > self.memory_limit_gb * 0.9:
                print(f"\nWARNING: Memory usage at {memory_gb:.2f} GB "
                      f"(limit: {self.memory_limit_gb} GB)")
                
                # Trigger garbage collection
                import gc
                gc.collect()
                
        except ImportError:
            pass
    
    def load_checkpoint(self, checkpoint_path: str) -> PPO:
        """Load a trained model from checkpoint."""
        def env_creator(config):
            return make_marl_trading_env(n_assets=self.n_assets)
        
        config = self.create_ppo_config(env_creator)
        algo = config.build()
        algo.restore(checkpoint_path)
        
        return algo
    
    def get_training_stats(self) -> Dict:
        """Get aggregated training statistics."""
        if not self.training_history:
            return {}
        
        rewards = []
        for h in self.training_history:
            total = 0.0
            for policy_id in self.policy_specs.keys():
                key = f"{policy_id}/policy_reward_mean"
                if key in h:
                    total += h[key]
            rewards.append(total)
        
        return {
            'total_iterations': len(self.training_history),
            'best_reward': self.best_reward,
            'final_reward': rewards[-1] if rewards else 0.0,
            'avg_reward': np.mean(rewards[-10:]) if rewards else 0.0,
            'reward_std': np.std(rewards[-10:]) if rewards else 0.0,
        }


def run_training(n_assets: int = 10,
                 total_timesteps: int = 5_000_000,
                 checkpoint_dir: str = "./marl_checkpoints"):
    """
    Convenience function to run a complete training session.
    """
    trainer = MultiAgentTrainer(
        n_assets=n_assets,
        num_cpus=4,
        num_gpus=0.5,
        memory_gb=6.0,
        checkpoint_dir=checkpoint_dir,
    )
    
    algo = trainer.train(
        total_timesteps=total_timesteps,
        checkpoint_frequency=25,
        verbose=True,
    )
    
    stats = trainer.get_training_stats()
    print("\nTraining Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return algo, trainer


if __name__ == "__main__":
    # Example training run
    print("Starting MARL Training...")
    
    # For quick testing, use fewer timesteps
    algo, trainer = run_training(
        n_assets=5,
        total_timesteps=100_000,  # Small for testing
        checkpoint_dir="./test_checkpoints",
    )
    
    # Shutdown Ray
    ray.shutdown()
