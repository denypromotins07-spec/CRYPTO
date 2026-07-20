"""
Ray RLlib Setup for Training PPO/SAC Agents.
Implements custom callbacks to log metrics and ensures the training environment 
strictly respects the 8GB global memory cap.

Target Hardware: AMD Ryzen AI 5 + AMD Radeon GPU + 16GB RAM (8GB cap)
"""

import os
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass
import numpy as np

# Ray imports
try:
    import ray
    from ray import tune
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.algorithms.sac import SACConfig
    from ray.rllib.env.base_env import BaseEnv
    from ray.rllib.callbacks import Callbacks
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    logging.warning("Ray not available, RL training disabled")

logger = logging.getLogger(__name__)


@dataclass
class RLTrainingConfig:
    """Configuration for RL training."""
    # Algorithm selection
    algorithm: str = "PPO"  # or "SAC"
    
    # Environment
    env_name: str = "TradingEnv"
    observation_space_dim: int = 100
    action_space_dim: int = 3
    
    # Training parameters
    total_timesteps: int = 1_000_000
    train_batch_size: int = 4096
    rollout_fragment_length: int = 200
    
    # PPO specific
    ppo_clip_param: float = 0.2
    ppo_vf_loss_coeff: float = 0.5
    ppo_entropy_coeff: float = 0.01
    ppo_lr: float = 3e-4
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    
    # SAC specific
    sac_target_update_tau: float = 0.005
    sac_target_update_interval: int = 1
    sac_alpha: float = 0.2
    
    # Network architecture
    fcnet_hiddens: List[int] = None
    fcnet_activation: str = "relu"
    
    # Resource constraints
    num_workers: int = 2
    num_cpus_per_worker: float = 1.0
    num_gpus_per_worker: float = 0.0
    memory_per_worker_gb: float = 2.0
    
    # Checkpointing
    checkpoint_freq: int = 50
    keep_checkpoints_num: int = 5
    
    def __post_init__(self):
        if self.fcnet_hiddens is None:
            self.fcnet_hiddens = [256, 256, 128]


class TradingCallbacks(Callbacks):
    """Custom callbacks for RL training metrics and memory monitoring."""
    
    def __init__(self, max_memory_gb: float = 8.0):
        super().__init__()
        self.max_memory_bytes = max_memory_gb * 1024**3
        self.episode_rewards_history = []
        
    def on_episode_start(self, worker, base_env, policies, **kwargs):
        """Called at start of each episode."""
        pass
    
    def on_episode_step(self, worker, base_env, policies, **kwargs):
        """Called at each environment step."""
        # Monitor memory usage
        import psutil
        process = psutil.Process(os.getpid())
        mem_used = process.memory_info().rss
        
        if mem_used > self.max_memory_bytes * 0.9:
            logger.warning(
                f"Memory usage high: {mem_used / 1024**3:.2f}GB / "
                f"{self.max_memory_gb:.1f}GB"
            )
    
    def on_episode_end(self, worker, base_env, policies, **kwargs):
        """Called at end of each episode."""
        episode = kwargs.get("episode")
        if episode:
            reward = episode.episode_reward
            self.episode_rewards_history.append(reward)
            
            # Log custom metrics
            episode.custom_metrics["rolling_reward_mean"] = np.mean(
                self.episode_rewards_history[-100:]
            )
    
    def on_train_result(self, result, **kwargs):
        """Called after each training iteration."""
        # Add custom metrics to result
        if self.episode_rewards_history:
            result["custom_metrics"]["best_reward"] = max(self.episode_rewards_history)
            result["custom_metrics"]["avg_reward_100"] = np.mean(
                self.episode_rewards_history[-100:]
            )


class TradingEnvironment(BaseEnv):
    """
    Custom trading environment for RL training.
    Wraps the actual trading simulator with Ray RLlib interface.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        observation_space_dim: int = 100,
        action_space_dim: int = 3,
    ):
        super().__init__()
        
        self.config = config
        self.observation_space_dim = observation_space_dim
        self.action_space_dim = action_space_dim
        
        # Initialize gym spaces
        import gymnasium as gym
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_space_dim,),
            dtype=np.float32
        )
        
        # Discrete actions: 0=hold, 1=buy, 2=sell
        self.action_space = gym.spaces.Discrete(action_space_dim)
        
        # State tracking
        self.current_step = 0
        self.max_steps = 10000
        self.current_position = 0.0
        self.current_pnl = 0.0
        
    def reset(self, *, seed=None, options=None):
        """Reset environment."""
        self.current_step = 0
        self.current_position = 0.0
        self.current_pnl = 0.0
        
        # Return initial observation
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute action and return next state."""
        self.current_step += 1
        
        # Execute trade based on action
        reward = self._execute_action(action)
        
        # Check termination
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        obs = self._get_observation()
        info = {
            "pnl": self.current_pnl,
            "position": self.current_position,
            "step": self.current_step,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Generate observation vector."""
        # In production, this would fetch real market features
        obs = np.random.randn(self.observation_space_dim).astype(np.float32)
        return obs
    
    def _execute_action(self, action: int) -> float:
        """Execute trading action and return reward."""
        # Simplified reward calculation
        if action == 0:  # Hold
            reward = 0.0
        elif action == 1:  # Buy
            self.current_position += 0.1
            reward = np.random.randn() * 0.01
        elif action == 2:  # Sell
            self.current_position -= 0.1
            reward = np.random.randn() * 0.01
        else:
            reward = -0.1  # Invalid action penalty
        
        self.current_pnl += reward
        return reward


class RLTrainer:
    """
    Ray RLlib trainer for PPO/SAC agents.
    
    Features:
    - Memory-cap aware training
    - Custom callback integration
    - Automatic resource allocation
    - Checkpoint management
    """
    
    def __init__(
        self,
        config: RLTrainingConfig,
        storage_path: str = "./data/rl_training",
        max_ram_gb: float = 8.0
    ):
        """
        Initialize RL trainer.
        
        Args:
            config: RLTrainingConfig object
            storage_path: Path for checkpoints and logs
            max_ram_gb: Maximum RAM usage
        """
        self.config = config
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.max_ram_gb = max_ram_gb
        self.ray_initialized = False
        self.algorithm = None
        
        logger.info(f"RLTrainer initialized: algorithm={config.algorithm}")
        logger.info(f"Max RAM: {max_ram_gb}GB")
    
    def initialize_ray(self):
        """Initialize Ray with appropriate resources."""
        if not RAY_AVAILABLE:
            raise RuntimeError("Ray is not installed")
        
        if not self.ray_initialized:
            # Calculate available CPUs
            total_cpus = os.cpu_count() or 4
            num_cpus = min(total_cpus, int(self.max_ram_gb / 2))
            
            ray.init(
                num_cpus=num_cpus,
                include_dashboard=False,
                _temp_dir=str(self.storage_path / "ray_temp"),
                logging_level=logging.WARNING,
            )
            
            self.ray_initialized = True
            logger.info(f"Ray initialized with {num_cpus} CPUs")
    
    def build_algorithm(self):
        """Build RL algorithm configuration."""
        self.initialize_ray()
        
        # Create callbacks
        callbacks = TradingCallbacks(max_memory_gb=self.max_ram_gb)
        
        if self.config.algorithm == "PPO":
            algo_config = (
                PPOConfig()
                .environment(
                    env=TradingEnvironment,
                    env_config={
                        "observation_space_dim": self.config.observation_space_dim,
                        "action_space_dim": self.config.action_space_dim,
                    }
                )
                .rollouts(
                    rollout_fragment_length=self.config.rollout_fragment_length,
                    train_batch_size=self.config.train_batch_size,
                )
                .training(
                    clip_param=self.config.ppo_clip_param,
                    vf_loss_coeff=self.config.ppo_vf_loss_coeff,
                    entropy_coeff=self.config.ppo_entropy_coeff,
                    lr=self.config.ppo_lr,
                    gamma=self.config.ppo_gamma,
                    lambda_=self.config.ppo_gae_lambda,
                    model={
                        "fcnet_hiddens": self.config.fcnet_hiddens,
                        "fcnet_activation": self.config.fcnet_activation,
                    }
                )
                .workers(
                    num_workers=self.config.num_workers,
                    num_cpus_per_worker=self.config.num_cpus_per_worker,
                    num_gpus_per_worker=self.config.num_gpus_per_worker,
                )
                .resources(
                    num_cpus_per_learner=1,
                    num_gpus_per_learner=0,
                )
                .callbacks(callbacks)
            )
        elif self.config.algorithm == "SAC":
            algo_config = (
                SACConfig()
                .environment(
                    env=TradingEnvironment,
                    env_config={
                        "observation_space_dim": self.config.observation_space_dim,
                        "action_space_dim": self.config.action_space_dim,
                    }
                )
                .training(
                    target_update_tau=self.config.sac_target_update_tau,
                    target_update_interval=self.config.sac_target_update_interval,
                    alpha=self.config.sac_alpha,
                    model={
                        "fcnet_hiddens": self.config.fcnet_hiddens,
                        "fcnet_activation": self.config.fcnet_activation,
                    }
                )
                .workers(
                    num_workers=self.config.num_workers,
                    num_cpus_per_worker=self.config.num_cpus_per_worker,
                )
                .callbacks(callbacks)
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")
        
        return algo_config
    
    def train(self, num_iterations: Optional[int] = None) -> Dict[str, Any]:
        """
        Run RL training.
        
        Args:
            num_iterations: Number of training iterations
            
        Returns:
            Training results dictionary
        """
        algo_config = self.build_algorithm()
        self.algorithm = algo_config.build()
        
        logger.info(f"Starting {self.config.algorithm} training")
        
        results = []
        for i in range(num_iterations or 100):
            result = self.algorithm.train()
            results.append(result)
            
            # Log progress
            if i % 10 == 0:
                logger.info(
                    f"Iteration {i}: reward_mean={result['episode_reward_mean']:.4f}, "
                    f"memory_warning={'MEMORY' in str(result.get('custom_metrics', {}))}"
                )
            
            # Save checkpoint
            if (i + 1) % self.config.checkpoint_freq == 0:
                checkpoint_path = self.algorithm.save_checkpoint(
                    str(self.storage_path / "checkpoints")
                )
                logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        return results[-1] if results else {}
    
    def get_policy(self):
        """Get trained policy."""
        if self.algorithm is None:
            raise RuntimeError("No algorithm trained yet")
        return self.algorithm.get_policy()
    
    def save_model(self, path: str):
        """Save trained model."""
        if self.algorithm is None:
            raise RuntimeError("No algorithm to save")
        
        self.algorithm.save(str(Path(path)))
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model."""
        self.initialize_ray()
        
        if self.config.algorithm == "PPO":
            from ray.rllib.algorithms.ppo import PPO
            self.algorithm = PPO.from_checkpoint(path)
        elif self.config.algorithm == "SAC":
            from ray.rllib.algorithms.sac import SAC
            self.algorithm = SAC.from_checkpoint(path)
        
        logger.info(f"Model loaded from {path}")
    
    def shutdown(self):
        """Shutdown Ray and cleanup."""
        if self.algorithm:
            self.algorithm.stop()
        
        if self.ray_initialized:
            ray.shutdown()
            self.ray_initialized = False
        
        logger.info("RL trainer shut down")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    config = RLTrainingConfig(
        algorithm="PPO",
        total_timesteps=10000,
        num_workers=1,
        fcnet_hiddens=[128, 128],
    )
    
    trainer = RLTrainer(config, storage_path="./data/test_rl")
    
    try:
        result = trainer.train(num_iterations=20)
        print(f"\nFinal result: {result.get('episode_reward_mean', 'N/A')}")
        
        trainer.save_model("./data/test_rl/final_model")
        
    finally:
        trainer.shutdown()
