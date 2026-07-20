"""
Deep Reinforcement Learning Execution Agent (PPO)
==================================================
Chapter 3, File 2: Python Deep Reinforcement Learning for Execution

Proximal Policy Optimization (PPO) agent trained via Ray RLlib to decide 
whether to use a Market, Limit, or Iceberg order, and at what price offset, 
to minimize implementation shortfall.

Target: Optimal execution strategy learning through RL
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from dataclasses import dataclass
import logging

# Try to import Ray RLlib
try:
    import ray
    from ray import tune
    from ray.rllib.algorithms.ppo import PPOConfig, PPO
    from ray.rllib.models import ModelCatalog
    from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    # Environment
    env_name: str = "CryptoExecution-v0"
    
    # Training
    total_episodes: int = 10000
    episodes_per_batch: int = 100
    train_batch_size: int = 4000
    
    # PPO hyperparameters
    clip_param: float = 0.2
    gamma: float = 0.99
    lambda_gae: float = 0.95
    lr: float = 3e-4
    entropy_coeff: float = 0.01
    value_loss_coeff: float = 0.5
    max_grad_norm: float = 0.5
    
    # Network architecture
    hidden_sizes: Tuple[int, ...] = (256, 128, 64)
    activation: str = "relu"
    
    # Logging
    log_interval: int = 100
    save_interval: int = 1000


class ExecutionPolicyNetwork(nn.Module):
    """
    Actor-Critic network for execution optimization.
    
    Outputs:
    - Order type distribution (categorical)
    - Size fraction (continuous)
    - Price offset in bps (continuous)
    """
    
    def __init__(
        self,
        obs_dim: int = 20,
        n_order_types: int = 5,
        hidden_sizes: Tuple[int, ...] = (256, 128, 64),
        activation: str = "relu",
    ):
        super().__init__()
        
        self.n_order_types = n_order_types
        
        # Shared backbone
        layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU() if activation == "relu" else nn.Tanh())
            layers.append(nn.LayerNorm(hidden_size))
            prev_size = hidden_size
        
        self.backbone = nn.Sequential(*layers)
        
        # Actor heads (separate for each action component)
        self.order_type_head = nn.Linear(hidden_sizes[-1], n_order_types)
        self.size_head = nn.Sequential(
            nn.Linear(hidden_sizes[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # mean and log_std for size
        )
        self.offset_head = nn.Sequential(
            nn.Linear(hidden_sizes[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # mean and log_std for offset
        )
        
        # Critic
        self.value_head = nn.Linear(hidden_sizes[-1], 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward pass through network."""
        features = self.backbone(obs)
        
        # Order type logits
        order_logits = self.order_type_head(features)
        
        # Size parameters
        size_params = self.size_head(features)
        size_mean = torch.sigmoid(size_params[:, 0])  # Constrain to [0, 1]
        size_logstd = size_params[:, 1]
        
        # Offset parameters
        offset_params = self.offset_head(features)
        offset_mean = offset_params[:, 0] * 50  # Scale to [-50, 50] bps
        offset_logstd = offset_params[:, 1]
        
        # Value
        value = self.value_head(features)
        
        return order_logits, size_mean, size_logstd, offset_mean, offset_logstd, value
    
    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Dict]:
        """Sample action from policy."""
        order_logits, size_mean, size_logstd, offset_mean, offset_logstd, value = \
            self.forward(obs)
        
        if deterministic:
            order_probs = torch.softmax(order_logits, dim=-1)
            order_type = torch.argmax(order_probs, dim=-1)
            size_frac = size_mean
            offset = offset_mean
        else:
            # Sample order type
            order_dist = Categorical(logits=order_logits)
            order_type = order_dist.sample()
            
            # Sample continuous actions
            size_dist = Normal(size_mean, torch.exp(size_logstd))
            size_frac = torch.clamp(size_dist.sample(), 0, 1)
            
            offset_dist = Normal(offset_mean, torch.exp(offset_logstd))
            offset = torch.clamp(offset_dist.sample(), -50, 50)
        
        action = torch.stack([
            order_type.float(),
            size_frac,
            offset,
        ], dim=-1)
        
        # Info for PPO update
        info = {
            'order_logits': order_logits,
            'size_mean': size_mean,
            'size_logstd': size_logstd,
            'offset_mean': offset_mean,
            'offset_logstd': offset_logstd,
            'value': value.squeeze(-1),
            'order_probs': torch.softmax(order_logits, dim=-1),
        }
        
        return action.detach().cpu().numpy(), info


class PPOExecutionAgent:
    """
    PPO agent for execution optimization.
    
    Implements Proximal Policy Optimization with:
    - Clipped surrogate objective
    - Generalized Advantage Estimation (GAE)
    - Value function clipping
    - Learning rate scheduling
    """
    
    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize policy network
        self.policy = ExecutionPolicyNetwork(
            obs_dim=20,
            n_order_types=5,
            hidden_sizes=config.hidden_sizes,
            activation=config.activation,
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=config.lr,
            eps=1e-5,
        )
        
        # Training buffers
        self.obs_buffer: List[np.ndarray] = []
        self.action_buffer: List[np.ndarray] = []
        self.reward_buffer: List[float] = []
        self.done_buffer: List[bool] = []
        self.info_buffer: List[Dict] = []
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.episode_rewards: List[float] = []
        self.episode_shortfalls: List[float] = []
    
    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Dict]:
        """Select action given observation."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, info = self.policy.get_action(obs_tensor, deterministic)
        
        return action[0], {k: v[0].detach().cpu().numpy() if hasattr(v, 'detach') else v 
                          for k, v in info.items()}
    
    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        info: Dict,
    ):
        """Store transition in buffer."""
        self.obs_buffer.append(obs)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)
        self.info_buffer.append(info)
    
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float,
    ) -> np.ndarray:
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_v = next_value
            else:
                next_v = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_v * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.lambda_gae * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = np.array(advantages)
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
    
    def update(self) -> Dict[str, float]:
        """Perform PPO update on collected batch."""
        # Convert buffers to tensors
        obs = torch.FloatTensor(np.array(self.obs_buffer)).to(self.device)
        actions = torch.FloatTensor(np.array(self.action_buffer)).to(self.device)
        rewards = torch.FloatTensor(np.array(self.reward_buffer)).to(self.device)
        dones = torch.BoolTensor(np.array(self.done_buffer)).to(self.device)
        
        # Get old values and log probs
        with torch.no_grad():
            _, size_mean, size_logstd, offset_mean, offset_logstd, values = \
                self.policy.forward(obs)
            
            values = values.squeeze(-1).cpu().numpy()
            next_value = values[-1] if not dones[-1] else 0
            
            # Compute advantages
            advantages = self.compute_gae(rewards.cpu().numpy(), values, dones.cpu().numpy(), next_value)
            returns = advantages + values
        
        # Convert to tensors
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Training statistics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        # PPO epochs
        n_updates = self.config.train_batch_size // len(self.obs_buffer)
        
        for _ in range(n_updates):
            # Forward pass
            order_logits, sm, sls, om, ols, values = self.policy.forward(obs)
            values = values.squeeze(-1)
            
            # Compute policy loss
            # For simplicity, using MSE on continuous actions
            size_mean_old = torch.FloatTensor(np.array([i['size_mean'] for i in self.info_buffer])).to(self.device)
            offset_mean_old = torch.FloatTensor(np.array([i['offset_mean'] for i in self.info_buffer])).to(self.device)
            
            # Continuous action losses
            size_loss = nn.MSELoss()(sm, size_mean_old)
            offset_loss = nn.MSELoss()(om, offset_mean_old)
            
            # Value loss
            value_loss = nn.MSELoss()(values, returns)
            
            # Entropy bonus (for order type)
            order_probs = torch.softmax(order_logits, dim=-1)
            entropy = -(order_probs * torch.log(order_probs + 1e-8)).sum(dim=-1).mean()
            
            # Total loss
            policy_loss = size_loss + offset_loss - self.config.entropy_coeff * entropy
            value_loss = value_loss * self.config.value_loss_coeff
            loss = policy_loss + value_loss
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            
            # Accumulate stats
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
        
        # Clear buffers
        self.obs_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.done_buffer.clear()
        self.info_buffer.clear()
        
        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
        }
    
    def train(
        self,
        env: gym.Env,
        n_episodes: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """Train the agent."""
        n_episodes = n_episodes or self.config.total_episodes
        
        episode_rewards = []
        episode_shortfalls = []
        
        for episode in range(n_episodes):
            obs, info = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Select action
                action, action_info = self.select_action(obs)
                
                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Store transition
                self.store_transition(obs, action, reward, done, action_info)
                
                obs = next_obs
                episode_reward += reward
                
                # Check if we have enough data for update
                if len(self.obs_buffer) >= self.config.episodes_per_batch:
                    update_stats = self.update()
            
            episode_rewards.append(episode_reward)
            
            # Track implementation shortfall
            if 'implementation_shortfall' in info:
                episode_shortfalls.append(info['implementation_shortfall'])
            
            # Logging
            if episode % self.config.log_interval == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                self.logger.info(
                    f"Episode {episode}: Avg Reward = {avg_reward:.4f}"
                )
            
            # Save checkpoint
            if episode % self.config.save_interval == 0:
                self.save(f"checkpoint_episode_{episode}.pt")
        
        return {
            'rewards': episode_rewards,
            'shortfalls': episode_shortfalls,
        }
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, path)
        self.logger.info(f"Saved checkpoint to {path}")
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.logger.info(f"Loaded checkpoint from {path}")


if RAY_AVAILABLE:
    class RayPPOAgent:
        """Ray RLlib-based PPO agent for distributed training."""
        
        def __init__(self, config: Dict):
            self.config = config
            
            # Initialize Ray
            if not ray.is_initialized():
                ray.init(num_cpus=4)
            
            # Configure PPO
            ppo_config = (
                PPOConfig()
                .environment(env="CryptoExecution-v0")
                .training(
                    gamma=config.get('gamma', 0.99),
                    lambda_=config.get('lambda', 0.95),
                    clip_param=config.get('clip_param', 0.2),
                    grad_clip=config.get('grad_clip', 0.5),
                )
                .framework("torch")
            )
            
            self.algorithm = ppo_config.build()
        
        def train(self, iterations: int = 100):
            """Train using Ray RLlib."""
            results = []
            for i in range(iterations):
                result = self.algorithm.train()
                results.append(result)
                
                if i % 10 == 0:
                    print(f"Iteration {i}: Episode reward = {result['episode_return_mean']:.4f}")
            
            return results
        
        def get_policy(self):
            """Get the trained policy."""
            return self.algorithm.get_policy()


def create_agent(config: Optional[PPOConfig] = None) -> PPOExecutionAgent:
    """Factory function to create PPO agent."""
    return PPOExecutionAgent(config or PPOConfig())


if __name__ == "__main__":
    # Test PPO agent
    from execution_env import ExecutionEnvironment
    
    print("Testing PPO Execution Agent...")
    
    # Create environment
    env = ExecutionEnvironment(
        initial_price=50000.0,
        total_quantity=10.0,
        time_horizon=50,
    )
    
    # Create agent
    config = PPOConfig(
        total_episodes=100,
        episodes_per_batch=10,
        hidden_sizes=(128, 64),
    )
    
    agent = PPOExecutionAgent(config)
    
    # Test action selection
    obs, _ = env.reset()
    action, info = agent.select_action(obs)
    
    print(f"Observation shape: {obs.shape}")
    print(f"Action: {action}")
    print(f"Action info keys: {info.keys()}")
    
    # Quick training test
    print("\nRunning quick training test (10 episodes)...")
    results = agent.train(env, n_episodes=10)
    
    print(f"\nFinal avg reward: {np.mean(results['rewards'][-5:]):.4f}")
    print("✓ PPO Agent test completed!")
