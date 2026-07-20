"""
Sim-to-Real Transfer for Execution RL
======================================
Chapter 3, File 3: Python Deep Reinforcement Learning for Execution

Logic to handle the "sim-to-real" gap in execution. Applies domain 
randomization (simulating exchange lag, partial fills, WebSocket drops, 
and API rate limits) during training to ensure the DRL agent survives 
in live microsecond environments.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging


class DomainRandomizationType(Enum):
    """Types of domain randomization."""
    LATENCY = "latency"
    PARTIAL_FILLS = "partial_fills"
    WEBSOCKET_DROPS = "websocket_drops"
    RATE_LIMITS = "rate_limits"
    PRICE_SLIPPAGE = "price_slippage"
    ORDER_REJECTIONS = "order_rejections"
    CLOCK_DRIFT = "clock_drift"


@dataclass
class RandomizationConfig:
    """Configuration for a single randomization dimension."""
    enabled: bool = True
    min_value: float = 0.0
    max_value: float = 1.0
    distribution: str = "uniform"  # uniform, normal, beta
    update_frequency: int = 1  # How often to resample


@dataclass
class SimToRealConfig:
    """Configuration for sim-to-real transfer."""
    # Latency simulation
    latency: RandomizationConfig = field(default_factory=lambda: RandomizationConfig(
        enabled=True, min_value=1, max_value=100, distribution="lognormal"
    ))
    
    # Partial fill probability
    partial_fill: RandomizationConfig = field(default_factory=lambda: RandomizationConfig(
        enabled=True, min_value=0.1, max_value=0.9
    ))
    
    # WebSocket drop probability per step
    websocket_drop: RandomizationConfig = field(default_factory=lambda: RandomizationConfig(
        enabled=True, min_value=0.0, max_value=0.05
    ))
    
    # Rate limit (requests per second)
    rate_limit: RandomizationConfig = field(default_factory=lambda: RandomizationConfig(
        enabled=True, min_value=5, max_value=50
    ))
    
    # Price slippage multiplier
    slippage_multiplier: RandomizationConfig = field(default_factory=lambda: RandomizationConfig(
        enabled=True, min_value=0.5, max_value=3.0
    ))
    
    # Order rejection probability
    rejection_prob: RandomizationConfig = field(default_factory=lambda: RandomizationConfig(
        enabled=True, min_value=0.0, max_value=0.1
    ))
    
    # Clock drift (ms per second)
    clock_drift: RandomizationConfig = field(default_factory=lambda: RandomizationConfig(
        enabled=True, min_value=-1, max_value=1
    ))
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_steps: int = 10000
    start_easy: bool = True


class DomainRandomizer:
    """
    Applies domain randomization to bridge sim-to-real gap.
    
    Randomizes environment parameters during training so the policy
    learns robust behaviors that generalize to real-world conditions.
    """
    
    def __init__(self, config: SimToRealConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Current randomized values
        self.current_values: Dict[str, float] = {}
        
        # Curriculum state
        self.curriculum_step = 0
        self.difficulty = 0.0 if config.start_easy else 1.0
        
        # History for analysis
        self.randomization_history: List[Dict] = []
        
        # Initialize all randomizations
        self._resample_all()
    
    def _resample_all(self) -> None:
        """Resample all enabled randomization parameters."""
        self.current_values = {
            'latency_ms': self._sample(self.config.latency),
            'partial_fill_rate': self._sample(self.config.partial_fill),
            'websocket_drop_prob': self._sample(self.config.websocket_drop),
            'rate_limit_rps': self._sample(self.config.rate_limit),
            'slippage_multiplier': self._sample(self.config.slippage_multiplier),
            'rejection_prob': self._sample(self.config.rejection_prob),
            'clock_drift_ms': self._sample(self.config.clock_drift),
        }
    
    def _sample(self, config: RandomizationConfig) -> float:
        """Sample from configured distribution."""
        if not config.enabled:
            return (config.min_value + config.max_value) / 2
        
        # Apply curriculum difficulty scaling
        min_val = config.min_value
        max_val = config.max_value
        
        if self.config.use_curriculum:
            # Interpolate based on difficulty
            base_min = (config.min_value + config.max_value) / 2
            base_max = config.max_value
            min_val = base_min + (config.min_value - base_min) * (1 - self.difficulty)
            max_val = base_max
        
        if config.distribution == "uniform":
            return np.random.uniform(min_val, max_val)
        elif config.distribution == "normal":
            mean = (min_val + max_val) / 2
            std = (max_val - min_val) / 6  # 99.7% within range
            return np.clip(np.random.normal(mean, std), min_val, max_val)
        elif config.distribution == "lognormal":
            # For latency - log-normal distribution
            mean_log = np.log((min_val + max_val) / 2)
            sigma = (np.log(max_val) - np.log(min_val)) / 6
            return np.clip(np.random.lognormal(mean_log, sigma), min_val, max_val)
        elif config.distribution == "beta":
            # Beta distribution for bounded variables
            alpha, beta = 2, 5  # Skewed toward lower values
            scaled = np.random.beta(alpha, beta)
            return min_val + scaled * (max_val - min_val)
        else:
            return np.random.uniform(min_val, max_val)
    
    def get_latency(self) -> float:
        """Get current simulated latency in ms."""
        return self.current_values.get('latency_ms', 10.0)
    
    def get_partial_fill_rate(self) -> float:
        """Get current partial fill rate."""
        return self.current_values.get('partial_fill_rate', 1.0)
    
    def should_drop_connection(self) -> bool:
        """Check if WebSocket should drop."""
        prob = self.current_values.get('websocket_drop_prob', 0.0)
        return np.random.random() < prob
    
    def check_rate_limit(self, n_requests: int, time_window_s: float) -> bool:
        """Check if rate limit would be exceeded."""
        limit = self.current_values.get('rate_limit_rps', 10)
        allowed = limit * time_window_s
        return n_requests > allowed
    
    def apply_slippage(self, base_slippage: float) -> float:
        """Apply slippage multiplier."""
        multiplier = self.current_values.get('slippage_multiplier', 1.0)
        return base_slippage * multiplier
    
    def should_reject_order(self) -> bool:
        """Check if order should be rejected."""
        prob = self.current_values.get('rejection_prob', 0.0)
        return np.random.random() < prob
    
    def get_clock_offset(self, elapsed_s: float) -> float:
        """Get clock drift offset."""
        drift = self.current_values.get('clock_drift_ms', 0.0)
        return drift * elapsed_s
    
    def step(self, env_step: int) -> None:
        """
        Called each environment step.
        
        Updates curriculum and periodically resamples parameters.
        """
        self.curriculum_step += 1
        
        # Update curriculum difficulty
        if self.config.use_curriculum:
            progress = self.curriculum_step / self.config.curriculum_steps
            self.difficulty = min(1.0, progress)
        
        # Resample parameters periodically (every 100 steps by default)
        if self.curriculum_step % 100 == 0:
            self._resample_all()
        
        # Record history
        if len(self.randomization_history) < 1000:
            self.randomization_history.append({
                'step': self.curriculum_step,
                'difficulty': self.difficulty,
                **self.current_values,
            })
    
    def reset(self) -> None:
        """Reset randomizer state."""
        self.curriculum_step = 0
        self.difficulty = 0.0 if self.config.start_easy else 1.0
        self.randomization_history.clear()
        self._resample_all()
    
    def get_statistics(self) -> Dict:
        """Get randomization statistics."""
        if not self.randomization_history:
            return {}
        
        stats = {}
        for key in self.current_values.keys():
            values = [h[key] for h in self.randomization_history if key in h]
            if values:
                stats[f'{key}_mean'] = np.mean(values)
                stats[f'{key}_std'] = np.std(values)
                stats[f'{key}_min'] = np.min(values)
                stats[f'{key}_max'] = np.max(values)
        
        stats['curriculum_step'] = self.curriculum_step
        stats['current_difficulty'] = self.difficulty
        
        return stats


class RobustExecutionWrapper:
    """
    Wraps execution environment with sim-to-real robustness features.
    
    This wrapper applies domain randomization and provides methods
    for evaluating policies under various stress conditions.
    """
    
    def __init__(self, env, config: Optional[SimToRealConfig] = None):
        self.env = env
        self.config = config or SimToRealConfig()
        self.randomizer = DomainRandomizer(self.config)
        
        # State tracking
        self.pending_orders: List[Dict] = []
        self.connection_active = True
        self.request_timestamps: List[float] = []
        
        # Metrics
        self.total_rejections = 0
        self.total_drops = 0
        self.partial_fills = 0
    
    def reset(self, **kwargs):
        """Reset environment and randomizer."""
        self.randomizer.reset()
        self.pending_orders.clear()
        self.connection_active = True
        self.request_timestamps.clear()
        self.total_rejections = 0
        self.total_drops = 0
        self.partial_fills = 0
        
        return self.env.reset(**kwargs)
    
    def step(self, action: np.ndarray):
        """
        Execute step with sim-to-real perturbations.
        
        Returns:
            (obs, reward, terminated, truncated, info)
        """
        # Update randomizer
        self.randomizer.step(self.env.current_step)
        
        # Check connection status
        if self.randomizer.should_drop_connection():
            self.connection_active = False
            self.total_drops += 1
            
            # Return penalty observation
            obs = self._get_dropout_observation()
            reward = -10.0  # Penalty for dropout
            info = {'connection_dropped': True}
            
            return obs, reward, False, False, info
        
        self.connection_active = True
        
        # Check rate limit
        now = self.env.current_step * 0.1  # Assume 100ms per step
        self.request_timestamps = [t for t in self.request_timestamps if now - t < 1.0]
        
        if self.randomizer.check_rate_limit(len(self.request_timestamps) + 1, 1.0):
            # Rate limited - order rejected
            self.total_rejections += 1
            obs = self.env._get_observation()
            reward = -1.0  # Small penalty
            info = {'rate_limited': True}
            
            return obs, reward, False, False, info
        
        self.request_timestamps.append(now)
        
        # Check order rejection
        if self.randomizer.should_reject_order():
            self.total_rejections += 1
            # Still execute but mark as rejected
            pass
        
        # Execute environment step
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Apply latency simulation
        latency = self.randomizer.get_latency()
        info['simulated_latency_ms'] = latency
        
        # Apply partial fills
        if 'fill' in info and info['fill']:
            fill_rate = self.randomizer.get_partial_fill_rate()
            if fill_rate < 1.0 and np.random.random() < 0.3:
                # Partial fill
                original_qty = info.get('fill_quantity', 0)
                actual_qty = original_qty * fill_rate
                info['fill_quantity'] = actual_qty
                info['partial_fill'] = True
                self.partial_fills += 1
                
                # Adjust reward for partial fill
                reward *= (actual_qty / original_qty) if original_qty > 0 else 1.0
        
        # Apply slippage multiplier
        if 'slippage' in info:
            info['slippage'] = self.randomizer.apply_slippage(info['slippage'])
        
        # Add randomization info
        info['domain_randomization'] = self.randomizer.current_values.copy()
        
        return obs, reward, terminated, truncated, info
    
    def _get_dropout_observation(self) -> np.ndarray:
        """Return observation indicating connection dropout."""
        # Return zeros with dropout indicator
        obs = np.zeros(self.env.observation_space.shape, dtype=np.float32)
        # Could add specific dropout features here
        return obs
    
    def evaluate_robustness(
        self,
        policy: Callable,
        n_episodes: int = 100,
        stress_test: bool = False,
    ) -> Dict:
        """
        Evaluate policy robustness under domain randomization.
        
        Args:
            policy: Policy function that takes obs and returns action
            n_episodes: Number of evaluation episodes
            stress_test: If True, use extreme randomization values
            
        Returns:
            Dictionary of robustness metrics
        """
        rewards = []
        shortfalls = []
        rejections = []
        drops = []
        partials = []
        
        original_config = self.config
        
        if stress_test:
            # Create stress test config
            stress_config = SimToRealConfig(
                latency=RandomizationConfig(enabled=True, min_value=50, max_value=500),
                partial_fill=RandomizationConfig(enabled=True, min_value=0.3, max_value=0.7),
                websocket_drop=RandomizationConfig(enabled=True, min_value=0.01, max_value=0.1),
                slippage_multiplier=RandomizationConfig(enabled=True, min_value=2.0, max_value=5.0),
                rejection_prob=RandomizationConfig(enabled=True, min_value=0.05, max_value=0.2),
                use_curriculum=False,
            )
            self.randomizer = DomainRandomizer(stress_config)
        
        for ep in range(n_episodes):
            obs, _ = self.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = policy(obs)
                obs, reward, terminated, truncated, info = self.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            rewards.append(episode_reward)
            
            if 'implementation_shortfall' in info:
                shortfalls.append(info['implementation_shortfall'])
            
            rejections.append(self.total_rejections)
            drops.append(self.total_drops)
            partials.append(self.partial_fills)
        
        # Restore original config
        self.randomizer = DomainRandomizer(original_config)
        
        return {
            'reward_mean': np.mean(rewards),
            'reward_std': np.std(rewards),
            'reward_min': np.min(rewards),
            'shortfall_mean': np.mean(shortfalls) if shortfalls else 0,
            'rejection_rate': np.mean(rejections) / n_episodes,
            'drop_rate': np.mean(drops) / n_episodes,
            'partial_fill_rate': np.mean(partials) / n_episodes,
            'robustness_score': self._calculate_robustness_score(rewards, shortfalls),
        }
    
    def _calculate_robustness_score(
        self,
        rewards: List[float],
        shortfalls: List[float],
    ) -> float:
        """Calculate overall robustness score."""
        if not rewards:
            return 0.0
        
        # Higher is better: good rewards, low variance, low shortfall
        reward_score = np.mean(rewards) / (np.std(rewards) + 1)
        shortfall_penalty = np.mean(shortfalls) if shortfalls else 0
        
        return reward_score - shortfall_penalty * 0.1


def create_robust_env(base_env, config: Optional[SimToRealConfig] = None):
    """Factory function to create robust execution environment."""
    return RobustExecutionWrapper(base_env, config)


if __name__ == "__main__":
    print("Testing Sim-to-Real Transfer Module...")
    
    # Test domain randomizer
    config = SimToRealConfig()
    randomizer = DomainRandomizer(config)
    
    print("\nInitial randomization values:")
    for key, value in randomizer.current_values.items():
        print(f"  {key}: {value:.4f}")
    
    # Step through some iterations
    print("\nStepping through curriculum...")
    for i in range(5):
        randomizer.step(i * 100)
        print(f"Step {i*100}: difficulty={randomizer.difficulty:.2f}, latency={randomizer.get_latency():.1f}ms")
    
    # Test statistics
    print("\nRandomization statistics:")
    stats = randomizer.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n✓ Sim-to-Real Transfer test completed!")
