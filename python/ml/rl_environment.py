# python/ml/rl_environment.py
# =============================================================================
# STAGE 2 - CHAPTER 3 - FILE 1
# Focus: Custom Gymnasium environment for trading with AMD Radeon GPU optimization.
# Target: Minimal VRAM/RAM usage while maximizing throughput.
# =============================================================================

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import threading
import time

# Try to import ROCm/DirectML for AMD GPU acceleration
try:
    import torch
    # Check for AMD GPU (ROCm)
    if torch.cuda.is_available() and 'AMD' in torch.cuda.get_device_name(0):
        DEVICE = torch.device('cuda')
        print(f"[RL] Using AMD GPU: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = torch.device('cpu')
        print("[RL] Using CPU (no AMD GPU detected)")
except ImportError:
    DEVICE = torch.device('cpu')
    print("[RL] PyTorch not available, using CPU")


class TradingEnvironment(gym.Env):
    """
    Custom Gymnasium environment for crypto trading.
    
    Observation Space:
    - Order book features (bid/ask prices, volumes, spread)
    - Technical indicators (RSI, MACD, Bollinger Bands)
    - Order flow metrics (CVD, Imbalance)
    - Portfolio state (position, PnL, cash)
    
    Action Space:
    - Discrete: 0=Hold, 1=Buy, 2=Sell
    - Continuous: Position size (0.0 to 1.0 of available capital)
    """
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(
        self,
        feature_dim: int = 50,
        max_steps: int = 10000,
        initial_balance: float = 10000.0,
        commission_rate: float = 0.001,  # 0.1% Binance fee
        slippage_rate: float = 0.0005,   # 0.05% estimated slippage
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.max_steps = max_steps
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.render_mode = render_mode
        
        # Action space: [action_type (0-2), position_size (0.0-1.0)]
        # Using Box for continuous control
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([2.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation space: features + portfolio state
        # Features: price data, indicators, order flow
        # Portfolio: position, entry_price, unrealized_pnl, cash_ratio
        portfolio_state_dim = 5
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(feature_dim + portfolio_state_dim,),
            dtype=np.float32
        )
        
        # State variables
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0.0  # Positive = long, Negative = short
        self.entry_price = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        
        # Data buffers
        self._features_buffer: Optional[np.ndarray] = None
        self._current_features: Optional[np.ndarray] = None
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Performance metrics
        self.episode_rewards: List[float] = []
        self.episode_returns: List[float] = []
        
        # GPU tensors for fast computation (if available)
        if DEVICE.type == 'cuda':
            self._gpu_enabled = True
            self._state_tensor = torch.zeros(feature_dim + portfolio_state_dim, 
                                              dtype=torch.float32, device=DEVICE)
        else:
            self._gpu_enabled = False
            self._state_tensor = None
    
    def reset(
        self, 
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        with self._lock:
            self.current_step = 0
            self.balance = self.initial_balance
            self.position = 0.0
            self.entry_price = 0.0
            self.total_trades = 0
            self.winning_trades = 0
            
            # Reset episode tracking
            self.episode_rewards = []
            self.episode_returns = []
        
        # Get initial observation
        obs = self._get_observation()
        
        info = {
            'balance': self.balance,
            'position': self.position,
            'step': self.current_step,
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: [action_type, position_size]
                - action_type: 0=Hold, 1=Buy, 2=Sell
                - position_size: 0.0 to 1.0 of available capital
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        with self._lock:
            self.current_step += 1
            
            # Parse action
            action_type = int(np.clip(action[0], 0, 2))
            position_size = float(np.clip(action[1], 0, 1))
            
            # Get current price from features (assuming price is at index 4)
            current_price = self._current_features[4] if self._current_features is not None else 0.0
            
            # Execute action and calculate reward
            reward = self._execute_action(action_type, position_size, current_price)
            
            # Check termination conditions
            terminated = self._check_termination()
            truncated = self.current_step >= self.max_steps
            
            # Get new observation
            obs = self._get_observation()
            
            # Info dict
            info = {
                'balance': self.balance,
                'position': self.position,
                'entry_price': self.entry_price,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'step': self.current_step,
                'current_price': current_price,
            }
        
        return obs, reward, terminated, truncated, info
    
    def _execute_action(self, action_type: int, position_size: float, price: float) -> float:
        """
        Execute trading action and return immediate reward.
        
        Reward shaping:
        - Penalize excessive trading (commission costs)
        - Penalize drawdowns heavily
        - Reward risk-adjusted returns
        """
        reward = 0.0
        prev_balance = self.balance
        
        if action_type == 0:  # Hold
            # Update unrealized PnL
            if self.position != 0:
                pnl_pct = (price - self.entry_price) / self.entry_price
                if self.position > 0:  # Long
                    unrealized_pnl = self.position * pnl_pct
                else:  # Short
                    unrealized_pnl = -self.position * pnl_pct
                reward = unrealized_pnl * 0.01  # Small reward for holding profitable position
        
        elif action_type == 1:  # Buy
            if position_size > 0:
                # Close short position if exists
                if self.position < 0:
                    pnl = -self.position * (price - self.entry_price) / self.entry_price
                    self.balance += pnl
                    reward += pnl  # Reward realized PnL
                    if pnl > 0:
                        self.winning_trades += 1
                    self.total_trades += 1
                
                # Open long position
                trade_value = self.balance * position_size
                cost = trade_value * (self.commission_rate + self.slippage_rate)
                
                new_position = trade_value / price
                self.position = new_position
                self.entry_price = price
                self.balance -= cost
                reward -= cost / self.initial_balance  # Penalize transaction costs
        
        elif action_type == 2:  # Sell
            if position_size > 0:
                # Close long position if exists
                if self.position > 0:
                    pnl = self.position * (price - self.entry_price) / self.entry_price
                    self.balance += pnl
                    reward += pnl  # Reward realized PnL
                    if pnl > 0:
                        self.winning_trades += 1
                    self.total_trades += 1
                
                # Open short position
                trade_value = self.balance * position_size
                cost = trade_value * (self.commission_rate + self.slippage_rate)
                
                new_position = -(trade_value / price)
                self.position = new_position
                self.entry_price = price
                self.balance -= cost
                reward -= cost / self.initial_balance
        
        # Store episode reward
        self.episode_rewards.append(reward)
        
        return reward
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate (e.g., bankruptcy)."""
        if self.balance <= self.initial_balance * 0.5:  # 50% drawdown limit
            return True
        return False
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        if self._features_buffer is None or len(self._features_buffer) == 0:
            # Return zero observation if no data
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # Get latest features
        idx = min(self.current_step, len(self._features_buffer) - 1)
        self._current_features = self._features_buffer[idx].astype(np.float32)
        
        # Build portfolio state
        portfolio_state = np.array([
            self.position,  # Current position
            self.entry_price if self.position != 0 else 0.0,  # Entry price
            self.balance / self.initial_balance,  # Cash ratio
            self.total_trades / max(1, self.current_step),  # Trade frequency
            self.winning_trades / max(1, self.total_trades),  # Win rate
        ], dtype=np.float32)
        
        # Concatenate features and portfolio state
        observation = np.concatenate([self._current_features, portfolio_state])
        
        # Normalize observation for neural network input
        observation = self._normalize_observation(observation)
        
        # Copy to GPU if available
        if self._gpu_enabled:
            self._state_tensor.copy_(torch.from_numpy(observation).to(DEVICE))
        
        return observation
    
    def _normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation values for stable training."""
        # Apply tanh to bound extreme values
        obs_normalized = np.tanh(obs / 10.0)  # Scale down large values
        return obs_normalized
    
    def set_features(self, features: np.ndarray) -> None:
        """
        Set the feature buffer for the environment.
        
        Args:
            features: Array of shape (n_steps, feature_dim)
        """
        with self._lock:
            self._features_buffer = features.astype(np.float32)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get episode metrics."""
        total_return = (self.balance - self.initial_balance) / self.initial_balance
        
        if len(self.episode_rewards) > 1:
            sharpe = np.mean(self.episode_rewards) / (np.std(self.episode_rewards) + 1e-8)
        else:
            sharpe = 0.0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'win_rate': self.winning_trades / max(1, self.total_trades),
            'total_trades': self.total_trades,
            'final_balance': self.balance,
        }
    
    def render(self):
        """Render the environment (optional)."""
        if self.render_mode == 'human':
            print(f"Step: {self.current_step}, Balance: ${self.balance:.2f}, "
                  f"Position: {self.position:.4f}, Trades: {self.total_trades}")


class MultiSymbolEnvironment(gym.Env):
    """
    Environment wrapper for multi-symbol trading.
    Manages multiple TradingEnvironment instances in parallel.
    """
    
    def __init__(
        self,
        symbols: List[str],
        feature_dim: int = 50,
        **env_kwargs
    ):
        super().__init__()
        
        self.symbols = symbols
        self.n_symbols = len(symbols)
        
        # Create individual environments
        self.envs = [
            TradingEnvironment(feature_dim=feature_dim, **env_kwargs)
            for _ in range(self.n_symbols)
        ]
        
        # Combined action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0] * self.n_symbols, dtype=np.float32),
            high=np.array([2.0, 1.0] * self.n_symbols, dtype=np.float32),
            dtype=np.float32
        )
        
        obs_dim = (feature_dim + 5) * self.n_symbols
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        """Reset all environments."""
        observations = []
        for env in self.envs:
            obs, _ = env.reset(seed=seed)
            observations.append(obs)
        return np.concatenate(observations), {}
    
    def step(self, action):
        """Step all environments."""
        observations = []
        rewards = []
        dones = []
        infos = []
        
        for i, env in enumerate(self.envs):
            # Extract action for this symbol
            symbol_action = action[i*2:(i+1)*2]
            obs, reward, terminated, truncated, info = env.step(symbol_action)
            observations.append(obs)
            rewards.append(reward)
            dones.append(terminated or truncated)
            infos.append(info)
        
        return (
            np.concatenate(observations),
            np.sum(rewards),
            any(dones),
            False,
            {'individual_infos': infos}
        )
