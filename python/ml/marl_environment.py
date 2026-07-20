"""
Multi-Agent Reinforcement Learning Environment for Portfolio Management
========================================================================
Implements a PettingZoo/Gymnasium-compatible MARL environment where specialized
agents manage different aspects of trading:

1. Alpha Generator Agent - Generates trading signals from market data
2. Risk Manager Agent - Monitors and controls portfolio risk exposure  
3. Executor Agent - Optimizes order execution to minimize slippage

Uses cooperative-competitive reward structure where:
- All agents share portfolio P&L as base reward
- Risk Manager penalizes Alpha Generator for excessive drawdowns
- Executor is rewarded for beating implementation shortfall benchmarks

Memory-bounded design for 8GB RAM constraint with efficient state representation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import gymnasium as gym
from gymnasium import spaces


class AgentType(Enum):
    """Types of agents in the MARL system."""
    ALPHA_GENERATOR = "alpha"
    RISK_MANAGER = "risk"
    EXECUTOR = "executor"


@dataclass
class MarketState:
    """Current market state observation."""
    # Price features (normalized)
    returns_1m: float = 0.0
    returns_5m: float = 0.0
    returns_15m: float = 0.0
    returns_1h: float = 0.0
    
    # Volatility features
    volatility_1m: float = 0.0
    volatility_5m: float = 0.0
    rv_1h: float = 0.0  # Realized volatility
    
    # Order book features
    spread_bps: float = 0.0
    bid_ask_imbalance: float = 0.0
    order_flow_imbalance: float = 0.0
    depth_ratio: float = 0.0
    
    # Volume features
    volume_ratio: float = 0.0  # Current vs average
    vpin: float = 0.0  # Toxic flow estimate
    
    # Trend indicators
    rsi: float = 50.0
    momentum: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for neural network input."""
        return np.array([
            self.returns_1m, self.returns_5m, self.returns_15m, self.returns_1h,
            self.volatility_1m, self.volatility_5m, self.rv_1h,
            self.spread_bps, self.bid_ask_imbalance, self.order_flow_imbalance,
            self.depth_ratio, self.volume_ratio, self.vpin,
            self.rsi / 100.0,  # Normalize RSI
            self.momentum
        ], dtype=np.float32)
    
    @property
    def dim(self) -> int:
        return len(self.to_array())


@dataclass
class PortfolioState:
    """Current portfolio state."""
    # Positions
    positions: Dict[str, float] = field(default_factory=dict)
    
    # P&L
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_pnl: float = 0.0
    
    # Risk metrics
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    var_95: float = 0.0
    drawdown: float = 0.0
    max_drawdown: float = 0.0
    
    # Execution metrics
    pending_orders: int = 0
    fill_rate: float = 0.0
    avg_slippage_bps: float = 0.0
    
    def to_array(self, n_assets: int = 10) -> np.ndarray:
        """Convert to fixed-size array."""
        # Position features (top n_assets by absolute value)
        pos_sorted = sorted(self.positions.items(), key=lambda x: abs(x[1]), reverse=True)
        pos_features = [p[1] for _, p in zip(range(n_assets), pos_sorted)] if pos_sorted else []
        pos_features += [0.0] * max(0, n_assets - len(pos_features))
        
        return np.array([
            self.unrealized_pnl,
            self.realized_pnl,
            self.total_pnl,
            self.gross_exposure,
            self.net_exposure,
            self.var_95,
            self.drawdown,
            self.max_drawdown,
            self.pending_orders,
            self.fill_rate,
            self.avg_slippage_bps,
        ] + pos_features, dtype=np.float32)
    
    @property
    def dim(self) -> int:
        return 11  # Base features (positions handled separately)


@dataclass 
class ActionSpace:
    """Action specifications for each agent type."""
    # Alpha Generator: [-1, 1] signal per asset
    alpha_signal_dim: int = 10
    
    # Risk Manager: [0, 1] position limit multiplier per asset
    risk_limit_dim: int = 10
    
    # Executor: Aggressiveness [0, 1] per pending order
    executor_aggression_dim: int = 5


class MultiAgentTradingEnv(gym.Env):
    """
    Multi-Agent Trading Environment following PettingZoo API conventions.
    
    Agents:
    - alpha_generator: Produces directional signals
    - risk_manager: Sets position limits and risk constraints
    - executor: Determines execution aggressiveness
    
    Observation Space:
    - Shared market state
    - Agent-specific portfolio state
    
    Action Space:
    - Continuous actions normalized to [-1, 1] or [0, 1]
    
    Rewards:
    - Shared: Portfolio return
    - Individual: Role-specific bonuses/penalties
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'name': 'multi_agent_trading_v0'
    }
    
    def __init__(self, 
                 n_assets: int = 10,
                 max_positions: int = 20,
                 initial_capital: float = 1e6,
                 transaction_cost_bps: float = 5.0,
                 risk_free_rate: float = 0.02):
        super().__init__()
        
        self.n_assets = n_assets
        self.max_positions = max_positions
        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps
        self.risk_free_rate = risk_free_rate
        
        # Agent definitions
        self.agents = ['alpha_generator', 'risk_manager', 'executor']
        self.possible_agents = self.agents.copy()
        
        # Action spaces
        self.action_spaces = {
            'alpha_generator': spaces.Box(
                low=-1.0, high=1.0, shape=(n_assets,), dtype=np.float32
            ),
            'risk_manager': spaces.Box(
                low=0.0, high=1.0, shape=(n_assets,), dtype=np.float32
            ),
            'executor': spaces.Box(
                low=0.0, high=1.0, shape=(5,), dtype=np.float32
            ),
        }
        
        # Observation spaces (will be set in reset)
        self.observation_spaces = {}
        
        # State tracking
        self.current_step = 0
        self.max_steps = 1000  # Episodes are finite
        
        # Portfolio state
        self.capital = initial_capital
        self.positions = {}
        self.pnl_history = []
        self.max_portfolio_value = initial_capital
        
        # Market data (placeholder - would be fed from real data)
        self.market_state = MarketState()
        
        # Reward weights
        self.reward_weights = {
            'return': 1.0,
            'sharpe': 0.5,
            'drawdown_penalty': 2.0,
            'turnover_penalty': 0.1,
            'risk_bonus': 0.3,
            'execution_bonus': 0.2,
        }
        
    def reset(self, seed: Optional[int] = None, 
              options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.capital = self.initial_capital
        self.positions = {}
        self.pnl_history = [self.initial_capital]
        self.max_portfolio_value = self.initial_capital
        
        # Reset market state
        self.market_state = MarketState()
        
        # Set up observation spaces
        obs_dim = self.market_state.dim + self.max_positions + 10
        for agent in self.agents:
            self.observation_spaces[agent] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
        
        # Get initial observations
        observations = self._get_observations()
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
    
    def step(self, actions: Dict[str, np.ndarray]) -> \
             Tuple[Dict[str, np.ndarray], Dict[str, float], 
                   Dict[str, bool], Dict[str, bool], Dict[str, Dict]]:
        """
        Execute one environment step.
        
        Args:
            actions: Dictionary mapping agent names to actions
            
        Returns:
            observations, rewards, terminated, truncated, infos
        """
        self.current_step += 1
        
        # Process actions in order
        alpha_action = actions.get('alpha_generator', np.zeros(self.n_assets))
        risk_action = actions.get('risk_manager', np.ones(self.n_assets))
        executor_action = actions.get('executor', np.ones(5) * 0.5)
        
        # Apply alpha signals (with risk constraints)
        constrained_signals = alpha_action * risk_action
        self._apply_trading_signals(constrained_signals, executor_action)
        
        # Simulate market movement (placeholder)
        self._simulate_market_step()
        
        # Update portfolio valuation
        portfolio_value = self._calculate_portfolio_value()
        self.pnl_history.append(portfolio_value)
        
        # Track max for drawdown
        if portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = portfolio_value
        
        # Calculate rewards
        rewards = self._calculate_rewards()
        
        # Get new observations
        observations = self._get_observations()
        
        # Check termination
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps
        
        # Build info dicts
        infos = {
            agent: {
                'portfolio_value': portfolio_value,
                'step': self.current_step,
                'market_state': self.market_state,
            } for agent in self.agents
        }
        
        return observations, rewards, terminated, truncated, infos
    
    def _apply_trading_signals(self, signals: np.ndarray, 
                               execution_agg: np.ndarray):
        """Apply trading signals to portfolio."""
        # Simple implementation: adjust positions based on signals
        avg_agg = np.mean(execution_agg)
        
        for i in range(min(len(signals), self.n_assets)):
            signal = signals[i]
            asset_id = f"asset_{i}"
            
            # Position size proportional to signal and execution aggressiveness
            target_position = signal * avg_agg * 0.1  # Scale factor
            
            current_pos = self.positions.get(asset_id, 0.0)
            position_change = target_position - current_pos
            
            # Apply transaction costs
            cost = abs(position_change) * (self.transaction_cost_bps / 1e4)
            self.capital -= cost
            
            # Update position
            if abs(target_position) < 0.01:
                self.positions.pop(asset_id, None)
            else:
                self.positions[asset_id] = target_position
    
    def _simulate_market_step(self):
        """Simulate one step of market movement."""
        # Random walk with mean reversion (simplified)
        mean_rev = -0.01 * self.market_state.returns_1m
        shock = self.np_random.standard_normal() * 0.02
        
        # Update returns
        self.market_state.returns_1m = mean_rev + shock
        self.market_state.returns_5m *= 0.9 + 0.1 * self.market_state.returns_1m
        
        # Update volatility (GARCH-like)
        self.market_state.volatility_1m = (
            0.9 * self.market_state.volatility_1m + 
            0.1 * abs(shock)
        )
        
        # Random order book changes
        self.market_state.spread_bps = np.clip(
            self.market_state.spread_bps + self.np_random.standard_normal() * 0.5,
            1.0, 50.0
        )
        
        self.market_state.vpin = np.clip(
            self.market_state.vpin + self.np_random.standard_normal() * 0.05,
            0.0, 1.0
        )
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate current portfolio value."""
        # Simplified: assume assets move with market
        market_return = self.market_state.returns_5m
        
        position_value = sum(
            pos * market_return * 100  # Simplified P&L
            for pos in self.positions.values()
        )
        
        return self.capital + position_value
    
    def _calculate_rewards(self) -> Dict[str, float]:
        """Calculate individual rewards for each agent."""
        # Base portfolio return
        if len(self.pnl_history) < 2:
            base_return = 0.0
        else:
            prev_value = self.pnl_history[-2]
            curr_value = self.pnl_history[-1]
            base_return = (curr_value - prev_value) / prev_value if prev_value > 0 else 0.0
        
        # Drawdown penalty
        current_dd = 1.0 - (self.pnl_history[-1] / self.max_portfolio_value)
        dd_penalty = max(0.0, current_dd - 0.05)  # Penalty only beyond 5% DD
        
        # Sharpe approximation (rolling)
        if len(self.pnl_history) > 20:
            returns = np.diff(self.pnl_history[-20:]) / np.array(self.pnl_history[-20:-1])
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Agent-specific rewards
        rewards = {}
        
        # Alpha Generator: P&L focused but penalized for drawdowns
        rewards['alpha_generator'] = (
            self.reward_weights['return'] * base_return -
            self.reward_weights['drawdown_penalty'] * dd_penalty +
            self.reward_weights['sharpe'] * np.clip(sharpe, -2, 2) / 10
        )
        
        # Risk Manager: Penalized for drawdowns, rewarded for stability
        rewards['risk_manager'] = (
            self.reward_weights['return'] * base_return * 0.5 -
            self.reward_weights['drawdown_penalty'] * dd_penalty * 2.0 +
            self.reward_weights['risk_bonus'] * (1.0 - min(current_dd * 10, 1.0))
        )
        
        # Executor: Rewarded for good fills, penalized for slippage
        exec_bonus = self.reward_weights['execution_bonus'] * (1.0 - self.market_state.vpin)
        rewards['executor'] = (
            self.reward_weights['return'] * base_return * 0.3 +
            exec_bonus -
            self.reward_weights['turnover_penalty'] * abs(base_return) * 10
        )
        
        return rewards
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all agents."""
        market_obs = self.market_state.to_array()
        portfolio_obs = self._get_portfolio_features()
        
        # Concatenate features
        full_obs = np.concatenate([market_obs, portfolio_obs])
        
        return {agent: full_obs.copy() for agent in self.agents}
    
    def _get_portfolio_features(self) -> np.ndarray:
        """Extract portfolio features for observation."""
        features = [
            len(self.positions),  # Number of positions
            sum(abs(p) for p in self.positions.values()),  # Gross exposure
            sum(self.positions.values()),  # Net exposure
        ]
        
        # Top positions
        pos_sorted = sorted(self.positions.values(), key=abs, reverse=True)
        features.extend(pos_sorted[:self.max_positions])
        features.extend([0.0] * max(0, self.max_positions - len(pos_sorted)))
        
        # P&L
        if len(self.pnl_history) >= 1:
            current_pnl = self.pnl_history[-1] - self.initial_capital
            features.append(current_pnl / self.initial_capital)
        else:
            features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # Terminate on large drawdown
        if len(self.pnl_history) > 0:
            current_dd = 1.0 - (self.pnl_history[-1] / self.max_portfolio_value)
            if current_dd > 0.20:  # 20% max drawdown
                return True
        
        # Terminate if capital exhausted
        if self.capital < self.initial_capital * 0.5:
            return True
        
        return False
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: {self.pnl_history[-1]:,.2f}")
            print(f"Positions: {len(self.positions)}")
            print(f"Market Return: {self.market_state.returns_5m:.4f}")
            print("-" * 40)


# Convenience function for creating environment
def make_marl_trading_env(n_assets: int = 10, **kwargs) -> MultiAgentTradingEnv:
    """Factory function for creating MARL trading environment."""
    return MultiAgentTradingEnv(n_assets=n_assets, **kwargs)


if __name__ == "__main__":
    # Test the environment
    env = make_marl_trading_env(n_assets=5)
    
    obs, info = env.reset(seed=42)
    
    print("Initial observations:")
    for agent, ob in obs.items():
        print(f"  {agent}: shape={ob.shape}, mean={ob.mean():.4f}")
    
    # Run a few steps with random actions
    for step in range(10):
        actions = {
            agent: env.action_spaces[agent].sample()
            for agent in env.agents
        }
        
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        print(f"\nStep {step + 1}:")
        for agent, reward in rewards.items():
            print(f"  {agent}: reward={reward:.4f}")
        
        if terminated or truncated:
            break
    
    print("\nEnvironment test completed successfully!")
