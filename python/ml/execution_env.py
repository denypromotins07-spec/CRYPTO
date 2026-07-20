"""
Custom Gymnasium Environment for Execution Optimization
========================================================
Chapter 3, File 1: Python Deep Reinforcement Learning for Execution

A custom Gymnasium environment simulating the Binance matching engine, 
queue position, and latency. Focuses heavily on the cost of execution 
(slippage + maker/taker fees) for training RL agents to minimize 
implementation shortfall.

Target: Train execution agents to minimize market impact and slippage
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import IntEnum
import logging


class OrderType(IntEnum):
    """Order types available to the agent."""
    MARKET = 0
    LIMIT_PASSIVE = 1  # Limit order inside spread (maker)
    LIMIT_AGGRESSIVE = 2  # Limit order at touch (taker)
    ICEBERG = 3  # Iceberg order
    HOLD = 4  # Do nothing


@dataclass
class OrderBookState:
    """Snapshot of order book state."""
    best_bid: float
    best_ask: float
    bid_volume: float
    ask_volume: float
    spread_bps: float
    mid_price: float
    imbalance: float  # (bid_vol - ask_vol) / (bid_vol + ask_vol)
    
    def to_array(self) -> np.ndarray:
        """Convert to feature array."""
        return np.array([
            self.mid_price,
            self.spread_bps,
            self.imbalance,
            self.bid_volume,
            self.ask_volume,
        ], dtype=np.float32)


@dataclass
class ExecutionFill:
    """Represents a fill from an order."""
    price: float
    quantity: float
    fee: float
    is_maker: bool
    queue_position: int = 0
    latency_ms: float = 0.0


@dataclass
class MarketImpactModel:
    """Simple linear market impact model."""
    temporary_impact_coefficient: float = 1e-5  # Impact per unit
    permanent_impact_coefficient: float = 1e-6
    fixed_cost_bps: float = 1.0  # Fixed cost in basis points
    
    def calculate_impact(
        self, 
        quantity: float, 
        volume: float,
        direction: int  # 1 for buy, -1 for sell
    ) -> Tuple[float, float]:
        """
        Calculate market impact for given order size.
        
        Returns:
            (temporary_impact_bps, permanent_impact_bps)
        """
        if volume <= 0:
            return 0.0, 0.0
        
        participation_rate = abs(quantity) / volume
        
        # Temporary impact (square root law)
        temp_impact = (
            self.fixed_cost_bps + 
            self.temporary_impact_coefficient * np.sqrt(participation_rate * 1e6)
        )
        
        # Permanent impact (linear)
        perm_impact = self.permanent_impact_coefficient * abs(quantity)
        
        return temp_impact * direction, perm_impact * direction


class BinanceMatchingEngine:
    """
    Simplified simulation of Binance matching engine.
    
    Models:
    - Order book dynamics
    - Queue positioning
    - Fill probability for limit orders
    - Latency simulation
    """
    
    def __init__(
        self,
        initial_price: float = 100.0,
        volatility: float = 0.0001,
        tick_size: float = 0.01,
        lot_size: float = 0.001,
    ):
        self.initial_price = initial_price
        self.current_price = initial_price
        self.volatility = volatility
        self.tick_size = tick_size
        self.lot_size = lot_size
        
        # Order book state
        self.spread_bps = 1.0  # Initial spread in bps
        self.book_depth = 100.0  # Base depth in units
        
        # Queue simulation
        self.queue_position = 0
        self.orders_ahead = 0
        
        # Latency model
        self.base_latency_ms = 5.0
        self.latency_std_ms = 2.0
        
        # Fee structure (Binance VIP levels)
        self.maker_fee_bps = 2.0
        self.taker_fee_bps = 4.0
        
        # Impact model
        self.impact_model = MarketImpactModel()
        
        # Price history
        self.price_history: List[float] = [initial_price]
        
    def step(self, time_step: int) -> OrderBookState:
        """
        Simulate one time step of market evolution.
        
        Args:
            time_step: Current time step
            
        Returns:
            Current order book state
        """
        # Random walk with mean reversion
        noise = np.random.normal(0, self.volatility)
        
        # Mean reversion toward initial price
        mean_reversion = 0.001 * (self.initial_price - self.current_price)
        
        self.current_price += self.current_price * (noise + mean_reversion * 0.01)
        self.current_price = max(self.current_price, self.tick_size)
        
        # Round to tick size
        self.current_price = round(self.current_price / self.tick_size) * self.tick_size
        
        # Update spread (wider during volatile moves)
        self.spread_bps = max(0.5, 1.0 + abs(noise) * 100)
        
        # Update book depth (less depth during volatility)
        self.book_depth = max(10, 100 * (1 - abs(noise) * 10))
        
        # Add to history
        self.price_history.append(self.current_price)
        
        return self.get_state()
    
    def get_state(self) -> OrderBookState:
        """Get current order book state."""
        half_spread = (self.spread_bps / 10000) * self.current_price / 2
        
        best_bid = self.current_price - half_spread
        best_ask = self.current_price + half_spread
        
        # Volume based on depth
        bid_vol = self.book_depth * (1 + np.random.uniform(-0.2, 0.2))
        ask_vol = self.book_depth * (1 + np.random.uniform(-0.2, 0.2))
        
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-6)
        
        return OrderBookState(
            best_bid=best_bid,
            best_ask=best_ask,
            bid_volume=bid_vol,
            ask_volume=ask_vol,
            spread_bps=self.spread_bps,
            mid_price=self.current_price,
            imbalance=imbalance,
        )
    
    def execute_order(
        self,
        order_type: OrderType,
        side: int,  # 1 for buy, -1 for sell
        quantity: float,
        limit_price: Optional[float] = None,
    ) -> Optional[ExecutionFill]:
        """
        Execute an order and return fill details.
        
        Args:
            order_type: Type of order
            side: Buy (1) or Sell (-1)
            quantity: Order quantity
            limit_price: Limit price for limit orders
            
        Returns:
            ExecutionFill or None if not filled
        """
        state = self.get_state()
        
        # Simulate latency
        latency = np.random.normal(self.base_latency_ms, self.latency_std_ms)
        latency = max(1.0, latency)
        
        if order_type == OrderType.MARKET:
            # Immediate fill at adverse price
            fill_price = state.best_ask if side == 1 else state.best_bid
            
            # Apply market impact
            temp_impact, perm_impact = self.impact_model.calculate_impact(
                quantity, self.book_depth, side
            )
            
            # Adjust price for impact
            fill_price *= (1 + temp_impact / 10000)
            
            # Update permanent impact
            self.initial_price *= (1 + perm_impact / 10000)
            
            fee = abs(fill_price * quantity) * (self.taker_fee_bps / 10000)
            
            return ExecutionFill(
                price=fill_price,
                quantity=quantity,
                fee=fee,
                is_maker=False,
                latency_ms=latency,
            )
            
        elif order_type == OrderType.LIMIT_PASSIVE:
            # Place limit order inside spread
            if limit_price is None:
                if side == 1:  # Buy
                    limit_price = state.best_bid + self.tick_size
                else:  # Sell
                    limit_price = state.best_ask - self.tick_size
            
            # Fill probability based on queue position and market movement
            fill_prob = self._calculate_limit_fill_probability(
                limit_price, side, state
            )
            
            if np.random.random() < fill_prob:
                fee = abs(limit_price * quantity) * (self.maker_fee_bps / 10000)
                return ExecutionFill(
                    price=limit_price,
                    quantity=quantity,
                    fee=fee,
                    is_maker=True,
                    queue_position=self.orders_ahead,
                    latency_ms=latency,
                )
            
        elif order_type == OrderType.LIMIT_AGGRESSIVE:
            # Place at touch (immediate fill likely)
            if limit_price is None:
                limit_price = state.best_ask if side == 1 else state.best_bid
            
            # High fill probability
            if np.random.random() < 0.95:
                fee = abs(limit_price * quantity) * (self.taker_fee_bps / 10000)
                return ExecutionFill(
                    price=limit_price,
                    quantity=quantity,
                    fee=fee,
                    is_maker=False,
                    latency_ms=latency,
                )
                
        elif order_type == OrderType.ICEBERG:
            # Iceberg order: partial fills over time
            visible_qty = quantity * 0.1  # 10% visible
            fill_prob = min(0.5, self.book_depth / (quantity * 10))
            
            if np.random.random() < fill_prob:
                fill_qty = visible_qty * np.random.uniform(0.5, 1.0)
                fill_price = state.best_ask if side == 1 else state.best_bid
                
                fee = abs(fill_price * fill_qty) * (self.maker_fee_bps / 10000)
                return ExecutionFill(
                    price=fill_price,
                    quantity=fill_qty,
                    fee=fee,
                    is_maker=True,
                    latency_ms=latency,
                )
        
        return None
    
    def _calculate_limit_fill_probability(
        self,
        limit_price: float,
        side: int,
        state: OrderBookState,
    ) -> float:
        """Calculate probability of limit order fill."""
        # Distance from mid price
        if side == 1:  # Buy
            distance = (state.mid_price - limit_price) / state.mid_price
        else:  # Sell
            distance = (limit_price - state.mid_price) / state.mid_price
        
        distance_bps = distance * 10000
        
        # Base probability decreases with distance
        base_prob = np.exp(-abs(distance_bps) / 10)
        
        # Adjust for imbalance
        imbalance_factor = 1.0 + side * state.imbalance * 0.2
        
        # Queue position penalty
        queue_penalty = 1.0 / (1 + self.orders_ahead / 10)
        
        prob = base_prob * imbalance_factor * queue_penalty
        return np.clip(prob, 0.01, 0.99)
    
    def reset(self, initial_price: Optional[float] = None) -> None:
        """Reset the matching engine."""
        if initial_price is not None:
            self.initial_price = initial_price
        self.current_price = self.initial_price
        self.price_history = [self.initial_price]
        self.spread_bps = 1.0
        self.book_depth = 100.0
        self.queue_position = 0
        self.orders_ahead = 0


class ExecutionEnvironment(gym.Env):
    """
    Gymnasium environment for execution optimization.
    
    The agent learns to execute orders while minimizing:
    - Implementation shortfall
    - Market impact
    - Timing risk
    
    State space: Order book features, inventory, time remaining
    Action space: Order type, size, price offset
    Reward: Negative of execution cost (to be maximized)
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(
        self,
        initial_price: float = 100.0,
        total_quantity: float = 10.0,
        time_horizon: int = 100,
        max_order_size: float = 1.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        self.initial_price = initial_price
        self.total_quantity = total_quantity
        self.time_horizon = time_horizon
        self.max_order_size = max_order_size
        self.render_mode = render_mode
        
        # Initialize matching engine
        self.engine = BinanceMatchingEngine(initial_price=initial_price)
        
        # State variables
        self.remaining_quantity = total_quantity
        self.executed_quantity = 0.0
        self.executed_value = 0.0
        self.current_step = 0
        self.side = 1  # 1 for buy, -1 for sell (randomized in reset)
        
        # Benchmark (arrival price)
        self.arrival_price = initial_price
        
        # Cost tracking
        self.total_fees = 0.0
        self.total_slippage = 0.0
        
        # Define action space
        # [order_type (5), size_fraction (continuous), price_offset_bps (continuous)]
        self.action_space = spaces.Box(
            low=np.array([0, 0, -50], dtype=np.float32),
            high=np.array([4, 1, 50], dtype=np.float32),
            dtype=np.float32,
        )
        
        # Define observation space
        # Order book (5) + Inventory (3) + Time (2) + History (10) = 20
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(20,),
            dtype=np.float32,
        )
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Reset engine
        self.engine.reset(self.initial_price)
        
        # Randomize side
        self.side = np.random.choice([-1, 1])
        
        # Reset state
        self.remaining_quantity = self.total_quantity
        self.executed_quantity = 0.0
        self.executed_value = 0.0
        self.current_step = 0
        self.arrival_price = self.engine.current_price
        self.total_fees = 0.0
        self.total_slippage = 0.0
        
        # Get initial observation
        obs = self._get_observation()
        
        # Info dict
        info = {
            'remaining_qty': self.remaining_quantity,
            'executed_qty': self.executed_quantity,
            'avg_price': self.executed_value / max(self.executed_quantity, 1e-6),
            'arrival_price': self.arrival_price,
        }
        
        return obs, info
    
    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: [order_type_idx, size_fraction, price_offset_bps]
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Parse action
        order_type_idx = int(np.clip(action[0], 0, 4))
        size_fraction = np.clip(action[1], 0, 1)
        price_offset_bps = np.clip(action[2], -50, 50)
        
        order_type = OrderType(order_type_idx)
        
        # Calculate order size
        if order_type == OrderType.HOLD:
            order_size = 0.0
        else:
            order_size = min(
                self.remaining_quantity * size_fraction,
                self.max_order_size,
                self.remaining_quantity,
            )
        
        # Calculate limit price if needed
        state = self.engine.get_state()
        if price_offset_bps != 0 and order_type in [
            OrderType.LIMIT_PASSIVE,
            OrderType.LIMIT_AGGRESSIVE,
        ]:
            offset = (price_offset_bps / 10000) * state.mid_price
            if self.side == 1:  # Buying
                limit_price = state.best_ask + offset
            else:  # Selling
                limit_price = state.best_bid + offset
        else:
            limit_price = None
        
        # Execute order
        fill = None
        if order_size > 0:
            fill = self.engine.execute_order(
                order_type=order_type,
                side=self.side,
                quantity=order_size,
                limit_price=limit_price,
            )
        
        # Process fill
        if fill:
            self.executed_quantity += fill.quantity
            self.executed_value += fill.price * fill.quantity
            self.remaining_quantity -= fill.quantity
            self.total_fees += fill.fee
            
            # Calculate slippage
            if self.side == 1:  # Buy
                slippage = (fill.price - self.arrival_price) * fill.quantity
            else:  # Sell
                slippage = (self.arrival_price - fill.price) * fill.quantity
            self.total_slippage += slippage
        
        # Step the market
        self.engine.step(self.current_step)
        self.current_step += 1
        
        # Check termination
        terminated = (
            self.remaining_quantity <= 0 or
            self.current_step >= self.time_horizon
        )
        truncated = False
        
        # Calculate reward (negative implementation shortfall)
        reward = self._calculate_reward(fill)
        
        # Get new observation
        obs = self._get_observation()
        
        # Info dict
        info = {
            'remaining_qty': self.remaining_quantity,
            'executed_qty': self.executed_quantity,
            'total_qty': self.total_quantity,
            'avg_price': self.executed_value / max(self.executed_quantity, 1e-6),
            'arrival_price': self.arrival_price,
            'total_fees': self.total_fees,
            'total_slippage': self.total_slippage,
            'implementation_shortfall': self._calculate_implementation_shortfall(),
            'fill': fill is not None,
            'step': self.current_step,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        state = self.engine.get_state()
        
        # Order book features (normalized)
        ob_features = state.to_array() / self.initial_price
        
        # Inventory features
        inventory_features = np.array([
            self.remaining_quantity / self.total_quantity,
            self.executed_quantity / self.total_quantity,
            self.current_step / self.time_horizon,
        ], dtype=np.float32)
        
        # Time features
        time_features = np.array([
            self.current_step / self.time_horizon,
            1.0 - self.current_step / self.time_horizon,
        ], dtype=np.float32)
        
        # Price history (last 10 returns)
        if len(self.engine.price_history) >= 10:
            recent_prices = self.engine.price_history[-10:]
            returns = np.diff(recent_prices) / np.array(recent_prices[:-1])
        else:
            returns = np.zeros(9)
        history_features = np.pad(returns, (0, 1), 'constant')[:10]
        
        # Combine all features
        obs = np.concatenate([
            ob_features,
            inventory_features,
            time_features,
            history_features,
        ])
        
        return obs.astype(np.float32)
    
    def _calculate_reward(self, fill: Optional[ExecutionFill]) -> float:
        """
        Calculate reward for this step.
        
        Reward design:
        - Negative of implementation shortfall
        - Penalty for unexecuted quantity at end
        - Bonus for maker fills
        - Penalty for large orders causing impact
        """
        # Running cost
        current_is = self._calculate_implementation_shortfall()
        
        # Reward is negative cost
        reward = -current_is
        
        # Maker bonus
        if fill and fill.is_maker:
            reward += abs(fill.fee) * 0.5  # Half of fee saved
        
        # Small penalty for waiting (encourages execution)
        if self.remaining_quantity > 0:
            reward -= 0.001 * (self.remaining_quantity / self.total_quantity)
        
        return reward
    
    def _calculate_implementation_shortfall(self) -> float:
        """Calculate implementation shortfall in basis points."""
        if self.executed_quantity <= 0:
            return 0.0
        
        avg_price = self.executed_value / self.executed_quantity
        
        if self.side == 1:  # Buy
            # Paying more than arrival price is bad
            shortfall_bps = (avg_price - self.arrival_price) / self.arrival_price * 10000
        else:  # Sell
            # Receiving less than arrival price is bad
            shortfall_bps = (self.arrival_price - avg_price) / self.arrival_price * 10000
        
        # Add fees and remaining quantity penalty
        fees_bps = self.total_fees / self.executed_value * 10000
        remaining_penalty = (self.remaining_quantity / self.total_quantity) * 100
        
        return shortfall_bps + fees_bps + remaining_penalty
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            print(f"Step: {self.current_step}/{self.time_horizon}")
            print(f"Remaining: {self.remaining_quantity:.4f} / {self.total_quantity:.4f}")
            print(f"Executed: {self.executed_quantity:.4f} @ avg ${self.executed_value/max(self.executed_quantity, 1e-6):.2f}")
            print(f"Arrival price: ${self.arrival_price:.2f}")
            print(f"Implementation shortfall: {self._calculate_implementation_shortfall():.2f} bps")
            print("-" * 50)
    
    def close(self):
        """Clean up."""
        pass


# Register environment
def register_environment():
    """Register the environment with Gymnasium."""
    from gymnasium.envs.registration import register
    
    register(
        id='CryptoExecution-v0',
        entry_point='python.ml.execution_env:ExecutionEnvironment',
        max_episode_steps=100,
    )


if __name__ == "__main__":
    # Test the environment
    env = ExecutionEnvironment(
        initial_price=50000.0,
        total_quantity=10.0,
        time_horizon=50,
        render_mode="human",
    )
    
    print("Testing ExecutionEnvironment...")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Run random actions
    total_reward = 0.0
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if i % 10 == 0:
            env.render()
        
        if terminated or truncated:
            break
    
    print(f"\nTotal reward: {total_reward:.4f}")
    print(f"Final info: {info}")
    print("\n✓ Environment test completed!")
