# python/ml/reward_functions.py
# =============================================================================
# STAGE 2 - CHAPTER 3 - FILE 2
# Focus: Advanced reward shaping for RL trading agent.
# Heavily penalizes drawdowns, slippage; rewards Sortino ratio improvements.
# =============================================================================

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class RewardType(Enum):
    """Types of rewards/penalties."""
    PROFIT = "profit"
    LOSS = "loss"
    TRANSACTION_COST = "transaction_cost"
    DRAWDOWN = "drawdown"
    SHARPE_IMPROVEMENT = "sharpe_improvement"
    SORTINO_IMPROVEMENT = "sortino_improvement"
    LIQUIDITY_SWEEP = "liquidity_sweep"
    SLIPPAGE = "slippage"


@dataclass
class RewardComponents:
    """Container for individual reward components."""
    profit_reward: float = 0.0
    loss_penalty: float = 0.0
    transaction_cost_penalty: float = 0.0
    drawdown_penalty: float = 0.0
    sharpe_bonus: float = 0.0
    sortino_bonus: float = 0.0
    liquidity_bonus: float = 0.0
    slippage_penalty: float = 0.0
    
    @property
    def total(self) -> float:
        """Calculate total reward."""
        return (
            self.profit_reward +
            self.loss_penalty +
            self.transaction_cost_penalty +
            self.drawdown_penalty +
            self.sharpe_bonus +
            self.sortino_bonus +
            self.liquidity_bonus +
            self.slippage_penalty
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            'profit_reward': self.profit_reward,
            'loss_penalty': self.loss_penalty,
            'transaction_cost_penalty': self.transaction_cost_penalty,
            'drawdown_penalty': self.drawdown_penalty,
            'sharpe_bonus': self.sharpe_bonus,
            'sortino_bonus': self.sortino_bonus,
            'liquidity_bonus': self.liquidity_bonus,
            'slippage_penalty': self.slippage_penalty,
            'total': self.total,
        }


class AsymmetricRewardShaper:
    """
    Advanced reward shaper with asymmetric penalties.
    
    Key principles:
    1. Losses hurt more than gains feel good (loss aversion)
    2. Drawdowns are heavily penalized exponentially
    3. Transaction costs always reduce reward
    4. Risk-adjusted returns (Sortino) are rewarded
    5. Successful liquidity sweeps get bonus
    """
    
    def __init__(
        self,
        loss_aversion_factor: float = 2.5,  # Losses hurt 2.5x more
        drawdown_exponent: float = 2.0,     # Exponential penalty for drawdowns
        max_drawdown_threshold: float = 0.10,  # 10% max drawdown before severe penalty
        transaction_cost_weight: float = 1.0,
        sortino_weight: float = 0.5,
        liquidity_bonus_weight: float = 0.3,
    ):
        self.loss_aversion_factor = loss_aversion_factor
        self.drawdown_exponent = drawdown_exponent
        self.max_drawdown_threshold = max_drawdown_threshold
        self.transaction_cost_weight = transaction_cost_weight
        self.sortino_weight = sortino_weight
        self.liquidity_bonus_weight = liquidity_bonus_weight
        
        # State tracking
        self.peak_balance = 0.0
        self.initial_balance = 0.0
        self.returns_history: List[float] = []
        self.downside_returns: List[float] = []
        
        # Liquidity sweep tracking
        self._prev_order_book_imbalance = 0.0
        self._sweep_detected = False
    
    def reset(self, initial_balance: float):
        """Reset reward shaper state for new episode."""
        self.initial_balance = initial_balance
        self.peak_balance = initial_balance
        self.returns_history = []
        self.downside_returns = []
        self._prev_order_book_imbalance = 0.0
        self._sweep_detected = False
    
    def calculate_reward(
        self,
        current_balance: float,
        realized_pnl: float,
        unrealized_pnl: float,
        transaction_costs: float,
        current_drawdown: float,
        order_book_imbalance: float,
        trade_direction: int = 0,  # 1=buy, -1=sell, 0=no trade
    ) -> RewardComponents:
        """
        Calculate comprehensive reward with all components.
        
        Args:
            current_balance: Current account balance
            realized_pnl: Realized PnL from closed trades
            unrealized_pnl: Unrealized PnL from open positions
            transaction_costs: Total transaction costs (commission + slippage)
            current_drawdown: Current drawdown from peak (0.0 to 1.0)
            order_book_imbalance: Current order book imbalance (-1 to 1)
            trade_direction: Direction of last trade (1=buy, -1=sell)
        
        Returns:
            RewardComponents with all individual rewards/penalties
        """
        components = RewardComponents()
        
        # Update peak balance
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        
        # Calculate current drawdown
        if self.peak_balance > 0:
            current_drawdown = (self.peak_balance - current_balance) / self.peak_balance
        else:
            current_drawdown = 0.0
        
        # 1. Profit/Loss Reward (asymmetric)
        total_pnl = realized_pnl + unrealized_pnl
        if total_pnl > 0:
            components.profit_reward = total_pnl / self.initial_balance
        else:
            # Apply loss aversion - losses hurt more
            components.loss_penalty = -abs(total_pnl) * self.loss_aversion_factor / self.initial_balance
        
        # Record return for Sharpe/Sortino calculation
        if self.initial_balance > 0:
            period_return = total_pnl / self.initial_balance
            self.returns_history.append(period_return)
            if period_return < 0:
                self.downside_returns.append(period_return)
        
        # 2. Transaction Cost Penalty
        if transaction_costs > 0 and self.initial_balance > 0:
            components.transaction_cost_penalty = (
                -transaction_costs * self.transaction_cost_weight / self.initial_balance
            )
        
        # 3. Drawdown Penalty (exponential)
        if current_drawdown > 0:
            # Mild penalty for small drawdowns, severe for large ones
            if current_drawdown <= self.max_drawdown_threshold:
                # Linear penalty up to threshold
                components.drawdown_penalty = -current_drawdown
            else:
                # Exponential penalty beyond threshold
                excess_dd = current_drawdown - self.max_drawdown_threshold
                components.drawdown_penalty = -(
                    self.max_drawdown_threshold + 
                    (excess_dd ** self.drawdown_exponent) * 10
                )
        
        # 4. Sharpe Ratio Improvement Bonus
        if len(self.returns_history) >= 10:
            current_sharpe = self._calculate_sharpe()
            # Compare to previous Sharpe (simplified - would need history)
            if current_sharpe > 0:
                components.sharpe_bonus = current_sharpe * 0.1  # Small bonus
        
        # 5. Sortino Ratio Improvement Bonus (more important than Sharpe)
        if len(self.downside_returns) >= 5:
            current_sortino = self._calculate_sortino()
            if current_sortino > 0:
                components.sortino_bonus = (
                    current_sortino * self.sortino_weight * 0.2
                )
        
        # 6. Liquidity Sweep Detection Bonus
        sweep_bonus = self._detect_liquidity_sweep(
            order_book_imbalance, trade_direction
        )
        components.liquidity_bonus = sweep_bonus * self.liquidity_bonus_weight
        
        # 7. Slippage Penalty (separate from transaction costs)
        # This is additional penalty for adverse price movement during execution
        if trade_direction != 0:
            slippage_penalty = self._estimate_slippage_penalty(
                trade_direction, order_book_imbalance
            )
            components.slippage_penalty = slippage_penalty
        
        return components
    
    def _calculate_sharpe(self, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio from returns history."""
        if len(self.returns_history) < 2:
            return 0.0
        
        mean_return = np.mean(self.returns_history)
        std_return = np.std(self.returns_history)
        
        if std_return == 0:
            return 0.0
        
        # Annualize (assuming daily returns)
        sharpe = (mean_return - risk_free_rate) / std_return * np.sqrt(252)
        return sharpe
    
    def _calculate_sortino(self, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sortino ratio (uses downside deviation instead of total std).
        More appropriate for trading strategies.
        """
        if len(self.returns_history) < 2 or len(self.downside_returns) < 2:
            return 0.0
        
        mean_return = np.mean(self.returns_history)
        
        # Downside deviation (only negative returns)
        downside_std = np.std(self.downside_returns)
        
        if downside_std == 0:
            # If no downside volatility, return high value if positive returns
            return 10.0 if mean_return > 0 else 0.0
        
        # Annualize
        sortino = (mean_return - risk_free_rate) / downside_std * np.sqrt(252)
        return sortino
    
    def _detect_liquidity_sweep(
        self, 
        current_imbalance: float, 
        trade_direction: int
    ) -> float:
        """
        Detect if the agent successfully swept liquidity.
        
        A liquidity sweep occurs when:
        - Large order book imbalance exists
        - Agent trades in the direction that absorbs the imbalance
        - Price moves favorably after the trade
        """
        bonus = 0.0
        
        # Check for significant imbalance
        if abs(current_imbalance) > 0.3:
            # Check if trading with the imbalance (absorbing liquidity)
            if trade_direction > 0 and current_imbalance > 0:
                # Buying when asks dominate (absorbing sell pressure)
                bonus = 0.5
                self._sweep_detected = True
            elif trade_direction < 0 and current_imbalance < 0:
                # Selling when bids dominate (absorbing buy pressure)
                bonus = 0.5
                self._sweep_detected = True
        
        # Reset detection flag
        if trade_direction == 0:
            self._sweep_detected = False
        
        return bonus
    
    def _estimate_slippage_penalty(
        self,
        trade_direction: int,
        order_book_imbalance: float
    ) -> float:
        """
        Estimate penalty for adverse slippage.
        
        Slippage is worse when:
        - Trading against order book imbalance
        - Low liquidity (high imbalance often indicates this)
        """
        penalty = 0.0
        
        # Penalty for trading against imbalance
        if trade_direction > 0 and order_book_imbalance < 0:
            # Buying when bids dominate (chasing price)
            penalty = abs(order_book_imbalance) * 0.1
        elif trade_direction < 0 and order_book_imbalance > 0:
            # Selling when asks dominate (panic selling)
            penalty = abs(order_book_imbalance) * 0.1
        
        return -penalty


class PortfolioRewardAggregator:
    """
    Aggregates rewards across multiple symbols/positions.
    Applies diversification bonus and correlation penalties.
    """
    
    def __init__(self, reward_shaper: AsymmetricRewardShaper):
        self.reward_shaper = reward_shaper
        self.symbol_rewards: Dict[str, RewardComponents] = {}
    
    def add_symbol_reward(self, symbol: str, components: RewardComponents):
        """Add reward components for a symbol."""
        self.symbol_rewards[symbol] = components
    
    def get_aggregated_reward(
        self,
        apply_diversification_bonus: bool = True,
        apply_correlation_penalty: bool = True
    ) -> Tuple[float, Dict[str, float]]:
        """
        Get aggregated reward across all symbols.
        
        Args:
            apply_diversification_bonus: Bonus for uncorrelated positions
            apply_correlation_penalty: Penalty for highly correlated positions
        
        Returns:
            Tuple of (total_reward, breakdown_by_symbol)
        """
        if not self.symbol_rewards:
            return 0.0, {}
        
        # Sum individual rewards
        total = sum(c.total for c in self.symbol_rewards.values())
        breakdown = {k: v.total for k, v in self.symbol_rewards.items()}
        
        # Diversification bonus
        if apply_diversification_bonus and len(self.symbol_rewards) > 1:
            # Bonus for having multiple positions
            n_positions = len([r for r in self.symbol_rewards.values() if r.total != 0])
            if n_positions > 1:
                total *= (1 + 0.05 * (n_positions - 1))  # 5% bonus per additional position
        
        # Correlation penalty (simplified - would need actual correlation matrix)
        if apply_correlation_penalty:
            # Check if all rewards have same sign (highly correlated outcomes)
            signs = [np.sign(r.total) for r in self.symbol_rewards.values() if r.total != 0]
            if len(signs) > 1 and len(set(signs)) == 1:
                # All same sign - apply penalty
                total *= 0.9  # 10% penalty for perfect correlation
        
        return total, breakdown
    
    def clear(self):
        """Clear all symbol rewards."""
        self.symbol_rewards.clear()


# Example usage and testing
if __name__ == "__main__":
    # Test the reward shaper
    shaper = AsymmetricRewardShaper(
        loss_aversion_factor=2.5,
        drawdown_exponent=2.0
    )
    
    shaper.reset(initial_balance=10000.0)
    
    # Simulate a trade sequence
    print("Testing Asymmetric Reward Shaper")
    print("=" * 50)
    
    # Scenario 1: Profitable trade
    components = shaper.calculate_reward(
        current_balance=10100.0,
        realized_pnl=100.0,
        unrealized_pnl=0.0,
        transaction_costs=1.0,
        current_drawdown=0.0,
        order_book_imbalance=0.2,
        trade_direction=1
    )
    print(f"\nScenario 1 - Profitable Trade:")
    print(f"  Total Reward: {components.total:.4f}")
    print(f"  Components: {components.to_dict()}")
    
    # Scenario 2: Losing trade with drawdown
    shaper.returns_history.clear()
    components = shaper.calculate_reward(
        current_balance=9800.0,
        realized_pnl=-200.0,
        unrealized_pnl=-50.0,
        transaction_costs=2.0,
        current_drawdown=0.05,
        order_book_imbalance=-0.3,
        trade_direction=-1
    )
    print(f"\nScenario 2 - Losing Trade:")
    print(f"  Total Reward: {components.total:.4f}")
    print(f"  Components: {components.to_dict()}")
    
    # Scenario 3: Severe drawdown
    shaper.peak_balance = 10000.0
    components = shaper.calculate_reward(
        current_balance=8500.0,
        realized_pnl=-1000.0,
        unrealized_pnl=-500.0,
        transaction_costs=10.0,
        current_drawdown=0.15,
        order_book_imbalance=0.1,
        trade_direction=0
    )
    print(f"\nScenario 3 - Severe Drawdown (15%):")
    print(f"  Total Reward: {components.total:.4f}")
    print(f"  Note: Exponential penalty applied!")
