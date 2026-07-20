"""
strategy_router.py
------------------
Dynamic capital allocation engine using Kelly Criterion and Risk Parity models.
Continuously evaluates active strategies based on real-time rolling Sharpe/Sortino ratios
and dynamically re-weights their allocated capital.

Optimized for low-latency rebalancing with pre-allocated arrays.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AllocationModel(Enum):
    """Capital allocation models."""
    EQUAL_WEIGHT = 1
    KELLY_CRITERION = 2
    RISK_PARITY = 3
    SHARPE_WEIGHTED = 4
    SORTINO_WEIGHTED = 5


@dataclass
class StrategyMetrics:
    """Real-time metrics for a strategy."""
    strategy_id: str
    allocated_capital: float
    current_exposure: float
    unrealized_pnl: float
    realized_pnl: float
    rolling_sharpe: float
    rolling_sortino: float
    max_drawdown: float
    win_rate: float
    volatility: float


class StrategyRouter:
    """
    Dynamic capital allocation router.
    
    Features:
    - Multiple allocation models (Kelly, Risk Parity, Sharpe-weighted)
    - Real-time performance tracking
    - Automatic rebalancing based on metrics
    - Strict capital limits per strategy
    """

    # Configuration constants
    MAX_STRATEGIES = 20
    DEFAULT_LOOKBACK_DAYS = 30
    MIN_CAPITAL_PER_STRATEGY = 100.0  # USDT
    MAX_CAPITAL_PER_STRATEGY_RATIO = 0.5  # Max 50% to single strategy
    
    def __init__(
        self,
        total_capital: float,
        allocation_model: AllocationModel = AllocationModel.RISK_PARITY,
        lookback_days: int = 30
    ):
        self._total_capital = total_capital
        self._allocation_model = allocation_model
        self._lookback_days = lookback_days
        
        # Pre-allocated arrays for strategy data
        self._strategy_ids: np.ndarray = np.zeros(self.MAX_STRATEGIES, dtype='U32')
        self._allocated_capital: np.ndarray = np.zeros(self.MAX_STRATEGIES, dtype='f8')
        self._current_exposure: np.ndarray = np.zeros(self.MAX_STRATEGIES, dtype='f8')
        self._rolling_returns: np.ndarray = np.zeros((self.MAX_STRATEGIES, lookback_days), dtype='f8')
        self._sharpe_ratios: np.ndarray = np.zeros(self.MAX_STRATEGIES, dtype='f8')
        self._sortino_ratios: np.ndarray = np.zeros(self.MAX_STRATEGIES, dtype='f8')
        self._max_drawdowns: np.ndarray = np.zeros(self.MAX_STRATEGIES, dtype='f8')
        self._volatilities: np.ndarray = np.zeros(self.MAX_STRATEGIES, dtype='f8')
        self._is_active: np.ndarray = np.zeros(self.MAX_STRATEGIES, dtype='i1')
        
        self._strategy_count = 0
        self._return_write_idx = 0
        
        # Risk-free rate (annualized)
        self._risk_free_rate = 0.05 / 252  # Daily
        
        # Last rebalance time
        self._last_rebalance = datetime.utcnow()
        self._rebalance_interval = timedelta(hours=1)
        
        # Lock for thread-safe updates
        self._lock = asyncio.Lock()

    def _find_strategy(self, strategy_id: str) -> int:
        """Find strategy index by ID. Returns -1 if not found."""
        mask = self._strategy_ids[:self._strategy_count] == strategy_id
        if np.any(mask):
            return int(np.argmax(mask))
        return -1

    def register_strategy(self, strategy_id: str, initial_allocation: float = None) -> bool:
        """Register a new strategy."""
        if self._strategy_count >= self.MAX_STRATEGIES:
            logger.error("Max strategy limit reached")
            return False
        
        if self._find_strategy(strategy_id) >= 0:
            logger.warning(f"Strategy {strategy_id} already registered")
            return False
        
        idx = self._strategy_count
        self._strategy_ids[idx] = strategy_id
        
        # Equal initial allocation if not specified
        if initial_allocation is None:
            initial_allocation = self._total_capital / (self._strategy_count + 1)
        
        # Enforce max allocation limit
        initial_allocation = min(
            initial_allocation,
            self._total_capital * self.MAX_CAPITAL_PER_STRATEGY_RATIO
        )
        
        self._allocated_capital[idx] = max(initial_allocation, self.MIN_CAPITAL_PER_STRATEGY)
        self._is_active[idx] = 1
        self._strategy_count += 1
        
        logger.info(f"Registered strategy: {strategy_id}, allocation={initial_allocation:.2f}")
        return True

    def unregister_strategy(self, strategy_id: str) -> bool:
        """Unregister a strategy (requires closing positions first)."""
        idx = self._find_strategy(strategy_id)
        if idx < 0:
            return False
        
        # Check if exposure is zero
        if abs(self._current_exposure[idx]) > 1e-6:
            logger.warning(f"Cannot unregister {strategy_id}: still has exposure")
            return False
        
        # Remove by swapping with last
        last_idx = self._strategy_count - 1
        if idx != last_idx:
            self._strategy_ids[idx] = self._strategy_ids[last_idx]
            self._allocated_capital[idx] = self._allocated_capital[last_idx]
            self._current_exposure[idx] = self._current_exposure[last_idx]
            self._rolling_returns[idx, :] = self._rolling_returns[last_idx, :]
            self._sharpe_ratios[idx] = self._sharpe_ratios[last_idx]
            self._sortino_ratios[idx] = self._sortino_ratios[last_idx]
            self._max_drawdowns[idx] = self._max_drawdowns[last_idx]
            self._volatilities[idx] = self._volatilities[last_idx]
            self._is_active[idx] = self._is_active[last_idx]
        
        # Clear last
        self._strategy_ids[last_idx] = ''
        self._is_active[last_idx] = 0
        self._strategy_count -= 1
        
        logger.info(f"Unregistered strategy: {strategy_id}")
        return True

    def update_return(self, strategy_id: str, daily_return: float) -> None:
        """Update daily return for a strategy."""
        idx = self._find_strategy(strategy_id)
        if idx < 0:
            return
        
        # Store return in circular buffer
        self._rolling_returns[idx, self._return_write_idx] = daily_return
        
        # Update metrics
        self._update_metrics(idx)

    def _update_metrics(self, idx: int) -> None:
        """Calculate updated metrics for a strategy."""
        returns = self._rolling_returns[idx, :]
        
        # Only use non-zero returns
        valid_returns = returns[returns != 0]
        
        if len(valid_returns) < 5:
            return  # Not enough data
        
        # Calculate mean and std
        mean_return = np.mean(valid_returns)
        std_return = np.std(valid_returns)
        
        # Annualized volatility
        self._volatilities[idx] = std_return * np.sqrt(252)
        
        # Sharpe ratio
        if std_return > 0:
            self._sharpe_ratios[idx] = (mean_return - self._risk_free_rate) / std_return * np.sqrt(252)
        else:
            self._sharpe_ratios[idx] = 0.0
        
        # Sortino ratio (downside deviation)
        negative_returns = valid_returns[valid_returns < 0]
        if len(negative_returns) > 0:
            downside_std = np.std(negative_returns)
            if downside_std > 0:
                self._sortino_ratios[idx] = (mean_return - self._risk_free_rate) / downside_std * np.sqrt(252)
            else:
                self._sortino_ratios[idx] = 0.0
        else:
            self._sortino_ratios[idx] = float('inf') if mean_return > 0 else 0.0
        
        # Max drawdown
        cumulative = np.cumprod(1 + valid_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        self._max_drawdowns[idx] = np.min(drawdown) if len(drawdown) > 0 else 0.0

    async def rebalance(self) -> Dict[str, float]:
        """
        Rebalance capital allocation based on current model.
        Returns dict of strategy_id -> new_allocation.
        """
        async with self._lock:
            now = datetime.utcnow()
            
            # Check rebalance interval
            if now - self._last_rebalance < self._rebalance_interval:
                return {}
            
            if self._strategy_count == 0:
                return {}
            
            logger.info(f"Rebalancing allocations using {self._allocation_model.name}")
            
            # Calculate new weights based on model
            weights = self._calculate_weights()
            
            # Apply weights to total capital
            new_allocations = {}
            for i in range(self._strategy_count):
                strategy_id = self._strategy_ids[i]
                new_alloc = weights[i] * self._total_capital
                
                # Enforce limits
                new_alloc = max(new_alloc, self.MIN_CAPITAL_PER_STRATEGY)
                new_alloc = min(new_alloc, self._total_capital * self.MAX_CAPITAL_PER_STRATEGY_RATIO)
                
                new_allocations[strategy_id] = new_alloc
            
            self._last_rebalance = now
            logger.info(f"Rebalance complete. New allocations: {new_allocations}")
            
            return new_allocations

    def _calculate_weights(self) -> np.ndarray:
        """Calculate allocation weights based on selected model."""
        if self._allocation_model == AllocationModel.EQUAL_WEIGHT:
            return np.ones(self._strategy_count) / self._strategy_count
        
        elif self._allocation_model == AllocationModel.KELLY_CRITERION:
            return self._kelly_weights()
        
        elif self._allocation_model == AllocationModel.RISK_PARITY:
            return self._risk_parity_weights()
        
        elif self._allocation_model == AllocationModel.SHARPE_WEIGHTED:
            return self._sharpe_weights()
        
        elif self._allocation_model == AllocationModel.SORTINO_WEIGHTED:
            return self._sortino_weights()
        
        else:
            return np.ones(self._strategy_count) / self._strategy_count

    def _kelly_weights(self) -> np.ndarray:
        """
        Calculate Kelly Criterion weights.
        f* = (p * b - q) / b where p=win_prob, b=win_loss_ratio, q=1-p
        Simplified version using Sharpe ratios.
        """
        sharpe_ratios = self._sharpe_ratios[:self._strategy_count].copy()
        
        # Kelly fraction approximated as Sharpe / Variance
        variances = self._volatilities[:self._strategy_count] ** 2
        kelly_fractions = np.zeros(self._strategy_count)
        
        mask = variances > 0
        kelly_fractions[mask] = sharpe_ratios[mask] / variances[mask]
        
        # Normalize and cap
        kelly_fractions = np.maximum(kelly_fractions, 0)  # No short strategies
        total = np.sum(kelly_fractions)
        
        if total > 0:
            return kelly_fractions / total
        return np.ones(self._strategy_count) / self._strategy_count

    def _risk_parity_weights(self) -> np.ndarray:
        """
        Calculate Risk Parity weights.
        Each strategy contributes equally to portfolio risk.
        """
        volatilities = self._volatilities[:self._strategy_count].copy()
        
        # Inverse volatility weighting
        inv_vol = np.zeros(self._strategy_count)
        mask = volatilities > 0
        inv_vol[mask] = 1.0 / volatilities[mask]
        
        total = np.sum(inv_vol)
        if total > 0:
            return inv_vol / total
        return np.ones(self._strategy_count) / self._strategy_count

    def _sharpe_weights(self) -> np.ndarray:
        """Weight by Sharpe ratio."""
        sharpe_ratios = self._sharpe_ratios[:self._strategy_count].copy()
        
        # Use positive Sharpe only
        sharpe_ratios = np.maximum(sharpe_ratios, 0)
        
        total = np.sum(sharpe_ratios)
        if total > 0:
            return sharpe_ratios / total
        return np.ones(self._strategy_count) / self._strategy_count

    def _sortino_weights(self) -> np.ndarray:
        """Weight by Sortino ratio."""
        sortino_ratios = self._sortino_ratios[:self._strategy_count].copy()
        
        # Handle infinity
        sortino_ratios = np.clip(sortino_ratios, 0, 100)
        
        total = np.sum(sortino_ratios)
        if total > 0:
            return sortino_ratios / total
        return np.ones(self._strategy_count) / self._strategy_count

    def get_strategy_metrics(self, strategy_id: str) -> Optional[StrategyMetrics]:
        """Get current metrics for a strategy."""
        idx = self._find_strategy(strategy_id)
        if idx < 0:
            return None
        
        return StrategyMetrics(
            strategy_id=strategy_id,
            allocated_capital=self._allocated_capital[idx],
            current_exposure=self._current_exposure[idx],
            unrealized_pnl=0.0,  # Would be fetched from position tracker
            realized_pnl=0.0,
            rolling_sharpe=self._sharpe_ratios[idx],
            rolling_sortino=self._sortino_ratios[idx],
            max_drawdown=self._max_drawdowns[idx],
            win_rate=0.0,  # Would be calculated from trade history
            volatility=self._volatilities[idx]
        )

    def get_all_allocations(self) -> Dict[str, float]:
        """Get current capital allocations."""
        allocations = {}
        for i in range(self._strategy_count):
            allocations[self._strategy_ids[i]] = self._allocated_capital[i]
        return allocations

    def set_allocation_model(self, model: AllocationModel) -> None:
        """Change the allocation model."""
        self._allocation_model = model
        logger.info(f"Allocation model changed to {model.name}")
