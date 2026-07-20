"""
python/nautilus/strategy_lifecycle.py

Automated strategy lifecycle manager. Promotes shadow strategies to live if they
pass out-of-sample walk-forward metrics, and automatically demotes live strategies
to shadow or halts them if their live Sharpe ratio degrades.

Features:
- Walk-forward validation pipeline
- Automatic promotion/demotion logic
- Statistical significance testing
- Circuit breaker integration
"""

import asyncio
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from scipy import stats


class StrategyStatus(Enum):
    """Strategy lifecycle states."""
    DEVELOPMENT = "development"
    SHADOW = "shadow"
    LIVE_SMALL = "live_small"  # Limited capital
    LIVE_FULL = "live_full"
    HALTED = "halted"
    RETIRED = "retired"


@dataclass
class StrategyConfig:
    """Configuration for a trading strategy."""
    id: str
    name: str
    status: StrategyStatus = StrategyStatus.DEVELOPMENT
    capital_allocation: float = 0.0
    max_capital: float = 100_000.0
    min_sharpe: float = 1.5  # Minimum Sharpe for promotion
    max_drawdown: float = 0.15  # Maximum allowed drawdown
    created_at: datetime = field(default_factory=datetime.now)
    promoted_at: Optional[datetime] = None
    halted_at: Optional[datetime] = None


@dataclass
class WalkForwardResult:
    """Results from walk-forward analysis."""
    mean_sharpe: float
    std_sharpe: float
    t_statistic: float
    p_value: float
    is_significant: bool
    oos_periods: int
    avg_oos_sharpe: float


class StrategyLifecycleManager:
    """
    Manages the full lifecycle of trading strategies.
    
    Handles:
    - Promotion from shadow to live
    - Demotion from live to shadow
    - Emergency halting
    - Retirement
    """
    
    def __init__(self):
        self.strategies: Dict[str, StrategyConfig] = {}
        self.metrics_history: Dict[str, List[Dict]] = {}
        
        # Callbacks for state changes
        self.on_promote_callbacks: List[Callable] = []
        self.on_demote_callbacks: List[Callable] = []
        self.on_halt_callbacks: List[Callable] = []
        
        # Running state
        self.running = False
        
    def register_strategy(self, config: StrategyConfig):
        """Register a new strategy."""
        self.strategies[config.id] = config
        self.metrics_history[config.id] = []
    
    def record_metrics(self, strategy_id: str, metrics: Dict):
        """Record performance metrics for a strategy."""
        if strategy_id not in self.metrics_history:
            self.metrics_history[strategy_id] = []
        
        metrics['timestamp'] = datetime.now()
        self.metrics_history[strategy_id].append(metrics)
        
        # Keep last 10000 records
        if len(self.metrics_history[strategy_id]) > 10000:
            self.metrics_history[strategy_id] = self.metrics_history[strategy_id][-10000:]
    
    async def start(self):
        """Start lifecycle monitoring loop."""
        self.running = True
        await self._monitor_loop()
    
    def stop(self):
        """Stop monitoring."""
        self.running = False
    
    async def _monitor_loop(self):
        """Main monitoring loop - checks strategies every minute."""
        while self.running:
            for strategy_id, config in self.strategies.items():
                await self._evaluate_strategy(strategy_id, config)
            await asyncio.sleep(60)  # Check every minute
    
    async def _evaluate_strategy(self, strategy_id: str, config: StrategyConfig):
        """Evaluate a single strategy for state transitions."""
        if config.status == StrategyStatus.HALTED:
            return
        
        metrics = self.metrics_history.get(strategy_id, [])
        if len(metrics) < 100:  # Need minimum data
            return
        
        # Get recent metrics (last 100)
        recent = metrics[-100:]
        
        sharpe_ratios = [m.get('sharpe', 0) for m in recent if 'sharpe' in m]
        drawdowns = [m.get('drawdown', 0) for m in recent if 'drawdown' in m]
        
        if not sharpe_ratios:
            return
        
        avg_sharpe = np.mean(sharpe_ratios)
        max_dd = max(drawdowns) if drawdowns else 0
        
        # Check for demotion/halt conditions
        if config.status in [StrategyStatus.LIVE_SMALL, StrategyStatus.LIVE_FULL]:
            if max_dd > config.max_drawdown:
                await self._halt_strategy(strategy_id, "Max drawdown exceeded")
            elif avg_sharpe < config.min_sharpe * 0.5:
                await self._demote_strategy(strategy_id, "Sharpe degradation")
        
        # Check for promotion conditions
        if config.status == StrategyStatus.SHADOW:
            wf_result = self.run_walk_forward(strategy_id)
            if wf_result and wf_result.is_significant:
                if wf_result.mean_sharpe >= config.min_sharpe:
                    await self._promote_strategy(strategy_id, StrategyStatus.LIVE_SMALL)
    
    def run_walk_forward(self, strategy_id: str, n_splits: int = 5) -> Optional[WalkForwardResult]:
        """
        Run walk-forward analysis on strategy metrics.
        
        Splits data into training/test periods and evaluates
        out-of-sample performance statistical significance.
        """
        metrics = self.metrics_history.get(strategy_id, [])
        if len(metrics) < n_splits * 200:
            return None
        
        # Extract Sharpe ratios
        sharpe_series = [m.get('sharpe', 0) for m in metrics if 'sharpe' in m]
        if len(sharpe_series) < n_splits * 200:
            return None
        
        sharpe_array = np.array(sharpe_series)
        chunk_size = len(sharpe_array) // n_splits
        
        oos_sharpes = []
        
        for i in range(n_splits):
            # In-sample: first 70% of chunk
            # Out-of-sample: last 30% of chunk
            start = i * chunk_size
            end = (i + 1) * chunk_size
            chunk = sharpe_array[start:end]
            
            split_point = int(len(chunk) * 0.7)
            oos = chunk[split_point:]
            
            if len(oos) > 0:
                oos_sharpes.append(np.mean(oos))
        
        if not oos_sharpes:
            return None
        
        oos_array = np.array(oos_sharpes)
        mean_sharpe = np.mean(oos_array)
        std_sharpe = np.std(oos_array)
        
        # T-test against null hypothesis (Sharpe = 0)
        t_stat, p_value = stats.ttest_1samp(oos_array, 0)
        
        result = WalkForwardResult(
            mean_sharpe=mean_sharpe,
            std_sharpe=std_sharpe,
            t_statistic=t_stat,
            p_value=p_value,
            is_significant=p_value < 0.05 and mean_sharpe > 0,
            oos_periods=len(oos_sharpes),
            avg_oos_sharpe=mean_sharpe,
        )
        
        return result
    
    async def _promote_strategy(self, strategy_id: str, new_status: StrategyStatus):
        """Promote a strategy to higher status."""
        config = self.strategies[strategy_id]
        old_status = config.status
        
        config.status = new_status
        config.promoted_at = datetime.now()
        
        # Increase capital allocation
        if new_status == StrategyStatus.LIVE_SMALL:
            config.capital_allocation = config.max_capital * 0.1
        elif new_status == StrategyStatus.LIVE_FULL:
            config.capital_allocation = config.max_capital
        
        # Notify callbacks
        for callback in self.on_promote_callbacks:
            try:
                callback(strategy_id, old_status, new_status)
            except Exception:
                pass
    
    async def _demote_strategy(self, strategy_id: str, reason: str):
        """Demote a strategy to lower status."""
        config = self.strategies[strategy_id]
        old_status = config.status
        
        if config.status == StrategyStatus.LIVE_FULL:
            config.status = StrategyStatus.LIVE_SMALL
            config.capital_allocation = config.max_capital * 0.1
        elif config.status == StrategyStatus.LIVE_SMALL:
            config.status = StrategyStatus.SHADOW
            config.capital_allocation = 0
        
        # Notify callbacks
        for callback in self.on_demote_callbacks:
            try:
                callback(strategy_id, old_status, config.status, reason)
            except Exception:
                pass
    
    async def _halt_strategy(self, strategy_id: str, reason: str):
        """Emergency halt a strategy."""
        config = self.strategies[strategy_id]
        old_status = config.status
        
        config.status = StrategyStatus.HALTED
        config.halted_at = datetime.now()
        config.capital_allocation = 0
        
        # Notify callbacks
        for callback in self.on_halt_callbacks:
            try:
                callback(strategy_id, old_status, reason)
            except Exception:
                pass
    
    def get_strategy_status(self, strategy_id: str) -> Optional[StrategyStatus]:
        """Get current status of a strategy."""
        config = self.strategies.get(strategy_id)
        return config.status if config else None
    
    def get_all_strategies_summary(self) -> List[Dict]:
        """Get summary of all strategies."""
        summaries = []
        for sid, config in self.strategies.items():
            metrics = self.metrics_history.get(sid, [])
            recent_sharpe = np.mean([m.get('sharpe', 0) for m in metrics[-100:]]) if metrics else 0
            
            summaries.append({
                'id': sid,
                'name': config.name,
                'status': config.status.value,
                'capital': config.capital_allocation,
                'recent_sharpe': recent_sharpe,
                'created': config.created_at.isoformat(),
            })
        
        return summaries


if __name__ == '__main__':
    print("Strategy Lifecycle Manager - Import StrategyLifecycleManager class")
