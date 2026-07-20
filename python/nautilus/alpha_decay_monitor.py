"""
python/nautilus/alpha_decay_monitor.py

Continuously monitors the rolling Sharpe, Sortino, and Calmar ratios of every
deployed strategy. Triggers automatic retraining via the Ray cluster when alpha
decay crosses a predefined statistical threshold.

Features:
- Rolling performance metrics calculation
- Statistical significance testing for decay detection
- Automatic Ray cluster retraining triggers
- Multi-timeframe analysis (1h, 4h, 24h)
"""

import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import numpy as np
from scipy import stats


@dataclass
class DecayAlert:
    """Alert triggered when alpha decay is detected."""
    strategy_id: str
    metric_name: str  # sharpe, sortino, calmar
    current_value: float
    historical_avg: float
    decay_percent: float
    p_value: float
    timestamp: datetime = field(default_factory=datetime.now)
    action_taken: str = ""  # "retrain", "demote", "halt"


@dataclass
class StrategyMetrics:
    """Rolling metrics for a strategy."""
    sharpe_1h: float = 0.0
    sharpe_4h: float = 0.0
    sharpe_24h: float = 0.0
    sortino_1h: float = 0.0
    sortino_4h: float = 0.0
    sortino_24h: float = 0.0
    calmar_1h: float = 0.0
    calmar_4h: float = 0.0
    calmar_24h: float = 0.0


class AlphaDecayMonitor:
    """
    Monitors strategies for alpha decay and triggers retraining.
    
    Uses statistical tests to determine if performance degradation
    is significant vs random noise.
    """
    
    def __init__(
        self,
        decay_threshold: float = 0.5,  # 50% decay triggers alert
        min_samples: int = 100,
        confidence_level: float = 0.95,
    ):
        self.decay_threshold = decay_threshold
        self.min_samples = min_samples
        self.confidence_level = confidence_level
        
        # PnL history per strategy
        self.pnl_history: Dict[str, deque] = {}
        
        # Historical metrics baselines
        self.baselines: Dict[str, Dict[str, List[float]]] = {}
        
        # Active alerts
        self.alerts: List[DecayAlert] = []
        
        # Retraining callbacks
        self.on_retrain_request: Optional[callable] = None
        
        # Running state
        self.running = False
        
    def record_pnl(self, strategy_id: str, pnl: float, timestamp: datetime = None):
        """Record a PnL observation for a strategy."""
        if strategy_id not in self.pnl_history:
            self.pnl_history[strategy_id] = deque(maxlen=100000)
            self.baselines[strategy_id] = {
                'sharpe': [],
                'sortino': [],
                'calmar': [],
            }
        
        ts = timestamp or datetime.now()
        self.pnl_history[strategy_id].append((ts, pnl))
    
    async def start(self):
        """Start monitoring loop."""
        self.running = True
        await self._monitor_loop()
    
    def stop(self):
        """Stop monitoring."""
        self.running = False
    
    async def _monitor_loop(self):
        """Main monitoring loop - runs every 5 minutes."""
        while self.running:
            for strategy_id in self.pnl_history.keys():
                await self._check_strategy_decay(strategy_id)
            await asyncio.sleep(300)  # 5 minutes
    
    async def _check_strategy_decay(self, strategy_id: str):
        """Check a single strategy for alpha decay."""
        pnl_data = self.pnl_history.get(strategy_id, [])
        if len(pnl_data) < self.min_samples:
            return
        
        # Convert to array
        pnls = np.array([p[1] for p in pnl_data])
        
        # Calculate metrics for different timeframes
        metrics = self._calculate_metrics(pnls)
        
        # Update baselines (exponential moving average)
        self._update_baselines(strategy_id, metrics)
        
        # Check for decay
        alerts = self._detect_decay(strategy_id, metrics)
        
        for alert in alerts:
            self.alerts.append(alert)
            
            # Keep only last 100 alerts
            if len(self.alerts) > 100:
                self.alerts = self.alerts[-100:]
            
            # Trigger retraining if decay is significant
            if alert.decay_percent > self.decay_threshold:
                await self._trigger_retrain(strategy_id, alert)
    
    def _calculate_metrics(self, pnls: np.ndarray) -> StrategyMetrics:
        """Calculate rolling metrics from PnL series."""
        # Split into timeframes (assuming 1-minute data)
        n = len(pnls)
        
        # 1 hour = last 60 samples
        pnl_1h = pnls[-60:] if n >= 60 else pnls
        
        # 4 hours = last 240 samples
        pnl_4h = pnls[-240:] if n >= 240 else pnls
        
        # 24 hours = last 1440 samples
        pnl_24h = pnls[-1440:] if n >= 1440 else pnls
        
        def calc_sharpe(returns):
            if len(returns) < 2 or np.std(returns) == 0:
                return 0.0
            return np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        def calc_sortino(returns):
            if len(returns) < 2:
                return 0.0
            downside = returns[returns < 0]
            if len(downside) < 2:
                return 0.0
            return np.mean(returns) / np.std(downside) * np.sqrt(252)
        
        def calc_calmar(returns):
            if len(returns) < 2:
                return 0.0
            cumulative = np.cumsum(returns)
            peak = np.maximum.accumulate(cumulative)
            drawdown = (peak - cumulative) / (np.abs(peak) + 1e-10)
            max_dd = np.max(drawdown)
            if max_dd == 0:
                return 0.0
            return np.sum(returns) / max_dd
        
        return StrategyMetrics(
            sharpe_1h=calc_sharpe(pnl_1h),
            sharpe_4h=calc_sharpe(pnl_4h),
            sharpe_24h=calc_sharpe(pnl_24h),
            sortino_1h=calc_sortino(pnl_1h),
            sortino_4h=calc_sortino(pnl_4h),
            sortino_24h=calc_sortino(pnl_24h),
            calmar_1h=calc_calmar(pnl_1h),
            calmar_4h=calc_calmar(pnl_4h),
            calmar_24h=calc_calmar(pnl_24h),
        )
    
    def _update_baselines(self, strategy_id: str, metrics: StrategyMetrics):
        """Update baseline metrics with exponential moving average."""
        baseline = self.baselines[strategy_id]
        
        # EMA factor
        alpha = 0.1
        
        for metric_name in ['sharpe', 'sortino', 'calmar']:
            value_24h = getattr(metrics, f'{metric_name}_24h')
            
            if value_24h != 0:
                if not baseline[metric_name]:
                    baseline[metric_name] = [value_24h]
                else:
                    ema = alpha * value_24h + (1 - alpha) * baseline[metric_name][-1]
                    baseline[metric_name].append(ema)
                    
                    # Keep last 1000 baseline values
                    if len(baseline[metric_name]) > 1000:
                        baseline[metric_name] = baseline[metric_name][-1000:]
    
    def _detect_decay(self, strategy_id: str, metrics: StrategyMetrics) -> List[DecayAlert]:
        """Detect significant alpha decay."""
        alerts = []
        baseline = self.baselines.get(strategy_id, {})
        
        for metric_name in ['sharpe', 'sortino', 'calmar']:
            current = getattr(metrics, f'{metric_name}_24h')
            historical = baseline.get(metric_name, [])
            
            if len(historical) < self.min_samples:
                continue
            
            historical_avg = np.mean(historical[-100:])
            
            if abs(historical_avg) < 0.1:  # Ignore near-zero baselines
                continue
            
            # Calculate decay percentage
            decay = (historical_avg - current) / abs(historical_avg)
            
            if decay < self.decay_threshold:
                continue
            
            # Statistical test: is current significantly lower?
            t_stat, p_value = stats.ttest_1samp(historical[-100:], current)
            
            if p_value < (1 - self.confidence_level):
                alert = DecayAlert(
                    strategy_id=strategy_id,
                    metric_name=metric_name,
                    current_value=current,
                    historical_avg=historical_avg,
                    decay_percent=decay,
                    p_value=p_value,
                )
                alerts.append(alert)
        
        return alerts
    
    async def _trigger_retrain(self, strategy_id: str, alert: DecayAlert):
        """Trigger retraining for a decaying strategy."""
        alert.action_taken = "retrain_requested"
        
        if self.on_retrain_request:
            try:
                await self.on_retrain_request(strategy_id, alert)
            except Exception as e:
                print(f"Retrain request failed: {e}")
    
    def get_current_metrics(self, strategy_id: str) -> Optional[StrategyMetrics]:
        """Get current metrics for a strategy."""
        pnl_data = self.pnl_history.get(strategy_id, [])
        if len(pnl_data) < 10:
            return None
        
        pnls = np.array([p[1] for p in pnl_data])
        return self._calculate_metrics(pnls)
    
    def get_alerts(self, strategy_id: Optional[str] = None) -> List[DecayAlert]:
        """Get recent alerts, optionally filtered by strategy."""
        if strategy_id:
            return [a for a in self.alerts if a.strategy_id == strategy_id]
        return self.alerts.copy()
    
    def get_decay_summary(self) -> Dict[str, float]:
        """Get summary of decay across all strategies."""
        summary = {}
        for strategy_id in self.pnl_history.keys():
            metrics = self.get_current_metrics(strategy_id)
            if metrics:
                baseline = self.baselines.get(strategy_id, {})
                hist_sharpe = np.mean(baseline.get('sharpe', [0])[-100:])
                
                if hist_sharpe != 0:
                    decay = (hist_sharpe - metrics.sharpe_24h) / abs(hist_sharpe)
                    summary[strategy_id] = max(0, decay)
        
        return summary


if __name__ == '__main__':
    print("Alpha Decay Monitor - Import AlphaDecayMonitor class")
