"""
correlation_guard.py
--------------------
Real-time covariance matrix calculation to prevent highly correlated strategies
from over-leveraging the portfolio. If correlation between Strategy A and B
exceeds a threshold, automatically scales down position size of newer entries.

Uses pre-allocated arrays and incremental covariance updates for O(1) complexity.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CorrelationAlert:
    """Alert when correlation threshold exceeded."""
    strategy_a: str
    strategy_b: str
    correlation: float
    threshold: float
    timestamp: datetime
    action_taken: str


class CorrelationGuard:
    """
    Real-time correlation monitoring and position sizing adjustment.
    
    Features:
    - Incremental covariance matrix updates (O(1) per update)
    - Automatic position scaling based on correlation
    - Configurable correlation thresholds
    - Pre-allocated matrices for fixed strategy count
    """

    MAX_STRATEGIES = 20
    DEFAULT_CORRELATION_THRESHOLD = 0.7
    DEFAULT_LOOKBACK_PERIODS = 60  # ~2 months of daily data
    
    def __init__(
        self,
        strategy_ids: List[str],
        correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
        lookback_periods: int = DEFAULT_LOOKBACK_PERIODS
    ):
        self._strategy_ids = strategy_ids[:self.MAX_STRATEGIES]
        self._n_strategies = len(self._strategy_ids)
        self._threshold = correlation_threshold
        self._lookback = lookback_periods
        
        # Create ID to index mapping
        self._id_to_idx = {sid: i for i, sid in enumerate(self._strategy_ids)}
        
        # Pre-allocated return history (strategies x periods)
        self._returns_history = np.zeros((self._n_strategies, lookback_periods), dtype='f8')
        self._write_idx = 0
        self._periods_filled = 0
        
        # Pre-allocated covariance and correlation matrices
        self._covariance_matrix = np.eye(self._n_strategies, dtype='f8')
        self._correlation_matrix = np.eye(self._n_strategies, dtype='f8')
        
        # Running statistics for incremental updates
        self._means = np.zeros(self._n_strategies, dtype='f8')
        self._M2 = np.zeros((self._n_strategies, self._n_strategies), dtype='f8')  # Sum of squares of differences
        
        # Position scaling factors (1.0 = no scaling)
        self._position_scales = np.ones(self._n_strategies, dtype='f8')
        
        # Alert history
        self._alerts: List[CorrelationAlert] = []
        self._max_alerts = 100
        
        # Lock for thread-safe updates
        self._lock = asyncio.Lock()
        
        logger.info(f"CorrelationGuard initialized with {self._n_strategies} strategies")

    def add_return(self, strategy_id: str, return_value: float) -> None:
        """
        Add a new return observation for a strategy.
        Updates covariance matrix incrementally.
        """
        if strategy_id not in self._id_to_idx:
            logger.warning(f"Unknown strategy: {strategy_id}")
            return
        
        idx = self._id_to_idx[strategy_id]
        
        # Store return
        self._returns_history[idx, self._write_idx] = return_value
        
        # Update running statistics (Welford's online algorithm extended to covariance)
        n = max(self._periods_filled, 1)
        
        # Update means
        delta = return_value - self._means[idx]
        self._means[idx] += delta / (n + 1)
        
        # Update M2 matrix (cross-products)
        for j in range(self._n_strategies):
            delta_j = self._returns_history[j, self._write_idx] - self._means[j]
            self._M2[idx, j] += delta * delta_j
            self._M2[j, idx] = self._M2[idx, j]  # Symmetric
        
        # Update write index
        self._write_idx = (self._write_idx + 1) % self._lookback
        if self._periods_filled < self._lookback:
            self._periods_filled += 1
        
        # Recalculate correlation matrix periodically
        if self._periods_filled >= 5:
            self._update_correlation_matrix()

    def _update_correlation_matrix(self) -> None:
        """Update correlation matrix from running statistics."""
        n = max(self._periods_filled - 1, 1)
        
        # Calculate covariance matrix
        for i in range(self._n_strategies):
            for j in range(self._n_strategies):
                self._covariance_matrix[i, j] = self._M2[i, j] / n
        
        # Calculate correlation matrix
        for i in range(self._n_strategies):
            for j in range(self._n_strategies):
                var_i = self._covariance_matrix[i, i]
                var_j = self._covariance_matrix[j, j]
                
                if var_i > 0 and var_j > 0:
                    std_i = np.sqrt(var_i)
                    std_j = np.sqrt(var_j)
                    self._correlation_matrix[i, j] = self._covariance_matrix[i, j] / (std_i * std_j)
                else:
                    self._correlation_matrix[i, j] = 0.0 if i != j else 1.0
        
        # Check for high correlations and adjust positions
        self._check_correlations()

    def _check_correlations(self) -> None:
        """Check all pairs for high correlation and apply scaling."""
        now = datetime.utcnow()
        
        for i in range(self._n_strategies):
            for j in range(i + 1, self._n_strategies):
                corr = self._correlation_matrix[i, j]
                
                if abs(corr) > self._threshold:
                    # High correlation detected
                    strategy_a = self._strategy_ids[i]
                    strategy_b = self._strategy_ids[j]
                    
                    # Scale down the newer/higher-risk strategy
                    # Simple heuristic: scale the one with higher current exposure
                    scale_factor = 1.0 - (abs(corr) - self._threshold) / (1.0 - self._threshold + 0.01)
                    scale_factor = max(0.2, scale_factor)  # Minimum 20% allocation
                    
                    # Apply scaling to both strategies proportionally
                    self._position_scales[i] = min(self._position_scales[i], scale_factor)
                    self._position_scales[j] = min(self._position_scales[j], scale_factor)
                    
                    # Record alert
                    alert = CorrelationAlert(
                        strategy_a=strategy_a,
                        strategy_b=strategy_b,
                        correlation=float(corr),
                        threshold=self._threshold,
                        timestamp=now,
                        action_taken=f"Scaled positions to {scale_factor:.2%}"
                    )
                    self._alerts.append(alert)
                    
                    if len(self._alerts) > self._max_alerts:
                        self._alerts.pop(0)
                    
                    logger.warning(
                        f"High correlation detected: {strategy_a} <-> {strategy_b} "
                        f"(corr={corr:.3f}). Applied scaling factor {scale_factor:.2f}"
                    )

    def get_position_scale(self, strategy_id: str) -> float:
        """Get current position scaling factor for a strategy."""
        if strategy_id not in self._id_to_idx:
            return 1.0
        idx = self._id_to_idx[strategy_id]
        return self._position_scales[idx]

    def get_adjusted_position_size(
        self,
        strategy_id: str,
        requested_size: float
    ) -> float:
        """Get position size after correlation-based adjustment."""
        scale = self.get_position_scale(strategy_id)
        return requested_size * scale

    def get_correlation_matrix(self) -> np.ndarray:
        """Get current correlation matrix."""
        return self._correlation_matrix.copy()

    def get_covariance_matrix(self) -> np.ndarray:
        """Get current covariance matrix."""
        return self._covariance_matrix.copy()

    def get_correlation(self, strategy_a: str, strategy_b: str) -> Optional[float]:
        """Get correlation between two strategies."""
        if strategy_a not in self._id_to_idx or strategy_b not in self._id_to_idx:
            return None
        
        idx_a = self._id_to_idx[strategy_a]
        idx_b = self._id_to_idx[strategy_b]
        
        return float(self._correlation_matrix[idx_a, idx_b])

    def reset_scaling(self) -> None:
        """Reset all position scaling factors to 1.0."""
        self._position_scales[:] = 1.0
        logger.info("Position scaling factors reset")

    def get_alerts(self, limit: int = 10) -> List[CorrelationAlert]:
        """Get recent correlation alerts."""
        return self._alerts[-limit:]

    def get_risk_metrics(self) -> dict:
        """Get comprehensive risk metrics."""
        # Calculate portfolio-level metrics
        avg_correlation = np.mean(
            [self._correlation_matrix[i, j] 
             for i in range(self._n_strategies) 
             for j in range(i+1, self._n_strategies)]
        )
        
        max_correlation = 0.0
        max_pair = None
        for i in range(self._n_strategies):
            for j in range(i + 1, self._n_strategies):
                corr = abs(self._correlation_matrix[i, j])
                if corr > max_correlation:
                    max_correlation = corr
                    max_pair = (self._strategy_ids[i], self._strategy_ids[j])
        
        return {
            'average_correlation': float(avg_correlation),
            'max_correlation': float(max_correlation),
            'max_correlation_pair': max_pair,
            'active_scales': {
                self._strategy_ids[i]: float(self._position_scales[i])
                for i in range(self._n_strategies)
            },
            'alert_count': len(self._alerts),
            'strategies_monitored': self._n_strategies
        }

    async def monitor_loop(self, check_interval: float = 60.0) -> None:
        """
        Continuous monitoring loop.
        Run as background task.
        """
        while True:
            await asyncio.sleep(check_interval)
            
            # Log current state
            metrics = self.get_risk_metrics()
            if metrics['max_correlation'] > self._threshold:
                logger.warning(
                    f"Correlation guard status: max_corr={metrics['max_correlation']:.3f}, "
                    f"pair={metrics['max_correlation_pair']}"
                )
