"""
regime_selector.py
------------------
Integrates with Hidden Markov Models (HMM) to activate/deactivate strategies
based on detected market regime.

Regimes: BULL_LOW_VOL, BULL_HIGH_VOL, BEAR_LOW_VOL, BEAR_HIGH_VOL, SIDEWAYS

Enables Trend Following in high volatility regimes, Mean Reversion in low volatility.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Detected market regimes."""
    BULL_LOW_VOL = auto()
    BULL_HIGH_VOL = auto()
    BEAR_LOW_VOL = auto()
    BEAR_HIGH_VOL = auto()
    SIDEWAYS_LOW_VOL = auto()
    SIDEWAYS_HIGH_VOL = auto()
    UNKNOWN = auto()


class StrategyType(Enum):
    """Types of trading strategies."""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"


# Optimal strategy types for each regime
REGIME_STRATEGY_MAP: Dict[MarketRegime, List[StrategyType]] = {
    MarketRegime.BULL_LOW_VOL: [StrategyType.TREND_FOLLOWING, StrategyType.MOMENTUM],
    MarketRegime.BULL_HIGH_VOL: [StrategyType.TREND_FOLLOWING, StrategyType.MOMENTUM],
    MarketRegime.BEAR_LOW_VOL: [StrategyType.MEAN_REVERSION],
    MarketRegime.BEAR_HIGH_VOL: [StrategyType.MEAN_REVERSION, StrategyType.ARBITRAGE],
    MarketRegime.SIDEWAYS_LOW_VOL: [StrategyType.MEAN_REVERSION, StrategyType.MARKET_MAKING],
    MarketRegime.SIDEWAYS_HIGH_VOL: [StrategyType.ARBITRAGE, StrategyType.MARKET_MAKING],
    MarketRegime.UNKNOWN: [],
}


@dataclass
class RegimeMetrics:
    """Current regime detection metrics."""
    regime: MarketRegime
    confidence: float
    trend_strength: float
    volatility_level: float
    momentum_score: float
    timestamp: datetime


class RegimeSelector:
    """
    Market regime detection and strategy activation controller.
    
    Features:
    - HMM-based regime detection (integrated from Stage 3)
    - Multi-factor regime classification (trend, vol, momentum)
    - Automatic strategy enable/disable based on regime
    - Confidence-weighted position sizing
    """

    # Thresholds for regime classification
    VOLATILITY_HIGH_THRESHOLD = 0.02  # 2% daily vol
    TREND_THRESHOLD = 0.5  # Correlation threshold for trend
    MOMENTUM_THRESHOLD = 0.3  # Momentum score threshold
    
    # Smoothing window for signals
    SMOOTHING_WINDOW = 5

    def __init__(self, hmm_model=None):
        self._hmm_model = hmm_model
        
        # Pre-allocated buffers for calculations
        self._returns_buffer = np.zeros(100, dtype='f8')
        self._volatility_buffer = np.zeros(50, dtype='f8')
        self._trend_buffer = np.zeros(50, dtype='f8')
        self._momentum_buffer = np.zeros(50, dtype='f8')
        
        self._write_idx = 0
        self._samples_collected = 0
        
        # Current regime state
        self._current_regime = MarketRegime.UNKNOWN
        self._regime_confidence = 0.0
        self._last_regime_change = datetime.utcnow()
        
        # Active strategies per regime
        self._active_strategies: Dict[str, bool] = {}  # strategy_id -> is_active
        self._strategy_types: Dict[str, StrategyType] = {}  # strategy_id -> type
        
        # Regime change callbacks
        self._regime_callbacks = []
        
        logger.info("RegimeSelector initialized")

    def register_strategy(self, strategy_id: str, strategy_type: StrategyType) -> None:
        """Register a strategy with its type."""
        self._strategy_types[strategy_id] = strategy_type
        self._active_strategies[strategy_id] = True  # Default active
        logger.info(f"Registered strategy: {strategy_id} ({strategy_type.value})")

    def add_market_data(
        self,
        return_value: float,
        volatility: float,
        trend_signal: float,
        momentum: float
    ) -> None:
        """
        Add new market data point for regime analysis.
        Called on each new bar/tick.
        """
        idx = self._write_idx
        
        self._returns_buffer[idx] = return_value
        self._volatility_buffer[idx] = volatility
        self._trend_buffer[idx] = trend_signal
        self._momentum_buffer[idx] = momentum
        
        self._write_idx = (idx + 1) % 100
        if self._samples_collected < 100:
            self._samples_collected += 1
        
        # Update regime periodically
        if self._samples_collected >= self.SMOOTHING_WINDOW:
            self._update_regime()

    def _update_regime(self) -> None:
        """Update regime classification based on current data."""
        # Get valid samples (non-zero)
        valid_vol = self._volatility_buffer[self._volatility_buffer != 0]
        valid_trend = self._trend_buffer[self._trend_buffer != 0]
        valid_momentum = self._momentum_buffer[self._momentum_buffer != 0]
        
        if len(valid_vol) < 3 or len(valid_trend) < 3:
            return  # Not enough data
        
        # Calculate smoothed metrics
        avg_volatility = np.mean(valid_vol[-self.SMOOTHING_WINDOW:])
        avg_trend = np.mean(valid_trend[-self.SMOOTHING_WINDOW:])
        avg_momentum = np.mean(valid_momentum[-self.SMOOTHING_WINDOW:])
        
        # Classify regime
        new_regime = self._classify_regime(avg_volatility, avg_trend, avg_momentum)
        
        # Calculate confidence based on signal strength
        confidence = self._calculate_confidence(avg_volatility, avg_trend, avg_momentum)
        
        # Check for regime change
        if new_regime != self._current_regime:
            old_regime = self._current_regime
            self._current_regime = new_regime
            self._regime_confidence = confidence
            self._last_regime_change = datetime.utcnow()
            
            logger.info(
                f"Regime change: {old_regime.name} -> {new_regime.name} "
                f"(confidence: {confidence:.2%})"
            )
            
            # Update active strategies
            self._update_strategy_activation()
            
            # Notify callbacks
            self._notify_regime_change(new_regime)

    def _classify_regime(
        self,
        volatility: float,
        trend: float,
        momentum: float
    ) -> MarketRegime:
        """Classify market regime based on metrics."""
        is_high_vol = volatility > self.VOLATILITY_HIGH_THRESHOLD
        is_trending = abs(trend) > self.TREND_THRESHOLD
        is_bullish = momentum > self.MOMENTUM_THRESHOLD
        is_bearish = momentum < -self.MOMENTUM_THRESHOLD
        
        if is_trending:
            if is_bullish:
                return MarketRegime.BULL_HIGH_VOL if is_high_vol else MarketRegime.BULL_LOW_VOL
            elif is_bearish:
                return MarketRegime.BEAR_HIGH_VOL if is_high_vol else MarketRegime.BEAR_LOW_VOL
            else:
                return MarketRegime.SIDEWAYS_HIGH_VOL if is_high_vol else MarketRegime.SIDEWAYS_LOW_VOL
        else:
            return MarketRegime.SIDEWAYS_HIGH_VOL if is_high_vol else MarketRegime.SIDEWAYS_LOW_VOL

    def _calculate_confidence(
        self,
        volatility: float,
        trend: float,
        momentum: float
    ) -> float:
        """Calculate confidence score for regime classification."""
        # Confidence based on signal clarity
        vol_clarity = min(volatility / self.VOLATILITY_HIGH_THRESHOLD, 1.0)
        trend_clarity = min(abs(trend) / self.TREND_THRESHOLD, 1.0)
        momentum_clarity = min(abs(momentum) / self.MOMENTUM_THRESHOLD, 1.0)
        
        # Weighted average
        confidence = 0.3 * vol_clarity + 0.4 * trend_clarity + 0.3 * momentum_clarity
        return min(confidence, 1.0)

    def _update_strategy_activation(self) -> None:
        """Enable/disable strategies based on current regime."""
        optimal_types = REGIME_STRATEGY_MAP.get(self._current_regime, [])
        
        for strategy_id, strategy_type in self._strategy_types.items():
            should_be_active = strategy_type in optimal_types
            currently_active = self._active_strategies.get(strategy_id, False)
            
            if should_be_active != currently_active:
                self._active_strategies[strategy_id] = should_be_active
                status = "ENABLED" if should_be_active else "DISABLED"
                logger.info(
                    f"Strategy {strategy_id} {status} "
                    f"(regime: {self._current_regime.name}, type: {strategy_type.value})"
                )

    def is_strategy_active(self, strategy_id: str) -> bool:
        """Check if a strategy should be active."""
        return self._active_strategies.get(strategy_id, False)

    def get_position_scale(self, strategy_id: str) -> float:
        """
        Get position scaling factor based on regime confidence.
        Reduces position size when regime confidence is low.
        """
        if not self.is_strategy_active(strategy_id):
            return 0.0
        
        # Scale by confidence
        # Minimum 20% even at low confidence to avoid complete shutdown
        return max(0.2, self._regime_confidence)

    def _notify_regime_change(self, new_regime: MarketRegime) -> None:
        """Notify registered callbacks of regime change."""
        for callback in self._regime_callbacks:
            try:
                callback(new_regime)
            except Exception as e:
                logger.error(f"Regime callback error: {e}")

    def register_callback(self, callback) -> None:
        """Register callback for regime changes."""
        self._regime_callbacks.append(callback)

    def get_current_regime(self) -> RegimeMetrics:
        """Get current regime information."""
        # Calculate current metrics
        valid_vol = self._volatility_buffer[self._volatility_buffer != 0]
        valid_trend = self._trend_buffer[self._trend_buffer != 0]
        valid_momentum = self._momentum_buffer[self._momentum_buffer != 0]
        
        return RegimeMetrics(
            regime=self._current_regime,
            confidence=self._regime_confidence,
            trend_strength=float(np.mean(valid_trend[-5:])) if len(valid_trend) > 0 else 0.0,
            volatility_level=float(np.mean(valid_vol[-5:])) if len(valid_vol) > 0 else 0.0,
            momentum_score=float(np.mean(valid_momentum[-5:])) if len(valid_momentum) > 0 else 0.0,
            timestamp=datetime.utcnow()
        )

    def get_optimal_strategies(self) -> List[StrategyType]:
        """Get list of optimal strategy types for current regime."""
        return REGIME_STRATEGY_MAP.get(self._current_regime, [])

    def get_all_strategy_states(self) -> Dict[str, dict]:
        """Get state of all registered strategies."""
        states = {}
        for strategy_id, strategy_type in self._strategy_types.items():
            states[strategy_id] = {
                'type': strategy_type.value,
                'is_active': self.is_strategy_active(strategy_id),
                'position_scale': self.get_position_scale(strategy_id),
                'optimal_for_regime': strategy_type in self.get_optimal_strategies()
            }
        return states

    def force_regime(self, regime: MarketRegime) -> None:
        """Force a specific regime (for testing/override)."""
        old_regime = self._current_regime
        self._current_regime = regime
        self._regime_confidence = 1.0
        self._last_regime_change = datetime.utcnow()
        
        self._update_strategy_activation()
        
        logger.warning(f"Regime manually forced: {old_regime.name} -> {regime.name}")
