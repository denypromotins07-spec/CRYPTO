# python/quant/cointegration.py
# =============================================================================
# COINTEGRATION & PAIRS TRADING ENGINE
# =============================================================================
# Purpose: Real-time cointegration tracking between crypto pairs using Kalman
# filters and rolling statistics for statistical arbitrage opportunities.
#
# Key Features:
# - Kalman Filter for dynamic hedge ratio estimation
# - Rolling ADF test approximation for cointegration status
# - Z-score calculation for mean-reversion signals
# - Multi-pair monitoring for basket trading

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class PairState:
    """State of a cointegrated pair"""
    symbol_a: str
    symbol_b: str
    hedge_ratio: float
    spread: float
    z_score: float
    is_cointegrated: bool
    half_life: float  # Mean reversion half-life in bars
    correlation: float


class KalmanFilter:
    """
    1D Kalman Filter for dynamic hedge ratio estimation.
    Provides online, recursive estimation without storing full history.
    """
    
    def __init__(self, initial_state: float = 0.0, process_var: float = 1e-5, 
                 measurement_var: float = 1e-2):
        # State estimate
        self.x = initial_state
        
        # Error covariance
        self.P = 1.0
        
        # Process noise (how much we expect hedge ratio to change)
        self.Q = process_var
        
        # Measurement noise (price noise)
        self.R = measurement_var
    
    def update(self, measurement: float, control: float = 1.0) -> float:
        """
        Update filter with new observation.
        
        Args:
            measurement: Observed value (e.g., price return of asset A)
            control: Control input (e.g., price return of asset B)
            
        Returns:
            Updated state estimate (hedge ratio)
        """
        # Prediction step
        x_pred = self.x  # State transition is identity
        P_pred = self.P + self.Q
        
        # Update step
        K = P_pred * control / (control * control * P_pred + self.R)  # Kalman gain
        self.x = x_pred + K * (measurement - control * x_pred)
        self.P = (1 - K * control) * P_pred
        
        return self.x
    
    def get_state(self) -> float:
        return self.x


class CointegrationTracker:
    """
    Tracks cointegration relationships between pairs in real-time.
    Uses Kalman filter for hedge ratio and rolling statistics for spread analysis.
    """
    
    def __init__(self, lookback: int = 100, half_life_window: int = 50):
        self.lookback = lookback
        self.half_life_window = half_life_window
        
        # Storage for each pair
        self.pairs: Dict[str, dict] = {}
        
        # Pre-allocated buffers for efficiency
        self.max_history = 500
    
    def add_pair(self, symbol_a: str, symbol_b: str):
        """Initialize tracking for a new pair"""
        key = f"{symbol_a}_{symbol_b}"
        self.pairs[key] = {
            'symbol_a': symbol_a,
            'symbol_b': symbol_b,
            'kalman': KalmanFilter(initial_state=1.0),
            'spread_history': deque(maxlen=self.max_history),
            'price_a_history': deque(maxlen=self.max_history),
            'price_b_history': deque(maxlen=self.max_history),
            'returns_a': deque(maxlen=self.lookback),
            'returns_b': deque(maxlen=self.lookback),
        }
    
    def update(self, symbol_a: str, symbol_b: str, price_a: float, 
               price_b: float, timestamp_ns: int) -> Optional[PairState]:
        """
        Update pair with new prices and calculate spread/z-score.
        
        Returns:
            PairState if enough data, None otherwise
        """
        key = f"{symbol_a}_{symbol_b}"
        if key not in self.pairs:
            self.add_pair(symbol_a, symbol_b)
        
        pair = self.pairs[key]
        
        # Calculate returns
        if len(pair['price_a_history']) > 0:
            prev_a = pair['price_a_history'][-1]
            prev_b = pair['price_b_history'][-1]
            ret_a = (price_a - prev_a) / prev_a
            ret_b = (price_b - prev_b) / prev_b
            
            pair['returns_a'].append(ret_a)
            pair['returns_b'].append(ret_b)
            
            # Update Kalman filter for hedge ratio
            # Using returns to estimate dynamic beta
            hedge_ratio = pair['kalman'].update(ret_a, ret_b)
        else:
            hedge_ratio = 1.0
        
        # Store prices
        pair['price_a_history'].append(price_a)
        pair['price_b_history'].append(price_b)
        
        # Calculate spread: price_a - hedge_ratio * price_b
        spread = price_a - hedge_ratio * price_b
        pair['spread_history'].append(spread)
        
        # Need enough data for statistics
        if len(pair['spread_history']) < self.lookback:
            return None
        
        # Calculate z-score of spread
        spreads = np.array(pair['spread_history'])
        spread_mean = np.mean(spreads)
        spread_std = np.std(spreads)
        
        if spread_std > 0:
            z_score = (spread - spread_mean) / spread_std
        else:
            z_score = 0.0
        
        # Estimate half-life of mean reversion
        half_life = self._estimate_half_life(spreads)
        
        # Check cointegration status (simplified ADF approximation)
        # In production, use proper rolling ADF test
        is_cointegrated = self._check_cointegration_status(
            pair['returns_a'], 
            pair['returns_b'],
            hedge_ratio
        )
        
        # Calculate correlation
        if len(pair['returns_a']) >= 20:
            correlation = np.corrcoef(
                list(pair['returns_a']), 
                list(pair['returns_b'])
            )[0, 1]
        else:
            correlation = 0.0
        
        return PairState(
            symbol_a=symbol_a,
            symbol_b=symbol_b,
            hedge_ratio=hedge_ratio,
            spread=spread,
            z_score=z_score,
            is_cointegrated=is_cointegrated,
            half_life=half_life,
            correlation=correlation
        )
    
    def _estimate_half_life(self, spreads: np.ndarray) -> float:
        """
        Estimate mean reversion half-life using autocorrelation.
        Half-life = -log(2) / lambda, where lambda is from Ornstein-Uhlenbeck process
        """
        if len(spreads) < self.half_life_window:
            return 0.0
        
        # Calculate spread changes
        spread_changes = np.diff(spreads)
        spread_lagged = spreads[:-1]
        
        # Simple regression: spread_change = lambda * spread_lagged + epsilon
        if np.var(spread_lagged) > 0:
            lambda_est = np.cov(spread_lagged, spread_changes)[0, 1] / np.var(spread_lagged)
            
            if lambda_est < 0:
                half_life = -np.log(2) / lambda_est
                return min(half_life, self.half_life_window * 2)  # Cap at reasonable value
        
        return self.half_life_window  # Default
    
    def _check_cointegration_status(self, returns_a: deque, returns_b: deque, 
                                     hedge_ratio: float) -> bool:
        """
        Simplified cointegration check using correlation and variance ratio.
        In production, implement rolling Engle-Granger or Johansen test.
        """
        if len(returns_a) < 30:
            return False
        
        ra = np.array(list(returns_a))
        rb = np.array(list(returns_b))
        
        # Create spread returns
        spread_returns = ra - hedge_ratio * rb
        
        # Cointegration heuristic: spread variance should be lower than individual variances
        var_ratio = np.var(spread_returns) / (np.var(ra) + np.var(rb) + 1e-10)
        
        # High correlation and low spread variance suggests cointegration
        correlation = np.corrcoef(ra, rb)[0, 1]
        
        return var_ratio < 0.5 and abs(correlation) > 0.7
    
    def get_trading_signal(self, symbol_a: str, symbol_b: str, 
                           z_threshold: float = 2.0) -> Optional[Tuple[str, float]]:
        """
        Generate trading signal based on z-score.
        
        Returns:
            Tuple of (direction, size_multiplier) or None
            direction: 'LONG_SPREAD' (long A, short B) or 'SHORT_SPREAD'
        """
        key = f"{symbol_a}_{symbol_b}"
        if key not in self.pairs:
            return None
        
        pair = self.pairs[key]
        if len(pair['spread_history']) < self.lookback:
            return None
        
        spreads = np.array(pair['spread_history'])
        z_score = (spreads[-1] - np.mean(spreads)) / (np.std(spreads) + 1e-10)
        
        if z_score > z_threshold:
            # Spread is high - short spread (short A, long B)
            return ('SHORT_SPREAD', min(abs(z_score) / z_threshold, 2.0))
        elif z_score < -z_threshold:
            # Spread is low - long spread (long A, short B)
            return ('LONG_SPREAD', min(abs(z_score) / z_threshold, 2.0))
        
        return None
    
    def get_all_pairs_state(self) -> List[PairState]:
        """Get current state of all tracked pairs"""
        states = []
        for key, pair in self.pairs.items():
            if len(pair['spread_history']) >= self.lookback:
                spreads = np.array(pair['spread_history'])
                z_score = (spreads[-1] - np.mean(spreads)) / (np.std(spreads) + 1e-10)
                
                states.append(PairState(
                    symbol_a=pair['symbol_a'],
                    symbol_b=pair['symbol_b'],
                    hedge_ratio=pair['kalman'].get_state(),
                    spread=spreads[-1],
                    z_score=z_score,
                    is_cointegrated=self._check_cointegration_status(
                        pair['returns_a'], pair['returns_b'], pair['kalman'].get_state()
                    ),
                    half_life=self._estimate_half_life(spreads),
                    correlation=np.corrcoef(list(pair['returns_a']), list(pair['returns_b']))[0, 1] 
                                if len(pair['returns_a']) >= 20 else 0.0
                ))
        return states
