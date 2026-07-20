# python/features/technical_indicators.py
# =============================================================================
# STAGE 2 - CHAPTER 2 - FILE 1
# Focus: Vectorized, zero-allocation technical indicators using Numba.
# Target: AMD Ryzen AI 5 with AVX2 optimizations via LLVM.
# =============================================================================

import numpy as np
from numba import jit, prange
from typing import Tuple, Optional

# Configure Numba for maximum performance on AMD Ryzen
# parallel=True enables multi-threading across CPU cores
# fastmath=True allows aggressive floating-point optimizations
NUMBA_CONFIG = {
    'nopython': True,
    'parallel': True,
    'fastmath': True,
    'cache': True
}


@jit(**NUMBA_CONFIG)
def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index (RSI) using Wilder's smoothing method.
    
    Args:
        prices: Array of closing prices (must be contiguous in memory)
        period: RSI calculation period (default 14)
    
    Returns:
        Array of RSI values (0-100 scale), NaN for initial period
    """
    n = len(prices)
    rsi = np.empty(n, dtype=np.float64)
    rsi[:] = np.nan
    
    if n <= period:
        return rsi
    
    # Calculate price changes
    deltas = np.diff(prices)
    
    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    # Initial SMA for first RSI value
    avg_gain = np.sum(gains[:period]) / period
    avg_loss = np.sum(losses[:period]) / period
    
    # First RSI calculation
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))
    
    # Wilder's smoothing for subsequent values
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


@jit(**NUMBA_CONFIG)
def calculate_macd(
    prices: np.ndarray, 
    fast_period: int = 12, 
    slow_period: int = 26, 
    signal_period: int = 9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        prices: Array of closing prices
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line EMA period (default 9)
    
    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    n = len(prices)
    macd_line = np.empty(n, dtype=np.float64)
    signal_line = np.empty(n, dtype=np.float64)
    histogram = np.empty(n, dtype=np.float64)
    
    macd_line[:] = np.nan
    signal_line[:] = np.nan
    histogram[:] = np.nan
    
    if n < slow_period:
        return macd_line, signal_line, histogram
    
    # EMA smoothing constants
    fast_mult = 2.0 / (fast_period + 1)
    slow_mult = 2.0 / (slow_period + 1)
    signal_mult = 2.0 / (signal_period + 1)
    
    # Initialize EMAs with SMA
    fast_ema = np.sum(prices[:fast_period]) / fast_period
    slow_ema = np.sum(prices[:slow_period]) / slow_period
    
    # Calculate MACD line
    for i in range(fast_period, n):
        fast_ema = (prices[i] - fast_ema) * fast_mult + fast_ema
        if i >= slow_period:
            slow_ema = (prices[i] - slow_ema) * slow_mult + slow_ema
            macd_line[i] = fast_ema - slow_ema
    
    # Find first valid MACD value for signal calculation
    first_valid = slow_period
    while first_valid < n and np.isnan(macd_line[first_valid]):
        first_valid += 1
    
    if first_valid >= n:
        return macd_line, signal_line, histogram
    
    # Initialize signal EMA
    signal_ema = macd_line[first_valid]
    signal_line[first_valid] = signal_ema
    histogram[first_valid] = macd_line[first_valid] - signal_ema
    
    # Calculate signal line and histogram
    for i in range(first_valid + 1, n):
        if not np.isnan(macd_line[i]):
            signal_ema = (macd_line[i] - signal_ema) * signal_mult + signal_ema
            signal_line[i] = signal_ema
            histogram[i] = macd_line[i] - signal_ema
    
    return macd_line, signal_line, histogram


@jit(**NUMBA_CONFIG)
def calculate_bollinger_bands(
    prices: np.ndarray, 
    period: int = 20, 
    std_dev: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Bollinger Bands (Middle, Upper, Lower).
    
    Args:
        prices: Array of closing prices
        period: Moving average period (default 20)
        std_dev: Standard deviation multiplier (default 2.0)
    
    Returns:
        Tuple of (Middle Band, Upper Band, Lower Band)
    """
    n = len(prices)
    middle = np.empty(n, dtype=np.float64)
    upper = np.empty(n, dtype=np.float64)
    lower = np.empty(n, dtype=np.float64)
    
    middle[:] = np.nan
    upper[:] = np.nan
    lower[:] = np.nan
    
    if n < period:
        return middle, upper, lower
    
    for i in range(period - 1, n):
        # Calculate SMA
        window = prices[i - period + 1:i + 1]
        sma = np.sum(window) / period
        middle[i] = sma
        
        # Calculate standard deviation
        variance = np.sum((window - sma) ** 2) / period
        std = np.sqrt(variance)
        
        upper[i] = sma + (std_dev * std)
        lower[i] = sma - (std_dev * std)
    
    return middle, upper, lower


@jit(**NUMBA_CONFIG)
def calculate_vwap(
    highs: np.ndarray, 
    lows: np.ndarray, 
    closes: np.ndarray, 
    volumes: np.ndarray
) -> np.ndarray:
    """
    Calculate Volume Weighted Average Price (VWAP).
    
    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of close prices
        volumes: Array of volumes
    
    Returns:
        Array of VWAP values
    """
    n = len(closes)
    vwap = np.empty(n, dtype=np.float64)
    vwap[:] = np.nan
    
    if n == 0 or len(highs) != n or len(lows) != n or len(volumes) != n:
        return vwap
    
    cumulative_pv = 0.0
    cumulative_vol = 0.0
    
    for i in range(n):
        # Typical price = (High + Low + Close) / 3
        typical_price = (highs[i] + lows[i] + closes[i]) / 3.0
        pv = typical_price * volumes[i]
        
        cumulative_pv += pv
        cumulative_vol += volumes[i]
        
        if cumulative_vol > 0:
            vwap[i] = cumulative_pv / cumulative_vol
    
    return vwap


@jit(**NUMBA_CONFIG)
def calculate_atr(
    highs: np.ndarray, 
    lows: np.ndarray, 
    closes: np.ndarray, 
    period: int = 14
) -> np.ndarray:
    """
    Calculate Average True Range (ATR) for volatility measurement.
    
    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of close prices
        period: ATR period (default 14)
    
    Returns:
        Array of ATR values
    """
    n = len(closes)
    atr = np.empty(n, dtype=np.float64)
    atr[:] = np.nan
    
    if n <= period:
        return atr
    
    # Calculate True Range for each bar
    tr = np.empty(n, dtype=np.float64)
    tr[0] = highs[0] - lows[0]
    
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)
    
    # Initial ATR (SMA of first TR values)
    current_atr = np.sum(tr[:period]) / period
    atr[period - 1] = current_atr
    
    # Wilder's smoothing
    for i in range(period, n):
        current_atr = (current_atr * (period - 1) + tr[i]) / period
        atr[i] = current_atr
    
    return atr


class TechnicalIndicators:
    """
    High-performance technical indicator calculator with memory pooling.
    Reuses arrays to minimize allocations during real-time streaming.
    """
    
    def __init__(self, max_length: int = 10000):
        """
        Initialize with pre-allocated buffers.
        
        Args:
            max_length: Maximum array size to pre-allocate
        """
        self.max_length = max_length
        self._price_buffer = np.zeros(max_length, dtype=np.float64)
        self._volume_buffer = np.zeros(max_length, dtype=np.float64)
        self._high_buffer = np.zeros(max_length, dtype=np.float64)
        self._low_buffer = np.zeros(max_length, dtype=np.float64)
        self._close_buffer = np.zeros(max_length, dtype=np.float64)
    
    def update_buffers(
        self,
        prices: np.ndarray,
        highs: Optional[np.ndarray] = None,
        lows: Optional[np.ndarray] = None,
        volumes: Optional[np.ndarray] = None
    ) -> None:
        """Update internal buffers with new data (zero-copy when possible)."""
        length = min(len(prices), self.max_length)
        self._price_buffer[:length] = prices[:length]
        
        if highs is not None:
            self._high_buffer[:length] = highs[:length]
        if lows is not None:
            self._low_buffer[:length] = lows[:length]
        if volumes is not None:
            self._volume_buffer[:length] = volumes[:length]
    
    def get_rsi(self, period: int = 14) -> np.ndarray:
        """Calculate RSI on current buffer."""
        return calculate_rsi(self._price_buffer, period)
    
    def get_macd(
        self, 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD on current buffer."""
        return calculate_macd(self._price_buffer, fast, slow, signal)
    
    def get_bollinger_bands(
        self, 
        period: int = 20, 
        std_dev: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands on current buffer."""
        return calculate_bollinger_bands(self._price_buffer, period, std_dev)
    
    def get_vwap(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray
    ) -> np.ndarray:
        """Calculate VWAP."""
        return calculate_vwap(highs, lows, self._price_buffer, volumes)


# Example usage and benchmarking
if __name__ == "__main__":
    import time
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 10000
    prices = np.cumsum(np.random.randn(n_samples)) + 100
    
    # Benchmark RSI
    start = time.perf_counter()
    for _ in range(1000):
        rsi = calculate_rsi(prices, 14)
    elapsed = time.perf_counter() - start
    print(f"RSI (1000 iterations, {n_samples} samples): {elapsed*1000:.2f}ms")
    print(f"Per iteration: {elapsed*1000000:.2f}μs")
