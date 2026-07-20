# python/features/orderflow_metrics.py
# =============================================================================
# STAGE 2 - CHAPTER 2 - FILE 2
# Focus: Real-time order flow metrics (CVD, Imbalance, Volume Profile).
# Target: Microsecond calculation from raw tick data.
# =============================================================================

import numpy as np
from numba import jit
from typing import Tuple, Optional, List
from collections import deque
import threading


@jit(nopython=True, cache=True, fastmath=True)
def calculate_cvd(
    prices: np.ndarray,
    volumes: np.ndarray,
    is_buyer_maker: np.ndarray
) -> np.ndarray:
    """
    Calculate Cumulative Volume Delta (CVD).
    CVD = Sum of (buy volume - sell volume)
    
    Args:
        prices: Array of trade prices
        volumes: Array of trade volumes
        is_buyer_maker: Boolean array (True = seller initiated, False = buyer initiated)
    
    Returns:
        Cumulative CVD values
    """
    n = len(prices)
    cvd = np.empty(n, dtype=np.float64)
    cumulative = 0.0
    
    for i in range(n):
        if is_buyer_maker[i]:
            # Seller initiated (aggressive sell)
            cumulative -= volumes[i]
        else:
            # Buyer initiated (aggressive buy)
            cumulative += volumes[i]
        cvd[i] = cumulative
    
    return cvd


@jit(nopython=True, cache=True, fastmath=True)
def calculate_order_flow_imbalance(
    bid_volumes: np.ndarray,
    ask_volumes: np.ndarray,
    window: int = 100
) -> np.ndarray:
    """
    Calculate Order Flow Imbalance (OFI).
    OFI = (Ask Volume - Bid Volume) / (Ask Volume + Bid Volume)
    
    Args:
        bid_volumes: Array of bid-side volumes
        ask_volumes: Array of ask-side volumes
        window: Rolling window size
    
    Returns:
        Array of OFI values (-1 to 1)
    """
    n = len(bid_volumes)
    ofi = np.empty(n, dtype=np.float64)
    ofi[:] = np.nan
    
    if n < window:
        return ofi
    
    for i in range(window - 1, n):
        bid_sum = np.sum(bid_volumes[i - window + 1:i + 1])
        ask_sum = np.sum(ask_volumes[i - window + 1:i + 1])
        
        total = bid_sum + ask_sum
        if total > 0:
            ofi[i] = (ask_sum - bid_sum) / total
        else:
            ofi[i] = 0.0
    
    return ofi


@jit(nopython=True, cache=True)
def find_volume_profile_levels(
    prices: np.ndarray,
    volumes: np.ndarray,
    price_range_min: float,
    price_range_max: float,
    num_bins: int = 50
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Calculate Volume Profile with POC, VAH, VAL.
    
    Args:
        prices: Array of trade prices
        volumes: Array of trade volumes
        price_range_min: Minimum price for binning
        price_range_max: Maximum price for binning
        num_bins: Number of price bins
    
    Returns:
        Tuple of (bin_centers, bin_volumes, POC, VAH, VAL)
        VAH/VAL are Value Area High/Low (70% of volume)
    """
    bin_width = (price_range_max - price_range_min) / num_bins
    bin_volumes = np.zeros(num_bins, dtype=np.float64)
    bin_centers = np.empty(num_bins, dtype=np.float64)
    
    # Initialize bin centers
    for i in range(num_bins):
        bin_centers[i] = price_range_min + (i + 0.5) * bin_width
    
    # Accumulate volumes into bins
    n = len(prices)
    for i in range(n):
        if prices[i] < price_range_min or prices[i] >= price_range_max:
            continue
        bin_idx = int((prices[i] - price_range_min) / bin_width)
        if 0 <= bin_idx < num_bins:
            bin_volumes[bin_idx] += volumes[i]
    
    # Find POC (Point of Control - highest volume bin)
    poc_idx = 0
    max_vol = bin_volumes[0]
    for i in range(1, num_bins):
        if bin_volumes[i] > max_vol:
            max_vol = bin_volumes[i]
            poc_idx = i
    poc = bin_centers[poc_idx]
    
    # Calculate Value Area (70% of total volume around POC)
    total_volume = np.sum(bin_volumes)
    target_volume = total_volume * 0.70
    
    # Expand from POC to find VAH and VAL
    left_idx = poc_idx
    right_idx = poc_idx
    accumulated_vol = bin_volumes[poc_idx]
    
    while accumulated_vol < target_volume:
        # Check which side to expand
        left_vol = 0.0
        right_vol = 0.0
        
        if left_idx > 0:
            left_vol = bin_volumes[left_idx - 1]
        if right_idx < num_bins - 1:
            right_vol = bin_volumes[right_idx + 1]
        
        if left_vol >= right_vol and left_idx > 0:
            left_idx -= 1
            accumulated_vol += bin_volumes[left_idx]
        elif right_idx < num_bins - 1:
            right_idx += 1
            accumulated_vol += bin_volumes[right_idx]
        else:
            break
    
    vah = bin_centers[right_idx]
    val = bin_centers[left_idx]
    
    return bin_centers, bin_volumes, poc, vah, val


@jit(nopython=True, cache=True, fastmath=True)
def detect_absorption(
    prices: np.ndarray,
    volumes: np.ndarray,
    lookback: int = 10,
    volume_threshold: float = 2.0
) -> np.ndarray:
    """
    Detect absorption patterns (high volume with little price movement).
    
    Args:
        prices: Array of prices
        volumes: Array of volumes
        lookback: Number of bars to check
        volume_threshold: Multiplier above average volume to consider "high"
    
    Returns:
        Array of absorption signals (1 = bullish absorption, -1 = bearish, 0 = none)
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.int8)
    
    if n < lookback:
        return signals
    
    # Calculate average volume
    avg_vol = np.mean(volumes[:lookback])
    
    for i in range(lookback, n):
        # Check if current volume is significantly higher than average
        if volumes[i] < avg_vol * volume_threshold:
            continue
        
        # Calculate price range over lookback
        price_range = np.max(prices[i-lookback:i+1]) - np.min(prices[i-lookback:i+1])
        
        # If high volume but small price movement = absorption
        if price_range < (np.mean(prices[i-lookback:i]) * 0.001):  # 0.1% range
            # Determine direction based on close vs open of the range
            if prices[i] > prices[i - lookback]:
                signals[i] = 1  # Bullish absorption (buying despite resistance)
            else:
                signals[i] = -1  # Bearish absorption (selling despite support)
    
    return signals


class OrderFlowAnalyzer:
    """
    Real-time order flow analyzer with circular buffers for memory efficiency.
    Maintains strict memory bounds within the 8GB system cap.
    """
    
    def __init__(self, max_ticks: int = 100000, max_levels: int = 50):
        """
        Initialize order flow analyzer with pre-allocated buffers.
        
        Args:
            max_ticks: Maximum number of ticks to store (memory bound)
            max_levels: Maximum order book levels to track
        """
        self.max_ticks = max_ticks
        self.max_levels = max_levels
        
        # Circular buffers for tick data
        self.prices = np.zeros(max_ticks, dtype=np.float64)
        self.volumes = np.zeros(max_ticks, dtype=np.float64)
        self.is_buyer_maker = np.zeros(max_ticks, dtype=np.bool_)
        
        # Order book level tracking
        self.bid_volumes = np.zeros(max_levels, dtype=np.float64)
        self.ask_volumes = np.zeros(max_levels, dtype=np.float64)
        
        # Current position in circular buffer
        self.tick_index = 0
        self.tick_count = 0
        
        # Thread lock for concurrent updates
        self._lock = threading.Lock()
        
        # Cached results (updated on each tick)
        self._cvd_cache = None
        self._ofi_cache = None
    
    def add_tick(self, price: float, volume: float, is_seller: bool) -> None:
        """
        Add a new tick to the circular buffer.
        
        Args:
            price: Trade price
            volume: Trade volume
            is_seller: True if seller-initiated (buyer_maker), False if buyer-initiated
        """
        with self._lock:
            idx = self.tick_index
            self.prices[idx] = price
            self.volumes[idx] = volume
            self.is_buyer_maker[idx] = is_seller
            
            self.tick_index = (idx + 1) % self.max_ticks
            if self.tick_count < self.max_ticks:
                self.tick_count += 1
            
            # Invalidate caches
            self._cvd_cache = None
            self._ofi_cache = None
    
    def update_order_book_levels(
        self, 
        bid_vols: np.ndarray, 
        ask_vols: np.ndarray
    ) -> None:
        """Update order book level volumes."""
        with self._lock:
            length = min(len(bid_vols), self.max_levels)
            self.bid_volumes[:length] = bid_vols[:length]
            self.ask_volumes[:length] = ask_vols[:length]
    
    def get_cvd(self) -> np.ndarray:
        """Get cached or calculate CVD."""
        if self._cvd_cache is None:
            with self._lock:
                valid_data = self.tick_count
                self._cvd_cache = calculate_cvd(
                    self.prices[:valid_data],
                    self.volumes[:valid_data],
                    self.is_buyer_maker[:valid_data]
                )
        return self._cvd_cache
    
    def get_ofi(self, window: int = 100) -> np.ndarray:
        """Get cached or calculate Order Flow Imbalance."""
        if self._ofi_cache is None:
            with self._lock:
                # Use top 5 levels for OFI calculation
                self._ofi_cache = calculate_order_flow_imbalance(
                    self.bid_volumes[:5],
                    self.ask_volumes[:5],
                    window
                )
        return self._ofi_cache
    
    def get_volume_profile(
        self, 
        num_bins: int = 50,
        price_range_pct: float = 0.02
    ) -> dict:
        """
        Calculate current volume profile.
        
        Args:
            num_bins: Number of price bins
            price_range_pct: Price range as percentage of current price
        
        Returns:
            Dictionary with POC, VAH, VAL, and full profile data
        """
        with self._lock:
            valid_data = self.tick_count
            if valid_data == 0:
                return {}
            
            prices = self.prices[:valid_data]
            volumes = self.volumes[:valid_data]
            
            current_price = prices[-1]
            price_min = current_price * (1 - price_range_pct)
            price_max = current_price * (1 + price_range_pct)
            
            centers, vols, poc, vah, val = find_volume_profile_levels(
                prices, volumes, price_min, price_max, num_bins
            )
            
            return {
                'poc': poc,
                'vah': vah,
                'val': val,
                'bin_centers': centers,
                'bin_volumes': vols,
                'current_price': current_price
            }
    
    def get_absorption_signals(self, lookback: int = 10) -> np.ndarray:
        """Detect recent absorption patterns."""
        with self._lock:
            valid_data = self.tick_count
            return detect_absorption(
                self.prices[:valid_data],
                self.volumes[:valid_data],
                lookback
            )
    
    def get_imbalance_ratio(self) -> float:
        """Get current bid/ask imbalance ratio."""
        with self._lock:
            bid_total = np.sum(self.bid_volumes[:10])  # Top 10 levels
            ask_total = np.sum(self.ask_volumes[:10])
            total = bid_total + ask_total
            if total == 0:
                return 0.0
            return (ask_total - bid_total) / total


class TickAggregator:
    """
    Aggregates raw ticks into time/volume bars for efficient processing.
    Reduces data volume while preserving order flow information.
    """
    
    def __init__(self, bar_type: str = 'time', bar_size: int = 1000):
        """
        Initialize tick aggregator.
        
        Args:
            bar_type: 'time' (ms), 'volume', or 'tick'
            bar_size: Size threshold for bar completion
        """
        self.bar_type = bar_type
        self.bar_size = bar_size
        
        # Current bar state
        self.current_open: Optional[float] = None
        self.current_high: Optional[float] = None
        self.current_low: Optional[float] = None
        self.current_close: Optional[float] = None
        self.current_volume: float = 0.0
        self.current_buy_volume: float = 0.0
        self.current_sell_volume: float = 0.0
        self.current_tick_count: int = 0
        self.current_start_time: int = 0
        
        # Completed bars queue (bounded)
        self.bars: deque = deque(maxlen=10000)
        self._lock = threading.Lock()
    
    def process_tick(
        self, 
        price: float, 
        volume: float, 
        is_seller: bool,
        timestamp: int
    ) -> Optional[dict]:
        """
        Process a tick and return completed bar if threshold reached.
        
        Args:
            price: Trade price
            volume: Trade volume
            is_seller: True if seller-initiated
            timestamp: Unix timestamp in milliseconds
        
        Returns:
            Completed bar dict or None
        """
        with self._lock:
            # Initialize bar if needed
            if self.current_open is None:
                self.current_open = price
                self.current_high = price
                self.current_low = price
                self.current_start_time = timestamp
            
            # Update bar statistics
            self.current_high = max(self.current_high, price)
            self.current_low = min(self.current_low, price)
            self.current_close = price
            self.current_volume += volume
            self.current_tick_count += 1
            
            if is_seller:
                self.current_sell_volume += volume
            else:
                self.current_buy_volume += volume
            
            # Check if bar is complete
            bar_complete = False
            
            if self.bar_type == 'tick' and self.current_tick_count >= self.bar_size:
                bar_complete = True
            elif self.bar_type == 'volume' and self.current_volume >= self.bar_size:
                bar_complete = True
            elif self.bar_type == 'time' and (timestamp - self.current_start_time) >= self.bar_size:
                bar_complete = True
            
            if bar_complete:
                bar = {
                    'open': self.current_open,
                    'high': self.current_high,
                    'low': self.current_low,
                    'close': self.current_close,
                    'volume': self.current_volume,
                    'buy_volume': self.current_buy_volume,
                    'sell_volume': self.current_sell_volume,
                    'tick_count': self.current_tick_count,
                    'start_time': self.current_start_time,
                    'end_time': timestamp
                }
                
                self.bars.append(bar)
                
                # Reset for next bar
                self.current_open = price
                self.current_high = price
                self.current_low = price
                self.current_close = price
                self.current_volume = 0.0
                self.current_buy_volume = 0.0
                self.current_sell_volume = 0.0
                self.current_tick_count = 0
                self.current_start_time = timestamp
                
                return bar
            
            return None
    
    def get_bars(self, count: int = 100) -> List[dict]:
        """Get the last N completed bars."""
        with self._lock:
            return list(self.bars)[-count:]
