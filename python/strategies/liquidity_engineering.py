# python/strategies/liquidity_engineering.py
# =============================================================================
# LIQUIDITY ENGINEERING & STOP HUNT DETECTION
# =============================================================================
# Purpose: Identify liquidity pools, stop hunts, inducement patterns, and 
# premium/discount zones based on institutional order flow concepts.
#
# Key Concepts:
# - Equal Highs/Lows (EQH/EQL): Liquidity pools where retail stops accumulate
# - Stop Hunt: Price spikes through key levels to trigger stops before reversing
# - Inducement: Patterns that lure traders into wrong positions
# - Premium/Discount Zones: Relative price position within trading ranges
# - Mitigation Blocks: Retracement zones for entry after sweeps

import numpy as np
from numba import jit
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class LiquidityType(Enum):
    EQUAL_HIGHS = "EQH"
    EQUAL_LOWS = "EQL"
    STOP_HUNT_HIGH = "SH_HIGH"
    STOP_HUNT_LOW = "SH_LOW"
    INDUCEMENT_LONG = "IND_LONG"
    INDUCEMENT_SHORT = "IND_SHORT"
    PREMIUM_ZONE = "PREMIUM"
    DISCOUNT_ZONE = "DISCOUNT"
    MITIGATION_BLOCK = "MIT_BLOCK"


@dataclass
class LiquidityZone:
    """Represents a identified liquidity zone"""
    zone_type: LiquidityType
    price_level: float
    strength: float  # 0.0 to 1.0 based on touches/volume
    timestamp_ns: int
    touched_count: int
    swept: bool  # Whether liquidity has been taken
    sweep_timestamp_ns: Optional[int] = None


@jit(nopython=True, cache=True)
def detect_equal_highs_lows(highs: np.ndarray, lows: np.ndarray, 
                             tolerance_pct: float, min_touches: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect Equal Highs (EQH) and Equal Lows (EQL) - key liquidity pools.
    
    These are areas where price has tested the same level multiple times,
    creating a pool of stop orders above/below that institutions target.
    
    Args:
        highs: Array of high prices
        lows: Array of low prices
        tolerance_pct: Percentage tolerance for "equal" levels
        min_touches: Minimum number of touches to qualify
        
    Returns:
        Tuple of (eqh_prices, eqh_counts, eql_prices, eql_counts)
    """
    n = len(highs)
    if n < 10:
        return (np.array([], dtype=np.float64), np.array([], dtype=np.int32),
                np.array([], dtype=np.float64), np.array([], dtype=np.int32))
    
    # Group highs by price level (rounded to tolerance)
    eqh_levels = []
    eqh_counts = []
    eql_levels = []
    eql_counts = []
    
    tolerance_mult = 1.0 + tolerance_pct / 100.0
    
    # Simple clustering approach
    for i in range(n):
        # Check for equal highs
        count = 1
        for j in range(max(0, i-20), min(n, i+20)):
            if i != j:
                if abs(highs[i] - highs[j]) <= highs[i] * (tolerance_mult - 1.0):
                    count += 1
        
        if count >= min_touches:
            # Check if this level already recorded
            found = False
            for idx, lvl in enumerate(eqh_levels):
                if abs(lvl - highs[i]) <= lvl * (tolerance_mult - 1.0):
                    found = True
                    break
            if not found:
                eqh_levels.append(highs[i])
                eqh_counts.append(count)
        
        # Check for equal lows
        count = 1
        for j in range(max(0, i-20), min(n, i+20)):
            if i != j:
                if abs(lows[i] - lows[j]) <= lows[i] * (tolerance_mult - 1.0):
                    count += 1
        
        if count >= min_touches:
            found = False
            for idx, lvl in enumerate(eql_levels):
                if abs(lvl - lows[i]) <= lvl * (tolerance_mult - 1.0):
                    found = True
                    break
            if not found:
                eql_levels.append(lows[i])
                eql_counts.append(count)
    
    return (np.array(eqh_levels, dtype=np.float64), 
            np.array(eqh_counts, dtype=np.int32),
            np.array(eql_levels, dtype=np.float64),
            np.array(eql_counts, dtype=np.int32))


@jit(nopython=True, cache=True)
def calculate_premium_discount_zones(highs: np.ndarray, lows: np.ndarray, 
                                      closes: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Premium and Discount zones within the current trading range.
    
    Premium Zone: Upper 50% of range (selling opportunities)
    Discount Zone: Lower 50% of range (buying opportunities)
    Fair Value: Middle 50% (avoid trading)
    
    Args:
        highs, lows, closes: Price data
        window: Lookback window for range calculation
        
    Returns:
        Tuple of (premium_threshold, discount_threshold, fair_value_mid)
    """
    n = len(closes)
    if n < window:
        return np.array([0.5]), np.array([0.0]), np.array([0.25])
    
    premiums = []
    discounts = []
    mids = []
    
    for i in range(window - 1, n):
        range_high = np.max(highs[i-window+1:i+1])
        range_low = np.min(lows[i-window+1:i+1])
        range_size = range_high - range_low
        
        if range_size == 0:
            continue
            
        mid = (range_high + range_low) / 2
        premium_thresh = mid + (range_size * 0.25)  # Top 25%
        discount_thresh = mid - (range_size * 0.25)  # Bottom 25%
        
        premiums.append(premium_thresh)
        discounts.append(discount_thresh)
        mids.append(mid)
    
    return np.array(premiums, dtype=np.float64), np.array(discounts, dtype=np.float64), np.array(mids, dtype=np.float64)


@jit(nopython=True, cache=True)
def detect_stop_hunt_pattern(highs: np.ndarray, lows: np.ndarray, 
                              opens: np.ndarray, closes: np.ndarray,
                              eqh_level: float, eql_level: float,
                              tolerance_pct: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect stop hunt patterns around EQH/EQL levels.
    
    A stop hunt occurs when:
    1. Price spikes above EQH (or below EQL)
    2. Closes back below (or above) the level
    3. Creates a long wick indicating rejection
    
    Args:
        highs, lows, opens, closes: OHLC data
        eqh_level, eql_level: Known liquidity levels
        tolerance_pct: Tolerance for level breach
        
    Returns:
        Tuple of (bullish_stop_hunt_indices, bearish_stop_hunt_indices)
    """
    n = len(closes)
    bullish_hunts = []
    bearish_hunts = []
    
    tolerance = eqh_level * tolerance_pct / 100.0 if eqh_level > 0 else 0.01
    
    for i in range(1, n):
        # Bearish stop hunt (hunt above EQH then reverse)
        if eqh_level > 0:
            if highs[i] > eqh_level and highs[i] <= eqh_level + tolerance:
                if closes[i] < eqh_level and opens[i] < eqh_level:
                    # Long upper wick
                    wick = highs[i] - max(opens[i], closes[i])
                    body = abs(closes[i] - opens[i])
                    if wick > body * 2:  # Wick at least 2x body
                        bearish_hunts.append(i)
        
        # Bullish stop hunt (hunt below EQL then reverse)
        if eql_level > 0:
            if lows[i] < eql_level and lows[i] >= eql_level - tolerance:
                if closes[i] > eql_level and opens[i] > eql_level:
                    # Long lower wick
                    wick = min(opens[i], closes[i]) - lows[i]
                    body = abs(closes[i] - opens[i])
                    if wick > body * 2:
                        bullish_hunts.append(i)
    
    return np.array(bullish_hunts, dtype=np.int64), np.array(bearish_hunts, dtype=np.int64)


class LiquidityEngineering:
    """
    Main class for liquidity analysis and zone management.
    Integrates with SMC detector for comprehensive market structure view.
    """
    
    def __init__(self, tolerance_pct: float = 0.5, min_touches: int = 3):
        self.tolerance_pct = tolerance_pct
        self.min_touches = min_touches
        
        # Pre-allocated buffers
        self.max_bars = 1000
        self.highs = np.zeros(self.max_bars)
        self.lows = np.zeros(self.max_bars)
        self.opens = np.zeros(self.max_bars)
        self.closes = np.zeros(self.max_bars)
        self.timestamps = np.zeros(self.max_bars, dtype=np.int64)
        self.bar_count = 0
        
        # Detected zones
        self.eqh_zones: List[LiquidityZone] = []
        self.eql_zones: List[LiquidityZone] = []
        self.current_premium: float = 0.0
        self.current_discount: float = 0.0
        self.current_fair_value: float = 0.0
        
        # Stop hunt alerts
        self.stop_hunt_alerts: List[Tuple[int, str, float]] = []
    
    def add_bar(self, timestamp_ns: int, open_p: float, high: float,
                low: float, close: float) -> dict:
        """Add bar and update liquidity analysis"""
        idx = self.bar_count % self.max_bars
        self.highs[idx] = high
        self.lows[idx] = low
        self.opens[idx] = open_p
        self.closes[idx] = close
        self.timestamps[idx] = timestamp_ns
        self.bar_count += 1
        
        result = {
            'new_eqh': [],
            'new_eql': [],
            'stop_hunts': [],
            'zone_status': {}
        }
        
        if self.bar_count < 20:
            return result
        
        effective = min(self.bar_count, self.max_bars)
        
        # Update EQH/EQL
        eqh_prices, eqh_counts, eql_prices, eql_counts = detect_equal_highs_lows(
            self.highs[:effective],
            self.lows[:effective],
            self.tolerance_pct,
            self.min_touches
        )
        
        # Record new EQH zones
        for i, (price, count) in enumerate(zip(eqh_prices, eqh_counts)):
            strength = min(count / 5.0, 1.0)
            zone = LiquidityZone(
                zone_type=LiquidityType.EQUAL_HIGHS,
                price_level=price,
                strength=strength,
                timestamp_ns=timestamp_ns,
                touched_count=count,
                swept=False
            )
            # Check if new
            is_new = True
            for existing in self.eqh_zones:
                if abs(existing.price_level - price) < price * 0.001:
                    is_new = False
                    existing.touched_count = count
                    break
            if is_new:
                self.eqh_zones.append(zone)
                result['new_eqh'].append(price)
        
        # Record new EQL zones
        for i, (price, count) in enumerate(zip(eql_prices, eql_counts)):
            strength = min(count / 5.0, 1.0)
            zone = LiquidityZone(
                zone_type=LiquidityType.EQUAL_LOWS,
                price_level=price,
                strength=strength,
                timestamp_ns=timestamp_ns,
                touched_count=count,
                swept=False
            )
            is_new = True
            for existing in self.eql_zones:
                if abs(existing.price_level - price) < price * 0.001:
                    is_new = False
                    existing.touched_count = count
                    break
            if is_new:
                self.eql_zones.append(zone)
                result['new_eql'].append(price)
        
        # Update premium/discount zones
        premiums, discounts, mids = calculate_premium_discount_zones(
            self.highs[:effective],
            self.lows[:effective],
            self.closes[:effective],
            min(50, effective)
        )
        
        if len(premiums) > 0:
            self.current_premium = premiums[-1]
            self.current_discount = discounts[-1]
            self.current_fair_value = mids[-1]
            
            # Determine current zone status
            if close > self.current_premium:
                result['zone_status'] = {'zone': 'PREMIUM', 'bias': 'BEARISH'}
            elif close < self.current_discount:
                result['zone_status'] = {'zone': 'DISCOUNT', 'bias': 'BULLISH'}
            else:
                result['zone_status'] = {'zone': 'FAIR_VALUE', 'bias': 'NEUTRAL'}
        
        # Check for stop hunts against known levels
        for eqh in self.eqh_zones:
            bull_hunts, bear_hunts = detect_stop_hunt_pattern(
                self.highs[:effective],
                self.lows[:effective],
                self.opens[:effective],
                self.closes[:effective],
                eqh.price_level,
                0,  # No EQL for EQH check
                self.tolerance_pct
            )
            if len(bear_hunts) > 0 and bear_hunts[-1] == effective - 1:
                eqh.swept = True
                eqh.sweep_timestamp_ns = timestamp_ns
                result['stop_hunts'].append({
                    'type': 'BEARISH_STOP_HUNT',
                    'level': eqh.price_level,
                    'timestamp_ns': timestamp_ns
                })
                self.stop_hunt_alerts.append((timestamp_ns, 'BEARISH_SH', eqh.price_level))
        
        for eql in self.eql_zones:
            bull_hunts, bear_hunts = detect_stop_hunt_pattern(
                self.highs[:effective],
                self.lows[:effective],
                self.opens[:effective],
                self.closes[:effective],
                0,  # No EQH for EQL check
                eql.price_level,
                self.tolerance_pct
            )
            if len(bull_hunts) > 0 and bull_hunts[-1] == effective - 1:
                eql.swept = True
                eql.sweep_timestamp_ns = timestamp_ns
                result['stop_hunts'].append({
                    'type': 'BULLISH_STOP_HUNT',
                    'level': eql.price_level,
                    'timestamp_ns': timestamp_ns
                })
                self.stop_hunt_alerts.append((timestamp_ns, 'BULLISH_SH', eql.price_level))
        
        return result
    
    def get_liquidity_map(self) -> dict:
        """Return current liquidity landscape"""
        return {
            'eqh_zones': [(z.price_level, z.strength, z.swept) for z in self.eqh_zones],
            'eql_zones': [(z.price_level, z.strength, z.swept) for z in self.eql_zones],
            'premium': self.current_premium,
            'discount': self.current_discount,
            'fair_value': self.current_fair_value,
            'recent_stop_hunts': self.stop_hunt_alerts[-10:]
        }
    
    def clear_old_zones(self, max_age_bars: int = 100):
        """Remove zones that haven't been relevant recently"""
        cutoff = self.timestamps[self.bar_count - 1] - (max_age_bars * 60_000_000_000)  # Assume 1min bars
        self.eqh_zones = [z for z in self.eqh_zones if z.timestamp_ns > cutoff or not z.swept]
        self.eql_zones = [z for z in self.eql_zones if z.timestamp_ns > cutoff or not z.swept]
