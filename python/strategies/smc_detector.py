# python/strategies/smc_detector.py
# =============================================================================
# SMART MONEY CONCEPTS (SMC) DETECTOR
# =============================================================================
# Purpose: Vectorized detection of institutional trading patterns including:
# - Break of Structure (BOS): Price breaking previous high/low with momentum
# - Change of Character (CHoCH): First sign of trend reversal
# - Order Blocks: Institutional accumulation/distribution zones
# - Fair Value Gaps (FVG): Imbalance zones where price moved too fast
#
# Implementation: Uses Numba JIT compilation for microsecond-level detection
# on streaming data. Designed to work with Nautilus Trader's bar engine.

import numpy as np
from numba import jit, prange
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum


class SMCType(Enum):
    BOS_BULLISH = "BOS_BULL"
    BOS_BEARISH = "BOS_BEAR"
    CHOC_BULLISH = "CHOC_BULL"
    CHOC_BEARISH = "CHOC_BEAR"
    ORDER_BLOCK_BULL = "OB_BULL"
    ORDER_BLOCK_BEAR = "OB_BEAR"
    FVG_BULLISH = "FVG_BULL"
    FVG_BEARISH = "FVG_BEAR"


@dataclass
class SMCSignal:
    """Represents a detected Smart Money Concept pattern"""
    smc_type: SMCType
    timestamp_ns: int
    price: float
    confidence: float  # 0.0 to 1.0
    zone_start: float  # For zones like OB/FVG
    zone_end: float
    volume_confirmation: bool


@jit(nopython=True, cache=True)
def detect_break_of_structure(highs: np.ndarray, lows: np.ndarray, 
                               swing_window: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect Break of Structure (BOS) events.
    
    A bullish BOS occurs when price breaks above a previous swing high.
    A bearish BOS occurs when price breaks below a previous swing low.
    
    Args:
        highs: Array of high prices
        lows: Array of low prices
        swing_window: Number of bars to consider for swing points
        
    Returns:
        Tuple of (bullish_bos_indices, bearish_bos_indices)
    """
    n = len(highs)
    if n < swing_window * 2:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    
    bullish_bos = []
    bearish_bos = []
    
    # Track swing highs and lows
    prev_swing_high = -np.inf
    prev_swing_low = np.inf
    prev_swing_high_idx = -1
    prev_swing_low_idx = -1
    
    for i in range(swing_window, n - swing_window):
        # Check for swing high
        is_swing_high = True
        for j in range(i - swing_window, i + swing_window + 1):
            if j != i and highs[j] >= highs[i]:
                is_swing_high = False
                break
        
        # Check for swing low
        is_swing_low = True
        for j in range(i - swing_window, i + swing_window + 1):
            if j != i and lows[j] <= lows[i]:
                is_swing_low = False
                break
        
        if is_swing_high:
            # Check if we broke the previous swing high
            if i > prev_swing_high_idx and prev_swing_high_idx >= 0:
                if highs[i] > prev_swing_high:
                    bullish_bos.append(i)
            prev_swing_high = highs[i]
            prev_swing_high_idx = i
            
        if is_swing_low:
            # Check if we broke the previous swing low
            if i > prev_swing_low_idx and prev_swing_low_idx >= 0:
                if lows[i] < prev_swing_low:
                    bearish_bos.append(i)
            prev_swing_low = lows[i]
            prev_swing_low_idx = i
    
    return np.array(bullish_bos, dtype=np.int64), np.array(bearish_bos, dtype=np.int64)


@jit(nopython=True, cache=True)
def detect_fair_value_gaps(highs: np.ndarray, lows: np.ndarray, 
                            closes: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect Fair Value Gaps (FVG) - imbalance zones where price moved too quickly.
    
    A bullish FVG: Low of candle i > High of candle i-2 (gap in between)
    A bearish FVG: High of candle i < Low of candle i-2 (gap in between)
    
    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of close prices
        threshold: Minimum gap size as percentage of ATR
        
    Returns:
        Tuple of (bullish_fvg_indices, bearish_fvg_indices, fvg_starts, fvg_ends)
    """
    n = len(highs)
    if n < 3:
        return (np.array([], dtype=np.int64), np.array([], dtype=np.int64),
                np.array([], dtype=np.float64), np.array([], dtype=np.float64))
    
    bullish_fvg = []
    bearish_fvg = []
    fvg_starts = []
    fvg_ends = []
    
    # Simple ATR approximation (will be replaced with proper ATR in pipeline)
    atr = np.zeros(n)
    for i in range(1, n):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        atr[i] = (atr[i-1] * 13 + tr) / 14 if i > 1 else tr
    
    for i in range(2, n):
        # Bullish FVG: Current low > Candle i-2 high
        if lows[i] > highs[i-2]:
            gap_size = lows[i] - highs[i-2]
            if gap_size > threshold * atr[i]:
                bullish_fvg.append(i)
                fvg_starts.append(highs[i-2])  # Top of candle i-2
                fvg_ends.append(lows[i])        # Bottom of candle i
                
        # Bearish FVG: Current high < Candle i-2 low
        elif highs[i] < lows[i-2]:
            gap_size = lows[i-2] - highs[i]
            if gap_size > threshold * atr[i]:
                bearish_fvg.append(i)
                fvg_starts.append(lows[i-2])   # Bottom of candle i-2
                fvg_ends.append(highs[i])       # Top of candle i
    
    return (np.array(bullish_fvg, dtype=np.int64), 
            np.array(bearish_fvg, dtype=np.int64),
            np.array(fvg_starts, dtype=np.float64),
            np.array(fvg_ends, dtype=np.float64))


@jit(nopython=True, cache=True)
def detect_order_blocks(highs: np.ndarray, lows: np.ndarray, opens: np.ndarray,
                        closes: np.ndarray, volumes: np.ndarray, 
                        lookback: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect Order Blocks - candles representing institutional accumulation/distribution.
    
    Bullish OB: Last down candle before a strong upward move (BOS)
    Bearish OB: Last up candle before a strong downward move (BOS)
    
    Args:
        highs, lows, opens, closes: OHLC data
        volumes: Volume data
        lookback: Bars to check for strong move after OB
        
    Returns:
        Tuple of (ob_indices, ob_types, ob_confidence)
    """
    n = len(closes)
    if n < lookback + 5:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int8), np.array([], dtype=np.float64)
    
    ob_indices = []
    ob_types = []  # 1 = bullish, -1 = bearish
    ob_confidence = []
    
    for i in range(5, n - lookback):
        # Check for bullish OB (down candle followed by strong up move)
        if closes[i] < opens[i]:  # Down candle
            # Check if next 'lookback' bars show strong upward movement
            max_high_after = np.max(highs[i+1:i+1+lookback])
            move_strength = (max_high_after - closes[i]) / closes[i]
            
            if move_strength > 0.02:  # 2% move threshold
                volume_avg = np.mean(volumes[max(0, i-5):i])
                vol_confirmation = volumes[i] > volume_avg * 1.5
                
                confidence = min(move_strength / 0.05, 1.0)
                if vol_confirmation:
                    confidence = min(confidence + 0.2, 1.0)
                
                ob_indices.append(i)
                ob_types.append(1)  # Bullish
                ob_confidence.append(confidence)
        
        # Check for bearish OB (up candle followed by strong down move)
        elif closes[i] > opens[i]:  # Up candle
            max_low_after = np.min(lows[i+1:i+1+lookback])
            move_strength = (opens[i] - max_low_after) / opens[i]
            
            if move_strength > 0.02:
                volume_avg = np.mean(volumes[max(0, i-5):i])
                vol_confirmation = volumes[i] > volume_avg * 1.5
                
                confidence = min(move_strength / 0.05, 1.0)
                if vol_confirmation:
                    confidence = min(confidence + 0.2, 1.0)
                
                ob_indices.append(i)
                ob_types.append(-1)  # Bearish
                ob_confidence.append(confidence)
    
    return np.array(ob_indices, dtype=np.int64), np.array(ob_types, dtype=np.int8), np.array(ob_confidence, dtype=np.float64)


class SMCDetector:
    """
    Main detector class that wraps the Numba functions and maintains state.
    Designed for real-time streaming integration.
    """
    
    def __init__(self, swing_window: int = 5, fvg_threshold: float = 0.5, ob_lookback: int = 10):
        self.swing_window = swing_window
        self.fvg_threshold = fvg_threshold
        self.ob_lookback = ob_lookback
        
        # Rolling buffers (pre-allocated for zero GC)
        self.max_bars = 1000
        self.highs = np.zeros(self.max_bars)
        self.lows = np.zeros(self.max_bars)
        self.opens = np.zeros(self.max_bars)
        self.closes = np.zeros(self.max_bars)
        self.volumes = np.zeros(self.max_bars)
        self.timestamps = np.zeros(self.max_bars, dtype=np.int64)
        self.bar_count = 0
        
        # Detected signals queue
        self.signals: List[SMCSignal] = []
    
    def add_bar(self, timestamp_ns: int, open_p: float, high: float, 
                low: float, close: float, volume: float) -> List[SMCSignal]:
        """Add a new bar and check for SMC patterns"""
        idx = self.bar_count % self.max_bars
        self.highs[idx] = high
        self.lows[idx] = low
        self.opens[idx] = open_p
        self.closes[idx] = close
        self.volumes[idx] = volume
        self.timestamps[idx] = timestamp_ns
        self.bar_count += 1
        
        # Need minimum bars for detection
        if self.bar_count < self.swing_window * 3:
            return []
        
        new_signals = []
        effective_count = min(self.bar_count, self.max_bars)
        
        # Detect BOS
        bull_bos, bear_bos = detect_break_of_structure(
            self.highs[:effective_count], 
            self.lows[:effective_count], 
            self.swing_window
        )
        
        # Only report recent detections (last bar)
        if len(bull_bos) > 0 and bull_bos[-1] == effective_count - 1:
            new_signals.append(SMCSignal(
                smc_type=SMCType.BOS_BULLISH,
                timestamp_ns=timestamp_ns,
                price=high,
                confidence=0.8,
                zone_start=0, zone_end=0,
                volume_confirmation=True
            ))
        
        if len(bear_bos) > 0 and bear_bos[-1] == effective_count - 1:
            new_signals.append(SMCSignal(
                smc_type=SMCType.BOS_BEARISH,
                timestamp_ns=timestamp_ns,
                price=low,
                confidence=0.8,
                zone_start=0, zone_end=0,
                volume_confirmation=True
            ))
        
        # Detect FVG
        bull_fvg, bear_fvg, starts, ends = detect_fair_value_gaps(
            self.highs[:effective_count],
            self.lows[:effective_count],
            self.closes[:effective_count],
            self.fvg_threshold
        )
        
        if len(bull_fvg) > 0 and bull_fvg[-1] == effective_count - 1:
            idx_fvg = len(bull_fvg) - 1
            new_signals.append(SMCSignal(
                smc_type=SMCType.FVG_BULLISH,
                timestamp_ns=timestamp_ns,
                price=(starts[idx_fvg] + ends[idx_fvg]) / 2,
                confidence=0.7,
                zone_start=starts[idx_fvg],
                zone_end=ends[idx_fvg],
                volume_confirmation=False
            ))
        
        if len(bear_fvg) > 0 and bear_fvg[-1] == effective_count - 1:
            idx_fvg = len(bear_fvg) - 1
            new_signals.append(SMCSignal(
                smc_type=SMCType.FVG_BEARISH,
                timestamp_ns=timestamp_ns,
                price=(starts[idx_fvg] + ends[idx_fvg]) / 2,
                confidence=0.7,
                zone_start=starts[idx_fvg],
                zone_end=ends[idx_fvg],
                volume_confirmation=False
            ))
        
        # Detect Order Blocks
        ob_idxs, ob_types, ob_confs = detect_order_blocks(
            self.highs[:effective_count],
            self.lows[:effective_count],
            self.opens[:effective_count],
            self.closes[:effective_count],
            self.volumes[:effective_count],
            self.ob_lookback
        )
        
        if len(ob_idxs) > 0 and ob_idxs[-1] == effective_count - 1:
            idx_ob = len(ob_idxs) - 1
            smc_type = SMCType.ORDER_BLOCK_BULL if ob_types[idx_ob] == 1 else SMCType.ORDER_BLOCK_BEAR
            new_signals.append(SMCSignal(
                smc_type=smc_type,
                timestamp_ns=timestamp_ns,
                price=self.closes[ob_idxs[idx_ob]],
                confidence=ob_confs[idx_ob],
                zone_start=self.lows[ob_idxs[idx_ob]],
                zone_end=self.highs[ob_idxs[idx_ob]],
                volume_confirmation=True
            ))
        
        self.signals.extend(new_signals)
        return new_signals
    
    def drain_signals(self) -> List[SMCSignal]:
        """Consume all pending signals"""
        signals = self.signals.copy()
        self.signals.clear()
        return signals
