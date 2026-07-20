# python/strategies/smc_strategy.py
# =============================================================================
# SMC STRATEGY - NAUTILUS TRADER INTEGRATION
# =============================================================================
# Purpose: Core Nautilus Trader strategy class that integrates Smart Money
# Concepts (SMC) signals with order flow confirmation from the Rust microstructure
# engine to generate high-probability limit order entries.
#
# Architecture:
# - Subscribes to Rust IPC for real-time order book and trade data
# - Uses SMC detector for pattern recognition
# - Uses Liquidity Engineering for zone analysis
# - Generates limit orders at optimal entry points (FVG, Order Blocks)

import numpy as np
from typing import Optional, Dict, List
from decimal import Decimal

from nautilus_trader.model.data import Bar, TradeTick, QuoteTick, OrderBookDepth10
from nautilus_trader.model.events import OrderFilled, OrderSubmitted
from nautilus_trader.model.orders import LimitOrder, MarketOrder
from nautilus_trader.live.strategy import Strategy
from nautilus_trader.model.identifiers import InstrumentId, PositionId
from nautilus_trader.model.currencies import BTC, USDT

# Import our custom detectors
import sys
sys.path.append('/workspace/python')
from strategies.smc_detector import SMCDetector, SMCType, SMCSignal
from strategies.liquidity_engineering import LiquidityEngineering, LiquidityType


class SMCNautilusStrategy(Strategy):
    """
    Nautilus Trader strategy implementing Smart Money Concepts.
    
    Entry Logic:
    1. Wait for BOS (Break of Structure) confirmation
    2. Identify FVG or Order Block as entry zone
    3. Confirm with order flow imbalance from Rust engine
    4. Place limit order at FVG/OB edge
    5. Stop loss beyond recent swing low/high
    6. Take profit at next liquidity pool (EQH/EQL)
    """
    
    def __init__(self, config: dict):
        super().__init__(config=config)
        
        # Configuration
        self.instrument_id = InstrumentId.from_str(config.get('instrument', 'BTC/USDT-BINANCE'))
        self.risk_per_trade = Decimal(str(config.get('risk_per_trade', 0.01)))  # 1% risk
        self.min_confidence = config.get('min_confidence', 0.6)
        
        # Initialize detectors (pre-allocated, zero-GC design)
        self.smc_detector = SMCDetector(
            swing_window=config.get('swing_window', 5),
            fvg_threshold=config.get('fvg_threshold', 0.5),
            ob_lookback=config.get('ob_lookback', 10)
        )
        
        self.liquidity_engine = LiquidityEngineering(
            tolerance_pct=config.get('tolerance_pct', 0.5),
            min_touches=config.get('min_touches', 3)
        )
        
        # State tracking
        self.position_open = False
        self.entry_price: Optional[float] = None
        self.stop_loss: Optional[float] = None
        self.take_profit: Optional[float] = None
        self.pending_signals: List[SMCSignal] = []
        
        # Order flow state from Rust (updated via IPC)
        self.last_ofi = 0.0  # Order Flow Imbalance
        self.last_vpin = 0.0
        self.iceberg_detected = False
        
        # Performance tracking
        self.trades_taken = 0
        self.trades_won = 0
        self.total_pnl = 0.0
    
    def on_start(self):
        """Strategy startup - subscribe to data feeds"""
        self.log.info(f"Starting SMC Strategy for {self.instrument_id}")
        
        # Subscribe to bar data (for SMC patterns)
        self.subscribe_bars(self.instrument_id)
        
        # Subscribe to order book and trades (for order flow confirmation)
        self.subscribe_order_book_depth_10(self.instrument_id)
        self.subscribe_trade_ticks(self.instrument_id)
        
        # Load historical data if available (backtesting mode)
        # In live mode, this will be fed from Rust IPC bridge
    
    def on_stop(self):
        """Strategy shutdown"""
        self.log.info(f"Stopping SMC Strategy. Trades: {self.trades_taken}, Win Rate: {self.trades_won/max(1,self.trades_taken)*100:.1f}%")
    
    def on_bar(self, bar: Bar):
        """
        Main entry point - called on each new bar.
        Orchestrates SMC detection, liquidity analysis, and order execution.
        """
        timestamp_ns = bar.ts_event  # Nanosecond timestamp
        
        # Update detectors with new bar
        smc_signals = self.smc_detector.add_bar(
            timestamp_ns=timestamp_ns,
            open_p=float(bar.open),
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
            volume=float(bar.volume)
        )
        
        liq_result = self.liquidity_engine.add_bar(
            timestamp_ns=timestamp_ns,
            open_p=float(bar.open),
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close)
        )
        
        # Process any new SMC signals
        for signal in smc_signals:
            if signal.confidence >= self.min_confidence:
                self.pending_signals.append(signal)
                self.log.debug(f"SMC Signal: {signal.smc_type.value} at {signal.price} (conf: {signal.confidence:.2f})")
        
        # Check for entry opportunities if no position open
        if not self.position_open:
            self._check_entry_opportunity(bar, liq_result)
        else:
            # Manage existing position
            self._manage_position(bar)
    
    def _check_entry_opportunity(self, bar: Bar, liq_result: dict):
        """
        Evaluate pending signals for entry based on confluence factors.
        """
        if not self.pending_signals:
            return
        
        # Get latest signal
        signal = self.pending_signals[-1]
        
        # Confluence checklist
        confluence_score = 0.0
        
        # 1. SMC Pattern confidence
        confluence_score += signal.confidence * 0.4
        
        # 2. Zone alignment (Premium/Discount)
        zone_status = liq_result.get('zone_status', {})
        if signal.smc_type in [SMCType.FVG_BULLISH, SMCType.ORDER_BLOCK_BULL]:
            if zone_status.get('bias') == 'BULLISH':
                confluence_score += 0.3
        elif signal.smc_type in [SMCType.FVG_BEARISH, SMCType.ORDER_BLOCK_BEAR]:
            if zone_status.get('bias') == 'BEARISH':
                confluence_score += 0.3
        
        # 3. Order flow confirmation from Rust
        if signal.smc_type in [SMCType.FVG_BULLISH, SMCType.ORDER_BLOCK_BULL]:
            if self.last_ofi > 0.3:  # Positive order flow
                confluence_score += 0.3
            if self.iceberg_detected:
                confluence_score += 0.1
        elif signal.smc_type in [SMCType.FVG_BEARISH, SMCType.ORDER_BLOCK_BEAR]:
            if self.last_ofi < -0.3:  # Negative order flow
                confluence_score += 0.3
            if self.iceberg_detected:
                confluence_score += 0.1
        
        # 4. Stop hunt confirmation
        if liq_result.get('stop_hunts'):
            confluence_score += 0.2
        
        self.log.debug(f"Confluence score: {confluence_score:.2f} for {signal.smc_type.value}")
        
        # Execute if confluence is sufficient
        if confluence_score >= 0.7:
            self._execute_entry(signal, bar)
            self.pending_signals.clear()  # Clear processed signals
    
    def _execute_entry(self, signal: SMCSignal, bar: Bar):
        """
        Place limit order at optimal entry price.
        """
        current_price = float(bar.close)
        
        # Determine direction
        is_long = signal.smc_type in [SMCType.FVG_BULLISH, SMCType.ORDER_BLOCK_BULL, SMCType.BOS_BULLISH]
        
        # Calculate entry price (at FVG/OB zone edge)
        if signal.zone_start > 0 and signal.zone_end > 0:
            if is_long:
                entry_price = max(signal.zone_start, signal.zone_end) * 1.001  # Slight buffer
            else:
                entry_price = min(signal.zone_start, signal.zone_end) * 0.999
        else:
            entry_price = current_price
        
        # Calculate stop loss (beyond recent swing)
        if is_long:
            stop_loss = float(bar.low) * 0.998
            take_profit = entry_price + (entry_price - stop_loss) * 2.0  # 2:1 RR
        else:
            stop_loss = float(bar.high) * 1.002
            take_profit = entry_price - (stop_loss - entry_price) * 2.0
        
        # Calculate position size based on risk
        risk_amount = float(self.account.balance_usd()) * float(self.risk_per_trade)
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit > 0:
            quantity = risk_amount / risk_per_unit
        else:
            quantity = 0.01  # Minimum fallback
        
        # Round quantity to instrument precision
        quantity = round(quantity, 5)
        
        self.log.info(
            f"Executing {'LONG' if is_long else 'SHORT'} entry: "
            f"Entry={entry_price}, SL={stop_loss}, TP={take_profit}, Qty={quantity}"
        )
        
        # Create and submit limit order
        order = LimitOrder(
            instrument_id=self.instrument_id,
            side=1 if is_long else 2,  # 1=Buy, 2=Sell
            quantity=Decimal(str(quantity)),
            price=Decimal(str(round(entry_price, 2))),
            expire_time=None,
            post_only=True,  # Maker order for lower fees
            reduce_only=False,
            display_qty=None,
            emulation_trigger=None,
            trigger_type=None,
            trigger_price=None,
            limit_price=None,
            ts_init=bar.ts_event,
        )
        
        self.submit_order(order)
        
        # Update state
        self.position_open = True
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trades_taken += 1
    
    def _manage_position(self, bar: Bar):
        """
        Manage open position - check for stop loss, take profit, or early exit.
        """
        current_high = float(bar.high)
        current_low = float(bar.low)
        
        # Check stop loss
        if self.stop_loss:
            if self.entry_price > self.stop_loss:  # Long position
                if current_low <= self.stop_loss:
                    self._close_position(reason="STOP_LOSS")
                    return
            else:  # Short position
                if current_high >= self.stop_loss:
                    self._close_position(reason="STOP_LOSS")
                    return
        
        # Check take profit
        if self.take_profit:
            if self.entry_price < self.take_profit:  # Long
                if current_high >= self.take_profit:
                    self._close_position(reason="TAKE_PROFIT")
                    return
            else:  # Short
                if current_low <= self.take_profit:
                    self._close_position(reason="TAKE_PROFIT")
                    return
        
        # Check for invalidation (opposite BOS)
        # This would be implemented by checking new SMC signals
    
    def _close_position(self, reason: str):
        """Close all positions for the instrument"""
        self.log.info(f"Closing position: {reason}")
        self.position_open = False
        
        # Track performance
        if reason == "TAKE_PROFIT":
            self.trades_won += 1
        
        # In production, this would send a close order via Nautilus
        # For now, we reset state
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
    
    def on_order_book(self, order_book: OrderBookDepth10):
        """
        Receive order book updates from Rust IPC bridge.
        Used for real-time order flow calculation.
        """
        # Extract best bid/ask sizes for OFI calculation
        if order_book.bids and order_book.asks:
            bid_size = float(order_book.bids[0].size)
            ask_size = float(order_book.asks[0].size)
            
            # Simple imbalance metric (full OFI calculated in Rust)
            if bid_size + ask_size > 0:
                self.last_ofi = (bid_size - ask_size) / (bid_size + ask_size)
    
    def on_trade_tick(self, tick: TradeTick):
        """
        Receive trade ticks from Rust IPC bridge.
        Used for iceberg detection and VPIN calculation.
        """
        # In production, this data would feed into the Rust imbalance engine
        # and the result would be read back via shared memory
        pass
    
    def on_order_filled(self, event: OrderFilled):
        """Handle order fill events"""
        self.log.info(f"Order filled: {event}")
        # Update PnL tracking
        # In production, calculate realized PnL here
    
    # -------------------------------------------------------------------------
    # IPC Integration Methods (called from Rust bridge)
    # -------------------------------------------------------------------------
    
    def update_rust_microstructure(self, ofi: float, vpin: float, iceberg: bool):
        """
        Called by the IPC bridge when new microstructure data arrives from Rust.
        Updates internal state for entry confluence calculation.
        """
        self.last_ofi = ofi
        self.last_vpin = vpin
        self.iceberg_detected = iceberg
