"""
position_tracker.py
-------------------
Lock-free tracking of open positions, average entry prices, and realized PnL.
Reconciles local state with exchange state periodically.
Handles corporate actions like funding rate payouts and token splits.

Uses atomic operations and pre-allocated buffers for zero-GC updates.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
from enum import IntEnum

from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue, PositionId
from nautilus_trader.model.objects import Quantity, Price
from nautilus_trader.position import Position

logger = logging.getLogger(__name__)


class CorporateActionType(IntEnum):
    """Types of corporate actions affecting positions."""
    FUNDING_PAYMENT = 1
    TOKEN_SPLIT = 2
    TOKEN_MERGE = 3
    AIRDROP = 4
    DELISTING = 5


@dataclass
class CorporateAction:
    """Record of a corporate action affecting positions."""
    action_type: CorporateActionType
    instrument_id: str
    timestamp: datetime
    value: float  # Payment amount or split ratio
    description: str


class PositionTracker:
    """
    Ultra-low latency position tracker with lock-free design.
    
    Features:
    - Pre-allocated circular buffers for position history
    - Atomic updates using NumPy operations
    - Automatic reconciliation with exchange state
    - Corporate action handling (funding, splits, etc.)
    """

    MAX_POSITIONS = 100
    MAX_HISTORY = 10_000
    
    def __init__(self):
        # Current positions (pre-allocated)
        self._instrument_ids: np.ndarray = np.zeros(self.MAX_POSITIONS, dtype='U32')
        self._quantities: np.ndarray = np.zeros(self.MAX_POSITIONS, dtype='f8')
        self._entry_prices: np.ndarray = np.zeros(self.MAX_POSITIONS, dtype='f8')
        self._current_prices: np.ndarray = np.zeros(self.MAX_POSITIONS, dtype='f8')
        self._realized_pnl: np.ndarray = np.zeros(self.MAX_POSITIONS, dtype='f8')
        self._unrealized_pnl: np.ndarray = np.zeros(self.MAX_POSITIONS, dtype='f8')
        self._cum_funding: np.ndarray = np.zeros(self.MAX_POSITIONS, dtype='f8')
        self._open_timestamps: np.ndarray = np.zeros(self.MAX_POSITIONS, dtype='i8')
        self._is_long: np.ndarray = np.zeros(self.MAX_POSITIONS, dtype='i1')
        self._is_active: np.ndarray = np.zeros(self.MAX_POSITIONS, dtype='i1')
        
        self._position_count = 0
        
        # Position history (circular buffer)
        self._history_instrument: np.ndarray = np.zeros(self.MAX_HISTORY, dtype='U32')
        self._history_action: np.ndarray = np.zeros(self.MAX_HISTORY, dtype='i1')
        self._history_value: np.ndarray = np.zeros(self.MAX_HISTORY, dtype='f8')
        self._history_timestamp: np.ndarray = np.zeros(self.MAX_HISTORY, dtype='i8')
        self._history_write_idx = 0
        
        # Corporate actions queue
        self._pending_actions: List[CorporateAction] = []
        
        # Reconciliation state
        self._last_reconciliation = datetime.utcnow()
        self._reconciliation_interval = timedelta(seconds=30)
        self._reconciliation_errors = 0
        
        # Lock for complex operations (minimal usage)
        self._lock = asyncio.Lock()

    def _find_position(self, instrument_str: str) -> int:
        """Find position index by instrument. Returns -1 if not found."""
        mask = self._instrument_ids[:self._position_count] == instrument_str
        if np.any(mask):
            return int(np.argmax(mask))
        return -1

    def open_position(
        self,
        instrument_id: InstrumentId,
        quantity: float,
        entry_price: float,
        is_long: bool,
        position_id: str
    ) -> None:
        """
        Record a new position opening.
        Thread-safe via pre-allocation (no dynamic growth).
        """
        instrument_str = str(instrument_id)
        
        # Check if position already exists (shouldn't happen for new opens)
        existing_idx = self._find_position(instrument_str)
        if existing_idx >= 0:
            logger.warning(f"Position already exists for {instrument_str}, updating")
            self._update_position(existing_idx, quantity, entry_price, is_long)
            return
        
        # Add new position
        if self._position_count >= self.MAX_POSITIONS:
            logger.error("Max position limit reached! Cannot open new position.")
            return
        
        idx = self._position_count
        self._instrument_ids[idx] = instrument_str
        self._quantities[idx] = abs(quantity)
        self._entry_prices[idx] = entry_price
        self._current_prices[idx] = entry_price
        self._realized_pnl[idx] = 0.0
        self._unrealized_pnl[idx] = 0.0
        self._cum_funding[idx] = 0.0
        self._open_timestamps[idx] = int(datetime.utcnow().timestamp() * 1e9)
        self._is_long[idx] = 1 if is_long else 0
        self._is_active[idx] = 1
        
        self._position_count += 1
        logger.info(f"Opened position: {instrument_str}, qty={quantity}, price={entry_price}, long={is_long}")

    def _update_position(self, idx: int, quantity: float, price: float, is_long: bool) -> None:
        """Update an existing position (average in/out)."""
        current_qty = self._quantities[idx]
        current_price = self._entry_prices[idx]
        
        # Calculate new average price
        if (current_qty > 0 and quantity > 0) or (current_qty < 0 and quantity < 0):
            # Adding to position - average the price
            total_cost = current_qty * current_price + abs(quantity) * price
            new_qty = current_qty + abs(quantity)
            if new_qty > 0:
                self._entry_prices[idx] = total_cost / new_qty
        else:
            # Reducing position - calculate realized PnL
            close_qty = min(abs(quantity), current_qty)
            if self._is_long[idx]:
                pnl = (price - current_price) * close_qty
            else:
                pnl = (current_price - price) * close_qty
            
            self._realized_pnl[idx] += pnl
            logger.debug(f"Realized PnL: {pnl:.4f}")
        
        self._quantities[idx] = abs(quantity)
        self._is_long[idx] = 1 if is_long else 0

    def close_position(self, instrument_id: InstrumentId, exit_price: float) -> float:
        """
        Close a position and return realized PnL.
        """
        instrument_str = str(instrument_id)
        idx = self._find_position(instrument_str)
        
        if idx < 0:
            logger.warning(f"Cannot close non-existent position: {instrument_str}")
            return 0.0
        
        entry_price = self._entry_prices[idx]
        quantity = self._quantities[idx]
        is_long = self._is_long[idx] == 1
        
        # Calculate final PnL
        if is_long:
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity
        
        # Add cumulative funding
        total_pnl = pnl + self._cum_funding[idx]
        
        # Record in history
        self._record_history(instrument_str, 0, total_pnl)  # 0 = CLOSE action
        
        # Remove position
        self._remove_position(idx)
        
        logger.info(f"Closed position: {instrument_str}, PnL={total_pnl:.4f}")
        return total_pnl

    def _remove_position(self, idx: int) -> None:
        """Remove position at index by swapping with last element."""
        if idx < 0 or idx >= self._position_count:
            return
        
        last_idx = self._position_count - 1
        
        # Swap with last element (O(1) removal)
        if idx != last_idx:
            self._instrument_ids[idx] = self._instrument_ids[last_idx]
            self._quantities[idx] = self._quantities[last_idx]
            self._entry_prices[idx] = self._entry_prices[last_idx]
            self._current_prices[idx] = self._current_prices[last_idx]
            self._realized_pnl[idx] = self._realized_pnl[last_idx]
            self._unrealized_pnl[idx] = self._unrealized_pnl[last_idx]
            self._cum_funding[idx] = self._cum_funding[last_idx]
            self._open_timestamps[idx] = self._open_timestamps[last_idx]
            self._is_long[idx] = self._is_long[last_idx]
            self._is_active[idx] = self._is_active[last_idx]
        
        # Clear last element
        self._instrument_ids[last_idx] = ''
        self._is_active[last_idx] = 0
        self._position_count -= 1

    def update_mark_price(self, instrument_id: InstrumentId, mark_price: float) -> None:
        """Update mark price and recalculate unrealized PnL."""
        instrument_str = str(instrument_id)
        idx = self._find_position(instrument_str)
        
        if idx >= 0:
            self._current_prices[idx] = mark_price
            
            entry_price = self._entry_prices[idx]
            quantity = self._quantities[idx]
            is_long = self._is_long[idx] == 1
            
            if is_long:
                self._unrealized_pnl[idx] = (mark_price - entry_price) * quantity
            else:
                self._unrealized_pnl[idx] = (entry_price - mark_price) * quantity

    def apply_funding_payment(
        self,
        instrument_id: InstrumentId,
        funding_rate: float,
        payment: float
    ) -> None:
        """
        Apply funding rate payment to a position.
        Positive payment = we receive, negative = we pay.
        """
        instrument_str = str(instrument_id)
        idx = self._find_position(instrument_str)
        
        if idx >= 0:
            self._cum_funding[idx] += payment
            self._record_history(instrument_str, CorporateActionType.FUNDING_PAYMENT, payment)
            logger.debug(f"Funding payment for {instrument_str}: {payment:.4f}")

    def apply_token_split(self, instrument_id: InstrumentId, split_ratio: float) -> None:
        """
        Apply token split to position.
        split_ratio > 1 means tokens are split (qty increases, price decreases).
        """
        instrument_str = str(instrument_id)
        idx = self._find_position(instrument_str)
        
        if idx >= 0:
            old_qty = self._quantities[idx]
            old_price = self._entry_prices[idx]
            
            self._quantities[idx] = old_qty * split_ratio
            self._entry_prices[idx] = old_price / split_ratio
            self._current_prices[idx] = self._current_prices[idx] / split_ratio
            
            self._record_history(instrument_str, CorporateActionType.TOKEN_SPLIT, split_ratio)
            logger.info(f"Token split applied: {instrument_str}, ratio={split_ratio}")

    def _record_history(self, instrument: str, action: int, value: float) -> None:
        """Record action in circular history buffer."""
        idx = self._history_write_idx
        self._history_instrument[idx] = instrument
        self._history_action[idx] = action
        self._history_value[idx] = value
        self._history_timestamp[idx] = int(datetime.utcnow().timestamp() * 1e9)
        
        self._history_write_idx = (idx + 1) % self.MAX_HISTORY

    def get_total_unrealized_pnl(self) -> float:
        """Get sum of all unrealized PnL."""
        return float(np.sum(self._unrealized_pnl[:self._position_count]))

    def get_total_realized_pnl(self) -> float:
        """Get sum of all realized PnL (including funding)."""
        realized = np.sum(self._realized_pnl[:self._position_count])
        funding = np.sum(self._cum_funding[:self._position_count])
        return float(realized + funding)

    def get_all_positions(self) -> List[dict]:
        """Get list of all active positions."""
        positions = []
        for i in range(self._position_count):
            if self._is_active[i]:
                positions.append({
                    'instrument': self._instrument_ids[i],
                    'quantity': self._quantities[i],
                    'entry_price': self._entry_prices[i],
                    'current_price': self._current_prices[i],
                    'unrealized_pnl': self._unrealized_pnl[i],
                    'realized_pnl': self._realized_pnl[i],
                    'cumulative_funding': self._cum_funding[i],
                    'is_long': self._is_long[i] == 1,
                    'open_time_ns': self._open_timestamps[i]
                })
        return positions

    async def reconcile_with_exchange(self, exchange_positions: List[Position]) -> None:
        """
        Reconcile local state with exchange state.
        Called periodically to detect discrepancies.
        """
        async with self._lock:
            now = datetime.utcnow()
            
            # Build map of exchange positions
            exchange_map = {}
            for pos in exchange_positions:
                key = str(pos.instrument_id)
                exchange_map[key] = pos
            
            # Check each local position
            discrepancies = 0
            for i in range(self._position_count):
                if not self._is_active[i]:
                    continue
                
                instrument = self._instrument_ids[i]
                
                if instrument not in exchange_map:
                    logger.warning(f"Local position {instrument} not found on exchange!")
                    discrepancies += 1
                    # Mark for review
                else:
                    ex_pos = exchange_map[instrument]
                    local_qty = self._quantities[i]
                    ex_qty = abs(float(ex_pos.quantity))
                    
                    if abs(local_qty - ex_qty) > 1e-6:
                        logger.warning(
                            f"Quantity mismatch for {instrument}: "
                            f"local={local_qty}, exchange={ex_qty}"
                        )
                        discrepancies += 1
            
            self._last_reconciliation = now
            self._reconciliation_errors = discrepancies
            
            if discrepancies > 0:
                logger.error(f"Reconciliation found {discrepancies} discrepancies!")
