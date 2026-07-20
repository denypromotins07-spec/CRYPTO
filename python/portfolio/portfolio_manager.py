"""
portfolio_manager.py
--------------------
Real-time tracking of wallet balances, unrealized PnL, and margin utilization.
Interfaces with Nautilus Portfolio component but uses custom pre-allocated NumPy arrays
to prevent memory fragmentation during high-frequency updates.

STRICT MEMORY RULE: All arrays pre-allocated. No dynamic growth during trading.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
from dataclasses import dataclass

from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue, AccountId
from nautilus_trader.model.objects import Balance, Money, Quantity, Price
from nautilus_trader.portfolio.portfolio import Portfolio
from nautilus_trader.position import Position

logger = logging.getLogger(__name__)


@dataclass
class AssetBalance:
    """Pre-allocated balance record."""
    asset: str
    free: float
    locked: float
    total: float


class PortfolioManager:
    """
    High-performance portfolio state tracker.
    
    Features:
    - Pre-allocated NumPy arrays for all balance/PnL data
    - Vectorized PnL calculations
    - Real-time margin utilization monitoring
    - 8GB RAM cap enforcement via bounded buffers
    """

    # Maximum assets to track (prevents unbounded growth)
    MAX_ASSETS = 50
    MAX_POSITIONS = 100
    
    def __init__(self, account_id: AccountId):
        self._account_id = account_id
        
        # Pre-allocated balance arrays
        self._asset_names: np.ndarray = np.zeros(self.MAX_ASSETS, dtype='U20')
        self._free_balances: np.ndarray = np.zeros(self.MAX_ASSETS, dtype='f8')
        self._locked_balances: np.ndarray = np.zeros(self.MAX_ASSETS, dtype='f8')
        self._total_balances: np.ndarray = np.zeros(self.MAX_ASSETS, dtype='f8')
        self._asset_count = 0
        
        # Pre-allocated position tracking
        self._position_instruments: np.ndarray = np.zeros(self.MAX_POSITIONS, dtype='U20')
        self._position_sizes: np.ndarray = np.zeros(self.MAX_POSITIONS, dtype='f8')
        self._position_entry_prices: np.ndarray = np.zeros(self.MAX_POSITIONS, dtype='f8')
        self._position_unrealized_pnl: np.ndarray = np.zeros(self.MAX_POSITIONS, dtype='f8')
        self._position_realized_pnl: np.ndarray = np.zeros(self.MAX_POSITIONS, dtype='f8')
        self._position_count = 0
        
        # Margin tracking
        self._initial_margin = 0.0
        self._maintenance_margin = 0.0
        self._account_equity = 0.0
        self._available_balance = 0.0
        
        # Reference to Nautilus portfolio (for compatibility)
        self._nautilus_portfolio: Optional[Portfolio] = None
        
        # Lock for thread-safe updates
        self._lock = asyncio.Lock()
        
        # Last update timestamp
        self._last_update = datetime.utcnow()

    def set_nautilus_portfolio(self, portfolio: Portfolio) -> None:
        """Set reference to Nautilus portfolio for sync."""
        self._nautilus_portfolio = portfolio

    async def update_balances(self, balances: List[Balance]) -> None:
        """
        Update balances from exchange.
        Uses pre-allocated arrays to avoid GC.
        """
        async with self._lock:
            self._asset_count = 0
            
            for balance in balances:
                if self._asset_count >= self.MAX_ASSETS:
                    logger.warning("Max asset limit reached, ignoring additional balances")
                    break
                
                asset = balance.currency.code
                total = float(balance.total)
                free = float(balance.free)
                locked = float(balance.locked)
                
                # Store in pre-allocated arrays
                idx = self._asset_count
                self._asset_names[idx] = asset
                self._free_balances[idx] = free
                self._locked_balances[idx] = locked
                self._total_balances[idx] = total
                
                self._asset_count += 1
            
            self._last_update = datetime.utcnow()
            
            # Recalculate derived values
            self._recalculate_equity()

    def _recalculate_equity(self) -> None:
        """Recalculate account equity and available balance."""
        # Sum all asset balances (simplified - assumes USD quote)
        # In production, convert all to base currency using current prices
        self._account_equity = np.sum(self._total_balances[:self._asset_count])
        
        # Available = Total - Locked - Initial Margin
        total_locked = np.sum(self._locked_balances[:self._asset_count])
        self._available_balance = self._account_equity - total_locked - self._initial_margin

    async def update_position(self, position: Position) -> None:
        """Update or add a position in the tracker."""
        async with self._lock:
            instrument_str = str(position.instrument_id)
            
            # Check if position already exists
            mask = self._position_instruments[:self._position_count] == instrument_str
            
            if np.any(mask):
                # Update existing position
                idx = np.argmax(mask)
                self._position_sizes[idx] = float(position.quantity)
                self._position_entry_prices[idx] = float(position.avg_price) if position.avg_price else 0.0
                self._position_realized_pnl[idx] = float(position.realized_return)
            else:
                # Add new position
                if self._position_count >= self.MAX_POSITIONS:
                    logger.warning("Max position limit reached")
                    return
                
                idx = self._position_count
                self._position_instruments[idx] = instrument_str
                self._position_sizes[idx] = float(position.quantity)
                self._position_entry_prices[idx] = float(position.avg_price) if position.avg_price else 0.0
                self._position_realized_pnl[idx] = 0.0
                self._position_count += 1

    def update_unrealized_pnl(self, instrument_id: InstrumentId, mark_price: float) -> None:
        """
        Update unrealized PnL for a position given current mark price.
        Called frequently during trading.
        """
        instrument_str = str(instrument_id)
        mask = self._position_instruments[:self._position_count] == instrument_str
        
        if np.any(mask):
            idx = np.argmax(mask)
            size = self._position_sizes[idx]
            entry_price = self._position_entry_prices[idx]
            
            if entry_price > 0:
                # Calculate PnL: (mark_price - entry_price) * size
                pnl = (mark_price - entry_price) * size
                self._position_unrealized_pnl[idx] = pnl

    def get_total_unrealized_pnl(self) -> float:
        """Get sum of all unrealized PnL."""
        return float(np.sum(self._position_unrealized_pnl[:self._position_count]))

    def get_total_realized_pnl(self) -> float:
        """Get sum of all realized PnL."""
        return float(np.sum(self._position_realized_pnl[:self._position_count]))

    def get_total_pnl(self) -> float:
        """Get total PnL (realized + unrealized)."""
        return self.get_total_realized_pnl() + self.get_total_unrealized_pnl()

    def get_margin_utilization(self) -> float:
        """
        Calculate margin utilization ratio.
        Returns value between 0.0 and 1.0+ (over-utilized)
        """
        if self._account_equity <= 0:
            return 0.0
        
        total_margin_used = self._initial_margin + self._maintenance_margin
        return total_margin_used / self._account_equity

    def get_available_balance(self, asset: str = 'USDT') -> float:
        """Get available balance for a specific asset."""
        mask = self._asset_names[:self._asset_count] == asset
        if np.any(mask):
            idx = np.argmax(mask)
            return self._free_balances[idx]
        return 0.0

    def get_total_balance(self, asset: str = 'USDT') -> float:
        """Get total balance for a specific asset."""
        mask = self._asset_names[:self._asset_count] == asset
        if np.any(mask):
            idx = np.argmax(mask)
            return self._total_balances[idx]
        return 0.0

    def is_margin_safe(self, threshold: float = 0.8) -> bool:
        """
        Check if margin utilization is below safe threshold.
        Returns True if safe to trade, False if approaching limits.
        """
        utilization = self.get_margin_utilization()
        return utilization < threshold

    def get_risk_metrics(self) -> dict:
        """
        Get comprehensive risk metrics.
        Used by risk management systems.
        """
        return {
            'account_equity': self._account_equity,
            'available_balance': self._available_balance,
            'initial_margin': self._initial_margin,
            'maintenance_margin': self._maintenance_margin,
            'margin_utilization': self.get_margin_utilization(),
            'total_unrealized_pnl': self.get_total_unrealized_pnl(),
            'total_realized_pnl': self.get_total_realized_pnl(),
            'position_count': self._position_count,
            'is_margin_safe': self.is_margin_safe(),
            'last_update': self._last_update.isoformat()
        }

    def update_margin(self, initial: float, maintenance: float) -> None:
        """Update margin requirements."""
        self._initial_margin = initial
        self._maintenance_margin = maintenance
        self._recalculate_equity()

    def get_all_positions(self) -> List[dict]:
        """Get list of all tracked positions."""
        positions = []
        for i in range(self._position_count):
            positions.append({
                'instrument': self._position_instruments[i],
                'size': self._position_sizes[i],
                'entry_price': self._position_entry_prices[i],
                'unrealized_pnl': self._position_unrealized_pnl[i],
                'realized_pnl': self._position_realized_pnl[i]
            })
        return positions
