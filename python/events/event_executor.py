"""
Event Executor Module - Event-Driven Trading Execution

Event-driven execution logic that instantly fires market/limit orders based on
sentiment thresholds. Includes "straddle" logic to trade high-volatility news
breakouts while protecting against fake-outs.

Hardware Target: AMD Ryzen AI 5 with async I/O
Memory Constraint: Bounded order queues, pre-allocated structures
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Set
from enum import Enum
from collections import deque
import time
import logging

# Import from sibling modules
try:
    from .news_ingest import NewsItem
    from .nlp_sentiment import SentimentResult
except ImportError:
    # Fallback for direct execution
    class NewsItem:
        pass
    
    class SentimentResult:
        pass


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class TradeSignal:
    """Generated trading signal from news event"""
    signal_id: str
    news_item_id: str
    ticker: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 0.0
    urgency: float = 0.0
    created_at_ns: int = field(default_factory=lambda: time.time_ns())
    expiry_ns: int = 0  # Signal expires after this time
    executed: bool = False
    execution_price: Optional[float] = None


@dataclass
class StraddleConfig:
    """Configuration for straddle strategy around news events"""
    enabled: bool = True
    buy_threshold: float = 0.3  # Sentiment threshold for long
    sell_threshold: float = -0.3  # Sentiment threshold for short
    position_size_pct: float = 0.1  # Position size as % of portfolio
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.05  # 5% take profit
    cooldown_ms: int = 5000  # Cooldown between signals per ticker
    max_signals_per_minute: int = 10
    fakeout_protection: bool = True  # Wait for confirmation
    confirmation_window_ms: int = 500  # Time to wait for confirmation


class FakeoutDetector:
    """
    Detects potential fake-outs by analyzing initial price movement
    after news release vs sustained direction.
    """
    
    def __init__(self, window_ms: int = 1000):
        self.window_ms = window_ms
        self._price_history: Dict[str, deque] = {}
        self._max_history = 100
    
    def record_price(self, ticker: str, price: float, timestamp_ns: int):
        """Record price for analysis"""
        if ticker not in self._price_history:
            self._price_history[ticker] = deque(maxlen=self._max_history)
        
        self._price_history[ticker].append((timestamp_ns, price))
    
    def is_fakeout(self, ticker: str, initial_direction: int) -> bool:
        """
        Detect if price movement is a fake-out.
        initial_direction: 1 for up, -1 for down
        Returns True if movement appears to be reversing (fake-out)
        """
        if ticker not in self._price_history:
            return False
        
        history = self._price_history[ticker]
        if len(history) < 5:
            return False
        
        now = time.time_ns()
        recent_prices = [p for t, p in list(history)[-10:]]
        
        if len(recent_prices) < 5:
            return False
        
        # Calculate initial move
        initial_move = recent_prices[0] - recent_prices[len(recent_prices)//2]
        
        # Calculate recent move
        recent_move = recent_prices[-1] - recent_prices[len(recent_prices)//2]
        
        # If directions oppose, likely a fake-out
        if initial_direction > 0 and recent_move < 0 and abs(recent_move) > abs(initial_move) * 0.5:
            return True
        if initial_direction < 0 and recent_move > 0 and abs(recent_move) > abs(initial_move) * 0.5:
            return True
        
        return False


class EventExecutor:
    """
    Main event-driven execution engine.
    Processes sentiment signals and executes trades accordingly.
    """
    
    def __init__(
        self,
        config: Optional[StraddleConfig] = None,
        max_pending_signals: int = 100,
    ):
        self.config = config or StraddleConfig()
        self.max_pending_signals = max_pending_signals
        
        # Pending signals queue
        self.pending_signals: deque[TradeSignal] = deque(maxlen=max_pending_signals)
        
        # Executed signals tracking
        self.executed_signals: Dict[str, TradeSignal] = {}
        
        # Cooldown tracking per ticker
        self._cooldowns: Dict[str, int] = {}  # ticker -> expiry_ns
        
        # Rate limiting
        self._signals_this_minute: Dict[int, int] = {}  # minute -> count
        
        # Fakeout detector
        self.fakeout_detector = FakeoutDetector()
        
        # Callbacks for order execution
        self._order_callbacks: List[Callable[[TradeSignal], Any]] = []
        
        # Statistics
        self.signals_generated = 0
        self.signals_executed = 0
        self.signals_rejected = 0
        self.fakeouts_avoided = 0
        
        # Running state
        self._running = False
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def register_order_callback(self, callback: Callable[[TradeSignal], Any]):
        """Register callback for order execution"""
        self._order_callbacks.append(callback)
    
    async def start(self):
        """Start the executor"""
        self._running = True
        self.logger.info("[EventExecutor] Started")
    
    async def stop(self):
        """Stop the executor"""
        self._running = False
        self.logger.info(f"[EventExecutor] Stopped. Executed: {self.signals_executed}")
    
    def process_news_item(self, news: NewsItem, sentiment: SentimentResult) -> Optional[TradeSignal]:
        """
        Process a news item with sentiment and generate trade signals.
        Returns TradeSignal if action should be taken, None otherwise.
        """
        if not self._running:
            return None
        
        # Check cooldowns
        for ticker in news.tickers:
            if ticker in self._cooldowns:
                if time.time_ns() < self._cooldowns[ticker]:
                    continue
        
        # Check rate limits
        current_minute = time.time_ns() // 60_000_000_000
        if self._signals_this_minute.get(current_minute, 0) >= self.config.max_signals_per_minute:
            self.signals_rejected += 1
            return None
        
        # Generate signals based on sentiment
        signals = []
        
        for ticker in news.tickers:
            # Determine direction based on sentiment
            if sentiment.compound >= self.config.buy_threshold:
                side = OrderSide.BUY
            elif sentiment.compound <= self.config.sell_threshold:
                side = OrderSide.SELL
            else:
                continue  # No signal
            
            # Check fakeout protection
            if self.config.fakeout_protection:
                direction = 1 if side == OrderSide.BUY else -1
                if self.fakeout_detector.is_fakeout(ticker, direction):
                    self.fakeouts_avoided += 1
                    self.logger.debug(f"[EventExecutor] Fakeout detected for {ticker}")
                    continue
            
            # Create signal
            signal = TradeSignal(
                signal_id=f"sig_{time.time_ns()}_{ticker}",
                news_item_id=news.id,
                ticker=ticker,
                side=side,
                order_type=OrderType.MARKET,
                quantity=0,  # To be calculated by position manager
                confidence=sentiment.confidence,
                urgency=sentiment.urgency,
                expiry_ns=time.time_ns() + 5_000_000_000,  # 5 second expiry
            )
            
            # Add stop loss and take profit
            if self.config.stop_loss_pct > 0:
                signal.stop_loss = 0  # Price to be filled by executor
            if self.config.take_profit_pct > 0:
                signal.take_profit = 0
            
            signals.append(signal)
        
        if not signals:
            return None
        
        # Return highest confidence signal
        best_signal = max(signals, key=lambda s: s.confidence * s.urgency)
        
        # Record for rate limiting
        self._signals_this_minute[current_minute] = \
            self._signals_this_minute.get(current_minute, 0) + 1
        
        # Set cooldown
        self._cooldowns[best_signal.ticker] = \
            time.time_ns() + self.config.cooldown_ms * 1_000_000
        
        self.signals_generated += 1
        self.pending_signals.append(best_signal)
        
        return best_signal
    
    async def execute_signal(self, signal: TradeSignal) -> bool:
        """Execute a trade signal"""
        if signal.executed:
            return False
        
        # Check expiry
        if time.time_ns() > signal.expiry_ns:
            self.logger.debug(f"[EventExecutor] Signal expired: {signal.signal_id}")
            return False
        
        try:
            # Call registered callbacks to actually place order
            for callback in self._order_callbacks:
                if asyncio.iscoroutinefunction(callback):
                    await callback(signal)
                else:
                    callback(signal)
            
            # Mark as executed
            signal.executed = True
            self.executed_signals[signal.signal_id] = signal
            self.signals_executed += 1
            
            self.logger.info(
                f"[EventExecutor] Executed: {signal.side.value} {signal.ticker} "
                f"(confidence: {signal.confidence:.2f})"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"[EventExecutor] Execution error: {e}")
            return False
    
    def update_price(self, ticker: str, price: float):
        """Update price for fakeout detection"""
        self.fakeout_detector.record_price(ticker, price, time.time_ns())
    
    def get_pending_signals(self, limit: int = 10) -> List[TradeSignal]:
        """Get pending signals ready for execution"""
        now = time.time_ns()
        pending = []
        
        for signal in self.pending_signals:
            if not signal.executed and now < signal.expiry_ns:
                pending.append(signal)
                if len(pending) >= limit:
                    break
        
        return pending
    
    def cleanup_expired(self):
        """Remove expired signals from queue"""
        now = time.time_ns()
        
        # Clean pending signals
        valid_signals = deque(maxlen=self.max_pending_signals)
        for signal in self.pending_signals:
            if now < signal.expiry_ns and not signal.executed:
                valid_signals.append(signal)
        self.pending_signals = valid_signals
        
        # Clean cooldowns
        expired_cooldowns = [
            ticker for ticker, expiry in self._cooldowns.items()
            if now > expiry
        ]
        for ticker in expired_cooldowns:
            del self._cooldowns[ticker]
        
        # Clean rate limit tracking (keep last 5 minutes)
        current_minute = now // 60_000_000_000
        old_minutes = [m for m in self._signals_this_minute if current_minute - m > 5]
        for minute in old_minutes:
            del self._signals_this_minute[minute]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics"""
        return {
            'signals_generated': self.signals_generated,
            'signals_executed': self.signals_executed,
            'signals_rejected': self.signals_rejected,
            'fakeouts_avoided': self.fakeouts_avoided,
            'pending_signals': len(self.pending_signals),
            'active_cooldowns': len(self._cooldowns),
            'execution_rate': (
                self.signals_executed / max(1, self.signals_generated)
            ),
        }


# Example usage
if __name__ == '__main__':
    async def main():
        executor = EventExecutor()
        
        # Mock order callback
        async def place_order(signal: TradeSignal):
            print(f"[ORDER] {signal.side.value} {signal.ticker} @ MARKET")
        
        executor.register_order_callback(place_order)
        
        await executor.start()
        
        # Simulate processing
        print("EventExecutor ready for signals")
        
        # Get stats
        stats = executor.get_stats()
        print(f"Stats: {stats}")
        
        await executor.stop()
    
    asyncio.run(main())
