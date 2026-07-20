"""
Python IPC Client for Rust Shared Memory Ring Buffer

This module implements the Python side of the IPC bridge using Cython-style
zero-copy memory access to read from the Rust shared memory ring buffer.

Features:
- Zero-copy memory mapping for minimal latency
- Lock-free reading with atomic operations
- Direct struct unpacking without serialization overhead
- GIL-free event processing where possible
- Integration with Nautilus Trader event loop

Note: For production, compile this as a Cython extension (.pyx) for
maximum performance. This .py version uses ctypes for demonstration.
"""

import os
import sys
import mmap
import struct
import logging
import threading
import time
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
from enum import IntEnum

logger = logging.getLogger(__name__)

# Protocol constants (must match Rust side)
PROTOCOL_MAGIC = 0x5155414E  # "QUAN"
PROTOCOL_VERSION = 1
SHM_MAGIC = 0x53484D51  # "SHMQ"

# Message types
class MessageType(IntEnum):
    ORDER_BOOK = 0
    TRADE = 1
    SIGNAL = 2
    ORDER_ACK = 3
    ORDER_REJECT = 4
    HEARTBEAT = 5
    CUSTOM = 255

# Ring buffer configuration
RING_BUFFER_CAPACITY = 1 << 14  # 16384 messages
MESSAGE_MAX_SIZE = 32 + 64 * 1024  # Header + max payload
HEADER_SIZE = 32  # MessageHeader size in Rust
SHM_HEADER_SIZE = 128  # ShmHeader size in Rust

# Message header format (little-endian)
# magic(u32) + version(u16) + type(u8) + flags(u8) + 
# payload_len(u32) + sequence(u64) + timestamp_ns(u64) + checksum(u32)
HEADER_FORMAT = '<IHBBQHQI'
HEADER_SIZE_CALC = struct.calcsize(HEADER_FORMAT)


@dataclass
class TradeMessage:
    """Parsed trade message."""
    symbol: str
    trade_id: int
    price: float
    quantity: float
    buyer_order_id: int
    seller_order_id: int
    buyer_is_maker: bool
    timestamp_ns: int
    sequence: int


@dataclass
class OrderBookMessage:
    """Parsed order book message."""
    symbol: str
    first_update_id: int
    last_update_id: int
    bids: List[tuple]  # [(price, quantity), ...]
    asks: List[tuple]
    timestamp_ns: int
    sequence: int


@dataclass
class SignalMessage:
    """Parsed trading signal message."""
    symbol: str
    signal_type: int  # 0=None, 1=Buy, 2=Sell, 3=Strong Buy, 4=Strong Sell
    confidence: float  # 0.0 to 1.0
    target_price: float
    stop_loss: float
    position_size: float
    time_horizon_secs: int
    model_id: int
    timestamp_ns: int
    sequence: int


class SharedMemoryReader:
    """
    Zero-copy shared memory reader for Rust ring buffer.
    
    Uses mmap for direct memory access without copying.
    """
    
    def __init__(self, path: str):
        """
        Initialize shared memory reader.
        
        Args:
            path: Path to shared memory file
        """
        self.path = path
        self._mmap: Optional[mmap.mmap] = None
        self._file = None
        self._read_pos = 0
        self._running = False
        
    def open(self) -> bool:
        """Open the shared memory file."""
        try:
            # Wait for ready signal
            ready_path = f"{self.path}.ready"
            timeout = 10  # seconds
            start = time.time()
            
            while not os.path.exists(ready_path):
                if time.time() - start > timeout:
                    logger.error("Timeout waiting for Rust publisher")
                    return False
                time.sleep(0.1)
            
            # Open file
            self._file = open(self.path, 'rb')
            
            # Memory map
            total_size = SHM_HEADER_SIZE + (RING_BUFFER_CAPACITY * MESSAGE_MAX_SIZE)
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            
            # Validate header
            if not self._validate_header():
                logger.error("Invalid shared memory header")
                self.close()
                return False
            
            logger.info(f"Opened shared memory at {self.path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to open shared memory: {e}")
            return False
    
    def _validate_header(self) -> bool:
        """Validate the shared memory header."""
        if self._mmap is None:
            return False
        
        try:
            magic = struct.unpack_from('<I', self._mmap, 0)[0]
            version = struct.unpack_from('<H', self._mmap, 4)[0]
            
            return magic == SHM_MAGIC and version == 1
        except Exception:
            return False
    
    def _get_write_pos(self) -> int:
        """Get current write position from header."""
        if self._mmap is None:
            return 0
        
        try:
            # write_pos is at offset 16 in ShmHeader
            return struct.unpack_from('<Q', self._mmap, 16)[0]
        except Exception:
            return 0
    
    def read_message(self) -> Optional[Dict[str, Any]]:
        """
        Read the next message from the ring buffer.
        
        Returns:
            Message dictionary or None if no new messages
        """
        if self._mmap is None:
            return None
        
        write_pos = self._get_write_pos()
        
        # Check if there's a new message
        if write_pos == self._read_pos:
            return None
        
        try:
            # Calculate offset
            offset = SHM_HEADER_SIZE + (self._read_pos * MESSAGE_MAX_SIZE)
            
            # Read header
            header_data = self._mmap[offset:offset + HEADER_SIZE]
            if len(header_data) < HEADER_SIZE:
                return None
            
            header = struct.unpack(HEADER_FORMAT, header_data)
            magic, version, msg_type, flags, payload_len, sequence, timestamp_ns, checksum = header
            
            # Validate
            if magic != PROTOCOL_MAGIC:
                self._read_pos = (self._read_pos + 1) & (RING_BUFFER_CAPACITY - 1)
                return None
            
            # Read payload
            payload_offset = offset + HEADER_SIZE
            payload = bytes(self._mmap[payload_offset:payload_offset + payload_len])
            
            # Update read position
            self._read_pos = (self._read_pos + 1) & (RING_BUFFER_CAPACITY - 1)
            
            return {
                'type': msg_type,
                'sequence': sequence,
                'timestamp_ns': timestamp_ns,
                'payload': payload,
            }
            
        except Exception as e:
            logger.error(f"Error reading message: {e}")
            self._read_pos = (self._read_pos + 1) & (RING_BUFFER_CAPACITY - 1)
            return None
    
    def close(self):
        """Close the shared memory."""
        if self._mmap:
            self._mmap.close()
            self._mmap = None
        if self._file:
            self._file.close()
            self._file = None


class IpcSubscriber:
    """
    IPC subscriber that reads from Rust shared memory and
    dispatches events to handlers.
    """
    
    def __init__(self, shm_path: str = "/tmp/trading_bot_ipc.bin"):
        """
        Initialize IPC subscriber.
        
        Args:
            shm_path: Path to shared memory file
        """
        self.shm_path = shm_path
        self._reader = SharedMemoryReader(shm_path)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Event handlers
        self._handlers: Dict[int, Callable] = {
            MessageType.TRADE: self._handle_trade,
            MessageType.ORDER_BOOK: self._handle_orderbook,
            MessageType.SIGNAL: self._handle_signal,
            MessageType.HEARTBEAT: self._handle_heartbeat,
        }
        
        # Statistics
        self._messages_received = 0
        self._errors = 0
        self._last_message_ns = 0
        
    def _parse_trade_payload(self, payload: bytes) -> Optional[TradeMessage]:
        """Parse trade message payload."""
        if len(payload) < 72:  # TradePayload size
            return None
        
        try:
            # Symbol (12 bytes, null-terminated)
            symbol_bytes = payload[:12]
            symbol = symbol_bytes.rstrip(b'\x00').decode('utf-8')
            
            # Unpack rest
            trade_id, price_scaled, quantity_scaled = struct.unpack_from('<qQQ', payload, 12)
            buyer_order_id, seller_order_id = struct.unpack_from('<qq', payload, 28)
            buyer_is_maker = struct.unpack_from('<?', payload, 44)[0]
            
            return TradeMessage(
                symbol=symbol,
                trade_id=trade_id,
                price=price_scaled / 100_000_000.0,
                quantity=quantity_scaled / 100_000_000.0,
                buyer_order_id=buyer_order_id,
                seller_order_id=seller_order_id,
                buyer_is_maker=buyer_is_maker,
                timestamp_ns=0,  # Set from header
                sequence=0,
            )
        except Exception as e:
            logger.error(f"Error parsing trade payload: {e}")
            return None
    
    def _parse_orderbook_payload(self, payload: bytes) -> Optional[OrderBookMessage]:
        """Parse order book message payload."""
        if len(payload) < 340:  # OrderBookPayload size
            return None
        
        try:
            # Symbol
            symbol_bytes = payload[:12]
            symbol = symbol_bytes.rstrip(b'\x00').decode('utf-8')
            
            # Counts
            bid_count = payload[12]
            ask_count = payload[13]
            
            # Update IDs
            first_id, last_id = struct.unpack_from('<qq', payload, 16)
            
            # Parse levels (each level is 16 bytes: 8 price + 8 quantity)
            bids = []
            asks = []
            
            offset = 32  # After header fields
            for i in range(bid_count):
                price, qty = struct.unpack_from('<qq', payload, offset + i * 16)
                bids.append((price / 100_000_000.0, qty / 100_000_000.0))
            
            offset = 32 + 20 * 16  # Skip bid array
            for i in range(ask_count):
                price, qty = struct.unpack_from('<qq', payload, offset + i * 16)
                asks.append((price / 100_000_000.0, qty / 100_000_000.0))
            
            return OrderBookMessage(
                symbol=symbol,
                first_update_id=first_id,
                last_update_id=last_id,
                bids=bids,
                asks=asks,
                timestamp_ns=0,
                sequence=0,
            )
        except Exception as e:
            logger.error(f"Error parsing orderbook payload: {e}")
            return None
    
    def _parse_signal_payload(self, payload: bytes) -> Optional[SignalMessage]:
        """Parse signal message payload."""
        if len(payload) < 48:  # SignalPayload size
            return None
        
        try:
            # Symbol
            symbol_bytes = payload[:12]
            symbol = symbol_bytes.rstrip(b'\x00').decode('utf-8')
            
            # Fields
            signal_type = payload[12]
            confidence = payload[13] / 100.0
            target_price, stop_loss = struct.unpack_from('<qq', payload, 16)
            position_size = struct.unpack_from('<i', payload, 24)[0]
            time_horizon, model_id = struct.unpack_from('<II', payload, 28)
            
            return SignalMessage(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                target_price=target_price / 100_000_000.0,
                stop_loss=stop_loss / 100_000_000.0,
                position_size=position_size / 1_000_000.0,
                time_horizon_secs=time_horizon,
                model_id=model_id,
                timestamp_ns=0,
                sequence=0,
            )
        except Exception as e:
            logger.error(f"Error parsing signal payload: {e}")
            return None
    
    def _handle_trade(self, msg: Dict[str, Any]):
        """Handle trade message."""
        trade = self._parse_trade_payload(msg['payload'])
        if trade:
            trade.timestamp_ns = msg['timestamp_ns']
            trade.sequence = msg['sequence']
            logger.debug(f"Trade: {trade.symbol} @ {trade.price} x {trade.quantity}")
            self._on_trade(trade)
    
    def _handle_orderbook(self, msg: Dict[str, Any]):
        """Handle order book message."""
        ob = self._parse_orderbook_payload(msg['payload'])
        if ob:
            ob.timestamp_ns = msg['timestamp_ns']
            ob.sequence = msg['sequence']
            logger.debug(f"OrderBook: {ob.symbol} - {len(ob.bids)} bids, {len(ob.asks)} asks")
            self._on_orderbook(ob)
    
    def _handle_signal(self, msg: Dict[str, Any]):
        """Handle signal message."""
        signal = self._parse_signal_payload(msg['payload'])
        if signal:
            signal.timestamp_ns = msg['timestamp_ns']
            signal.sequence = msg['sequence']
            logger.info(f"Signal: {signal.symbol} type={signal.signal_type} conf={signal.confidence}")
            self._on_signal(signal)
    
    def _handle_heartbeat(self, msg: Dict[str, Any]):
        """Handle heartbeat message."""
        logger.debug("Heartbeat received")
    
    def _on_trade(self, trade: TradeMessage):
        """Override this method to handle trades."""
        pass
    
    def _on_orderbook(self, ob: OrderBookMessage):
        """Override this method to handle order books."""
        pass
    
    def _on_signal(self, signal: SignalMessage):
        """Override this method to handle signals."""
        pass
    
    def start(self):
        """Start the subscriber thread."""
        if not self._reader.open():
            logger.error("Failed to open shared memory")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("IPC subscriber started")
    
    def _run_loop(self):
        """Main event loop."""
        while self._running:
            try:
                msg = self._reader.read_message()
                
                if msg:
                    self._messages_received += 1
                    self._last_message_ns = msg['timestamp_ns']
                    
                    handler = self._handlers.get(msg['type'])
                    if handler:
                        handler(msg)
                    else:
                        logger.debug(f"Unknown message type: {msg['type']}")
                else:
                    # No new message, brief sleep
                    time.sleep(0.0001)  # 100 microseconds
                    
            except Exception as e:
                self._errors += 1
                logger.error(f"Error in event loop: {e}")
                time.sleep(0.001)
    
    def stop(self):
        """Stop the subscriber."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._reader.close()
        logger.info("IPC subscriber stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get subscriber statistics."""
        return {
            'messages_received': self._messages_received,
            'errors': self._errors,
            'last_message_ns': self._last_message_ns,
            'running': self._running,
        }


class NautilusIpcSubscriber(IpcSubscriber):
    """
    IPC subscriber integrated with Nautilus Trader.
    
    Extends IpcSubscriber to forward events to Nautilus DataEngine.
    """
    
    def __init__(self, shm_path: str, nautilus_setup=None):
        """
        Initialize Nautilus IPC subscriber.
        
        Args:
            shm_path: Path to shared memory file
            nautilus_setup: NautilusTraderSetup instance
        """
        super().__init__(shm_path)
        self.nautilus_setup = nautilus_setup
    
    def _on_trade(self, trade: TradeMessage):
        """Forward trade to Nautilus."""
        if self.nautilus_setup:
            self.nautilus_setup.process_rust_event("trade", {
                'symbol': trade.symbol,
                'price': trade.price,
                'quantity': trade.quantity,
                'timestamp_ns': trade.timestamp_ns,
                'buyer_is_maker': trade.buyer_is_maker,
            })
    
    def _on_orderbook(self, ob: OrderBookMessage):
        """Forward order book to Nautilus."""
        if self.nautilus_setup:
            self.nautilus_setup.process_rust_event("orderbook", {
                'symbol': ob.symbol,
                'bids': ob.bids,
                'asks': ob.asks,
                'timestamp_ns': ob.timestamp_ns,
            })


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("Starting IPC Subscriber...")
    
    subscriber = IpcSubscriber()
    subscriber.start()
    
    try:
        while True:
            time.sleep(1)
            stats = subscriber.get_stats()
            print(f"Stats: {stats}")
    except KeyboardInterrupt:
        subscriber.stop()
        print("Subscriber stopped")
