"""
shared_memory.py
----------------
Zero-copy shared memory ring buffer for Rust-Python IPC.
Uses POSIX shared memory (mmap) for ultra-low latency data transfer.
"""

import mmap
import struct
import logging
from typing import Optional, Any
import numpy as np

logger = logging.getLogger(__name__)


class SharedMemoryRingBuffer:
    """
    Lock-free ring buffer using POSIX shared memory.
    
    Memory Layout:
    - Header (64 bytes): write_index (8), read_index (8), version (8), flags (8), reserved (32)
    - Data Region (remaining): Fixed-size slots for tick data
    
    Each slot (128 bytes):
    - symbol_id (4 bytes)
    - flags (4 bytes) - bit 0: quote, bit 1: trade, bit 2: bar
    - bid_price (8 bytes)
    - ask_price (8 bytes)
    - bid_size (8 bytes)
    - ask_size (8 bytes)
    - price (8 bytes)
    - size (8 bytes)
    - aggressor_side (1 byte)
    - padding (3 bytes)
    - ts_event (8 bytes)
    - ts_recv (8 bytes)
    - reserved (56 bytes)
    """
    
    HEADER_SIZE = 64
    SLOT_SIZE = 128
    DTYPE = np.dtype([
        ('symbol_id', 'i4'),
        ('flags', 'i4'),
        ('bid_price', 'f8'),
        ('ask_price', 'f8'),
        ('bid_size', 'f8'),
        ('ask_size', 'f8'),
        ('price', 'f8'),
        ('size', 'f8'),
        ('aggressor_side', 'i1'),
        ('padding', 'V3'),
        ('ts_event', 'i8'),
        ('ts_recv', 'i8'),
        ('reserved', 'V56')
    ])
    
    def __init__(self, name: str, size_mb: int = 512):
        self._name = f"/nautilus_{name}"
        self._size_bytes = size_mb * 1024 * 1024
        self._num_slots = (self._size_bytes - self.HEADER_SIZE) // self.SLOT_SIZE
        
        self._fd: Optional[int] = None
        self._mm: Optional[mmap.mmap] = None
        self._header_view: Optional[np.ndarray] = None
        self._data_view: Optional[np.ndarray] = None
        
        self._connect()
    
    def _connect(self):
        """Create or open shared memory segment."""
        try:
            # Try to open existing shared memory
            self._fd = open(f"/dev/shm{self._name}", "r+b")
            logger.info(f"Attached to existing shared memory: {self._name}")
        except FileNotFoundError:
            # Create new shared memory
            self._fd = open(f"/dev/shm{self._name}", "wb+")
            self._fd.write(b'\x00' * self._size_bytes)
            logger.info(f"Created new shared memory: {self._name} ({self._size_bytes} bytes)")
        
        # Memory map the file
        self._mm = mmap.mmap(self._fd.fileno(), self._size_bytes)
        
        # Create numpy views (zero-copy)
        header_data = np.frombuffer(self._mm[:self.HEADER_SIZE], dtype=np.uint64)
        self._header_view = header_data[:8]  # First 8 uint64s = 64 bytes
        
        data_buffer = self._mm[self.HEADER_SIZE:]
        self._data_view = np.frombuffer(data_buffer, dtype=self.DTYPE, count=self._num_slots)
    
    def close(self):
        """Cleanup resources."""
        if self._mm:
            self._mm.close()
        if self._fd:
            self._fd.close()
    
    @property
    def _write_index(self) -> int:
        return int(self._header_view[0])
    
    @_write_index.setter
    def _write_index(self, value: int):
        self._header_view[0] = value
    
    @property
    def _read_index(self) -> int:
        return int(self._header_view[1])
    
    @_read_index.setter
    def _read_index(self, value: int):
        self._header_view[1] = value
    
    def read_batch(self, max_items: int = 1000) -> Optional[np.ndarray]:
        """
        Read a batch of items from the ring buffer.
        Returns None if no new data available.
        """
        write_idx = self._write_index
        read_idx = self._read_index
        
        if write_idx == read_idx:
            return None  # No new data
        
        # Calculate available items (handle wrap-around)
        if write_idx > read_idx:
            available = write_idx - read_idx
            start = read_idx
        else:
            available = (self._num_slots - read_idx) + write_idx
            start = read_idx
        
        # Limit batch size
        count = min(available, max_items)
        
        if count == 0:
            return None
        
        # Extract data (handle wrap-around)
        if start + count <= self._num_slots:
            # No wrap-around
            batch = self._data_view[start:start + count].copy()
        else:
            # Wrap-around: need to copy two segments
            first_part = self._data_view[start:].copy()
            second_part = self._data_view[:count - len(first_part)].copy()
            batch = np.concatenate([first_part, second_part])
        
        # Update read index
        new_read_idx = (read_idx + count) % self._num_slots
        self._read_index = new_read_idx
        
        return batch
    
    def send_subscription_request(self, symbol: str):
        """
        Send a subscription request to the Rust producer.
        Uses a special control message format.
        """
        # Encode symbol as bytes and write to a control slot
        # This is a simplified implementation
        logger.debug(f"Subscription request for {symbol}")
        # In production, this would use a separate control channel
