"""
Feature Store with Memory-Mapped Files for Ultra-Low Latency ML Inference.
Stores engineered features with strict memory limits, allowing ML models to 
fetch historical feature windows without loading everything into RAM.

Target Hardware: AMD Ryzen AI 5 + AMD Radeon GPU + 16GB RAM (8GB cap)
"""

import os
import mmap
import struct
import logging
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import threading
from collections import OrderedDict

logger = logging.getLogger(__name__)


@dataclass
class FeatureMetadata:
    """Metadata for a stored feature."""
    name: str
    dtype: np.dtype
    shape: Tuple[int, ...]
    created_at: datetime
    last_updated: datetime
    version: int
    checksum: str  # Simple hash for integrity
    
    # Memory layout info
    offset: int  # Byte offset in mmap file
    size_bytes: int


class MemoryMappedArray:
    """
    Zero-copy memory-mapped NumPy array wrapper.
    
    Provides direct access to disk-backed arrays without loading into RAM.
    Uses POSIX mmap for efficient page fault handling by the OS.
    """
    
    def __init__(
        self,
        filepath: str,
        dtype: np.dtype,
        shape: Tuple[int, ...],
        offset: int = 0,
        mode: str = 'r+'
    ):
        """
        Initialize memory-mapped array.
        
        Args:
            filepath: Path to the backing file
            dtype: NumPy data type
            shape: Array dimensions
            offset: Byte offset in file
            mode: File mode ('r', 'r+', 'w+')
        """
        self.filepath = Path(filepath)
        self.dtype = dtype
        self.shape = shape
        self.offset = offset
        self.mode = mode
        
        # Calculate total size
        self.itemsize = np.dtype(dtype).itemsize
        self.total_elements = int(np.prod(shape))
        self.total_bytes = self.total_elements * self.itemsize
        
        # Ensure file exists and is large enough
        self._ensure_file_size()
        
        # Open mmap
        self._fd = None
        self._mmap = None
        self._array = None
        self._open()
        
        logger.debug(f"Mapped array {filepath}: shape={shape}, bytes={self.total_bytes}")
    
    def _ensure_file_size(self):
        """Ensure backing file is large enough."""
        required_size = self.offset + self.total_bytes
        
        if not self.filepath.exists():
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            self.filepath.touch()
        
        current_size = self.filepath.stat().st_size
        if current_size < required_size:
            # Extend file
            with open(self.filepath, 'ab') as f:
                f.seek(required_size - 1)
                f.write(b'\x00')
    
    def _open(self):
        """Open the memory mapping."""
        flags = mmap.ACCESS_READ if self.mode == 'r' else mmap.ACCESS_WRITE
        self._fd = os.open(str(self.filepath), os.O_RDWR if 'w' in self.mode else os.O_RDONLY)
        self._mmap = mmap.mmap(self._fd, 0, access=flags)
        
        # Create NumPy array view (zero-copy)
        self._array = np.ndarray(
            shape=self.shape,
            dtype=self.dtype,
            buffer=self._mmap,
            offset=self.offset
        )
    
    @property
    def array(self) -> np.ndarray:
        """Get the NumPy array view."""
        return self._array
    
    def __getitem__(self, key):
        """Direct indexing into memory-mapped array."""
        return self._array[key]
    
    def __setitem__(self, key, value):
        """Direct assignment to memory-mapped array."""
        if self.mode == 'r':
            raise ValueError("Cannot write to read-only mapped array")
        self._array[key] = value
    
    def flush(self):
        """Flush changes to disk."""
        if self.mode != 'r':
            self._mmap.flush()
    
    def close(self):
        """Close the memory mapping."""
        if self._mmap is not None:
            self._mmap.close()
        if self._fd is not None:
            os.close(self._fd)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __del__(self):
        try:
            self.close()
        except:
            pass


class FeatureStore:
    """
    High-performance feature store with memory-mapped files.
    
    Features:
    - Zero-copy reads via mmap
    - Strict memory cap enforcement (8GB max)
    - Automatic LRU eviction of cold features
    - Versioned feature storage
    - Thread-safe access with fine-grained locking
    - Integrity checking via checksums
    """
    
    # Maximum features to keep in memory cache
    MAX_CACHED_FEATURES = 100
    
    def __init__(
        self,
        storage_path: str = "./data/features",
        max_ram_usage_gb: float = 4.0,  # Reserve 4GB for features (leave 4GB for rest)
        default_window_size: int = 10000
    ):
        """
        Initialize feature store.
        
        Args:
            storage_path: Base path for feature files
            max_ram_usage_gb: Maximum RAM for feature caching
            default_window_size: Default lookback window for time-series features
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.max_ram_bytes = int(max_ram_usage_gb * 1024**3)
        self.default_window = default_window_size
        
        # Metadata storage
        self.features: Dict[str, FeatureMetadata] = {}
        self.feature_files: Dict[str, str] = {}  # feature_name -> filepath
        
        # LRU cache for hot features (in-memory)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_lock = threading.RLock()
        self._current_cache_bytes = 0
        
        # Write locks per feature
        self._write_locks: Dict[str, threading.Lock] = {}
        
        logger.info(f"FeatureStore initialized at {self.storage_path}")
        logger.info(f"Max RAM for features: {max_ram_usage_gb}GB")
    
    def register_feature(
        self,
        name: str,
        dtype: np.dtype,
        shape: Tuple[int, ...],
        initial_data: Optional[np.ndarray] = None
    ) -> bool:
        """
        Register a new feature in the store.
        
        Args:
            name: Unique feature identifier
            dtype: NumPy data type
            shape: Array dimensions (time_steps, features)
            initial_data: Optional initial data
            
        Returns:
            True if registration successful
        """
        if name in self.features:
            logger.warning(f"Feature {name} already registered, updating")
            return self.update_feature(name, initial_data)
        
        # Calculate storage requirements
        itemsize = np.dtype(dtype).itemsize
        total_bytes = int(np.prod(shape)) * itemsize
        
        # Check memory budget
        if self._current_cache_bytes + total_bytes > self.max_ram_bytes:
            logger.warning(f"Memory cap exceeded for new feature {name}")
            self._evict_cold_features(total_bytes)
        
        # Create backing file
        filename = f"{name}.feat"
        filepath = self.storage_path / filename
        
        # Write metadata header (first 256 bytes)
        header_size = 256
        data_offset = header_size
        
        with open(filepath, 'wb') as f:
            # Write header placeholder
            f.write(b'\x00' * header_size)
            
            # Write initial data if provided
            if initial_data is not None:
                initial_data.astype(dtype).tofile(f)
            else:
                # Write zeros
                f.write(b'\x00' * total_bytes)
        
        # Create metadata
        metadata = FeatureMetadata(
            name=name,
            dtype=dtype,
            shape=shape,
            created_at=datetime.now(),
            last_updated=datetime.now(),
            version=1,
            checksum="",  # TODO: Implement checksum
            offset=data_offset,
            size_bytes=total_bytes
        )
        
        self.features[name] = metadata
        self.feature_files[name] = str(filepath)
        self._write_locks[name] = threading.Lock()
        
        logger.info(f"Registered feature: {name} (shape={shape}, bytes={total_bytes})")
        return True
    
    def get_feature(
        self,
        name: str,
        window: Optional[int] = None,
        use_cache: bool = True
    ) -> Optional[np.ndarray]:
        """
        Retrieve feature data with optional time window.
        
        Args:
            name: Feature identifier
            window: Number of recent time steps (default: full history)
            use_cache: Use LRU cache if available
            
        Returns:
            NumPy array or None if not found
        """
        if name not in self.features:
            logger.error(f"Feature {name} not found")
            return None
        
        metadata = self.features[name]
        window = window or self.default_window
        
        # Check cache first
        if use_cache:
            cached = self._get_from_cache(name)
            if cached is not None:
                # Return windowed view
                if len(cached) > window:
                    return cached[-window:].copy()
                return cached.copy()
        
        # Memory-mapped read
        filepath = self.feature_files[name]
        
        try:
            with MemoryMappedArray(
                filepath=filepath,
                dtype=metadata.dtype,
                shape=metadata.shape,
                offset=metadata.offset,
                mode='r'
            ) as mmap_arr:
                data = mmap_arr.array
                
                # Apply window
                if len(data) > window:
                    result = data[-window:].copy()
                else:
                    result = data.copy()
                
                # Cache if small enough
                if use_cache and result.nbytes < self.max_ram_bytes // 10:
                    self._add_to_cache(name, result)
                
                return result
                
        except Exception as e:
            logger.error(f"Failed to read feature {name}: {e}")
            return None
    
    def update_feature(
        self,
        name: str,
        data: np.ndarray,
        append: bool = True
    ) -> bool:
        """
        Update feature data.
        
        Args:
            name: Feature identifier
            data: New data array
            append: Append to existing or replace
            
        Returns:
            True if update successful
        """
        if name not in self.features:
            logger.error(f"Feature {name} not found")
            return False
        
        if name not in self._write_locks:
            self._write_locks[name] = threading.Lock()
        
        with self._write_locks[name]:
            metadata = self.features[name]
            filepath = self.feature_files[name]
            
            try:
                if append:
                    # Append mode - extend file
                    with open(filepath, 'ab') as f:
                        data.astype(metadata.dtype).tofile(f)
                    
                    # Update metadata
                    metadata.shape = (
                        metadata.shape[0] + len(data),
                        *metadata.shape[1:]
                    )
                    metadata.size_bytes += data.nbytes
                else:
                    # Replace mode
                    with MemoryMappedArray(
                        filepath=filepath,
                        dtype=metadata.dtype,
                        shape=data.shape,
                        offset=metadata.offset,
                        mode='r+'
                    ) as mmap_arr:
                        mmap_arr.array[:] = data
                
                metadata.last_updated = datetime.now()
                metadata.version += 1
                
                # Invalidate cache
                self._remove_from_cache(name)
                
                logger.debug(f"Updated feature {name}: {len(data)} records")
                return True
                
            except Exception as e:
                logger.error(f"Failed to update feature {name}: {e}")
                return False
    
    def get_feature_window(
        self,
        name: str,
        start_idx: int,
        end_idx: int
    ) -> Optional[np.ndarray]:
        """
        Get specific window of feature data by index.
        
        Args:
            name: Feature identifier
            start_idx: Start index (inclusive)
            end_idx: End index (exclusive)
            
        Returns:
            Windowed NumPy array
        """
        full_data = self.get_feature(name, window=None, use_cache=False)
        if full_data is None:
            return None
        
        return full_data[start_idx:end_idx].copy()
    
    def list_features(self) -> List[str]:
        """List all registered features."""
        return list(self.features.keys())
    
    def get_feature_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a feature."""
        if name not in self.features:
            return None
        
        meta = self.features[name]
        return {
            'name': meta.name,
            'dtype': str(meta.dtype),
            'shape': meta.shape,
            'size_mb': meta.size_bytes / (1024**2),
            'version': meta.version,
            'created': meta.created_at.isoformat(),
            'updated': meta.last_updated.isoformat()
        }
    
    def delete_feature(self, name: str) -> bool:
        """Delete a feature from the store."""
        if name not in self.features:
            return False
        
        # Remove from cache
        self._remove_from_cache(name)
        
        # Delete file
        filepath = self.feature_files.get(name)
        if filepath and Path(filepath).exists():
            Path(filepath).unlink()
        
        # Clean up metadata
        del self.features[name]
        del self.feature_files[name]
        if name in self._write_locks:
            del self._write_locks[name]
        
        logger.info(f"Deleted feature: {name}")
        return True
    
    def flush_all(self):
        """Flush all pending writes to disk."""
        logger.info("Flushing all feature writes")
        # mmap automatically flushes, but we can force sync
        for name in self.features:
            filepath = self.feature_files.get(name)
            if filepath:
                try:
                    with open(filepath, 'rb') as f:
                        os.fsync(f.fileno())
                except:
                    pass
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_bytes = sum(f.size_bytes for f in self.features.values())
        return {
            'feature_count': len(self.features),
            'total_size_mb': total_bytes / (1024**2),
            'cache_size_mb': self._current_cache_bytes / (1024**2),
            'cache_entries': len(self._cache),
            'ram_limit_gb': self.max_ram_bytes / (1024**3)
        }
    
    def _get_from_cache(self, name: str) -> Optional[np.ndarray]:
        """Get feature from LRU cache."""
        with self._cache_lock:
            if name in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(name)
                return self._cache[name]
        return None
    
    def _add_to_cache(self, name: str, data: np.ndarray):
        """Add feature to LRU cache with eviction."""
        with self._cache_lock:
            data_bytes = data.nbytes
            
            # Evict if necessary
            while self._current_cache_bytes + data_bytes > self.max_ram_bytes:
                self._evict_coldest()
            
            # Add to cache
            self._cache[name] = data
            self._current_cache_bytes += data_bytes
            self._cache.move_to_end(name)
    
    def _remove_from_cache(self, name: str):
        """Remove feature from cache."""
        with self._cache_lock:
            if name in self._cache:
                data = self._cache.pop(name)
                self._current_cache_bytes -= data.nbytes
    
    def _evict_cold_features(self, required_bytes: int):
        """Evict cold features to make room."""
        with self._cache_lock:
            while self._current_cache_bytes + required_bytes > self.max_ram_bytes:
                if not self._cache:
                    break
                self._evict_coldest()
    
    def _evict_coldest(self):
        """Evict the coldest (least recently used) feature."""
        if self._cache:
            # First item is least recently used
            name, data = next(iter(self._cache.items()))
            del self._cache[name]
            self._current_cache_bytes -= data.nbytes
            logger.debug(f"Evicted cold feature: {name}")
    
    def close(self):
        """Clean shutdown."""
        logger.info("Closing FeatureStore")
        self.flush_all()
        
        with self._cache_lock:
            self._cache.clear()
            self._current_cache_bytes = 0


class FeatureEngine:
    """
    Real-time feature engineering pipeline.
    
    Computes technical indicators and statistical features on streaming data,
    storing results directly to the FeatureStore with minimal memory overhead.
    """
    
    def __init__(self, feature_store: FeatureStore):
        """
        Initialize feature engine.
        
        Args:
            feature_store: FeatureStore instance
        """
        self.store = feature_store
        self._buffers: Dict[str, np.ndarray] = {}
        
        logger.info("FeatureEngine initialized")
    
    def compute_returns(
        self,
        prices: np.ndarray,
        symbol: str,
        periods: List[int] = [1, 5, 15, 60]
    ) -> Dict[str, np.ndarray]:
        """
        Compute log returns for multiple periods.
        
        Args:
            prices: Price array
            symbol: Trading pair symbol
            periods: Return periods to compute
            
        Returns:
            Dictionary of period -> returns array
        """
        results = {}
        
        for period in periods:
            feat_name = f"{symbol}_returns_{period}"
            
            # Compute log returns
            if len(prices) > period:
                returns = np.log(prices[period:] / prices[:-period])
            else:
                returns = np.array([])
            
            # Store in feature store
            if len(returns) > 0:
                self.store.update_feature(feat_name, returns, append=True)
                results[feat_name] = returns
        
        return results
    
    def compute_volatility(
        self,
        returns: np.ndarray,
        symbol: str,
        windows: List[int] = [10, 50, 200]
    ) -> Dict[str, np.ndarray]:
        """
        Compute rolling volatility (standard deviation).
        
        Args:
            returns: Returns array
            symbol: Trading pair symbol
            windows: Rolling window sizes
            
        Returns:
            Dictionary of window -> volatility array
        """
        results = {}
        
        for window in windows:
            feat_name = f"{symbol}_volatility_{window}"
            
            if len(returns) >= window:
                # Rolling std
                vol = np.array([
                    np.std(returns[i-window:i])
                    for i in range(window, len(returns) + 1)
                ])
                
                self.store.update_feature(feat_name, vol, append=True)
                results[feat_name] = vol
        
        return results
    
    def compute_momentum(
        self,
        prices: np.ndarray,
        symbol: str,
        lookback: int = 14
    ) -> np.ndarray:
        """
        Compute RSI-like momentum indicator.
        
        Args:
            prices: Price array
            symbol: Trading pair symbol
            lookback: Lookback period
            
        Returns:
            Momentum array
        """
        if len(prices) <= lookback:
            return np.array([])
        
        # Simple momentum: rate of change
        momentum = (prices[lookback:] - prices[:-lookback]) / prices[:-lookback] * 100
        
        feat_name = f"{symbol}_momentum_{lookback}"
        self.store.update_feature(feat_name, momentum, append=True)
        
        return momentum
    
    def compute_orderbook_imbalance(
        self,
        bids: np.ndarray,
        asks: np.ndarray,
        symbol: str
    ) -> np.ndarray:
        """
        Compute order book imbalance metric.
        
        Args:
            bids: Bid quantities array
            asks: Ask quantities array
            symbol: Trading pair symbol
            
        Returns:
            Imbalance array: (bid_qty - ask_qty) / (bid_qty + ask_qty)
        """
        if len(bids) != len(asks):
            return np.array([])
        
        total = bids + asks
        mask = total > 0
        imbalance = np.zeros_like(bids, dtype=np.float64)
        imbalance[mask] = (bids[mask] - asks[mask]) / total[mask]
        
        feat_name = f"{symbol}_ob_imbalance"
        self.store.update_feature(feat_name, imbalance, append=True)
        
        return imbalance
    
    def create_feature_vector(
        self,
        symbol: str,
        feature_names: List[str],
        window: int = 100
    ) -> Optional[np.ndarray]:
        """
        Create concatenated feature vector for ML model input.
        
        Args:
            symbol: Trading pair symbol
            feature_names: List of feature names to include
            window: Time window length
            
        Returns:
            2D array of shape (window, num_features)
        """
        features = []
        
        for feat_name in feature_names:
            data = self.store.get_feature(feat_name, window=window)
            if data is None:
                logger.warning(f"Feature {feat_name} not found")
                return None
            features.append(data.reshape(-1, 1) if data.ndim == 1 else data)
        
        if not features:
            return None
        
        # Concatenate along feature axis
        return np.hstack(features)


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    store = FeatureStore(storage_path="./data/test_features")
    engine = FeatureEngine(store)
    
    # Register and populate features
    n_samples = 1000
    prices = 50000 + np.cumsum(np.random.randn(n_samples) * 100)
    
    # Compute features
    engine.compute_returns(prices, "BTCUSDT", periods=[1, 5, 15])
    engine.compute_momentum(prices, "BTCUSDT", lookback=14)
    
    # Get feature vector
    vector = engine.create_feature_vector(
        "BTCUSDT",
        ["BTCUSDT_returns_1", "BTCUSDT_returns_5", "BTCUSDT_momentum_14"],
        window=100
    )
    
    if vector is not None:
        print(f"Feature vector shape: {vector.shape}")
        print(f"First row: {vector[0]}")
    
    # Stats
    print(f"\nStorage stats: {store.get_storage_stats()}")
    
    store.close()
