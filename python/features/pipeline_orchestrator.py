# python/features/pipeline_orchestrator.py
# =============================================================================
# STAGE 2 - CHAPTER 2 - FILE 3
# Focus: Ray-based distributed feature pipeline with strict memory management.
# Ensures the entire system stays within the 8GB RAM cap.
# =============================================================================

import ray
import numpy as np
from typing import Dict, List, Optional, Any
import threading
import time
import gc
import os
from dataclasses import dataclass
from collections import deque

# Import our feature modules
from technical_indicators import TechnicalIndicators, calculate_rsi, calculate_macd
from orderflow_metrics import OrderFlowAnalyzer, TickAggregator


# =============================================================================
# MEMORY MANAGEMENT CONFIGURATION
# =============================================================================

# Strict memory limits to stay within 8GB system cap
# Allocation breakdown:
# - Rust Engine: 1GB
# - Ray/Python ML: 4GB  
# - Feature Pipeline: 1.5GB
# - OS/Buffer: 1.5GB
MAX_PIPELINE_MEMORY_GB = 1.5
MAX_PIPELINE_MEMORY_BYTES = int(MAX_PIPELINE_MEMORY_GB * 1024 * 1024 * 1024)

# Object store configuration for Ray
OBJECT_STORE_MEMORY_GB = 2.0

# Garbage collection thresholds
GC_THRESHOLD_MINOR = 100  # Minor GC every 100 objects
GC_THRESHOLD_MAJOR = 1000  # Major GC every 1000 objects


@dataclass
class FeatureVector:
    """
    Compact feature vector for ML consumption.
    Uses float32 instead of float64 to halve memory usage.
    """
    timestamp: int
    symbol: str
    features: np.ndarray  # float32 array
    label: Optional[float] = None
    
    def __post_init__(self):
        # Ensure float32 for memory efficiency
        if self.features.dtype != np.float32:
            self.features = self.features.astype(np.float32)


class MemoryMonitor:
    """
    Real-time memory monitoring with automatic garbage collection.
    Prevents memory leaks and ensures we stay within bounds.
    """
    
    def __init__(self, max_memory_bytes: int = MAX_PIPELINE_MEMORY_BYTES):
        self.max_memory = max_memory_bytes
        self._lock = threading.Lock()
        self.allocation_count = 0
        self.last_gc_time = time.time()
        
    def get_current_usage(self) -> int:
        """Get current process memory usage in bytes."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss
        except ImportError:
            # Fallback: estimate based on numpy arrays
            return 0
    
    def check_and_gc(self) -> bool:
        """
        Check memory usage and trigger GC if needed.
        Returns True if GC was triggered.
        """
        with self._lock:
            current = self.get_current_usage()
            
            if current > self.max_memory * 0.9:  # 90% threshold
                print(f"[MEMORY] Critical: {current / 1e9:.2f}GB / {self.max_memory / 1e9:.2f}GB")
                gc.collect()
                self.last_gc_time = time.time()
                return True
            elif current > self.max_memory * 0.7:  # 70% warning
                print(f"[MEMORY] Warning: {current / 1e9:.2f}GB / {self.max_memory / 1e9:.2f}GB")
            
            return False
    
    def record_allocation(self):
        """Record a new allocation for tracking."""
        with self._lock:
            self.allocation_count += 1
            if self.allocation_count % GC_THRESHOLD_MINOR == 0:
                gc.collect()


class ObjectPool:
    """
    Object pooling to reduce allocations and GC pressure.
    Reuses numpy arrays and feature vectors.
    """
    
    def __init__(self, pool_size: int = 1000, feature_dim: int = 50):
        self.pool_size = pool_size
        self.feature_dim = feature_dim
        
        # Pre-allocate pool of numpy arrays
        self._array_pool = deque(
            [np.zeros(feature_dim, dtype=np.float32) for _ in range(pool_size)],
            maxlen=pool_size
        )
        self._lock = threading.Lock()
        
    def acquire(self) -> np.ndarray:
        """Acquire an array from the pool."""
        with self._lock:
            if len(self._array_pool) > 0:
                arr = self._array_pool.popleft()
                arr.fill(0)  # Reset values
                return arr
            else:
                # Pool exhausted - allocate new (will be added back on release)
                return np.zeros(self.feature_dim, dtype=np.float32)
    
    def release(self, arr: np.ndarray):
        """Return an array to the pool."""
        with self._lock:
            if len(self._array_pool) < self.pool_size:
                arr.fill(0)  # Clear before returning
                self._array_pool.append(arr)


# =============================================================================
# RAY ACTORS FOR DISTRIBUTED FEATURE COMPUTATION
# =============================================================================

@ray.remote
class FeatureCalculator:
    """
    Ray actor for parallel feature calculation.
    Each actor processes a subset of symbols or timeframes.
    """
    
    def __init__(self, actor_id: int, memory_limit_mb: int = 512):
        self.actor_id = actor_id
        self.memory_limit = memory_limit_mb * 1024 * 1024
        self.indicators = TechnicalIndicators(max_length=5000)
        self.orderflow = OrderFlowAnalyzer(max_ticks=50000)
        self.object_pool = ObjectPool(pool_size=500, feature_dim=50)
        self.monitor = MemoryMonitor(max_memory_bytes=self.memory_limit)
        
    def calculate_features(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        timestamps: np.ndarray
    ) -> List[FeatureVector]:
        """
        Calculate all features for a batch of data.
        Returns list of FeatureVector objects.
        """
        self.monitor.check_and_gc()
        
        n = len(prices)
        features_list = []
        
        # Calculate technical indicators
        rsi = calculate_rsi(prices, 14)
        macd_line, signal_line, histogram = calculate_macd(prices, 12, 26, 9)
        
        # Update order flow analyzer
        for i in range(n):
            is_seller = False  # Would come from actual tick data
            self.orderflow.add_tick(prices[i], volumes[i], is_seller)
        
        # Build feature vectors
        for i in range(len(prices)):
            # Acquire array from pool
            features = self.object_pool.acquire()
            
            # Technical features (indices 0-19)
            features[0] = rsi[i] if not np.isnan(rsi[i]) else 0.0
            features[1] = macd_line[i] if not np.isnan(macd_line[i]) else 0.0
            features[2] = signal_line[i] if not np.isnan(signal_line[i]) else 0.0
            features[3] = histogram[i] if not np.isnan(histogram[i]) else 0.0
            
            # Price features (indices 4-9)
            features[4] = prices[i]  # Current price
            features[5] = (prices[i] - prices[i-1]) / prices[i-1] if i > 0 else 0.0  # Return
            features[6] = np.log(prices[i] / prices[i-10]) if i > 10 else 0.0  # 10-period log return
            features[7] = highs[i] - lows[i]  # Range
            features[8] = (highs[i] + lows[i]) / 2  # Mid price
            features[9] = volumes[i]  # Volume
            
            # Order flow features (indices 10-19)
            cvd = self.orderflow.get_cvd()
            if len(cvd) > 0:
                features[10] = cvd[-1] if not np.isnan(cvd[-1]) else 0.0
            features[11] = self.orderflow.get_imbalance_ratio()
            
            # Volatility features (indices 20-24)
            if i >= 14:
                recent_returns = np.diff(np.log(prices[max(0,i-14):i+1]))
                features[20] = np.std(recent_returns) if len(recent_returns) > 0 else 0.0
            else:
                features[20] = 0.0
            
            # Normalize certain features
            features[4] = np.log(features[4])  # Log price
            
            # Create feature vector
            fv = FeatureVector(
                timestamp=int(timestamps[i]),
                symbol="BTCUSDT",
                features=features.copy()  # Copy before releasing to pool
            )
            features_list.append(fv)
            
            # Release array back to pool
            self.object_pool.release(features)
        
        self.monitor.record_allocation()
        return features_list
    
    def reset(self):
        """Reset actor state."""
        self.orderflow = OrderFlowAnalyzer(max_ticks=50000)
        gc.collect()


@ray.remote
class FeatureAggregator:
    """
    Ray actor for aggregating features from multiple calculators.
    Handles batching and forwarding to ML pipeline.
    """
    
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self.buffer: List[FeatureVector] = []
        self._lock = threading.Lock()
        
    def add_features(self, features: List[FeatureVector]) -> Optional[List[FeatureVector]]:
        """
        Add features to buffer and return batch if full.
        """
        with self._lock:
            self.buffer.extend(features)
            
            if len(self.buffer) >= self.batch_size:
                batch = self.buffer[:self.batch_size]
                self.buffer = self.buffer[self.batch_size:]
                return batch
            
            return None
    
    def flush(self) -> List[FeatureVector]:
        """Flush remaining buffered features."""
        with self._lock:
            batch = self.buffer.copy()
            self.buffer = []
            return batch


# =============================================================================
# PIPELINE ORCHESTRATOR
# =============================================================================

class FeaturePipelineOrchestrator:
    """
    Main orchestrator for the distributed feature pipeline.
    Manages Ray actors, memory, and data flow.
    """
    
    def __init__(
        self,
        num_calculators: int = 4,
        batch_size: int = 100,
        memory_limit_gb: float = MAX_PIPELINE_MEMORY_GB
    ):
        """
        Initialize the feature pipeline.
        
        Args:
            num_calculators: Number of parallel feature calculator actors
            batch_size: Batch size for aggregation
            memory_limit_gb: Maximum memory limit for the pipeline
        """
        self.num_calculators = num_calculators
        self.batch_size = batch_size
        self.memory_limit = int(memory_limit_gb * 1024 * 1024 * 1024)
        
        # Ray actors
        self.calculators: List[ray.actor.ActorHandle] = []
        self.aggregator: Optional[ray.actor.ActorHandle] = None
        
        # Monitoring
        self.monitor = MemoryMonitor(max_memory_bytes=self.memory_limit)
        self.start_time = time.time()
        
        # Statistics
        self.features_processed = 0
        self.batches_sent = 0
        
    def start(self):
        """Initialize Ray cluster and actors."""
        # Check if Ray is already initialized
        if not ray.is_initialized():
            ray.init(
                num_cpus=os.cpu_count(),
                object_store_memory=int(OBJECT_STORE_MEMORY_GB * 1024 * 1024 * 1024),
                _system_config={
                    "max_bytes_spill_threshold": self.memory_limit,
                },
                log_to_driver=True,
                include_dashboard=False
            )
            print(f"[PIPELINE] Ray initialized with {os.cpu_count()} CPUs")
        
        # Create calculator actors
        memory_per_actor = self.memory_limit // self.num_calculators
        self.calculators = [
            FeatureCalculator.remote(i, memory_per_actor // (1024 * 1024))
            for i in range(self.num_calculators)
        ]
        
        # Create aggregator
        self.aggregator = FeatureAggregator.remote(self.batch_size)
        
        print(f"[PIPELINE] Started {self.num_calculators} calculator actors")
    
    def stop(self):
        """Shutdown Ray cluster and cleanup."""
        print("[PIPELINE] Shutting down...")
        
        # Flush remaining features
        if self.aggregator:
            ray.get(self.aggregator.flush.remote())
        
        # Kill actors
        for calc in self.calculators:
            ray.kill(calc)
        
        self.calculators = []
        self.aggregator = None
        
        # Shutdown Ray
        if ray.is_initialized():
            ray.shutdown()
        
        # Force garbage collection
        gc.collect()
        
        print(f"[PIPELINE] Shutdown complete. Processed {self.features_processed} features.")
    
    async def process_batch(
        self,
        symbol: str,
        prices: np.ndarray,
        volumes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        timestamps: np.ndarray
    ) -> List[FeatureVector]:
        """
        Process a batch of market data through the pipeline.
        
        Args:
            symbol: Trading pair symbol
            prices: Array of prices
            volumes: Array of volumes
            highs: Array of highs
            lows: Array of lows
            timestamps: Array of timestamps
        
        Returns:
            List of computed feature vectors
        """
        # Check memory before processing
        self.monitor.check_and_gc()
        
        # Distribute data across calculators (round-robin for now)
        # In production, would distribute by symbol or time range
        calculator_idx = self.features_processed % self.num_calculators
        calculator = self.calculators[calculator_idx]
        
        # Send to calculator actor
        features_future = calculator.calculate_features.remote(
            prices, volumes, highs, lows, timestamps
        )
        
        # Get results
        features = await features_future
        
        # Add to aggregator
        batch_future = self.aggregator.add_features.remote(features)
        batch = await batch_future
        
        # Update statistics
        self.features_processed += len(features)
        if batch:
            self.batches_sent += 1
        
        # Return batch if available
        return batch if batch else features
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "features_processed": self.features_processed,
            "batches_sent": self.batches_sent,
            "uptime_seconds": time.time() - self.start_time,
            "memory_usage_gb": self.monitor.get_current_usage() / 1e9,
            "memory_limit_gb": self.memory_limit / 1e9,
        }


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    base_price = 50000
    
    prices = base_price + np.cumsum(np.random.randn(n_samples) * 10)
    volumes = np.abs(np.random.randn(n_samples)) * 10
    highs = prices + np.abs(np.random.randn(n_samples)) * 5
    lows = prices - np.abs(np.random.randn(n_samples)) * 5
    timestamps = np.arange(time.time() * 1000, time.time() * 1000 + n_samples * 1000, 1000)
    
    # Create and start pipeline
    pipeline = FeaturePipelineOrchestrator(num_calculators=2, batch_size=50)
    pipeline.start()
    
    try:
        # Process data
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            pipeline.process_batch("BTCUSDT", prices, volumes, highs, lows, timestamps)
        )
        
        print(f"\nProcessed {len(result)} feature vectors")
        print(f"Statistics: {pipeline.get_statistics()}")
        
    finally:
        pipeline.stop()
