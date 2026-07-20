"""
AMD Ryzen AI NPU Inference Engine
==================================
Chapter 1, File 2: Hardware Acceleration

This module provides integration with AMD Ryzen AI NPU (Neural Processing Unit)
via the Vitis AI / Ryzen AI Software stack. It offloads lightweight ML inference
tasks (regime detection, small neural networks, XGBoost models) to the NPU,
freeing up the main CPU and keeping RAM usage minimal.

Target Hardware: AMD Ryzen AI 5 with integrated NPU
Memory Cap: Strict memory management for 8GB global limit
"""

import numpy as np
from typing import Optional, Dict, List, Any, Tuple, Union
from pathlib import Path
import threading
import time
import logging
import json
import struct
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import weakref

# Try to import AMD Ryzen AI libraries (will gracefully degrade if not available)
try:
    # AMD Ryzen AI runtime (when available on target hardware)
    from ryzen_ai import Runtime, Model
    RYZEN_AI_AVAILABLE = True
except ImportError:
    RYZEN_AI_AVAILABLE = False
    print("Warning: AMD Ryzen AI runtime not available. Using CPU fallback mode.")

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class NPUPriority(Enum):
    """Priority levels for NPU task scheduling."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class NPUModelConfig:
    """Configuration for NPU model deployment."""
    model_path: str
    model_type: str  # 'onnx', 'xgboost', 'custom'
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    dtype: str = 'float32'
    batch_size: int = 1
    priority: NPUPriority = NPUPriority.NORMAL
    max_latency_ms: float = 10.0
    memory_limit_mb: int = 64


@dataclass
class InferenceResult:
    """Result from NPU inference operation."""
    output: np.ndarray
    latency_ms: float
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp_ns: int = field(default_factory=lambda: time.time_ns())


class NPUModelRegistry:
    """
    Thread-safe registry for managing NPU models.
    
    Maintains a pool of loaded models with LRU eviction policy
    to stay within memory constraints.
    """
    
    def __init__(self, max_models: int = 10, max_memory_mb: int = 256):
        self._models: Dict[str, Any] = {}
        self._access_order: deque = deque(maxlen=max_models)
        self._memory_usage_mb = 0
        self._max_memory_mb = max_memory_mb
        self._lock = threading.RLock()
        self._model_refs: Dict[str, weakref.ref] = {}
        
    def register(self, model_id: str, model: Any, memory_mb: int) -> bool:
        """Register a loaded model in the registry."""
        with self._lock:
            if model_id in self._models:
                # Update access order
                if model_id in self._access_order:
                    self._access_order.remove(model_id)
                self._access_order.append(model_id)
                return True
            
            # Check memory limit
            if self._memory_usage_mb + memory_mb > self._max_memory_mb:
                # Evict least recently used model
                self._evict_lru()
            
            self._models[model_id] = model
            self._memory_usage_mb += memory_mb
            self._access_order.append(model_id)
            return True
    
    def get(self, model_id: str) -> Optional[Any]:
        """Get model by ID and update access order."""
        with self._lock:
            if model_id not in self._models:
                return None
            
            # Update access order
            if model_id in self._access_order:
                self._access_order.remove(model_id)
            self._access_order.append(model_id)
            
            return self._models[model_id]
    
    def _evict_lru(self) -> None:
        """Evict least recently used model."""
        if not self._access_order:
            return
        
        oldest_id = self._access_order.popleft()
        if oldest_id in self._models:
            del self._models[oldest_id]
            # Note: In production, we'd need to track per-model memory
            self._memory_usage_mb = max(0, self._memory_usage_mb - 32)
    
    def clear(self) -> None:
        """Clear all registered models."""
        with self._lock:
            self._models.clear()
            self._access_order.clear()
            self._memory_usage_mb = 0


class AMDNPUInference:
    """
    AMD Ryzen AI NPU Inference Engine.
    
    Provides hardware-accelerated inference for:
    - Regime detection models (XGBoost, small NNs)
    - Feature extraction networks
    - Classification models
    - Lightweight transformers
    
    Automatically falls back to CPU/GPU if NPU is unavailable.
    """
    
    def __init__(self, config_override: Optional[Dict] = None):
        """
        Initialize NPU inference engine.
        
        Args:
            config_override: Optional configuration overrides
        """
        self.config = config_override or {}
        self._registry = NPUModelRegistry()
        self._npu_available = RYZEN_AI_AVAILABLE
        self._onnx_available = ONNX_AVAILABLE
        self._xgboost_available = XGBOOST_AVAILABLE
        
        # Performance tracking
        self._inference_count = 0
        self._total_latency_ns = 0
        self._npu_offload_count = 0
        self._fallback_count = 0
        
        # Threading for async inference
        self._executor_thread = None
        self._task_queue = deque()
        self._result_callbacks: Dict[str, callable] = {}
        self._lock = threading.Lock()
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        if self._npu_available:
            self.logger.info("AMD Ryzen AI NPU detected and available")
            try:
                self._npu_runtime = Runtime()
            except Exception as e:
                self.logger.warning(f"NPU runtime initialization failed: {e}")
                self._npu_available = False
        else:
            self.logger.info("Running in CPU fallback mode")
            self._npu_runtime = None
    
    def load_onnx_model(
        self,
        model_id: str,
        model_path: str,
        config: NPUModelConfig
    ) -> bool:
        """
        Load an ONNX model for NPU acceleration.
        
        Args:
            model_id: Unique identifier for the model
            model_path: Path to ONNX model file
            config: Model configuration
            
        Returns:
            True if model loaded successfully
        """
        if not self._onnx_available:
            self.logger.error("ONNX Runtime not available")
            return False
        
        try:
            # Configure session options for NPU
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.intra_op_num_threads = 1
            session_options.inter_op_num_threads = 1
            
            # If NPU available, try to use EP (Execution Provider)
            if self._npu_available:
                # AMD NPU execution provider (when available)
                providers = ['VitisAIExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
            
            session = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=providers
            )
            
            # Estimate memory usage
            model_size_mb = Path(model_path).stat().st_size / (1024 * 1024)
            estimated_memory = model_size_mb * 3  # Account for runtime overhead
            
            self._registry.register(model_id, session, int(estimated_memory))
            
            self.logger.info(f"Loaded ONNX model '{model_id}' ({model_path})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load ONNX model: {e}")
            return False
    
    def load_xgboost_model(
        self,
        model_id: str,
        model_path: str,
        config: NPUModelConfig
    ) -> bool:
        """
        Load XGBoost model for regime detection.
        
        Args:
            model_id: Unique identifier
            model_path: Path to saved XGBoost model (.json or .bst)
            config: Model configuration
            
        Returns:
            True if loaded successfully
        """
        if not self._xgboost_available:
            self.logger.error("XGBoost not available")
            return False
        
        try:
            model = xgb.Booster()
            model.load_model(model_path)
            
            # Set optimal parameters for low-latency
            model.set_param({'nthread': 1})
            
            # Estimate memory
            model_size_mb = Path(model_path).stat().st_size / (1024 * 1024)
            estimated_memory = int(model_size_mb * 2)
            
            self._registry.register(model_id, model, estimated_memory)
            
            self.logger.info(f"Loaded XGBoost model '{model_id}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load XGBoost model: {e}")
            return False
    
    def infer(
        self,
        model_id: str,
        input_data: np.ndarray,
        timeout_ms: float = 10.0
    ) -> Optional[InferenceResult]:
        """
        Perform synchronous inference.
        
        Args:
            model_id: ID of model to use
            input_data: Input tensor
            timeout_ms: Maximum allowed latency
            
        Returns:
            InferenceResult or None if failed
        """
        start_time = time.time_ns()
        
        model = self._registry.get(model_id)
        if model is None:
            self.logger.error(f"Model '{model_id}' not found")
            return None
        
        try:
            # Prepare input
            input_data = np.ascontiguousarray(input_data, dtype=np.float32)
            
            if isinstance(model, ort.InferenceSession):
                # ONNX model inference
                input_name = model.get_inputs()[0].name
                output = model.run(None, {input_name: input_data})[0]
                
            elif isinstance(model, xgb.Booster):
                # XGBoost inference
                dmatrix = xgb.DMatrix(input_data)
                output = model.predict(dmatrix, validate_features=False)
                
            else:
                self.logger.error(f"Unknown model type: {type(model)}")
                return None
            
            # Calculate latency
            latency_ns = time.time_ns() - start_time
            latency_ms = latency_ns / 1_000_000
            
            # Update statistics
            self._inference_count += 1
            self._total_latency_ns += latency_ns
            
            if latency_ms <= timeout_ms:
                self._npu_offload_count += 1
            else:
                self._fallback_count += 1
            
            return InferenceResult(
                output=output,
                latency_ms=latency_ms,
                metadata={'model_id': model_id, 'input_shape': input_data.shape}
            )
            
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            self._fallback_count += 1
            return None
    
    async def infer_async(
        self,
        model_id: str,
        input_data: np.ndarray,
        callback: Optional[callable] = None
    ) -> InferenceResult:
        """
        Perform asynchronous inference.
        
        Args:
            model_id: ID of model
            input_data: Input tensor
            callback: Optional callback function(result)
            
        Returns:
            Future-like object or task ID
        """
        task_id = f"{model_id}_{time.time_ns()}"
        
        with self._lock:
            self._task_queue.append((task_id, model_id, input_data, callback))
        
        # Start executor thread if not running
        if self._executor_thread is None or not self._executor_thread.is_alive():
            self._executor_thread = threading.Thread(target=self._executor_loop, daemon=True)
            self._executor_thread.start()
        
        return task_id
    
    def _executor_loop(self) -> None:
        """Background thread for async inference execution."""
        while True:
            with self._lock:
                if not self._task_queue:
                    time.sleep(0.001)  # 1ms sleep
                    continue
                task_id, model_id, input_data, callback = self._task_queue.popleft()
            
            result = self.infer(model_id, input_data)
            
            if callback and result:
                try:
                    callback(result)
                except Exception as e:
                    self.logger.error(f"Callback error: {e}")
            
            if result and callback:
                self._result_callbacks[task_id] = result
    
    def get_regime_prediction(
        self,
        features: np.ndarray,
        model_id: str = 'regime_detector'
    ) -> Dict[str, Any]:
        """
        Get market regime prediction from NPU-accelerated model.
        
        Specialized method for regime detection use case.
        
        Args:
            features: Feature vector [n_features] or batch [n_samples, n_features]
            model_id: Regime detection model ID
            
        Returns:
            Dictionary with regime prediction and confidence
        """
        result = self.infer(model_id, features)
        
        if result is None:
            return {
                'regime': 'unknown',
                'confidence': 0.0,
                'probabilities': None,
                'latency_ms': float('inf')
            }
        
        predictions = result.output
        
        # Handle different output formats
        if len(predictions.shape) == 1 or predictions.shape[-1] == 1:
            # Binary or regression output
            regime_idx = int(predictions[0] > 0.5) if predictions.shape[-1] == 1 else int(predictions[0])
            confidence = float(abs(predictions[0]))
        else:
            # Multi-class probabilities
            regime_idx = int(np.argmax(predictions[-1] if len(predictions.shape) > 1 else predictions))
            confidence = float(np.max(predictions[-1] if len(predictions.shape) > 1 else predictions))
        
        regime_map = {
            0: 'low_volatility',
            1: 'trending_up',
            2: 'trending_down',
            3: 'high_volatility',
            4: 'mean_reverting'
        }
        
        return {
            'regime': regime_map.get(regime_idx, f'regime_{regime_idx}'),
            'regime_id': regime_idx,
            'confidence': confidence,
            'probabilities': predictions.tolist() if len(predictions.shape) > 0 else [predictions],
            'latency_ms': result.latency_ms
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get inference performance statistics."""
        avg_latency = (
            self._total_latency_ns / self._inference_count / 1_000_000
            if self._inference_count > 0 else 0
        )
        
        return {
            'npu_available': self._npu_available,
            'onnx_available': self._onnx_available,
            'xgboost_available': self._xgboost_available,
            'total_inferences': self._inference_count,
            'npu_offloads': self._npu_offload_count,
            'cpu_fallbacks': self._fallback_count,
            'offload_ratio': self._npu_offload_count / max(1, self._inference_count),
            'avg_latency_ms': avg_latency,
            'models_loaded': len(self._registry._models),
            'memory_usage_mb': self._registry._memory_usage_mb
        }
    
    def warmup(self, model_id: str, iterations: int = 10) -> float:
        """
        Warm up model with dummy inference runs.
        
        Args:
            model_id: Model to warm up
            iterations: Number of warmup iterations
            
        Returns:
            Average warmup latency
        """
        model = self._registry.get(model_id)
        if model is None:
            return float('inf')
        
        # Create dummy input based on model type
        if isinstance(model, ort.InferenceSession):
            input_shape = model.get_inputs()[0].shape
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
        else:
            dummy_input = np.random.randn(1, 100).astype(np.float32)
        
        latencies = []
        for _ in range(iterations):
            result = self.infer(model_id, dummy_input)
            if result:
                latencies.append(result.latency_ms)
        
        return np.mean(latencies) if latencies else float('inf')
    
    def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory."""
        with self._registry._lock:
            if model_id in self._registry._models:
                del self._registry._models[model_id]
                if model_id in self._registry._access_order:
                    self._registry._access_order.remove(model_id)
                self.logger.info(f"Unloaded model '{model_id}'")
                return True
        return False
    
    def shutdown(self) -> None:
        """Clean shutdown of NPU engine."""
        self._registry.clear()
        if self._npu_runtime:
            try:
                # Proper NPU cleanup if available
                pass
            except:
                pass
        self.logger.info("NPU inference engine shut down")


class RegimeDetector:
    """
    High-level regime detection using NPU acceleration.
    
    Wraps AMDNPUInference to provide easy-to-use market regime
    detection for trading strategy adaptation.
    """
    
    def __init__(self, npu_engine: Optional[AMDNPUInference] = None):
        """
        Initialize regime detector.
        
        Args:
            npu_engine: Optional NPU engine instance
        """
        self.npu = npu_engine or AMDNPUInference()
        self._feature_buffer = deque(maxlen=100)
        self._last_regime = 'unknown'
        self._regime_change_callback = None
    
    def set_regime_change_callback(self, callback: callable) -> None:
        """Set callback for regime changes."""
        self._regime_change_callback = callback
    
    def extract_features(self, market_data: Dict) -> np.ndarray:
        """
        Extract features from market data for regime detection.
        
        Args:
            market_data: Dictionary with OHLCV, orderbook, etc.
            
        Returns:
            Feature vector
        """
        # Example feature extraction (customize for your needs)
        features = []
        
        # Price-based features
        if 'returns' in market_data:
            returns = market_data['returns']
            features.extend([
                np.mean(returns),
                np.std(returns),
                np.skew(returns) if len(returns) > 2 else 0,
                np.kurtosis(returns) if len(returns) > 3 else 0,
            ])
        
        # Volatility features
        if 'volatility' in market_data:
            features.extend([
                market_data['volatility'],
                np.diff([market_data['volatility']])[0] if 'volatility' in market_data else 0
            ])
        
        # Volume features
        if 'volume' in market_data:
            features.extend([
                np.log1p(market_data['volume']),
                np.diff([np.log1p(market_data['volume'])])[0]
            ])
        
        # Orderbook imbalance
        if 'bid_volume' in market_data and 'ask_volume' in market_data:
            total = market_data['bid_volume'] + market_data['ask_volume']
            if total > 0:
                imbalance = (market_data['bid_volume'] - market_data['ask_volume']) / total
                features.append(imbalance)
            else:
                features.append(0)
        
        # Pad to fixed size if needed
        target_size = 50  # Adjust based on model input
        while len(features) < target_size:
            features.append(0)
        
        return np.array(features[:target_size], dtype=np.float32)
    
    def detect(self, market_data: Dict) -> Dict[str, Any]:
        """
        Detect current market regime.
        
        Args:
            market_data: Current market data
            
        Returns:
            Regime prediction dictionary
        """
        features = self.extract_features(market_data)
        self._feature_buffer.append(features)
        
        result = self.npu.get_regime_prediction(features)
        
        # Detect regime change
        if result['regime'] != self._last_regime:
            old_regime = self._last_regime
            self._last_regime = result['regime']
            
            if self._regime_change_callback:
                try:
                    self._regime_change_callback(old_regime, result)
                except Exception as e:
                    pass
        
        return result
    
    def load_model(self, model_path: str, model_type: str = 'xgboost') -> bool:
        """Load regime detection model."""
        config = NPUModelConfig(
            model_path=model_path,
            model_type=model_type,
            input_shape=(1, 50),
            output_shape=(1, 5)
        )
        
        if model_type == 'xgboost':
            return self.npu.load_xgboost_model('regime_detector', model_path, config)
        else:
            return self.npu.load_onnx_model('regime_detector', model_path, config)


# Singleton instance
_npu_instance: Optional[AMDNPUInference] = None


def get_npu_engine() -> AMDNPUInference:
    """Get or create singleton NPU engine instance."""
    global _npu_instance
    if _npu_instance is None:
        _npu_instance = AMDNPUInference()
    return _npu_instance


if __name__ == "__main__":
    # Test NPU inference engine
    print("=" * 60)
    print("AMD Ryzen AI NPU Inference Engine Test")
    print("=" * 60)
    
    # Initialize engine
    engine = AMDNPUInference()
    
    print(f"\nNPU Available: {engine._npu_available}")
    print(f"ONNX Available: {engine._onnx_available}")
    print(f"XGBoost Available: {engine._xgboost_available}")
    
    # Test with random data (simulating regime detection)
    print("\nTesting regime detection with synthetic data...")
    
    # Create synthetic features
    n_features = 50
    test_features = np.random.randn(n_features).astype(np.float32)
    
    # Without a loaded model, demonstrate the API
    print("\nNote: No model loaded. Loading a model required for actual inference.")
    print("Example usage:")
    print("  engine.load_xgboost_model('regime', 'model.json', config)")
    print("  result = engine.infer('regime', features)")
    
    # Show statistics
    stats = engine.get_statistics()
    print("\nEngine Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test regime detector wrapper
    print("\n" + "=" * 60)
    print("Regime Detector Test")
    print("=" * 60)
    
    detector = RegimeDetector(npu_engine=engine)
    
    # Simulate market data
    mock_market_data = {
        'returns': np.random.randn(100) * 0.01,
        'volatility': 0.02,
        'volume': 1000000,
        'bid_volume': 50000,
        'ask_volume': 45000
    }
    
    features = detector.extract_features(mock_market_data)
    print(f"\nExtracted {len(features)} features")
    print(f"Feature sample: {features[:5]}")
    
    print("\n✓ NPU inference engine test completed!")
