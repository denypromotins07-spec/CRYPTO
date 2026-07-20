"""
Online Learning Ensemble for Time-Series Forecasting
Combines XGBoost, LightGBM, and LSTMs with Ray parallel training
Implements object pooling to prevent memory fragmentation
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import threading
import time
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import xgboost as xgb
import lightgbm as lgb
import gc
import psutil


# ============================================================================
# OBJECT POOLING SYSTEM
# ============================================================================

class ObjectPool:
    """
    Generic object pool to prevent memory fragmentation.
    Reuses objects instead of frequent allocation/deallocation.
    """
    
    def __init__(self, factory, max_size: int = 100):
        self.factory = factory
        self.max_size = max_size
        self.pool = deque()
        self.lock = threading.Lock()
        self.created_count = 0
        
    def acquire(self):
        """Acquire an object from the pool or create new one."""
        with self.lock:
            if self.pool:
                return self.pool.popleft()
            else:
                self.created_count += 1
                return self.factory()
    
    def release(self, obj):
        """Return object to the pool."""
        with self.lock:
            if len(self.pool) < self.max_size:
                # Reset object state before returning to pool
                if hasattr(obj, 'reset'):
                    obj.reset()
                self.pool.append(obj)


class NumpyArrayPool(ObjectPool):
    """Specialized pool for NumPy arrays to reduce memory fragmentation."""
    
    def __init__(self, shape: tuple, dtype: np.dtype = np.float32, max_size: int = 50):
        super().__init__(
            factory=lambda: np.zeros(shape, dtype=dtype),
            max_size=max_size
        )
        self.shape = shape
        self.dtype = dtype
    
    def acquire(self) -> np.ndarray:
        arr = super().acquire()
        # Ensure correct shape and dtype
        if arr.shape != self.shape or arr.dtype != self.dtype:
            arr = np.zeros(self.shape, dtype=self.dtype)
        return arr
    
    def release(self, arr: np.ndarray):
        # Zero out array before returning to pool
        arr.fill(0)
        super().release(arr)


# ============================================================================
# LSTM MODEL WITH MEMORY CONSTRAINTS
# ============================================================================

class MemoryEfficientLSTM(nn.Module):
    """
    LSTM model optimized for online learning with strict memory limits.
    Supports incremental updates and gradient accumulation.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Gradient accumulation buffer
        self.accumulated_grads = None
        self.accumulation_steps = 0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LSTM."""
        lstm_out, _ = self.lstm(x)
        # Use last time step
        last_output = lstm_out[:, -1, :]
        return self.output_layer(last_output)
    
    def incremental_update(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        accumulate: bool = False
    ) -> float:
        """
        Perform incremental update with optional gradient accumulation.
        Returns loss value.
        """
        self.train()
        
        if not accumulate:
            optimizer.zero_grad()
        
        predictions = self.forward(x_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        
        if not accumulate:
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
        
        return loss.item()


# ============================================================================
# ENSEMBLE MANAGER
# ============================================================================

@ray.remote(num_cpus=2, memory=500*1024*1024)
class EnsembleWorker:
    """
    Ray worker for parallel ensemble training.
    Each worker handles one model type (XGBoost, LightGBM, or LSTM).
    """
    
    def __init__(self, model_type: str, config: Dict[str, Any]):
        self.model_type = model_type
        self.config = config
        self.model = None
        self.training_history = []
        
    def initialize_model(self):
        """Initialize the model based on type."""
        if self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                **self.config.get('xgboost_params', {})
            )
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMRegressor(
                **self.config.get('lightgbm_params', {})
            )
        elif self.model_type == 'lstm':
            self.model = MemoryEfficientLSTM(**self.config.get('lstm_params', {}))
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X: np.ndarray, y: np.ndarray, val_X: np.ndarray = None, val_y: np.ndarray = None):
        """Train the model on provided data."""
        if self.model is None:
            self.initialize_model()
        
        start_time = time.time()
        
        if self.model_type in ['xgboost', 'lightgbm']:
            # Tree-based models
            eval_set = [(val_X, val_y)] if val_X is not None else None
            self.model.fit(
                X, y,
                eval_set=eval_set,
                verbose=False
            )
        else:
            # LSTM model
            X_tensor = torch.from_numpy(X).float()
            y_tensor = torch.from_numpy(y).float()
            
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            epochs = self.config.get('lstm_epochs', 10)
            batch_size = self.config.get('lstm_batch_size', 32)
            
            for epoch in range(epochs):
                # Mini-batch training
                indices = np.random.permutation(len(X_tensor))
                total_loss = 0
                n_batches = 0
                
                for i in range(0, len(X_tensor), batch_size):
                    batch_idx = indices[i:i+batch_size]
                    batch_x = X_tensor[batch_idx]
                    batch_y = y_tensor[batch_idx]
                    
                    loss = self.model.incremental_update(
                        batch_x, batch_y, optimizer, criterion
                    )
                    total_loss += loss
                    n_batches += 1
                
                avg_loss = total_loss / n_batches
                self.training_history.append({'epoch': epoch, 'loss': avg_loss})
        
        train_time = time.time() - start_time
        
        return {
            'model_type': self.model_type,
            'train_time': train_time,
            'training_history': self.training_history
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        if self.model_type in ['xgboost', 'lightgbm']:
            return self.model.predict(X)
        else:
            X_tensor = torch.from_numpy(X).float()
            self.model.eval()
            with torch.no_grad():
                return self.model(X_tensor).numpy()
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores."""
        if self.model_type == 'xgboost':
            return self.model.feature_importances_
        elif self.model_type == 'lightgbm':
            return self.model.feature_importances_
        else:
            return None  # LSTM doesn't have direct feature importance


class OnlineEnsemble:
    """
    Main ensemble class combining multiple models with online learning.
    Implements weighted averaging and dynamic model selection.
    """
    
    def __init__(
        self,
        n_workers: int = 3,
        memory_limit_gb: float = 6.0,
        retrain_threshold: float = 0.05,
        window_size: int = 10000
    ):
        self.n_workers = n_workers
        self.memory_limit_gb = memory_limit_gb
        self.retrain_threshold = retrain_threshold
        self.window_size = window_size
        
        # Data buffers with fixed size
        self.X_buffer = deque(maxlen=window_size)
        self.y_buffer = deque(maxlen=window_size)
        
        # Model weights for ensemble
        self.model_weights = {
            'xgboost': 0.4,
            'lightgbm': 0.4,
            'lstm': 0.2
        }
        
        # Ray workers
        self.workers = []
        self._initialize_workers()
        
        # Performance tracking
        self.prediction_errors = deque(maxlen=1000)
        self.last_retrain_time = time.time()
        
        # Object pools for memory efficiency
        self.array_pool = NumpyArrayPool(shape=(1000, 50))  # Adjust dimensions as needed
        
    def _initialize_workers(self):
        """Initialize Ray workers for each model type."""
        config = {
            'xgboost_params': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'n_jobs': 1
            },
            'lightgbm_params': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'n_jobs': 1,
                'verbose': -1
            },
            'lstm_params': {
                'input_dim': 50,  # Will be updated dynamically
                'hidden_dim': 128,
                'num_layers': 2,
                'output_dim': 1
            },
            'lstm_epochs': 10,
            'lstm_batch_size': 32
        }
        
        for model_type in ['xgboost', 'lightgbm', 'lstm']:
            worker = EnsembleWorker.remote(model_type, config)
            self.workers.append((model_type, worker))
    
    def add_data(self, X: np.ndarray, y: np.ndarray):
        """Add new data to the buffer."""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        for x_sample, y_sample in zip(X, y):
            self.X_buffer.append(x_sample)
            self.y_buffer.append(y_sample)
    
    def _check_memory_usage(self) -> bool:
        """Check if memory usage is within limits."""
        current_ram_gb = psutil.virtual_memory().used / (1024**3)
        return current_ram_gb < self.memory_limit_gb
    
    def _force_garbage_collection(self):
        """Force garbage collection across all workers."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def train_ensemble(self, use_ray: bool = True):
        """Train all models in the ensemble in parallel."""
        if len(self.X_buffer) < 1000:
            print("Insufficient data for training")
            return
        
        # Convert buffers to arrays
        X_train = np.array(list(self.X_buffer))
        y_train = np.array(list(self.y_buffer))
        
        # Update LSTM input dimension
        input_dim = X_train.shape[1]
        for i, (model_type, worker) in enumerate(self.workers):
            if model_type == 'lstm':
                # Reshape for LSTM: [samples, seq_len, features]
                # For simplicity, using seq_len=10
                seq_len = 10
                if input_dim >= seq_len:
                    features = input_dim // seq_len
                    X_train_lstm = X_train.reshape(-1, seq_len, features)
                else:
                    X_train_lstm = X_train.reshape(-1, 1, input_dim)
                
                # Update worker config
                ray.get(worker.configure_input_dim.remote(features)) if hasattr(worker, 'configure_input_dim') else None
        
        # Split validation set
        val_split = int(0.8 * len(X_train))
        X_tr, X_val = X_train[:val_split], X_train[val_split:]
        y_tr, y_val = y_train[:val_split], y_train[val_split:]
        
        # Train in parallel using Ray
        if use_ray:
            futures = []
            for model_type, worker in self.workers:
                if model_type == 'lstm':
                    future = worker.train.remote(X_train_lstm, y_train)
                else:
                    future = worker.train.remote(X_tr, y_tr, X_val, y_val)
                futures.append(future)
            
            results = ray.get(futures)
            for result in results:
                print(f"Trained {result['model_type']} in {result['train_time']:.2f}s")
        else:
            # Sequential training fallback
            for model_type, worker in self.workers:
                result = ray.get(worker.train.remote(X_tr, y_tr, X_val, y_val))
                print(f"Trained {result['model_type']} in {result['train_time']:.2f}s")
        
        self.last_retrain_time = time.time()
        self._force_garbage_collection()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate ensemble predictions with weighted averaging.
        """
        if len(self.X_buffer) < 100:
            # Fallback to simple mean if insufficient training data
            return np.mean(list(self.y_buffer)) * np.ones(len(X))
        
        predictions = []
        weights = []
        
        for model_type, worker in self.workers:
            try:
                pred_future = worker.predict.remote(X)
                pred = ray.get(pred_future)
                predictions.append(pred)
                weights.append(self.model_weights[model_type])
            except Exception as e:
                print(f"Prediction error for {model_type}: {e}")
                continue
        
        if not predictions:
            return np.mean(list(self.y_buffer)) * np.ones(len(X))
        
        # Weighted average
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        return ensemble_pred
    
    def update_weights(self, X_val: np.ndarray, y_val: np.ndarray):
        """
        Dynamically update model weights based on recent performance.
        Uses inverse error weighting.
        """
        errors = {}
        
        for model_type, worker in self.workers:
            try:
                pred = ray.get(worker.predict.remote(X_val))
                mse = np.mean((pred - y_val) ** 2)
                errors[model_type] = mse
            except:
                errors[model_type] = float('inf')
        
        # Convert errors to weights (inverse proportional)
        total_inv_error = sum(1.0 / (e + 1e-6) for e in errors.values())
        
        for model_type in self.model_weights:
            inv_error = 1.0 / (errors[model_type] + 1e-6)
            self.model_weights[model_type] = inv_error / total_inv_error
        
        print(f"Updated model weights: {self.model_weights}")
    
    def should_retrain(self) -> bool:
        """
        Determine if ensemble should be retrained based on performance degradation.
        """
        # Check time since last retrain
        time_since_retrain = time.time() - self.last_retrain_time
        if time_since_retrain > 3600:  # 1 hour
            return True
        
        # Check prediction error trend
        if len(self.prediction_errors) < 100:
            return False
        
        recent_errors = list(self.prediction_errors)[-50:]
        older_errors = list(self.prediction_errors)[:50]
        
        recent_mse = np.mean(np.array(recent_errors) ** 2)
        older_mse = np.mean(np.array(older_errors) ** 2)
        
        # If recent error increased by more than threshold
        if (recent_mse - older_mse) / (older_mse + 1e-6) > self.retrain_threshold:
            return True
        
        return False
    
    def record_prediction_error(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Record prediction errors for drift detection."""
        errors = np.abs(y_true - y_pred).flatten()
        for error in errors:
            self.prediction_errors.append(error)


# ============================================================================
# RAY INITIALIZATION AND TESTING
# ============================================================================

def initialize_ray(memory_gb: float = 6.0):
    """Initialize Ray with memory constraints."""
    if not ray.is_initialized():
        ray.init(
            _temp_dir='/tmp/ray_temp',
            _memory=int(memory_gb * 1024 * 1024 * 1024),
            object_store_memory=int(memory_gb * 0.3 * 1024 * 1024 * 1024),
            include_dashboard=False,
            ignore_reinit_error=True
        )


if __name__ == '__main__':
    # Initialize Ray
    initialize_ray(memory_gb=6.0)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 5000
    n_features = 50
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.sin(X[:, 0]) + 0.5 * np.cos(X[:, 1]) + 0.1 * np.random.randn(n_samples)
    
    # Create ensemble
    ensemble = OnlineEnsemble(
        n_workers=3,
        memory_limit_gb=6.0,
        retrain_threshold=0.05,
        window_size=10000
    )
    
    # Add data
    ensemble.add_data(X, y)
    
    # Train ensemble
    print("Training ensemble...")
    ensemble.train_ensemble(use_ray=True)
    
    # Test prediction
    X_test = X[-100:]
    y_test = y[-100:]
    
    predictions = ensemble.predict(X_test)
    print(f"Predictions shape: {predictions.shape}")
    
    # Record errors
    ensemble.record_prediction_error(y_test, predictions)
    
    # Test weight update
    ensemble.update_weights(X_test, y_test)
    
    # Check if retrain needed
    print(f"Should retrain: {ensemble.should_retrain()}")
    
    # Cleanup
    ray.shutdown()
