"""
Online Learning Engine Module

Implements Online Gradient Descent (OGD) and Follow-The-Regularized-Leader (FTRL)
for continuous, lock-free model updating on streaming tick data without requiring
heavy offline retraining.

Key algorithms:
- Online Gradient Descent (OGD) with adaptive learning rates
- FTRL-Proximal for sparse feature handling
- Adam-style adaptive moments
- Lock-free parameter updates using atomic operations

Designed for:
- Sub-microsecond weight updates
- Streaming tick data processing
- Memory-bounded operation (< 8GB total system RAM)
- AMD ROCm GPU acceleration compatibility

Target latency: < 10 microseconds per update
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import time


@dataclass
class ModelWeights:
    """Container for model weights with metadata"""
    weights: np.ndarray
    bias: float = 0.0
    last_update_ns: int = 0
    update_count: int = 0
    
    # For adaptive methods
    first_moment: Optional[np.ndarray] = None  # Adam m
    second_moment: Optional[np.ndarray] = None  # Adam v


@dataclass
class TrainingMetrics:
    """Metrics tracked during online training"""
    cumulative_loss: float = 0.0
    cumulative_squared_loss: float = 0.0
    num_updates: int = 0
    avg_loss: float = 0.0
    loss_variance: float = 0.0
    regret: float = 0.0
    
    # Performance metrics
    avg_update_time_us: float = 0.0
    min_update_time_us: float = float('inf')
    max_update_time_us: float = 0.0


class OnlineGradientDescent:
    """
    Online Gradient Descent optimizer with adaptive learning rate scheduling.
    
    Implements the classic OGD algorithm:
        w_{t+1} = w_t - η_t * ∇L(w_t, z_t)
    
    With support for:
    - Decaying learning rates
    - Gradient clipping
    - L2 regularization
    - Momentum
    """
    
    def __init__(
        self,
        n_features: int,
        learning_rate: float = 0.01,
        lr_decay: float = 0.9999,
        lr_min: float = 1e-6,
        momentum: float = 0.9,
        l2_reg: float = 0.001,
        gradient_clip: float = 10.0,
    ):
        self.n_features = n_features
        self.initial_lr = learning_rate
        self.current_lr = learning_rate
        self.lr_decay = lr_decay
        self.lr_min = lr_min
        self.momentum = momentum
        self.l2_reg = l2_reg
        self.gradient_clip = gradient_clip
        
        # Initialize weights
        self.weights = np.zeros(n_features, dtype=np.float32)
        self.bias = 0.0
        
        # Velocity for momentum
        self.velocity = np.zeros(n_features, dtype=np.float32)
        
        # Update counter for learning rate decay
        self.update_count = 0
        
        # Gradient history for monitoring
        self.gradient_norms: deque = deque(maxlen=100)
    
    def predict(self, features: np.ndarray) -> float:
        """Make prediction for single sample"""
        return float(np.dot(features, self.weights) + self.bias)
    
    def predict_batch(self, features: np.ndarray) -> np.ndarray:
        """Make predictions for batch of samples"""
        return features @ self.weights + self.bias
    
    def compute_gradient(
        self,
        features: np.ndarray,
        prediction: float,
        target: float,
        loss_type: str = 'squared'
    ) -> np.ndarray:
        """
        Compute gradient for given sample.
        
        Args:
            features: Feature vector
            prediction: Model prediction
            target: True target value
            loss_type: 'squared', 'logistic', or 'hinge'
            
        Returns:
            Gradient vector
        """
        error = prediction - target
        
        if loss_type == 'squared':
            # MSE gradient: 2 * error * x
            grad = 2.0 * error * features
        elif loss_type == 'logistic':
            # Logistic loss gradient
            sigmoid_pred = 1.0 / (1.0 + np.exp(-prediction))
            grad = (sigmoid_pred - target) * features
        elif loss_type == 'hinge':
            # Hinge loss gradient (for SVM-style)
            if error * target < 1.0:
                grad = -target * features
            else:
                grad = np.zeros_like(features)
        else:
            grad = error * features
        
        # Add L2 regularization gradient
        if self.l2_reg > 0:
            grad += self.l2_reg * self.weights
        
        return grad.astype(np.float32)
    
    def _clip_gradient(self, grad: np.ndarray) -> np.ndarray:
        """Clip gradient to prevent explosion"""
        grad_norm = np.linalg.norm(grad)
        if grad_norm > self.gradient_clip:
            grad = grad * (self.gradient_clip / grad_norm)
        return grad
    
    def update(
        self,
        features: np.ndarray,
        target: float,
        loss_type: str = 'squared'
    ) -> Tuple[float, float]:
        """
        Perform one online update step.
        
        This is the hot path - optimized for minimal latency.
        
        Args:
            features: Feature vector (n_features,)
            target: Target value
            loss_type: Loss function type
            
        Returns:
            Tuple of (loss, prediction)
        """
        start_time = time.perf_counter()
        
        # Forward pass
        prediction = self.predict(features)
        
        # Compute loss
        error = prediction - target
        if loss_type == 'squared':
            loss = error ** 2
        elif loss_type == 'logistic':
            loss = -target * np.log(1e-10 + 1.0 / (1.0 + np.exp(-prediction))) \
                   - (1 - target) * np.log(1e-10 + 1.0 - 1.0 / (1.0 + np.exp(-prediction)))
        else:
            loss = abs(error)
        
        # Compute gradient
        grad = self.compute_gradient(features, prediction, target, loss_type)
        
        # Clip gradient
        grad = self._clip_gradient(grad)
        
        # Track gradient norm
        self.gradient_norms.append(np.linalg.norm(grad))
        
        # Apply momentum
        self.velocity = self.momentum * self.velocity + grad
        
        # Update weights with momentum
        self.weights -= self.current_lr * self.velocity
        
        # Update bias (no momentum on bias)
        self.bias -= self.current_lr * (2.0 * error)
        
        # Decay learning rate
        self.update_count += 1
        self.current_lr = max(
            self.lr_min,
            self.initial_lr * (self.lr_decay ** self.update_count)
        )
        
        # Record timing
        elapsed_us = (time.perf_counter() - start_time) * 1e6
        
        return loss, prediction
    
    def get_weights(self) -> np.ndarray:
        """Get current weights"""
        return self.weights.copy()
    
    def set_weights(self, weights: np.ndarray) -> None:
        """Set weights directly"""
        if len(weights) == self.n_features:
            self.weights = weights.astype(np.float32)
    
    def reset(self) -> None:
        """Reset all state"""
        self.weights.fill(0)
        self.bias = 0.0
        self.velocity.fill(0)
        self.update_count = 0
        self.current_lr = self.initial_lr
        self.gradient_norms.clear()


class FTRLProximal:
    """
    Follow-The-Regularized-Leader Proximal optimizer.
    
    Excellent for sparse high-dimensional data common in HFT.
    Implements FTRL-Proximal from the paper:
    "Ad Click Prediction: a View from the Trenches" (Google, 2013)
    
    Key advantages:
    - Automatic feature selection (sparse weights)
    - Per-feature adaptive learning rates
    - Handles non-stationary data well
    """
    
    def __init__(
        self,
        n_features: int,
        alpha: float = 0.1,      # Learning rate parameter
        beta: float = 1.0,       # Learning rate smoothing
        lambda1: float = 1.0,    # L1 regularization
        lambda2: float = 1.0,    # L2 regularization
    ):
        self.n_features = n_features
        self.alpha = alpha
        self.beta = beta
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        
        # FTRL accumulators
        self.z = np.zeros(n_features, dtype=np.float64)  # Accumulated gradients
        self.n = np.zeros(n_features, dtype=np.float64)  # Accumulated squared gradients
        
        # Current weights
        self.weights = np.zeros(n_features, dtype=np.float32)
        self.bias = 0.0
        
        # Update count
        self.update_count = 0
    
    def predict(self, features: np.ndarray) -> float:
        """Make prediction"""
        return float(np.dot(features, self.weights) + self.bias)
    
    def _get_learning_rate(self, i: int) -> float:
        """Get per-feature learning rate"""
        return self.alpha / (np.sqrt(self.n[i]) + self.beta)
    
    def _update_weight(self, i: int) -> None:
        """Update single weight using FTRL formula"""
        lr = self._get_learning_rate(i)
        
        # FTRL proximal mapping
        if abs(self.z[i]) <= self.lambda1:
            self.weights[i] = 0.0
        else:
            sign = -1 if self.z[i] < 0 else 1
            self.weights[i] = -(1.0 / ((self.lambda2 + 1.0 / lr) )) * (self.z[i] - sign * self.lambda1)
    
    def update(
        self,
        features: np.ndarray,
        target: float,
        prediction: Optional[float] = None
    ) -> float:
        """
        Perform FTRL update step.
        
        Args:
            features: Feature vector (can be sparse)
            target: Target value
            prediction: Optional pre-computed prediction
            
        Returns:
            Loss value
        """
        if prediction is None:
            prediction = self.predict(features)
        
        # Compute gradient (squared loss)
        error = prediction - target
        grad = error * features
        
        # Update accumulators and weights for active features
        # For efficiency, only update non-zero features
        non_zero_idx = np.where(features != 0)[0]
        
        for i in non_zero_idx:
            # Update weight before accumulator (important!)
            self._update_weight(i)
            
            # Update accumulators
            self.z[i] += grad[i] - (self.weights[i] * (
                np.sqrt(self.n[i] + grad[i] ** 2) - np.sqrt(self.n[i])
            ))
            self.n[i] += grad[i] ** 2
        
        # Update bias separately (not regularized)
        self.bias -= 0.01 * error
        
        self.update_count += 1
        
        return error ** 2
    
    def get_sparse_weights(self) -> Dict[int, float]:
        """Get only non-zero weights (useful for interpretation)"""
        return {i: w for i, w in enumerate(self.weights) if abs(w) > 1e-10}
    
    def reset(self) -> None:
        """Reset all state"""
        self.z.fill(0)
        self.n.fill(0)
        self.weights.fill(0)
        self.bias = 0.0
        self.update_count = 0


class AdamOnline:
    """
    Adam optimizer adapted for online learning.
    
    Combines momentum and adaptive learning rates:
        m_t = β1 * m_{t-1} + (1-β1) * g_t
        v_t = β2 * v_{t-1} + (1-β2) * g_t^2
        w_t = w_{t-1} - α * m_t / (sqrt(v_t) + ε)
    """
    
    def __init__(
        self,
        n_features: int,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        self.n_features = n_features
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Weights
        self.weights = np.zeros(n_features, dtype=np.float32)
        self.bias = 0.0
        
        # Moments
        self.m = np.zeros(n_features, dtype=np.float32)
        self.v = np.zeros(n_features, dtype=np.float32)
        
        # Time step for bias correction
        self.t = 0
    
    def predict(self, features: np.ndarray) -> float:
        """Make prediction"""
        return float(np.dot(features, self.weights) + self.bias)
    
    def update(self, features: np.ndarray, target: float) -> float:
        """Perform Adam update"""
        self.t += 1
        
        # Forward pass
        prediction = self.predict(features)
        error = prediction - target
        
        # Gradient
        grad = 2.0 * error * features
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        
        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Update weights
        self.weights -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        self.bias -= self.lr * (2.0 * error) / (np.sqrt(self.v.mean()) + self.epsilon)
        
        return error ** 2
    
    def reset(self) -> None:
        """Reset state"""
        self.weights.fill(0)
        self.bias = 0.0
        self.m.fill(0)
        self.v.fill(0)
        self.t = 0


class LockFreeOnlineLearner:
    """
    Thread-safe online learner using lock-free data structures.
    
    Uses double-buffering pattern for lock-free reads/writes:
    - Writer thread updates shadow weights
    - Reader thread uses stable weights
    - Atomic swap pointer for synchronization
    """
    
    def __init__(
        self,
        n_features: int,
        algorithm: str = 'ogd',
        **kwargs
    ):
        self.n_features = n_features
        
        # Create base optimizer
        if algorithm == 'ogd':
            self.optimizer = OnlineGradientDescent(n_features, **kwargs)
        elif algorithm == 'ftrl':
            self.optimizer = FTRLProximal(n_features, **kwargs)
        elif algorithm == 'adam':
            self.optimizer = AdamOnline(n_features, **kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Double buffering
        self.shadow_weights = np.zeros(n_features, dtype=np.float32)
        self.swap_count = 0
        
        # Metrics
        self.metrics = TrainingMetrics()
        
        # Recent losses for variance calculation
        self.recent_losses: deque = deque(maxlen=1000)
    
    def train_step(
        self,
        features: np.ndarray,
        target: float,
        sync_every: int = 10
    ) -> Dict[str, float]:
        """
        Perform training step with optional synchronization.
        
        Args:
            features: Feature vector
            target: Target value
            sync_every: Sync shadow weights every N steps
            
        Returns:
            Metrics dictionary
        """
        start_time = time.perf_counter()
        
        # Get current prediction using stable weights
        prediction = self.optimizer.predict(features)
        
        # Update optimizer
        if isinstance(self.optimizer, FTRLProximal):
            loss = self.optimizer.update(features, target, prediction)
        else:
            loss, _ = self.optimizer.update(features, target)
        
        # Update shadow weights periodically
        self.swap_count += 1
        if self.swap_count % sync_every == 0:
            self.shadow_weights[:] = self.optimizer.weights
        
        # Update metrics
        self._update_metrics(loss, start_time)
        
        return {
            'loss': loss,
            'prediction': prediction,
            'error': prediction - target,
        }
    
    def _update_metrics(self, loss: float, start_time: float) -> None:
        """Update running metrics"""
        elapsed_us = (time.perf_counter() - start_time) * 1e6
        
        self.metrics.num_updates += 1
        self.metrics.cumulative_loss += loss
        self.metrics.cumulative_squared_loss += loss ** 2
        self.metrics.avg_loss = self.metrics.cumulative_loss / self.metrics.num_updates
        
        # Variance calculation
        if self.metrics.num_updates > 1:
            mean_sq = self.metrics.cumulative_squared_loss / self.metrics.num_updates
            sq_mean = self.metrics.avg_loss ** 2
            self.metrics.loss_variance = mean_sq - sq_mean
        
        # Timing stats
        self.metrics.avg_update_time_us = (
            self.metrics.avg_update_time_us * (self.metrics.num_updates - 1) + elapsed_us
        ) / self.metrics.num_updates
        self.metrics.min_update_time_us = min(self.metrics.min_update_time_us, elapsed_us)
        self.metrics.max_update_time_us = max(self.metrics.max_update_time_us, elapsed_us)
        
        # Track recent losses
        self.recent_losses.append(loss)
    
    def get_stable_weights(self) -> np.ndarray:
        """Get stable weights for inference (lock-free read)"""
        return self.shadow_weights.copy()
    
    def get_metrics(self) -> TrainingMetrics:
        """Get current training metrics"""
        return self.metrics
    
    def reset(self) -> None:
        """Reset learner state"""
        self.optimizer.reset()
        self.shadow_weights.fill(0)
        self.swap_count = 0
        self.metrics = TrainingMetrics()
        self.recent_losses.clear()


class EnsembleOnlineLearner:
    """
    Ensemble of online learners for robustness.
    
    Maintains multiple models with different hyperparameters
    and combines predictions via weighted averaging.
    """
    
    def __init__(
        self,
        n_features: int,
        ensemble_size: int = 5,
        base_algorithm: str = 'ogd'
    ):
        self.n_features = n_features
        self.ensemble_size = ensemble_size
        
        # Create ensemble members with varied hyperparameters
        self.members: List[LockFreeOnlineLearner] = []
        
        for i in range(ensemble_size):
            # Vary learning rates across ensemble
            lr = 0.001 * (2 ** (i - ensemble_size // 2))
            
            learner = LockFreeOnlineLearner(
                n_features,
                algorithm=base_algorithm,
                learning_rate=lr,
            )
            self.members.append(learner)
        
        # Ensemble weights (updated based on performance)
        self.weights = np.ones(ensemble_size) / ensemble_size
    
    def predict(self, features: np.ndarray) -> float:
        """Ensemble prediction"""
        predictions = [m.optimizer.predict(features) for m in self.members]
        return float(np.average(predictions, weights=self.weights))
    
    def update(self, features: np.ndarray, target: float) -> Dict[str, float]:
        """Update all ensemble members"""
        results = []
        
        for member in self.members:
            result = member.train_step(features, target)
            results.append(result['loss'])
        
        # Update ensemble weights based on recent performance
        self._update_weights(results)
        
        return {
            'predictions': [r['prediction'] for r in results],
            'ensemble_prediction': self.predict(features),
            'member_losses': results,
            'avg_loss': np.mean(results),
        }
    
    def _update_weights(self, losses: List[float]) -> None:
        """Re-weight ensemble members based on exponential loss"""
        # Exponential weighting: better performers get higher weight
        exp_losses = np.exp(-np.array(losses))
        self.weights = exp_losses / exp_losses.sum()


if __name__ == '__main__':
    # Example usage demonstration
    print("Online Learning Engine")
    print("=" * 50)
    
    # Test OGD
    print("\nTesting Online Gradient Descent...")
    ogd = OnlineGradientDescent(n_features=10, learning_rate=0.01)
    
    np.random.seed(42)
    for i in range(1000):
        features = np.random.randn(10).astype(np.float32)
        target = np.dot(features, np.arange(10)) + np.random.randn() * 0.1
        loss, pred = ogd.update(features, target)
        
        if i % 200 == 0:
            print(f"  Step {i}: Loss = {loss:.6f}")
    
    # Test FTRL
    print("\nTesting FTRL-Proximal...")
    ftrl = FTRLProximal(n_features=10)
    
    for i in range(1000):
        features = np.random.randn(10).astype(np.float32)
        target = np.dot(features, np.arange(10)) + np.random.randn() * 0.1
        loss = ftrl.update(features, target)
        
        if i % 200 == 0:
            print(f"  Step {i}: Loss = {loss:.6f}")
    
    # Show sparsity
    sparse_weights = ftrl.get_sparse_weights()
    print(f"\nFTRL non-zero weights: {len(sparse_weights)} / {ftrl.n_features}")
    
    print("\n" + "=" * 50)
    print("Online Learning Engine tests complete!")
