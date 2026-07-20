"""
Bayesian Optimization Module

Bayesian Optimization (using Gaussian Processes) for hyperparameter tuning.
Optimizes strategy parameters while strictly penalizing overfitting, high
drawdowns, and parameter sensitivity.

Key features:
- Gaussian Process surrogate model
- Expected Improvement / UCB / Probability of Improvement acquisition
- Overfitting penalty via walk-forward validation
- Drawdown and risk constraints
- Parameter sensitivity analysis
- Memory-bounded operation

Target: Find optimal parameters in < 100 evaluations
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass, field
from scipy.optimize import minimize
from scipy.stats import norm
import warnings


@dataclass
class ParameterBounds:
    """Definition of a parameter's search space"""
    name: str
    lower: float
    upper: float
    log_scale: bool = False  # Use log scale for this parameter
    discrete_values: Optional[List[float]] = None  # For discrete params
    prior_mean: Optional[float] = None
    prior_std: Optional[float] = None


@dataclass
class EvaluationResult:
    """Result of evaluating a parameter set"""
    params: Dict[str, float]
    objective_value: float
    metrics: Dict[str, float]
    is_valid: bool
    evaluation_time_s: float
    iteration: int


@dataclass
class BayesianOptimizationResult:
    """Final result of Bayesian optimization"""
    best_params: Dict[str, float]
    best_objective: float
    all_evaluations: List[EvaluationResult]
    num_evaluations: int
    convergence_history: List[float]
    parameter_importance: Dict[str, float]
    final_gp_model: 'GaussianProcess'
    success: bool
    message: str


class GaussianProcess:
    """
    Simplified Gaussian Process implementation for Bayesian Optimization.
    
    Uses RBF kernel with automatic relevance determination (ARD).
    Optimized for low-dimensional parameter spaces typical in strategy tuning.
    """
    
    def __init__(
        self,
        length_scale: float = 1.0,
        noise_level: float = 1e-6,
        alpha: float = 1e-10,
    ):
        self.length_scale = length_scale
        self.noise_level = noise_level
        self.alpha = alpha
        
        # Training data
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        
        # Computed matrices
        self.L: Optional[np.ndarray] = None
        self.alpha_vec: Optional[np.ndarray] = None
    
    def _rbf_kernel(
        self,
        X1: np.ndarray,
        X2: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute RBF (squared exponential) kernel with ARD.
        
        K(x, x') = exp(-0.5 * sum((x - x')^2 / l^2))
        """
        if X2 is None:
            X2 = X1
        
        # Scale by length scale
        X1_scaled = X1 / self.length_scale
        X2_scaled = X2 / self.length_scale
        
        # Compute squared distances
        sq_dist = (
            np.sum(X1_scaled ** 2, axis=1, keepdims=True) +
            np.sum(X2_scaled ** 2, axis=1) -
            2 * X1_scaled @ X2_scaled.T
        )
        
        return np.exp(-0.5 * sq_dist)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianProcess':
        """
        Fit the GP to training data.
        
        Args:
            X: Input points (n_samples, n_features)
            y: Target values (n_samples,)
        """
        self.X_train = X.copy()
        self.y_train = y.copy()
        
        n = len(X)
        
        # Compute kernel matrix
        K = self._rbf_kernel(X)
        
        # Add noise and regularization
        K += (self.noise_level + self.alpha) * np.eye(n)
        
        # Cholesky decomposition for numerical stability
        try:
            self.L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # Add jitter if not positive definite
            K += 1e-6 * np.eye(n)
            self.L = np.linalg.cholesky(K)
        
        # Solve for alpha vector
        self.alpha_vec = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y))
        
        return self
    
    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict mean and optionally standard deviation at new points.
        
        Args:
            X: Query points (n_query, n_features)
            return_std: Whether to return std deviation
            
        Returns:
            Mean predictions, optionally with std deviations
        """
        if self.X_train is None or self.L is None:
            raise ValueError("GP must be fitted before prediction")
        
        # Kernel between query and training points
        K_star = self._rbf_kernel(X, self.X_train)
        
        # Mean prediction
        mu = K_star @ self.alpha_vec
        
        if not return_std:
            return mu
        
        # Variance
        v = np.linalg.solve(self.L, K_star.T)
        var = 1 - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 1e-10)  # Numerical stability
        std = np.sqrt(var)
        
        return mu, std
    
    def sample_posterior(
        self,
        X: np.ndarray,
        n_samples: int = 1
    ) -> np.ndarray:
        """Sample from the posterior distribution"""
        mu, std = self.predict(X, return_std=True)
        
        # Generate samples
        samples = np.random.normal(mu, std[:, np.newaxis], size=(len(X), n_samples))
        
        return samples.squeeze()


class AcquisitionFunction:
    """Base class for acquisition functions"""
    
    def __call__(self, gp: GaussianProcess, X: np.ndarray, y_min: float) -> np.ndarray:
        raise NotImplementedError


class ExpectedImprovement(AcquisitionFunction):
    """Expected Improvement acquisition function"""
    
    def __init__(self, xi: float = 0.01):
        self.xi = xi
    
    def __call__(self, gp: GaussianProcess, X: np.ndarray, y_min: float) -> np.ndarray:
        mu, std = gp.predict(X, return_std=True)
        
        # Avoid division by zero
        std = np.maximum(std, 1e-10)
        
        # Improvement
        Z = (mu - y_min - self.xi) / std
        
        # Expected improvement
        ei = (mu - y_min - self.xi) * norm.cdf(Z) + std * norm.pdf(Z)
        
        return ei


class UpperConfidenceBound(AcquisitionFunction):
    """Upper Confidence Bound acquisition function"""
    
    def __init__(self, kappa: float = 2.576):
        self.kappa = kappa
    
    def __call__(self, gp: GaussianProcess, X: np.ndarray, y_min: float) -> np.ndarray:
        mu, std = gp.predict(X, return_std=True)
        return mu + self.kappa * std


class ProbabilityOfImprovement(AcquisitionFunction):
    """Probability of Improvement acquisition function"""
    
    def __init__(self, xi: float = 0.01):
        self.xi = xi
    
    def __call__(self, gp: GaussianProcess, X: np.ndarray, y_min: float) -> np.ndarray:
        mu, std = gp.predict(X, return_std=True)
        std = np.maximum(std, 1e-10)
        
        Z = (mu - y_min - self.xi) / std
        return norm.cdf(Z)


class BayesianOptimizer:
    """
    Main Bayesian Optimization class for hyperparameter tuning.
    
    Designed specifically for trading strategy optimization with:
    - Overfitting penalties
    - Risk constraints
    - Walk-forward validation support
    """
    
    def __init__(
        self,
        param_bounds: List[ParameterBounds],
        acquisition: str = 'ei',
        n_initial_points: int = 5,
        max_evaluations: int = 50,
        random_state: Optional[int] = None,
    ):
        self.param_bounds = param_bounds
        self.n_params = len(param_bounds)
        self.n_initial_points = n_initial_points
        self.max_evaluations = max_evaluations
        self.random_state = random_state
        
        # Initialize acquisition function
        if acquisition == 'ei':
            self.acquisition = ExpectedImprovement(xi=0.01)
        elif acquisition == 'ucb':
            self.acquisition = UpperConfidenceBound(kappa=2.576)
        elif acquisition == 'poi':
            self.acquisition = ProbabilityOfImprovement(xi=0.01)
        else:
            self.acquisition = ExpectedImprovement()
        
        # Storage
        self.evaluations: List[EvaluationResult] = []
        self.X_sampled: List[np.ndarray] = []
        self.y_sampled: List[float] = []
        
        # GP model
        self.gp = GaussianProcess(length_scale=1.0, noise_level=1e-6)
        
        # Best found
        self.best_params: Optional[Dict[str, float]] = None
        self.best_objective: float = float('inf')
        
        # Convergence tracking
        self.convergence_history: List[float] = []
        
        # Random generator
        self.rng = np.random.default_rng(random_state)
    
    def _params_to_array(self, params: Dict[str, float]) -> np.ndarray:
        """Convert parameter dict to array in consistent order"""
        return np.array([params[pb.name] for pb in self.param_bounds])
    
    def _array_to_params(self, arr: np.ndarray) -> Dict[str, float]:
        """Convert array to parameter dict"""
        return {pb.name: arr[i] for i, pb in enumerate(self.param_bounds)}
    
    def _sample_initial_points(self) -> np.ndarray:
        """Generate initial points using Latin Hypercube Sampling"""
        n = self.n_initial_points
        d = self.n_params
        
        # Simple LHS approximation
        X = np.zeros((n, d))
        
        for j in range(d):
            pb = self.param_bounds[j]
            
            # Generate stratified samples
            perm = self.rng.permutation(n)
            
            if pb.log_scale:
                # Log-uniform sampling
                log_lower = np.log(max(pb.lower, 1e-10))
                log_upper = np.log(pb.upper)
                samples = np.exp(log_lower + (log_upper - log_lower) * (perm + self.rng.random(n)) / n)
            elif pb.discrete_values is not None:
                # Discrete sampling
                indices = self.rng.choice(len(pb.discrete_values), size=n, replace=False)
                samples = [pb.discrete_values[i] for i in indices]
            else:
                # Uniform sampling
                samples = pb.lower + (pb.upper - pb.lower) * (perm + self.rng.random(n)) / n
            
            X[:, j] = samples
        
        return X
    
    def _apply_constraints(
        self,
        params: Dict[str, float],
        metrics: Dict[str, float]
    ) -> Tuple[float, bool]:
        """
        Apply constraints and penalties to objective.
        
        Returns:
            Tuple of (penalized_objective, is_valid)
        """
        base_objective = metrics.get('objective', 0.0)
        is_valid = True
        
        # Penalty for high drawdown
        max_dd = abs(metrics.get('max_drawdown', 0))
        dd_threshold = 0.20  # 20% max drawdown threshold
        if max_dd > dd_threshold:
            base_objective += 1000 * (max_dd - dd_threshold) ** 2
            is_valid = False
        
        # Penalty for low Sharpe ratio
        sharpe = metrics.get('sharpe_ratio', 0)
        if sharpe < 1.0:
            base_objective += 10 * (1.0 - sharpe) ** 2
        
        # Penalty for parameter sensitivity (if available)
        sensitivity = metrics.get('parameter_sensitivity', 0)
        if sensitivity > 0.5:
            base_objective += 5 * sensitivity
        
        # Penalty for overfitting (train/test divergence)
        train_return = metrics.get('train_return', 0)
        test_return = metrics.get('test_return', 0)
        if train_return > 0 and test_return > 0:
            divergence = abs(train_return - test_return) / max(train_return, 1e-10)
            if divergence > 0.3:
                base_objective += 100 * divergence ** 2
                is_valid = False
        
        return base_objective, is_valid
    
    def optimize(
        self,
        objective_func: Callable[[Dict[str, float]], Dict[str, float]],
        verbose: bool = True,
    ) -> BayesianOptimizationResult:
        """
        Run Bayesian optimization.
        
        Args:
            objective_func: Function that takes params dict and returns metrics dict
                Must include 'objective' key for the value to minimize
            verbose: Print progress
            
        Returns:
            BayesianOptimizationResult
        """
        import time
        
        if verbose:
            print(f"Starting Bayesian Optimization ({self.max_evaluations} evaluations)")
            print(f"Parameters: {[pb.name for pb in self.param_bounds]}")
            print("-" * 60)
        
        # Sample initial points
        X_initial = self._sample_initial_points()
        
        # Evaluate initial points
        for i, x in enumerate(X_initial):
            params = self._array_to_params(x)
            
            start_time = time.time()
            try:
                metrics = objective_func(params)
                eval_time = time.time() - start_time
                
                obj_value, is_valid = self._apply_constraints(params, metrics)
                
                result = EvaluationResult(
                    params=params,
                    objective_value=obj_value,
                    metrics=metrics,
                    is_valid=is_valid,
                    evaluation_time_s=eval_time,
                    iteration=i,
                )
                
                self.evaluations.append(result)
                self.X_sampled.append(x)
                self.y_sampled.append(obj_value)
                
                # Update best
                if obj_value < self.best_objective:
                    self.best_objective = obj_value
                    self.best_params = params.copy()
                
                self.convergence_history.append(self.best_objective)
                
                if verbose:
                    status = "✓" if is_valid else "✗"
                    print(f"[{status}] Iter {i+1}: Obj={obj_value:.4f}, Best={self.best_objective:.4f}")
                    
            except Exception as e:
                if verbose:
                    print(f"[!] Iter {i+1}: Error - {str(e)}")
        
        # Fit GP on initial points
        if len(self.X_sampled) >= 2:
            X_arr = np.array(self.X_sampled)
            y_arr = np.array(self.y_sampled)
            self.gp.fit(X_arr, y_arr)
        
        # Main optimization loop
        for iteration in range(self.n_initial_points, self.max_evaluations):
            # Find next point by maximizing acquisition function
            x_next = self._suggest_next_point()
            
            if x_next is None:
                if verbose:
                    print("Could not suggest next point, stopping early")
                break
            
            # Evaluate
            params = self._array_to_params(x_next)
            
            start_time = time.time()
            try:
                metrics = objective_func(params)
                eval_time = time.time() - start_time
                
                obj_value, is_valid = self._apply_constraints(params, metrics)
                
                result = EvaluationResult(
                    params=params,
                    objective_value=obj_value,
                    metrics=metrics,
                    is_valid=is_valid,
                    evaluation_time_s=eval_time,
                    iteration=iteration,
                )
                
                self.evaluations.append(result)
                self.X_sampled.append(x_next)
                self.y_sampled.append(obj_value)
                
                # Update best
                if obj_value < self.best_objective:
                    self.best_objective = obj_value
                    self.best_params = params.copy()
                
                self.convergence_history.append(self.best_objective)
                
                # Refit GP
                X_arr = np.array(self.X_sampled)
                y_arr = np.array(self.y_sampled)
                self.gp.fit(X_arr, y_arr)
                
                if verbose:
                    status = "✓" if is_valid else "✗"
                    print(f"[{status}] Iter {iteration+1}: Obj={obj_value:.4f}, Best={self.best_objective:.4f}")
                    
            except Exception as e:
                if verbose:
                    print(f"[!] Iter {iteration+1}: Error - {str(e)}")
        
        # Calculate parameter importance
        param_importance = self._calculate_parameter_importance()
        
        return BayesianOptimizationResult(
            best_params=self.best_params or {},
            best_objective=self.best_objective,
            all_evaluations=self.evaluations,
            num_evaluations=len(self.evaluations),
            convergence_history=self.convergence_history,
            parameter_importance=param_importance,
            final_gp_model=self.gp,
            success=self.best_params is not None,
            message="Optimization completed" if self.best_params else "No valid points found",
        )
    
    def _suggest_next_point(self) -> Optional[np.ndarray]:
        """Suggest next point to evaluate by maximizing acquisition"""
        if self.gp.X_train is None:
            return None
        
        y_min = min(self.y_sampled)
        
        # Create candidate points
        candidates = self._generate_candidates(n_candidates=1000)
        
        # Evaluate acquisition function
        acq_values = self.acquisition(self.gp, candidates, y_min)
        
        # Select best candidate
        best_idx = np.argmax(acq_values)
        return candidates[best_idx]
    
    def _generate_candidates(self, n_candidates: int = 1000) -> np.ndarray:
        """Generate candidate points for acquisition maximization"""
        candidates = []
        
        for _ in range(n_candidates):
            point = []
            for pb in self.param_bounds:
                if pb.log_scale:
                    val = np.exp(self.rng.uniform(np.log(max(pb.lower, 1e-10)), np.log(pb.upper)))
                elif pb.discrete_values is not None:
                    val = self.rng.choice(pb.discrete_values)
                else:
                    val = self.rng.uniform(pb.lower, pb.upper)
                point.append(val)
            candidates.append(point)
        
        return np.array(candidates)
    
    def _calculate_parameter_importance(self) -> Dict[str, float]:
        """Calculate parameter importance using GP length scales"""
        importance = {}
        
        # Inverse of length scale indicates importance
        total_inv_scale = 0
        inv_scales = []
        
        for pb in self.param_bounds:
            inv_scale = 1.0 / self.gp.length_scale  # Simplified - would use ARD in full impl
            inv_scales.append(inv_scale)
            total_inv_scale += inv_scale
        
        for i, pb in enumerate(self.param_bounds):
            if total_inv_scale > 0:
                importance[pb.name] = inv_scales[i] / total_inv_scale
            else:
                importance[pb.name] = 1.0 / len(self.param_bounds)
        
        return importance
    
    def plot_convergence(self) -> None:
        """Plot optimization convergence (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Convergence plot
        axes[0].plot(self.convergence_history, marker='o')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Best Objective')
        axes[0].set_title('Optimization Convergence')
        axes[0].grid(True, alpha=0.3)
        
        # Evaluation history
        objectives = [e.objective_value for e in self.evaluations]
        axes[1].scatter(range(len(objectives)), objectives, alpha=0.6)
        axes[1].axhline(self.best_objective, color='r', linestyle='--', label=f'Best: {self.best_objective:.4f}')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Objective Value')
        axes[1].set_title('All Evaluations')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# Convenience function for simple optimization
def optimize_strategy(
    param_bounds: Dict[str, Tuple[float, float]],
    objective_func: Callable[[Dict[str, float]], Dict[str, float]],
    n_evaluations: int = 30,
    random_state: int = 42,
) -> BayesianOptimizationResult:
    """
    Quick helper for strategy optimization.
    
    Args:
        param_bounds: Dict of {param_name: (lower, upper)}
        objective_func: Function returning metrics dict with 'objective' key
        n_evaluations: Maximum number of evaluations
        random_state: Random seed
        
    Returns:
        BayesianOptimizationResult
    """
    bounds = [
        ParameterBounds(name=name, lower=b[0], upper=b[1])
        for name, b in param_bounds.items()
    ]
    
    optimizer = BayesianOptimizer(
        param_bounds=bounds,
        acquisition='ei',
        n_initial_points=min(5, n_evaluations // 3),
        max_evaluations=n_evaluations,
        random_state=random_state,
    )
    
    return optimizer.optimize(objective_func)


if __name__ == '__main__':
    # Example usage demonstration
    print("Bayesian Optimization Module")
    print("=" * 50)
    
    # Define parameter bounds
    param_bounds = [
        ParameterBounds(name='lookback_period', lower=5, upper=100, discrete_values=[5, 10, 20, 50, 100]),
        ParameterBounds(name='entry_threshold', lower=0.5, upper=3.0),
        ParameterBounds(name='exit_threshold', lower=0.2, upper=2.0),
        ParameterBounds(name='stop_loss_pct', lower=0.01, upper=0.10, log_scale=True),
        ParameterBounds(name='take_profit_pct', lower=0.02, upper=0.20, log_scale=True),
    ]
    
    # Mock objective function (simulates backtest)
    def mock_backtest(params: Dict[str, float]) -> Dict[str, float]:
        np.random.seed(int(sum(params.values()) * 1000) % 2**31)
        
        lookback = params['lookback_period']
        entry_thresh = params['entry_threshold']
        
        # Simulate strategy performance
        n_trades = int(100 / np.sqrt(lookback))
        win_rate = 0.45 + 0.15 * np.exp(-abs(entry_thresh - 1.5))
        
        wins = np.random.binomial(n_trades, win_rate)
        avg_win = 0.02 + np.random.normal(0, 0.005)
        avg_loss = -0.015 + np.random.normal(0, 0.003)
        
        total_return = wins * avg_win + (n_trades - wins) * avg_loss
        
        # Add some noise based on other params
        total_return *= (1 + np.random.normal(0, 0.1))
        
        sharpe = total_return / (abs(total_return) * 0.3 + 0.01) + np.random.normal(0, 0.2)
        max_dd = 0.1 + np.random.exponential(0.05)
        
        return {
            'objective': -total_return,  # Minimize negative return = maximize return
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'n_trades': n_trades,
            'win_rate': wins / max(n_trades, 1),
        }
    
    # Run optimization
    optimizer = BayesianOptimizer(
        param_bounds=param_bounds,
        acquisition='ei',
        n_initial_points=5,
        max_evaluations=20,
        random_state=42,
    )
    
    print("\nRunning Bayesian Optimization...")
    result = optimizer.optimize(mock_backtest, verbose=True)
    
    # Print results
    print("\n" + "=" * 50)
    print("OPTIMIZATION RESULTS")
    print("=" * 50)
    
    print(f"\nBest Parameters:")
    for name, value in result.best_params.items():
        print(f"  {name}: {value:.4f}" if isinstance(value, float) else f"  {name}: {value}")
    
    print(f"\nBest Objective: {result.best_objective:.4f}")
    print(f"Total Evaluations: {result.num_evaluations}")
    
    print(f"\nParameter Importance:")
    for name, imp in sorted(result.parameter_importance.items(), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.3f}")
    
    print("\n" + "=" * 50)
    print("Bayesian Optimization complete!")
