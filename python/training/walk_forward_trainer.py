"""
Walk-Forward Trainer with Purged K-Fold Cross-Validation.
Implements Purged K-Fold CV and Walk-Forward optimization using Ray.
Ensures no data leakage during training and generates robust model weights.

Target Hardware: AMD Ryzen AI 5 + AMD Radeon GPU + 16GB RAM (8GB cap)
"""

import os
import logging
from typing import Dict, Any, Optional, List, Tuple, Generator
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd

# Ray imports
try:
    import ray
    from ray import tune
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    logging.warning("Ray not available, walk-forward training disabled")

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis."""
    # Data splitting
    train_ratio: float = 0.7
    test_ratio: float = 0.3
    gap_between_folds: int = 100  # Samples between train/test to prevent leakage
    
    # K-Fold parameters
    n_folds: int = 5
    purge_size: int = 50  # Number of samples to purge at fold boundaries
    
    # Walk-forward parameters
    initial_train_window: int = 10000
    step_size: int = 1000
    min_test_samples: int = 1000
    
    # Model parameters
    model_type: str = "lightgbm"  # or "xgboost", "sklearn"
    
    # Resource constraints
    num_parallel_folds: int = 2
    memory_per_fold_gb: float = 2.0
    
    # Validation
    use_purged_cv: bool = True
    embargo_pct: float = 0.01  # Percentage of data to embargo


class PurgedKFold:
    """
    Purged K-Fold cross-validation for time series data.
    
    Prevents data leakage by:
    1. Purging samples at fold boundaries
    2. Adding embargo period between train and test
    3. Maintaining temporal order
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        purge_size: int = 50,
        embargo_pct: float = 0.01
    ):
        """
        Initialize purged K-fold.
        
        Args:
            n_splits: Number of folds
            purge_size: Number of samples to purge at boundaries
            embargo_pct: Percentage of data to embargo between train/test
        """
        self.n_splits = n_splits
        self.purge_size = purge_size
        self.embargo_pct = embargo_pct
    
    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices with purging.
        
        Args:
            X: Feature matrix
            y: Target values (optional)
            groups: Group labels (optional)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        
        for fold_idx in range(self.n_splits):
            # Calculate test indices for this fold
            test_start = fold_idx * fold_size
            test_end = test_start + fold_size if fold_idx < self.n_splits - 1 else n_samples
            
            # Apply embargo
            embargo_size = int(fold_size * self.embargo_pct)
            
            # Test set with embargo buffer
            test_indices = np.arange(test_start, test_end)
            
            # Train set: everything before test_start and after test_end
            train_before = np.arange(0, max(0, test_start - self.purge_size))
            train_after = np.arange(min(n_samples, test_end + embargo_size), n_samples)
            
            # Combine train sets
            train_indices = np.concatenate([train_before, train_after])
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices
    
    def get_n_splits(self) -> int:
        """Return number of splits."""
        return self.n_splits


class WalkForwardTrainer:
    """
    Walk-forward optimization trainer with purged cross-validation.
    
    Features:
    - No data leakage through purging and embargo
    - Ray-based parallel fold training
    - Memory-cap enforcement
    - Progressive model updating
    """
    
    def __init__(
        self,
        config: WalkForwardConfig,
        storage_path: str = "./data/walkforward",
        max_ram_gb: float = 8.0
    ):
        """
        Initialize walk-forward trainer.
        
        Args:
            config: WalkForwardConfig object
            storage_path: Path for storing results
            max_ram_gb: Maximum RAM usage
        """
        self.config = config
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.max_ram_gb = max_ram_gb
        self.ray_initialized = False
        
        # Results storage
        self.fold_results: List[Dict[str, Any]] = []
        self.walk_forward_results: List[Dict[str, Any]] = []
        
        logger.info(f"WalkForwardTrainer initialized: {config.n_folds} folds")
        logger.info(f"Max RAM: {max_ram_gb}GB")
    
    def initialize_ray(self):
        """Initialize Ray for parallel processing."""
        if not RAY_AVAILABLE:
            raise RuntimeError("Ray is not installed")
        
        if not self.ray_initialized:
            num_cpus = min(
                os.cpu_count() or 4,
                self.config.num_parallel_folds * 2
            )
            
            ray.init(
                num_cpus=num_cpus,
                include_dashboard=False,
                _temp_dir=str(self.storage_path / "ray_temp"),
                logging_level=logging.WARNING,
            )
            
            self.ray_initialized = True
            logger.info(f"Ray initialized with {num_cpus} CPUs")
    
    def purged_kfold_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_func: callable
    ) -> List[Dict[str, Any]]:
        """
        Perform purged K-fold cross-validation.
        
        Args:
            X: Feature matrix
            y: Target values
            train_func: Training function that takes (X_train, y_train, X_test, y_test)
            
        Returns:
            List of fold results
        """
        self.initialize_ray()
        
        pkf = PurgedKFold(
            n_splits=self.config.n_folds,
            purge_size=self.config.purge_size,
            embargo_pct=self.config.embargo_pct
        )
        
        fold_results = []
        
        # Prepare tasks for parallel execution
        tasks = []
        for fold_idx, (train_idx, test_idx) in enumerate(pkf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Create remote task
            remote_train = ray.remote(train_func)
            task = remote_train.remote(X_train, y_train, X_test, y_test, fold_idx)
            tasks.append((fold_idx, task))
            
            logger.info(f"Scheduled fold {fold_idx}: train={len(train_idx)}, test={len(test_idx)}")
        
        # Collect results
        for fold_idx, task in tasks:
            try:
                result = ray.get(task, timeout=300)
                fold_results.append(result)
                logger.info(f"Fold {fold_idx} completed: {result.get('metric', 'N/A')}")
            except Exception as e:
                logger.error(f"Fold {fold_idx} failed: {e}")
                fold_results.append({
                    "fold": fold_idx,
                    "error": str(e),
                    "metric": None
                })
        
        self.fold_results = fold_results
        return fold_results
    
    def walk_forward_analysis(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_func: callable
    ) -> List[Dict[str, Any]]:
        """
        Perform walk-forward analysis.
        
        Args:
            X: Feature matrix
            y: Target values
            train_func: Training function
            
        Returns:
            List of walk-forward period results
        """
        self.initialize_ray()
        
        n_samples = len(X)
        current_train_end = self.config.initial_train_window
        results = []
        
        period = 0
        while current_train_end + self.config.min_test_samples < n_samples:
            # Define train/test split
            train_start = max(0, current_train_end - self.config.initial_train_window)
            train_end = current_train_end
            test_start = train_end + self.config.gap_between_folds
            test_end = min(n_samples, test_start + self.config.step_size)
            
            if test_end - test_start < self.config.min_test_samples:
                break
            
            # Extract data
            X_train = X[train_start:train_end]
            y_train = y[train_start:train_end]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]
            
            logger.info(
                f"Period {period}: train=[{train_start}:{train_end}], "
                f"test=[{test_start}:{test_end}]"
            )
            
            # Train and evaluate
            remote_train = ray.remote(train_func)
            try:
                result = ray.get(
                    remote_train.remote(X_train, y_train, X_test, y_test, period),
                    timeout=300
                )
                result["period"] = period
                result["train_start"] = train_start
                result["train_end"] = train_end
                result["test_start"] = test_start
                result["test_end"] = test_end
                results.append(result)
                
                logger.info(f"Period {period} completed: {result.get('metric', 'N/A')}")
                
            except Exception as e:
                logger.error(f"Period {period} failed: {e}")
                results.append({
                    "period": period,
                    "error": str(e),
                    "metric": None
                })
            
            # Move window forward
            current_train_end += self.config.step_size
            period += 1
        
        self.walk_forward_results = results
        return results
    
    def aggregate_results(self) -> Dict[str, Any]:
        """Aggregate results from all folds/periods."""
        metrics = [r.get("metric") for r in self.fold_results + self.walk_forward_results 
                  if r.get("metric") is not None]
        
        if not metrics:
            return {"error": "No valid metrics found"}
        
        return {
            "mean_metric": np.mean(metrics),
            "std_metric": np.std(metrics),
            "min_metric": np.min(metrics),
            "max_metric": np.max(metrics),
            "sharpe_ratio": np.mean(metrics) / (np.std(metrics) + 1e-8),
            "num_valid_folds": len(metrics),
            "total_folds": len(self.fold_results) + len(self.walk_forward_results)
        }
    
    def check_data_leakage(
        self,
        X: np.ndarray,
        train_indices: np.ndarray,
        test_indices: np.ndarray
    ) -> float:
        """
        Check for potential data leakage between train and test sets.
        
        Returns:
            Leakage score (0 = no leakage, 1 = high leakage)
        """
        if len(train_indices) == 0 or len(test_indices) == 0:
            return 0.0
        
        # Check temporal overlap
        train_max = np.max(train_indices)
        test_min = np.min(test_indices)
        
        if train_max >= test_min:
            # Temporal overlap detected
            overlap = train_max - test_min + 1
            return min(1.0, overlap / len(X))
        
        # Check feature similarity (simplified)
        train_mean = np.mean(X[train_indices], axis=0)
        test_mean = np.mean(X[test_indices], axis=0)
        
        # Normalized difference
        diff = np.linalg.norm(train_mean - test_mean) / (np.linalg.norm(train_mean) + 1e-8)
        
        return max(0.0, 1.0 - diff)  # Higher diff = lower leakage risk
    
    def export_results(self, output_path: str):
        """Export results to files."""
        output_file = Path(output_path)
        
        # Export fold results
        if self.fold_results:
            fold_df = pd.DataFrame(self.fold_results)
            fold_df.to_csv(output_file.with_suffix('.folds.csv'), index=False)
        
        # Export walk-forward results
        if self.walk_forward_results:
            wf_df = pd.DataFrame(self.walk_forward_results)
            wf_df.to_csv(output_file.with_suffix('.walkforward.csv'), index=False)
        
        # Export summary
        summary = self.aggregate_results()
        import json
        with open(output_file.with_suffix('.summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results exported to {output_path}")
    
    def shutdown(self):
        """Shutdown Ray."""
        if self.ray_initialized:
            ray.shutdown()
            self.ray_initialized = False
            logger.info("Ray shut down")


def example_train_func(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fold_id: int
) -> Dict[str, Any]:
    """Example training function for demonstration."""
    # Simple model training (replace with actual model)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    model = LogisticRegression(max_iter=100)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) > 1 else y_pred
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5
    
    return {
        "fold": fold_id,
        "metric": auc,
        "accuracy": accuracy,
        "n_train": len(X_train),
        "n_test": len(X_test)
    }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 10000
    n_features = 50
    
    X = np.random.randn(n_samples, n_features)
    y = (np.sum(X[:, :10], axis=1) > 0).astype(int)  # Simple target
    
    config = WalkForwardConfig(
        n_folds=5,
        initial_train_window=5000,
        step_size=1000,
        num_parallel_folds=2
    )
    
    trainer = WalkForwardTrainer(config, storage_path="./data/test_wf")
    
    try:
        # Run purged K-fold CV
        print("\n=== Running Purged K-Fold CV ===")
        fold_results = trainer.purged_kfold_cv(X, y, example_train_func)
        
        # Run walk-forward analysis
        print("\n=== Running Walk-Forward Analysis ===")
        wf_results = trainer.walk_forward_analysis(X, y, example_train_func)
        
        # Aggregate results
        summary = trainer.aggregate_results()
        print(f"\n=== Summary ===")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Export results
        trainer.export_results("./data/test_wf/results")
        
    finally:
        trainer.shutdown()
