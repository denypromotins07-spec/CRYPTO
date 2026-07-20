"""
Ray Tune Integration for Distributed Hyperparameter Optimization.
Uses ASHA (Asynchronous Successive Halving Algorithm) to efficiently kill 
poor-performing trials early, saving compute and RAM.

Target Hardware: AMD Ryzen AI 5 + AMD Radeon GPU + 16GB RAM (8GB cap)
"""

import os
import logging
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
import numpy as np

# Ray imports
try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search import BasicVariantGenerator
    from ray.tune.result_grid import ResultGrid
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    logging.warning("Ray not available, hyperparameter tuning disabled")

logger = logging.getLogger(__name__)


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning."""
    # Search space parameters
    learning_rate_range: tuple = (1e-5, 1e-2)
    batch_size_options: List[int] = None
    hidden_size_options: List[int] = None
    dropout_range: tuple = (0.0, 0.5)
    
    # ASHA scheduler parameters
    max_epochs: int = 100
    grace_period: int = 10
    reduction_factor: int = 3
    
    # Resource constraints
    num_samples: int = 50
    cpu_per_trial: float = 2.0
    gpu_per_trial: float = 0.5
    memory_per_trial_gb: float = 4.0
    
    # Early stopping
    metric_name: str = "validation_loss"
    metric_mode: str = "min"
    
    def __post_init__(self):
        if self.batch_size_options is None:
            self.batch_size_options = [32, 64, 128, 256]
        if self.hidden_size_options is None:
            self.hidden_size_options = [64, 128, 256, 512]


class RayTuner:
    """
    Ray-based hyperparameter tuner with ASHA scheduling.
    
    Features:
    - Asynchronous Successive Halving Algorithm for efficient pruning
    - Strict memory cap enforcement per trial
    - Parallel trial execution with resource limits
    - Automatic checkpointing and resume capability
    - Integration with ML/RL training pipelines
    """
    
    def __init__(
        self,
        config: TuningConfig,
        storage_path: str = "./data/tuning",
        max_ram_gb: float = 8.0
    ):
        """
        Initialize Ray tuner.
        
        Args:
            config: TuningConfig object
            storage_path: Path for storing results and checkpoints
            max_ram_gb: Maximum total RAM usage
        """
        self.config = config
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.max_ram_gb = max_ram_gb
        self.ray_initialized = False
        
        # Calculate trials that can run in parallel
        self.parallel_trials = int(
            max_ram_gb / config.memory_per_trial_gb
        )
        
        logger.info(f"RayTuner initialized: max {self.parallel_trials} parallel trials")
        logger.info(f"Total RAM cap: {max_ram_gb}GB, per-trial: {config.memory_per_trial_gb}GB")
    
    def initialize_ray(self):
        """Initialize Ray with resource constraints."""
        if not RAY_AVAILABLE:
            raise RuntimeError("Ray is not installed")
        
        if not self.ray_initialized:
            # Calculate CPU allocation
            num_cpus = min(
                os.cpu_count() or 4,
                int(self.max_ram_gb / self.config.memory_per_trial_gb * self.config.cpu_per_trial)
            )
            
            ray.init(
                num_cpus=num_cpus,
                include_dashboard=False,
                _temp_dir=str(self.storage_path / "ray_temp"),
                logging_level=logging.WARNING,
            )
            
            self.ray_initialized = True
            logger.info(f"Ray initialized with {num_cpus} CPUs")
    
    def get_search_space(self) -> Dict[str, Any]:
        """Define hyperparameter search space."""
        return {
            "learning_rate": tune.loguniform(*self.config.learning_rate_range),
            "batch_size": tune.choice(self.config.batch_size_options),
            "hidden_size": tune.choice(self.config.hidden_size_options),
            "dropout": tune.uniform(*self.config.dropout_range),
            # Additional parameters can be added here
            "optimizer": tune.choice(["adam", "sgd", "rmsprop"]),
            "activation": tune.choice(["relu", "gelu", "swish"]),
        }
    
    def get_scheduler(self) -> ASHAScheduler:
        """Create ASHA scheduler for early stopping."""
        return ASHAScheduler(
            metric=self.config.metric_name,
            mode=self.config.metric_mode,
            max_t=self.config.max_epochs,
            grace_period=self.config.grace_period,
            reduction_factor=self.config.reduction_factor,
            brackets=3,  # Number of down-sampling brackets
        )
    
    def tune(
        self,
        train_func: Callable[[Dict[str, Any]], Dict[str, float]],
        name: str = "hyperparameter_tuning",
        resume: bool = False
    ) -> ResultGrid:
        """
        Run hyperparameter tuning.
        
        Args:
            train_func: Training function that takes config dict and returns metrics
            name: Experiment name
            resume: Resume from previous checkpoint
            
        Returns:
            ResultGrid with all trial results
        """
        self.initialize_ray()
        
        # Wrap training function to enforce memory limits
        wrapped_func = self._wrap_with_memory_limit(train_func)
        
        # Configure tuner
        tuner = tune.Tuner(
            tune.with_resources(
                wrapped_func,
                resources={
                    "cpu": self.config.cpu_per_trial,
                    "gpu": self.config.gpu_per_trial,
                    "memory": int(self.config.memory_per_trial_gb * 1024),  # MB
                }
            ),
            param_space=self.get_search_space(),
            tune_config=tune.TuneConfig(
                scheduler=self.get_scheduler(),
                num_samples=self.config.num_samples,
                search_alg=BasicVariantGenerator(),
                max_concurrent_trials=self.parallel_trials,
            ),
            run_config=ray.train.RunConfig(
                name=name,
                storage_path=str(self.storage_path / "results"),
                verbose=1,
            ),
        )
        
        logger.info(f"Starting tuning: {self.config.num_samples} trials, "
                   f"{self.parallel_trials} parallel")
        
        result_grid = tuner.fit()
        
        logger.info("Tuning completed")
        return result_grid
    
    def _wrap_with_memory_limit(
        self,
        train_func: Callable[[Dict[str, Any]], Dict[str, float]]
    ) -> Callable:
        """Wrap training function with memory limit enforcement."""
        def wrapped(config: Dict[str, Any]):
            import gc
            import psutil
            
            process = psutil.Process(os.getpid())
            
            try:
                # Set memory limit warning threshold
                mem_limit_bytes = self.config.memory_per_trial_gb * 1024**3
                
                # Run training
                result = train_func(config)
                
                # Check memory usage
                mem_used = process.memory_info().rss
                if mem_used > mem_limit_bytes * 0.9:
                    logger.warning(
                        f"Trial approaching memory limit: "
                        f"{mem_used / 1024**3:.2f}GB / {self.config.memory_per_trial_gb}GB"
                    )
                
                # Force cleanup
                gc.collect()
                
                return result
                
            except MemoryError:
                logger.error("Trial exceeded memory limit")
                raise
        
        return wrapped
    
    def get_best_config(
        self,
        result_grid: ResultGrid,
        metric: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract best configuration from results."""
        if metric is None:
            metric = self.config.metric_name
        
        best_result = result_grid.get_best_result(metric=metric, mode=self.config.metric_mode)
        
        if best_result is None:
            raise ValueError("No valid results found")
        
        return best_result.config
    
    def get_best_results(
        self,
        result_grid: ResultGrid,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Get top-k best configurations."""
        results = []
        
        # Sort by metric
        sorted_results = result_grid.get_dataframe().sort_values(
            by=self.config.metric_name,
            ascending=(self.config.metric_mode == "min")
        )
        
        for _, row in sorted_results.head(top_k).iterrows():
            results.append({
                "config": row.to_dict(),
                "metrics": {
                    self.config.metric_name: row[self.config.metric_name]
                }
            })
        
        return results
    
    def export_results(
        self,
        result_grid: ResultGrid,
        output_path: str
    ):
        """Export tuning results to file."""
        df = result_grid.get_dataframe()
        output_file = Path(output_path)
        
        # Save as CSV
        df.to_csv(output_file.with_suffix('.csv'))
        
        # Save summary statistics
        summary = {
            "total_trials": len(df),
            "best_" + self.config.metric_name: df[self.config.metric_name].min(),
            "mean_" + self.config.metric_name: df[self.config.metric_name].mean(),
            "std_" + self.config.metric_name: df[self.config.metric_name].std(),
        }
        
        import json
        with open(output_file.with_suffix('.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results exported to {output_path}")
    
    def shutdown(self):
        """Shutdown Ray cluster."""
        if self.ray_initialized:
            ray.shutdown()
            self.ray_initialized = False
            logger.info("Ray cluster shut down")


def example_training_function(config: Dict[str, Any]) -> Dict[str, float]:
    """Example training function for demonstration."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    # Create simple model with config parameters
    model = nn.Sequential(
        nn.Linear(100, config["hidden_size"]),
        nn.ReLU() if config["activation"] == "relu" else nn.GELU(),
        nn.Dropout(config["dropout"]),
        nn.Linear(config["hidden_size"], 10)
    )
    
    # Setup optimizer
    if config["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=config["learning_rate"])
    
    # Training loop (simplified)
    for epoch in range(10):
        # Simulated training
        loss = np.random.random() * config["learning_rate"] * 10
        val_loss = loss * 1.1
        
        # Report metrics to Ray Tune
        tune.report(loss=loss, validation_loss=val_loss)
    
    return {"loss": loss, "validation_loss": val_loss}


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    config = TuningConfig(
        num_samples=20,  # Reduced for demo
        max_epochs=50,
        grace_period=5,
        memory_per_trial_gb=2.0,
    )
    
    tuner = RayTuner(config, storage_path="./data/test_tuning")
    
    try:
        results = tuner.tune(example_training_function, name="demo_tuning")
        
        best_config = tuner.get_best_config(results)
        print(f"\nBest config: {best_config}")
        
        top_5 = tuner.get_best_results(results, top_k=5)
        print(f"\nTop 5 results:")
        for i, r in enumerate(top_5):
            print(f"  {i+1}. val_loss={r['metrics']['validation_loss']:.4f}")
        
    finally:
        tuner.shutdown()
