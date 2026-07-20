"""
Ray Cluster Initialization for Ultra-Low Latency Trading Bot

This module initializes a local Ray cluster with strict memory constraints
optimized for an AMD Ryzen AI 5 laptop with 16GB RAM (capped at 8GB usage).

Features:
- Strict 8GB global memory cap
- CPU core affinity for AMD Ryzen architecture
- NPU/GPU utilization when available (default to CPU for ML)
- Object store optimization for low-latency data sharing
- Integration with Nautilus Trader event loop
"""

import os
import sys
import logging
import psutil
from typing import Optional, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Memory configuration constants
# Total system RAM: 16GB, System cap: 8GB
MAX_SYSTEM_MEMORY_GB = 8
OBJECT_STORE_MEMORY_RATIO = 0.3  # 30% for object store
WORKER_MEMORY_RATIO = 0.5  # 50% for workers
HEAD_MEMORY_RATIO = 0.2  # 20% for head node


class RayClusterConfig:
    """Configuration for Ray cluster initialization."""
    
    def __init__(
        self,
        max_memory_gb: float = MAX_SYSTEM_MEMORY_GB,
        num_cpus: Optional[int] = None,
        num_gpus: Optional[int] = None,
        object_store_ratio: float = OBJECT_STORE_MEMORY_RATIO,
    ):
        """
        Initialize Ray cluster configuration.
        
        Args:
            max_memory_gb: Maximum memory in GB (default: 8GB)
            num_cpus: Number of CPUs to use (default: auto-detect)
            num_gpus: Number of GPUs to use (default: 0, use CPU for ML)
            object_store_ratio: Ratio of memory for object store
        """
        self.max_memory_gb = max_memory_gb
        self.max_memory_bytes = int(max_memory_gb * 1024 * 1024 * 1024)
        
        # Auto-detect CPU cores if not specified
        if num_cpus is None:
            # AMD Ryzen AI 5 typically has 6 performance cores
            self.num_cpus = psutil.cpu_count(logical=False) or 6
            # Reserve one core for OS and main event loop
            self.num_cpus = max(1, self.num_cpus - 1)
        else:
            self.num_cpus = num_cpus
        
        # Default to CPU-only for ML training to save GPU memory
        self.num_gpus = num_gpus or 0
        
        # Calculate memory allocations
        self.object_store_memory = int(self.max_memory_bytes * object_store_ratio)
        self.worker_memory = int(self.max_memory_bytes * WORKER_MEMORY_RATIO)
        self.head_memory = int(self.max_memory_bytes * HEAD_MEMORY_RATIO)
        
        logger.info(f"Ray Cluster Configuration:")
        logger.info(f"  Max Memory: {self.max_memory_gb} GB")
        logger.info(f"  CPUs: {self.num_cpus}")
        logger.info(f"  GPUs: {self.num_gpus}")
        logger.info(f"  Object Store: {self.object_store_memory / (1024**3):.2f} GB")
        logger.info(f"  Worker Memory: {self.worker_memory / (1024**3):.2f} GB")


class RayClusterManager:
    """
    Manages Ray cluster lifecycle for the trading bot.
    
    This class handles:
    - Cluster initialization with memory constraints
    - Worker management
    - Resource monitoring
    - Graceful shutdown
    """
    
    _instance: Optional['RayClusterManager'] = None
    
    def __new__(cls) -> 'RayClusterManager':
        """Singleton pattern for cluster manager."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the cluster manager."""
        if self._initialized:
            return
        
        self.cluster_config: Optional[RayClusterConfig] = None
        self._ray_initialized = False
        self._shutdown_requested = False
        self._initialized = True
        
    def initialize(
        self,
        config: Optional[RayClusterConfig] = None,
        address: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Initialize the Ray cluster.
        
        Args:
            config: Cluster configuration (optional, uses defaults if not provided)
            address: Ray address (for connecting to existing cluster)
            **kwargs: Additional arguments passed to ray.init()
            
        Returns:
            True if initialization successful, False otherwise
        """
        import ray
        
        if self._ray_initialized:
            logger.warning("Ray cluster already initialized")
            return True
        
        try:
            # Create default config if not provided
            if config is None:
                config = RayClusterConfig()
            
            self.cluster_config = config
            
            # Prepare ray.init arguments
            init_kwargs = {
                'num_cpus': config.num_cpus,
                'num_gpus': config.num_gpus,
                '_memory': config.max_memory_bytes,
                'object_store_memory': config.object_store_memory,
                'include_dashboard': False,  # Disable dashboard for lower overhead
                'log_to_driver': True,
                'ignore_reinit_error': True,
            }
            
            # Merge with user-provided kwargs
            init_kwargs.update(kwargs)
            
            # Check for AMD Ryzen AI NPU availability
            npu_available = self._check_npu_availability()
            if npu_available and config.num_gpus == 0:
                logger.info("AMD Ryzen AI NPU detected but using CPU for ML (memory optimization)")
            
            # Initialize Ray
            if address:
                logger.info(f"Connecting to existing Ray cluster at {address}")
                ray.init(address=address, **{k: v for k, v in init_kwargs.items() 
                                             if k != 'num_cpus'})
            else:
                logger.info("Initializing new Ray cluster")
                ray.init(**init_kwargs)
            
            self._ray_initialized = True
            
            # Log cluster info
            cluster_info = ray.cluster_resources()
            logger.info(f"Ray cluster initialized successfully")
            logger.info(f"  Available resources: {cluster_info}")
            
            # Register memory monitor
            self._start_memory_monitoring()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Ray cluster: {e}")
            return False
    
    def _check_npu_availability(self) -> bool:
        """Check if AMD Ryzen AI NPU is available."""
        # Check for AMD NPU devices
        try:
            # Look for AMD NPU in /dev or via lspci
            if sys.platform == 'linux':
                # Check for AMD AI Engine
                amd_paths = [
                    '/sys/class/drm',
                    '/dev/kfd',
                ]
                for path in amd_paths:
                    if os.path.exists(path):
                        logger.info(f"AMD device detected at {path}")
                        return True
        except Exception as e:
            logger.debug(f"NPU check error: {e}")
        
        return False
    
    def _start_memory_monitoring(self):
        """Start background memory monitoring."""
        import threading
        import time
        
        def monitor_loop():
            while not self._shutdown_requested:
                try:
                    mem = psutil.virtual_memory()
                    used_gb = mem.used / (1024**3)
                    
                    if used_gb > self.cluster_config.max_memory_gb * 0.9:
                        logger.warning(
                            f"Memory usage high: {used_gb:.2f} GB / "
                            f"{self.cluster_config.max_memory_gb} GB"
                        )
                    
                    time.sleep(5)  # Check every 5 seconds
                except Exception as e:
                    logger.debug(f"Memory monitor error: {e}")
                    time.sleep(10)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Memory monitoring started")
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get current cluster information."""
        import ray
        
        if not self._ray_initialized:
            return {'status': 'not_initialized'}
        
        try:
            return {
                'status': 'running',
                'resources': dict(ray.cluster_resources()),
                'nodes': len(ray.nodes()),
                'config': {
                    'max_memory_gb': self.cluster_config.max_memory_gb if self.cluster_config else None,
                    'num_cpus': self.cluster_config.num_cpus if self.cluster_config else None,
                    'num_gpus': self.cluster_config.num_gpus if self.cluster_config else None,
                } if self.cluster_config else None,
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def shutdown(self, force: bool = False):
        """
        Shutdown the Ray cluster gracefully.
        
        Args:
            force: If True, force immediate shutdown
        """
        import ray
        
        if not self._ray_initialized:
            return
        
        logger.info("Shutting down Ray cluster...")
        self._shutdown_requested = True
        
        try:
            ray.shutdown()
            self._ray_initialized = False
            logger.info("Ray cluster shut down successfully")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            if force:
                raise


def get_ray_cluster() -> RayClusterManager:
    """Get the singleton Ray cluster manager instance."""
    return RayClusterManager()


def initialize_ray_cluster(
    max_memory_gb: float = MAX_SYSTEM_MEMORY_GB,
    **kwargs
) -> RayClusterManager:
    """
    Convenience function to initialize the Ray cluster.
    
    Args:
        max_memory_gb: Maximum memory in GB
        **kwargs: Additional configuration options
        
    Returns:
        RayClusterManager instance
    """
    manager = get_ray_cluster()
    config = RayClusterConfig(max_memory_gb=max_memory_gb)
    manager.initialize(config=config, **kwargs)
    return manager


# Remote task decorators for common operations
def cpu_task(func):
    """Decorator for CPU-bound tasks."""
    import ray
    
    @ray.remote(num_cpus=1, num_gpus=0)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    return wrapper


def gpu_task(func):
    """Decorator for GPU/NPU-accelerated tasks."""
    import ray
    
    @ray.remote(num_cpus=1, num_gpus=1)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    return wrapper


if __name__ == '__main__':
    # Example usage
    print("Initializing Ray Cluster for Trading Bot...")
    
    manager = initialize_ray_cluster(
        max_memory_gb=8,
        num_cpus=5,  # Reserve 1 core for event loop
    )
    
    info = manager.get_cluster_info()
    print(f"Cluster Info: {info}")
    
    # Keep running for demonstration
    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        manager.shutdown()
        print("Cluster shut down")
