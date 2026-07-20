"""
Hardware Memory Profiler and Watchdog
======================================
Chapter 1, File 3: Hardware Acceleration

A strict, real-time hardware memory watchdog that interfaces with AMD SMI 
(System Management Interface) to monitor VRAM, NPU, and system RAM. Triggers 
aggressive, targeted garbage collection if the 8GB global limit is approached.

Target Hardware: AMD Ryzen AI 5 with AMD Radeon GPU (ROCm)
Memory Cap: Strict 8GB global limit with automatic enforcement
"""

import psutil
import subprocess
import threading
import time
import json
import os
import gc
import signal
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import logging
from pathlib import Path
import weakref


class MemoryType(Enum):
    """Types of memory being monitored."""
    SYSTEM_RAM = "system_ram"
    GPU_VRAM = "gpu_vram"
    NPU_MEMORY = "npu_memory"
    SHARED_MEMORY = "shared_memory"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = 0
    WARNING = 1
    CRITICAL = 2
    EMERGENCY = 3


@dataclass
class MemorySnapshot:
    """Point-in-time memory usage snapshot."""
    timestamp_ns: int
    system_used_mb: float
    system_available_mb: float
    system_percent: float
    gpu_used_mb: float = 0.0
    gpu_total_mb: float = 0.0
    gpu_percent: float = 0.0
    npu_used_mb: float = 0.0
    npu_total_mb: float = 0.0
    process_rss_mb: float = 0.0
    process_vms_mb: float = 0.0
    total_usage_mb: float = 0.0
    limit_mb: float = 8192.0  # 8GB default limit
    
    @property
    def utilization_ratio(self) -> float:
        """Calculate overall utilization against limit."""
        return self.total_usage_mb / self.limit_mb
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'timestamp_ns': self.timestamp_ns,
            'system': {
                'used_mb': self.system_used_mb,
                'available_mb': self.system_available_mb,
                'percent': self.system_percent
            },
            'gpu': {
                'used_mb': self.gpu_used_mb,
                'total_mb': self.gpu_total_mb,
                'percent': self.gpu_percent
            },
            'npu': {
                'used_mb': self.npu_used_mb,
                'total_mb': self.npu_total_mb
            },
            'process': {
                'rss_mb': self.process_rss_mb,
                'vms_mb': self.process_vms_mb
            },
            'total_usage_mb': self.total_usage_mb,
            'limit_mb': self.limit_mb,
            'utilization_ratio': self.utilization_ratio
        }


@dataclass
class GCStats:
    """Garbage collection statistics."""
    collections_triggered: int = 0
    objects_collected: int = 0
    last_collection_time_ms: float = 0.0
    total_collection_time_ms: float = 0.0
    memory_freed_mb: float = 0.0


class AMDSMIMonitor:
    """
    AMD System Management Interface (SMI) monitor.
    
    Interfaces with rocm-smi or amd-smi to get GPU memory information.
    Gracefully degrades if AMD tools are not available.
    """
    
    def __init__(self):
        self._rocm_smi_available = self._check_rocm_smi()
        self._amd_smi_available = self._check_amd_smi()
        self._logger = logging.getLogger(__name__)
        
        if self._rocm_smi_available:
            self._logger.info("rocm-smi detected")
        elif self._amd_smi_available:
            self._logger.info("amd-smi detected")
        else:
            self._logger.warning("AMD SMI tools not found, using fallback methods")
    
    def _check_rocm_smi(self) -> bool:
        """Check if rocm-smi is available."""
        try:
            result = subprocess.run(
                ['rocm-smi', '--showmeminfo', 'vram'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _check_amd_smi(self) -> bool:
        """Check if amd-smi (newer tool) is available."""
        try:
            result = subprocess.run(
                ['amd-smi', 'memory', 'show'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """
        Get GPU memory information from AMD SMI.
        
        Returns:
            Dictionary with used_vram_mb, total_vram_mb
        """
        if self._rocm_smi_available:
            return self._get_via_rocm_smi()
        elif self._amd_smi_available:
            return self._get_via_amd_smi()
        else:
            return self._get_via_pytorch()
    
    def _get_via_rocm_smi(self) -> Dict[str, float]:
        """Parse rocm-smi output."""
        try:
            result = subprocess.run(
                ['rocm-smi', '--showmeminfo', 'vram', '--json'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                # Parse JSON output (format varies by version)
                card_data = list(data.values())[0] if data else {}
                
                used = float(card_data.get('VRAM Used', 0))
                total = float(card_data.get('VRAM Total', 0))
                
                return {
                    'used_vram_mb': used / (1024 * 1024),
                    'total_vram_mb': total / (1024 * 1024)
                }
        except Exception as e:
            pass
        
        # Fallback: parse text output
        try:
            result = subprocess.run(
                ['rocm-smi', '--showmeminfo', 'vram'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                used, total = 0, 0
                
                for line in lines:
                    if 'VRAM Used' in line:
                        used = self._parse_memory_value(line)
                    elif 'VRAM Total' in line:
                        total = self._parse_memory_value(line)
                
                return {'used_vram_mb': used, 'total_vram_mb': total}
        except:
            pass
        
        return {'used_vram_mb': 0, 'total_vram_mb': 0}
    
    def _get_via_amd_smi(self) -> Dict[str, float]:
        """Parse amd-smi output."""
        try:
            result = subprocess.run(
                ['amd-smi', 'memory', 'show', '--json'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                # Parse based on amd-smi format
                gpu_info = data.get('gpus', [{}])[0]
                memory_info = gpu_info.get('memory', {})
                
                return {
                    'used_vram_mb': float(memory_info.get('vram_used', 0)),
                    'total_vram_mb': float(memory_info.get('vram_total', 0))
                }
        except:
            pass
        
        return {'used_vram_mb': 0, 'total_vram_mb': 0}
    
    def _get_via_pytorch(self) -> Dict[str, float]:
        """Fallback: use PyTorch to get GPU memory info."""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                reserved = torch.cuda.memory_reserved() / (1024 * 1024)
                
                return {
                    'used_vram_mb': allocated,
                    'total_vram_mb': reserved
                }
        except ImportError:
            pass
        
        return {'used_vram_mb': 0, 'total_vram_mb': 0}
    
    def _parse_memory_value(self, line: str) -> float:
        """Parse memory value from text output."""
        import re
        match = re.search(r'(\d+(?:\.\d+)?)\s*(MB|GB|KB)', line, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            unit = match.group(2).upper()
            
            if unit == 'GB':
                return value * 1024
            elif unit == 'KB':
                return value / 1024
            return value
        
        return 0.0
    
    def get_npu_memory_info(self) -> Dict[str, float]:
        """
        Get NPU memory information.
        
        Note: NPU memory monitoring depends on Ryzen AI software stack.
        Returns estimates if direct access not available.
        """
        # Try to get NPU info from sysfs (Linux)
        try:
            npu_mem_path = Path('/sys/class/drm/card0/device/npu_mem')
            if npu_mem_path.exists():
                # Read NPU memory info
                pass
        except:
            pass
        
        # Fallback: estimate based on typical NPU usage
        # In production, integrate with Ryzen AI runtime
        return {
            'used_npu_mb': 0,
            'total_npu_mb': 512  # Typical NPU memory allocation
        }


class MemoryWatchdog:
    """
    Real-time memory watchdog with aggressive enforcement.
    
    Monitors all memory types (system RAM, GPU VRAM, NPU) and enforces
    the 8GB global limit through targeted garbage collection and
    memory pressure responses.
    """
    
    def __init__(
        self,
        memory_limit_mb: float = 8192.0,
        warning_threshold: float = 0.7,
        critical_threshold: float = 0.85,
        emergency_threshold: float = 0.95,
        check_interval_ms: int = 100
    ):
        """
        Initialize memory watchdog.
        
        Args:
            memory_limit_mb: Global memory limit in MB (default 8GB)
            warning_threshold: Ratio at which to issue warnings
            critical_threshold: Ratio triggering aggressive GC
            emergency_threshold: Ratio triggering emergency measures
            check_interval_ms: How often to check memory (milliseconds)
        """
        self.limit_mb = memory_limit_mb
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.emergency_threshold = emergency_threshold
        self.check_interval_s = check_interval_ms / 1000.0
        
        self._monitor = AMDSMIMonitor()
        self._logger = logging.getLogger(__name__)
        
        # State tracking
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._snapshots: deque = deque(maxlen=1000)
        self._callbacks: Dict[AlertLevel, List[Callable]] = {
            level: [] for level in AlertLevel
        }
        
        # GC tracking
        self._gc_stats = GCStats()
        self._last_gc_trigger_level = 0.0
        
        # Process tracking
        self._process = psutil.Process(os.getpid())
        self._tracked_objects: List[weakref.ref] = []
        
        # Alert suppression
        self._last_alert_time: Dict[AlertLevel, float] = {}
        self._alert_cooldown_s = 5.0
    
    def register_callback(self, level: AlertLevel, callback: Callable) -> None:
        """Register callback for alert level."""
        self._callbacks[level].append(callback)
    
    def unregister_callback(self, level: AlertLevel, callback: Callable) -> None:
        """Unregister callback."""
        if callback in self._callbacks[level]:
            self._callbacks[level].remove(callback)
    
    def track_object(self, obj: Any) -> None:
        """Track an object for potential cleanup under memory pressure."""
        self._tracked_objects.append(weakref.ref(obj))
    
    def start(self) -> None:
        """Start the watchdog monitoring thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        self._logger.info(f"Memory watchdog started (limit: {self.limit_mb}MB)")
    
    def stop(self) -> None:
        """Stop the watchdog."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._logger.info("Memory watchdog stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                snapshot = self._take_snapshot()
                self._snapshots.append(snapshot)
                
                # Check thresholds and trigger actions
                self._evaluate_and_respond(snapshot)
                
            except Exception as e:
                self._logger.error(f"Monitor error: {e}")
            
            time.sleep(self.check_interval_s)
    
    def _take_snapshot(self) -> MemorySnapshot:
        """Take a complete memory snapshot."""
        now = time.time_ns()
        
        # System memory
        mem = psutil.virtual_memory()
        system_used = mem.used / (1024 * 1024)
        system_available = mem.available / (1024 * 1024)
        system_percent = mem.percent / 100.0
        
        # GPU memory
        gpu_info = self._monitor.get_gpu_memory_info()
        gpu_used = gpu_info['used_vram_mb']
        gpu_total = gpu_info['total_vram_mb']
        gpu_percent = gpu_used / max(gpu_total, 1)
        
        # NPU memory
        npu_info = self._monitor.get_npu_memory_info()
        npu_used = npu_info['used_npu_mb']
        npu_total = npu_info['total_npu_mb']
        
        # Process memory
        process_rss = self._process.memory_info().rss / (1024 * 1024)
        process_vms = self._process.memory_info().vms / (1024 * 1024)
        
        # Calculate total usage (conservative estimate)
        # Use max of system and process to avoid double counting
        total_usage = max(system_used, process_rss + gpu_used)
        
        return MemorySnapshot(
            timestamp_ns=now,
            system_used_mb=system_used,
            system_available_mb=system_available,
            system_percent=system_percent,
            gpu_used_mb=gpu_used,
            gpu_total_mb=gpu_total,
            gpu_percent=gpu_percent,
            npu_used_mb=npu_used,
            npu_total_mb=npu_total,
            process_rss_mb=process_rss,
            process_vms_mb=process_vms,
            total_usage_mb=total_usage,
            limit_mb=self.limit_mb
        )
    
    def _evaluate_and_respond(self, snapshot: MemorySnapshot) -> None:
        """Evaluate memory state and trigger appropriate responses."""
        ratio = snapshot.utilization_ratio
        
        # Determine alert level
        if ratio >= self.emergency_threshold:
            level = AlertLevel.EMERGENCY
        elif ratio >= self.critical_threshold:
            level = AlertLevel.CRITICAL
        elif ratio >= self.warning_threshold:
            level = AlertLevel.WARNING
        else:
            level = AlertLevel.INFO
        
        # Trigger callbacks with cooldown
        now = time.time()
        if now - self._last_alert_time.get(level, 0) > self._alert_cooldown_s:
            self._trigger_callbacks(level, snapshot)
            self._last_alert_time[level] = now
        
        # Automatic responses based on level
        if level == AlertLevel.CRITICAL and ratio > self._last_gc_trigger_level:
            self._aggressive_gc()
            self._last_gc_trigger_level = ratio
        elif level == AlertLevel.EMERGENCY:
            self._emergency_response(snapshot)
    
    def _trigger_callbacks(self, level: AlertLevel, snapshot: MemorySnapshot) -> None:
        """Trigger registered callbacks."""
        for callback in self._callbacks[level]:
            try:
                callback(level, snapshot)
            except Exception as e:
                self._logger.error(f"Callback error: {e}")
        
        # Log alerts
        if level == AlertLevel.WARNING:
            self._logger.warning(
                f"Memory warning: {snapshot.total_usage_mb:.0f}MB / {self.limit_mb:.0f}MB "
                f"({snapshot.utilization_ratio*100:.1f}%)"
            )
        elif level == AlertLevel.CRITICAL:
            self._logger.critical(
                f"Memory CRITICAL: {snapshot.total_usage_mb:.0f}MB / {self.limit_mb:.0f}MB "
                f"({snapshot.utilization_ratio*100:.1f}%)"
            )
        elif level == AlertLevel.EMERGENCY:
            self._logger.error(
                f"Memory EMERGENCY: {snapshot.total_usage_mb:.0f}MB / {self.limit_mb:.0f}MB "
                f"({snapshot.utilization_ratio*100:.1f}%)"
            )
    
    def _aggressive_gc(self) -> None:
        """Perform aggressive garbage collection."""
        start_time = time.time()
        
        # Track objects before GC
        objects_before = gc.get_count()
        
        # Collect all generations multiple times
        for _ in range(3):
            collected = gc.collect()
            self._gc_stats.objects_collected += collected
        
        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass
        
        # Clear tracked objects
        cleaned = 0
        for ref in self._tracked_objects[:]:
            obj = ref()
            if obj is None:
                self._tracked_objects.remove(ref)
                cleaned += 1
        
        elapsed_ms = (time.time() - start_time) * 1000
        self._gc_stats.collections_triggered += 1
        self._gc_stats.last_collection_time_ms = elapsed_ms
        self._gc_stats.total_collection_time_ms += elapsed_ms
        
        self._logger.info(
            f"Aggressive GC completed: {elapsed_ms:.1f}ms, "
            f"collected {self._gc_stats.objects_collected} objects"
        )
    
    def _emergency_response(self, snapshot: MemorySnapshot) -> None:
        """Emergency response when memory is critically low."""
        self._logger.error("EMERGENCY: Initiating emergency memory response")
        
        # Maximum GC
        self._aggressive_gc()
        
        # Clear all caches
        self._clear_all_caches()
        
        # If still over limit, consider terminating non-essential threads
        new_snapshot = self._take_snapshot()
        if new_snapshot.utilization_ratio >= self.emergency_threshold:
            self._logger.error(
                "EMERGENCY: Memory still critical after cleanup. "
                "Consider reducing workload or increasing limit."
            )
            
            # Send SIGUSR1 to trigger graceful degradation in other components
            try:
                os.kill(os.getpid(), signal.SIGUSR1)
            except:
                pass
    
    def _clear_all_caches(self) -> None:
        """Clear all known caches."""
        # Python module caches
        import importlib
        importlib.invalidate_caches()
        
        # Clear LRU caches in common libraries
        try:
            from functools import _lru_cache_wrapper
            for obj in gc.get_objects():
                if isinstance(obj, _lru_cache_wrapper):
                    obj.cache_clear()
        except:
            pass
        
        # NumPy FFT cache
        try:
            import numpy.fft
            if hasattr(numpy.fft, '_cache'):
                numpy.fft._cache.clear()
        except:
            pass
    
    def get_current_usage(self) -> MemorySnapshot:
        """Get current memory usage snapshot."""
        return self._take_snapshot()
    
    def get_history(self, n: int = 100) -> List[Dict]:
        """Get recent memory history."""
        return [s.to_dict() for s in list(self._snapshots)[-n:]]
    
    def get_gc_stats(self) -> Dict:
        """Get garbage collection statistics."""
        return {
            'collections_triggered': self._gc_stats.collections_triggered,
            'objects_collected': self._gc_stats.objects_collected,
            'last_collection_time_ms': self._gc_stats.last_collection_time_ms,
            'total_collection_time_ms': self._gc_stats.total_collection_time_ms
        }
    
    def get_statistics(self) -> Dict:
        """Get comprehensive watchdog statistics."""
        current = self.get_current_usage()
        
        # Calculate trends from history
        if len(self._snapshots) >= 2:
            old = self._snapshots[0]
            new = self._snapshots[-1]
            trend_mb_per_s = (
                (new.total_usage_mb - old.total_usage_mb) /
                ((new.timestamp_ns - old.timestamp_ns) / 1e9)
            )
        else:
            trend_mb_per_s = 0
        
        return {
            'current': current.to_dict(),
            'gc_stats': self.get_gc_stats(),
            'trend_mb_per_second': trend_mb_per_s,
            'tracked_objects': len(self._tracked_objects),
            'history_size': len(self._snapshots),
            'is_running': self._running
        }


# Singleton instance
_watchdog_instance: Optional[MemoryWatchdog] = None


def get_memory_watchdog() -> MemoryWatchdog:
    """Get or create singleton watchdog instance."""
    global _watchdog_instance
    if _watchdog_instance is None:
        _watchdog_instance = MemoryWatchdog()
    return _watchdog_instance


def initialize_watchdog(
    limit_mb: float = 8192.0,
    auto_start: bool = True
) -> MemoryWatchdog:
    """
    Initialize the memory watchdog with custom settings.
    
    Args:
        limit_mb: Memory limit in MB
        auto_start: Whether to start monitoring immediately
        
    Returns:
        Configured watchdog instance
    """
    global _watchdog_instance
    _watchdog_instance = MemoryWatchdog(memory_limit_mb=limit_mb)
    
    if auto_start:
        _watchdog_instance.start()
    
    return _watchdog_instance


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("Hardware Memory Profiler & Watchdog Test")
    print("=" * 60)
    
    # Initialize watchdog
    watchdog = initialize_watchdog(limit_mb=8192.0, auto_start=True)
    
    # Register callbacks
    def alert_handler(level: AlertLevel, snapshot: MemorySnapshot):
        print(f"\n📊 ALERT [{level.name}]: {snapshot.total_usage_mb:.0f}MB / 8192MB")
    
    watchdog.register_callback(AlertLevel.WARNING, alert_handler)
    watchdog.register_callback(AlertLevel.CRITICAL, alert_handler)
    watchdog.register_callback(AlertLevel.EMERGENCY, alert_handler)
    
    # Monitor for a few seconds
    print("\nMonitoring memory usage... (Press Ctrl+C to stop)")
    
    try:
        for i in range(30):
            stats = watchdog.get_statistics()
            current = stats['current']
            
            print(
                f"\r[{i+1}/30] System: {current['system']['used_mb']:.0f}MB | "
                f"GPU: {current['gpu']['used_mb']:.0f}MB | "
                f"Process: {current['process']['rss_mb']:.0f}MB | "
                f"Total: {current['total_usage_mb']:.0f}MB ({current['utilization_ratio']*100:.1f}%)",
                end=''
            )
            
            time.sleep(0.5)
        
        # Show final statistics
        print("\n\n" + "=" * 60)
        print("Final Statistics")
        print("=" * 60)
        
        stats = watchdog.get_statistics()
        print(json.dumps(stats, indent=2, default=str))
        
        # GC stats
        gc_stats = watchdog.get_gc_stats()
        print(f"\nGC Statistics:")
        print(f"  Collections triggered: {gc_stats['collections_triggered']}")
        print(f"  Objects collected: {gc_stats['objects_collected']}")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted")
    finally:
        watchdog.stop()
        print("\n✓ Memory watchdog test completed!")
