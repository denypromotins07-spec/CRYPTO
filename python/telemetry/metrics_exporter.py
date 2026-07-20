"""
Prometheus-compatible Metrics Exporter for System Telemetry
Exports metrics from Ray, Nautilus Trader, and Python ML models
Includes memory usage, queue depths, and ML inference times
"""

import time
import threading
import psutil
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from collections import deque
from dataclasses import dataclass, field
import json


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


class MetricType:
    """Metric type enumeration."""
    GAUGE = "gauge"
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricDefinition:
    """Metric definition with metadata."""
    name: str
    description: str
    metric_type: str
    label_names: List[str] = field(default_factory=list)


class PrometheusMetric:
    """Base class for Prometheus-style metrics."""
    
    def __init__(self, name: str, description: str, labels: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.labels = labels or []
        self.values: Dict[str, MetricPoint] = {}
        self.lock = threading.Lock()
        
    def _make_label_key(self, label_values: Dict[str, str]) -> str:
        """Create unique key from label values."""
        sorted_labels = sorted(label_values.items())
        return ",".join(f"{k}={v}" for k, v in sorted_labels)
    
    def collect(self) -> str:
        """Collect metrics in Prometheus format."""
        raise NotImplementedError


class Gauge(PrometheusMetric):
    """Gauge metric that can go up or down."""
    
    def set(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Set gauge value."""
        label_values = labels or {}
        key = self._make_label_key(label_values)
        
        with self.lock:
            self.values[key] = MetricPoint(
                timestamp=time.time(),
                value=value,
                labels=label_values
            )
    
    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment gauge by amount."""
        label_values = labels or {}
        key = self._make_label_key(label_values)
        
        with self.lock:
            if key in self.values:
                new_value = self.values[key].value + amount
            else:
                new_value = amount
            
            self.values[key] = MetricPoint(
                timestamp=time.time(),
                value=new_value,
                labels=label_values
            )
    
    def dec(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Decrement gauge by amount."""
        self.inc(-amount, labels)
    
    def collect(self) -> str:
        """Export in Prometheus format."""
        lines = [f"# HELP {self.name} {self.description}", f"# TYPE {self.name} gauge"]
        
        with self.lock:
            for point in self.values.values():
                if point.labels:
                    label_str = "{" + ",".join(f'{k}="{v}"' for k, v in point.labels.items()) + "}"
                else:
                    label_str = ""
                lines.append(f"{self.name}{label_str} {point.value}")
        
        return "\n".join(lines)


class Counter(PrometheusMetric):
    """Counter metric that only increases."""
    
    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment counter."""
        label_values = labels or {}
        key = self._make_label_key(label_values)
        
        with self.lock:
            if key in self.values:
                new_value = self.values[key].value + amount
            else:
                new_value = amount
            
            self.values[key] = MetricPoint(
                timestamp=time.time(),
                value=new_value,
                labels=label_values
            )
    
    def collect(self) -> str:
        """Export in Prometheus format."""
        lines = [f"# HELP {self.name} {self.description}", f"# TYPE {self.name} counter"]
        
        with self.lock:
            for point in self.values.values():
                if point.labels:
                    label_str = "{" + ",".join(f'{k}="{v}"' for k, v in point.labels.items()) + "}"
                else:
                    label_str = ""
                lines.append(f"{self.name}{label_str} {point.value}")
        
        return "\n".join(lines)


class Histogram(PrometheusMetric):
    """Histogram metric for distributions."""
    
    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf'))
    
    def __init__(self, name: str, description: str, labels: Optional[List[str]] = None,
                 buckets: Optional[tuple] = None):
        super().__init__(name, description, labels)
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self.bucket_counts: Dict[str, Dict[float, int]] = {}
        self.sum_values: Dict[str, float] = {}
        self.count_values: Dict[str, int] = {}
    
    def observe(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe a value."""
        label_values = labels or {}
        key = self._make_label_key(label_values)
        
        with self.lock:
            if key not in self.bucket_counts:
                self.bucket_counts[key] = {b: 0 for b in self.buckets}
                self.sum_values[key] = 0.0
                self.count_values[key] = 0
            
            # Update bucket counts
            for bucket in self.buckets:
                if value <= bucket:
                    self.bucket_counts[key][bucket] += 1
            
            self.sum_values[key] += value
            self.count_values[key] += 1
    
    def collect(self) -> str:
        """Export in Prometheus format."""
        lines = [f"# HELP {self.name} {self.description}", f"# TYPE {self.name} histogram"]
        
        with self.lock:
            for key, bucket_counts in self.bucket_counts.items():
                point = self.values.get(key)
                labels = point.labels if point else {}
                
                if labels:
                    label_str = "{" + ",".join(f'{k}="{v}"' for k, v in labels.items()) + "}"
                else:
                    label_str = ""
                
                # Export bucket counts
                cumulative = 0
                for bucket in self.buckets:
                    cumulative += bucket_counts[bucket]
                    bucket_label = f'{label_str},le="{bucket}"' if label_str else f'{{le="{bucket}"}}'
                    lines.append(f"{self.name}_bucket{bucket_label} {cumulative}")
                
                # Export sum and count
                lines.append(f"{self.name}_sum{label_str} {self.sum_values[key]}")
                lines.append(f"{self.name}_count{label_str} {self.count_values[key]}")
        
        return "\n".join(lines)


class MetricsExporter:
    """
    Main metrics exporter for the trading system.
    Collects and exports metrics from all components.
    """
    
    def __init__(self, prefix: str = "crypto_bot"):
        self.prefix = prefix
        self.metrics: Dict[str, PrometheusMetric] = {}
        self.export_lock = threading.Lock()
        
        # Initialize standard metrics
        self._init_system_metrics()
        self._init_ml_metrics()
        self._init_trading_metrics()
        self._init_ray_metrics()
        
        # Background export thread
        self.running = False
        self.export_thread: Optional[threading.Thread] = None
        
        # Callbacks for custom metrics
        self.custom_collectors: List[Callable[[], Dict[str, float]]] = []
    
    def _register_metric(self, metric: PrometheusMetric):
        """Register a metric."""
        with self.export_lock:
            self.metrics[metric.name] = metric
    
    def _init_system_metrics(self):
        """Initialize system health metrics."""
        # Memory usage
        self._register_metric(Gauge(
            f"{self.prefix}_memory_usage_bytes",
            "Current memory usage in bytes"
        ))
        
        self._register_metric(Gauge(
            f"{self.prefix}_memory_usage_percent",
            "Current memory usage percentage"
        ))
        
        self._register_metric(Gauge(
            f"{self.prefix}_ram_limit_gb",
            "RAM limit in GB (should be 8.0)"
        ))
        
        # CPU usage
        self._register_metric(Gauge(
            f"{self.prefix}_cpu_usage_percent",
            "Current CPU usage percentage"
        ))
        
        # Process count
        self._register_metric(Gauge(
            f"{self.prefix}_process_count",
            "Number of active processes"
        ))
    
    def _init_ml_metrics(self):
        """Initialize ML model metrics."""
        # Inference latency
        self._register_metric(Histogram(
            f"{self.prefix}_ml_inference_latency_seconds",
            "ML model inference latency distribution",
            buckets=(0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
        ))
        
        # Model predictions
        self._register_metric(Counter(
            f"{self.prefix}_ml_predictions_total",
            "Total number of ML predictions"
        ))
        
        # Queue depth for prediction requests
        self._register_metric(Gauge(
            f"{self.prefix}_ml_queue_depth",
            "Current ML prediction queue depth"
        ))
        
        # Model version
        self._register_metric(Gauge(
            f"{self.prefix}_model_version",
            "Current model version"
        ))
    
    def _init_trading_metrics(self):
        """Initialize trading metrics."""
        # Order book depth
        self._register_metric(Gauge(
            f"{self.prefix}_orderbook_bid_depth",
            "Order book bid depth"
        ))
        
        self._register_metric(Gauge(
            f"{self.prefix}_orderbook_ask_depth",
            "Order book ask depth"
        ))
        
        # Spread
        self._register_metric(Gauge(
            f"{self.prefix}_spread_bps",
            "Current spread in basis points"
        ))
        
        # Latency
        self._register_metric(Histogram(
            f"{self.prefix}_tick_to_trade_latency_seconds",
            "Tick-to-trade latency distribution",
            buckets=(0.000001, 0.000005, 0.00001, 0.000025, 0.00005, 0.0001, 0.00025, 0.0005, 0.001)
        ))
        
        # PnL
        self._register_metric(Gauge(
            f"{self.prefix}_pnl_unrealized",
            "Current unrealized PnL"
        ))
        
        self._register_metric(Gauge(
            f"{self.prefix}_pnl_realized",
            "Realized PnL"
        ))
    
    def _init_ray_metrics(self):
        """Initialize Ray-specific metrics."""
        self._register_metric(Gauge(
            f"{self.prefix}_ray_workers_active",
            "Number of active Ray workers"
        ))
        
        self._register_metric(Gauge(
            f"{self.prefix}_ray_object_store_usage",
            "Ray object store usage bytes"
        ))
        
        self._register_metric(Gauge(
            f"{self.prefix}_ray_task_queue_depth",
            "Ray task queue depth"
        ))
    
    def update_memory_metrics(self):
        """Update memory-related metrics."""
        mem_info = psutil.virtual_memory()
        
        # Convert to GB for limit check
        mem_used_gb = mem_info.used / (1024 ** 3)
        mem_percent = mem_info.percent
        
        self.metrics[f"{self.prefix}_memory_usage_bytes"].set(mem_info.used)
        self.metrics[f"{self.prefix}_memory_usage_percent"].set(mem_percent)
        self.metrics[f"{self.prefix}_ram_limit_gb"].set(8.0)
        
        # Check if approaching limit
        if mem_used_gb > 7.5:  # 7.5GB out of 8GB limit
            print(f"WARNING: Memory usage at {mem_used_gb:.2f}GB / 8GB limit")
    
    def update_cpu_metrics(self):
        """Update CPU-related metrics."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        process_count = len(psutil.pids())
        
        self.metrics[f"{self.prefix}_cpu_usage_percent"].set(cpu_percent)
        self.metrics[f"{self.prefix}_process_count"].set(process_count)
    
    def record_inference(self, latency_seconds: float, model_name: str = "default"):
        """Record an ML inference event."""
        self.metrics[f"{self.prefix}_ml_inference_latency_seconds"].observe(
            latency_seconds,
            labels={"model": model_name}
        )
        self.metrics[f"{self.prefix}_ml_predictions_total"].inc(
            labels={"model": model_name}
        )
    
    def update_queue_depth(self, queue_name: str, depth: int):
        """Update queue depth metric."""
        self.metrics[f"{self.prefix}_ml_queue_depth"].set(
            depth,
            labels={"queue": queue_name}
        )
    
    def record_tick_to_trade(self, latency_seconds: float):
        """Record tick-to-trade latency."""
        self.metrics[f"{self.prefix}_tick_to_trade_latency_seconds"].observe(latency_seconds)
    
    def register_custom_collector(self, collector: Callable[[], Dict[str, float]]):
        """Register a custom metric collector function."""
        self.custom_collectors.append(collector)
    
    def collect_all(self) -> str:
        """Collect all metrics in Prometheus format."""
        # Update system metrics
        self.update_memory_metrics()
        self.update_cpu_metrics()
        
        # Collect from all registered metrics
        lines = []
        with self.export_lock:
            for metric in self.metrics.values():
                lines.append(metric.collect())
                lines.append("")  # Empty line between metrics
        
        # Collect custom metrics
        for collector in self.custom_collectors:
            try:
                custom_metrics = collector()
                for name, value in custom_metrics.items():
                    lines.append(f"# TYPE {name} gauge")
                    lines.append(f"{name} {value}")
            except Exception as e:
                print(f"Error in custom collector: {e}")
        
        return "\n".join(lines)
    
    def collect_json(self) -> Dict[str, Any]:
        """Collect metrics as JSON."""
        result = {
            "timestamp": time.time(),
            "metrics": {}
        }
        
        with self.export_lock:
            for name, metric in self.metrics.items():
                if hasattr(metric, 'values'):
                    values = {}
                    for key, point in metric.values.items():
                        values[key] = {
                            "value": point.value,
                            "timestamp": point.timestamp,
                            "labels": point.labels
                        }
                    result["metrics"][name] = values
        
        return result
    
    def start_background_export(self, interval_seconds: float = 1.0):
        """Start background metric collection."""
        self.running = True
        
        def export_loop():
            while self.running:
                try:
                    self.collect_all()
                except Exception as e:
                    print(f"Error in background export: {e}")
                time.sleep(interval_seconds)
        
        self.export_thread = threading.Thread(target=export_loop, daemon=True)
        self.export_thread.start()
    
    def stop_background_export(self):
        """Stop background metric collection."""
        self.running = False
        if self.export_thread:
            self.export_thread.join(timeout=2.0)
    
    def get_memory_status(self) -> Dict[str, float]:
        """Get current memory status."""
        mem_info = psutil.virtual_memory()
        return {
            "used_gb": mem_info.used / (1024 ** 3),
            "percent": mem_info.percent,
            "available_gb": mem_info.available / (1024 ** 3),
            "limit_gb": 8.0,
            "approaching_limit": mem_info.used / (1024 ** 3) > 7.5
        }


# Example usage
if __name__ == "__main__":
    exporter = MetricsExporter(prefix="crypto_bot")
    
    # Simulate some metrics
    exporter.record_inference(0.0025, model_name="transformer")
    exporter.record_inference(0.001, model_name="lstm")
    exporter.record_tick_to_trade(0.00005)  # 50 microseconds
    
    exporter.update_queue_depth("prediction_queue", 15)
    exporter.metrics[f"{exporter.prefix}_pnl_unrealized"].set(1250.50)
    exporter.metrics[f"{exporter.prefix}_spread_bps"].set(5.2)
    
    # Get Prometheus format
    print("=== Prometheus Format ===")
    print(exporter.collect_all())
    
    # Get JSON format
    print("\n=== JSON Format ===")
    print(json.dumps(exporter.collect_json(), indent=2))
    
    # Check memory status
    print("\n=== Memory Status ===")
    print(exporter.get_memory_status())
