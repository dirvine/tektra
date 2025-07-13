#!/usr/bin/env python3
"""
Performance Monitoring System

Real-time performance monitoring, profiling, and metrics collection with
distributed tracing support.

UV Dependencies:
# Install UV if not available: curl -LsSf https://astral.sh/uv/install.sh | sh
# Run with dependencies: uv run --with psutil,prometheus-client,opentelemetry-api,opentelemetry-sdk,loguru python performance_monitor.py
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "psutil>=5.9.0",
#     "prometheus-client>=0.19.0",
#     "opentelemetry-api>=1.20.0",
#     "opentelemetry-sdk>=1.20.0",
#     "loguru>=0.7.0",
# ]
# ///

import asyncio
import cProfile
import io
import pstats
import sys
import time
import traceback
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import threading
import gc

import psutil
from prometheus_client import Counter, Gauge, Histogram, Summary, CollectorRegistry, generate_latest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from loguru import logger


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"      # Monotonically increasing value
    GAUGE = "gauge"          # Value that can go up or down
    HISTOGRAM = "histogram"  # Distribution of values
    SUMMARY = "summary"      # Similar to histogram with percentiles


@dataclass
class MetricPoint:
    """A single metric data point."""
    
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate metric point."""
        if self.timestamp < 0:
            raise ValueError("Timestamp must be non-negative")


@dataclass
class MetricTimeSeries:
    """Time series data for a metric."""
    
    name: str
    metric_type: MetricType
    description: str = ""
    unit: str = ""
    
    points: deque[MetricPoint] = field(default_factory=lambda: deque(maxlen=10000))
    
    # Statistics
    min_value: float = float('inf')
    max_value: float = float('-inf')
    sum_value: float = 0.0
    count: int = 0
    
    def add_point(self, value: float, timestamp: Optional[float] = None, labels: Optional[Dict[str, str]] = None) -> None:
        """Add a data point to the time series."""
        if timestamp is None:
            timestamp = time.time()
        
        point = MetricPoint(timestamp=timestamp, value=value, labels=labels or {})
        self.points.append(point)
        
        # Update statistics
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)
        self.sum_value += value
        self.count += 1
    
    @property
    def average(self) -> float:
        """Get average value."""
        return self.sum_value / self.count if self.count > 0 else 0.0
    
    @property
    def latest(self) -> Optional[float]:
        """Get latest value."""
        return self.points[-1].value if self.points else None
    
    def get_window(self, duration_seconds: float) -> List[MetricPoint]:
        """Get points within a time window."""
        cutoff = time.time() - duration_seconds
        return [p for p in self.points if p.timestamp >= cutoff]


@dataclass
class ProfileResult:
    """Results from a profiling session."""
    
    name: str
    duration_seconds: float
    
    # CPU profiling
    cpu_time: float = 0.0
    function_calls: Dict[str, int] = field(default_factory=dict)
    time_by_function: Dict[str, float] = field(default_factory=dict)
    
    # Memory profiling
    memory_allocated_mb: float = 0.0
    memory_peak_mb: float = 0.0
    gc_collections: Dict[int, int] = field(default_factory=dict)
    
    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system.
    
    Features:
    - Real-time metrics collection
    - CPU and memory profiling
    - Distributed tracing
    - Prometheus metrics export
    - Performance anomaly detection
    - Resource usage tracking
    - Bottleneck identification
    """
    
    def __init__(
        self,
        enable_tracing: bool = True,
        enable_profiling: bool = True,
        enable_prometheus: bool = True,
        metrics_port: int = 9090,
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize performance monitor.
        
        Args:
            enable_tracing: Enable distributed tracing
            enable_profiling: Enable CPU/memory profiling
            enable_prometheus: Enable Prometheus metrics
            metrics_port: Port for Prometheus metrics endpoint
            alert_thresholds: Thresholds for performance alerts
        """
        self.enable_tracing = enable_tracing
        self.enable_profiling = enable_profiling
        self.enable_prometheus = enable_prometheus
        self.metrics_port = metrics_port
        self.alert_thresholds = alert_thresholds or {}
        
        # Metrics storage
        self.metrics: Dict[str, MetricTimeSeries] = {}
        
        # Profiling data
        self.profile_results: Dict[str, ProfileResult] = {}
        self.active_profiles: Dict[str, cProfile.Profile] = {}
        
        # System metrics
        self.process = psutil.Process()
        self._last_cpu_time = self.process.cpu_times()
        
        # Tracing setup
        if enable_tracing:
            self._setup_tracing()
        
        # Prometheus setup
        if enable_prometheus:
            self._setup_prometheus()
        
        # Alert tracking
        self.active_alerts: Set[str] = set()
        self.alert_history: List[Dict[str, Any]] = []
        
        # Background monitoring
        self._monitor_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("Performance monitor initialized")
    
    def _setup_tracing(self) -> None:
        """Set up distributed tracing."""
        # Set up the tracer provider
        trace.set_tracer_provider(TracerProvider())
        
        # Add console exporter for demo
        console_exporter = ConsoleSpanExporter()
        span_processor = BatchSpanProcessor(console_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
        
        self.tracer = trace.get_tracer(__name__)
        logger.debug("Distributed tracing initialized")
    
    def _setup_prometheus(self) -> None:
        """Set up Prometheus metrics."""
        self.registry = CollectorRegistry()
        
        # System metrics
        self.prom_cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.prom_memory_usage = Gauge(
            'system_memory_usage_mb',
            'Memory usage in MB',
            registry=self.registry
        )
        
        self.prom_disk_io = Counter(
            'system_disk_io_bytes',
            'Disk I/O in bytes',
            ['direction'],  # read/write
            registry=self.registry
        )
        
        # Application metrics
        self.prom_request_count = Counter(
            'app_requests_total',
            'Total number of requests',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.prom_request_duration = Histogram(
            'app_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.prom_active_tasks = Gauge(
            'app_active_tasks',
            'Number of active tasks',
            ['task_type'],
            registry=self.registry
        )
        
        logger.debug("Prometheus metrics initialized")
    
    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        labels: Optional[Dict[str, str]] = None,
        description: str = "",
        unit: str = ""
    ) -> None:
        """
        Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            labels: Optional labels
            description: Metric description
            unit: Unit of measurement
        """
        with self._lock:
            if name not in self.metrics:
                self.metrics[name] = MetricTimeSeries(
                    name=name,
                    metric_type=metric_type,
                    description=description,
                    unit=unit
                )
            
            self.metrics[name].add_point(value, labels=labels)
            
            # Check alert thresholds
            if name in self.alert_thresholds:
                threshold = self.alert_thresholds[name]
                if value > threshold and name not in self.active_alerts:
                    self._trigger_alert(name, value, threshold)
                elif value <= threshold and name in self.active_alerts:
                    self._clear_alert(name)
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        with self._lock:
            if name not in self.metrics:
                self.metrics[name] = MetricTimeSeries(
                    name=name,
                    metric_type=MetricType.COUNTER
                )
            
            current = self.metrics[name].latest or 0.0
            self.metrics[name].add_point(current + value, labels=labels)
    
    @contextmanager
    def measure_time(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager to measure execution time."""
        start_time = time.time()
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_metric(
                name=f"{name}_duration",
                value=duration,
                metric_type=MetricType.HISTOGRAM,
                labels=labels,
                unit="seconds"
            )
    
    @contextmanager
    def profile(self, name: str):
        """Context manager for profiling code."""
        if not self.enable_profiling:
            yield
            return
        
        profile = cProfile.Profile()
        
        # Memory before
        gc.collect()
        mem_before = self.process.memory_info().rss / (1024 * 1024)
        gc_before = gc.get_count()
        
        profile.enable()
        start_time = time.time()
        
        try:
            yield
        finally:
            profile.disable()
            duration = time.time() - start_time
            
            # Memory after
            gc.collect()
            mem_after = self.process.memory_info().rss / (1024 * 1024)
            gc_after = gc.get_count()
            
            # Process profile data
            result = ProfileResult(
                name=name,
                duration_seconds=duration,
                memory_allocated_mb=max(0, mem_after - mem_before),
                memory_peak_mb=mem_after
            )
            
            # Extract function statistics
            stats = pstats.Stats(profile)
            stats.sort_stats('cumulative')
            
            for func, (cc, nc, tt, ct, callers) in stats.stats.items():
                func_name = f"{func[0]}:{func[1]}:{func[2]}"
                result.function_calls[func_name] = nc
                result.time_by_function[func_name] = ct
                result.cpu_time += ct
            
            # GC collections
            for i in range(3):
                result.gc_collections[i] = gc_after[i] - gc_before[i]
            
            with self._lock:
                self.profile_results[name] = result
            
            logger.debug(f"Profile completed: {name} ({duration:.3f}s)")
    
    def trace_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Create a tracing span."""
        if not self.enable_tracing:
            return nullcontext()
        
        return self.tracer.start_as_current_span(
            name,
            attributes=attributes or {}
        )
    
    async def collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        try:
            # CPU usage
            cpu_percent = self.process.cpu_percent(interval=0.1)
            
            # Memory usage
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            memory_percent = self.process.memory_percent()
            
            # Disk I/O
            io_counters = self.process.io_counters()
            
            # Network (system-wide)
            net_io = psutil.net_io_counters()
            
            # Thread count
            thread_count = self.process.num_threads()
            
            # File descriptors
            try:
                fd_count = self.process.num_fds()
            except:
                fd_count = 0  # Not available on all platforms
            
            metrics = {
                "cpu_percent": cpu_percent,
                "memory_mb": memory_mb,
                "memory_percent": memory_percent,
                "disk_read_mb": io_counters.read_bytes / (1024 * 1024),
                "disk_write_mb": io_counters.write_bytes / (1024 * 1024),
                "net_sent_mb": net_io.bytes_sent / (1024 * 1024),
                "net_recv_mb": net_io.bytes_recv / (1024 * 1024),
                "thread_count": thread_count,
                "fd_count": fd_count,
            }
            
            # Record metrics
            for name, value in metrics.items():
                self.record_metric(f"system_{name}", value)
            
            # Update Prometheus metrics
            if self.enable_prometheus:
                self.prom_cpu_usage.set(cpu_percent)
                self.prom_memory_usage.set(memory_mb)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}
    
    def detect_anomalies(self, window_seconds: float = 300.0) -> List[Dict[str, Any]]:
        """
        Detect performance anomalies.
        
        Args:
            window_seconds: Time window to analyze
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        with self._lock:
            for metric_name, series in self.metrics.items():
                points = series.get_window(window_seconds)
                if len(points) < 10:  # Need enough data
                    continue
                
                values = [p.value for p in points]
                avg = sum(values) / len(values)
                
                # Simple anomaly detection: values > 2 standard deviations
                variance = sum((v - avg) ** 2 for v in values) / len(values)
                std_dev = variance ** 0.5
                
                recent_values = values[-5:]  # Last 5 values
                for i, value in enumerate(recent_values):
                    if abs(value - avg) > 2 * std_dev:
                        anomalies.append({
                            "metric": metric_name,
                            "value": value,
                            "average": avg,
                            "std_dev": std_dev,
                            "timestamp": points[-(5-i)].timestamp,
                            "severity": "high" if abs(value - avg) > 3 * std_dev else "medium"
                        })
        
        return anomalies
    
    def identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        with self._lock:
            # Check CPU bottlenecks
            cpu_metric = self.metrics.get("system_cpu_percent")
            if cpu_metric and cpu_metric.latest > 80:
                bottlenecks.append({
                    "type": "cpu",
                    "severity": "high" if cpu_metric.latest > 90 else "medium",
                    "value": cpu_metric.latest,
                    "description": f"High CPU usage: {cpu_metric.latest:.1f}%"
                })
            
            # Check memory bottlenecks
            mem_metric = self.metrics.get("system_memory_percent")
            if mem_metric and mem_metric.latest > 80:
                bottlenecks.append({
                    "type": "memory",
                    "severity": "high" if mem_metric.latest > 90 else "medium",
                    "value": mem_metric.latest,
                    "description": f"High memory usage: {mem_metric.latest:.1f}%"
                })
            
            # Check slow functions from profiling
            for name, result in self.profile_results.items():
                slow_functions = [
                    (func, time) for func, time in result.time_by_function.items()
                    if time > 0.1  # Functions taking > 100ms
                ]
                
                if slow_functions:
                    bottlenecks.append({
                        "type": "function",
                        "severity": "medium",
                        "profile": name,
                        "slow_functions": sorted(slow_functions, key=lambda x: x[1], reverse=True)[:5],
                        "description": f"Slow functions detected in profile '{name}'"
                    })
        
        return bottlenecks
    
    def get_metrics_summary(self, window_seconds: float = 60.0) -> Dict[str, Any]:
        """Get summary of all metrics."""
        with self._lock:
            summary = {}
            
            for name, series in self.metrics.items():
                points = series.get_window(window_seconds)
                if not points:
                    continue
                
                values = [p.value for p in points]
                
                summary[name] = {
                    "latest": series.latest,
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                    "type": series.metric_type.value,
                    "unit": series.unit
                }
            
            return summary
    
    def get_profile_summary(self, name: str) -> Optional[Dict[str, Any]]:
        """Get summary of a profile result."""
        with self._lock:
            result = self.profile_results.get(name)
            if not result:
                return None
            
            # Get top functions by time
            top_functions = sorted(
                result.time_by_function.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            return {
                "name": result.name,
                "duration_seconds": result.duration_seconds,
                "cpu_time": result.cpu_time,
                "memory_allocated_mb": result.memory_allocated_mb,
                "memory_peak_mb": result.memory_peak_mb,
                "total_function_calls": sum(result.function_calls.values()),
                "gc_collections": result.gc_collections,
                "top_functions": [
                    {"name": func, "time": time, "calls": result.function_calls.get(func, 0)}
                    for func, time in top_functions
                ],
                "custom_metrics": result.custom_metrics
            }
    
    def export_prometheus_metrics(self) -> bytes:
        """Export metrics in Prometheus format."""
        if not self.enable_prometheus:
            return b""
        
        return generate_latest(self.registry)
    
    def _trigger_alert(self, metric_name: str, value: float, threshold: float) -> None:
        """Trigger a performance alert."""
        self.active_alerts.add(metric_name)
        
        alert = {
            "metric": metric_name,
            "value": value,
            "threshold": threshold,
            "timestamp": time.time(),
            "status": "triggered"
        }
        
        self.alert_history.append(alert)
        logger.warning(f"Performance alert: {metric_name} = {value:.2f} (threshold: {threshold})")
    
    def _clear_alert(self, metric_name: str) -> None:
        """Clear a performance alert."""
        if metric_name in self.active_alerts:
            self.active_alerts.remove(metric_name)
            
            alert = {
                "metric": metric_name,
                "timestamp": time.time(),
                "status": "cleared"
            }
            
            self.alert_history.append(alert)
            logger.info(f"Performance alert cleared: {metric_name}")
    
    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while not self._shutdown:
            try:
                # Collect system metrics
                await self.collect_system_metrics()
                
                # Check for anomalies
                anomalies = self.detect_anomalies()
                if anomalies:
                    logger.warning(f"Detected {len(anomalies)} performance anomalies")
                
                # Sleep
                await asyncio.sleep(1.0)  # Collect every second
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
    
    async def start(self) -> None:
        """Start background monitoring."""
        if self._monitor_task is None:
            self._monitor_task = asyncio.create_task(self._monitor_loop())
            logger.info("Started performance monitoring")
    
    async def stop(self) -> None:
        """Stop background monitoring."""
        self._shutdown = True
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped performance monitoring")


def create_performance_monitor(**kwargs) -> PerformanceMonitor:
    """
    Create a performance monitor with the given configuration.
    
    Args:
        **kwargs: Monitor configuration
        
    Returns:
        Configured performance monitor
    """
    return PerformanceMonitor(**kwargs)


# Null context for when tracing is disabled
class nullcontext:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


if __name__ == "__main__":
    import random
    
    async def demo_performance_monitor():
        """Demonstrate performance monitoring functionality."""
        print("📊 Performance Monitor Demo")
        print("=" * 40)
        
        # Create monitor
        monitor = create_performance_monitor(
            enable_tracing=False,  # Disable console spam for demo
            enable_prometheus=True,
            alert_thresholds={
                "demo_response_time": 0.5,  # Alert if response time > 0.5s
                "demo_error_rate": 0.1,     # Alert if error rate > 10%
            }
        )
        
        await monitor.start()
        print("Performance monitor started")
        
        # Simulate some workload
        async def simulate_request(endpoint: str) -> float:
            """Simulate a request with variable response time."""
            # Measure response time
            with monitor.measure_time(f"request_{endpoint}"):
                # Simulate work
                delay = random.uniform(0.1, 0.8)
                await asyncio.sleep(delay)
                
                # Record custom metrics
                monitor.increment_counter("requests_total", labels={"endpoint": endpoint})
                monitor.record_metric("demo_response_time", delay, unit="seconds")
                
                # Simulate errors sometimes
                if random.random() < 0.05:  # 5% error rate
                    monitor.increment_counter("errors_total", labels={"endpoint": endpoint})
                    raise Exception("Simulated error")
                
                return delay
        
        # Profile a function
        def cpu_intensive_work():
            """Simulate CPU-intensive work."""
            result = 0
            for i in range(1000000):
                result += i ** 2
            return result
        
        print("\nSimulating workload...")
        
        # Run some requests
        tasks = []
        endpoints = ["api/users", "api/products", "api/orders"]
        
        for i in range(20):
            endpoint = random.choice(endpoints)
            task = asyncio.create_task(simulate_request(endpoint))
            tasks.append(task)
        
        # Profile CPU work
        with monitor.profile("cpu_intensive"):
            result = cpu_intensive_work()
            print(f"CPU work result: {result}")
        
        # Wait for requests
        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful = sum(1 for r in results if not isinstance(r, Exception))
        print(f"Completed {successful}/{len(tasks)} requests successfully")
        
        # Check system metrics
        await asyncio.sleep(2.0)  # Let metrics collect
        
        print("\nSystem Metrics:")
        system_metrics = await monitor.collect_system_metrics()
        for name, value in system_metrics.items():
            print(f"   {name}: {value:.2f}")
        
        # Get metrics summary
        print("\nMetrics Summary (last 60s):")
        summary = monitor.get_metrics_summary()
        for name, stats in summary.items():
            if name.startswith("demo_") or name.startswith("request_"):
                print(f"   {name}:")
                print(f"      Latest: {stats['latest']:.3f} {stats['unit']}")
                print(f"      Average: {stats['average']:.3f}")
                print(f"      Min/Max: {stats['min']:.3f} / {stats['max']:.3f}")
        
        # Check profile results
        print("\nProfile Results:")
        profile_summary = monitor.get_profile_summary("cpu_intensive")
        if profile_summary:
            print(f"   Duration: {profile_summary['duration_seconds']:.3f}s")
            print(f"   CPU time: {profile_summary['cpu_time']:.3f}s")
            print(f"   Memory allocated: {profile_summary['memory_allocated_mb']:.2f}MB")
            print(f"   Top functions:")
            for func in profile_summary['top_functions'][:3]:
                print(f"      {func['name']}: {func['time']:.3f}s ({func['calls']} calls)")
        
        # Check for anomalies
        print("\nChecking for anomalies...")
        anomalies = monitor.detect_anomalies(window_seconds=30.0)
        if anomalies:
            print(f"Found {len(anomalies)} anomalies:")
            for anomaly in anomalies:
                print(f"   {anomaly['metric']}: {anomaly['value']:.3f} (avg: {anomaly['average']:.3f})")
        else:
            print("No anomalies detected")
        
        # Check bottlenecks
        print("\nIdentifying bottlenecks...")
        bottlenecks = monitor.identify_bottlenecks()
        if bottlenecks:
            print(f"Found {len(bottlenecks)} bottlenecks:")
            for bottleneck in bottlenecks:
                print(f"   {bottleneck['type']}: {bottleneck['description']}")
        else:
            print("No bottlenecks identified")
        
        # Check alerts
        if monitor.active_alerts:
            print(f"\nActive alerts: {monitor.active_alerts}")
        
        # Export Prometheus metrics
        prometheus_data = monitor.export_prometheus_metrics()
        print(f"\nPrometheus metrics exported ({len(prometheus_data)} bytes)")
        
        # Stop monitor
        await monitor.stop()
        print("\n📊 Performance Monitor Demo Complete")
    
    # Run demo
    asyncio.run(demo_performance_monitor())