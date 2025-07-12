"""
Metrics collection for MSRK v3
Provides application metrics for monitoring and observability.
"""

import time
import asyncio
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import threading
import logging

logger = logging.getLogger("msrk.metrics")

class MetricsCollector:
    """Collect and store application metrics."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._metrics = defaultdict(deque)
        self._counters = defaultdict(int)
        self._gauges = defaultdict(float)
        self._timers = defaultdict(list)
        self._lock = threading.Lock()
        
        # System-level metrics
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        
        logger.info("Metrics collector initialized")
    
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        with self._lock:
            metric_key = self._format_metric_key(name, tags)
            self._counters[metric_key] += value
            
            # Also store timestamped data
            self._metrics[metric_key].append({
                "timestamp": datetime.now(),
                "value": self._counters[metric_key],
                "increment": value
            })
            
            # Limit history size
            if len(self._metrics[metric_key]) > self.max_history:
                self._metrics[metric_key].popleft()
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric value."""
        with self._lock:
            metric_key = self._format_metric_key(name, tags)
            self._gauges[metric_key] = value
            
            # Store timestamped data
            self._metrics[metric_key].append({
                "timestamp": datetime.now(),
                "value": value
            })
            
            # Limit history size
            if len(self._metrics[metric_key]) > self.max_history:
                self._metrics[metric_key].popleft()
    
    def record_timer(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record a timer metric."""
        with self._lock:
            metric_key = self._format_metric_key(name, tags)
            self._timers[metric_key].append(duration)
            
            # Store timestamped data
            self._metrics[metric_key].append({
                "timestamp": datetime.now(),
                "duration": duration
            })
            
            # Limit history size for timers
            if len(self._timers[metric_key]) > self.max_history:
                self._timers[metric_key] = self._timers[metric_key][-self.max_history:]
            
            if len(self._metrics[metric_key]) > self.max_history:
                self._metrics[metric_key].popleft()
    
    def get_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> int:
        """Get current counter value."""
        metric_key = self._format_metric_key(name, tags)
        return self._counters.get(metric_key, 0)
    
    def get_gauge(self, name: str, tags: Optional[Dict[str, str]] = None) -> float:
        """Get current gauge value."""
        metric_key = self._format_metric_key(name, tags)
        return self._gauges.get(metric_key, 0.0)
    
    def get_timer_stats(self, name: str, tags: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get timer statistics."""
        metric_key = self._format_metric_key(name, tags)
        durations = self._timers.get(metric_key, [])
        
        if not durations:
            return {"count": 0, "min": 0, "max": 0, "avg": 0, "p95": 0, "p99": 0}
        
        sorted_durations = sorted(durations)
        count = len(sorted_durations)
        
        return {
            "count": count,
            "min": min(sorted_durations),
            "max": max(sorted_durations),
            "avg": sum(sorted_durations) / count,
            "p95": sorted_durations[int(count * 0.95)] if count > 0 else 0,
            "p99": sorted_durations[int(count * 0.99)] if count > 0 else 0
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics."""
        with self._lock:
            current_time = time.time()
            uptime = current_time - self.start_time
            
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": uptime,
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "timers": {}
            }
            
            # Calculate timer statistics
            for metric_key in self._timers:
                metrics["timers"][metric_key] = self.get_timer_stats(
                    metric_key.split("|")[0],  # Extract name from formatted key
                    None  # Tags would need to be parsed from key
                )
            
            return metrics
    
    def get_metric_history(self, name: str, tags: Optional[Dict[str, str]] = None,
                          hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical data for a metric."""
        metric_key = self._format_metric_key(name, tags)
        history = list(self._metrics.get(metric_key, []))
        
        # Filter by time window
        cutoff_time = datetime.now() - timedelta(hours=hours)
        filtered_history = [
            entry for entry in history 
            if entry["timestamp"] > cutoff_time
        ]
        
        return filtered_history
    
    def reset_metrics(self):
        """Reset all metrics."""
        with self._lock:
            self._metrics.clear()
            self._counters.clear()
            self._gauges.clear()
            self._timers.clear()
            self.start_time = time.time()
            self.request_count = 0
            self.error_count = 0
            
        logger.info("All metrics reset")
    
    def _format_metric_key(self, name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """Format metric key with tags."""
        if not tags:
            return name
        
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}|{tag_str}"

# Timer context manager
class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, metrics: MetricsCollector, name: str, tags: Optional[Dict[str, str]] = None):
        self.metrics = metrics
        self.name = name
        self.tags = tags
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.metrics.record_timer(self.name, duration, self.tags)

# Global metrics collector instance
metrics = MetricsCollector()

# Convenience functions
def increment(name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
    """Increment a counter."""
    metrics.increment_counter(name, value, tags)

def gauge(name: str, value: float, tags: Optional[Dict[str, str]] = None):
    """Set a gauge value."""
    metrics.set_gauge(name, value, tags)

def timer(name: str, tags: Optional[Dict[str, str]] = None):
    """Get a timer context manager."""
    return Timer(metrics, name, tags)

def record_time(name: str, duration: float, tags: Optional[Dict[str, str]] = None):
    """Record a timer value."""
    metrics.record_timer(name, duration, tags)

# Application-specific metrics functions
def record_api_call(endpoint: str, method: str, status_code: int, duration: float):
    """Record API call metrics."""
    tags = {
        "endpoint": endpoint,
        "method": method,
        "status": str(status_code)
    }
    
    increment("api.requests.total", 1, tags)
    record_time("api.request.duration", duration, tags)
    
    if status_code >= 400:
        increment("api.errors.total", 1, tags)

def record_database_operation(operation: str, table: str, count: int, duration: float):
    """Record database operation metrics."""
    tags = {
        "operation": operation,
        "table": table
    }
    
    increment("db.operations.total", 1, tags)
    increment("db.records.total", count, tags)
    record_time("db.operation.duration", duration, tags)

def record_model_inference(model_name: str, input_size: int, duration: float):
    """Record model inference metrics."""
    tags = {
        "model": model_name
    }
    
    increment("ml.inferences.total", 1, tags)
    gauge("ml.input.size", input_size, tags)
    record_time("ml.inference.duration", duration, tags)

def record_data_ingestion(source: str, records: int, duration: float, success: bool):
    """Record data ingestion metrics."""
    tags = {
        "source": source,
        "success": str(success)
    }
    
    increment("data.ingestion.jobs.total", 1, tags)
    increment("data.ingestion.records.total", records, tags)
    record_time("data.ingestion.duration", duration, tags)

def get_system_metrics() -> Dict[str, Any]:
    """Get all system metrics."""
    return metrics.get_all_metrics()

def get_application_health_metrics() -> Dict[str, Any]:
    """Get application-specific health metrics."""
    return {
        "api_requests_total": metrics.get_counter("api.requests.total"),
        "api_errors_total": metrics.get_counter("api.errors.total"),
        "db_operations_total": metrics.get_counter("db.operations.total"),
        "ml_inferences_total": metrics.get_counter("ml.inferences.total"),
        "data_ingestion_jobs_total": metrics.get_counter("data.ingestion.jobs.total"),
        "uptime_seconds": time.time() - metrics.start_time
    }
