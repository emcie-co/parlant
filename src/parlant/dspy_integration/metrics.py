"""Metrics module for DSPy integration."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any
import time

from parlant.core.metrics import ModelMetrics


@dataclass
class DSPyMetrics(ModelMetrics):
    """Extended metrics for DSPy operations.
    
    This class extends the base ModelMetrics to add DSPy-specific tracking
    for optimizations, classifications, and errors.
    
    Attributes:
        total_optimizations: Total number of optimization operations
        total_classifications: Total number of classification operations
        avg_optimization_latency: Average latency for optimization operations
        avg_classification_latency: Average latency for classification operations
        last_error: Last error message encountered
        last_error_time: Timestamp of last error
        operation_counts: Counter for different operation types
        operation_latencies: Total latency for each operation type
    """
    total_optimizations: int = 0
    total_classifications: int = 0
    avg_optimization_latency: float = 0.0
    avg_classification_latency: float = 0.0
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None
    operation_counts: dict[str, int] = field(default_factory=dict)
    operation_latencies: dict[str, float] = field(default_factory=dict)
    
    def record_operation(self, operation: str, latency: float, metadata: Optional[dict[str, Any]] = None) -> None:
        """Record metrics for an operation.
        
        Args:
            operation: Name of the operation
            latency: Operation latency in seconds
            metadata: Optional metadata about the operation
        """
        self.operation_counts[operation] = self.operation_counts.get(operation, 0) + 1
        self.operation_latencies[operation] = self.operation_latencies.get(operation, 0.0) + latency
        
        if operation == "optimization":
            self.total_optimizations += 1
            self.avg_optimization_latency = (
                (self.avg_optimization_latency * (self.total_optimizations - 1) + latency)
                / self.total_optimizations
            )
        elif operation == "classification":
            self.total_classifications += 1
            self.avg_classification_latency = (
                (self.avg_classification_latency * (self.total_classifications - 1) + latency)
                / self.total_classifications
            )
    
    def record_error(self, error: str) -> None:
        """Record an error.
        
        Args:
            error: Error message to record
        """
        self.last_error = error
        self.last_error_time = datetime.now()
    
    @property
    def avg_latency(self) -> dict[str, float]:
        """Calculate average latency for each operation type.
        
        Returns:
            Dictionary mapping operation names to average latencies
        """
        return {
            op: self.operation_latencies[op] / count
            for op, count in self.operation_counts.items()
            if count > 0
        }
    
    def reset(self) -> None:
        """Reset all metrics to initial values."""
        super().reset()
        self.total_optimizations = 0
        self.total_classifications = 0
        self.avg_optimization_latency = 0.0
        self.avg_classification_latency = 0.0
        self.last_error = None
        self.last_error_time = None
        self.operation_counts.clear()
        self.operation_latencies.clear()


class MetricsTimer:
    """Context manager for timing operations.
    
    Example:
        ```python
        metrics = DSPyMetrics()
        with MetricsTimer(metrics, "optimization") as timer:
            result = optimize_guidelines()
        ```
    """
    
    def __init__(self, metrics: DSPyMetrics, operation: str, metadata: Optional[dict[str, Any]] = None) -> None:
        """Initialize the timer.
        
        Args:
            metrics: DSPyMetrics instance to record to
            operation: Name of the operation being timed
            metadata: Optional metadata about the operation
        """
        self.metrics = metrics
        self.operation = operation
        self.metadata = metadata or {}
        self.start_time: float = 0.0
        
    def __enter__(self) -> 'MetricsTimer':
        """Start timing the operation."""
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop timing and record metrics.
        
        If an exception occurred, it will be recorded as an error.
        """
        end_time = time.time()
        latency = end_time - self.start_time
        
        if exc_val is not None:
            self.metrics.record_error(str(exc_val))
        
        self.metrics.record_operation(
            operation=self.operation,
            latency=latency,
            metadata=self.metadata
        ) 