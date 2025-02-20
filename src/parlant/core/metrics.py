"""Metrics tracking for model performance."""
import time
from dataclasses import dataclass


@dataclass
class ModelMetrics:
    """Metrics for tracking model performance."""
    total_tokens: int = 0
    total_api_calls: int = 0
    total_time: float = 0.0
    start_time: float = 0.0

    def start(self) -> None:
        """Start tracking metrics."""
        self.start_time = time.time()

    def stop(self) -> None:
        """Stop tracking metrics."""
        self.total_time = time.time() - self.start_time
