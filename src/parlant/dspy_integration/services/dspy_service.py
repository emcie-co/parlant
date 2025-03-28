"""DSPy integration services.

This module provides service interfaces for DSPy integration with Parlant.
"""

from typing import Optional

import structlog

from parlant.core.logging import Logger
from parlant.core.metrics import ModelMetrics


class DSPyService:
    """Service interface for DSPy operations."""
    
    def __init__(
        self,
        logger: Logger,
        api_key: Optional[str] = None,
        model_name: str = "openai/gpt-3.5-turbo",
        metrics: Optional[ModelMetrics] = None,
        use_optimizer: bool = True
    ) -> None:
        """Initialize the DSPy service.
        
        Args:
            logger: Logger instance for tracking operations
            api_key: Optional API key for the model provider
            model_name: Name of the model to use
            metrics: Optional metrics tracker
            use_optimizer: Whether to use optimization
        """
        self._logger = logger
        self.model_name = model_name
        self.metrics = metrics or ModelMetrics()
        self.use_optimizer = use_optimizer
        
        self._logger.debug(f"[DSPy] Initializing DSPy service with model {model_name}")
        if use_optimizer:
            self._logger.debug("[DSPy] DSPy optimizer enabled")
        
        # Log configuration details
        self._logger.debug(f"[DSPy] Service configuration: model={model_name}, use_optimizer={use_optimizer}") 