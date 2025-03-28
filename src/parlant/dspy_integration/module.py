"""DSPy integration module configuration.

This module configures the DSPy integration with the core message pipeline.
"""

from typing import Any, Dict, Optional
from functools import partial

from lagom import Container

from parlant.core.engines.alpha.message_event_composer import MessageEventComposer
from parlant.core.logging import Logger
from parlant.core.metrics import ModelMetrics
from parlant.dspy_integration.composers import DSPyEnhancedMessageComposer
from parlant.dspy_integration.services import DSPyService
from parlant.dspy_integration.guideline_optimizer import BatchOptimizedGuidelineManager
from parlant.dspy_integration.guideline_classifier import GuidelineClassifier
from parlant.core.logging import Logger, LogLevel

async def initialize_module(container: Container) -> Optional[Container]:
    """Initialize the DSPy integration module.
    
    Args:
        container: The dependency injection container
        
    Returns:
        Optional[Container]: The modified container if any changes were made
    """
    logger = container.resolve(Logger)
    metrics = container.resolve(ModelMetrics)
    
    # Set DSPy components to debug level
    logger.set_level(LogLevel.DEBUG)
    
    # Initialize DSPy services with debug logging
    dspy_service = DSPyService(logger)
    guideline_optimizer = BatchOptimizedGuidelineManager(metrics=metrics, use_optimizer=True, logger=logger)
    context_classifier = GuidelineClassifier(logger)
    
    # Create enhanced composer with debug logging
    enhanced_composer = DSPyEnhancedMessageComposer(
        dspy_service=dspy_service,
        guideline_optimizer=guideline_optimizer,
        context_classifier=context_classifier,
        logger=logger
    )
    
    # Add operation logging for DSPy operations
    logger.info("[DSPy] Initializing DSPy integration with debug logging")
    
    # Register the enhanced composer as the default
    container[MessageEventComposer] = enhanced_composer
    
    logger.info("[DSPy] DSPy integration module initialized successfully")
    return container


async def shutdown_module() -> None:
    """Shut down the DSPy integration module."""
    pass  # No cleanup needed yet 