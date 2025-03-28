"""Container module for DSPy integration."""

from typing import Optional
from contextlib import asynccontextmanager
from typing import AsyncIterator

from lagom import Container
from fastapi import FastAPI

from parlant.core.logging import Logger
from parlant.dspy_integration.config import DSPyConfig
from parlant.dspy_integration.metrics import DSPyMetrics
from parlant.dspy_integration.server import create_dspy_router
from parlant.dspy_integration.engine.proposer import DSPyGuidelineProposer


def setup_dspy_container(container: Container) -> Container:
    """Set up DSPy components in the container.
    
    This function registers DSPy-related components in the dependency injection
    container, including configuration, metrics, and the guideline proposer.
    
    Args:
        container: Base container to extend
        
    Returns:
        Container with DSPy components registered
    """
    # Get logger from container
    logger = container[Logger]
    
    # Create and register config
    config = DSPyConfig.from_env(logger=logger)
    container[DSPyConfig] = config
    
    # Create and register metrics with explicit initialization
    # to avoid container resolution issues
    metrics = DSPyMetrics(
        operation_counts={},
        operation_latencies={}
    )
    container[DSPyMetrics] = metrics
    
    # Create and register proposer
    container[DSPyGuidelineProposer] = DSPyGuidelineProposer(
        logger=logger,
        api_key=config.api_key,
        model_name=config.model_name,
        metrics=metrics
    )
    
    return container


@asynccontextmanager
async def setup_dspy_app(app: FastAPI, container: Container) -> AsyncIterator[None]:
    """Set up FastAPI application with DSPy components.
    
    This context manager sets up the FastAPI application with DSPy routes
    and ensures proper cleanup on exit.
    
    Args:
        app: FastAPI application to configure
        container: Container with DSPy components
        
    Yields:
        None
    """
    # Create DSPy router
    dspy_router = create_dspy_router(
        logger=container[Logger],
        config=container[DSPyConfig],
        metrics=container[DSPyMetrics]
    )
    
    # Add DSPy router to the app
    app.mount("/dspy", dspy_router)
    
    try:
        yield
    finally:
        # Clean up resources if needed
        pass 