"""Server integration module for DSPy."""

from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, APIRouter
from lagom import Container

from parlant.core.logging import Logger
from parlant.dspy_integration.config import DSPyConfig
from parlant.dspy_integration.metrics import DSPyMetrics
from parlant.dspy_integration.server import create_dspy_router
from parlant.dspy_integration.engine.proposer import DSPyGuidelineProposer
from parlant.dspy_integration.services import DSPyService


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
    
    # Create and register metrics
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


def create_dspy_api_router(container: Container) -> APIRouter:
    """Create FastAPI router for DSPy endpoints.
    
    This function creates a router with DSPy endpoints that can be included
    in the main FastAPI application.
    
    Args:
        container: Container with DSPy components
        
    Returns:
        APIRouter with DSPy endpoints
    """
    # Create a basic router that will act as a wrapper
    router = APIRouter(prefix="/dspy", tags=["DSPy"])
    
    # We need to add a sub-router or endpoints directly here
    @router.get("/")
    async def dspy_root():
        return {"status": "DSPy API Router is active"}
    
    return router


def setup_dspy_routes(app: FastAPI, container: Container) -> None:
    """Set up DSPy routes in the FastAPI application.
    
    This function adds DSPy routes to an existing FastAPI application.
    
    Args:
        app: Existing FastAPI application
        container: Container with DSPy components
    """
    # Get DSPy service from container
    dspy_service = container[DSPyService]
    
    # Create DSPy router app with just the service
    dspy_app = create_dspy_router(dspy_service=dspy_service)
    
    # Mount it directly
    app.mount("/dspy", dspy_app)


@asynccontextmanager
async def setup_dspy_app(app: FastAPI, container: Container) -> AsyncIterator[None]:
    """Set up FastAPI application with DSPy routes.
    
    This context manager adds DSPy routes to an existing FastAPI application
    and ensures proper cleanup on exit.
    
    Args:
        app: Existing FastAPI application
        container: Container with DSPy components
        
    Yields:
        None
    """
    # Mount DSPy endpoints
    setup_dspy_routes(app, container)
    
    try:
        yield
    finally:
        # Cleanup if needed
        pass 