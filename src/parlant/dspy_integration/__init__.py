"""DSPy integration module for Parlant."""

from .config import DSPyConfig
from .metrics import DSPyMetrics, MetricsTimer
from .container import setup_dspy_container, setup_dspy_app
from .server import create_dspy_router
from .server_integration import setup_dspy_routes, create_dspy_api_router
from .engine.proposer import DSPyGuidelineProposer
from .services import DSPyService

__all__ = [
    'DSPyConfig',
    'DSPyMetrics',
    'MetricsTimer',
    'setup_dspy_container',
    'setup_dspy_app',
    'create_dspy_router',
    'setup_dspy_routes',
    'create_dspy_api_router',
    'DSPyGuidelineProposer',
    'DSPyService',
]
