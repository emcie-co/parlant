"""DSPy integration server module.

This module provides FastAPI router and endpoints for DSPy integration.
"""

from typing import Dict, Any

from fastapi import APIRouter
from pydantic import BaseModel

from parlant.core.guidelines import Guideline
from parlant.core.engines.types import Context
from parlant.dspy_integration.services import DSPyService


class DSPyRequest(BaseModel):
    """Base model for DSPy API requests."""
    context: Dict[str, Any]
    guidelines: Dict[str, Any]


def create_dspy_router(dspy_service: DSPyService) -> APIRouter:
    """Create FastAPI router for DSPy integration endpoints.
    
    Args:
        dspy_service: Service for DSPy operations
        
    Returns:
        FastAPI router with DSPy endpoints
    """
    router = APIRouter(prefix="/dspy", tags=["dspy"])
    
    @router.post("/enhance")
    async def enhance_guidelines(request: DSPyRequest) -> Dict[str, Any]:
        """Enhance guidelines using DSPy."""
        context = Context(**request.context)
        guidelines = Guideline(**request.guidelines)
        enhanced = await dspy_service.enhance_guidelines(context, guidelines)
        return {"success": True, "data": enhanced.dict()}
    
    return router 