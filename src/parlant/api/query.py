from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional

from parlant.api.common import apigen_config
from parlant.core.services.tools.service_registry import ServiceRegistry

API_GROUP = "query"

class QueryRequestDTO(BaseModel):
    """Request model for query endpoint"""
    query: str = Field(..., description="The query to process")
    agent_id: str = Field(..., description="The ID of the agent to process the query")

class QueryResponseDTO(BaseModel):
    """Response model for query endpoint"""
    answer: str = Field(..., description="The answer to the query")

def create_router(service_registry: ServiceRegistry) -> APIRouter:
    """Creates a router for the query API"""
    router = APIRouter()

    @router.post(
        "",
        status_code=status.HTTP_200_OK,
        operation_id="process_query",
        response_model=QueryResponseDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Query successfully processed",
                "content": {"application/json": {"example": {"answer": "The answer to your query"}}},
            },
            status.HTTP_404_NOT_FOUND: {"description": "Agent not found"},
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
                "description": "Validation error in request parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="process_query"),
    )
    async def process_query(params: QueryRequestDTO) -> QueryResponseDTO:
        """
        Process a query using the specified agent.
        
        This endpoint takes a query string and an agent ID, then returns the agent's response.
        The agent processes the query according to its configuration and capabilities.
        """
        try:
            answer = await service_registry.process_query(params.query, params.agent_id)
            return QueryResponseDTO(answer=answer)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e)
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )

    return router 