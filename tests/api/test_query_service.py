import pytest
from fastapi import status
import httpx

from parlant.api.common import apigen_config
from parlant.core.services.tools.service_registry import ServiceRegistry

API_GROUP = "query"

@pytest.fixture
def mock_service_registry(mocker):
    return mocker.Mock(spec=ServiceRegistry)

async def test_query_service_success(async_client: httpx.AsyncClient, mock_service_registry):
    # Test data
    query = "What is the weather in London?"
    agent_id = "test-agent-1"
    expected_response = "The weather in London is currently sunny with a temperature of 20°C"
    
    # Mock the service registry response
    mock_service_registry.process_query.return_value = expected_response
    
    # Make the request
    response = await async_client.post(
        "/query",
        json={
            "query": query,
            "agent_id": agent_id
        }
    )
    
    # Assert response
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["answer"] == expected_response
    mock_service_registry.process_query.assert_called_once_with(query, agent_id)

async def test_query_service_missing_query(async_client: httpx.AsyncClient):
    response = await async_client.post(
        "/query",
        json={
            "agent_id": "test-agent-1"
        }
    )
    
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

async def test_query_service_missing_agent_id(async_client: httpx.AsyncClient):
    response = await async_client.post(
        "/query",
        json={
            "query": "What is the weather in London?"
        }
    )
    
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

async def test_query_service_invalid_agent_id(async_client: httpx.AsyncClient, mock_service_registry):
    # Mock service registry to raise an exception for invalid agent
    mock_service_registry.process_query.side_effect = ValueError("Invalid agent ID")
    
    response = await async_client.post(
        "/query",
        json={
            "query": "What is the weather in London?",
            "agent_id": "invalid-agent"
        }
    )
    
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "Invalid agent ID" in response.json()["detail"] 