import asyncio
import json
from typing import Any

from pydantic import BaseModel
import pytest

from parlant.core.container import Container

from parlant.core.agents import Agent
from parlant.core.emissions import EventEmitterFactory
from parlant.core.loggers import StdoutLogger
from parlant.core.services.tools.mcp_service import MCPToolClient, MCPToolServer
from parlant.core.tracer import LocalTracer
from parlant.sdk import ToolContext
from tests.test_utilities import SERVER_BASE_URL, get_random_port


def create_client(
    server: MCPToolServer,
    container: Container,
) -> MCPToolClient:
    tracer = LocalTracer()
    logger = StdoutLogger(tracer)
    return MCPToolClient(
        url=SERVER_BASE_URL,
        event_emitter_factory=container[EventEmitterFactory],
        logger=logger,
        tracer=tracer,
        port=server._port,
    )


async def greet_me_like_pirate(name: str, lucky_number: int, am_i_the_goat: bool = True) -> str:
    message = f"Ahoy {name}! I doubled your lucky number to {lucky_number * 2} !"
    if am_i_the_goat:
        message += " You are the GOAT!"
    return message


async def test_that_mcp_client_reconnects_after_its_session_is_closed(
    container: Container,
    agent: Agent,
) -> None:
    async with MCPToolServer([greet_me_like_pirate], port=get_random_port()) as server:
        client = create_client(server, container)

        async with client:
            result = await client.call_tool(
                "greet_me_like_pirate",
                ToolContext("", "", ""),
                {"name": "Short Jon Nickel", "lucky_number": 7},
            )
            assert "Ahoy Short Jon Nickel! I doubled your lucky number to 14 !" in result.data

            assert client._client is not None
            await client._client.close()  # type: ignore[no-untyped-call]

            reconnected_result = await client.call_tool(
                "greet_me_like_pirate",
                ToolContext("", "", ""),
                {"name": "Another Pirate", "lucky_number": 9},
            )

            assert (
                "Ahoy Another Pirate! I doubled your lucky number to 18 !"
                in reconnected_result.data
            )


async def test_that_mcp_client_retries_initial_connection(
    container: Container,
) -> None:
    client = MCPToolClient(
        url=SERVER_BASE_URL,
        event_emitter_factory=container[EventEmitterFactory],
        logger=StdoutLogger(LocalTracer()),
        tracer=LocalTracer(),
        port=get_random_port(),
    )

    class FakeClient:
        def __init__(self, should_fail: bool) -> None:
            self.should_fail = should_fail
            self.connected = False

        async def __aenter__(self) -> "FakeClient":
            if self.should_fail:
                raise asyncio.TimeoutError()
            self.connected = True
            return self

        async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
            self.connected = False
            return False

        def is_connected(self) -> bool:
            return self.connected

    attempted_clients: list[FakeClient] = []

    def fake_create_client() -> FakeClient:
        fake_client = FakeClient(should_fail=not attempted_clients)
        attempted_clients.append(fake_client)
        return fake_client

    client._create_client = fake_create_client  # type: ignore[method-assign, assignment]

    async with client:
        assert len(attempted_clients) == 2
        assert attempted_clients[-1].is_connected()


class Order(BaseModel):
    order_id: str
    status: str


async def get_order(order_id: str) -> Order:
    return Order(order_id=order_id, status="shipped")


async def is_in_stock(sku: str) -> bool:
    return False


async def count_returns(sku: str) -> int:
    return 0


async def list_discounts(sku: str) -> dict[str, int]:
    return {}


async def test_that_a_structured_mcp_tool_result_is_returned_as_json_native_data(
    container: Container,
) -> None:
    async with MCPToolServer([get_order], port=get_random_port()) as server:
        async with create_client(server, container) as client:
            result = await client.call_tool(
                "get_order",
                ToolContext("", "", ""),
                {"order_id": "A-1"},
            )

    assert result.data == {"order_id": "A-1", "status": "shipped"}
    assert json.loads(json.dumps(result.data)) == result.data


@pytest.mark.parametrize(
    ("tool_name", "expected"),
    [
        ("is_in_stock", False),
        ("count_returns", 0),
        ("list_discounts", {}),
    ],
)
async def test_that_falsey_structured_mcp_results_are_preserved(
    container: Container,
    tool_name: str,
    expected: Any,
) -> None:
    async with MCPToolServer(
        [is_in_stock, count_returns, list_discounts], port=get_random_port()
    ) as server:
        async with create_client(server, container) as client:
            result = await client.call_tool(tool_name, ToolContext("", "", ""), {"sku": "S-1"})

    assert result.data == expected
    assert type(result.data) is type(expected)
