# Copyright 2026 Emcie Co Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from datetime import datetime, date, timedelta, timezone
from enum import Enum
import uuid
from pathlib import Path
from typing import Any, cast

from parlant.adapters.db.json_file import JSONFileDocumentDatabase
from mcp.types import CallToolResult, TextContent, Tool as McpTool
from parlant.core.services.tools.mcp_service import (
    MCPToolServer,
    MCPToolClient,
    mcp_result_to_tool_result_data,
    mcp_tool_to_parlant_tool,
)
from lagom import Container
from parlant.core.agents import Agent
from parlant.core.emissions import EventEmitterFactory
from parlant.core.tracer import LocalTracer
from parlant.core.loggers import StdoutLogger
from parlant.core.tools import Tool, ToolOverlap
from parlant.core.loggers import Logger
from parlant.sdk import ToolContext
from tests.test_utilities import SERVER_BASE_URL, get_random_port
from parlant.core.services.tools.service_registry import ServiceRegistry, ServiceDocumentRegistry


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


class StubMCPClient:
    def __init__(self, result: CallToolResult) -> None:
        self._result = result
        self.connected = True

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        return self._result

    def is_connected(self) -> bool:
        return self.connected


def create_stubbed_tool_client(result: CallToolResult) -> MCPToolClient:
    client = object.__new__(MCPToolClient)
    client._client = StubMCPClient(result)  # type: ignore[assignment,attr-defined]
    client._client_lock = asyncio.Lock()  # type: ignore[attr-defined]

    async def read_tool(name: str) -> Tool:
        return Tool(
            name=name,
            creation_utc=datetime.now(timezone.utc),
            description="",
            metadata={},
            parameters={},
            required=[],
            consequential=True,
            overlap=ToolOverlap.ALWAYS,
        )

    client.read_tool = read_tool  # type: ignore[method-assign]
    return client


class FastMCPStyleResult:
    def __init__(
        self,
        *,
        content: list[TextContent],
        data: Any = None,
        structured_content: Any = None,
    ) -> None:
        self.content = content
        self.data = data
        self.structured_content = structured_content


async def greet_me_like_pirate(name: str, lucky_number: int, am_i_the_goat: bool = True) -> str:
    message = f"Ahoy {name}! I doubled your lucky number to {lucky_number * 2} !"
    if am_i_the_goat:
        message += " You are the GOAT!"
    return message


async def tool_with_date_and_float(when: datetime, factor: float) -> str:
    assert isinstance(when, datetime), "when must be a datetime"
    assert isinstance(factor, float), "factor must be a float"
    return f"The date is {when.isoformat()} and the factor is {factor}"


async def test_that_simple_mcp_tool_is_listed_and_called(
    container: Container,
    agent: Agent,
) -> None:
    async with MCPToolServer([greet_me_like_pirate], port=get_random_port()) as server:
        client = create_client(server, container)
        async with client:
            tool = await client.read_tool("greet_me_like_pirate")
            assert tool is not None
            result = await client.call_tool(
                tool.name,
                ToolContext("", "", ""),
                {"name": "Short Jon Nickel", "lucky_number": 7},
            )
            assert "Ahoy Short Jon Nickel! I doubled your lucky number to 14 !" in result.data


async def test_that_another_simple_mcp_tool_is_listed_resolved_and_called(
    container: Container,
    agent: Agent,
) -> None:
    async with MCPToolServer(
        [tool_with_date_and_float, greet_me_like_pirate], port=get_random_port()
    ) as server:
        client = create_client(server, container)
        async with client:
            tools = await client.list_tools()
            assert tools is not None and len(tools) == 2
            tool = await client.resolve_tool("tool_with_date_and_float", ToolContext("", "", ""))
            assert tool is not None
            result = await client.call_tool(
                tool.name, ToolContext("", "", ""), {"when": "2025-01-20 12:05", "factor": 2.3}
            )
            assert "The date is 2025-01-20T12:05:00 and the factor is 2.3" in result.data


def test_that_an_mcp_tool_schema_without_required_defaults_to_no_required_parameters() -> None:
    tool = mcp_tool_to_parlant_tool(
        McpTool(
            name="missing_required",
            description="",
            inputSchema={
                "type": "object",
                "properties": {
                    "payload": {"type": "string", "title": "Payload"},
                },
            },
        )
    )

    assert tool.required == []
    assert tool.parameters["payload"][0]["type"] == "string"


def test_that_object_parameters_and_object_arrays_are_degraded_to_string_descriptors() -> None:
    tool = mcp_tool_to_parlant_tool(
        McpTool(
            name="object_params",
            description="",
            inputSchema={
                "type": "object",
                "properties": {
                    "payload": {
                        "type": "object",
                        "title": "Payload",
                    },
                    "items": {
                        "type": "array",
                        "title": "Items",
                        "items": {"type": "object"},
                    },
                },
                "required": [],
            },
        )
    )

    assert tool.parameters["payload"][0]["type"] == "string"
    assert tool.parameters["items"][0]["type"] == "array"
    assert tool.parameters["items"][0]["item_type"] == "string"


async def test_that_an_mcp_tool_can_be_called_with_enum_and_bool_lists(
    container: Container,
    agent: Agent,
) -> None:
    class JustEnum(Enum):
        a = "a"
        b = "b"
        c = "c"

    def tool_with_two_lists(
        enum_list: list[JustEnum],
        bool_list: list[bool],
    ) -> str:
        return f"The enum list is {enum_list} and the bool list is {bool_list}"

    async with MCPToolServer([tool_with_two_lists], port=get_random_port()) as server:
        client = create_client(server, container)
        async with client:
            tool = await client.read_tool("tool_with_two_lists")
            assert tool is not None
            result = await client.call_tool(
                tool.name,
                ToolContext("", "", ""),
                {"enum_list": ["a", "b", "c", "a"], "bool_list": [True, False, True]},
            )
            assert "The enum list is" in result.data


async def test_that_mcp_call_tool_preserves_multiple_text_blocks() -> None:
    client = create_stubbed_tool_client(
        CallToolResult(
            content=[
                TextContent(type="text", text="alpha"),
                TextContent(type="text", text="beta"),
            ]
        )
    )

    result = await client.call_tool("multi_text", ToolContext("", "", ""), {})

    assert result.data == ["alpha", "beta"]


def test_that_mcp_result_data_is_deserialized_from_json_text() -> None:
    data = mcp_result_to_tool_result_data(
        CallToolResult(
            content=[
                TextContent(type="text", text='{"ok": true, "items": [1, 2]}'),
            ]
        )
    )

    assert data == {"ok": True, "items": [1, 2]}


def test_that_mcp_result_data_uses_native_fastmcp_data_when_available() -> None:
    data = mcp_result_to_tool_result_data(
        FastMCPStyleResult(
            content=[TextContent(type="text", text="33")],
            data=33,
            structured_content={"result": 33},
        )
    )

    assert data == 33


async def test_that_mcp_call_tool_prefers_structured_content_over_serialized_text() -> None:
    client = create_stubbed_tool_client(
        CallToolResult(
            content=[
                TextContent(type="text", text='"{\\"ok\\": true}"'),
            ],
            structuredContent={"ok": True},
        )
    )

    result = await client.call_tool("structured", ToolContext("", "", ""), {})

    assert result.data == {"ok": True}


async def test_that_an_mcp_tool_can_be_called_with_a_list_of_date_and_datetime(
    container: Container,
    agent: Agent,
) -> None:
    def tool_with_date_list_and_datetime(
        date_list: list[date],
        date_time: datetime,
    ) -> str:
        return f"The dates are {date_list} and the datetime is {date_time}"

    async with MCPToolServer([tool_with_date_list_and_datetime], port=get_random_port()) as server:
        client = create_client(server, container)
        async with client:
            tool = await client.read_tool("tool_with_date_list_and_datetime")
            assert tool is not None
            result = await client.call_tool(
                tool.name,
                ToolContext("", "", ""),
                {
                    "date_list": [
                        "2025-05-25",
                        "2020-10-10",
                    ],
                    "date_time": "1948-05-14 16:00",
                },
            )
            assert "The dates are" in result.data


async def test_that_an_mcp_tool_can_be_called_with_timedelta_path_and_uuid(
    container: Container,
    agent: Agent,
) -> None:
    def tool_with_timedelta_path_and_uuid(
        delta: timedelta,
        path: Path,
        uuid: uuid.UUID,
    ) -> str:
        return f"uuid {uuid} reports it took {delta} seconds to navigate to {path}"

    async with MCPToolServer([tool_with_timedelta_path_and_uuid], port=get_random_port()) as server:
        client = create_client(server, container)
        async with client:
            tool = await client.read_tool("tool_with_timedelta_path_and_uuid")
            assert tool is not None
            result = await client.call_tool(
                tool.name,
                ToolContext("", "", ""),
                {
                    "uuid": str(uuid.uuid1()),
                    "delta": str(timedelta(seconds=10)),
                    "path": str(Path("/dev/null")),
                },
            )
            assert "reports it took" in result.data


async def test_that_reading_an_existing_mcp_service_returns_its_tools_and_can_call_them(
    container: Container,
) -> None:
    def my_tool(arg_1: int, arg_2: int) -> int:
        return arg_1 + arg_2

    async def my_async_tool(message: str) -> str:
        return f"Echo: {message}"

    service_registry = container[ServiceRegistry]

    async with MCPToolServer([my_tool, my_async_tool]) as server:
        await service_registry.update_tool_service(
            name="my_mcp_service",
            kind="mcp",
            url=f"{SERVER_BASE_URL}:{server.get_port()}",
        )

        await service_registry.list_tool_services()

        # service_data = (await service_registry.list_tool_services()).raise_for_status().json()
        service = await service_registry.read_tool_service("my_mcp_service")

        tools_list = await service.list_tools()
        assert len(tools_list) == 2
        assert "my_tool" in [t.name for t in tools_list]
        assert "my_async_tool" in [t.name for t in tools_list]

        result = await service.call_tool(
            "my_tool", ToolContext("", "", ""), {"arg_1": 11, "arg_2": 22}
        )
        assert str(result.data) == "33"

        result = await service.call_tool(
            "my_async_tool", ToolContext("", "", ""), {"message": "Hello"}
        )
        assert str(result.data) == "Echo: Hello"


async def test_that_updating_an_mcp_service_closes_the_previous_client_connection(
    container: Container,
) -> None:
    service_registry = container[ServiceRegistry]

    async with MCPToolServer([greet_me_like_pirate], port=get_random_port()) as first_server:
        first_service = await service_registry.update_tool_service(
            name="my_mcp_service",
            kind="mcp",
            url=f"{SERVER_BASE_URL}:{first_server.get_port()}",
        )

        first_client = cast(MCPToolClient, first_service)._client
        assert first_client is not None and first_client.is_connected()

        async with MCPToolServer(
            [tool_with_date_and_float], port=get_random_port()
        ) as second_server:
            second_service = await service_registry.update_tool_service(
                name="my_mcp_service",
                kind="mcp",
                url=f"{SERVER_BASE_URL}:{second_server.get_port()}",
            )

            assert first_client is not None and not first_client.is_connected()
            second_client = cast(MCPToolClient, second_service)._client
            assert second_client is not None and second_client.is_connected()


async def test_that_mcp_service_endpoint_roundtrips_through_persistence(
    container: Container,
    logger: Logger,
    tmp_path: Path,
) -> None:
    service_url = ""

    async with MCPToolServer([greet_me_like_pirate], port=get_random_port()) as server:
        service_url = f"{SERVER_BASE_URL}:{server.get_port()}"

        async with JSONFileDocumentDatabase(logger, tmp_path / "services.json") as database:
            async with ServiceDocumentRegistry(
                database=database,
                event_emitter_factory=container[EventEmitterFactory],
                logger=logger,
                tracer=LocalTracer(),
                nlp_services_provider=lambda: {},
            ) as registry:
                service = await registry.update_tool_service(
                    name="my_mcp_service",
                    kind="mcp",
                    url=service_url,
                )

                assert isinstance(service, MCPToolClient)
                assert service.endpoint_url == service_url

        async with JSONFileDocumentDatabase(logger, tmp_path / "services.json") as database:
            async with ServiceDocumentRegistry(
                database=database,
                event_emitter_factory=container[EventEmitterFactory],
                logger=logger,
                tracer=LocalTracer(),
                nlp_services_provider=lambda: {},
            ) as restored_registry:
                restored_service = await restored_registry.read_tool_service("my_mcp_service")

                assert isinstance(restored_service, MCPToolClient)
                assert restored_service.endpoint_url == service_url


async def test_that_mcp_service_port_is_preserved_through_persistence_roundtrip(
    container: Container,
    logger: Logger,
    tmp_path: Path,
) -> None:
    service_url = ""

    async with MCPToolServer([greet_me_like_pirate], port=get_random_port()) as server:
        service_url = f"{SERVER_BASE_URL}:{server.get_port()}"

        async with JSONFileDocumentDatabase(logger, tmp_path / "services.json") as database:
            async with ServiceDocumentRegistry(
                database=database,
                event_emitter_factory=container[EventEmitterFactory],
                logger=logger,
                tracer=LocalTracer(),
                nlp_services_provider=lambda: {},
            ) as registry:
                service = await registry.update_tool_service(
                    name="my_mcp_service",
                    kind="mcp",
                    url=service_url,
                )

                assert isinstance(service, MCPToolClient)
                assert f"{service.url}:{service.port}" == service_url

        async with JSONFileDocumentDatabase(logger, tmp_path / "services.json") as database:
            async with ServiceDocumentRegistry(
                database=database,
                event_emitter_factory=container[EventEmitterFactory],
                logger=logger,
                tracer=LocalTracer(),
                nlp_services_provider=lambda: {},
            ) as restored_registry:
                restored_service = await restored_registry.read_tool_service("my_mcp_service")

                assert isinstance(restored_service, MCPToolClient)
                assert f"{restored_service.url}:{restored_service.port}" == service_url
