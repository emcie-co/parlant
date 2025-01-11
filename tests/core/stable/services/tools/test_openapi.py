# Copyright 2024 Emcie Co Ltd.
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

from typing import Any
from pytest import mark

from parlant.core.tools import ToolContext
from parlant.core.services.tools.openapi import OpenAPIClient

from tests.test_utilities import (
    OPENAPI_SERVER_URL,
    TOOLS,
    get_openapi_spec,
    one_required_body_param,
    one_required_query_param,
    one_required_query_param_one_required_body_param,
    rng_app,
    run_openapi_server,
    two_required_body_params,
    two_required_query_params,
)


async def test_that_tools_are_exposed_via_an_openapi_server() -> None:
    async with run_openapi_server(rng_app()):
        openapi_json = await get_openapi_spec(OPENAPI_SERVER_URL)

        async with OpenAPIClient(OPENAPI_SERVER_URL, openapi_json) as client:
            tools = await client.list_tools()

            for tool_name, tool in {t.__name__: t for t in TOOLS}.items():
                listed_tool = next((t for t in tools if t.name == tool_name), None)
                assert listed_tool


async def test_that_tools_can_be_read_via_an_openapi_server() -> None:
    async with run_openapi_server(rng_app()):
        openapi_json = await get_openapi_spec(OPENAPI_SERVER_URL)

        async with OpenAPIClient(OPENAPI_SERVER_URL, openapi_json) as client:
            tools = await client.list_tools()

            for t in tools:
                assert (await client.read_tool(t.name)) == t


@mark.parametrize(
    ["tool_name", "tool_args", "expected_result"],
    [
        (
            one_required_query_param.__name__,
            {"query_param": 123},
            {"result": 123},
        ),
        (
            two_required_query_params.__name__,
            {"query_param_1": 123, "query_param_2": 321},
            {"result": 123 + 321},
        ),
        (
            one_required_body_param.__name__,
            {"body_param": "hello"},
            {"result": "hello"},
        ),
        (
            two_required_body_params.__name__,
            {"body_param_1": "hello ", "body_param_2": "world"},
            {"result": "hello world"},
        ),
        (
            one_required_query_param_one_required_body_param.__name__,
            {"body_param": "banana", "query_param": 123},
            {"result": "banana: 123"},
        ),
    ],
)
async def test_that_a_tool_can_be_called_via_an_openapi_server(
    tool_name: str,
    tool_args: dict[str, Any],
    expected_result: Any,
) -> None:
    async with run_openapi_server(rng_app()):
        openapi_json = await get_openapi_spec(OPENAPI_SERVER_URL)

        async with OpenAPIClient(OPENAPI_SERVER_URL, openapi_json) as client:
            stub_context = ToolContext(
                agent_id="test-agent",
                session_id="test_session",
                customer_id="test_customer",
            )
            result = await client.call_tool(tool_name, stub_context, tool_args)
            assert result.data == expected_result
