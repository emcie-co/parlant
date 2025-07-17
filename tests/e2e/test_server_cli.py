# Copyright 2025 Emcie Co Ltd.
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
import os
import signal
import pytest
import httpx

from parlant.core.tools import ToolContext, ToolResult
from parlant.core.services.tools.plugins import tool

from tests.e2e.test_utilities import (
    ContextOfTest,
    run_server,
)
from tests.test_utilities import nlp_test, run_service_server


REASONABLE_AMOUNT_OF_TIME = 5
EXTENDED_AMOUNT_OF_TIME = 10


@pytest.mark.parametrize(
    "extra_args",
    [
        [],
        ["--api-mode", "deployment"],
    ],
)
async def test_that_the_server_starts_and_shuts_down_cleanly_on_interrupt(
    context: ContextOfTest,
    extra_args: list[str],
) -> None:
    with run_server(context, extra_args) as server_process:
        await asyncio.sleep(EXTENDED_AMOUNT_OF_TIME)
        server_process.send_signal(signal.SIGINT)
        server_process.wait(timeout=REASONABLE_AMOUNT_OF_TIME)
        assert server_process.returncode == os.EX_OK


async def test_that_the_server_starts_and_generates_a_message(
    context: ContextOfTest,
) -> None:
    with run_server(context):
        await asyncio.sleep(EXTENDED_AMOUNT_OF_TIME)

        agent = await context.api.get_first_agent()
        customer = await context.api.create_customer("test-customer")
        session = await context.api.create_session(agent["id"], customer["id"])

        agent_replies = await context.api.get_agent_replies(
            session_id=session["id"],
            message="Hello",
            number_of_replies_to_expect=1,
        )

        assert await nlp_test(
            agent_replies[0],
            "It greets the customer",
        )


async def test_that_guidelines_are_loaded_after_server_restarts(
    context: ContextOfTest,
) -> None:
    with run_server(context) as server_process:
        await asyncio.sleep(EXTENDED_AMOUNT_OF_TIME)

        first = await context.api.create_guideline(
            condition="the customer greets you",
            action="greet them back with 'Hello'",
        )

        second = await context.api.create_guideline(
            condition="the customer say goodbye",
            action="say goodbye",
        )

        server_process.send_signal(signal.SIGINT)
        server_process.wait(timeout=EXTENDED_AMOUNT_OF_TIME)
        assert server_process.returncode == os.EX_OK

    with run_server(context) as server_process:
        await asyncio.sleep(EXTENDED_AMOUNT_OF_TIME)

        guidelines = await context.api.list_guidelines()

        assert any(first["condition"] == g["condition"] for g in guidelines)
        assert any(first["action"] == g["action"] for g in guidelines)

        assert any(second["condition"] == g["condition"] for g in guidelines)
        assert any(second["action"] == g["action"] for g in guidelines)


async def test_that_context_variable_values_load_after_server_restart(
    context: ContextOfTest,
) -> None:
    variable_name = "test_variable_with_value"
    variable_description = "Variable with values"
    key = "test_key"
    data = "test_value"

    with run_server(context) as server_process:
        await asyncio.sleep(EXTENDED_AMOUNT_OF_TIME)

        variable = await context.api.create_context_variable(variable_name, variable_description)
        await context.api.update_context_variable_value(variable["id"], key, data)

        server_process.send_signal(signal.SIGINT)
        server_process.wait(timeout=EXTENDED_AMOUNT_OF_TIME)
        assert server_process.returncode == os.EX_OK

    with run_server(context):
        await asyncio.sleep(EXTENDED_AMOUNT_OF_TIME)

        variable_value = await context.api.read_context_variable_value(variable["id"], key)

        assert variable_value["data"] == data


async def test_that_services_load_after_server_restart(context: ContextOfTest) -> None:
    service_name = "test_service"
    service_kind = "sdk"

    @tool
    def sample_tool(context: ToolContext, param: int) -> ToolResult:
        return ToolResult(param * 2)

    with run_server(context) as server_process:
        await asyncio.sleep(EXTENDED_AMOUNT_OF_TIME)

        async with run_service_server([sample_tool]) as server:
            await context.api.create_sdk_service(service_name, server.url)

        server_process.send_signal(signal.SIGINT)
        server_process.wait(timeout=EXTENDED_AMOUNT_OF_TIME)
        assert server_process.returncode == os.EX_OK

    with run_server(context):
        await asyncio.sleep(EXTENDED_AMOUNT_OF_TIME)

        services = await context.api.list_services()
        assert any(s["name"] == service_name for s in services)
        assert any(s["kind"] == service_kind for s in services)


async def test_that_glossary_terms_load_after_server_restart(context: ContextOfTest) -> None:
    term_name = "test_term"
    description = "Term added before server restart"

    with run_server(context) as server_process:
        await asyncio.sleep(EXTENDED_AMOUNT_OF_TIME)

        await context.api.create_term(term_name, description)

        server_process.send_signal(signal.SIGINT)
        server_process.wait(timeout=REASONABLE_AMOUNT_OF_TIME)
        assert server_process.returncode == os.EX_OK

    with run_server(context):
        await asyncio.sleep(EXTENDED_AMOUNT_OF_TIME)

        terms = await context.api.list_terms()

        assert any(t["name"] == term_name for t in terms)
        assert any(t["description"] == description for t in terms)


async def test_that_server_starts_with_single_module(context: ContextOfTest) -> None:
    with run_server(context, extra_args=["--module", "tests.modules.tech_store"]):
        await asyncio.sleep(EXTENDED_AMOUNT_OF_TIME)

        agent = await context.api.get_first_agent()

        guideline = await context.api.create_guideline(
            condition="the user asks about product categories",
            action="tell them what product categories are available",
        )
        _ = await context.api.add_association(
            guideline_id=guideline["id"],
            service_name="tech-store",
            tool_name="list_categories",
        )

        session = await context.api.create_session(agent["id"])

        agent_replies = await context.api.get_agent_replies(
            session_id=session["id"],
            message="Hello, what product categories do you have?",
            number_of_replies_to_expect=1,
        )

        assert await nlp_test(
            agent_replies[0]["data"]["message"],
            "laptops and chairs",
        )


async def test_that_server_in_deployment_mode_does_not_allow_creation_calls(
    context: ContextOfTest,
) -> None:
    with run_server(context, extra_args=["--deploy"]):
        with pytest.raises(httpx.HTTPStatusError):
            await context.api.create_guideline(
                condition="the user asks about product categories",
                action="tell them what product categories are available",
            )
        with pytest.raises(httpx.HTTPStatusError):
            await context.api.create_term("name", "description")

        with pytest.raises(httpx.HTTPStatusError):
            await context.api.create_agent(name="test_agent")
