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

import asyncio
from contextlib import asynccontextmanager
import json
import os
import tempfile
from typing import Any, AsyncIterator, Awaitable, Callable
from fastapi import FastAPI, Query, Request, Response
from fastapi.responses import JSONResponse
import httpx
import uvicorn

from tests.e2e.test_utilities import (
    CLI_CLIENT_PATH,
    SERVER_ADDRESS,
    SERVER_PORT,
    ContextOfTest,
    is_server_responsive,
    run_server,
)
from parlant.core.services.tools.plugins import tool, ToolEntry, PluginServer
from parlant.core.tools import ToolResult, ToolContext

REASONABLE_AMOUNT_OF_TIME_FOR_TERM_CREATION = 0.25

OPENAPI_SERVER_PORT = 8091
OPENAPI_SERVER_URL = f"http://localhost:{OPENAPI_SERVER_PORT}"


@asynccontextmanager
async def run_openapi_server(
    app: FastAPI,
) -> AsyncIterator[None]:
    config = uvicorn.Config(app=app, port=OPENAPI_SERVER_PORT)
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    yield
    server.should_exit = True
    await task


async def one_required_query_param(
    query_param: int = Query(),
) -> JSONResponse:
    return JSONResponse({"result": query_param})


async def two_required_query_params(
    query_param_1: int = Query(),
    query_param_2: int = Query(),
) -> JSONResponse:
    return JSONResponse({"result": query_param_1 + query_param_2})


TOOLS = (
    one_required_query_param,
    two_required_query_params,
)


def rng_app() -> FastAPI:
    app = FastAPI(servers=[{"url": OPENAPI_SERVER_URL}])

    @app.middleware("http")
    async def debug_request(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        return response

    for t in TOOLS:
        registration_func = app.post if "body" in t.__name__ else app.get
        registration_func(f"/{t.__name__}", operation_id=t.__name__)(t)

    return app


@asynccontextmanager
async def run_service_server(
    tools: list[ToolEntry],
) -> AsyncIterator[PluginServer]:
    async with PluginServer(
        tools=tools,
        port=8091,
        host="127.0.0.1",
    ) as server:
        try:
            yield server
        finally:
            await server.shutdown()


async def run_cli(*args: str, **kwargs: Any) -> asyncio.subprocess.Process:
    exec_args = [
        "poetry",
        "run",
        "python",
        CLI_CLIENT_PATH.as_posix(),
        "--server",
        SERVER_ADDRESS,
    ] + list(args)

    return await asyncio.create_subprocess_exec(*exec_args, **kwargs)


async def run_cli_and_get_exit_status(*args: str) -> int:
    exec_args = [
        "poetry",
        "run",
        "python",
        CLI_CLIENT_PATH.as_posix(),
        "--server",
        SERVER_ADDRESS,
    ] + list(args)

    process = await asyncio.create_subprocess_exec(*exec_args)
    return await process.wait()


async def test_that_an_agent_can_be_added(context: ContextOfTest) -> None:
    name = "TestAgent"
    description = "This is a test agent"

    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        process = await run_cli(
            "agent",
            "create",
            "--name",
            name,
            "--description",
            description,
            "--max-engine-iterations",
            str(123),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        agents = await context.api.list_agents()
        new_agent = next((a for a in agents if a["name"] == name), None)
        assert new_agent
        assert new_agent["description"] == description
        assert new_agent["max_engine_iterations"] == 123

        process = await run_cli(
            "agent",
            "create",
            "--name",
            "Test Agent With No Description",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        agents = await context.api.list_agents()
        new_agent_no_desc = next(
            (a for a in agents if a["name"] == "Test Agent With No Description"), None
        )
        assert new_agent_no_desc
        assert new_agent_no_desc["description"] is None


async def test_that_an_agent_can_be_updated(
    context: ContextOfTest,
) -> None:
    new_name = "Updated Agent"
    new_description = "Updated description"
    new_max_engine_iterations = 5

    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        process = await run_cli(
            "agent",
            "update",
            "--name",
            new_name,
            "--description",
            new_description,
            "--max-engine-iterations",
            str(new_max_engine_iterations),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        agent = (await context.api.list_agents())[0]
        assert agent["name"] == new_name
        assert agent["description"] == new_description
        assert agent["max_engine_iterations"] == new_max_engine_iterations


async def test_that_an_agent_can_be_deleted(
    context: ContextOfTest,
) -> None:
    name = "Test Agent"

    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        agent = await context.api.create_agent(name=name)

        assert (
            await run_cli_and_get_exit_status(
                "agent",
                "delete",
                "--id",
                agent["id"],
            )
            == os.EX_OK
        )

        assert not any(a["name"] == name for a in await context.api.list_agents())


async def test_that_an_agent_can_be_viewed(
    context: ContextOfTest,
) -> None:
    name = "Test Agent"
    description = "Bananas"
    max_engine_iterations = 2

    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        agent = await context.api.create_agent(
            name=name,
            description=description,
            max_engine_iterations=max_engine_iterations,
        )

        process = await run_cli(
            "agent",
            "view",
            "--id",
            agent["id"],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert process.returncode == os.EX_OK

        assert agent["id"] in output_view
        assert name in output_view
        assert description in output_view
        assert str(max_engine_iterations) in output_view


async def test_that_sessions_can_be_listed(
    context: ContextOfTest,
) -> None:
    first_customer = "First Customer"
    second_customer = "Second Customer"

    first_title = "First Title"
    second_title = "Second Title"
    third_title = "Third Title"

    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        agent_id = (await context.api.get_first_agent())["id"]
        _ = await context.api.create_session(
            agent_id=agent_id, customer_id=first_customer, title=first_title
        )
        _ = await context.api.create_session(
            agent_id=agent_id, customer_id=first_customer, title=second_title
        )
        _ = await context.api.create_session(
            agent_id=agent_id, customer_id=second_customer, title=third_title
        )

        process = await run_cli(
            "session",
            "list",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        output_list = stdout.decode() + stderr.decode()
        assert process.returncode == os.EX_OK

        assert first_title in output_list
        assert second_title in output_list
        assert third_title in output_list

        process = await run_cli(
            "session",
            "list",
            "--customer-id",
            first_customer,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        output_list = stdout.decode() + stderr.decode()
        assert process.returncode == os.EX_OK

        assert first_title in output_list
        assert second_title in output_list
        assert third_title not in output_list


async def test_that_session_can_be_updated(
    context: ContextOfTest,
) -> None:
    session_title = "Old Title"

    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        agent_id = (await context.api.get_first_agent())["id"]
        session_id = (await context.api.create_session(agent_id=agent_id, title=session_title))[
            "id"
        ]

        assert (
            await run_cli_and_get_exit_status(
                "session",
                "update",
                "--id",
                session_id,
                "--title",
                "New Title",
            )
            == os.EX_OK
        )

        session = await context.api.read_session(session_id)
        assert session["title"] == "New Title"


async def test_that_a_term_can_be_created_with_synonyms(
    context: ContextOfTest,
) -> None:
    term_name = "guideline"
    description = "when and then statements"
    synonyms = "rule, principle"

    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        agent_id = (await context.api.get_first_agent())["id"]

        process = await run_cli(
            "glossary",
            "create",
            "--agent-id",
            agent_id,
            "--name",
            term_name,
            "--description",
            description,
            "--synonyms",
            synonyms,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK


async def test_that_a_term_can_be_created_without_synonyms(
    context: ContextOfTest,
) -> None:
    term_name = "guideline_no_synonyms"
    description = "simple guideline with no synonyms"

    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        agent_id = (await context.api.get_first_agent())["id"]

        process = await run_cli(
            "glossary",
            "create",
            "--agent-id",
            agent_id,
            "--name",
            term_name,
            "--description",
            description,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        terms = await context.api.list_terms(agent_id)
        assert any(t["name"] == term_name for t in terms)
        assert any(t["description"] == description for t in terms)
        assert any(t["synonyms"] == [] for t in terms)


async def test_that_a_term_can_be_updated(
    context: ContextOfTest,
) -> None:
    name = "guideline"
    description = "when and then statements"
    synonyms = "rule, principle"

    new_name = "updated guideline"
    new_description = "then and when statements "
    new_synonyms = "instructions"

    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        agent_id = (await context.api.get_first_agent())["id"]

        term_to_update = await context.api.create_term(agent_id, name, description, synonyms)

        process = await run_cli(
            "glossary",
            "update",
            "--agent-id",
            agent_id,
            "--id",
            term_to_update["id"],
            "--name",
            new_name,
            "--description",
            new_description,
            "--synonyms",
            new_synonyms,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        updated_term = await context.api.read_term(agent_id=agent_id, term_id=term_to_update["id"])
        assert updated_term["name"] == new_name
        assert updated_term["description"] == new_description
        assert updated_term["synonyms"] == [new_synonyms]


async def test_that_a_term_can_be_deleted(
    context: ContextOfTest,
) -> None:
    name = "guideline_delete"
    description = "to be deleted"
    synonyms = "rule, principle"

    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        agent_id = (await context.api.get_first_agent())["id"]

        term = await context.api.create_term(agent_id, name, description, synonyms)

        process = await run_cli(
            "glossary",
            "delete",
            "--agent-id",
            agent_id,
            "--id",
            term["id"],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        terms = await context.api.list_terms(agent_id)
        assert len(terms) == 0


async def test_that_a_guideline_can_be_added(
    context: ContextOfTest,
) -> None:
    condition = "the customer greets you"
    action = "greet them back with 'Hello'"

    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        agent_id = (await context.api.get_first_agent())["id"]

        process = await run_cli(
            "guideline",
            "create",
            "--agent-id",
            agent_id,
            "--condition",
            condition,
            "--action",
            action,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        guidelines = await context.api.list_guidelines(agent_id)
        assert any(g["condition"] == condition and g["action"] == action for g in guidelines)


async def test_that_a_guideline_can_be_updated(
    context: ContextOfTest,
) -> None:
    condition = "the customer asks for help"
    initial_action = "offer assistance"
    updated_action = "provide detailed support information"

    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        agent_id = (await context.api.get_first_agent())["id"]

        guideline = await context.api.create_guideline(
            agent_id=agent_id, condition=condition, action=initial_action
        )

        process = await run_cli(
            "guideline",
            "update",
            "--agent-id",
            agent_id,
            "--id",
            guideline["id"],
            "--condition",
            condition,
            "--action",
            updated_action,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        updated_guideline = (
            await context.api.read_guideline(agent_id=agent_id, guideline_id=guideline["id"])
        )["guideline"]

        assert updated_guideline["condition"] == condition
        assert updated_guideline["action"] == updated_action


async def test_that_adding_a_contradictory_guideline_shows_coherence_errors(
    context: ContextOfTest,
) -> None:
    condition = "the customer greets you"
    action = "greet them back with 'Hello'"

    conflicting_action = "ignore the customer"

    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        agent_id = (await context.api.get_first_agent())["id"]

        process = await run_cli(
            "guideline",
            "create",
            "--agent-id",
            agent_id,
            "--condition",
            condition,
            "--action",
            action,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        process = await run_cli(
            "guideline",
            "create",
            "--agent-id",
            agent_id,
            "--condition",
            condition,
            "--action",
            conflicting_action,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()
        output = stdout.decode() + stderr.decode()

        assert "Detected potential incoherence with other guidelines" in output

        guidelines = await context.api.list_guidelines(agent_id)

        assert not any(
            g["condition"] == condition and g["action"] == conflicting_action for g in guidelines
        )


async def test_that_adding_connected_guidelines_creates_connections(
    context: ContextOfTest,
) -> None:
    condition1 = "the customer asks about the weather"
    action1 = "provide a weather update"

    condition2 = "providing a weather update"
    action2 = "include temperature and humidity"

    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        agent_id = (await context.api.get_first_agent())["id"]

        process = await run_cli(
            "guideline",
            "create",
            "--agent-id",
            agent_id,
            "--condition",
            condition1,
            "--action",
            action1,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        process = await run_cli(
            "guideline",
            "create",
            "--agent-id",
            agent_id,
            "--condition",
            condition2,
            "--action",
            action2,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        guidelines = await context.api.list_guidelines(agent_id)

        assert len(guidelines) == 2
        source = guidelines[0]
        target = guidelines[1]

        source_guideline = await context.api.read_guideline(agent_id, source["id"])
        source_connections = source_guideline["connections"]

        assert len(source_connections) == 1
        connection = source_connections[0]

        assert connection["source"] == source
        assert connection["target"] == target
        assert connection["kind"] == "entails"


async def test_that_a_guideline_can_be_viewed(
    context: ContextOfTest,
) -> None:
    condition = "the customer says goodbye"
    action = "say 'Goodbye' back"

    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        agent_id = (await context.api.get_first_agent())["id"]

        guideline = await context.api.create_guideline(
            agent_id=agent_id, condition=condition, action=action
        )

        process = await run_cli(
            "guideline",
            "view",
            "--agent-id",
            agent_id,
            "--id",
            guideline["id"],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        output = stdout.decode() + stderr.decode()
        assert process.returncode == os.EX_OK

        assert guideline["id"] in output
        assert condition in output
        assert action in output


async def test_that_guidelines_can_be_listed(
    context: ContextOfTest,
) -> None:
    condition1 = "the customer asks for help"
    action1 = "provide assistance"

    condition2 = "the customer needs support"
    action2 = "offer support"

    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        agent_id = (await context.api.get_first_agent())["id"]

        _ = await context.api.create_guideline(
            agent_id=agent_id, condition=condition1, action=action1
        )
        _ = await context.api.create_guideline(
            agent_id=agent_id, condition=condition2, action=action2
        )

        process = await run_cli(
            "guideline",
            "list",
            "--agent-id",
            agent_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        output_list = stdout.decode() + stderr.decode()
        assert process.returncode == os.EX_OK

        assert condition1 in output_list
        assert action1 in output_list
        assert condition2 in output_list
        assert action2 in output_list


async def test_that_guidelines_can_be_entailed(
    context: ContextOfTest,
) -> None:
    condition1 = "the customer needs assistance"
    action1 = "provide help"

    condition2 = "customer ask about a certain subject"
    action2 = "offer detailed explanation"

    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        agent_id = (await context.api.get_first_agent())["id"]

        process = await run_cli(
            "guideline",
            "create",
            "--agent-id",
            agent_id,
            "--no-check",
            "--no-connect",
            "--condition",
            condition1,
            "--action",
            action1,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        process = await run_cli(
            "guideline",
            "create",
            "--agent-id",
            agent_id,
            "--no-check",
            "--no-connect",
            "--condition",
            condition2,
            "--action",
            action2,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        guidelines = await context.api.list_guidelines(agent_id)

        first_guideline = next(
            g for g in guidelines if g["condition"] == condition1 and g["action"] == action1
        )
        second_guideline = next(
            g for g in guidelines if g["condition"] == condition2 and g["action"] == action2
        )

        process = await run_cli(
            "guideline",
            "entail",
            "--agent-id",
            agent_id,
            "--source",
            first_guideline["id"],
            "--target",
            second_guideline["id"],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await process.communicate()
        await process.wait()
        assert process.returncode == os.EX_OK

        guideline = await context.api.read_guideline(agent_id, first_guideline["id"])
        assert "connections" in guideline and len(guideline["connections"]) == 1
        connection = guideline["connections"][0]
        assert (
            connection["source"] == first_guideline
            and connection["target"] == second_guideline
            and connection["kind"] == "entails"
        )


async def test_that_guidelines_can_be_suggestively_entailed(
    context: ContextOfTest,
) -> None:
    condition1 = "the customer needs assistance"
    action1 = "provide help"

    condition2 = "customer ask about a certain subject"
    action2 = "offer detailed explanation"

    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        agent_id = (await context.api.get_first_agent())["id"]

        process = await run_cli(
            "guideline",
            "create",
            "--agent-id",
            agent_id,
            "--no-check",
            "--no-connect",
            "--condition",
            condition1,
            "--action",
            action1,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        process = await run_cli(
            "guideline",
            "create",
            "--agent-id",
            agent_id,
            "--no-check",
            "--no-connect",
            "--condition",
            condition2,
            "--action",
            action2,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        guidelines = await context.api.list_guidelines(agent_id)

        first_guideline = next(
            g for g in guidelines if g["condition"] == condition1 and g["action"] == action1
        )
        second_guideline = next(
            g for g in guidelines if g["condition"] == condition2 and g["action"] == action2
        )

        process = await run_cli(
            "guideline",
            "entail",
            "--agent-id",
            agent_id,
            "--suggestive",
            "--source",
            first_guideline["id"],
            "--target",
            second_guideline["id"],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await process.communicate()
        await process.wait()
        assert process.returncode == os.EX_OK

        guideline = await context.api.read_guideline(agent_id, first_guideline["id"])

        assert "connections" in guideline and len(guideline["connections"]) == 1
        connection = guideline["connections"][0]
        assert (
            connection["source"] == first_guideline
            and connection["target"] == second_guideline
            and connection["kind"] == "suggests"
        )


async def test_that_a_guideline_can_be_removed(
    context: ContextOfTest,
) -> None:
    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        agent_id = (await context.api.get_first_agent())["id"]

        guideline = await context.api.create_guideline(
            agent_id, condition="the customer greets you", action="greet them back with 'Hello'"
        )

        process = await run_cli(
            "guideline",
            "delete",
            "--agent-id",
            agent_id,
            "--id",
            guideline["id"],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        guidelines = await context.api.list_guidelines(agent_id)
        assert len(guidelines) == 0


async def test_that_a_connection_can_be_removed(
    context: ContextOfTest,
) -> None:
    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        agent_id = (await context.api.get_first_agent())["id"]

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(30),
        ) as client:
            guidelines_response = await client.post(
                f"{SERVER_ADDRESS}/agents/{agent_id}/guidelines/",
                json={
                    "invoices": [
                        {
                            "payload": {
                                "kind": "guideline",
                                "guideline": {
                                    "content": {
                                        "condition": "the customer greets you",
                                        "action": "greet them back with 'Hello'",
                                    },
                                    "operation": "add",
                                    "coherence_check": True,
                                    "connection_proposition": True,
                                },
                            },
                            "checksum": "checksum_value",
                            "approved": True,
                            "data": {
                                "guideline": {
                                    "coherence_checks": [],
                                    "connection_propositions": [
                                        {
                                            "check_kind": "connection_with_another_evaluated_guideline",
                                            "source": {
                                                "condition": "the customer greets you",
                                                "action": "greet them back with 'Hello'",
                                            },
                                            "target": {
                                                "condition": "greeting the customer",
                                                "action": "ask for his health condition",
                                            },
                                            "connection_kind": "entails",
                                        }
                                    ],
                                },
                            },
                            "error": None,
                        },
                        {
                            "payload": {
                                "kind": "guideline",
                                "guideline": {
                                    "content": {
                                        "condition": "greeting the customer",
                                        "action": "ask for his health condition",
                                    },
                                    "operation": "add",
                                    "coherence_check": True,
                                    "connection_proposition": True,
                                },
                            },
                            "checksum": "checksum_value",
                            "approved": True,
                            "data": {
                                "guideline": {
                                    "coherence_checks": [],
                                    "connection_propositions": [
                                        {
                                            "check_kind": "connection_with_another_evaluated_guideline",
                                            "source": {
                                                "condition": "the customer greets you",
                                                "action": "greet them back with 'Hello'",
                                            },
                                            "target": {
                                                "condition": "greeting the customer",
                                                "action": "ask for his health condition",
                                            },
                                            "connection_kind": "entails",
                                        }
                                    ],
                                },
                            },
                            "error": None,
                        },
                    ]
                },  # type: ignore
            )

            guidelines_response.raise_for_status()
            first = guidelines_response.json()["items"][0]["guideline"]["id"]
            second = guidelines_response.json()["items"][1]["guideline"]["id"]

        process = await run_cli(
            "guideline",
            "disentail",
            "--agent-id",
            agent_id,
            "--source",
            first,
            "--target",
            second,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        guideline = await context.api.read_guideline(agent_id, first)
        assert len(guideline["connections"]) == 0


async def test_that_a_tool_can_be_enabled_for_a_guideline(
    context: ContextOfTest,
) -> None:
    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        agent_id = (await context.api.get_first_agent())["id"]

        guideline = await context.api.create_guideline(
            agent_id,
            condition="the customer wants to get meeting details",
            action="get meeting event information",
        )

        service_name = "google_calendar"
        tool_name = "fetch_event_data"
        service_kind = "sdk"

        @tool
        def fetch_event_data(context: ToolContext, event_id: str) -> ToolResult:
            """Fetch event data based on event ID."""
            return ToolResult({"event_id": event_id})

        async with run_service_server([fetch_event_data]) as server:
            assert (
                await run_cli_and_get_exit_status(
                    "service",
                    "create",
                    "--name",
                    service_name,
                    "--kind",
                    service_kind,
                    "--url",
                    server.url,
                )
                == os.EX_OK
            )

            assert (
                await run_cli_and_get_exit_status(
                    "guideline",
                    "tool-enable",
                    "--agent-id",
                    agent_id,
                    "--id",
                    guideline["id"],
                    "--service",
                    service_name,
                    "--tool",
                    tool_name,
                )
                == os.EX_OK
            )

            guideline = await context.api.read_guideline(
                agent_id=agent_id, guideline_id=guideline["id"]
            )

            assert any(
                assoc["tool_id"]["service_name"] == service_name
                and assoc["tool_id"]["tool_name"] == tool_name
                for assoc in guideline["tool_associations"]
            )


async def test_that_a_tool_can_be_disabled_for_a_guideline(
    context: ContextOfTest,
) -> None:
    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        agent_id = (await context.api.get_first_agent())["id"]

        guideline = await context.api.create_guideline(
            agent_id,
            condition="the customer wants to get meeting details",
            action="get meeting event information",
        )

        service_name = "local_service"
        tool_name = "fetch_event_data"
        service_kind = "sdk"

        @tool
        def fetch_event_data(context: ToolContext, event_id: str) -> ToolResult:
            """Fetch event data based on event ID."""
            return ToolResult({"event_id": event_id})

        async with run_service_server([fetch_event_data]) as server:
            assert (
                await run_cli_and_get_exit_status(
                    "service",
                    "create",
                    "--name",
                    service_name,
                    "--kind",
                    service_kind,
                    "--url",
                    server.url,
                )
                == os.EX_OK
            )

            _ = await context.api.add_association(
                agent_id, guideline["id"], service_name, tool_name
            )

            assert (
                await run_cli_and_get_exit_status(
                    "guideline",
                    "tool-disable",
                    "--agent-id",
                    agent_id,
                    "--id",
                    guideline["id"],
                    "--service",
                    service_name,
                    "--tool",
                    tool_name,
                )
                == os.EX_OK
            )

            guideline = await context.api.read_guideline(
                agent_id=agent_id, guideline_id=guideline["id"]
            )

            assert guideline["tool_associations"] == []


async def test_that_variables_can_be_listed(
    context: ContextOfTest,
) -> None:
    name1 = "VAR1"
    description1 = "FIRST"

    name2 = "VAR2"
    description2 = "SECOND"

    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        agent_id = (await context.api.get_first_agent())["id"]
        _ = await context.api.create_context_variable(agent_id, name1, description1)
        _ = await context.api.create_context_variable(agent_id, name2, description2)

        process = await run_cli(
            "variable",
            "list",
            "--agent-id",
            agent_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()
        output = stdout.decode() + stderr.decode()
        assert process.returncode == os.EX_OK

        assert name1 in output
        assert description1 in output
        assert name2 in output
        assert description2 in output


async def test_that_a_variable_can_be_added(
    context: ContextOfTest,
) -> None:
    name = "test_variable_cli"
    description = "Variable added via CLI"

    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        agent_id = (await context.api.get_first_agent())["id"]

        service_name = "local_service"
        tool_name = "fetch_event_data"
        service_kind = "sdk"
        freshness_rules = "0 0,6,12,18 * * *"

        @tool
        def fetch_event_data(context: ToolContext, event_id: str) -> ToolResult:
            """Fetch event data based on event ID."""
            return ToolResult({"event_id": event_id})

        async with run_service_server([fetch_event_data]) as server:
            assert (
                await run_cli_and_get_exit_status(
                    "service",
                    "create",
                    "--name",
                    service_name,
                    "--kind",
                    service_kind,
                    "--url",
                    server.url,
                )
                == os.EX_OK
            )

            assert (
                await run_cli_and_get_exit_status(
                    "variable",
                    "create",
                    "--agent-id",
                    agent_id,
                    "--description",
                    description,
                    "--name",
                    name,
                    "--service",
                    service_name,
                    "--tool",
                    tool_name,
                    "--freshness-rules",
                    freshness_rules,
                )
                == os.EX_OK
            )

        variables = await context.api.list_context_variables(agent_id)

        variable = next(
            (
                v
                for v in variables
                if v["name"] == name
                and v["description"] == description
                and v["tool_id"]
                == {
                    "service_name": "local_service",
                    "tool_name": "fetch_event_data",
                }
                and v["freshness_rules"] == freshness_rules
            ),
            None,
        )
        assert variable is not None, "Variable was not added"


async def test_that_a_variable_can_be_updated(
    context: ContextOfTest,
) -> None:
    name = "test_variable_cli"
    description = "Variable added via CLI"
    new_description = "Variable updated via CLI"
    service_name = "local"
    tool_name = "fetch_account_balance"
    freshness_rules = "0 0,6,12,18 * * *"

    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        agent_id = (await context.api.get_first_agent())["id"]
        variable = await context.api.create_context_variable(agent_id, name, description)

        service_name = "local_service"
        tool_name = "fetch_event_data"
        service_kind = "sdk"
        freshness_rules = "0 0,6,12,18 * * *"

        @tool
        def fetch_event_data(context: ToolContext, event_id: str) -> ToolResult:
            """Fetch event data based on event ID."""
            return ToolResult({"event_id": event_id})

        async with run_service_server([fetch_event_data]) as server:
            assert (
                await run_cli_and_get_exit_status(
                    "service",
                    "create",
                    "--name",
                    service_name,
                    "--kind",
                    service_kind,
                    "--url",
                    server.url,
                )
                == os.EX_OK
            )

            assert (
                await run_cli_and_get_exit_status(
                    "variable",
                    "update",
                    "--agent-id",
                    agent_id,
                    "--id",
                    variable["id"],
                    "--description",
                    new_description,
                    "--service",
                    service_name,
                    "--tool",
                    tool_name,
                    "--freshness-rules",
                    freshness_rules,
                )
                == os.EX_OK
            )

        updated_variable = await context.api.read_context_variable(agent_id, variable["id"])
        assert updated_variable["context_variable"]["name"] == name
        assert updated_variable["context_variable"]["description"] == new_description
        assert updated_variable["context_variable"]["tool_id"] == {
            "service_name": "local_service",
            "tool_name": "fetch_event_data",
        }
        assert updated_variable["context_variable"]["freshness_rules"] == freshness_rules


async def test_that_a_variable_can_be_removed(
    context: ContextOfTest,
) -> None:
    name = "test_variable_to_remove"
    description = "Variable to be removed via CLI"

    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        agent_id = (await context.api.get_first_agent())["id"]

        variable = await context.api.create_context_variable(agent_id, name, description)

        process = await run_cli(
            "variable",
            "delete",
            "--agent-id",
            agent_id,
            "--id",
            variable["id"],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        variables = await context.api.list_context_variables(agent_id)
        assert len(variables) == 0


async def test_that_a_variable_value_can_be_set_with_json(
    context: ContextOfTest,
) -> None:
    variable_name = "test_variable_set"
    variable_description = "Variable to test setting value via CLI"
    key = "test_key"
    data: dict[str, Any] = {"test": "data", "type": 27}

    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        agent_id = (await context.api.get_first_agent())["id"]
        variable = await context.api.create_context_variable(
            agent_id, variable_name, variable_description
        )

        process = await run_cli(
            "variable",
            "set",
            "--agent-id",
            agent_id,
            "--id",
            variable["id"],
            "--key",
            key,
            "--value",
            json.dumps(data),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        value = await context.api.read_context_variable_value(agent_id, variable["id"], key)
        assert json.loads(value["data"]) == data


async def test_that_a_variable_value_can_be_set_with_string(
    context: ContextOfTest,
) -> None:
    variable_name = "test_variable_set"
    variable_description = "Variable to test setting value via CLI"
    key = "test_key"
    data = "test_string"

    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        agent_id = (await context.api.get_first_agent())["id"]
        variable = await context.api.create_context_variable(
            agent_id, variable_name, variable_description
        )

        process = await run_cli(
            "variable",
            "set",
            "--agent-id",
            agent_id,
            "--id",
            variable["id"],
            "--key",
            key,
            "--value",
            data,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        value = await context.api.read_context_variable_value(agent_id, variable["id"], key)

        assert value["data"] == data


async def test_that_a_variables_values_can_be_retrieved(
    context: ContextOfTest,
) -> None:
    variable_name = "test_variable_get"
    variable_description = "Variable to test retrieving values via CLI"
    values = {
        "key1": "data1",
        "key2": "data2",
        "key3": "data3",
    }

    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        agent_id = (await context.api.get_first_agent())["id"]
        variable = await context.api.create_context_variable(
            agent_id, variable_name, variable_description
        )

        for key, data in values.items():
            await context.api.update_context_variable_value(agent_id, variable["id"], key, data)

        process = await run_cli(
            "variable",
            "get",
            "--agent-id",
            agent_id,
            "--id",
            variable["id"],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_get_all_values, stderr_get_all = await process.communicate()
        output_get_all_values = stdout_get_all_values.decode() + stderr_get_all.decode()
        assert process.returncode == os.EX_OK

        for key, data in values.items():
            assert key in output_get_all_values
            assert data in output_get_all_values

        specific_key = "key2"
        expected_value = values[specific_key]

        process = await run_cli(
            "variable",
            "get",
            "--agent-id",
            agent_id,
            "--id",
            variable["id"],
            "--key",
            specific_key,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        output = stdout.decode() + stderr.decode()
        assert process.returncode == os.EX_OK

        assert specific_key in output
        assert expected_value in output


async def test_that_a_message_can_be_inspected(
    context: ContextOfTest,
) -> None:
    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        agent_id = (await context.api.get_first_agent())["id"]

        guideline = await context.api.create_guideline(
            agent_id=agent_id,
            condition="the customer talks about cows",
            action="address the customer by his first name and say you like Pepsi",
        )

        term = await context.api.create_term(
            agent_id=agent_id,
            name="Bazoo",
            description="a type of cow",
        )

        variable = await context.api.create_context_variable(
            agent_id=agent_id,
            name="Customer first name",
            description="",
        )

        customer = await context.api.create_customer("John Smith")

        await context.api.update_context_variable_value(
            agent_id=agent_id,
            variable_id=variable["id"],
            key=customer["id"],
            value="Johnny",
        )

        session = await context.api.create_session(agent_id, customer["id"])

        reply_event = await context.api.get_agent_reply(session["id"], "Oh do I like bazoos")

        assert "Johnny" in reply_event["data"]["message"]
        assert "Pepsi" in reply_event["data"]["message"]

        process = await run_cli(
            "session",
            "inspect",
            "--session-id",
            session["id"],
            "--event-id",
            reply_event["id"],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()
        output = stdout.decode() + stderr.decode()
        assert process.returncode == os.EX_OK

        assert guideline["condition"] in output
        assert guideline["action"] in output
        assert term["name"] in output
        assert term["description"] in output
        assert variable["name"] in output
        assert customer["id"] in output


async def test_that_an_openapi_service_can_be_added_via_file(
    context: ContextOfTest,
) -> None:
    service_name = "test_openapi_service"
    service_kind = "openapi"

    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        async with run_openapi_server(rng_app()):
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{OPENAPI_SERVER_URL}/openapi.json")
                response.raise_for_status()
                openapi_json = response.text

            with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as temp_file:
                temp_file.write(openapi_json)
                temp_file.flush()
                source = temp_file.name

                assert (
                    await run_cli_and_get_exit_status(
                        "service",
                        "create",
                        "--name",
                        service_name,
                        "--kind",
                        service_kind,
                        "--source",
                        source,
                        "--url",
                        OPENAPI_SERVER_URL,
                    )
                    == os.EX_OK
                )

                async with context.api.make_client() as client:
                    response = await client.get("/services/")
                    response.raise_for_status()
                    services = response.json()
                    assert any(
                        s["name"] == service_name and s["kind"] == service_kind for s in services
                    )


async def test_that_an_openapi_service_can_be_added_via_url(
    context: ContextOfTest,
) -> None:
    service_name = "test_openapi_service_via_url"
    service_kind = "openapi"

    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        async with run_openapi_server(rng_app()):
            source = OPENAPI_SERVER_URL + "/openapi.json"

            assert (
                await run_cli_and_get_exit_status(
                    "service",
                    "create",
                    "--name",
                    service_name,
                    "--kind",
                    service_kind,
                    "--source",
                    source,
                    "--url",
                    OPENAPI_SERVER_URL,
                )
                == os.EX_OK
            )

            async with context.api.make_client() as client:
                response = await client.get("/services/")
                response.raise_for_status()
                services = response.json()
                assert any(
                    s["name"] == service_name and s["kind"] == service_kind for s in services
                )


async def test_that_a_sdk_service_can_be_added(
    context: ContextOfTest,
) -> None:
    service_name = "test_sdk_service"
    service_kind = "sdk"

    @tool
    def sample_tool(context: ToolContext, param: int) -> ToolResult:
        """I want to check also the description here.
        So for that, I will just write multiline text, so I can test both the
        limit of chars in one line, and also, test that multiline works as expected
        and displayed such that the customer can easily read and understand it."""
        return ToolResult(param * 2)

    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        async with run_service_server([sample_tool]) as server:
            assert (
                await run_cli_and_get_exit_status(
                    "service",
                    "create",
                    "--name",
                    service_name,
                    "--kind",
                    service_kind,
                    "--url",
                    server.url,
                )
                == os.EX_OK
            )

            async with context.api.make_client() as client:
                response = await client.get("/services/")
                response.raise_for_status()
                services = response.json()
                assert any(
                    s["name"] == service_name and s["kind"] == service_kind for s in services
                )


async def test_that_a_service_can_be_removed(
    context: ContextOfTest,
) -> None:
    service_name = "test_service_to_remove"

    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        async with run_openapi_server(rng_app()):
            await context.api.create_openapi_service(service_name, OPENAPI_SERVER_URL)

        process = await run_cli(
            "service",
            "delete",
            "--name",
            service_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        async with context.api.make_client() as client:
            response = await client.get("/services")
            response.raise_for_status()
            services = response.json()
            assert not any(s["name"] == service_name for s in services)


async def test_that_services_can_be_listed(
    context: ContextOfTest,
) -> None:
    service_name_1 = "test_openapi_service_1"
    service_name_2 = "test_openapi_service_2"

    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        async with run_openapi_server(rng_app()):
            await context.api.create_openapi_service(service_name_1, OPENAPI_SERVER_URL)
            await context.api.create_openapi_service(service_name_2, OPENAPI_SERVER_URL)

        process = await run_cli(
            "service",
            "list",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()
        output = stdout.decode() + stderr.decode()
        assert process.returncode == os.EX_OK

        assert service_name_1 in output
        assert service_name_2 in output
        assert "openapi" in output, "Service type 'openapi' was not found in the output"


async def test_that_a_service_can_be_viewed(
    context: ContextOfTest,
) -> None:
    service_name = "test_service_view"
    service_url = OPENAPI_SERVER_URL

    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        async with run_openapi_server(rng_app()):
            await context.api.create_openapi_service(service_name, service_url)

        process = await run_cli(
            "service",
            "view",
            "--name",
            service_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()
        output = stdout.decode() + stderr.decode()
        assert process.returncode == os.EX_OK

        assert service_name in output
        assert "openapi" in output
        assert service_url in output

        assert "one_required_query_param" in output
        assert "query_param:"

        assert "two_required_query_params" in output
        assert "query_param_1:"
        assert "query_param_2:"


async def test_that_customers_can_be_listed(context: ContextOfTest) -> None:
    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        await context.api.create_customer(name="First Customer")
        await context.api.create_customer(name="Second Customer")

        process = await run_cli(
            "customer",
            "list",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        output = stdout.decode() + stderr.decode()
        assert process.returncode == os.EX_OK

        assert "First Customer" in output
        assert "Second Customer" in output


async def test_that_a_customer_can_be_added(context: ContextOfTest) -> None:
    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        assert (
            await run_cli_and_get_exit_status(
                "customer",
                "create",
                "--name",
                "TestCustomer",
            )
            == os.EX_OK
        )

        customers = await context.api.list_customers()
        assert any(c["name"] == "TestCustomer" for c in customers)


async def test_that_a_customer_can_be_updated(context: ContextOfTest) -> None:
    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        customer = await context.api.create_customer("TestCustomer")

        assert (
            await run_cli_and_get_exit_status(
                "customer",
                "update",
                "--id",
                customer["id"],
                "--name",
                "UpdatedTestCustomer",
            )
            == os.EX_OK
        )

        updated_customer = await context.api.read_customer(customer["id"])
        assert updated_customer["name"] == "UpdatedTestCustomer"


async def test_that_a_customer_can_be_viewed(context: ContextOfTest) -> None:
    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        customer_id = (await context.api.create_customer(name="TestCustomer"))["id"]

        process = await run_cli(
            "customer",
            "view",
            "--id",
            customer_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        output = stdout.decode() + stderr.decode()
        assert process.returncode == os.EX_OK

        assert customer_id in output
        assert "TestCustomer" in output


async def test_that_a_customer_can_be_deleted(context: ContextOfTest) -> None:
    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        customer_id = (await context.api.create_customer(name="TestCustomer"))["id"]

        assert (
            await run_cli_and_get_exit_status(
                "customer",
                "delete",
                "--id",
                customer_id,
            )
            == os.EX_OK
        )

        customers = await context.api.list_customers()
        assert not any(c["name"] == "TestCustomer" for c in customers)


async def test_that_a_customer_extra_can_be_added(context: ContextOfTest) -> None:
    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        customer_id = (await context.api.create_customer(name="TestCustomer"))["id"]

        assert (
            await run_cli_and_get_exit_status(
                "customer",
                "set",
                "--id",
                customer_id,
                "--key",
                "key1",
                "--value",
                "value1",
            )
            == os.EX_OK
        )

        customer = await context.api.read_customer(id=customer_id)
        assert customer["extra"].get("key1") == "value1"


async def test_that_a_customer_extra_can_be_removed(context: ContextOfTest) -> None:
    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        customer_id = (
            await context.api.create_customer(name="TestCustomer", extra={"key1": "value1"})
        )["id"]

        assert (
            await run_cli_and_get_exit_status(
                "customer",
                "unset",
                "--id",
                customer_id,
                "--key",
                "key1",
            )
            == os.EX_OK
        )

        customer = await context.api.read_customer(id=customer_id)
        assert "key1" not in customer["extra"]


async def test_that_a_customer_tag_can_be_added(context: ContextOfTest) -> None:
    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        customer_id = (await context.api.create_customer(name="TestCustomer"))["id"]
        tag_id = (await context.api.create_tag(name="TestTag"))["id"]

        assert (
            await run_cli_and_get_exit_status(
                "customer",
                "tag",
                "--id",
                customer_id,
                "--tag-id",
                tag_id,
            )
            == os.EX_OK
        )
        customer = await context.api.read_customer(id=customer_id)
        tags = customer["tags"]
        assert tag_id in tags


async def test_that_a_customer_tag_can_be_removed(context: ContextOfTest) -> None:
    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        customer_id = (await context.api.create_customer(name="TestCustomer"))["id"]
        tag_id = (await context.api.create_tag(name="TestTag"))["id"]
        await context.api.add_customer_tag(customer_id, tag_id)

        assert (
            await run_cli_and_get_exit_status(
                "customer",
                "untag",
                "--id",
                customer_id,
                "--tag-id",
                tag_id,
            )
            == os.EX_OK
        )
        customer = await context.api.read_customer(id=customer_id)
        tags = customer["tags"]
        assert tag_id not in tags


async def test_that_a_tag_can_be_added(context: ContextOfTest) -> None:
    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        tag_name = "TestTag"

        assert (
            await run_cli_and_get_exit_status(
                "tag",
                "create",
                "--name",
                tag_name,
            )
            == os.EX_OK
        )

        tags = await context.api.list_tags()
        assert any(t["name"] == tag_name for t in tags)


async def test_that_tags_can_be_listed(context: ContextOfTest) -> None:
    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        await context.api.create_tag("FirstTag")
        await context.api.create_tag("SecondTag")

        process = await run_cli(
            "tag",
            "list",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        output = stdout.decode() + stderr.decode()
        assert process.returncode == os.EX_OK

        assert "FirstTag" in output
        assert "SecondTag" in output


async def test_that_a_tag_can_be_viewed(context: ContextOfTest) -> None:
    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        tag_id = (await context.api.create_tag("TestViewTag"))["id"]

        process = await run_cli(
            "tag",
            "view",
            "--id",
            tag_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        output = stdout.decode() + stderr.decode()
        assert process.returncode == os.EX_OK

        assert "TestViewTag" in output
        assert tag_id in output


async def test_that_a_tag_can_be_updated(context: ContextOfTest) -> None:
    with run_server(context):
        while not is_server_responsive(SERVER_PORT):
            pass

        tag_id = (await context.api.create_tag("TestViewTag"))["id"]
        new_name = "UpdatedTagName"

        assert (
            await run_cli_and_get_exit_status(
                "tag",
                "update",
                "--id",
                tag_id,
                "--name",
                new_name,
            )
            == os.EX_OK
        )

        updated_tag = await context.api.read_tag(tag_id)
        assert updated_tag["name"] == new_name
