import asyncio
from contextlib import asynccontextmanager
import json
import os
import time
import traceback
import tempfile
from typing import Any, AsyncIterator, Awaitable, Callable, Optional
from fastapi import FastAPI, Query, Request, Response
from fastapi.responses import JSONResponse
import httpx
import uvicorn

from tests.e2e.test_utilities import (
    CLI_CLIENT_PATH,
    SERVER_ADDRESS,
    ContextOfTest,
    run_server,
)
from parlant.core.services.tools.plugins import tool, ToolEntry, PluginServer
from parlant.core.tools import ToolResult, ToolContext

REASONABLE_AMOUNT_OF_TIME = 15
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


class API:
    @staticmethod
    @asynccontextmanager
    async def make_client() -> AsyncIterator[httpx.AsyncClient]:
        async with httpx.AsyncClient(
            base_url=SERVER_ADDRESS,
            follow_redirects=True,
            timeout=httpx.Timeout(30),
        ) as client:
            yield client

    @staticmethod
    async def get_first_agent_id() -> str:
        async with API.make_client() as client:
            response = await client.get("/agents/")
            agent = response.raise_for_status().json()["agents"][0]
            return str(agent["id"])

    @staticmethod
    async def create_agent(
        name: str,
        description: Optional[str],
        max_engine_iterations: Optional[int],
    ) -> Any:
        async with API.make_client() as client:
            response = await client.post(
                "/agents",
                json={
                    "name": name,
                    "description": description,
                    "max_engine_iterations": max_engine_iterations,
                },
            )

            return response.raise_for_status().json()["agent"]

    @staticmethod
    async def list_agents() -> Any:
        async with API.make_client() as client:
            response = await client.get("/agents/")
            return response.raise_for_status().json()["agents"]

    @staticmethod
    async def create_session(
        agent_id: str,
        customer_id: str,
        title: Optional[str] = None,
    ) -> Any:
        async with API.make_client() as client:
            response = await client.post(
                "/sessions",
                params={"allow_greeting": False},
                json={
                    "agent_id": agent_id,
                    "customer_id": customer_id,
                    "title": title,
                },
            )

            return response.raise_for_status().json()["session"]

    @staticmethod
    async def get_agent_reply(
        session_id: str,
        message: str,
    ) -> Any:
        return next(iter(await API.get_agent_replies(session_id, message, 1)))

    @staticmethod
    async def get_agent_replies(
        session_id: str,
        message: str,
        number_of_replies_to_expect: int,
    ) -> list[Any]:
        async with API.make_client() as client:
            try:
                customer_message_response = await client.post(
                    f"/sessions/{session_id}/events",
                    json={
                        "kind": "message",
                        "source": "customer",
                        "content": message,
                    },
                )
                customer_message_response.raise_for_status()
                customer_message_offset = int(customer_message_response.json()["event"]["offset"])

                last_known_offset = customer_message_offset

                replies: list[Any] = []
                start_time = time.time()
                timeout = 300

                while len(replies) < number_of_replies_to_expect:
                    response = await client.get(
                        f"/sessions/{session_id}/events",
                        params={
                            "min_offset": last_known_offset + 1,
                            "kinds": "message",
                            "wait": True,
                        },
                    )
                    response.raise_for_status()
                    events = response.json()["events"]

                    if message_events := [e for e in events if e["kind"] == "message"]:
                        replies.append(message_events[0])

                    last_known_offset = events[-1]["offset"]

                    if (time.time() - start_time) >= timeout:
                        raise TimeoutError()

                return replies
            except:
                traceback.print_exc()
                raise

    @staticmethod
    async def create_term(
        agent_id: str,
        name: str,
        description: str,
        synonyms: str = "",
    ) -> Any:
        async with API.make_client() as client:
            response = await client.post(
                f"/agents/{agent_id}/terms/",
                json={
                    "name": name,
                    "description": description,
                    **({"synonyms": synonyms.split(",")} if synonyms else {}),
                },
            )

            return response.raise_for_status().json()["term"]

    @staticmethod
    async def list_terms(agent_id: str) -> Any:
        async with API.make_client() as client:
            response = await client.get(
                f"/agents/{agent_id}/terms/",
            )
            response.raise_for_status()

            return response.json()["terms"]

    @staticmethod
    async def read_term(
        agent_id: str,
        term_id: str,
    ) -> Any:
        async with API.make_client() as client:
            response = await client.get(
                f"/agents/{agent_id}/terms/{term_id}",
            )
            response.raise_for_status()

            return response.json()

    @staticmethod
    async def list_guidelines(agent_id: str) -> Any:
        async with API.make_client() as client:
            response = await client.get(
                f"/agents/{agent_id}/guidelines/",
            )

            response.raise_for_status()

            return response.json()["guidelines"]

    @staticmethod
    async def read_guideline(
        agent_id: str,
        guideline_id: str,
    ) -> Any:
        async with API.make_client() as client:
            response = await client.get(
                f"/agents/{agent_id}/guidelines/{guideline_id}",
            )

            response.raise_for_status()

            return response.json()

    @staticmethod
    async def create_guideline(
        agent_id: str,
        condition: str,
        action: str,
        coherence_check: Optional[dict[str, Any]] = None,
        connection_propositions: Optional[dict[str, Any]] = None,
        operation: str = "add",
        updated_id: Optional[str] = None,
    ) -> Any:
        async with API.make_client() as client:
            response = await client.post(
                f"/agents/{agent_id}/guidelines/",
                json={
                    "invoices": [
                        {
                            "payload": {
                                "content": {
                                    "condition": condition,
                                    "action": action,
                                },
                                "operation": operation,
                                "updated_id": updated_id,
                                "coherence_check": True,
                                "connection_proposition": True,
                            },
                            "checksum": "checksum_value",
                            "approved": True if coherence_check is None else False,
                            "data": {
                                "coherence_checks": coherence_check if coherence_check else [],
                                "connection_propositions": connection_propositions
                                if connection_propositions
                                else None,
                            },
                            "error": None,
                        }
                    ]
                },  # type: ignore
            )

            response.raise_for_status()

            return response.json()["items"][0]["guideline"]

    @staticmethod
    async def add_association(
        agent_id: str,
        guideline_id: str,
        service_name: str,
        tool_name: str,
    ) -> Any:
        async with API.make_client() as client:
            response = await client.patch(
                f"/agents/{agent_id}/guidelines/{guideline_id}",
                json={
                    "tool_associations": {
                        "add": [
                            {
                                "service_name": service_name,
                                "tool_name": tool_name,
                            }
                        ]
                    }
                },
            )

            response.raise_for_status()

        return response.json()["tool_associations"]

    @staticmethod
    async def create_context_variable(
        agent_id: str,
        name: str,
        description: str,
    ) -> Any:
        async with API.make_client() as client:
            response = await client.post(
                f"/agents/{agent_id}/context-variables",
                json={
                    "name": name,
                    "description": description,
                },
            )

            response.raise_for_status()

            return response.json()["context_variable"]

    @staticmethod
    async def list_context_variables(agent_id: str) -> Any:
        async with API.make_client() as client:
            response = await client.get(f"/agents/{agent_id}/context-variables/")

            response.raise_for_status()

            return response.json()["context_variables"]

    @staticmethod
    async def update_context_variable_value(
        agent_id: str,
        variable_id: str,
        key: str,
        value: Any,
    ) -> Any:
        async with API.make_client() as client:
            response = await client.put(
                f"/agents/{agent_id}/context-variables/{variable_id}/{key}",
                json={"data": value},
            )
            response.raise_for_status()

    @staticmethod
    async def read_context_variable_value(
        agent_id: str,
        variable_id: str,
        key: str,
    ) -> Any:
        async with API.make_client() as client:
            response = await client.get(
                f"{SERVER_ADDRESS}/agents/{agent_id}/context-variables/{variable_id}/{key}",
            )

            response.raise_for_status()

            return response.json()

    @staticmethod
    async def create_openapi_service(
        service_name: str,
        url: str,
    ) -> None:
        payload = {"kind": "openapi", "openapi": {"source": f"{url}/openapi.json", "url": url}}

        async with API.make_client() as client:
            response = await client.put(f"/services/{service_name}", json=payload)
            response.raise_for_status()

    @staticmethod
    async def create_tag(name: str) -> Any:
        async with API.make_client() as client:
            response = await client.post("/tags", json={"name": name})
        return response.json()

    @staticmethod
    async def list_tags() -> Any:
        async with API.make_client() as client:
            response = await client.get("/tags")
        return response.json()

    @staticmethod
    async def read_tag(id: str) -> Any:
        async with API.make_client() as client:
            response = await client.get(f"/tags/{id}")
        return response.json()

    @staticmethod
    async def create_customer(
        name: str,
        extra: Optional[dict[str, Any]] = {},
    ) -> Any:
        async with API.make_client() as client:
            respone = await client.post("/customers", json={"name": name, "extra": extra})
            respone.raise_for_status()

        return respone.json()

    @staticmethod
    async def list_customers() -> Any:
        async with API.make_client() as client:
            respone = await client.get("/customers")
            respone.raise_for_status()

        return respone.json()["customers"]

    @staticmethod
    async def read_customer(id: str) -> Any:
        async with API.make_client() as client:
            respone = await client.get(f"/customers/{id}")
            respone.raise_for_status()

        return respone.json()

    @staticmethod
    async def add_customer_tag(id: str, tag_id: str) -> None:
        async with API.make_client() as client:
            respone = await client.patch(f"/customers/{id}", json={"tags": {"add": [tag_id]}})
            respone.raise_for_status()


async def test_that_an_agent_can_be_added(context: ContextOfTest) -> None:
    name = "TestAgent"
    description = "This is a test agent"

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        process = await run_cli(
            "agent",
            "add",
            name,
            "-d",
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

        agents = await API.list_agents()
        new_agent = next((a for a in agents if a["name"] == name), None)
        assert new_agent
        assert new_agent["description"] == description
        assert new_agent["max_engine_iterations"] == 123

        process = await run_cli(
            "agent",
            "add",
            "Test Agent With No Description",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        agents = await API.list_agents()
        new_agent_no_desc = next(
            (a for a in agents if a["name"] == "Test Agent With No Description"), None
        )
        assert new_agent_no_desc
        assert new_agent_no_desc["description"] is None


async def test_that_an_agent_can_be_updated(
    context: ContextOfTest,
) -> None:
    new_description = "Updated description"
    new_max_engine_iterations = 5

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        process = await run_cli(
            "agent",
            "update",
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

        agent = (await API.list_agents())[0]
        assert agent["description"] == new_description
        assert agent["max_engine_iterations"] == new_max_engine_iterations


async def test_that_an_agent_can_be_viewed(
    context: ContextOfTest,
) -> None:
    name = "Test Agent"
    description = "Bananas"
    max_engine_iterations = 2

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent = await API.create_agent(
            name=name,
            description=description,
            max_engine_iterations=max_engine_iterations,
        )

        process = await run_cli(
            "agent",
            "view",
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
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()
        _ = await API.create_session(
            agent_id=agent_id, customer_id=first_customer, title=first_title
        )
        _ = await API.create_session(
            agent_id=agent_id, customer_id=first_customer, title=second_title
        )
        _ = await API.create_session(
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
            "-u",
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


async def test_that_a_term_can_be_created_with_synonyms(
    context: ContextOfTest,
) -> None:
    term_name = "guideline"
    description = "when and then statements"
    synonyms = "rule, principle"

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

        process = await run_cli(
            "glossary",
            "add",
            "--agent-id",
            agent_id,
            term_name,
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
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

        process = await run_cli(
            "glossary",
            "add",
            "--agent-id",
            agent_id,
            term_name,
            description,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        terms = await API.list_terms(agent_id)
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
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

        term_to_update = await API.create_term(agent_id, name, description, synonyms)

        process = await run_cli(
            "glossary",
            "update",
            "--agent-id",
            agent_id,
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

        updated_term = await API.read_term(agent_id=agent_id, term_id=term_to_update["id"])
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
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

        term = await API.create_term(agent_id, name, description, synonyms)

        process = await run_cli(
            "glossary",
            "remove",
            "--agent-id",
            agent_id,
            term["id"],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        terms = await API.list_terms(agent_id)
        assert len(terms) == 0


async def test_that_terms_are_loaded_on_server_startup(
    context: ContextOfTest,
) -> None:
    name = "guideline_no_synonyms"
    description = "simple guideline with no synonyms"

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

        term = await API.create_term(agent_id, name, description)

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

        term = await API.read_term(agent_id, term["id"])
        assert term["name"] == name
        assert term["description"] == description
        assert term["synonyms"] == []


async def test_that_a_guideline_can_be_added(
    context: ContextOfTest,
) -> None:
    condition = "the customer greets you"
    action = "greet them back with 'Hello'"

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

        process = await run_cli(
            "guideline",
            "add",
            "-a",
            agent_id,
            condition,
            action,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        guidelines = await API.list_guidelines(agent_id)
        assert any(g["condition"] == condition and g["action"] == action for g in guidelines)


async def test_that_a_guideline_can_be_updated(
    context: ContextOfTest,
) -> None:
    condition = "the customer asks for help"
    initial_action = "offer assistance"
    updated_action = "provide detailed support information"

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

        guideline = await API.create_guideline(
            agent_id=agent_id, condition=condition, action=initial_action
        )

        process = await run_cli(
            "guideline",
            "update",
            "-a",
            agent_id,
            guideline["id"],
            condition,
            updated_action,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        updated_guideline = (
            await API.read_guideline(agent_id=agent_id, guideline_id=guideline["id"])
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
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

        process = await run_cli(
            "guideline",
            "add",
            "-a",
            agent_id,
            condition,
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
            "add",
            "-a",
            agent_id,
            condition,
            conflicting_action,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()
        output = stdout.decode() + stderr.decode()

        assert "Detected potential incoherence with other guidelines" in output

        guidelines = await API.list_guidelines(agent_id)

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
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

        process = await run_cli(
            "guideline",
            "add",
            "-a",
            agent_id,
            condition1,
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
            "add",
            "-a",
            agent_id,
            condition2,
            action2,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        guidelines = await API.list_guidelines(agent_id)

        assert len(guidelines) == 2
        source = guidelines[0]
        target = guidelines[1]

        source_guideline = await API.read_guideline(agent_id, source["id"])
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
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

        guideline = await API.create_guideline(
            agent_id=agent_id, condition=condition, action=action
        )

        process = await run_cli(
            "guideline",
            "view",
            "-a",
            agent_id,
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
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

        _ = await API.create_guideline(agent_id=agent_id, condition=condition1, action=action1)
        _ = await API.create_guideline(agent_id=agent_id, condition=condition2, action=action2)

        process = await run_cli(
            "guideline",
            "list",
            "-a",
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
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

        process = await run_cli(
            "guideline",
            "add",
            "-a",
            agent_id,
            "--no-check",
            "--no-index",
            condition1,
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
            "add",
            "-a",
            agent_id,
            "--no-check",
            "--no-index",
            condition2,
            action2,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        guidelines = await API.list_guidelines(agent_id)

        first_guideline = next(
            g for g in guidelines if g["condition"] == condition1 and g["action"] == action1
        )
        second_guideline = next(
            g for g in guidelines if g["condition"] == condition2 and g["action"] == action2
        )

        process = await run_cli(
            "guideline",
            "entail",
            "-a",
            agent_id,
            first_guideline["id"],
            second_guideline["id"],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await process.communicate()
        await process.wait()
        assert process.returncode == os.EX_OK

        guideline = await API.read_guideline(agent_id, first_guideline["id"])
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
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

        process = await run_cli(
            "guideline",
            "add",
            "-a",
            agent_id,
            "--no-check",
            "--no-index",
            condition1,
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
            "add",
            "-a",
            agent_id,
            "--no-check",
            "--no-index",
            condition2,
            action2,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        guidelines = await API.list_guidelines(agent_id)

        first_guideline = next(
            g for g in guidelines if g["condition"] == condition1 and g["action"] == action1
        )
        second_guideline = next(
            g for g in guidelines if g["condition"] == condition2 and g["action"] == action2
        )

        process = await run_cli(
            "guideline",
            "entail",
            "-a",
            agent_id,
            "--suggestive",
            first_guideline["id"],
            second_guideline["id"],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await process.communicate()
        await process.wait()
        assert process.returncode == os.EX_OK

        guideline = await API.read_guideline(agent_id, first_guideline["id"])

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
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

        guideline = await API.create_guideline(
            agent_id, condition="the customer greets you", action="greet them back with 'Hello'"
        )

        process = await run_cli(
            "guideline",
            "remove",
            "-a",
            agent_id,
            guideline["id"],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        guidelines = await API.list_guidelines(agent_id)
        assert len(guidelines) == 0


async def test_that_a_connection_can_be_removed(
    context: ContextOfTest,
) -> None:
    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

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
                                "content": {
                                    "condition": "the customer greets you",
                                    "action": "greet them back with 'Hello'",
                                },
                                "operation": "add",
                                "coherence_check": True,
                                "connection_proposition": True,
                            },
                            "checksum": "checksum_value",
                            "approved": True,
                            "data": {
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
                            "error": None,
                        },
                        {
                            "payload": {
                                "content": {
                                    "condition": "greeting the customer",
                                    "action": "ask for his health condition",
                                },
                                "operation": "add",
                                "coherence_check": True,
                                "connection_proposition": True,
                            },
                            "checksum": "checksum_value",
                            "approved": True,
                            "data": {
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
            "-a",
            agent_id,
            first,
            second,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        guideline = await API.read_guideline(agent_id, first)
        assert len(guideline["connections"]) == 0


async def test_that_a_tool_can_be_enabled_for_a_guideline(
    context: ContextOfTest,
) -> None:
    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

        guideline = await API.create_guideline(
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
                    "add",
                    service_name,
                    "-k",
                    service_kind,
                    "-u",
                    server.url,
                )
                == os.EX_OK
            )

            assert (
                await run_cli_and_get_exit_status(
                    "guideline",
                    "enable-tool",
                    "-a",
                    agent_id,
                    guideline["id"],
                    service_name,
                    tool_name,
                )
                == os.EX_OK
            )

            guideline = await API.read_guideline(agent_id=agent_id, guideline_id=guideline["id"])

            assert any(
                assoc["tool_id"]["service_name"] == service_name
                and assoc["tool_id"]["tool_name"] == tool_name
                for assoc in guideline["tool_associations"]
            )


async def test_that_a_tool_can_be_disabled_for_a_guideline(
    context: ContextOfTest,
) -> None:
    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

        guideline = await API.create_guideline(
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
                    "add",
                    service_name,
                    "-k",
                    service_kind,
                    "-u",
                    server.url,
                )
                == os.EX_OK
            )

            _ = await API.add_association(agent_id, guideline["id"], service_name, tool_name)

            assert (
                await run_cli_and_get_exit_status(
                    "guideline",
                    "disable-tool",
                    "-a",
                    agent_id,
                    guideline["id"],
                    service_name,
                    tool_name,
                )
                == os.EX_OK
            )

            guideline = await API.read_guideline(agent_id=agent_id, guideline_id=guideline["id"])

            assert guideline["tool_associations"] == []


async def test_that_a_variables_can_be_listed(
    context: ContextOfTest,
) -> None:
    name1 = "VAR1"
    description1 = "FIRST"

    name2 = "VAR2"
    description2 = "SECOND"

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()
        _ = await API.create_context_variable(agent_id, name1, description1)
        _ = await API.create_context_variable(agent_id, name2, description2)

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
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

        process = await run_cli(
            "variable",
            "add",
            "--agent-id",
            agent_id,
            "--description",
            description,
            name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        variables = await API.list_context_variables(agent_id)

        variable = next(
            (v for v in variables if v["name"] == name and v["description"] == description),
            None,
        )
        assert variable is not None, "Variable was not added"


async def test_that_a_variable_can_be_removed(
    context: ContextOfTest,
) -> None:
    name = "test_variable_to_remove"
    description = "Variable to be removed via CLI"

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

        _ = await API.create_context_variable(agent_id, name, description)

        process = await run_cli(
            "variable",
            "remove",
            "--agent-id",
            agent_id,
            name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        variables = await API.list_context_variables(agent_id)
        assert len(variables) == 0


async def test_that_a_variable_value_can_be_set_with_json(
    context: ContextOfTest,
) -> None:
    variable_name = "test_variable_set"
    variable_description = "Variable to test setting value via CLI"
    key = "test_key"
    data: dict[str, Any] = {"test": "data", "type": 27}

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()
        variable = await API.create_context_variable(agent_id, variable_name, variable_description)

        process = await run_cli(
            "variable",
            "set",
            "--agent-id",
            agent_id,
            variable_name,
            key,
            json.dumps(data),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        value = await API.read_context_variable_value(agent_id, variable["id"], key)
        assert json.loads(value["data"]) == data


async def test_that_a_variable_value_can_be_set_with_string(
    context: ContextOfTest,
) -> None:
    variable_name = "test_variable_set"
    variable_description = "Variable to test setting value via CLI"
    key = "test_key"
    data = "test_string"

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()
        variable = await API.create_context_variable(agent_id, variable_name, variable_description)

        process = await run_cli(
            "variable",
            "set",
            "--agent-id",
            agent_id,
            variable_name,
            key,
            data,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        value = await API.read_context_variable_value(agent_id, variable["id"], key)

        assert value["data"] == data


async def test_that_a_variable_values_can_be_retrieved(
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
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()
        variable = await API.create_context_variable(agent_id, variable_name, variable_description)

        for key, data in values.items():
            await API.update_context_variable_value(agent_id, variable["id"], key, data)

        process = await run_cli(
            "variable",
            "get",
            "--agent-id",
            agent_id,
            variable_name,
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
            variable_name,
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
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

        guideline = await API.create_guideline(
            agent_id=agent_id,
            condition="the customer talks about cows",
            action="address the customer by his first name and say you like Pepsi",
        )

        term = await API.create_term(
            agent_id=agent_id,
            name="Bazoo",
            description="a type of cow",
        )

        variable = await API.create_context_variable(
            agent_id=agent_id,
            name="Customer first name",
            description="",
        )

        customer_id = "john.s@peppery.co"

        await API.update_context_variable_value(
            agent_id=agent_id,
            variable_id=variable["id"],
            key=customer_id,
            value="Johnny",
        )

        session = await API.create_session(agent_id, customer_id)

        reply_event = await API.get_agent_reply(session["id"], "Oh do I like bazoos")

        assert "Johnny" in reply_event["data"]["message"]
        assert "Pepsi" in reply_event["data"]["message"]

        process = await run_cli(
            "session",
            "inspect",
            session["id"],
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
        assert customer_id in output


async def test_that_an_openapi_service_can_be_added_via_file(
    context: ContextOfTest,
) -> None:
    service_name = "test_openapi_service"
    service_kind = "openapi"

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

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
                        "add",
                        service_name,
                        "-k",
                        service_kind,
                        "-s",
                        source,
                        "-u",
                        OPENAPI_SERVER_URL,
                    )
                    == os.EX_OK
                )

                async with API.make_client() as client:
                    response = await client.get("/services/")
                    response.raise_for_status()
                    services = response.json()["services"]
                    assert any(
                        s["name"] == service_name and s["kind"] == service_kind for s in services
                    )


async def test_that_an_openapi_service_can_be_added_via_url(
    context: ContextOfTest,
) -> None:
    service_name = "test_openapi_service_via_url"
    service_kind = "openapi"

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        async with run_openapi_server(rng_app()):
            source = OPENAPI_SERVER_URL + "/openapi.json"

            assert (
                await run_cli_and_get_exit_status(
                    "service",
                    "add",
                    service_name,
                    "-k",
                    service_kind,
                    "-s",
                    source,
                    "-u",
                    OPENAPI_SERVER_URL,
                )
                == os.EX_OK
            )

            async with API.make_client() as client:
                response = await client.get("/services/")
                response.raise_for_status()
                services = response.json()["services"]
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
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        async with run_service_server([sample_tool]) as server:
            assert (
                await run_cli_and_get_exit_status(
                    "service",
                    "add",
                    service_name,
                    "-k",
                    service_kind,
                    "-u",
                    server.url,
                )
                == os.EX_OK
            )

            async with API.make_client() as client:
                response = await client.get("/services/")
                response.raise_for_status()
                services = response.json()["services"]
                assert any(
                    s["name"] == service_name and s["kind"] == service_kind for s in services
                )


async def test_that_a_service_can_be_removed(
    context: ContextOfTest,
) -> None:
    service_name = "test_service_to_remove"

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        async with run_openapi_server(rng_app()):
            await API.create_openapi_service(service_name, OPENAPI_SERVER_URL)

        process = await run_cli(
            "service",
            "remove",
            service_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        async with API.make_client() as client:
            response = await client.get("/services/")
            response.raise_for_status()
            services = response.json()["services"]
            assert not any(s["name"] == service_name for s in services)


async def test_that_services_can_be_listed(
    context: ContextOfTest,
) -> None:
    service_name_1 = "test_openapi_service_1"
    service_name_2 = "test_openapi_service_2"

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        async with run_openapi_server(rng_app()):
            await API.create_openapi_service(service_name_1, OPENAPI_SERVER_URL)
            await API.create_openapi_service(service_name_2, OPENAPI_SERVER_URL)

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
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        async with run_openapi_server(rng_app()):
            await API.create_openapi_service(service_name, service_url)

        process = await run_cli(
            "service",
            "view",
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
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        await API.create_customer(name="First Customer")
        await API.create_customer(name="Second Customer")

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
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        assert (
            await run_cli_and_get_exit_status(
                "customer",
                "add",
                "TestCustomer",
            )
            == os.EX_OK
        )

        customers = await API.list_customers()
        assert any(c["name"] == "TestCustomer" for c in customers)


async def test_that_a_customer_can_be_viewed(context: ContextOfTest) -> None:
    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        customer_id = (await API.create_customer(name="TestCustomer"))["customer"]["id"]

        process = await run_cli(
            "customer",
            "view",
            customer_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        output = stdout.decode() + stderr.decode()
        assert process.returncode == os.EX_OK

        assert customer_id in output
        assert "TestCustomer" in output


async def test_that_a_customer_extra_can_be_added(context: ContextOfTest) -> None:
    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        customer_id = (await API.create_customer(name="TestCustomer"))["customer"]["id"]

        assert (
            await run_cli_and_get_exit_status(
                "customer",
                "add-extra",
                customer_id,
                "key1",
                "value1",
            )
            == os.EX_OK
        )

        customer = await API.read_customer(id=customer_id)
        assert customer["extra"].get("key1") == "value1"


async def test_that_a_customer_extra_can_be_removed(context: ContextOfTest) -> None:
    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        customer_id = (await API.create_customer(name="TestCustomer", extra={"key1": "value1"}))[
            "customer"
        ]["id"]

        assert (
            await run_cli_and_get_exit_status(
                "customer",
                "remove-extra",
                customer_id,
                "key1",
            )
            == os.EX_OK
        )

        customer = await API.read_customer(id=customer_id)
        assert "key1" not in customer["extra"]


async def test_that_a_customer_tag_can_be_added(context: ContextOfTest) -> None:
    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        customer_id = (await API.create_customer(name="TestCustomer"))["customer"]["id"]
        tag_id = (await API.create_tag(name="TestTag"))["tag"]["id"]

        assert (
            await run_cli_and_get_exit_status(
                "customer",
                "add-tag",
                customer_id,
                tag_id,
            )
            == os.EX_OK
        )
        customer = await API.read_customer(id=customer_id)
        tags = customer["tags"]
        assert tag_id in tags


async def test_that_a_customer_tag_can_be_removed(context: ContextOfTest) -> None:
    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        customer_id = (await API.create_customer(name="TestCustomer"))["customer"]["id"]
        tag_id = (await API.create_tag(name="TestTag"))["tag"]["id"]
        await API.add_customer_tag(customer_id, tag_id)

        assert (
            await run_cli_and_get_exit_status(
                "customer",
                "remove-tag",
                customer_id,
                tag_id,
            )
            == os.EX_OK
        )
        customer = await API.read_customer(id=customer_id)
        tags = customer["tags"]
        assert tag_id not in tags


async def test_that_a_tag_can_be_added(context: ContextOfTest) -> None:
    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        tag_name = "TestTag"

        assert (
            await run_cli_and_get_exit_status(
                "tag",
                "add",
                tag_name,
            )
            == os.EX_OK
        )

        tags = (await API.list_tags())["tags"]
        assert any(t["name"] == tag_name for t in tags)


async def test_that_tags_can_be_listed(context: ContextOfTest) -> None:
    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        await API.create_tag("FirstTag")
        await API.create_tag("SecondTag")

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
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        tag_id = (await API.create_tag("TestViewTag"))["tag"]["id"]

        process = await run_cli(
            "tag",
            "view",
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
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        tag_id = (await API.create_tag("TestViewTag"))["tag"]["id"]
        new_name = "UpdatedTagName"

        assert (
            await run_cli_and_get_exit_status(
                "tag",
                "update",
                tag_id,
                new_name,
            )
            == os.EX_OK
        )

        updated_tag = await API.read_tag(tag_id)
        assert updated_tag["name"] == new_name
