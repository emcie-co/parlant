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

from tests.test_utilities import (
    CLI_CLIENT_PATH,
    SERVER_ADDRESS,
    ContextOfTest,
    run_server,
)
from emcie.common.plugin import tool, ToolEntry, PluginServer
from emcie.common.tools import ToolResult, ToolContext

REASONABLE_AMOUNT_OF_TIME = 5
REASONABLE_AMOUNT_OF_TIME_FOR_TERM_CREATION = 0.25

OPENAPI_SERVER_PORT = 8091
OPENAPI_SERVER_URL = f"http://localhost:{OPENAPI_SERVER_PORT}"


@asynccontextmanager
async def run_openapi_server(app: FastAPI) -> AsyncIterator[None]:
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
async def run_service_server(tools: list[ToolEntry]) -> AsyncIterator[PluginServer]:
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
    async def list_agents() -> Any:
        async with API.make_client() as client:
            response = await client.get("/agents/")
            return response.raise_for_status().json()["agents"]

    @staticmethod
    async def create_session(agent_id: str, end_user_id: str) -> Any:
        async with API.make_client() as client:
            response = await client.post(
                "/sessions",
                params={"allow_greeting": False},
                json={
                    "agent_id": agent_id,
                    "end_user_id": end_user_id,
                },
            )

            return response.raise_for_status().json()["session"]

    @staticmethod
    async def get_agent_reply(session_id: str, message: str) -> Any:
        return next(iter(await API.get_agent_replies(session_id, message, 1)))

    @staticmethod
    async def get_agent_replies(
        session_id: str,
        message: str,
        number_of_replies_to_expect: int,
    ) -> list[Any]:
        async with API.make_client() as client:
            try:
                user_message_response = await client.post(
                    f"/sessions/{session_id}/events",
                    json={
                        "content": message,
                    },
                )
                user_message_response.raise_for_status()
                user_message_offset = int(user_message_response.json()["event"]["offset"])

                last_known_offset = user_message_offset

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
    async def create_term(agent_id: str, name: str, description: str, synonyms: str = "") -> Any:
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
    async def read_term(agent_id: str, term_name: str) -> Any:
        async with API.make_client() as client:
            response = await client.get(
                f"/agents/{agent_id}/terms/{term_name}",
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
    async def read_guideline(agent_id: str, guideline_id: str) -> Any:
        async with API.make_client() as client:
            response = await client.get(
                f"/agents/{agent_id}/guidelines/{guideline_id}",
            )

            response.raise_for_status()

            return response.json()

    @staticmethod
    async def create_guideline(
        agent_id: str,
        predicate: str,
        action: str,
        coherence_check: Optional[dict[str, Any]] = None,
        connection_propositions: Optional[dict[str, Any]] = None,
    ) -> Any:
        async with API.make_client() as client:
            response = await client.post(
                f"/agents/{agent_id}/guidelines/",
                json={
                    "invoices": [
                        {
                            "payload": {
                                "kind": "guideline",
                                "predicate": predicate,
                                "action": action,
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
                },
            )

            response.raise_for_status()

            return response.json()["items"][0]["guideline"]

    @staticmethod
    async def create_context_variable(agent_id: str, name: str, description: str) -> Any:
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
    async def read_context_variable_value(agent_id: str, variable_id: str, key: str) -> Any:
        async with API.make_client() as client:
            response = await client.get(
                f"{SERVER_ADDRESS}/agents/{agent_id}/context-variables/{variable_id}/{key}",
            )

            response.raise_for_status()

            return response.json()

    @staticmethod
    async def create_openapi_service(service_name: str, url: str) -> None:
        payload = {"kind": "openapi", "source": f"{url}/openapi.json", "url": url}

        async with API.make_client() as client:
            response = await client.put(f"/services/{service_name}", json=payload)
            response.raise_for_status()


async def test_that_agent_can_be_updated(
    context: ContextOfTest,
) -> None:
    new_description = "Updated description"
    new_max_engine_iterations = 5

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        await run_cli_and_get_exit_status(
            "agent",
            "update",
            "--description",
            new_description,
            "--max-engine-iterations",
            str(new_max_engine_iterations),
        ) == os.EX_OK

        agent = (await API.list_agents())[0]

        assert agent["description"] == new_description
        assert agent["max_engine_iterations"] == new_max_engine_iterations


async def test_that_a_term_can_be_created_with_synonyms(
    context: ContextOfTest,
) -> None:
    term_name = "guideline"
    description = "when and then statements"
    synonyms = "rule, principle"

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

        await run_cli_and_get_exit_status(
            "glossary",
            "add",
            "--agent-id",
            agent_id,
            term_name,
            description,
            "--synonyms",
            synonyms,
        ) == os.EX_OK


async def test_that_a_term_can_be_created_without_synonyms(
    context: ContextOfTest,
) -> None:
    term_name = "guideline_no_synonyms"
    description = "simple guideline with no synonyms"

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

        await run_cli_and_get_exit_status(
            "glossary",
            "add",
            "--agent-id",
            agent_id,
            term_name,
            description,
        ) == os.EX_OK

        term = await API.read_term(agent_id, term_name)
        assert term["name"] == term_name
        assert term["description"] == description
        assert term["synonyms"] == []


async def test_that_terms_can_be_listed(
    context: ContextOfTest,
) -> None:
    guideline_term_name = "guideline"
    tool_term_name = "tool"
    guideline_description = "when and then statements"
    tool_description = "techniuqe to fetch external data"
    guideline_synonyms = "rule, instruction"

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

        _ = await API.create_term(
            agent_id, guideline_term_name, guideline_description, guideline_synonyms
        )

        _ = await API.create_term(agent_id, tool_term_name, tool_description)

        terms = await API.list_terms(agent_id)
        assert len(terms) == 2

        term_names = {term["name"] for term in terms}
        assert guideline_term_name in term_names
        assert tool_term_name in term_names


async def test_that_a_term_can_be_deleted(
    context: ContextOfTest,
) -> None:
    name = "guideline_delete"
    description = "to be deleted"
    synonyms = "rule, principle"

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

        _ = await API.create_term(agent_id, name, description, synonyms)

        assert (
            await run_cli_and_get_exit_status(
                "glossary",
                "remove",
                "--agent-id",
                agent_id,
                name,
            )
            == os.EX_OK
        )

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

        _ = await API.create_term(agent_id, name, description)

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

        term = await API.read_term(agent_id, name)
        assert term["name"] == name
        assert term["description"] == description
        assert term["synonyms"] == []


async def test_that_guideline_can_be_added(
    context: ContextOfTest,
) -> None:
    predicate = "the user greets you"
    action = "greet them back with 'Hello'"

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

        assert (
            await run_cli_and_get_exit_status(
                "guideline",
                "add",
                "-a",
                agent_id,
                predicate,
                action,
            )
            == os.EX_OK
        )

        guidelines = await API.list_guidelines(agent_id)
        assert any(g["predicate"] == predicate and g["action"] == action for g in guidelines)


async def test_that_adding_a_contradictory_guideline_shows_coherence_errors(
    context: ContextOfTest,
) -> None:
    predicate = "the user greets you"
    action = "greet them back with 'Hello'"

    conflicting_action = "ignore the user"

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

        assert (
            await run_cli_and_get_exit_status(
                "guideline",
                "add",
                "-a",
                agent_id,
                predicate,
                action,
            )
            == os.EX_OK
        )

        process = await run_cli(
            "guideline",
            "add",
            "-a",
            agent_id,
            predicate,
            conflicting_action,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()
        output = stdout.decode() + stderr.decode()

        assert "Detected incoherence with other guidelines" in output

        guidelines = await API.list_guidelines(agent_id)

        assert not any(
            g["predicate"] == predicate and g["action"] == conflicting_action for g in guidelines
        )


async def test_that_adding_connected_guidelines_creates_connections(
    context: ContextOfTest,
) -> None:
    predicate1 = "the user asks about the weather"
    action1 = "provide a weather update"

    predicate2 = "providing a weather update"
    action2 = "include temperature and humidity"

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

        assert (
            await run_cli_and_get_exit_status(
                "guideline",
                "add",
                "-a",
                agent_id,
                predicate1,
                action1,
            )
            == os.EX_OK
        )

        assert (
            await run_cli_and_get_exit_status(
                "guideline",
                "add",
                "-a",
                agent_id,
                predicate2,
                action2,
            )
            == os.EX_OK
        )

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


async def test_that_guideline_can_be_viewed(
    context: ContextOfTest,
) -> None:
    predicate = "the user says goodbye"
    action = "say 'Goodbye' back"

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

        guideline = await API.create_guideline(
            agent_id=agent_id, predicate=predicate, action=action
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
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert process.returncode == os.EX_OK

        assert guideline["id"] in output_view
        assert predicate in output_view
        assert action in output_view


async def test_that_view_guideline_with_connections_displays_indirect_and_direct_connections(
    context: ContextOfTest,
) -> None:
    predicate1 = "AAA"
    action1 = "BBB"

    predicate2 = "BBB"
    action2 = "CCC"

    predicate3 = "CCC"
    action3 = "DDD"

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(30),
        ) as client:
            add_guidelines_response = await client.post(
                f"{SERVER_ADDRESS}/agents/{agent_id}/guidelines/",
                json={
                    "invoices": [
                        {
                            "payload": {
                                "kind": "guideline",
                                "predicate": predicate1,
                                "action": action1,
                            },
                            "checksum": "checksum_value",
                            "approved": True,
                            "data": {
                                "coherence_checks": [],
                                "connection_propositions": [
                                    {
                                        "check_kind": "connection_with_another_evaluated_guideline",
                                        "source": {
                                            "predicate": predicate1,
                                            "action": action1,
                                        },
                                        "target": {
                                            "predicate": predicate2,
                                            "action": action2,
                                        },
                                        "connection_kind": "entails",
                                    }
                                ],
                            },
                            "error": None,
                        },
                        {
                            "payload": {
                                "kind": "guideline",
                                "predicate": predicate2,
                                "action": action2,
                            },
                            "checksum": "checksum_value",
                            "approved": True,
                            "data": {
                                "coherence_checks": [],
                                "connection_propositions": [
                                    {
                                        "check_kind": "connection_with_another_evaluated_guideline",
                                        "source": {
                                            "predicate": predicate1,
                                            "action": action1,
                                        },
                                        "target": {
                                            "predicate": predicate2,
                                            "action": action2,
                                        },
                                        "connection_kind": "entails",
                                    },
                                    {
                                        "check_kind": "connection_with_another_evaluated_guideline",
                                        "source": {
                                            "predicate": predicate2,
                                            "action": action2,
                                        },
                                        "target": {
                                            "predicate": predicate3,
                                            "action": action3,
                                        },
                                        "connection_kind": "entails",
                                    },
                                ],
                            },
                            "error": None,
                        },
                        {
                            "payload": {
                                "kind": "guideline",
                                "predicate": predicate3,
                                "action": action3,
                            },
                            "checksum": "checksum_value",
                            "approved": True,
                            "data": {
                                "coherence_checks": [],
                                "connection_propositions": [
                                    {
                                        "check_kind": "connection_with_another_evaluated_guideline",
                                        "source": {
                                            "predicate": predicate2,
                                            "action": action2,
                                        },
                                        "target": {
                                            "predicate": predicate3,
                                            "action": action3,
                                        },
                                        "connection_kind": "entails",
                                    }
                                ],
                            },
                            "error": None,
                        },
                    ]
                },
            )

            add_guidelines_response.raise_for_status()

        first = add_guidelines_response.json()["items"][0]["guideline"]
        first_connection = add_guidelines_response.json()["items"][0]["connections"][0]
        second_connection = add_guidelines_response.json()["items"][1]["connections"][0]

        process = await run_cli(
            "guideline",
            "view",
            "-a",
            agent_id,
            first["id"],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert process.returncode == os.EX_OK

        assert first["id"] in output_view
        assert predicate1 in output_view
        assert action1 in output_view

        assert "Direct Entailments:" in output_view
        assert first_connection["id"] in output_view
        assert predicate2 in output_view
        assert action2 in output_view

        assert "Indirect Entailments:" in output_view
        assert second_connection["id"] in output_view
        assert predicate3 in output_view
        assert action3 in output_view


async def test_that_guidelines_can_be_listed(
    context: ContextOfTest,
) -> None:
    predicate1 = "the user asks for help"
    action1 = "provide assistance"

    predicate2 = "the user needs support"
    action2 = "offer support"

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

        _ = await API.create_guideline(agent_id=agent_id, predicate=predicate1, action=action1)
        _ = await API.create_guideline(agent_id=agent_id, predicate=predicate2, action=action2)

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

        assert predicate1 in output_list
        assert action1 in output_list
        assert predicate2 in output_list
        assert action2 in output_list


async def test_that_guidelines_can_be_entailed(
    context: ContextOfTest,
) -> None:
    predicate1 = "the user needs assistance"
    action1 = "provide help"

    predicate2 = "user ask about a certain subject"
    action2 = "offer detailed explanation"

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

        assert (
            await run_cli_and_get_exit_status(
                "guideline",
                "add",
                "-a",
                agent_id,
                "--no-check",
                "--no-index",
                predicate1,
                action1,
            )
            == os.EX_OK
        )

        assert (
            await run_cli_and_get_exit_status(
                "guideline",
                "add",
                "-a",
                agent_id,
                "--no-check",
                "--no-index",
                predicate2,
                action2,
            )
            == os.EX_OK
        )

        guidelines = await API.list_guidelines(agent_id)

        first_guideline = next(
            g for g in guidelines if g["predicate"] == predicate1 and g["action"] == action1
        )
        second_guideline = next(
            g for g in guidelines if g["predicate"] == predicate2 and g["action"] == action2
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
    predicate1 = "the user needs assistance"
    action1 = "provide help"

    predicate2 = "user ask about a certain subject"
    action2 = "offer detailed explanation"

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

        assert (
            await run_cli_and_get_exit_status(
                "guideline",
                "add",
                "-a",
                agent_id,
                "--no-check",
                "--no-index",
                predicate1,
                action1,
            )
            == os.EX_OK
        )

        assert (
            await run_cli_and_get_exit_status(
                "guideline",
                "add",
                "-a",
                agent_id,
                "--no-check",
                "--no-index",
                predicate2,
                action2,
            )
            == os.EX_OK
        )

        guidelines = await API.list_guidelines(agent_id)

        first_guideline = next(
            g for g in guidelines if g["predicate"] == predicate1 and g["action"] == action1
        )
        second_guideline = next(
            g for g in guidelines if g["predicate"] == predicate2 and g["action"] == action2
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


async def test_that_guideline_can_be_removed(
    context: ContextOfTest,
) -> None:
    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

        guideline = await API.create_guideline(
            agent_id, predicate="the user greets you", action="greet them back with 'Hello'"
        )

        assert (
            await run_cli_and_get_exit_status(
                "guideline",
                "remove",
                "-a",
                agent_id,
                guideline["id"],
            )
            == os.EX_OK
        )

        guidelines = await API.list_guidelines(agent_id)
        assert len(guidelines) == 0


async def test_that_connection_can_be_removed(
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
                                "kind": "guideline",
                                "predicate": "the user greets you",
                                "action": "greet them back with 'Hello'",
                            },
                            "checksum": "checksum_value",
                            "approved": True,
                            "data": {
                                "coherence_checks": [],
                                "connection_propositions": [
                                    {
                                        "check_kind": "connection_with_another_evaluated_guideline",
                                        "source": {
                                            "predicate": "the user greets you",
                                            "action": "greet them back with 'Hello'",
                                        },
                                        "target": {
                                            "predicate": "greeting the user",
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
                                "kind": "guideline",
                                "predicate": "greeting the user",
                                "action": "ask for his health condition",
                            },
                            "checksum": "checksum_value",
                            "approved": True,
                            "data": {
                                "coherence_checks": [],
                                "connection_propositions": [
                                    {
                                        "check_kind": "connection_with_another_evaluated_guideline",
                                        "source": {
                                            "predicate": "the user greets you",
                                            "action": "greet them back with 'Hello'",
                                        },
                                        "target": {
                                            "predicate": "greeting the user",
                                            "action": "ask for his health condition",
                                        },
                                        "connection_kind": "entails",
                                    }
                                ],
                            },
                            "error": None,
                        },
                    ]
                },
            )

            guidelines_response.raise_for_status()
            first = guidelines_response.json()["items"][0]["guideline"]["id"]
            second = guidelines_response.json()["items"][1]["guideline"]["id"]

        assert (
            await run_cli_and_get_exit_status(
                "guideline",
                "disentail",
                "-a",
                agent_id,
                first,
                second,
            )
            == os.EX_OK
        )

        guideline = await API.read_guideline(agent_id, first)
        assert len(guideline["connections"]) == 0


async def test_that_variables_can_be_listed(
    context: ContextOfTest,
) -> None:
    name1 = "test_variable1"
    description1 = "test variable one"

    name2 = "test_variable2"
    description2 = "test variable two"

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
        process_output = stdout.decode() + stderr.decode()
        assert process.returncode == os.EX_OK

        assert name1 in process_output
        assert description1 in process_output
        assert name2 in process_output
        assert description2 in process_output


async def test_that_variable_can_be_added(
    context: ContextOfTest,
) -> None:
    name = "test_variable_cli"
    description = "Variable added via CLI"

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

        assert (
            await run_cli_and_get_exit_status(
                "variable",
                "add",
                "--agent-id",
                agent_id,
                "--description",
                description,
                name,
            )
            == os.EX_OK
        )

        variables = await API.list_context_variables(agent_id)

        variable = next(
            (v for v in variables if v["name"] == name and v["description"] == description),
            None,
        )
        assert variable is not None, "Variable was not added"


async def test_that_variable_can_be_removed(
    context: ContextOfTest,
) -> None:
    name = "test_variable_to_remove"
    description = "Variable to be removed via CLI"

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

        _ = await API.create_context_variable(agent_id, name, description)

        assert (
            await run_cli_and_get_exit_status(
                "variable",
                "remove",
                "--agent-id",
                agent_id,
                name,
            )
            == os.EX_OK
        )

        variables = await API.list_context_variables(agent_id)
        assert len(variables) == 0


async def test_that_variable_value_can_be_set_with_json(
    context: ContextOfTest,
) -> None:
    variable_name = "test_variable_set"
    variable_description = "Variable to test setting value via CLI"
    key = "test_key"
    data = {"test": "data", "type": 27}

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()
        variable = await API.create_context_variable(agent_id, variable_name, variable_description)

        assert (
            await run_cli_and_get_exit_status(
                "variable",
                "set",
                "--agent-id",
                agent_id,
                variable_name,
                key,
                json.dumps(data),
            )
            == os.EX_OK
        )

        value = await API.read_context_variable_value(agent_id, variable["id"], key)
        assert json.loads(value["data"]) == data


async def test_that_variable_value_can_be_set_with_string(
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

        assert (
            await run_cli_and_get_exit_status(
                "variable",
                "set",
                "--agent-id",
                agent_id,
                variable_name,
                key,
                data,
            )
            == os.EX_OK
        )

        value = await API.read_context_variable_value(agent_id, variable["id"], key)

        assert value["data"] == data


async def test_that_variable_values_can_be_retrieved(
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


async def test_that_a_message_interaction_can_be_inspected(
    context: ContextOfTest,
) -> None:
    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent_id = await API.get_first_agent_id()

        guideline = await API.create_guideline(
            agent_id=agent_id,
            predicate="the user talks about cows",
            action="address the user by his first name and say you like Pepsi",
        )

        term = await API.create_term(
            agent_id=agent_id,
            name="Bazoo",
            description="a type of cow",
        )

        variable = await API.create_context_variable(
            agent_id=agent_id,
            name="User first name",
            description="",
        )

        end_user_id = "john.s@peppery.co"

        await API.update_context_variable_value(
            agent_id=agent_id,
            variable_id=variable["id"],
            key=end_user_id,
            value="Johnny",
        )

        session = await API.create_session(agent_id, end_user_id)

        reply_event = await API.get_agent_reply(session["id"], "Oh do I like bazoos")

        assert "Johnny" in reply_event["data"]["message"]
        assert "Pepsi" in reply_event["data"]["message"]

        process = await run_cli(
            "session",
            "inspect",
            session["id"],
            reply_event["correlation_id"],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()
        output = stdout.decode() + stderr.decode()
        assert process.returncode == os.EX_OK

        assert guideline["predicate"] in output
        assert guideline["action"] in output
        assert term["name"] in output
        assert term["description"] in output
        assert variable["name"] in output
        assert end_user_id in output


async def test_that_openapi_service_can_be_added_via_file(
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


async def test_that_openapi_service_can_be_added_via_url(
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


async def test_that_sdk_service_can_be_added(
    context: ContextOfTest,
) -> None:
    service_name = "test_sdk_service"
    service_kind = "sdk"

    @tool
    def sample_tool(context: ToolContext, param: int) -> ToolResult:
        """I want to check also the description here.
        So for that, I will just write multiline text, so I can test both the
        limit of chars in one line, and also, test that multiline works as expected
        and displayed such that the user can easily read and understand it."""
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


async def test_that_service_can_be_removed(
    context: ContextOfTest,
) -> None:
    service_name = "test_service_to_remove"

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        async with run_openapi_server(rng_app()):
            await API.create_openapi_service(service_name, OPENAPI_SERVER_URL)

        assert (
            await run_cli_and_get_exit_status(
                "service",
                "remove",
                service_name,
            )
            == os.EX_OK
        )

        async with API.make_client() as client:
            response = await client.get("/services/")
            response.raise_for_status()
            services = response.json()["services"]
            assert not any(s["name"] == service_name for s in services)


async def test_that_services_can_be_listed(context: ContextOfTest) -> None:
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


async def test_that_services_can_be_viewed(context: ContextOfTest) -> None:
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
