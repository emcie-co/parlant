import asyncio
import os
import signal
import traceback
import httpx

from emcie.common.tools import ToolContext, ToolResult
from emcie.common.plugin import tool
from emcie.server.core.async_utils import Timeout
from emcie.server.core.agents import Agent

from tests.e2e.test_utilities import (
    SERVER_ADDRESS,
    ContextOfTest,
    create_context_variable,
    create_context_variable_value,
    create_guideline,
    create_sdk_service,
    create_term,
    read_context_variable_value,
    list_guidelines,
    list_services,
    get_term_list,
    run_server,
)
from tests.test_utilities import nlp_test
from tests.core.services.tools.test_plugin_client import run_service_server


REASONABLE_AMOUNT_OF_TIME = 5
EXTENDED_AMOUNT_OF_TIME = 10


async def get_first_agent() -> Agent:
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(30),
    ) as client:
        try:
            agents_response = await client.get(
                f"{SERVER_ADDRESS}/agents/",
            )
            agents_response.raise_for_status()

            assert len(agents_response.json()["agents"]) > 0
            agent = agents_response.json()["agents"][0]

            return Agent(
                id=agent["id"],
                name=agent["name"],
                description=agent["description"],
                creation_utc=agent["creation_utc"],
                max_engine_iterations=agent["max_engine_iterations"],
            )
        except:
            traceback.print_exc()
            raise


async def get_agent_replies(
    context: ContextOfTest,
    message: str,
    number_of_replies_to_expect: int,
    agent: Agent,
) -> list[str]:
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(30),
    ) as client:
        try:
            session_creation_response = await client.post(
                f"{SERVER_ADDRESS}/sessions",
                json={
                    "end_user_id": "test_user",
                    "agent_id": agent.id,
                },
            )
            session_creation_response.raise_for_status()
            session_id = session_creation_response.json()["session"]["id"]

            await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

            user_message_response = await client.post(
                f"{SERVER_ADDRESS}/sessions/{session_id}/events",
                json={
                    "content": message,
                },
            )
            user_message_response.raise_for_status()
            user_message_offset = int(user_message_response.json()["event"]["offset"])

            last_known_offset = user_message_offset

            replies: list[str] = []
            timeout = Timeout(300)

            while len(replies) < number_of_replies_to_expect:
                response = await client.get(
                    f"{SERVER_ADDRESS}/sessions/{session_id}/events",
                    params={
                        "min_offset": last_known_offset + 1,
                        "kinds": "message",
                        "wait": True,
                    },
                )
                response.raise_for_status()
                events = response.json()["events"]

                if message_events := [e for e in events if e["kind"] == "message"]:
                    replies.append(str(message_events[0]["data"]["message"]))

                last_known_offset = events[-1]["offset"]

                if timeout.expired():
                    raise TimeoutError()

            return replies
        except:
            traceback.print_exc()
            raise


async def test_that_the_server_starts_and_shuts_down_cleanly_on_interrupt(
    context: ContextOfTest,
) -> None:
    with run_server(context) as server_process:
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)
        server_process.send_signal(signal.SIGINT)
        server_process.wait(timeout=REASONABLE_AMOUNT_OF_TIME)
        assert server_process.returncode == os.EX_OK


async def test_that_the_server_starts_when_there_are_no_state_changes_and_told_not_to_(
    context: ContextOfTest,
) -> None:
    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent = await get_first_agent()

        agent_replies = await get_agent_replies(
            context,
            message="Hello",
            agent=agent,
            number_of_replies_to_expect=1,
        )

        assert await nlp_test(
            agent_replies[0],
            "It greeting the user",
        )


async def test_that_the_server_starts_and_gernerate_a_message(
    context: ContextOfTest,
) -> None:
    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent = await get_first_agent()

        agent_replies = await get_agent_replies(
            context,
            message="Hello",
            agent=agent,
            number_of_replies_to_expect=1,
        )

        assert await nlp_test(
            agent_replies[0],
            "It greeting the user",
        )


async def test_that_the_server_recovery_restarts_all_active_evaluation_tasks(
    context: ContextOfTest,
) -> None:
    with run_server(context) as server_process:
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent = await get_first_agent()

        payloads = {
            "payloads": [
                {
                    "kind": "guideline",
                    "predicate": "the user greets you",
                    "action": "greet them back with 'Hello'",
                },
                {
                    "kind": "guideline",
                    "predicate": "the user greeting you",
                    "action": "greet them back with 'Hola'",
                },
            ],
        }
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(30),
        ) as client:
            try:
                evaluation_creation_response = await client.post(
                    f"{SERVER_ADDRESS}/agents/{agent.id}/index/evaluations",
                    json=payloads,
                )
                evaluation_creation_response.raise_for_status()
                evaluation_id = evaluation_creation_response.json()["evaluation_id"]

                server_process.send_signal(signal.SIGINT)
                server_process.wait(timeout=1)
                assert server_process.returncode == os.EX_OK
            except:
                traceback.print_exc()
                raise

    with run_server(context) as server_process:
        EXTRA_TIME_TO_LET_THE_TASK_COMPLETE = 5
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME + EXTRA_TIME_TO_LET_THE_TASK_COMPLETE)

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(30),
        ) as client:
            try:
                evaluation_response = await client.get(
                    f"{SERVER_ADDRESS}/agents/index/evaluations/{evaluation_id}",
                )
                evaluation_creation_response.raise_for_status()
                assert evaluation_response.json()["status"] == "completed"
            except:
                traceback.print_exc()
                raise


async def test_that_guidelines_are_loaded_after_server_restarts(
    context: ContextOfTest,
) -> None:
    with run_server(context) as server_process:
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent = await get_first_agent()

        first = await create_guideline(
            agent_id=agent.id,
            predicate="the user greets you",
            action="greet them back with 'Hello'",
        )

        second = await create_guideline(
            agent_id=agent.id,
            predicate="the user say goodbye",
            action="say goodbye",
        )

        server_process.send_signal(signal.SIGINT)
        server_process.wait(timeout=1)
        assert server_process.returncode == os.EX_OK

    with run_server(context) as server_process:
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent = await get_first_agent()

        guidelines = await list_guidelines(agent_id=agent.id)

        assert any(first["predicate"] == g["predicate"] for g in guidelines)
        assert any(first["action"] == g["action"] for g in guidelines)

        assert any(second["predicate"] == g["predicate"] for g in guidelines)
        assert any(second["action"] == g["action"] for g in guidelines)


async def test_that_context_variable_values_load_after_server_restart(
    context: ContextOfTest,
) -> None:
    variable_name = "test_variable_with_value"
    variable_description = "Variable with values"
    key = "test_key"
    data = "test_value"

    with run_server(context) as server_process:
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent = await get_first_agent()

        variable = await create_context_variable(agent.id, variable_name, variable_description)
        await create_context_variable_value(agent.id, variable["id"], key, data)

        server_process.send_signal(signal.SIGINT)
        server_process.wait(timeout=1)
        assert server_process.returncode == os.EX_OK

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent = await get_first_agent()
        variable_value = await read_context_variable_value(agent.id, variable["id"], key)

        assert variable_value["data"] == data


async def test_that_services_load_after_server_restart(context: ContextOfTest) -> None:
    service_name = "test_service"
    service_kind = "sdk"

    @tool
    def sample_tool(context: ToolContext, param: int) -> ToolResult:
        return ToolResult(param * 2)

    with run_server(context) as server_process:
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        async with run_service_server([sample_tool]) as server:
            await create_sdk_service(service_name, server.url)

        server_process.send_signal(signal.SIGINT)
        server_process.wait(timeout=1)
        assert server_process.returncode == os.EX_OK

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        services = await list_services()
        assert any(s["name"] == service_name for s in services)
        assert any(s["kind"] == service_kind for s in services)


async def test_that_glossary_terms_load_after_server_restart(context: ContextOfTest) -> None:
    term_name = "test_term"
    description = "Term added before server restart"

    with run_server(context) as server_process:
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent = await get_first_agent()

        await create_term(agent.id, term_name, description)

        server_process.send_signal(signal.SIGINT)
        server_process.wait(timeout=3)
        assert server_process.returncode == os.EX_OK

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent = await get_first_agent()
        terms = await get_term_list(agent.id)

        assert any(t["name"] == term_name for t in terms)
        assert any(t["description"] == description for t in terms)
