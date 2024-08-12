import asyncio
import os
import signal
import traceback
import httpx

from emcie.common.plugin import PluginServer, ToolContext, tool

from emcie.server.core.sessions import Event
from tests.e2e.test_utilities import (
    DEFAULT_AGENT_NAME,
    SERVER_ADDRESS,
    _Guideline,
    _TestContext,
    find_guideline,
    load_active_agent,
    read_guideline_config,
    read_loaded_guidelines,
    run_server,
    write_guideline_config,
    write_service_config,
)
from tests.test_utilities import nlp_test


REASONABLE_AMOUNT_OF_TIME = 5


async def get_quick_reply_from_agent(
    context: _TestContext,
    message: str,
    agent_name: str = DEFAULT_AGENT_NAME,
) -> str:
    agent = load_active_agent(home_dir=context.home_dir, agent_name=agent_name)

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(30),
    ) as client:
        try:
            session_creation_response = await client.post(
                f"{SERVER_ADDRESS}/sessions",
                json={
                    "end_user_id": "test_user",
                    "agent_id": agent["id"],
                },
            )
            session_creation_response.raise_for_status()
            session_id = session_creation_response.json()["session_id"]

            await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

            user_message_response = await client.post(
                f"{SERVER_ADDRESS}/sessions/{session_id}/events",
                json={
                    "content": message,
                },
            )
            user_message_response.raise_for_status()
            user_message_offset = int(user_message_response.json()["event_offset"])

            last_known_offset = user_message_offset

            for _ in range(10):
                response = await client.get(
                    f"{SERVER_ADDRESS}/sessions/{session_id}/events",
                    params={
                        "min_offset": last_known_offset + 1,
                        "wait": True,
                    },
                )

                response.raise_for_status()

                events = response.json()["events"]

                if message_events := [e for e in events if e["kind"] == Event.MESSAGE_KIND]:
                    return str(message_events[0]["data"]["message"])
                else:
                    last_known_offset = events[-1]["offset"]

            raise TimeoutError()
        except:
            traceback.print_exc()
            raise


async def test_that_the_server_starts_and_shuts_down_cleanly_on_interrupt(
    context: _TestContext,
) -> None:
    with run_server(context) as server_process:
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)
        server_process.send_signal(signal.SIGINT)
        server_process.wait(timeout=REASONABLE_AMOUNT_OF_TIME)
        assert server_process.returncode == os.EX_OK


async def test_that_the_server_hot_reloads_guideline_changes(
    context: _TestContext,
) -> None:
    with run_server(context):
        initial_guidelines = read_guideline_config(context.config_file)

        new_guideline: _Guideline = {
            "when": "talking about bananas",
            "then": "say they're very tasty",
        }

        write_guideline_config(
            new_guidelines=initial_guidelines + [new_guideline],
            config_file=context.config_file,
        )

        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        loaded_guidelines = read_loaded_guidelines(context.home_dir)

        assert find_guideline(new_guideline, within=loaded_guidelines)

        agent_reply = await get_quick_reply_from_agent(context, message="what are bananas?")

        assert nlp_test(agent_reply, "It says that bananas are very tasty")


async def test_that_the_server_loads_and_interacts_with_a_plugin(
    context: _TestContext,
) -> None:
    @tool(id="about_dor", name="about_dor")
    def about_dor(context: ToolContext) -> str:
        """Gets information about Dor"""
        return "Dor makes great pizza"

    plugin_port = 8090

    write_service_config(
        [
            {
                "type": "plugin",
                "name": "my_plugin",
                "url": f"http://localhost:{plugin_port}",
            }
        ],
        context.config_file,
    )

    write_guideline_config(
        [
            {
                "when": "the user says hello",
                "then": "tell the user about Dor",
                "enabled_tools": ["my_plugin__about_dor"],
            }
        ],
        config_file=context.config_file,
    )

    async with PluginServer(name="my_plugin", tools=[about_dor], port=plugin_port) as plugin_server:
        try:
            with run_server(context):
                await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

                agent_reply = await get_quick_reply_from_agent(context, message="Hello")

                assert nlp_test(agent_reply, "It says that Dor makes great pizza")

        finally:
            await plugin_server.shutdown()
