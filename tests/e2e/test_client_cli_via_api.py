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
from asyncio import subprocess
import json
import os
import tempfile
from typing import Any, Optional
import httpx

from parlant.core.services.tools.plugins import tool
from parlant.core.tools import ToolResult, ToolContext

from tests.e2e.test_utilities import (
    CLI_CLIENT_PATH,
    ContextOfTest,
    run_server,
)
from tests.test_utilities import (
    SERVER_ADDRESS,
    run_openapi_server,
    run_service_server,
    run_mcp_server,
)

REASONABLE_AMOUNT_OF_TIME_FOR_TERM_CREATION = 0.25


async def run_cli(*args: str, address: str = SERVER_ADDRESS, **kwargs: Any) -> subprocess.Process:
    exec_args = [
        "uv",
        "run",
        "python",
        CLI_CLIENT_PATH.as_posix(),
        "--server",
        address,
    ] + list(args)

    return await asyncio.create_subprocess_exec(*exec_args, **kwargs)


async def run_cli_and_get_exit_status(*args: str, address: str = SERVER_ADDRESS) -> int:
    exec_args = [
        "uv",
        "run",
        "python",
        CLI_CLIENT_PATH.as_posix(),
        "--server",
        address,
    ] + list(args)

    process = await asyncio.create_subprocess_exec(*exec_args)
    return await process.wait()


async def test_that_an_agent_can_be_added(context: ContextOfTest) -> None:
    name = "TestAgent"
    description = "This is a test agent"

    with run_server(context):
        process = await run_cli(
            "agent",
            "create",
            "--name",
            name,
            "--description",
            description,
            "--max-engine-iterations",
            str(123),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            address=context.api.server_address,
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
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            address=context.api.server_address,
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
        process = await run_cli(
            "agent",
            "update",
            "--name",
            new_name,
            "--description",
            new_description,
            "--max-engine-iterations",
            str(new_max_engine_iterations),
            "--composition-mode",
            "strict_canned",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            address=context.api.server_address,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        agent = (await context.api.list_agents())[0]
        assert agent["name"] == new_name
        assert agent["description"] == new_description
        assert agent["max_engine_iterations"] == new_max_engine_iterations
        assert agent["composition_mode"] == "strict_canned"


async def test_that_an_agent_can_be_deleted(
    context: ContextOfTest,
) -> None:
    name = "Test Agent"

    with run_server(context):
        agent = await context.api.create_agent(name=name)

        assert (
            await run_cli_and_get_exit_status(
                "agent", "delete", "--id", agent["id"], address=context.api.server_address
            )
            == os.EX_OK
        )

        assert not any(a["name"] == name for a in await context.api.list_agents())


async def test_that_sessions_can_be_listed(
    context: ContextOfTest,
) -> None:
    first_customer = "First Customer"
    second_customer = "Second Customer"

    first_title = "First Title"
    second_title = "Second Title"
    third_title = "Third Title"

    with run_server(context):
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
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            address=context.api.server_address,
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
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            address=context.api.server_address,
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
                address=context.api.server_address,
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
        process = await run_cli(
            "glossary",
            "create",
            "--name",
            term_name,
            "--description",
            description,
            "--synonyms",
            synonyms,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            address=context.api.server_address,
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
        process = await run_cli(
            "glossary",
            "create",
            "--name",
            term_name,
            "--description",
            description,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            address=context.api.server_address,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        terms = await context.api.list_terms()
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
        term_to_update = await context.api.create_term(name, description, synonyms)

        process = await run_cli(
            "glossary",
            "update",
            "--id",
            term_to_update["id"],
            "--name",
            new_name,
            "--description",
            new_description,
            "--synonyms",
            new_synonyms,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            address=context.api.server_address,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        updated_term = await context.api.read_term(term_id=term_to_update["id"])
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
        term = await context.api.create_term(name, description, synonyms)

        process = await run_cli(
            "glossary",
            "delete",
            "--id",
            term["id"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            address=context.api.server_address,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        terms = await context.api.list_terms()
        assert len(terms) == 0


async def test_that_a_guideline_can_be_added(
    context: ContextOfTest,
) -> None:
    condition = "the customer greets you"
    action = "greet them back with 'Hello'"

    with run_server(context):
        process = await run_cli(
            "guideline",
            "create",
            "--condition",
            condition,
            "--action",
            action,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            address=context.api.server_address,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        guidelines = await context.api.list_guidelines()
        assert any(g["condition"] == condition and g["action"] == action for g in guidelines)


async def test_that_a_guideline_can_be_updated(
    context: ContextOfTest,
) -> None:
    condition = "the customer asks for help"
    initial_action = "offer assistance"
    updated_action = "provide detailed support information"

    with run_server(context):
        guideline = await context.api.create_guideline(condition=condition, action=initial_action)

        process = await run_cli(
            "guideline",
            "update",
            "--id",
            guideline["id"],
            "--condition",
            condition,
            "--action",
            updated_action,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            address=context.api.server_address,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        updated_guideline = (await context.api.read_guideline(guideline_id=guideline["id"]))[
            "guideline"
        ]

        assert updated_guideline["condition"] == condition
        assert updated_guideline["action"] == updated_action


async def test_that_guidelines_can_be_entailed(
    context: ContextOfTest,
) -> None:
    condition1 = "the customer needs assistance"
    action1 = "provide help"

    condition2 = "customer ask about a certain subject"
    action2 = "offer detailed explanation"

    with run_server(context):
        process = await run_cli(
            "guideline",
            "create",
            "--condition",
            condition1,
            "--action",
            action1,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            address=context.api.server_address,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        process = await run_cli(
            "guideline",
            "create",
            "--condition",
            condition2,
            "--action",
            action2,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            address=context.api.server_address,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        guidelines = await context.api.list_guidelines()

        first_guideline = next(
            g for g in guidelines if g["condition"] == condition1 and g["action"] == action1
        )
        second_guideline = next(
            g for g in guidelines if g["condition"] == condition2 and g["action"] == action2
        )

        process = await run_cli(
            "relationship",
            "create",
            "--kind",
            "entailment",
            "--source",
            first_guideline["id"],
            "--target",
            second_guideline["id"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            address=context.api.server_address,
        )
        await process.communicate()
        await process.wait()
        assert process.returncode == os.EX_OK

        guideline = await context.api.read_guideline(guideline_id=first_guideline["id"])
        assert "relationships" in guideline and len(guideline["relationships"]) == 1
        connection = guideline["relationships"][0]
        assert (
            connection["source_guideline"] == first_guideline
            and connection["target_guideline"] == second_guideline
        )


async def test_that_a_guideline_can_be_deleted(
    context: ContextOfTest,
) -> None:
    with run_server(context):
        guideline = await context.api.create_guideline(
            condition="the customer greets you", action="greet them back with 'Hello'"
        )

        process = await run_cli(
            "guideline",
            "delete",
            "--id",
            guideline["id"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            address=context.api.server_address,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        guidelines = await context.api.list_guidelines()
        assert len(guidelines) == 0


async def test_that_a_tool_can_be_enabled_for_a_guideline(
    context: ContextOfTest,
) -> None:
    with run_server(context):
        guideline = await context.api.create_guideline(
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
                    address=context.api.server_address,
                )
                == os.EX_OK
            )

            assert (
                await run_cli_and_get_exit_status(
                    "guideline",
                    "tool-enable",
                    "--id",
                    guideline["id"],
                    "--service",
                    service_name,
                    "--tool",
                    tool_name,
                    address=context.api.server_address,
                )
                == os.EX_OK
            )

            guideline = await context.api.read_guideline(guideline_id=guideline["id"])

            assert any(
                assoc["tool_id"]["service_name"] == service_name
                and assoc["tool_id"]["tool_name"] == tool_name
                for assoc in guideline["tool_associations"]
            )


async def test_that_a_tool_can_be_disabled_for_a_guideline(
    context: ContextOfTest,
) -> None:
    with run_server(context):
        guideline = await context.api.create_guideline(
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
                    address=context.api.server_address,
                )
                == os.EX_OK
            )

            _ = await context.api.add_association(guideline["id"], service_name, tool_name)

            assert (
                await run_cli_and_get_exit_status(
                    "guideline",
                    "tool-disable",
                    "--id",
                    guideline["id"],
                    "--service",
                    service_name,
                    "--tool",
                    tool_name,
                    address=context.api.server_address,
                )
                == os.EX_OK
            )

            guideline = await context.api.read_guideline(guideline_id=guideline["id"])

            assert guideline["tool_associations"] == []


async def test_that_variables_can_be_listed(
    context: ContextOfTest,
) -> None:
    name1 = "VAR1"
    description1 = "FIRST"

    name2 = "VAR2"
    description2 = "SECOND"

    with run_server(context):
        _ = await context.api.create_context_variable(name1, description1)
        _ = await context.api.create_context_variable(name2, description2)

        process = await run_cli(
            "variable",
            "list",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            address=context.api.server_address,
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
                    address=context.api.server_address,
                )
                == os.EX_OK
            )

            assert (
                await run_cli_and_get_exit_status(
                    "variable",
                    "create",
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
                    address=context.api.server_address,
                )
                == os.EX_OK
            )

        variables = await context.api.list_context_variables()

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
        variable = await context.api.create_context_variable(name, description)

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
                    address=context.api.server_address,
                )
                == os.EX_OK
            )

            assert (
                await run_cli_and_get_exit_status(
                    "variable",
                    "update",
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
                    address=context.api.server_address,
                )
                == os.EX_OK
            )

        updated_variable = await context.api.read_context_variable(variable_id=variable["id"])
        assert updated_variable["context_variable"]["name"] == name
        assert updated_variable["context_variable"]["description"] == new_description
        assert updated_variable["context_variable"]["tool_id"] == {
            "service_name": "local_service",
            "tool_name": "fetch_event_data",
        }
        assert updated_variable["context_variable"]["freshness_rules"] == freshness_rules


async def test_that_a_variable_can_be_deleted(
    context: ContextOfTest,
) -> None:
    name = "test_variable_to_delete"
    description = "Variable to be deleted via CLI"

    with run_server(context):
        variable = await context.api.create_context_variable(name, description)

        process = await run_cli(
            "variable",
            "delete",
            "--id",
            variable["id"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            address=context.api.server_address,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        variables = await context.api.list_context_variables()
        assert len(variables) == 0


async def test_that_a_variable_value_can_be_set_with_json(
    context: ContextOfTest,
) -> None:
    variable_name = "test_variable"
    variable_description = "Variable to test setting value via CLI"
    key = "test_key"
    data: dict[str, Any] = {"test": "data", "type": 27}

    with run_server(context):
        variable = await context.api.create_context_variable(variable_name, variable_description)

        process = await run_cli(
            "variable",
            "set",
            "--id",
            variable["id"],
            "--key",
            key,
            "--value",
            json.dumps(data),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            address=context.api.server_address,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        value = await context.api.read_context_variable_value(variable_id=variable["id"], key=key)
        assert json.loads(value["data"]) == data


async def test_that_a_variable_value_can_be_set_with_string(
    context: ContextOfTest,
) -> None:
    variable_name = "test_variable"
    variable_description = "Variable to test setting value via CLI"
    key = "test_key"
    data = "test_string"

    with run_server(context):
        variable = await context.api.create_context_variable(variable_name, variable_description)

        process = await run_cli(
            "variable",
            "set",
            "--id",
            variable["id"],
            "--key",
            key,
            "--value",
            data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            address=context.api.server_address,
        )
        stdout_view, stderr_view = await process.communicate()
        output_view = stdout_view.decode() + stderr_view.decode()
        assert "Traceback (most recent call last):" not in output_view
        assert process.returncode == os.EX_OK

        value = await context.api.read_context_variable_value(variable_id=variable["id"], key=key)

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
        variable = await context.api.create_context_variable(variable_name, variable_description)

        for key, data in values.items():
            await context.api.update_context_variable_value(
                variable_id=variable["id"], key=key, value=data
            )

        process = await run_cli(
            "variable",
            "get",
            "--id",
            variable["id"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            address=context.api.server_address,
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
            "--id",
            variable["id"],
            "--key",
            specific_key,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            address=context.api.server_address,
        )
        stdout, stderr = await process.communicate()
        output = stdout.decode() + stderr.decode()
        assert process.returncode == os.EX_OK

        assert specific_key in output
        assert expected_value in output


async def test_that_a_variable_value_can_be_deleted(
    context: ContextOfTest,
) -> None:
    name = "test_variable"
    key = "DEFAULT"
    value = "test-value"

    with run_server(context):
        variable = await context.api.create_context_variable(name, description="")
        _ = await context.api.update_context_variable_value(
            variable_id=variable["id"],
            key=key,
            value=value,
        )

        assert (
            await run_cli_and_get_exit_status(
                "variable",
                "delete-value",
                "--id",
                variable["id"],
                "--key",
                key,
                address=context.api.server_address,
            )
            == os.EX_OK
        )

        variable = await context.api.read_context_variable(variable_id=variable["id"])
        assert len(variable["key_value_pairs"]) == 0


async def test_that_an_openapi_service_can_be_added_via_file(
    context: ContextOfTest,
) -> None:
    service_name = "test_openapi_service"
    service_kind = "openapi"

    with run_server(context):
        async with run_openapi_server() as server_info:
            url = f"{server_info.url}:{server_info.port}"
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{url}/openapi.json")
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
                        url,
                        address=context.api.server_address,
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
        async with run_openapi_server() as server_info:
            url = f"{server_info.url}:{server_info.port}"
            source = url + "/openapi.json"

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
                    url,
                    address=context.api.server_address,
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
                    address=context.api.server_address,
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


async def test_that_a_service_can_be_deleted(
    context: ContextOfTest,
) -> None:
    service_name = "test_service_to_delete"

    with run_server(context):
        async with run_openapi_server() as server_info:
            url = f"{server_info.url}:{server_info.port}"
            await context.api.create_openapi_service(service_name, url)

        process = await run_cli(
            "service",
            "delete",
            "--name",
            service_name,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            address=context.api.server_address,
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
        async with run_openapi_server() as server_info:
            url = f"{server_info.url}:{server_info.port}"
            await context.api.create_openapi_service(service_name_1, url)
            await context.api.create_openapi_service(service_name_2, url)

        process = await run_cli(
            "service",
            "list",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            address=context.api.server_address,
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

    with run_server(context):
        async with run_openapi_server() as server_info:
            service_url = f"{server_info.url}:{server_info.port}"
            await context.api.create_openapi_service(service_name, service_url)

        process = await run_cli(
            "service",
            "view",
            "--name",
            service_name,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            address=context.api.server_address,
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
        await context.api.create_customer(name="First Customer")
        await context.api.create_customer(name="Second Customer")

        process = await run_cli(
            "customer",
            "list",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            address=context.api.server_address,
        )
        stdout, stderr = await process.communicate()
        output = stdout.decode() + stderr.decode()
        assert process.returncode == os.EX_OK

        assert "First Customer" in output
        assert "Second Customer" in output


async def test_that_a_customer_can_be_added(context: ContextOfTest) -> None:
    with run_server(context):
        assert (
            await run_cli_and_get_exit_status(
                "customer",
                "create",
                "--name",
                "TestCustomer",
                address=context.api.server_address,
            )
            == os.EX_OK
        )

        customers = await context.api.list_customers()
        assert any(c["name"] == "TestCustomer" for c in customers)


async def test_that_a_customer_can_be_updated(context: ContextOfTest) -> None:
    with run_server(context):
        customer = await context.api.create_customer("TestCustomer")

        assert (
            await run_cli_and_get_exit_status(
                "customer",
                "update",
                "--id",
                customer["id"],
                "--name",
                "UpdatedTestCustomer",
                address=context.api.server_address,
            )
            == os.EX_OK
        )

        updated_customer = await context.api.read_customer(customer["id"])
        assert updated_customer["name"] == "UpdatedTestCustomer"


async def test_that_a_customer_can_be_viewed(context: ContextOfTest) -> None:
    with run_server(context):
        customer_id = (await context.api.create_customer(name="TestCustomer"))["id"]

        process = await run_cli(
            "customer",
            "view",
            "--id",
            customer_id,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            address=context.api.server_address,
        )
        stdout, stderr = await process.communicate()
        output = stdout.decode() + stderr.decode()
        assert process.returncode == os.EX_OK

        assert customer_id in output
        assert "TestCustomer" in output


async def test_that_a_customer_can_be_deleted(context: ContextOfTest) -> None:
    with run_server(context):
        customer_id = (await context.api.create_customer(name="TestCustomer"))["id"]

        assert (
            await run_cli_and_get_exit_status(
                "customer",
                "delete",
                "--id",
                customer_id,
                address=context.api.server_address,
            )
            == os.EX_OK
        )

        customers = await context.api.list_customers()
        assert not any(c["name"] == "TestCustomer" for c in customers)


async def test_that_a_customer_metadata_can_be_set(context: ContextOfTest) -> None:
    with run_server(context):
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
                address=context.api.server_address,
            )
            == os.EX_OK
        )

        customer = await context.api.read_customer(id=customer_id)
        assert customer["metadata"].get("key1") == "value1"


async def test_that_a_customer_metadata_can_be_unset(context: ContextOfTest) -> None:
    with run_server(context):
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
                address=context.api.server_address,
            )
            == os.EX_OK
        )

        customer = await context.api.read_customer(id=customer_id)
        assert "key1" not in customer["metadata"]


async def test_that_a_customer_tag_can_be_added(context: ContextOfTest) -> None:
    with run_server(context):
        customer_id = (await context.api.create_customer(name="TestCustomer"))["id"]
        tag_id = (await context.api.create_tag(name="TestTag"))["id"]

        assert (
            await run_cli_and_get_exit_status(
                "customer",
                "tag",
                "--id",
                customer_id,
                "--tag",
                "TestTag",
                address=context.api.server_address,
            )
            == os.EX_OK
        )
        customer = await context.api.read_customer(id=customer_id)
        tags = customer["tags"]
        assert tag_id in tags


async def test_that_a_customer_tag_can_be_deleted(context: ContextOfTest) -> None:
    with run_server(context):
        customer_id = (await context.api.create_customer(name="TestCustomer"))["id"]
        tag_id = (await context.api.create_tag(name="TestTag"))["id"]
        await context.api.add_customer_tag(customer_id, tag_id)

        assert (
            await run_cli_and_get_exit_status(
                "customer",
                "untag",
                "--id",
                customer_id,
                "--tag",
                tag_id,
                address=context.api.server_address,
            )
            == os.EX_OK
        )
        customer = await context.api.read_customer(id=customer_id)
        tags = customer["tags"]
        assert tag_id not in tags


async def test_that_a_tag_can_be_added(context: ContextOfTest) -> None:
    with run_server(context):
        tag_name = "TestTag"

        assert (
            await run_cli_and_get_exit_status(
                "tag",
                "create",
                "--name",
                tag_name,
                address=context.api.server_address,
            )
            == os.EX_OK
        )

        tags = await context.api.list_tags()
        assert any(t["name"] == tag_name for t in tags)


async def test_that_tags_can_be_listed(context: ContextOfTest) -> None:
    with run_server(context):
        await context.api.create_tag("FirstTag")
        await context.api.create_tag("SecondTag")

        process = await run_cli(
            "tag",
            "list",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            address=context.api.server_address,
        )
        stdout, stderr = await process.communicate()
        output = stdout.decode() + stderr.decode()
        assert process.returncode == os.EX_OK

        assert "FirstTag" in output
        assert "SecondTag" in output


async def test_that_a_tag_can_be_updated(context: ContextOfTest) -> None:
    with run_server(context):
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
                address=context.api.server_address,
            )
            == os.EX_OK
        )

        updated_tag = await context.api.read_tag(tag_id)
        assert updated_tag["name"] == new_name


async def test_that_canned_responses_can_be_initialized(context: ContextOfTest) -> None:
    with run_server(context):
        tmp_file = tempfile.NamedTemporaryFile(delete=False)
        tmp_file_path = tmp_file.name
        tmp_file.close()

        assert (
            await run_cli_and_get_exit_status(
                "canned-response",
                "init",
                tmp_file_path,
                address=context.api.server_address,
            )
            == os.EX_OK
        )

        with open(tmp_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert len(data.get("canned_responses", [])) > 0
        assert all("value" in f for f in data["canned_responses"])

        os.remove(tmp_file_path)


async def test_that_canned_responses_can_be_loaded(context: ContextOfTest) -> None:
    with run_server(context):
        await context.api.create_tag("testTag1")
        await context.api.create_tag("testTag2")

        test_canned_responses = {
            "canned_responses": [
                {
                    "value": "Hello, {{username}}!",
                    "fields": [
                        {
                            "name": "username",
                            "description": "The user's name",
                            "examples": ["Alice", "Bob"],
                        }
                    ],
                    "tags": ["testTag1", "testTag2"],
                },
                {
                    "value": "Your balance is {{balance}}.",
                    "fields": [
                        {
                            "name": "balance",
                            "description": "Account balance",
                            "examples": ["1000", "2000"],
                        }
                    ],
                    "tags": [],
                },
                {
                    "value": "You are welcome (:",
                },
            ]
        }

        tmp_file = tempfile.NamedTemporaryFile(delete=False, mode="w")
        tmp_file_path = tmp_file.name
        json.dump(test_canned_responses, tmp_file, indent=2)
        tmp_file.close()

        assert (
            await run_cli_and_get_exit_status(
                "canned-response", "load", tmp_file_path, address=context.api.server_address
            )
            == os.EX_OK
        )

        canned_responses_in_system = await context.api.list_canned_responses()
        assert len(canned_responses_in_system) == 3

        first = canned_responses_in_system[0]
        assert first["value"] == "Hello, {{username}}!"
        assert "tags" in first
        assert "fields" in first

        os.remove(tmp_file_path)


async def test_that_guidelines_can_be_enabled(context: ContextOfTest) -> None:
    with run_server(context):
        first_guideline = await context.api.create_guideline(
            condition="the customer greets you",
            action="greet them back with 'Hello'",
        )

        second_guideline = await context.api.create_guideline(
            condition="the customer greets you",
            action="greet them back with 'Goodbye'",
        )

        disabled_first_guideline = await context.api.update_guideline(
            first_guideline["id"],
            enabled=False,
        )

        disabled_second_guideline = await context.api.update_guideline(
            second_guideline["id"],
            enabled=False,
        )

        assert disabled_first_guideline["enabled"] is False
        assert disabled_second_guideline["enabled"] is False

        assert (
            await run_cli_and_get_exit_status(
                "guideline",
                "enable",
                "--id",
                first_guideline["id"],
                "--id",
                second_guideline["id"],
                address=context.api.server_address,
            )
        ) == os.EX_OK

        enabled_first_guideline = await context.api.read_guideline(first_guideline["id"])
        assert enabled_first_guideline["guideline"]["enabled"] is True

        enabled_second_guideline = await context.api.read_guideline(second_guideline["id"])
        assert enabled_second_guideline["guideline"]["enabled"] is True


async def test_that_guidelines_can_be_disabled(context: ContextOfTest) -> None:
    with run_server(context):
        first_guideline = await context.api.create_guideline(
            condition="the customer greets you",
            action="greet them back with 'Hello'",
        )

        second_guideline = await context.api.create_guideline(
            condition="the customer greets you",
            action="greet them back with 'Goodbye'",
        )

        assert (
            await run_cli_and_get_exit_status(
                "guideline",
                "disable",
                "--id",
                first_guideline["id"],
                "--id",
                second_guideline["id"],
                address=context.api.server_address,
            )
        ) == os.EX_OK

        disabled_guideline = await context.api.read_guideline(first_guideline["id"])
        assert disabled_guideline["guideline"]["enabled"] is False

        disabled_guideline = await context.api.read_guideline(second_guideline["id"])
        assert disabled_guideline["guideline"]["enabled"] is False


async def test_that_a_guideline_can_be_created_with_tool_id(
    context: ContextOfTest,
) -> None:
    condition = "user provides list of numbers and an optional number"
    tool_id = "parameter_types:give_number_types"

    with run_server(context):
        service_name = "parameter_types"
        tool_name = "give_number_types"

        @tool
        def give_number_types(
            context: ToolContext,
            numbers: list[int],
            optional_number: Optional[int] = None,
        ) -> ToolResult:
            result = {"list_count": len(numbers)}
            if optional_number is not None:
                result["optional_provided"] = True
                result["optional_value"] = optional_number
            return ToolResult(result)

        async with run_service_server([give_number_types]) as server:
            await context.api.create_sdk_service(
                service_name=service_name,
                url=server.url,
            )

            assert (
                await run_cli_and_get_exit_status(
                    "guideline",
                    "create",
                    "--condition",
                    condition,
                    "--tool-id",
                    tool_id,
                    address=context.api.server_address,
                )
            ) == os.EX_OK

            guidelines = await context.api.list_guidelines()
            created_guideline = next((g for g in guidelines if g["condition"] == condition), None)
            assert created_guideline is not None, "Guideline was not created"

            guideline_details = await context.api.read_guideline(
                guideline_id=created_guideline["id"]
            )
            assert any(
                assoc["tool_id"]["service_name"] == service_name
                and assoc["tool_id"]["tool_name"] == tool_name
                for assoc in guideline_details["tool_associations"]
            ), "Tool association was not created"


async def test_that_a_mcp_service_can_be_added(
    context: ContextOfTest,
) -> None:
    service_name = "test_mcp_service"
    service_kind = "mcp"

    def sample_tool(param: int) -> int:
        return param * 2

    with run_server(context):
        async with run_mcp_server([sample_tool]) as server:
            assert (
                await run_cli_and_get_exit_status(
                    "service",
                    "create",
                    "--name",
                    service_name,
                    "--kind",
                    service_kind,
                    "--url",
                    f"{server.url}:{server.port}",
                    address=context.api.server_address,
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


async def test_that_a_mcp_tool_can_be_enabled_and_disabled_for_a_guideline(
    context: ContextOfTest,
) -> None:
    with run_server(context):
        guideline = await context.api.create_guideline(
            condition="the customer wants to get meeting details",
            action="get meeting event information",
        )
        guideline_id = guideline["id"]

        service_name = "google_calendar"
        tool_name = "fetch_event_data"
        service_kind = "mcp"

        def fetch_event_data(event_id: str) -> str:
            return event_id

        async with run_mcp_server([fetch_event_data]) as server:
            assert (
                await run_cli_and_get_exit_status(
                    "service",
                    "create",
                    "--name",
                    service_name,
                    "--kind",
                    service_kind,
                    "--url",
                    f"{server.url}:{server.port}",
                    address=context.api.server_address,
                )
                == os.EX_OK
            )

            assert (
                await run_cli_and_get_exit_status(
                    "guideline",
                    "tool-enable",
                    "--id",
                    guideline_id,
                    "--service",
                    service_name,
                    "--tool",
                    tool_name,
                    address=context.api.server_address,
                )
                == os.EX_OK
            )

            guideline = await context.api.read_guideline(guideline_id=guideline_id)

            assert any(
                assoc["tool_id"]["service_name"] == service_name
                and assoc["tool_id"]["tool_name"] == tool_name
                for assoc in guideline["tool_associations"]
            )

            assert (
                await run_cli_and_get_exit_status(
                    "guideline",
                    "tool-disable",
                    "--id",
                    guideline_id,
                    "--service",
                    service_name,
                    "--tool",
                    tool_name,
                    address=context.api.server_address,
                )
                == os.EX_OK
            )

            guideline = await context.api.read_guideline(guideline_id=guideline_id)

            assert guideline["tool_associations"] == []


async def test_that_a_variable_can_be_added_with_mcp_tool_then_updated(
    context: ContextOfTest,
) -> None:
    name = "test_variable_cli_with mcp"
    description = "Variable added via CLI bound with MCP tool"
    new_description = "Variable (mcp-bound) updated via CLI"

    with run_server(context):
        service_name = "local_service"
        tool_name = "fetch_event_data"
        service_kind = "mcp"
        freshness_rules = "0 0,6,12,18 * * *"

        def fetch_event_data(event_id: str) -> str:
            """Fetch event data based on event ID."""
            return event_id

        async with run_mcp_server([fetch_event_data]) as server:
            assert (
                await run_cli_and_get_exit_status(
                    "service",
                    "create",
                    "--name",
                    service_name,
                    "--kind",
                    service_kind,
                    "--url",
                    f"{server.url}:{server.port}",
                    address=context.api.server_address,
                )
                == os.EX_OK
            )

            assert (
                await run_cli_and_get_exit_status(
                    "variable",
                    "create",
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
                    address=context.api.server_address,
                )
                == os.EX_OK
            )

            variables = await context.api.list_context_variables()

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

            assert (
                await run_cli_and_get_exit_status(
                    "variable",
                    "update",
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
                    address=context.api.server_address,
                )
                == os.EX_OK
            )

        updated_variable = await context.api.read_context_variable(variable_id=variable["id"])
        assert updated_variable["context_variable"]["name"] == name
        assert updated_variable["context_variable"]["description"] == new_description
        assert updated_variable["context_variable"]["tool_id"] == {
            "service_name": "local_service",
            "tool_name": "fetch_event_data",
        }
        assert updated_variable["context_variable"]["freshness_rules"] == freshness_rules
