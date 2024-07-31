from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, AsyncIterator
import tempfile
from lagom import Container
from pytest import fixture, mark
import pytest

from emcie.server.core.agents import AgentDocumentStore, AgentId, AgentStore
from emcie.server.core.context_variables import (
    ContextVariableDocumentStore,
)
from emcie.server.core.end_users import EndUserDocumentStore, EndUserId
from emcie.server.core.guidelines import (
    GuidelineDocumentStore,
    GuidelineId,
)
from emcie.server.core.persistence.json_file_database import JSONFileDocumentDatabase
from emcie.server.core.sessions import Event, SessionDocumentStore
from emcie.server.core.tools import LocalToolService, ToolId
from emcie.server.core.guideline_tool_associations import (
    GuidelineToolAssociationDocumentStore,
)
from tests.test_utilities import SyncAwaiter


@fixture
def agent_id(
    container: Container,
    sync_await: SyncAwaiter,
) -> AgentId:
    store = container[AgentStore]
    agent = sync_await(store.create_agent(name="test-agent"))
    return agent.id


@dataclass
class _TestContext:
    container: Container
    agent_id: AgentId
    sync_await: SyncAwaiter


@fixture
def context(
    container: Container,
    agent_id: AgentId,
    sync_await: SyncAwaiter,
) -> _TestContext:
    return _TestContext(container, agent_id, sync_await)


@fixture
async def new_file() -> AsyncIterator[Path]:
    with tempfile.NamedTemporaryFile() as file:
        yield Path(file.name)


@mark.parametrize(
    ("agent_configuration"),
    [
        ({"name": "Test Agent"}),
        ({"name": "Test Agent", "description": "You are a test agent"}),
    ],
)
async def test_agent_creation(
    new_file: Path,
    agent_configuration: dict[str, Any],
) -> None:
    async with JSONFileDocumentDatabase(new_file) as agent_db:
        agent_store = AgentDocumentStore(agent_db)
        agent = await agent_store.create_agent(**agent_configuration)

        agents = list(await agent_store.list_agents())

        assert len(agents) == 1
        assert agents[0] == agent

    with open(new_file) as f:
        agents_from_json = json.load(f)

    assert len(agents_from_json["agents"]) == 1

    json_agent = agents_from_json["agents"][0]
    assert json_agent["id"] == agent.id
    assert json_agent["name"] == agent.name
    assert json_agent["description"] == agent.description
    assert datetime.fromisoformat(json_agent["creation_utc"]) == agent.creation_utc


async def test_session_creation(
    context: _TestContext,
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(new_file) as session_db:
        session_store = SessionDocumentStore(session_db)
        end_user_id = EndUserId("test_user")
        session = await session_store.create_session(
            end_user_id=end_user_id,
            agent_id=context.agent_id,
        )

    with open(new_file) as f:
        sessions_from_json = json.load(f)

    assert len(sessions_from_json["sessions"]) == 1
    json_session = sessions_from_json["sessions"][0]
    assert json_session["id"] == session.id
    assert json_session["end_user_id"] == end_user_id
    assert json_session["agent_id"] == context.agent_id
    assert json_session["consumption_offsets"] == {
        "client": 0,
    }


async def test_event_creation(
    context: _TestContext,
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(new_file) as session_db:
        session_store = SessionDocumentStore(session_db)
        end_user_id = EndUserId("test_user")
        session = await session_store.create_session(
            end_user_id=end_user_id,
            agent_id=context.agent_id,
        )

        event = await session_store.create_event(
            session_id=session.id,
            source="client",
            kind=Event.MESSAGE_KIND,
            data={"message": "Hello, world!"},
            creation_utc=datetime.now(timezone.utc),
        )

    with open(new_file) as f:
        events_from_json = json.load(f)

    assert len(events_from_json["events"]) == 1
    json_event = events_from_json["events"][0]
    assert json_event["kind"] == event.kind
    assert json_event["data"] == event.data
    assert json_event["source"] == event.source
    assert datetime.fromisoformat(json_event["creation_utc"]) == event.creation_utc


async def test_guideline_creation_and_loading_data_from_file(
    context: _TestContext,
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(new_file) as guideline_db:
        guideline_store = GuidelineDocumentStore(guideline_db)
        guideline = await guideline_store.create_guideline(
            guideline_set=context.agent_id,
            predicate="Creating a guideline with JSONFileDatabase implementation",
            content="Expecting it to show in the guidelines json file",
        )

    with open(new_file) as f:
        guidelines_from_json = json.load(f)

    assert len(guidelines_from_json["guidelines"]) == 1

    json_guideline = guidelines_from_json["guidelines"][0]
    assert json_guideline["guideline_set"] == context.agent_id

    assert json_guideline["predicate"] == guideline.predicate
    assert json_guideline["content"] == guideline.content
    assert datetime.fromisoformat(json_guideline["creation_utc"]) == guideline.creation_utc

    async with JSONFileDocumentDatabase(new_file) as guideline_db:
        guideline_store = GuidelineDocumentStore(guideline_db)

        second_guideline = await guideline_store.create_guideline(
            guideline_set=context.agent_id,
            predicate="Second guideline creation",
            content="Additional test entry in the JSON file",
        )

    with open(new_file) as f:
        guidelines_from_json = json.load(f)

    assert len(guidelines_from_json["guidelines"]) == 2

    second_json_guideline = guidelines_from_json["guidelines"][1]
    assert second_json_guideline["guideline_set"] == context.agent_id

    assert second_json_guideline["predicate"] == second_guideline.predicate
    assert second_json_guideline["content"] == second_guideline.content
    assert (
        datetime.fromisoformat(second_json_guideline["creation_utc"])
        == second_guideline.creation_utc
    )


async def test_guideline_retrieval(
    context: _TestContext,
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(new_file) as guideline_db:
        guideline_store = GuidelineDocumentStore(guideline_db)
        await guideline_store.create_guideline(
            guideline_set=context.agent_id,
            predicate="Test predicate for loading",
            content="Test content for loading guideline",
        )

        loaded_guidelines = await guideline_store.list_guidelines(context.agent_id)
        loaded_guideline_list = list(loaded_guidelines)

        assert len(loaded_guideline_list) == 1
        loaded_guideline = loaded_guideline_list[0]
        assert loaded_guideline.predicate == "Test predicate for loading"
        assert loaded_guideline.content == "Test content for loading guideline"


async def test_tool_creation(
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(new_file) as tool_db:
        tool_store = LocalToolService(tool_db)
        tool = await tool_store.create_tool(
            name="Unique tool name",
            module_path="path/to/module",
            description="A tool for testing JSON persistence",
            parameters={"param1": {"type": "string"}, "param2": {"type": "number"}},
            required=["param1"],
            consequential=True,
        )

    with open(new_file) as f:
        tools_from_json = json.load(f)

    assert len(tools_from_json["tools"]) == 1
    json_tool = tools_from_json["tools"][0]

    assert json_tool["name"] == tool.name
    assert json_tool["description"] == tool.description
    assert json_tool["parameters"] == tool.parameters
    assert json_tool["required"] == tool.required
    assert json_tool["consequential"] == tool.consequential


async def test_tool_retrieval(
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(new_file) as tool_db:
        tool_store = LocalToolService(tool_db)
        tool = await tool_store.create_tool(
            name="Tool for loading test",
            module_path="path/to/tool/module",
            description="Testing tool load functionality",
            parameters={"param1": {"type": "string"}},
            required=["param1"],
            consequential=False,
        )

        loaded_tools = await tool_store.list_tools()
        loaded_tool_list = list(loaded_tools)

        assert len(loaded_tool_list) == 1
        loaded_tool = loaded_tool_list[0]
        assert loaded_tool == tool


async def test_end_user_creation(
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(new_file) as end_user_db:
        end_user_store = EndUserDocumentStore(end_user_db)
        name = "Jane Doe"
        email = "jane.doe@example.com"
        created_user = await end_user_store.create_end_user(
            name=name,
            email=email,
        )

    with open(new_file, "r") as file:
        data = json.load(file)

    assert len(data["end_users"]) == 1
    json_end_user = data["end_users"][0]
    assert json_end_user["name"] == name
    assert json_end_user["email"] == email
    assert datetime.fromisoformat(json_end_user["creation_utc"]) == created_user.creation_utc


async def test_end_user_retrieval(
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(new_file) as end_user_db:
        end_user_store = EndUserDocumentStore(end_user_db)
        name = "John Doe"
        email = "john.doe@example.com"

        created_user = await end_user_store.create_end_user(name=name, email=email)

        retrieved_user = await end_user_store.read_end_user(created_user.id)

        assert created_user == retrieved_user


async def test_context_variable_creation(
    context: _TestContext,
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(new_file) as context_variable_db:
        context_variable_store = ContextVariableDocumentStore(context_variable_db)
        tool_id = ToolId("test_tool")
        variable = await context_variable_store.create_variable(
            variable_set=context.agent_id,
            name="Sample Variable",
            description="A test variable for persistence.",
            tool_id=tool_id,
            freshness_rules=None,
        )

    with open(new_file) as f:
        variables_from_json = json.load(f)

    assert len(variables_from_json["variables"]) == 1
    json_variable = variables_from_json["variables"][0]

    assert json_variable["variable_set"] == context.agent_id
    assert json_variable["name"] == variable.name
    assert json_variable["description"] == variable.description
    assert json_variable["tool_id"] == tool_id


async def test_context_variable_value_update_and_retrieval(
    context: _TestContext,
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(new_file) as context_variable_db:
        context_variable_store = ContextVariableDocumentStore(context_variable_db)
        tool_id = ToolId("test_tool")
        end_user_id = EndUserId("test_user")
        variable = await context_variable_store.create_variable(
            variable_set=context.agent_id,
            name="Sample Variable",
            description="A test variable for persistence.",
            tool_id=tool_id,
            freshness_rules=None,
        )

        await context_variable_store.update_value(
            variable_set=context.agent_id,
            key=end_user_id,
            variable_id=variable.id,
            data={"key": "value"},
        )
        value = await context_variable_store.read_value(
            variable_set=context.agent_id, key=end_user_id, variable_id=variable.id
        )

    with open(new_file) as f:
        values_from_json = json.load(f)

    assert len(values_from_json["values"]) == 1
    json_value = values_from_json["values"][0]

    assert json_value["data"] == value.data
    assert json_value["variable_id"] == value.variable_id
    assert json_value["data"] == value.data


async def test_context_variable_listing(
    context: _TestContext,
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(new_file) as context_variable_db:
        context_variable_store = ContextVariableDocumentStore(context_variable_db)
        tool_id = ToolId("test_tool")
        var1 = await context_variable_store.create_variable(
            variable_set=context.agent_id,
            name="Variable One",
            description="First test variable",
            tool_id=tool_id,
            freshness_rules=None,
        )

        var2 = await context_variable_store.create_variable(
            variable_set=context.agent_id,
            name="Variable Two",
            description="Second test variable",
            tool_id=tool_id,
            freshness_rules=None,
        )

        variables = list(await context_variable_store.list_variables(context.agent_id))
        assert var1 in variables
        assert var2 in variables
        assert len(variables) == 2


async def test_context_variable_deletion(
    context: _TestContext,
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(new_file) as context_variable_db:
        context_variable_store = ContextVariableDocumentStore(context_variable_db)
        tool_id = ToolId("test_tool")
        variable = await context_variable_store.create_variable(
            variable_set=context.agent_id,
            name="Deletable Variable",
            description="A variable to be deleted.",
            tool_id=tool_id,
            freshness_rules=None,
        )

        value_data = {"key": "test", "data": "This is a test value"}
        await context_variable_store.update_value(
            variable_set=context.agent_id,
            key="test_user",
            variable_id=variable.id,
            data=value_data,
        )

        variables_before_deletion = list(
            await context_variable_store.list_variables(context.agent_id)
        )

        assert any(v.id == variable.id for v in variables_before_deletion)

        value_before_deletion = await context_variable_store.read_value(
            variable_set=context.agent_id,
            key="test_user",
            variable_id=variable.id,
        )

        assert value_before_deletion.data == value_data

        await context_variable_store.delete_variable(
            variable_set=context.agent_id,
            id=variable.id,
        )

        variables_after_deletion = list(
            await context_variable_store.list_variables(context.agent_id)
        )

        assert all(var.id != variable.id for var in variables_after_deletion)

        with pytest.raises(ValueError):
            await context_variable_store.read_value(
                variable_set=context.agent_id, key="test_user", variable_id=variable.id
            )


async def test_guideline_tool_association_creation(
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(new_file) as guideline_tool_association_db:
        guideline_tool_association_store = GuidelineToolAssociationDocumentStore(
            guideline_tool_association_db
        )
        guideline_id = GuidelineId("guideline-789")
        tool_id = ToolId("tool-012")

        await guideline_tool_association_store.create_association(
            guideline_id=guideline_id, tool_id=tool_id
        )

    with open(new_file, "r") as f:
        guideline_tool_associations_from_json = json.load(f)

    assert len(guideline_tool_associations_from_json["associations"]) == 1
    json_variable = guideline_tool_associations_from_json["associations"][0]

    assert json_variable["guideline_id"] == guideline_id
    assert json_variable["tool_id"] == tool_id


async def test_guideline_tool_association_retrieval(
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(new_file) as guideline_tool_association_db:
        guideline_tool_association_store = GuidelineToolAssociationDocumentStore(
            guideline_tool_association_db
        )
        guideline_id = GuidelineId("test_guideline")
        tool_id = ToolId("test_tool")
        creation_utc = datetime.now(timezone.utc)

        created_association = await guideline_tool_association_store.create_association(
            guideline_id=guideline_id,
            tool_id=tool_id,
            creation_utc=creation_utc,
        )

        associations = list(await guideline_tool_association_store.list_associations())
        assert len(associations) == 1
        retrieved_association = list(associations)[0]

        assert retrieved_association.id == created_association.id
        assert retrieved_association.guideline_id == guideline_id
        assert retrieved_association.tool_id == tool_id
        assert retrieved_association.creation_utc == creation_utc


async def test_successful_loading_of_an_empty_json_file(
    context: _TestContext,
    new_file: Path,
) -> None:
    # Create an empty file
    new_file.touch()
    async with JSONFileDocumentDatabase(new_file) as guideline_db:
        guideline_store = GuidelineDocumentStore(guideline_db)
        await guideline_store.create_guideline(
            guideline_set=context.agent_id,
            predicate="Create a guideline just for testing",
            content="Expect it to appear in the guidelines JSON file eventually",
        )

    with open(new_file) as f:
        guidelines_from_json = json.load(f)

    assert len(guidelines_from_json["guidelines"]) == 1

    json_guideline = guidelines_from_json["guidelines"][0]
    assert json_guideline["guideline_set"] == context.agent_id
