from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, AsyncIterator
import tempfile
from lagom import Container
from pytest import fixture, mark

from emcie.server.core.agents import AgentDocumentStore, AgentId, AgentStore
from emcie.server.core.context_variables import (
    ContextVariableDocumentStore,
)
from emcie.server.core.end_users import EndUserDocumentStore, EndUserId
from emcie.server.core.evaluations import (
    EvaluationDocumentStore,
    GuidelinePayload,
    Invoice,
    InvoiceData,
    InvoiceGuidelineData,
    PayloadDescriptor,
    PayloadKind,
)
from emcie.server.core.guidelines import (
    GuidelineContent,
    GuidelineDocumentStore,
    GuidelineId,
)
from emcie.server.adapters.db.json_file import JSONFileDocumentDatabase
from emcie.server.core.sessions import SessionDocumentStore
from emcie.server.core.guideline_tool_associations import (
    GuidelineToolAssociationDocumentStore,
)
from emcie.server.core.logging import Logger
from emcie.server.core.tools import ToolId
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
    context: _TestContext,
    new_file: Path,
    agent_configuration: dict[str, Any],
) -> None:
    async with JSONFileDocumentDatabase(context.container[Logger], new_file) as agent_db:
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
    async with JSONFileDocumentDatabase(context.container[Logger], new_file) as session_db:
        session_store = SessionDocumentStore(session_db)
        end_user_id = EndUserId("test_user")
        utc_now = datetime.now(timezone.utc)
        session = await session_store.create_session(
            creation_utc=utc_now,
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
    async with JSONFileDocumentDatabase(context.container[Logger], new_file) as session_db:
        session_store = SessionDocumentStore(session_db)
        end_user_id = EndUserId("test_user")
        utc_now = datetime.now(timezone.utc)
        session = await session_store.create_session(
            creation_utc=utc_now,
            end_user_id=end_user_id,
            agent_id=context.agent_id,
        )

        event = await session_store.create_event(
            session_id=session.id,
            source="end_user",
            kind="message",
            correlation_id="test_correlation_id",
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
    async with JSONFileDocumentDatabase(context.container[Logger], new_file) as guideline_db:
        guideline_store = GuidelineDocumentStore(guideline_db)
        guideline = await guideline_store.create_guideline(
            guideline_set=context.agent_id,
            predicate="Creating a guideline with JSONFileDatabase implementation",
            action="Expecting it to show in the guidelines json file",
        )

    with open(new_file) as f:
        guidelines_from_json = json.load(f)

    assert len(guidelines_from_json["guidelines"]) == 1

    json_guideline = guidelines_from_json["guidelines"][0]
    assert json_guideline["guideline_set"] == context.agent_id

    assert json_guideline["predicate"] == guideline.content.predicate
    assert json_guideline["action"] == guideline.content.action
    assert datetime.fromisoformat(json_guideline["creation_utc"]) == guideline.creation_utc

    async with JSONFileDocumentDatabase(context.container[Logger], new_file) as guideline_db:
        guideline_store = GuidelineDocumentStore(guideline_db)

        second_guideline = await guideline_store.create_guideline(
            guideline_set=context.agent_id,
            predicate="Second guideline creation",
            action="Additional test entry in the JSON file",
        )

    with open(new_file) as f:
        guidelines_from_json = json.load(f)

    assert len(guidelines_from_json["guidelines"]) == 2

    second_json_guideline = guidelines_from_json["guidelines"][1]
    assert second_json_guideline["guideline_set"] == context.agent_id

    assert second_json_guideline["predicate"] == second_guideline.content.predicate
    assert second_json_guideline["action"] == second_guideline.content.action
    assert (
        datetime.fromisoformat(second_json_guideline["creation_utc"])
        == second_guideline.creation_utc
    )


async def test_guideline_retrieval(
    context: _TestContext,
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(context.container[Logger], new_file) as guideline_db:
        guideline_store = GuidelineDocumentStore(guideline_db)
        await guideline_store.create_guideline(
            guideline_set=context.agent_id,
            predicate="Test predicate for loading",
            action="Test content for loading guideline",
        )

        loaded_guidelines = await guideline_store.list_guidelines(context.agent_id)
        loaded_guideline_list = list(loaded_guidelines)

        assert len(loaded_guideline_list) == 1
        loaded_guideline = loaded_guideline_list[0]
        assert loaded_guideline.content.predicate == "Test predicate for loading"
        assert loaded_guideline.content.action == "Test content for loading guideline"


async def test_end_user_creation(
    context: _TestContext,
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(context.container[Logger], new_file) as end_user_db:
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
    context: _TestContext,
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(context.container[Logger], new_file) as end_user_db:
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
    async with JSONFileDocumentDatabase(context.container[Logger], new_file) as context_variable_db:
        context_variable_store = ContextVariableDocumentStore(context_variable_db)
        tool_id = ToolId("local", "test_tool")
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

    assert json_variable["tool_id"]
    assert json_variable["tool_id"] == tool_id.to_string()


async def test_context_variable_value_update_and_retrieval(
    context: _TestContext,
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(context.container[Logger], new_file) as context_variable_db:
        context_variable_store = ContextVariableDocumentStore(context_variable_db)
        tool_id = ToolId("local", "test_tool")
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
            variable_set=context.agent_id,
            key=end_user_id,
            variable_id=variable.id,
        )

    with open(new_file) as f:
        values_from_json = json.load(f)

    assert len(values_from_json["values"]) == 1
    json_value = values_from_json["values"][0]

    assert json_value["data"] == value.data


async def test_context_variable_listing(
    context: _TestContext,
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(context.container[Logger], new_file) as context_variable_db:
        context_variable_store = ContextVariableDocumentStore(context_variable_db)
        tool_id = ToolId("local", "test_tool")
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
    async with JSONFileDocumentDatabase(context.container[Logger], new_file) as context_variable_db:
        context_variable_store = ContextVariableDocumentStore(context_variable_db)
        tool_id = ToolId("local", "test_tool")
        variable = await context_variable_store.create_variable(
            variable_set=context.agent_id,
            name="Deletable Variable",
            description="A variable to be deleted.",
            tool_id=tool_id,
            freshness_rules=None,
        )

        for k, d in [("k1", "d1"), ("k2", "d2"), ("k3", "d3")]:
            await context_variable_store.update_value(
                variable_set=context.agent_id,
                key=k,
                variable_id=variable.id,
                data=d,
            )

        values = await context_variable_store.list_values(
            variable_set=context.agent_id,
            variable_id=variable.id,
        )

        assert len(values) == 3

        await context_variable_store.delete_variable(
            variable_set=context.agent_id,
            id=variable.id,
        )

        assert not any(
            variable.id == v.id
            for v in await context_variable_store.list_variables(context.agent_id)
        )

        values = await context_variable_store.list_values(
            variable_set=context.agent_id,
            variable_id=variable.id,
        )

        assert len(values) == 0


async def test_guideline_tool_association_creation(
    context: _TestContext,
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(
        context.container[Logger], new_file
    ) as guideline_tool_association_db:
        guideline_tool_association_store = GuidelineToolAssociationDocumentStore(
            guideline_tool_association_db
        )
        guideline_id = GuidelineId("guideline-789")
        tool_id = ToolId("local", "test_tool")

        await guideline_tool_association_store.create_association(
            guideline_id=guideline_id, tool_id=tool_id
        )

    with open(new_file, "r") as f:
        guideline_tool_associations_from_json = json.load(f)

    assert len(guideline_tool_associations_from_json["associations"]) == 1
    json_variable = guideline_tool_associations_from_json["associations"][0]

    assert json_variable["guideline_id"] == guideline_id
    assert json_variable["tool_id"] == tool_id.to_string()


async def test_guideline_tool_association_retrieval(
    context: _TestContext,
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(
        context.container[Logger], new_file
    ) as guideline_tool_association_db:
        guideline_tool_association_store = GuidelineToolAssociationDocumentStore(
            guideline_tool_association_db
        )
        guideline_id = GuidelineId("test_guideline")
        tool_id = ToolId("local", "test_tool")
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
    async with JSONFileDocumentDatabase(context.container[Logger], new_file) as guideline_db:
        guideline_store = GuidelineDocumentStore(guideline_db)
        await guideline_store.create_guideline(
            guideline_set=context.agent_id,
            predicate="Create a guideline just for testing",
            action="Expect it to appear in the guidelines JSON file eventually",
        )

    with open(new_file) as f:
        guidelines_from_json = json.load(f)

    assert len(guidelines_from_json["guidelines"]) == 1

    json_guideline = guidelines_from_json["guidelines"][0]
    assert json_guideline["guideline_set"] == context.agent_id


async def test_evaluation_creation(
    context: _TestContext,
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(context.container[Logger], new_file) as evaluation_db:
        evaluation_store = EvaluationDocumentStore(evaluation_db)

        payloads = [
            GuidelinePayload(
                content=GuidelineContent(
                    predicate="Test evaluation creation with invoice",
                    action="Ensure the evaluation with invoice is persisted in the JSON file",
                ),
                operation="add",
                coherence_check=True,
                connection_proposition=True,
            )
        ]

        evaluation = await evaluation_store.create_evaluation(
            agent_id=context.agent_id,
            payload_descriptors=[PayloadDescriptor(PayloadKind.GUIDELINE, p) for p in payloads],
        )

    with open(new_file) as f:
        evaluations_from_json = json.load(f)

    assert len(evaluations_from_json["evaluations"]) == 1
    json_evaluation = evaluations_from_json["evaluations"][0]

    assert json_evaluation["id"] == evaluation.id

    assert len(json_evaluation["invoices"]) == 1


async def test_evaluation_update(
    context: _TestContext,
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(context.container[Logger], new_file) as evaluation_db:
        evaluation_store = EvaluationDocumentStore(evaluation_db)

        payloads = [
            GuidelinePayload(
                content=GuidelineContent(
                    predicate="Initial evaluation payload with invoice",
                    action="This content will be updated",
                ),
                operation="add",
                coherence_check=True,
                connection_proposition=True,
            )
        ]

        evaluation = await evaluation_store.create_evaluation(
            agent_id=context.agent_id,
            payload_descriptors=[PayloadDescriptor(PayloadKind.GUIDELINE, p) for p in payloads],
        )

        invoice_data: InvoiceData = InvoiceGuidelineData(
            coherence_checks=[],
            connection_propositions=None,
        )

        invoice = Invoice(
            kind=PayloadKind.GUIDELINE,
            payload=payloads[0],
            state_version="123",
            checksum="initial_checksum",
            approved=True,
            data=invoice_data,
            error=None,
        )

        await evaluation_store.update_evaluation(
            evaluation_id=evaluation.id, params={"invoices": [invoice]}
        )

    with open(new_file) as f:
        evaluations_from_json = json.load(f)

    assert len(evaluations_from_json["evaluations"]) == 1
    json_evaluation = evaluations_from_json["evaluations"][0]

    assert json_evaluation["id"] == evaluation.id

    assert json_evaluation["invoices"][0]["data"] is not None
    assert json_evaluation["invoices"][0]["checksum"] == "initial_checksum"
    assert json_evaluation["invoices"][0]["approved"] is True
