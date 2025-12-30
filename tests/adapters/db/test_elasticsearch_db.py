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

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Optional, Sequence, cast
from typing_extensions import Self
from lagom import Container
from pytest import fixture, mark, raises

from parlant.core.agents import Agent, AgentDocumentStore, AgentId, AgentStore
from parlant.core.common import IdGenerator, Version
from parlant.core.context_variables import (
    ContextVariable,
    ContextVariableDocumentStore,
    ContextVariableValue,
)
from parlant.core.customers import CustomerDocumentStore, CustomerId
from parlant.core.evaluations import (
    Evaluation,
    EvaluationDocumentStore,
    GuidelinePayload,
    PayloadOperation,
    Invoice,
    InvoiceData,
    InvoiceGuidelineData,
    PayloadDescriptor,
    PayloadKind,
)
from parlant.core.guidelines import (
    Guideline,
    GuidelineContent,
    GuidelineDocumentStore,
    GuidelineId,
)
from parlant.adapters.db.elasticsearch import (
    ElasticsearchDocumentDatabase,
    create_elasticsearch_document_client_from_env,
)
from parlant.core.persistence.common import MigrationRequired
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentCollection,
    identity_loader,
)
from parlant.core.persistence.document_database_helper import DocumentStoreMigrationHelper
from parlant.core.sessions import Event, EventKind, EventSource, Session, SessionDocumentStore
from parlant.core.guideline_tool_associations import (
    GuidelineToolAssociation,
    GuidelineToolAssociationDocumentStore,
)
from parlant.core.loggers import Logger
from parlant.core.tags import Tag
from parlant.core.tools import ToolId

from tests.test_utilities import SyncAwaiter

try:
    from elasticsearch import AsyncElasticsearch
except ImportError:
    AsyncElasticsearch = None  # type: ignore


@fixture
def agent_id(
    container: Container,
    sync_await: SyncAwaiter,
) -> AgentId:
    store = container[AgentStore]
    agent = sync_await(store.create_agent(name="test-agent", max_engine_iterations=2))
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
async def test_index_prefix() -> AsyncIterator[str]:
    # Use unique prefix per test to avoid test isolation issues
    unique_id = uuid.uuid4().hex[:8]
    yield f"test_parlant_doc_{unique_id}"


async def cleanup_test_indices(index_prefix: str) -> None:
    """Helper function to cleanup all test indices.

    Note: Document database doesn't create templates, so only indices need cleanup.
    """
    if AsyncElasticsearch is None:
        return

    es_client = create_elasticsearch_document_client_from_env()
    try:
        all_indices = await es_client.indices.get(index=f"{index_prefix}_*")
        for index_name in all_indices.keys():
            try:
                await es_client.indices.delete(index=index_name)
            except Exception:
                pass
    except Exception:
        pass  # No indices to clean
    finally:
        await es_client.close()


@fixture(scope="function", autouse=True)
async def cleanup_elasticsearch_indices(test_index_prefix: str) -> AsyncIterator[None]:
    """Automatically cleanup test indices before and after each test."""
    if AsyncElasticsearch is None:
        import pytest

        pytest.skip("Elasticsearch not available")

    # Cleanup before test
    await cleanup_test_indices(test_index_prefix)
    yield
    # Cleanup after test
    await cleanup_test_indices(test_index_prefix)


@fixture
async def test_elasticsearch_client() -> AsyncIterator[AsyncElasticsearch]:
    if AsyncElasticsearch is None:
        import pytest

        pytest.skip("Elasticsearch not available")

    client = create_elasticsearch_document_client_from_env()
    yield client
    await client.close()


class ElasticsearchTestDocument(BaseDocument):
    name: str


# ============================================================================
# AGENT TESTS
# ============================================================================


@mark.parametrize(
    ("agent_configuration"),
    [
        ({"name": "Test Agent"}),
        ({"name": "Test Agent", "description": "You are a test agent"}),
    ],
)
async def test_agent_creation(
    context: _TestContext,
    test_elasticsearch_client: AsyncElasticsearch,
    test_index_prefix: str,
    agent_configuration: dict[str, Any],
) -> None:
    """Test creating agents in Elasticsearch."""
    created_agent: Optional[Agent] = None

    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client,
        test_index_prefix,
        context.container[Logger],
    ) as agent_db:
        async with AgentDocumentStore(IdGenerator(), agent_db) as agent_store:
            created_agent = await agent_store.create_agent(**agent_configuration)

            agents = list(await agent_store.list_agents())
            assert len(agents) == 1
            assert agents[0] == created_agent

    assert created_agent

    # Verify persistence
    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client,
        test_index_prefix,
        context.container[Logger],
    ) as agent_db:
        async with AgentDocumentStore(IdGenerator(), agent_db) as agent_store:
            actual_agents = await agent_store.list_agents()
            assert len(actual_agents) == 1

            db_agent = actual_agents[0]
            assert db_agent.id == created_agent.id
            assert db_agent.name == created_agent.name
            assert db_agent.description == created_agent.description
            assert db_agent.creation_utc == created_agent.creation_utc


# ============================================================================
# SESSION TESTS
# ============================================================================


async def test_session_creation(
    context: _TestContext,
    test_elasticsearch_client: AsyncElasticsearch,
    test_index_prefix: str,
) -> None:
    """Test creating sessions in Elasticsearch."""
    session: Optional[Session] = None

    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client, test_index_prefix, context.container[Logger]
    ) as session_db:
        async with SessionDocumentStore(session_db) as session_store:
            customer_id = CustomerId("test_customer")
            utc_now = datetime.now(timezone.utc)
            session = await session_store.create_session(
                creation_utc=utc_now,
                customer_id=customer_id,
                agent_id=context.agent_id,
            )

    assert session

    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client, test_index_prefix, context.container[Logger]
    ) as session_db:
        async with SessionDocumentStore(session_db) as session_store:
            actual_sessions = await session_store.list_sessions()
            assert len(actual_sessions) == 1
            db_session = actual_sessions.items[0]
            assert db_session.id == session.id
            assert db_session.customer_id == session.customer_id
            assert db_session.agent_id == context.agent_id
            assert db_session.consumption_offsets == {
                "client": 0,
            }


async def test_event_creation(
    context: _TestContext,
    test_elasticsearch_client: AsyncElasticsearch,
    test_index_prefix: str,
) -> None:
    """Test creating events in Elasticsearch."""
    session: Optional[Session] = None
    event: Optional[Event] = None

    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client, test_index_prefix, context.container[Logger]
    ) as session_db:
        async with SessionDocumentStore(session_db) as session_store:
            customer_id = CustomerId("test_customer")
            utc_now = datetime.now(timezone.utc)
            session = await session_store.create_session(
                creation_utc=utc_now,
                customer_id=customer_id,
                agent_id=context.agent_id,
            )

            event = await session_store.create_event(
                session_id=session.id,
                source=EventSource.CUSTOMER,
                kind=EventKind.MESSAGE,
                trace_id="<main>",
                data={"message": "Hello, world!"},
                creation_utc=datetime.now(timezone.utc),
            )

    assert session
    assert event

    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client, test_index_prefix, context.container[Logger]
    ) as session_db:
        async with SessionDocumentStore(session_db) as session_store:
            actual_events = await session_store.list_events(session.id)
            assert len(actual_events) == 1
            db_event = actual_events[0]
            assert db_event.id == event.id
            assert db_event.kind == event.kind
            assert db_event.data == event.data
            assert db_event.source == event.source
            assert db_event.creation_utc == event.creation_utc


# ============================================================================
# GUIDELINE TESTS
# ============================================================================


async def test_guideline_creation(
    context: _TestContext,
    test_elasticsearch_client: AsyncElasticsearch,
    test_index_prefix: str,
) -> None:
    """Test creating guidelines in Elasticsearch."""
    guideline: Optional[Guideline] = None

    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client, test_index_prefix, context.container[Logger]
    ) as guideline_db:
        async with GuidelineDocumentStore(IdGenerator(), guideline_db) as guideline_store:
            guideline = await guideline_store.create_guideline(
                condition="Creating a guideline with Elasticsearch implementation",
                action="Expecting it to be stored in the Elasticsearch database",
                tags=[Tag.for_agent_id(context.agent_id)],
            )

    assert guideline

    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client, test_index_prefix, context.container[Logger]
    ) as guideline_db:
        async with GuidelineDocumentStore(IdGenerator(), guideline_db) as guideline_store:
            guidelines = await guideline_store.list_guidelines([Tag.for_agent_id(context.agent_id)])
            guideline_list = list(guidelines)

            assert len(guideline_list) == 1
            db_guideline = guideline_list[0]
            assert db_guideline.id == guideline.id
            assert db_guideline.content.condition == guideline.content.condition
            assert db_guideline.content.action == guideline.content.action
            assert db_guideline.creation_utc == guideline.creation_utc


async def test_multiple_guideline_creation(
    context: _TestContext,
    test_elasticsearch_client: AsyncElasticsearch,
    test_index_prefix: str,
) -> None:
    """Test creating multiple guidelines in Elasticsearch."""
    first_guideline: Optional[Guideline] = None
    second_guideline: Optional[Guideline] = None

    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client, test_index_prefix, context.container[Logger]
    ) as guideline_db:
        async with GuidelineDocumentStore(IdGenerator(), guideline_db) as guideline_store:
            first_guideline = await guideline_store.create_guideline(
                condition="First guideline creation",
                action="Test entry in Elasticsearch",
                tags=[Tag.for_agent_id(context.agent_id)],
            )

            second_guideline = await guideline_store.create_guideline(
                condition="Second guideline creation",
                action="Additional test entry in Elasticsearch",
                tags=[Tag.for_agent_id(context.agent_id)],
            )

    assert first_guideline
    assert second_guideline

    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client, test_index_prefix, context.container[Logger]
    ) as guideline_db:
        async with GuidelineDocumentStore(IdGenerator(), guideline_db) as guideline_store:
            guidelines = list(
                await guideline_store.list_guidelines([Tag.for_agent_id(context.agent_id)])
            )

            assert len(guidelines) == 2

            guideline_ids = [g.id for g in guidelines]
            assert first_guideline.id in guideline_ids
            assert second_guideline.id in guideline_ids

            for guideline in guidelines:
                if guideline.id == first_guideline.id:
                    assert guideline.content.condition == "First guideline creation"
                    assert guideline.content.action == "Test entry in Elasticsearch"
                elif guideline.id == second_guideline.id:
                    assert guideline.content.condition == "Second guideline creation"
                    assert guideline.content.action == "Additional test entry in Elasticsearch"


async def test_guideline_retrieval(
    context: _TestContext,
    test_elasticsearch_client: AsyncElasticsearch,
    test_index_prefix: str,
) -> None:
    """Test retrieving guidelines from Elasticsearch."""
    created_guideline: Optional[Guideline] = None

    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client, test_index_prefix, context.container[Logger]
    ) as guideline_db:
        async with GuidelineDocumentStore(IdGenerator(), guideline_db) as guideline_store:
            created_guideline = await guideline_store.create_guideline(
                condition="Test condition for loading",
                action="Test content for loading guideline",
                tags=[Tag.for_agent_id(context.agent_id)],
            )

            loaded_guidelines = await guideline_store.list_guidelines(
                [Tag.for_agent_id(context.agent_id)]
            )
            loaded_guideline_list = list(loaded_guidelines)

            assert len(loaded_guideline_list) == 1
            loaded_guideline = loaded_guideline_list[0]
            assert loaded_guideline.content.condition == "Test condition for loading"
            assert loaded_guideline.content.action == "Test content for loading guideline"
            assert loaded_guideline.id == created_guideline.id


# ============================================================================
# CUSTOMER TESTS
# ============================================================================


async def test_customer_creation(
    context: _TestContext,
    test_elasticsearch_client: AsyncElasticsearch,
    test_index_prefix: str,
) -> None:
    """Test creating customers in Elasticsearch."""
    created_customer = None

    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client, test_index_prefix, context.container[Logger]
    ) as customer_db:
        async with CustomerDocumentStore(IdGenerator(), customer_db) as customer_store:
            name = "Jane Doe"
            extra = {"email": "jane.doe@example.com"}
            created_customer = await customer_store.create_customer(
                name=name,
                extra=extra,
            )

    assert created_customer
    assert created_customer.name == created_customer.name
    assert created_customer.extra == created_customer.extra

    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client, test_index_prefix, context.container[Logger]
    ) as customer_db:
        async with CustomerDocumentStore(IdGenerator(), customer_db) as customer_store:
            customers = await customer_store.list_customers()

            customer_list = list(customers)
            assert len(customer_list) == 2

            retrieved_customer_guest = customer_list[0]
            assert retrieved_customer_guest
            assert "guest" in retrieved_customer_guest.name.lower()

            retrieved_customer = customer_list[1]
            assert retrieved_customer.id == created_customer.id
            assert retrieved_customer.name == created_customer.name
            assert retrieved_customer.extra == created_customer.extra


async def test_customer_retrieval(
    context: _TestContext,
    test_elasticsearch_client: AsyncElasticsearch,
    test_index_prefix: str,
) -> None:
    """Test retrieving customers from Elasticsearch."""
    created_customer = None

    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client, test_index_prefix, context.container[Logger]
    ) as customer_db:
        async with CustomerDocumentStore(IdGenerator(), customer_db) as customer_store:
            name = "John Doe"
            extra = {"email": "john.doe@example.com"}

            created_customer = await customer_store.create_customer(name=name, extra=extra)

            retrieved_customer = await customer_store.read_customer(created_customer.id)

            assert created_customer == retrieved_customer


# ============================================================================
# CONTEXT VARIABLE TESTS
# ============================================================================


async def test_context_variable_creation(
    context: _TestContext,
    test_elasticsearch_client: AsyncElasticsearch,
    test_index_prefix: str,
) -> None:
    """Test creating context variables in Elasticsearch."""
    variable: Optional[ContextVariable] = None

    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client, test_index_prefix, context.container[Logger]
    ) as context_variable_db:
        async with ContextVariableDocumentStore(
            IdGenerator(), context_variable_db
        ) as context_variable_store:
            tool_id = ToolId("local", "test_tool")
            variable = await context_variable_store.create_variable(
                name="Sample Variable",
                description="A test variable for persistence.",
                tool_id=tool_id,
                freshness_rules=None,
                tags=[Tag.for_agent_id(context.agent_id)],
            )

    assert variable
    assert variable.name == "Sample Variable"
    assert variable.description == "A test variable for persistence."

    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client, test_index_prefix, context.container[Logger]
    ) as context_variable_db:
        async with ContextVariableDocumentStore(
            IdGenerator(), context_variable_db
        ) as context_variable_store:
            variables = list(
                await context_variable_store.list_variables([Tag.for_agent_id(context.agent_id)])
            )

            assert len(variables) == 1
            db_variable = variables[0]
            assert db_variable.id == variable.id
            assert db_variable.name == variable.name
            assert db_variable.description == variable.description
            assert db_variable.tool_id == variable.tool_id


async def test_context_variable_value_update_and_retrieval(
    context: _TestContext,
    test_elasticsearch_client: AsyncElasticsearch,
    test_index_prefix: str,
) -> None:
    """Test updating and retrieving context variable values in Elasticsearch."""
    variable: Optional[ContextVariable] = None
    value: Optional[ContextVariableValue] = None

    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client, test_index_prefix, context.container[Logger]
    ) as context_variable_db:
        async with ContextVariableDocumentStore(
            IdGenerator(), context_variable_db
        ) as context_variable_store:
            tool_id = ToolId("local", "test_tool")
            customer_id = CustomerId("test_customer")
            variable = await context_variable_store.create_variable(
                name="Sample Variable",
                description="A test variable for persistence.",
                tool_id=tool_id,
                freshness_rules=None,
                tags=[Tag.for_agent_id(context.agent_id)],
            )

            test_data = {"key": "value"}
            await context_variable_store.update_value(
                key=customer_id,
                variable_id=variable.id,
                data=test_data,
            )

            value = await context_variable_store.read_value(
                key=customer_id,
                variable_id=variable.id,
            )

            assert value
            assert value.data == test_data


async def test_context_variable_listing(
    context: _TestContext,
    test_elasticsearch_client: AsyncElasticsearch,
    test_index_prefix: str,
) -> None:
    """Test listing context variables in Elasticsearch."""
    var1 = None
    var2 = None

    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client, test_index_prefix, context.container[Logger]
    ) as context_variable_db:
        async with ContextVariableDocumentStore(
            IdGenerator(), context_variable_db
        ) as context_variable_store:
            tool_id = ToolId("local", "test_tool")
            var1 = await context_variable_store.create_variable(
                name="Variable One",
                description="First test variable",
                tool_id=tool_id,
                freshness_rules=None,
                tags=[Tag.for_agent_id(context.agent_id)],
            )

            var2 = await context_variable_store.create_variable(
                name="Variable Two",
                description="Second test variable",
                tool_id=tool_id,
                freshness_rules=None,
                tags=[Tag.for_agent_id(context.agent_id)],
            )

            variables = list(
                await context_variable_store.list_variables([Tag.for_agent_id(context.agent_id)])
            )
            assert len(variables) == 2

            variable_ids = [v.id for v in variables]
            assert var1.id in variable_ids
            assert var2.id in variable_ids


async def test_context_variable_deletion(
    context: _TestContext,
    test_elasticsearch_client: AsyncElasticsearch,
    test_index_prefix: str,
) -> None:
    """Test deleting context variables in Elasticsearch."""
    variable = None

    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client, test_index_prefix, context.container[Logger]
    ) as context_variable_db:
        async with ContextVariableDocumentStore(
            IdGenerator(), context_variable_db
        ) as context_variable_store:
            tool_id = ToolId("local", "test_tool")
            variable = await context_variable_store.create_variable(
                name="Deletable Variable",
                description="A variable to be deleted.",
                tool_id=tool_id,
                freshness_rules=None,
                tags=[Tag.for_agent_id(context.agent_id)],
            )

            for k, d in [("k1", "d1"), ("k2", "d2"), ("k3", "d3")]:
                await context_variable_store.update_value(
                    key=k,
                    variable_id=variable.id,
                    data=d,
                )

            values = await context_variable_store.list_values(
                variable_id=variable.id,
            )

            assert len(values) == 3

            await context_variable_store.delete_variable(
                variable_id=variable.id,
            )

            variables = await context_variable_store.list_variables(
                [Tag.for_agent_id(context.agent_id)]
            )
            assert not any(variable.id == v.id for v in variables)

            values = await context_variable_store.list_values(
                variable_id=variable.id,
            )
            assert len(values) == 0


# ============================================================================
# GUIDELINE TOOL ASSOCIATION TESTS
# ============================================================================


async def test_guideline_tool_association_creation(
    context: _TestContext,
    test_elasticsearch_client: AsyncElasticsearch,
    test_index_prefix: str,
) -> None:
    """Test creating guideline-tool associations in Elasticsearch."""
    association: Optional[GuidelineToolAssociation] = None

    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client,
        test_index_prefix,
        context.container[Logger],
    ) as guideline_tool_association_db:
        async with GuidelineToolAssociationDocumentStore(
            IdGenerator(), guideline_tool_association_db
        ) as guideline_tool_association_store:
            guideline_id = GuidelineId("guideline-789")
            tool_id = ToolId("local", "test_tool")

            association = await guideline_tool_association_store.create_association(
                guideline_id=guideline_id, tool_id=tool_id
            )

    assert association
    assert association.guideline_id == association.guideline_id
    assert association.tool_id == association.tool_id

    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client,
        test_index_prefix,
        context.container[Logger],
    ) as guideline_tool_association_db:
        async with GuidelineToolAssociationDocumentStore(
            IdGenerator(), guideline_tool_association_db
        ) as guideline_tool_association_store:
            associations = list(await guideline_tool_association_store.list_associations())

            assert len(associations) == 1
            stored_association = associations[0]
            assert stored_association.id == association.id
            assert stored_association.guideline_id == association.guideline_id
            assert stored_association.tool_id == association.tool_id


async def test_guideline_tool_association_retrieval(
    context: _TestContext,
    test_elasticsearch_client: AsyncElasticsearch,
    test_index_prefix: str,
) -> None:
    """Test retrieving guideline-tool associations from Elasticsearch."""
    created_association = None

    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client,
        test_index_prefix,
        context.container[Logger],
    ) as guideline_tool_association_db:
        async with GuidelineToolAssociationDocumentStore(
            IdGenerator(), guideline_tool_association_db
        ) as guideline_tool_association_store:
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
            retrieved_association = associations[0]

            assert retrieved_association.id == created_association.id
            assert retrieved_association.guideline_id == guideline_id
            assert retrieved_association.tool_id == tool_id
            assert retrieved_association.creation_utc == creation_utc


# ============================================================================
# EVALUATION TESTS
# ============================================================================


async def test_evaluation_creation(
    context: _TestContext,
    test_elasticsearch_client: AsyncElasticsearch,
    test_index_prefix: str,
) -> None:
    """Test creating evaluations in Elasticsearch."""
    evaluation: Optional[Evaluation] = None

    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client, test_index_prefix, context.container[Logger]
    ) as evaluation_db:
        async with EvaluationDocumentStore(evaluation_db) as evaluation_store:
            payloads = [
                GuidelinePayload(
                    content=GuidelineContent(
                        condition="Test evaluation creation with invoice",
                        action="Ensure the evaluation with invoice is persisted in Elasticsearch",
                    ),
                    tool_ids=[],
                    operation=PayloadOperation.ADD,
                    action_proposition=True,
                    properties_proposition=True,
                    journey_node_proposition=False,
                )
            ]

            evaluation = await evaluation_store.create_evaluation(
                payload_descriptors=[PayloadDescriptor(PayloadKind.GUIDELINE, p) for p in payloads],
            )

    assert evaluation

    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client, test_index_prefix, context.container[Logger]
    ) as evaluation_db:
        async with EvaluationDocumentStore(evaluation_db) as evaluation_store:
            evaluations = await evaluation_store.list_evaluations()
            evaluations_list = list(evaluations)

            assert len(evaluations_list) == 1
            db_evaluation = evaluations_list[0]
            assert db_evaluation.id == evaluation.id
            assert len(db_evaluation.invoices) == 1


async def test_evaluation_update(
    context: _TestContext,
    test_elasticsearch_client: AsyncElasticsearch,
    test_index_prefix: str,
) -> None:
    """Test updating evaluations in Elasticsearch."""
    evaluation = None

    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client, test_index_prefix, context.container[Logger]
    ) as evaluation_db:
        async with EvaluationDocumentStore(evaluation_db) as evaluation_store:
            payloads = [
                GuidelinePayload(
                    content=GuidelineContent(
                        condition="Ask for a book recommendation",
                        action=None,
                    ),
                    tool_ids=[],
                    operation=PayloadOperation.ADD,
                    action_proposition=True,
                    properties_proposition=True,
                    journey_node_proposition=False,
                )
            ]

            evaluation = await evaluation_store.create_evaluation(
                payload_descriptors=[PayloadDescriptor(PayloadKind.GUIDELINE, p) for p in payloads],
            )

            invoice_data: InvoiceData = InvoiceGuidelineData(
                properties_proposition={
                    "continuous": True,
                    "internal_action": "Provide a list of book recommendations",
                },
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

            updated_evaluation = await evaluation_store.read_evaluation(evaluation.id)
            assert updated_evaluation.invoices is not None
            assert len(updated_evaluation.invoices) == 1
            assert updated_evaluation.invoices[0].checksum == "initial_checksum"
            assert updated_evaluation.invoices[0].approved is True


# ============================================================================
# MIGRATION TESTS
# ============================================================================


class DummyStore:
    VERSION = Version.from_string("2.0.0")

    class DummyDocumentV1(BaseDocument):
        name: str

    class DummyDocumentV2(BaseDocument):
        name: str
        additional_field: str

    def __init__(
        self, database: ElasticsearchDocumentDatabase, allow_migration: bool = True
    ) -> None:
        self._database: ElasticsearchDocumentDatabase = database
        self._collection: DocumentCollection[DummyStore.DummyDocumentV2]
        self.allow_migration = allow_migration

    async def _document_loader(self, doc: BaseDocument) -> Optional[DummyDocumentV2]:
        if doc["version"] == "1.0.0":
            doc = cast(DummyStore.DummyDocumentV1, doc)
            return self.DummyDocumentV2(
                id=doc["id"],
                version=Version.String("2.0.0"),
                name=doc["name"],
                additional_field="default_value",
            )
        elif doc["version"] == "2.0.0":
            return cast(DummyStore.DummyDocumentV2, doc)
        return None

    async def __aenter__(self) -> Self:
        async with DocumentStoreMigrationHelper(
            store=self,
            database=self._database,
            allow_migration=self.allow_migration,
        ):
            self._collection = await self._database.get_or_create_collection(
                name="dummy_collection",
                schema=DummyStore.DummyDocumentV2,
                document_loader=self._document_loader,
            )

        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        pass

    async def list_dummy(self) -> Sequence[DummyDocumentV2]:
        result = await self._collection.find({})
        return list(result.items)


async def test_document_upgrade_during_loading_of_store(
    context: _TestContext,
    test_elasticsearch_client: AsyncElasticsearch,
    test_index_prefix: str,
) -> None:
    """Test that documents are upgraded during store loading."""
    logger = context.container[Logger]
    utc_now = datetime.now(timezone.utc).isoformat()

    # First, create indices with proper mappings by opening the database
    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client, test_index_prefix, logger
    ) as db:
        # Create the collection to establish proper index mappings
        _ = await db.get_or_create_collection(
            name="dummy_collection",
            schema=DummyStore.DummyDocumentV2,
            document_loader=identity_loader,
        )
        # Also ensure metadata index exists with proper mappings
        await db.get_or_create_collection(
            name="metadata",
            schema=BaseDocument,
            document_loader=identity_loader,
        )

    # Now delete all documents and insert old version documents for migration testing
    await test_elasticsearch_client.delete_by_query(
        index=f"{test_index_prefix}_dummy_collection",
        body={"query": {"match_all": {}}},
        refresh=True,
    )
    await test_elasticsearch_client.delete_by_query(
        index=f"{test_index_prefix}_metadata",
        body={"query": {"match_all": {}}},
        refresh=True,
    )

    # Insert old version documents directly (indices now have proper mappings)
    await test_elasticsearch_client.index(
        index=f"{test_index_prefix}_metadata",
        id="123",
        document={"id": "123", "version": "1.0.0", "creation_utc": utc_now},
        refresh="wait_for",
    )
    await test_elasticsearch_client.index(
        index=f"{test_index_prefix}_dummy_collection",
        id="dummy_id",
        document={
            "id": "dummy_id",
            "version": "1.0.0",
            "name": "Test Document",
            "creation_utc": utc_now,
        },
        refresh="wait_for",
    )

    # Now open the store and verify migration happens
    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client, test_index_prefix, logger
    ) as db:
        async with DummyStore(db, allow_migration=True) as store:
            documents = await store.list_dummy()

            assert len(documents) == 1
            upgraded_doc = documents[0]
            assert upgraded_doc["version"] == "2.0.0"
            assert upgraded_doc["name"] == "Test Document"
            assert upgraded_doc["additional_field"] == "default_value"


async def test_that_migration_is_not_needed_for_new_store(
    context: _TestContext,
    test_elasticsearch_client: AsyncElasticsearch,
    test_index_prefix: str,
) -> None:
    """Test that no migration is needed for new stores."""
    logger = context.container[Logger]

    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client, test_index_prefix, logger
    ) as db:
        async with DummyStore(db, allow_migration=False):
            meta_collection = await db.get_or_create_collection(
                name="metadata", schema=BaseDocument, document_loader=identity_loader
            )
            meta_document = await meta_collection.find_one({})

            assert meta_document
            assert meta_document["version"] == "2.0.0"


async def test_failed_migration_collection(
    container: Container,
    test_elasticsearch_client: AsyncElasticsearch,
    test_index_prefix: str,
) -> None:
    """Test that failed migrations are stored in a separate collection."""
    # Insert old version metadata and unmigrateable document
    await test_elasticsearch_client.index(
        index=f"{test_index_prefix}_metadata",
        id="meta_id",
        document={"id": "meta_id", "version": "1.0.0"},
        refresh="wait_for",
    )
    await test_elasticsearch_client.index(
        index=f"{test_index_prefix}_dummy_collection",
        id="invalid_dummy_id",
        document={
            "id": "invalid_dummy_id",
            "version": "3.0",
            "name": "Unmigratable Document",
        },
        refresh="wait_for",
    )

    logger = container[Logger]

    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client, test_index_prefix, logger
    ) as db:
        async with DummyStore(db, allow_migration=True) as store:
            documents = await store.list_dummy()

            assert len(documents) == 0

            # Check failed migrations index directly
            failed_migrations_index = f"{test_index_prefix}_dummy_collection_failed_migrations"
            response = await test_elasticsearch_client.search(
                index=failed_migrations_index,
                query={"match_all": {}},
                size=10,
            )

            failed_docs = [hit["_source"] for hit in response["hits"]["hits"]]
            assert len(failed_docs) == 1
            failed_doc = failed_docs[0]
            assert failed_doc["id"] == "invalid_dummy_id"
            assert failed_doc["version"] == "3.0"
            assert failed_doc.get("name") == "Unmigratable Document"


async def test_that_version_mismatch_raises_error_when_migration_is_required_but_disabled(
    context: _TestContext,
    test_elasticsearch_client: AsyncElasticsearch,
    test_index_prefix: str,
) -> None:
    """Test that version mismatch raises error when migration is disabled."""
    # Insert old version metadata
    await test_elasticsearch_client.index(
        index=f"{test_index_prefix}_metadata",
        id="meta_id",
        document={"id": "meta_id", "version": "NotRealVersion"},
        refresh="wait_for",
    )

    logger = context.container[Logger]

    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client, test_index_prefix, logger
    ) as db:
        with raises(MigrationRequired) as exc_info:
            async with DummyStore(db, allow_migration=False) as _:
                pass

        assert "Migration required for DummyStore." in str(exc_info.value)


async def test_that_persistence_and_store_version_match_allows_store_to_open_when_migrate_is_disabled(
    context: _TestContext,
    test_elasticsearch_client: AsyncElasticsearch,
    test_index_prefix: str,
) -> None:
    """Test that matching versions allow store to open without migration."""
    # Insert matching version metadata
    await test_elasticsearch_client.index(
        index=f"{test_index_prefix}_metadata",
        id="meta_id",
        document={"id": "meta_id", "version": "2.0.0"},
        refresh="wait_for",
    )

    logger = context.container[Logger]

    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client, test_index_prefix, logger
    ) as db:
        async with DummyStore(db, allow_migration=False):
            meta_collection = await db.get_or_create_collection(
                name="metadata",
                schema=BaseDocument,
                document_loader=identity_loader,
            )
            meta_document = await meta_collection.find_one({})

            assert meta_document
            assert meta_document["version"] == "2.0.0"


# ============================================================================
# DELETE OPERATIONS TESTS
# ============================================================================


async def test_delete_one_in_collection(
    context: _TestContext,
    test_elasticsearch_client: AsyncElasticsearch,
    test_index_prefix: str,
) -> None:
    """Test deleting a single document from a collection."""
    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client, test_index_prefix, context.container[Logger]
    ) as guideline_db:
        async with GuidelineDocumentStore(IdGenerator(), guideline_db) as guideline_store:
            guideline = await guideline_store.create_guideline(
                condition="Guideline to be deleted",
                action="This guideline will be deleted in the test",
                tags=[Tag.for_agent_id(context.agent_id)],
            )

            await guideline_store.delete_guideline(guideline.id)

            guidelines = list(
                await guideline_store.list_guidelines([Tag.for_agent_id(context.agent_id)])
            )
            assert len(guidelines) == 0


async def test_delete_collection(
    context: _TestContext,
    test_elasticsearch_client: AsyncElasticsearch,
    test_index_prefix: str,
) -> None:
    """Test deleting an entire collection."""
    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client, test_index_prefix, context.container[Logger]
    ) as es_db:
        async with GuidelineDocumentStore(IdGenerator(), es_db) as guideline_store:
            await guideline_store.create_guideline(
                condition="Test collection deletion",
                action="This collection will be deleted",
                tags=[Tag.for_agent_id(context.agent_id)],
            )

        # Verify index exists
        index_name = f"{test_index_prefix}_guidelines"
        exists = await test_elasticsearch_client.indices.exists(index=index_name)
        assert exists

        await es_db.delete_collection("guidelines")

        # Verify index no longer exists
        exists = await test_elasticsearch_client.indices.exists(index=index_name)
        assert not exists


# ============================================================================
# DATABASE INITIALIZATION TEST
# ============================================================================


async def test_database_initialization(
    context: _TestContext,
    test_elasticsearch_client: AsyncElasticsearch,
    test_index_prefix: str,
) -> None:
    """Test database initialization creates indices correctly."""
    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client, test_index_prefix, context.container[Logger]
    ) as guideline_db:
        async with GuidelineDocumentStore(IdGenerator(), guideline_db) as guideline_store:
            await guideline_store.create_guideline(
                condition="Create a guideline for initialization test",
                action="Verify it's stored in Elasticsearch correctly",
                tags=[Tag.for_agent_id(context.agent_id)],
            )

    # Verify index exists
    index_name = f"{test_index_prefix}_guidelines"
    exists = await test_elasticsearch_client.indices.exists(index=index_name)
    assert exists


# ============================================================================
# PAGINATION TESTS
# ============================================================================


class PaginationTestDocument(BaseDocument):
    """Test document for pagination tests."""

    name: str
    creation_utc: str


async def test_that_find_returns_find_result_with_items(
    context: _TestContext,
    test_elasticsearch_client: AsyncElasticsearch,
    test_index_prefix: str,
) -> None:
    """Test that find() returns a FindResult object with items."""
    from parlant.core.persistence.document_database import FindResult, identity_loader_for

    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client, test_index_prefix, context.container[Logger]
    ) as db:
        collection = await db.get_or_create_collection(
            name="pagination_test",
            schema=PaginationTestDocument,
            document_loader=identity_loader_for(PaginationTestDocument),
        )

        # Insert test documents
        for i in range(5):
            await collection.insert_one(
                PaginationTestDocument(
                    id=f"doc_{i}",
                    version="1.0.0",
                    name=f"Document {i}",
                    creation_utc=f"2024-01-0{i + 1}T00:00:00Z",
                )
            )

        # Test find without limit returns all items
        result = await collection.find({})

        assert isinstance(result, FindResult)
        assert len(result.items) == 5
        assert result.has_more is False
        assert result.next_cursor is None


async def test_that_find_with_limit_returns_paginated_results(
    context: _TestContext,
    test_elasticsearch_client: AsyncElasticsearch,
    test_index_prefix: str,
) -> None:
    """Test that find() with limit returns paginated results."""
    from parlant.core.persistence.document_database import identity_loader_for

    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client, test_index_prefix, context.container[Logger]
    ) as db:
        collection = await db.get_or_create_collection(
            name="pagination_limit_test",
            schema=PaginationTestDocument,
            document_loader=identity_loader_for(PaginationTestDocument),
        )

        # Insert test documents
        for i in range(10):
            await collection.insert_one(
                PaginationTestDocument(
                    id=f"doc_{i:02d}",
                    version="1.0.0",
                    name=f"Document {i}",
                    creation_utc=f"2024-01-{i + 1:02d}T00:00:00Z",
                )
            )

        # Test find with limit
        result = await collection.find({}, limit=3)

        assert len(result.items) == 3
        assert result.has_more is True
        assert result.next_cursor is not None


async def test_that_find_with_cursor_returns_next_page(
    context: _TestContext,
    test_elasticsearch_client: AsyncElasticsearch,
    test_index_prefix: str,
) -> None:
    """Test that find() with cursor returns the next page of results."""
    from parlant.core.persistence.document_database import identity_loader_for

    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client, test_index_prefix, context.container[Logger]
    ) as db:
        collection = await db.get_or_create_collection(
            name="pagination_cursor_test",
            schema=PaginationTestDocument,
            document_loader=identity_loader_for(PaginationTestDocument),
        )

        # Insert test documents with distinct creation times
        for i in range(6):
            await collection.insert_one(
                PaginationTestDocument(
                    id=f"doc_{i:02d}",
                    version="1.0.0",
                    name=f"Document {i}",
                    creation_utc=f"2024-01-{i + 1:02d}T00:00:00Z",
                )
            )

        # Get first page
        first_page = await collection.find({}, limit=2)
        assert len(first_page.items) == 2
        assert first_page.has_more is True
        assert first_page.next_cursor is not None

        # Get second page using cursor
        second_page = await collection.find({}, limit=2, cursor=first_page.next_cursor)
        assert len(second_page.items) == 2
        assert second_page.has_more is True

        # Verify no overlap between pages
        first_page_ids = {item["id"] for item in first_page.items}
        second_page_ids = {item["id"] for item in second_page.items}
        assert first_page_ids.isdisjoint(second_page_ids)


async def test_that_find_with_sort_direction_desc_returns_newest_first(
    context: _TestContext,
    test_elasticsearch_client: AsyncElasticsearch,
    test_index_prefix: str,
) -> None:
    """Test that find() with DESC sort returns newest items first."""
    from parlant.core.persistence.common import SortDirection
    from parlant.core.persistence.document_database import identity_loader_for

    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client, test_index_prefix, context.container[Logger]
    ) as db:
        collection = await db.get_or_create_collection(
            name="pagination_sort_test",
            schema=PaginationTestDocument,
            document_loader=identity_loader_for(PaginationTestDocument),
        )

        # Insert test documents with distinct creation times
        for i in range(5):
            await collection.insert_one(
                PaginationTestDocument(
                    id=f"doc_{i:02d}",
                    version="1.0.0",
                    name=f"Document {i}",
                    creation_utc=f"2024-01-{i + 1:02d}T00:00:00Z",
                )
            )

        # Test find with descending sort
        result = await collection.find({}, sort_direction=SortDirection.DESC)

        assert len(result.items) == 5
        # Verify descending order by creation_utc
        creation_dates = [item["creation_utc"] for item in result.items]
        assert creation_dates == sorted(creation_dates, reverse=True)


async def test_that_find_iterating_through_all_pages_returns_all_documents(
    context: _TestContext,
    test_elasticsearch_client: AsyncElasticsearch,
    test_index_prefix: str,
) -> None:
    """Test that iterating through all pages returns all documents."""
    from parlant.core.persistence.document_database import identity_loader_for

    async with ElasticsearchDocumentDatabase(
        test_elasticsearch_client, test_index_prefix, context.container[Logger]
    ) as db:
        collection = await db.get_or_create_collection(
            name="pagination_full_iteration_test",
            schema=PaginationTestDocument,
            document_loader=identity_loader_for(PaginationTestDocument),
        )

        # Insert 7 test documents
        expected_ids = set()
        for i in range(7):
            doc_id = f"doc_{i:02d}"
            expected_ids.add(doc_id)
            await collection.insert_one(
                PaginationTestDocument(
                    id=doc_id,
                    version="1.0.0",
                    name=f"Document {i}",
                    creation_utc=f"2024-01-{i + 1:02d}T00:00:00Z",
                )
            )

        # Iterate through all pages with limit=3
        all_items: list[PaginationTestDocument] = []
        cursor = None
        page_count = 0

        while True:
            result = await collection.find({}, limit=3, cursor=cursor)
            all_items.extend(result.items)
            page_count += 1

            if not result.has_more:
                break
            cursor = result.next_cursor

        # Verify all documents were retrieved
        retrieved_ids = {item["id"] for item in all_items}
        assert retrieved_ids == expected_ids
        assert len(all_items) == 7
        assert page_count == 3  # 3 + 3 + 1 = 7 documents
