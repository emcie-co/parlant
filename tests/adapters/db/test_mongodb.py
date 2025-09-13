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
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from typing import Any, AsyncIterator, Optional, Sequence, cast
from pymongo import AsyncMongoClient
import pytest
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
from parlant.adapters.db.mongo_db import MongoDocumentDatabase
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
async def test_database_name() -> AsyncIterator[str]:
    yield "test_db"


async def pymongo_tasks_still_running() -> None:
    while any("pymongo" in str(t) for t in asyncio.all_tasks()):
        print(str(t) for t in asyncio.all_tasks())
        await asyncio.sleep(1)


@fixture
async def test_mongo_client() -> AsyncIterator[AsyncMongoClient[Any]]:
    test_mongo_server = os.environ.get("TEST_MONGO_SERVER")
    if test_mongo_server:
        client = AsyncMongoClient[Any](test_mongo_server)
        yield client
        await client.close()
        await pymongo_tasks_still_running()
    else:
        print("could not find `TEST_MONGO_SERVER` in environment, skipping mongo tests...")
        raise pytest.skip()


class MongoTestDocument(BaseDocument):
    name: str


@mark.parametrize(
    ("agent_configuration"),
    [
        ({"name": "Test Agent"}),
        ({"name": "Test Agent", "description": "You are a test agent"}),
    ],
)
async def test_agent_creation(
    context: _TestContext,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
    agent_configuration: dict[str, Any],
) -> None:
    await test_mongo_client.drop_database(test_database_name)

    created_agent: Optional[Agent] = None

    async with MongoDocumentDatabase(
        test_mongo_client,
        test_database_name,
        context.container[Logger],
    ) as agent_db:
        async with AgentDocumentStore(IdGenerator(), agent_db) as agent_store:
            created_agent = await agent_store.create_agent(**agent_configuration)

            agents = list(await agent_store.list_agents())
            assert len(agents) == 1
            assert agents[0] == created_agent

    assert created_agent

    async with MongoDocumentDatabase(
        test_mongo_client,
        test_database_name,
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


async def test_session_creation(
    context: _TestContext,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    await test_mongo_client.drop_database(test_database_name)

    session: Optional[Session] = None

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, context.container[Logger]
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

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, context.container[Logger]
    ) as session_db:
        async with SessionDocumentStore(session_db) as session_store:
            actual_sessions = await session_store.list_sessions()
            assert len(actual_sessions) == 1
            db_session = actual_sessions[0]
            assert db_session.id == session.id
            assert db_session.customer_id == session.customer_id
            assert db_session.agent_id == context.agent_id
            assert db_session.consumption_offsets == {
                "client": 0,
            }


async def test_event_creation(
    context: _TestContext,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    await test_mongo_client.drop_database(test_database_name)

    session: Optional[Session] = None
    event: Optional[Event] = None

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, context.container[Logger]
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

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, context.container[Logger]
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


async def test_guideline_creation(
    context: _TestContext,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    await test_mongo_client.drop_database(test_database_name)

    guideline: Optional[Guideline] = None

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, context.container[Logger]
    ) as guideline_db:
        async with GuidelineDocumentStore(IdGenerator(), guideline_db) as guideline_store:
            guideline = await guideline_store.create_guideline(
                condition="Creating a guideline with MongoDB implementation",
                action="Expecting it to be stored in the MongoDB database",
                tags=[Tag.for_agent_id(context.agent_id)],
            )

    assert guideline

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, context.container[Logger]
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
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    await test_mongo_client.drop_database(test_database_name)

    first_guideline: Optional[Guideline] = None
    second_guideline: Optional[Guideline] = None

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, context.container[Logger]
    ) as guideline_db:
        async with GuidelineDocumentStore(IdGenerator(), guideline_db) as guideline_store:
            first_guideline = await guideline_store.create_guideline(
                condition="First guideline creation",
                action="Test entry in MongoDB",
                tags=[Tag.for_agent_id(context.agent_id)],
            )

            second_guideline = await guideline_store.create_guideline(
                condition="Second guideline creation",
                action="Additional test entry in MongoDB",
                tags=[Tag.for_agent_id(context.agent_id)],
            )

    assert first_guideline
    assert second_guideline

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, context.container[Logger]
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
                    assert guideline.content.action == "Test entry in MongoDB"
                elif guideline.id == second_guideline.id:
                    assert guideline.content.condition == "Second guideline creation"
                    assert guideline.content.action == "Additional test entry in MongoDB"


async def test_guideline_retrieval(
    context: _TestContext,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    await test_mongo_client.drop_database(test_database_name)

    created_guideline: Optional[Guideline] = None

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, context.container[Logger]
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


async def test_customer_creation(
    context: _TestContext,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    await test_mongo_client.drop_database(test_database_name)

    created_customer = None

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, context.container[Logger]
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

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, context.container[Logger]
    ) as customer_db:
        async with CustomerDocumentStore(IdGenerator(), customer_db) as customer_store:
            customers = await customer_store.list_customers()

            customer_list = list(customers)
            assert len(customer_list) == 2

            retrieved_customer_guest = customer_list[0]
            assert retrieved_customer_guest
            assert "guest" in retrieved_customer_guest.name

            retrieved_customer = customer_list[1]
            assert retrieved_customer.id == created_customer.id
            assert retrieved_customer.name == created_customer.name
            assert retrieved_customer.extra == created_customer.extra


async def test_customer_retrieval(
    context: _TestContext,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    await test_mongo_client.drop_database(test_database_name)

    created_customer = None

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, context.container[Logger]
    ) as customer_db:
        async with CustomerDocumentStore(IdGenerator(), customer_db) as customer_store:
            name = "John Doe"
            extra = {"email": "john.doe@example.com"}

            created_customer = await customer_store.create_customer(name=name, extra=extra)

            retrieved_customer = await customer_store.read_customer(created_customer.id)

            assert created_customer == retrieved_customer


async def test_context_variable_creation(
    context: _TestContext,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    await test_mongo_client.drop_database(test_database_name)

    variable: Optional[ContextVariable] = None

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, context.container[Logger]
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

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, context.container[Logger]
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
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    await test_mongo_client.drop_database(test_database_name)

    variable: Optional[ContextVariable] = None
    value: Optional[ContextVariableValue] = None

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, context.container[Logger]
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
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    await test_mongo_client.drop_database(test_database_name)

    var1 = None
    var2 = None

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, context.container[Logger]
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
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    await test_mongo_client.drop_database(test_database_name)

    variable = None

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, context.container[Logger]
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


async def test_guideline_tool_association_creation(
    context: _TestContext,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    await test_mongo_client.drop_database(test_database_name)

    association: Optional[GuidelineToolAssociation] = None

    async with MongoDocumentDatabase(
        test_mongo_client,
        test_database_name,
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

    async with MongoDocumentDatabase(
        test_mongo_client,
        test_database_name,
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
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    await test_mongo_client.drop_database(test_database_name)

    created_association = None

    async with MongoDocumentDatabase(
        test_mongo_client,
        test_database_name,
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


async def test_database_initialization(
    context: _TestContext,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    await test_mongo_client.drop_database(test_database_name)

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, context.container[Logger]
    ) as guideline_db:
        async with GuidelineDocumentStore(IdGenerator(), guideline_db) as guideline_store:
            await guideline_store.create_guideline(
                condition="Create a guideline for initialization test",
                action="Verify it's stored in MongoDB correctly",
                tags=[Tag.for_agent_id(context.agent_id)],
            )

    collections = await test_mongo_client[test_database_name].list_collection_names()
    assert "guidelines" in collections


async def test_evaluation_creation(
    context: _TestContext,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    await test_mongo_client.drop_database(test_database_name)

    evaluation: Optional[Evaluation] = None

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, context.container[Logger]
    ) as evaluation_db:
        async with EvaluationDocumentStore(evaluation_db) as evaluation_store:
            payloads = [
                GuidelinePayload(
                    content=GuidelineContent(
                        condition="Test evaluation creation with invoice",
                        action="Ensure the evaluation with invoice is persisted in MongoDB",
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

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, context.container[Logger]
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
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    await test_mongo_client.drop_database(test_database_name)

    evaluation = None

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, context.container[Logger]
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


class DummyStore:
    VERSION = Version.from_string("2.0.0")

    class DummyDocumentV1(BaseDocument):
        name: str

    class DummyDocumentV2(BaseDocument):
        name: str
        additional_field: str

    def __init__(self, database: MongoDocumentDatabase, allow_migration: bool = True):
        self._database: MongoDocumentDatabase = database
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
        return await self._collection.find({})


async def test_document_upgrade_during_loading_of_store(
    context: _TestContext,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    await test_mongo_client.drop_database(test_database_name)

    adb = test_mongo_client[test_database_name]
    await adb.metadata.insert_one({"id": "123", "version": "1.0.0"})
    await adb.dummy_collection.insert_one(
        {"id": "dummy_id", "version": "1.0.0", "name": "Test Document"}
    )

    logger = context.container[Logger]

    async with MongoDocumentDatabase(test_mongo_client, "test_db", logger) as db:
        async with DummyStore(db, allow_migration=True) as store:
            documents = await store.list_dummy()

            assert len(documents) == 1
            upgraded_doc = documents[0]
            assert upgraded_doc["version"] == "2.0.0"
            assert upgraded_doc["name"] == "Test Document"
            assert upgraded_doc["additional_field"] == "default_value"


async def test_that_migration_is_not_needed_for_new_store(
    context: _TestContext,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    await test_mongo_client.drop_database(test_database_name)

    logger = context.container[Logger]

    async with MongoDocumentDatabase(test_mongo_client, "test_db", logger) as db:
        async with DummyStore(db, allow_migration=False):
            meta_collection = await db.get_or_create_collection(
                name="metadata", schema=BaseDocument, document_loader=identity_loader
            )
            meta_document = await meta_collection.find_one({})

            assert meta_document
            assert meta_document["version"] == "2.0.0"


async def test_failed_migration_collection(
    container: Container,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    await test_mongo_client.drop_database(test_database_name)

    adb = test_mongo_client[test_database_name]
    await adb.metadata.insert_one({"id": "meta_id", "version": "1.0.0"})
    await adb.dummy_collection.insert_one(
        {
            "id": "invalid_dummy_id",
            "version": "3.0",
            "name": "Unmigratable Document",
        }
    )

    logger = container[Logger]

    async with MongoDocumentDatabase(test_mongo_client, "test_db", logger) as db:
        async with DummyStore(db, allow_migration=True) as store:
            documents = await store.list_dummy()

            assert len(documents) == 0

            failed_migrations_collection = await db.get_collection(
                "test_db_dummy_collection_failed_migrations",
                BaseDocument,
                identity_loader,
            )
            failed_docs = await failed_migrations_collection.find({})

            assert len(failed_docs) == 1
            failed_doc = failed_docs[0]
            assert failed_doc["id"] == "invalid_dummy_id"
            assert failed_doc["version"] == "3.0"
            assert failed_doc.get("name") == "Unmigratable Document"


async def test_that_version_mismatch_raises_error_when_migration_is_required_but_disabled(
    context: _TestContext,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    await test_mongo_client.drop_database(test_database_name)

    adb = test_mongo_client[test_database_name]
    await adb.metadata.insert_one({"id": "meta_id", "version": "NotRealVersion"})

    logger = context.container[Logger]

    async with MongoDocumentDatabase(test_mongo_client, "test_db", logger) as db:
        with raises(MigrationRequired) as exc_info:
            async with DummyStore(db, allow_migration=False) as _:
                pass

        assert "Migration required for DummyStore." in str(exc_info.value)


async def test_that_persistence_and_store_version_match_allows_store_to_open_when_migrate_is_disabled(
    context: _TestContext,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    await test_mongo_client.drop_database(test_database_name)

    adb = test_mongo_client[test_database_name]
    await adb.metadata.insert_one({"id": "meta_id", "version": "2.0.0"})

    logger = context.container[Logger]

    async with MongoDocumentDatabase(test_mongo_client, "test_db", logger) as db:
        async with DummyStore(db, allow_migration=False):
            meta_collection = await db.get_or_create_collection(
                name="metadata",
                schema=BaseDocument,
                document_loader=identity_loader,
            )
            meta_document = await meta_collection.find_one({})

            assert meta_document
            assert meta_document["version"] == "2.0.0"


async def test_delete_one_in_collection(
    context: _TestContext,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    await test_mongo_client.drop_database(test_database_name)

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, context.container[Logger]
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
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    await test_mongo_client.drop_database(test_database_name)

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, context.container[Logger]
    ) as mongo_db:
        async with GuidelineDocumentStore(IdGenerator(), mongo_db) as guideline_store:
            await guideline_store.create_guideline(
                condition="Test collection deletion",
                action="This collection will be deleted",
                tags=[Tag.for_agent_id(context.agent_id)],
            )

        collections = await test_mongo_client[test_database_name].list_collection_names()
        assert "guidelines" in collections

        await mongo_db.delete_collection("guidelines")

        collections = await test_mongo_client[test_database_name].list_collection_names()
        assert "guidelines" not in collections
