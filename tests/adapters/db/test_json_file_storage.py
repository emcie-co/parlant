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

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, AsyncIterator, Optional, cast
from typing_extensions import Self
import tempfile
from lagom import Container
from pytest import fixture, mark, raises

from parlant.core.agents import AgentDocumentStore, AgentId, AgentStore
from parlant.core.common import IdGenerator, Version
from parlant.core.context_variables import (
    ContextVariableDocumentStore,
)
from parlant.core.customers import CustomerDocumentStore, CustomerId
from parlant.core.evaluations import (
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
    GuidelineContent,
    GuidelineDocumentStore,
    GuidelineId,
)
from parlant.adapters.db.json_file import JSONFileDocumentDatabase
from parlant.core.persistence.common import MigrationRequired, ObjectId
from parlant.core.persistence.document_database import (
    BaseDocument,
    Cursor,
    DocumentCollection,
    FindResult,
    Sort,
    SortDirection,
    SortField,
    identity_loader,
)
from parlant.core.persistence.document_database_helper import DocumentStoreMigrationHelper
from parlant.core.sessions import EventKind, EventSource, SessionDocumentStore
from parlant.core.guideline_tool_associations import (
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
        async with AgentDocumentStore(IdGenerator(), agent_db) as agent_store:
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
        async with SessionDocumentStore(session_db) as session_store:
            customer_id = CustomerId("test_customer")
            utc_now = datetime.now(timezone.utc)
            session = await session_store.create_session(
                creation_utc=utc_now,
                customer_id=customer_id,
                agent_id=context.agent_id,
            )

    with open(new_file) as f:
        sessions_from_json = json.load(f)

    assert len(sessions_from_json["sessions"]) == 1
    json_session = sessions_from_json["sessions"][0]
    assert json_session["id"] == session.id
    assert json_session["customer_id"] == customer_id
    assert json_session["agent_id"] == context.agent_id
    assert json_session["consumption_offsets"] == {
        "client": 0,
    }


async def test_event_creation(
    context: _TestContext,
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(context.container[Logger], new_file) as session_db:
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

    with open(new_file) as f:
        events_from_json = json.load(f)

    assert len(events_from_json["events"]) == 1
    json_event = events_from_json["events"][0]
    assert json_event["kind"] == "message"
    assert json_event["data"] == event.data
    assert json_event["source"] == "customer"
    assert datetime.fromisoformat(json_event["creation_utc"]) == event.creation_utc


async def test_guideline_creation_and_loading_data_from_file(
    context: _TestContext,
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(context.container[Logger], new_file) as guideline_db:
        async with GuidelineDocumentStore(IdGenerator(), guideline_db) as guideline_store:
            guideline = await guideline_store.create_guideline(
                condition="Creating a guideline with JSONFileDatabase implementation",
                action="Expecting it to show in the guidelines json file",
            )

    with open(new_file) as f:
        guidelines_from_json = json.load(f)

    assert len(guidelines_from_json["guidelines"]) == 1

    json_guideline = guidelines_from_json["guidelines"][0]

    assert json_guideline["condition"] == guideline.content.condition
    assert json_guideline["action"] == guideline.content.action
    assert datetime.fromisoformat(json_guideline["creation_utc"]) == guideline.creation_utc

    async with JSONFileDocumentDatabase(context.container[Logger], new_file) as guideline_db:
        async with GuidelineDocumentStore(IdGenerator(), guideline_db) as guideline_store:
            second_guideline = await guideline_store.create_guideline(
                condition="Second guideline creation",
                action="Additional test entry in the JSON file",
            )

    with open(new_file) as f:
        guidelines_from_json = json.load(f)

    assert len(guidelines_from_json["guidelines"]) == 2

    second_json_guideline = guidelines_from_json["guidelines"][1]

    assert second_json_guideline["condition"] == second_guideline.content.condition
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
        async with GuidelineDocumentStore(IdGenerator(), guideline_db) as guideline_store:
            await guideline_store.create_guideline(
                condition="Test condition for loading",
                action="Test content for loading guideline",
            )

            loaded_guidelines = await guideline_store.list_guidelines()

        loaded_guideline_list = list(loaded_guidelines)

        assert len(loaded_guideline_list) == 1
        loaded_guideline = loaded_guideline_list[0]
        assert loaded_guideline.content.condition == "Test condition for loading"
        assert loaded_guideline.content.action == "Test content for loading guideline"


async def test_customer_creation(
    context: _TestContext,
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(context.container[Logger], new_file) as customer_db:
        async with CustomerDocumentStore(IdGenerator(), customer_db) as customer_store:
            name = "Jane Doe"
            extra = {"email": "jane.doe@example.com"}
            created_customer = await customer_store.create_customer(
                name=name,
                extra=extra,
            )

    with open(new_file, "r") as file:
        data = json.load(file)

    assert len(data["customers"]) == 1
    json_customer = data["customers"][0]
    assert json_customer["name"] == name
    assert json_customer["extra"] == extra
    assert datetime.fromisoformat(json_customer["creation_utc"]) == created_customer.creation_utc


async def test_customer_retrieval(
    context: _TestContext,
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(context.container[Logger], new_file) as customer_db:
        async with CustomerDocumentStore(IdGenerator(), customer_db) as customer_store:
            name = "John Doe"
            extra = {"email": "john.doe@example.com"}

            created_customer = await customer_store.create_customer(name=name, extra=extra)

            retrieved_customer = await customer_store.read_customer(created_customer.id)

            assert created_customer == retrieved_customer


async def test_context_variable_creation(
    context: _TestContext,
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(context.container[Logger], new_file) as context_variable_db:
        async with ContextVariableDocumentStore(
            IdGenerator(), context_variable_db
        ) as context_variable_store:
            tool_id = ToolId("local", "test_tool")
            variable = await context_variable_store.create_variable(
                name="Sample Variable",
                description="A test variable for persistence.",
                tool_id=tool_id,
                freshness_rules=None,
            )

    with open(new_file) as f:
        variables_from_json = json.load(f)

    assert len(variables_from_json["variables"]) == 1
    json_variable = variables_from_json["variables"][0]

    assert json_variable["name"] == variable.name
    assert json_variable["description"] == variable.description

    assert json_variable["tool_id"]
    assert json_variable["tool_id"] == tool_id.to_string()


async def test_context_variable_value_update_and_retrieval(
    context: _TestContext,
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(context.container[Logger], new_file) as context_variable_db:
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
            )

            await context_variable_store.update_value(
                variable_id=variable.id,
                key=customer_id,
                data={"key": "value"},
            )
            value = await context_variable_store.read_value(
                variable_id=variable.id,
                key=customer_id,
            )

    assert value

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
        async with ContextVariableDocumentStore(
            IdGenerator(), context_variable_db
        ) as context_variable_store:
            tool_id = ToolId("local", "test_tool")
            var1 = await context_variable_store.create_variable(
                name="Variable One",
                description="First test variable",
                tool_id=tool_id,
                freshness_rules=None,
            )

            await context_variable_store.add_variable_tag(
                variable_id=var1.id,
                tag_id=Tag.for_agent_id(context.agent_id),
            )

            var2 = await context_variable_store.create_variable(
                name="Variable Two",
                description="Second test variable",
                tool_id=tool_id,
                freshness_rules=None,
            )

            await context_variable_store.add_variable_tag(
                variable_id=var2.id,
                tag_id=Tag.for_agent_id(context.agent_id),
            )

            variables = list(
                await context_variable_store.list_variables(
                    tags=[Tag.for_agent_id(context.agent_id)]
                )
            )
            assert any(v.id == var1.id for v in variables)
            assert any(v.id == var2.id for v in variables)
            assert len(variables) == 2


async def test_context_variable_deletion(
    context: _TestContext,
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(context.container[Logger], new_file) as context_variable_db:
        async with ContextVariableDocumentStore(
            IdGenerator(), context_variable_db
        ) as context_variable_store:
            tool_id = ToolId("local", "test_tool")
            variable = await context_variable_store.create_variable(
                name="Deletable Variable",
                description="A variable to be deleted.",
                tool_id=tool_id,
                freshness_rules=None,
            )

            await context_variable_store.add_variable_tag(
                variable_id=variable.id,
                tag_id=Tag.for_agent_id(context.agent_id),
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

            assert not any(
                variable.id == v.id
                for v in await context_variable_store.list_variables(
                    tags=[Tag.for_agent_id(context.agent_id)]
                )
            )

            values = await context_variable_store.list_values(
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
        async with GuidelineToolAssociationDocumentStore(
            IdGenerator(), guideline_tool_association_db
        ) as guideline_tool_association_store:
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
        async with GuidelineDocumentStore(IdGenerator(), guideline_db) as guideline_store:
            await guideline_store.create_guideline(
                condition="Create a guideline just for testing",
                action="Expect it to appear in the guidelines JSON file eventually",
            )

    with open(new_file) as f:
        guidelines_from_json = json.load(f)

    assert len(guidelines_from_json["guidelines"]) == 1

    json_guideline = guidelines_from_json["guidelines"][0]

    assert json_guideline["condition"] == "Create a guideline just for testing"
    assert json_guideline["action"] == "Expect it to appear in the guidelines JSON file eventually"


async def test_evaluation_creation(
    context: _TestContext,
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(context.container[Logger], new_file) as evaluation_db:
        async with EvaluationDocumentStore(evaluation_db) as evaluation_store:
            payloads = [
                GuidelinePayload(
                    content=GuidelineContent(
                        condition="Test evaluation creation with invoice",
                        action="Ensure the evaluation with invoice is persisted in the JSON file",
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
        async with EvaluationDocumentStore(evaluation_db) as evaluation_store:
            payloads = [
                GuidelinePayload(
                    content=GuidelineContent(
                        condition="User asks for book recommendations",
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

        with open(new_file) as f:
            evaluations_from_json = json.load(f)

        assert len(evaluations_from_json["evaluations"]) == 1
        json_evaluation = evaluations_from_json["evaluations"][0]

        assert json_evaluation["id"] == evaluation.id

        assert json_evaluation["invoices"][0]["data"] is not None
        assert json_evaluation["invoices"][0]["checksum"] == "initial_checksum"
        assert json_evaluation["invoices"][0]["approved"] is True


class DummyStore:
    VERSION = Version.from_string("2.0.0")

    class DummyDocumentV1(BaseDocument):
        name: str

    class DummyDocumentV2(BaseDocument):
        name: str
        additional_field: str
        creation_utc: str

    def __init__(self, database: JSONFileDocumentDatabase, allow_migration: bool = True):
        self._database = database
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
                creation_utc=str(doc.get("creation_utc", "2023-01-01T00:00:00Z")),
            )
        elif doc["version"] == "2.0.0":
            # Ensure creation_utc field exists for existing documents
            doc_with_creation = dict(doc)
            if "creation_utc" not in doc_with_creation:
                doc_with_creation["creation_utc"] = "2023-01-01T00:00:00Z"
            return cast(DummyStore.DummyDocumentV2, doc_with_creation)
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

    async def list_dummy(
        self,
        sort: Optional[Sort] = None,
        limit: Optional[int] = None,
        cursor: Optional[Cursor] = None,
    ) -> FindResult[DummyDocumentV2]:
        return await self._collection.find({}, sort=sort, limit=limit, cursor=cursor)

    async def create_dummy(self, name: str, additional_field: str = "default") -> DummyDocumentV2:
        doc = self.DummyDocumentV2(
            id=ObjectId(f"dummy_{name}"),
            version=Version.String("2.0.0"),
            name=name,
            additional_field=additional_field,
            creation_utc=datetime.now(timezone.utc).isoformat(),
        )
        await self._collection.insert_one(doc)
        return doc


async def test_document_upgrade_during_loading_of_store(
    container: Container,
    new_file: Path,
) -> None:
    with open(new_file, "w") as f:
        json.dump(
            {
                "metadata": [
                    {
                        "id": "123",
                        "version": "1.0.0",
                    }
                ],
                "dummy_collection": [
                    {
                        "id": "dummy_id",
                        "version": "1.0.0",
                        "name": "Test Document",
                    }
                ],
            },
            f,
        )

    logger = container[Logger]

    async with JSONFileDocumentDatabase(logger, new_file) as db:
        async with DummyStore(db, allow_migration=True) as store:
            result = await store.list_dummy()

            assert result.total_count == 1
            upgraded_doc = result.items[0]
            assert upgraded_doc["version"] == "2.0.0"
            assert upgraded_doc["name"] == "Test Document"
            assert upgraded_doc["additional_field"] == "default_value"


async def test_that_migration_is_not_needed_for_new_store(
    context: _TestContext,
    new_file: Path,
) -> None:
    logger = context.container[Logger]

    async with JSONFileDocumentDatabase(logger, new_file) as db:
        async with DummyStore(db, allow_migration=False):
            meta_collection = await db.get_or_create_collection(
                name="metadata", schema=BaseDocument, document_loader=identity_loader
            )
            meta_document = await meta_collection.find_one({})

            assert meta_document
            assert meta_document["version"] == "2.0.0"


async def test_failed_migration_collection(
    container: Container,
    new_file: Path,
) -> None:
    with open(new_file, "w") as f:
        json.dump(
            {
                "metadata": [
                    {
                        "id": "meta_id",
                        "version": "1.0.0",
                    },
                ],
                "dummy_collection": [
                    {
                        "id": "invalid_dummy_id",
                        "version": "3.0",
                        "name": "Unmigratable Document",
                    }
                ],
            },
            f,
        )

    logger = container[Logger]

    async with JSONFileDocumentDatabase(logger, new_file) as db:
        async with DummyStore(db, allow_migration=True) as store:
            result = await store.list_dummy()

            assert result.total_count == 0

            failed_migrations_collection = await db.get_collection(
                "failed_migrations", BaseDocument, identity_loader
            )
            result_of_failed_migrations = await failed_migrations_collection.find({})

            assert result_of_failed_migrations.total_count == 1
            failed_doc = result_of_failed_migrations.items[0]
            assert failed_doc["id"] == "invalid_dummy_id"
            assert failed_doc["version"] == "3.0"
            assert failed_doc.get("name") == "Unmigratable Document"


async def test_that_version_mismatch_raises_error_when_migration_is_required_but_disabled(
    context: _TestContext,
    new_file: Path,
) -> None:
    logger = context.container[Logger]

    with open(new_file, "w") as f:
        json.dump(
            {
                "metadata": [
                    {"id": "meta_id", "version": "0.0.1"},
                ]
            },
            f,
        )

    async with JSONFileDocumentDatabase(logger, new_file) as db:
        with raises(MigrationRequired) as exc_info:
            async with DummyStore(db, allow_migration=False) as _:
                pass

        assert "Migration required for DummyStore." in str(exc_info.value)


async def test_that_persistence_and_store_version_match_allows_store_to_open_when_migrate_is_disabled(
    context: _TestContext,
    new_file: Path,
) -> None:
    logger = context.container[Logger]

    with open(new_file, "w") as f:
        json.dump(
            {
                "metadata": [
                    {"id": "meta_id", "version": "2.0.0"},
                ]
            },
            f,
        )

    async with JSONFileDocumentDatabase(logger, new_file) as db:
        async with DummyStore(db, allow_migration=False):
            meta_collection = await db.get_or_create_collection(
                name="metadata",
                schema=BaseDocument,
                document_loader=identity_loader,
            )
            meta_document = await meta_collection.find_one({})

            assert meta_document
            assert meta_document["version"] == "2.0.0"


async def test_that_find_with_limit_returns_number_of_documents(
    context: _TestContext,
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(context.container[Logger], new_file) as db:
        async with DummyStore(db) as dummy_store:
            for i in range(5):
                await dummy_store.create_dummy(f"doc{i}", f"field{i}")

            result = await dummy_store.list_dummy(limit=3)

            assert len(result.items) == 3
            assert result.total_count == 5  # Total should include all documents
            assert result.has_more
            assert result.next_cursor is not None


async def test_that_find_with_sort_orders_documents(
    context: _TestContext,
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(context.container[Logger], new_file) as db:
        async with DummyStore(db) as dummy_store:
            # Create dummy documents with different names for sorting
            doc1 = await dummy_store.create_dummy("apple", "field1")
            doc2 = await dummy_store.create_dummy("banana", "field2")
            doc3 = await dummy_store.create_dummy("cherry", "field3")

            # Sort by name ascending
            sort_asc = Sort.by_field("name", SortDirection.ASC)
            result = await dummy_store.list_dummy(sort=sort_asc)

            assert len(result.items) == 3
            assert str(result.items[0]["id"]) == str(doc1["id"])
            assert str(result.items[1]["id"]) == str(doc2["id"])
            assert str(result.items[2]["id"]) == str(doc3["id"])

            # Sort by name descending
            sort_desc = Sort.by_field("name", SortDirection.DESC)
            result = await dummy_store.list_dummy(sort=sort_desc)

            assert len(result.items) == 3
            assert str(result.items[0]["id"]) == str(doc3["id"])
            assert str(result.items[1]["id"]) == str(doc2["id"])
            assert str(result.items[2]["id"]) == str(doc1["id"])


async def test_that_find_with_cursor_returns_documents_after_cursor(
    context: _TestContext,
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(context.container[Logger], new_file) as db:
        async with DummyStore(db) as dummy_store:
            # Create dummy documents with different names for sorting
            doc1 = await dummy_store.create_dummy("first", "field1")
            await dummy_store.create_dummy("second", "field2")
            await dummy_store.create_dummy("third", "field3")

            # Sort by name
            sort_by_name = Sort.by_field("name", SortDirection.ASC)

            # Create cursor from doc1 using its actual creation_utc
            cursor = Cursor(creation_utc=doc1["creation_utc"], id=doc1["id"])

            # Find documents after cursor
            result = await dummy_store.list_dummy(sort=sort_by_name, cursor=cursor)

            assert len(result.items) == 2
            # Should get the documents created after doc1 (second and third)


async def test_that_cursor_based_pagination_works_with_configurable_page_size(
    context: _TestContext,
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(context.container[Logger], new_file) as db:
        async with DummyStore(db) as dummy_store:
            # Create 5 dummy documents
            docs = []
            for i in range(5):
                doc = await dummy_store.create_dummy(f"doc{i:02d}", f"field{i}")
                docs.append(doc)

            sort_by_name = Sort.by_field("name", SortDirection.ASC)

            # First page: get first 2 documents
            result1 = await dummy_store.list_dummy(sort=sort_by_name, limit=2)

            assert len(result1.items) == 2
            assert str(result1.items[0]["id"]) == str(docs[0]["id"])
            assert str(result1.items[1]["id"]) == str(docs[1]["id"])
            assert result1.has_more
            assert result1.next_cursor is not None

            # Second page: use cursor from first page
            result2 = await dummy_store.list_dummy(
                sort=sort_by_name, limit=2, cursor=result1.next_cursor
            )

            assert len(result2.items) == 2
            assert result2.has_more
            assert result2.next_cursor is not None

            # Third page: use cursor from second page
            result3 = await dummy_store.list_dummy(
                sort=sort_by_name, limit=2, cursor=result2.next_cursor
            )

            assert len(result3.items) == 1
            assert not result3.has_more
            assert result3.next_cursor is None


class ComplexDummyStore:
    VERSION = Version.from_string("1.0.0")

    class ComplexDocument(BaseDocument):
        name: str
        priority: int
        score: float
        creation_utc: str

    def __init__(self, database: JSONFileDocumentDatabase):
        self._database = database
        self._collection: DocumentCollection[ComplexDummyStore.ComplexDocument]

    async def __aenter__(self) -> Self:
        self._collection = await self._database.get_or_create_collection(
            name="complex_docs",
            schema=self.ComplexDocument,
            document_loader=self._document_loader,
        )
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None:
        pass

    async def _document_loader(
        self, doc: BaseDocument
    ) -> Optional["ComplexDummyStore.ComplexDocument"]:
        return cast(ComplexDummyStore.ComplexDocument, doc)

    async def create(
        self,
        name: str,
        priority: int,
        score: float,
    ) -> ComplexDocument:
        doc = self.ComplexDocument(
            id=ObjectId(f"complex_{name}_{priority}_{score}"),
            version=Version.String("1.0.0"),
            name=name,
            priority=priority,
            score=score,
            creation_utc=datetime.now(timezone.utc).isoformat(),
        )
        await self._collection.insert_one(doc)
        return doc

    async def list(
        self,
        sort: Optional[Sort] = None,
        limit: Optional[int] = None,
        cursor: Optional[Cursor] = None,
    ) -> FindResult[ComplexDocument]:
        return await self._collection.find({}, sort=sort, limit=limit, cursor=cursor)


async def test_that_complex_sorting_works_with_mixed_data_types(
    context: _TestContext,
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(context.container[Logger], new_file) as db:
        async with ComplexDummyStore(db) as store:
            await store.create("charlie", priority=1, score=9.5)
            await store.create("alice", priority=2, score=8.0)
            await store.create("bob", priority=1, score=9.5)
            await store.create("david", priority=2, score=7.5)
            await store.create("eve", priority=1, score=10.0)

            sort_multi = Sort(
                fields=[
                    SortField(field="priority", direction=SortDirection.ASC),
                    SortField(field="score", direction=SortDirection.DESC),
                    SortField(field="name", direction=SortDirection.ASC),
                ]
            )

            result = await store.list(sort=sort_multi)
            assert len(result.items) == 5

            expected_names = ["eve", "bob", "charlie", "alice", "david"]
            actual_names = [doc["name"] for doc in result.items]
            assert actual_names == expected_names

            sort_multi2 = Sort(
                fields=[
                    SortField(field="score", direction=SortDirection.DESC),
                    SortField(field="priority", direction=SortDirection.ASC),
                    SortField(field="name", direction=SortDirection.DESC),
                ]
            )

            result2 = await store.list(sort=sort_multi2)
            assert len(result2.items) == 5

            expected_names2 = ["eve", "charlie", "bob", "alice", "david"]
            actual_names2 = [doc["name"] for doc in result2.items]
            assert actual_names2 == expected_names2


async def test_that_complex_sorting_handles_equal_values_correctly(
    context: _TestContext,
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(context.container[Logger], new_file) as db:
        async with ComplexDummyStore(db) as store:
            await store.create("zebra", priority=5, score=3.14)
            await store.create("alpha", priority=5, score=3.14)
            await store.create("beta", priority=5, score=3.14)
            await store.create("gamma", priority=5, score=3.14)

            sort_equal = Sort(
                fields=[
                    SortField(field="priority", direction=SortDirection.ASC),
                    SortField(field="score", direction=SortDirection.ASC),
                    SortField(field="name", direction=SortDirection.ASC),
                ]
            )

            result = await store.list(sort=sort_equal)
            assert len(result.items) == 4

            expected_names = ["alpha", "beta", "gamma", "zebra"]
            actual_names = [doc["name"] for doc in result.items]
            assert actual_names == expected_names

            sort_equal_desc = Sort(
                fields=[
                    SortField(field="priority", direction=SortDirection.ASC),
                    SortField(field="score", direction=SortDirection.ASC),
                    SortField(field="name", direction=SortDirection.DESC),
                ]
            )

            result_desc = await store.list(sort=sort_equal_desc)
            assert len(result_desc.items) == 4

            expected_names_desc = ["zebra", "gamma", "beta", "alpha"]
            actual_names_desc = [doc["name"] for doc in result_desc.items]
            assert actual_names_desc == expected_names_desc


async def test_that_multi_field_sorting_with_numeric_edge_cases(
    context: _TestContext,
    new_file: Path,
) -> None:
    async with JSONFileDocumentDatabase(context.container[Logger], new_file) as db:
        async with ComplexDummyStore(db) as store:
            await store.create("negative_int", priority=-5, score=-1.5)
            await store.create("zero_int", priority=0, score=0.0)
            await store.create("positive_int", priority=10, score=2.7)
            await store.create("large_float", priority=1, score=999.99)
            await store.create("small_float", priority=1, score=0.001)

            sort_numeric = Sort(
                fields=[
                    SortField(field="priority", direction=SortDirection.ASC),
                    SortField(field="score", direction=SortDirection.ASC),
                ]
            )

            result = await store.list(sort=sort_numeric)
            assert len(result.items) == 5

            expected_names = [
                "negative_int",
                "zero_int",
                "small_float",
                "large_float",
                "positive_int",
            ]
            actual_names = [doc["name"] for doc in result.items]
            assert actual_names == expected_names

            sort_numeric_desc = Sort(
                fields=[
                    SortField(field="priority", direction=SortDirection.DESC),
                    SortField(field="score", direction=SortDirection.DESC),
                ]
            )

            result_desc = await store.list(sort=sort_numeric_desc)
            assert len(result_desc.items) == 5

            expected_names_desc = [
                "positive_int",
                "large_float",
                "small_float",
                "zero_int",
                "negative_int",
            ]
            actual_names_desc = [doc["name"] for doc in result_desc.items]
            assert actual_names_desc == expected_names_desc
