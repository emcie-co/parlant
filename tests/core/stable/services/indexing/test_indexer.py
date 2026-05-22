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

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping

import xxhash
from lagom import Container

from parlant.core.agents import AgentStore, CompositionMode, MessageOutputMode
from parlant.core.canned_responses import CannedResponseField, CannedResponseStore
from parlant.core.common import Criticality
from parlant.core.context_variables import ContextVariableStore
from parlant.core.glossary import GlossaryStore
from parlant.core.guidelines import GuidelineStore
from parlant.core.journeys import JourneyStore
from parlant.core.relationships import (
    RelationshipEntity,
    RelationshipEntityKind,
    RelationshipKind,
    RelationshipStore,
)
from parlant.core.services.indexing.common import ProgressReport
from parlant.core.services.indexing.indexer import IndexRequest, Indexer
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.tags import TagId
from parlant.core.tools import LocalToolService, ToolId, ToolOverlap, ToolParameterOptions


class _CapturingIndexer(Indexer):
    def __init__(
        self,
        agent_store: AgentStore,
        guideline_store: GuidelineStore,
        journey_store: JourneyStore,
        relationship_store: RelationshipStore,
        glossary_store: GlossaryStore,
        context_variable_store: ContextVariableStore,
        canned_response_store: CannedResponseStore,
        service_registry: ServiceRegistry,
    ) -> None:
        super().__init__(
            agent_store=agent_store,
            guideline_store=guideline_store,
            journey_store=journey_store,
            relationship_store=relationship_store,
            glossary_store=glossary_store,
            context_variable_store=context_variable_store,
            canned_response_store=canned_response_store,
            service_registry=service_registry,
        )
        self.captured_payload: Mapping[str, Mapping[str, IndexRequest]] = {}

    async def index(
        self,
        payload: Mapping[str, Mapping[str, IndexRequest]],
        progress_report: ProgressReport,
    ) -> None:
        self.captured_payload = payload


def _make_indexer(container: Container) -> _CapturingIndexer:
    return _CapturingIndexer(
        agent_store=container[AgentStore],
        guideline_store=container[GuidelineStore],
        journey_store=container[JourneyStore],
        relationship_store=container[RelationshipStore],
        glossary_store=container[GlossaryStore],
        context_variable_store=container[ContextVariableStore],
        canned_response_store=container[CannedResponseStore],
        service_registry=container[ServiceRegistry],
    )


async def test_that_indexer_run_walks_all_stores_and_yields_one_request_per_entity(
    container: Container,
) -> None:
    agent = await container[AgentStore].create_agent(name="Agent A", description="d")
    guideline = await container[GuidelineStore].create_guideline(
        condition="customer says hi",
        action="reply",
    )
    journey = await container[JourneyStore].create_journey(
        title="Onboarding",
        description="Walks the customer through onboarding",
        triggers=[],
    )
    term = await container[GlossaryStore].create_term(
        name="SLA",
        description="Service-level agreement",
    )
    variable = await container[ContextVariableStore].create_variable(
        name="account_tier",
        description="Customer's account tier",
    )
    canned = await container[CannedResponseStore].create_canned_response(
        value="Hello!",
    )
    relationship = await container[RelationshipStore].create_relationship(
        source=RelationshipEntity(id=guideline.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=guideline.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.ENTAILMENT,
    )

    indexer = _make_indexer(container)
    await indexer.run()

    payload = indexer.captured_payload

    assert agent.id in payload["agents"]
    assert guideline.id in payload["guidelines"]
    assert journey.id in payload["journeys"]
    assert term.id in payload["glossary"]
    assert variable.id in payload["context_variables"]
    assert canned.id in payload["canned_responses"]
    assert relationship.id in payload["relationships"]


async def test_that_run_invokes_progress_callback_and_reaches_one_hundred_percent(
    container: Container,
) -> None:
    await container[AgentStore].create_agent(name="Agent", description=None)

    class _CountingIndexer(Indexer):
        async def index(
            self,
            payload: Mapping[str, Mapping[str, IndexRequest]],
            progress_report: ProgressReport,
        ) -> None:
            for bucket in payload.values():
                for _ in bucket:
                    await progress_report.increment()

    indexer = _CountingIndexer(
        agent_store=container[AgentStore],
        guideline_store=container[GuidelineStore],
        journey_store=container[JourneyStore],
        relationship_store=container[RelationshipStore],
        glossary_store=container[GlossaryStore],
        context_variable_store=container[ContextVariableStore],
        canned_response_store=container[CannedResponseStore],
        service_registry=container[ServiceRegistry],
    )

    captured: list[float] = []

    async def callback(pct: float) -> None:
        captured.append(pct)

    await indexer.run(progress_callback=callback)

    assert captured, "progress callback was never invoked"
    assert captured[-1] == 100.0


async def test_that_null_indexer_advances_progress_to_full(
    container: Container,
) -> None:
    from parlant.core.services.indexing.indexer import NullIndexer

    await container[AgentStore].create_agent(name="Agent", description=None)
    await container[GuidelineStore].create_guideline(condition="c", action="a")

    captured: list[float] = []

    async def callback(pct: float) -> None:
        captured.append(pct)

    indexer = NullIndexer(
        agent_store=container[AgentStore],
        guideline_store=container[GuidelineStore],
        journey_store=container[JourneyStore],
        relationship_store=container[RelationshipStore],
        glossary_store=container[GlossaryStore],
        context_variable_store=container[ContextVariableStore],
        canned_response_store=container[CannedResponseStore],
        service_registry=container[ServiceRegistry],
    )

    await indexer.run(progress_callback=callback)

    assert captured[-1] == 100.0


async def test_that_payload_is_keyed_by_category_then_entity_id(
    container: Container,
) -> None:
    g1 = await container[GuidelineStore].create_guideline(condition="c1", action="a1")
    g2 = await container[GuidelineStore].create_guideline(condition="c2", action="a2")

    indexer = _make_indexer(container)
    await indexer.run()

    payload = indexer.captured_payload

    assert "guidelines" in payload
    assert set(payload["guidelines"].keys()) >= {g1.id, g2.id}
    assert payload["guidelines"][g1.id].id == g1.id
    assert payload["guidelines"][g2.id].id == g2.id


async def test_that_existing_checksum_is_preferred_over_computed_one(
    container: Container,
) -> None:
    @dataclass(frozen=True)
    class _EntityWithoutChecksum:
        id: str
        creation_utc: datetime

    @dataclass(frozen=True)
    class _EntityWithChecksum:
        id: str
        creation_utc: datetime
        checksum: str

    indexer = _make_indexer(container)
    moment = datetime.now(timezone.utc)

    _, computed = indexer._serialize_with_checksum(
        _EntityWithoutChecksum(id="x", creation_utc=moment)
    )
    assert computed != "precomputed-checksum"

    request = indexer._build_request(
        type="x",
        entity=_EntityWithChecksum(
            id="x",
            creation_utc=moment,
            checksum="precomputed-checksum",
        ),
    )
    assert request.checksum == "precomputed-checksum"


async def test_that_serialize_with_checksum_uses_xxh3_and_is_stable_for_the_same_input(
    container: Container,
) -> None:
    indexer = _make_indexer(container)

    @dataclass(frozen=True)
    class _Entity:
        id: str
        name: str
        creation_utc: datetime

    moment = datetime(2026, 1, 1, tzinfo=timezone.utc)
    e1 = _Entity(id="a", name="hello", creation_utc=moment)
    e2 = _Entity(id="a", name="hello", creation_utc=moment)
    e3 = _Entity(id="a", name="world", creation_utc=moment)

    _, c1 = indexer._serialize_with_checksum(e1)
    _, c2 = indexer._serialize_with_checksum(e2)
    _, c3 = indexer._serialize_with_checksum(e3)

    assert c1 == c2
    assert c1 != c3

    expected_length = len(xxhash.xxh3_64_hexdigest(b"x"))
    assert len(c1) == expected_length


async def test_that_last_modified_falls_back_to_creation_utc_then_now_when_neither_last_modification_nor_created_present(
    container: Container,
) -> None:
    indexer = _make_indexer(container)

    @dataclass(frozen=True)
    class _Modified:
        id: str
        last_modification_utc: datetime
        creation_utc: datetime

    @dataclass(frozen=True)
    class _CreatedOnly:
        id: str
        creation_utc: datetime

    @dataclass(frozen=True)
    class _Bare:
        id: str

    modified_at = datetime(2026, 3, 1, tzinfo=timezone.utc)
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)

    r_modified = indexer._build_request(
        type="x",
        entity=_Modified(id="a", last_modification_utc=modified_at, creation_utc=created_at),
    )
    assert r_modified.last_modification_utc == modified_at

    r_created = indexer._build_request(
        type="x",
        entity=_CreatedOnly(id="b", creation_utc=created_at),
    )
    assert r_created.last_modification_utc == created_at

    before = datetime.now(timezone.utc)
    r_bare = indexer._build_request(type="x", entity=_Bare(id="c"))
    after = datetime.now(timezone.utc)
    assert before <= r_bare.last_modification_utc <= after


async def test_that_tools_are_enumerated_one_request_per_service_tool_pair(
    container: Container,
) -> None:
    local = container[LocalToolService]

    t1 = await local.create_tool(
        name="tool_one",
        module_path="some.module",
        description="first",
        parameters={},
        required=[],
    )
    t2 = await local.create_tool(
        name="tool_two",
        module_path="some.module",
        description="second",
        parameters={},
        required=[],
    )

    indexer = _make_indexer(container)
    await indexer.run()

    payload = indexer.captured_payload
    assert "tools" in payload

    expected_keys = {f"local:{t1.name}", f"local:{t2.name}"}
    assert expected_keys.issubset(set(payload["tools"].keys()))

    one = payload["tools"][f"local:{t1.name}"]
    assert one.id == f"local:{t1.name}"


async def test_that_agent_is_serialized_with_all_fields_including_composition_and_message_modes(
    container: Container,
) -> None:
    agent = await container[AgentStore].create_agent(
        name="Agent A",
        description="An agent",
        max_engine_iterations=5,
        composition_mode=CompositionMode.CANNED_STRICT,
        message_output_mode=MessageOutputMode.STREAM,
    )

    indexer = _make_indexer(container)
    await indexer.run()

    data = indexer.captured_payload["agents"][agent.id].data
    assert isinstance(data, dict)
    assert data["id"] == agent.id
    assert data["name"] == "Agent A"
    assert data["description"] == "An agent"
    assert data["max_engine_iterations"] == 5
    assert data["composition_mode"] == "canned_strict"
    assert data["message_output_mode"] == "stream"
    assert data["tags"] == []


async def test_that_actionable_guideline_is_serialized_with_action_metadata_and_criticality(
    container: Container,
) -> None:
    guideline = await container[GuidelineStore].create_guideline(
        condition="customer says hi",
        action="reply with a greeting",
        criticality=Criticality.HIGH,
        metadata={"foo": "bar"},
    )

    indexer = _make_indexer(container)
    await indexer.run()

    data = indexer.captured_payload["guidelines"][guideline.id].data
    assert isinstance(data, dict)
    assert data["id"] == guideline.id
    assert isinstance(data["content"], dict)
    assert data["content"]["condition"] == "customer says hi"
    assert data["content"]["action"] == "reply with a greeting"
    assert data["enabled"] is True
    assert data["criticality"] == "high"
    assert data["metadata"] == {"foo": "bar"}


async def test_that_observational_guideline_is_serialized_with_null_action(
    container: Container,
) -> None:
    guideline = await container[GuidelineStore].create_guideline(
        condition="customer is silent",
        action=None,
    )

    indexer = _make_indexer(container)
    await indexer.run()

    data = indexer.captured_payload["guidelines"][guideline.id].data
    assert isinstance(data, dict)
    assert isinstance(data["content"], dict)
    assert data["content"]["condition"] == "customer is silent"
    assert data["content"]["action"] is None


async def test_that_journey_is_serialized_with_triggers_and_root_node_id(
    container: Container,
) -> None:
    trigger_guideline = await container[GuidelineStore].create_guideline(
        condition="customer asks to onboard",
        action=None,
    )
    journey = await container[JourneyStore].create_journey(
        title="Onboarding",
        description="Walks the customer through onboarding",
        triggers=[trigger_guideline.id],
    )

    indexer = _make_indexer(container)
    await indexer.run()

    data = indexer.captured_payload["journeys"][journey.id].data
    assert isinstance(data, dict)
    assert data["id"] == journey.id
    assert data["title"] == "Onboarding"
    assert data["description"] == "Walks the customer through onboarding"
    assert data["triggers"] == [trigger_guideline.id]
    assert data["root_id"] == journey.root_id


async def test_that_term_is_serialized_with_synonyms_and_tags(
    container: Container,
) -> None:
    term = await container[GlossaryStore].create_term(
        name="SLA",
        description="Service-level agreement",
        synonyms=["service level agreement", "service-level"],
        tags=[TagId("compliance"), TagId("legal")],
    )

    indexer = _make_indexer(container)
    await indexer.run()

    data = indexer.captured_payload["glossary"][term.id].data
    assert isinstance(data, dict)
    assert data["name"] == "SLA"
    assert data["description"] == "Service-level agreement"
    assert data["synonyms"] == ["service level agreement", "service-level"]
    assert isinstance(data["tags"], list)
    assert set(data["tags"]) == {"compliance", "legal"}


async def test_that_context_variable_with_tool_id_is_serialized_with_tool_id_and_freshness_rules(
    container: Container,
) -> None:
    variable = await container[ContextVariableStore].create_variable(
        name="account_tier",
        description="Customer's account tier",
        tool_id=ToolId(service_name="local", tool_name="lookup_tier"),
        freshness_rules="0 * * * *",
    )

    indexer = _make_indexer(container)
    await indexer.run()

    data = indexer.captured_payload["context_variables"][variable.id].data
    assert isinstance(data, dict)
    assert data["name"] == "account_tier"
    assert data["description"] == "Customer's account tier"
    assert data["freshness_rules"] == "0 * * * *"
    assert data["tool_id"] == ["local", "lookup_tier"]


async def test_that_canned_response_is_serialized_with_fields_signals_and_metadata(
    container: Container,
) -> None:
    canned = await container[CannedResponseStore].create_canned_response(
        value="Hello {{name}}!",
        fields=[
            CannedResponseField(
                name="name",
                description="Customer's name",
                examples=["Alice", "Bob"],
            )
        ],
        signals=["greeting"],
        metadata={"locale": "en-US"},
        field_dependencies=["customer.name"],
    )

    indexer = _make_indexer(container)
    await indexer.run()

    data = indexer.captured_payload["canned_responses"][canned.id].data
    assert isinstance(data, dict)
    assert data["value"] == "Hello {{name}}!"
    assert data["signals"] == ["greeting"]
    assert data["metadata"] == {"locale": "en-US"}
    assert data["field_dependencies"] == ["customer.name"]
    assert data["fields"] == [
        {"name": "name", "description": "Customer's name", "examples": ["Alice", "Bob"]}
    ]


async def test_that_relationship_is_serialized_with_source_target_and_kind(
    container: Container,
) -> None:
    guideline = await container[GuidelineStore].create_guideline(
        condition="customer is angry", action="apologize"
    )

    relationship = await container[RelationshipStore].create_relationship(
        source=RelationshipEntity(id=guideline.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=TagId("vip"), kind=RelationshipEntityKind.TAG_ANY),
        kind=RelationshipKind.PRIORITY,
    )

    indexer = _make_indexer(container)
    await indexer.run()

    data = indexer.captured_payload["relationships"][relationship.id].data
    assert isinstance(data, dict)
    assert data["kind"] == "priority"
    assert data["source"] == {"id": guideline.id, "kind": "guideline"}
    assert data["target"] == {"id": "vip", "kind": "tag_any"}


async def test_that_tool_is_serialized_with_parameters_and_required(
    container: Container,
) -> None:
    local = container[LocalToolService]

    tool = await local.create_tool(
        name="lookup_tier",
        module_path="some.module",
        description="Look up the customer's tier",
        parameters={
            "customer_id": (
                {"type": "string", "description": "Customer's identifier"},
                ToolParameterOptions(),
            ),
        },
        required=["customer_id"],
        consequential=True,
        overlap=ToolOverlap.ALWAYS,
    )

    indexer = _make_indexer(container)
    await indexer.run()

    data = indexer.captured_payload["tools"][f"local:{tool.name}"].data
    assert isinstance(data, dict)
    assert data["name"] == "lookup_tier"
    assert data["description"] == "Look up the customer's tier"
    assert data["required"] == ["customer_id"]
    assert data["consequential"] is True
    assert data["overlap"] == ToolOverlap.ALWAYS.value
    assert "customer_id" in data["parameters"]
