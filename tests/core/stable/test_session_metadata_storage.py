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

import pytest

from parlant.adapters.db.transient import TransientDocumentDatabase
from parlant.core.agents import AgentId
from parlant.core.customers import CustomerId
from parlant.core import metadata
from parlant.core.sessions import (
    EventKind,
    EventSource,
    MetadataRevisionConflict,
    SessionDocumentStore,
    patch_internal_event_metadata,
    patch_public_session_metadata,
)


@pytest.mark.asyncio
async def test_session_metadata_commit_uses_compare_and_swap() -> None:
    async with SessionDocumentStore(TransientDocumentDatabase()) as store:
        session = await store.create_session(CustomerId("customer"), AgentId("agent"))

        committed = await store.commit_session_metadata(
            session.id,
            expected_revision=0,
            metadata={"application": "first"},
        )

        assert committed.metadata_revision == 1
        assert committed.metadata == {"application": "first"}

        with pytest.raises(MetadataRevisionConflict):
            await store.commit_session_metadata(
                session.id,
                expected_revision=0,
                metadata={"application": "stale"},
            )

        assert (await store.read_session(session.id)).metadata == {"application": "first"}


@pytest.mark.asyncio
async def test_public_metadata_writes_increment_only_metadata_revision() -> None:
    async with SessionDocumentStore(TransientDocumentDatabase()) as store:
        session = await store.create_session(CustomerId("customer"), AgentId("agent"))

        session = await store.set_metadata(session.id, "priority", "high")
        assert session.metadata_revision == 1

        session = await store.upsert_labels(session.id, {"vip"})
        assert session.metadata_revision == 1

        session = await store.unset_metadata(session.id, "priority")
        assert session.metadata_revision == 2


@pytest.mark.asyncio
async def test_public_store_metadata_methods_reject_the_internal_root() -> None:
    async with SessionDocumentStore(TransientDocumentDatabase()) as store:
        session = await store.create_session(CustomerId("customer"), AgentId("agent"))

        with pytest.raises(metadata.ReservedMetadataKeyError):
            await store.set_metadata(session.id, metadata.PARLANT_METADATA_KEY, {})

        with pytest.raises(metadata.ReservedMetadataKeyError):
            await store.unset_metadata(session.id, metadata.PARLANT_METADATA_KEY)


@pytest.mark.asyncio
async def test_public_metadata_batch_is_one_revision_and_preserves_internal_state() -> None:
    async with SessionDocumentStore(TransientDocumentDatabase()) as store:
        session = await store.create_session(
            CustomerId("customer"),
            AgentId("agent"),
            metadata=metadata.set_internal_value(
                {"remove": "old"},
                ("compass", "discovered_rule_ids"),
                ["rule-1"],
            ),
        )

        committed = await patch_public_session_metadata(
            store,
            session.id,
            {"priority": "high", "project": "support"},
            ["remove"],
        )

        assert committed.metadata_revision == 1
        assert committed.metadata["priority"] == "high"
        assert committed.metadata["project"] == "support"
        assert "remove" not in committed.metadata
        assert metadata.internal_value(
            committed.metadata,
            ("compass", "discovered_rule_ids"),
        ) == ["rule-1"]


@pytest.mark.asyncio
async def test_event_metadata_patch_uses_compare_and_swap() -> None:
    async with SessionDocumentStore(TransientDocumentDatabase()) as store:
        session = await store.create_session(CustomerId("customer"), AgentId("agent"))
        event = await store.create_event(
            session.id,
            EventSource.CUSTOMER,
            EventKind.CUSTOM,
            trace_id="trace",
            data={},
        )

        committed = await store.patch_event_metadata(
            session.id,
            event.id,
            expected_revision=0,
            set_values={"application": "first"},
        )
        assert committed.metadata_revision == 1

        with pytest.raises(MetadataRevisionConflict):
            await store.patch_event_metadata(
                session.id,
                event.id,
                expected_revision=0,
                set_values={"application": "stale"},
            )


@pytest.mark.asyncio
async def test_internal_event_patch_preserves_concurrent_public_metadata() -> None:
    async with SessionDocumentStore(TransientDocumentDatabase()) as store:
        session = await store.create_session(CustomerId("customer"), AgentId("agent"))
        event = await store.create_event(
            session.id,
            EventSource.AI_AGENT,
            EventKind.MESSAGE,
            trace_id="trace",
            data={"message": "hello"},
            metadata=metadata.set_internal_value({}, ("voice", "speculative"), True),
        )

        event = await store.patch_event_metadata(
            session.id,
            event.id,
            expected_revision=0,
            set_values={"application": "preserved"},
        )
        event = await patch_internal_event_metadata(
            store,
            session.id,
            event.id,
            ("voice", "spoken_text"),
            "hello",
        )

        assert event.metadata["application"] == "preserved"
        assert metadata.internal_value(event.metadata, ("voice", "speculative")) is True
        assert metadata.internal_value(event.metadata, ("voice", "spoken_text")) == "hello"
