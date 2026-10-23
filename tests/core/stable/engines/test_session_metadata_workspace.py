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

import ast
from dataclasses import MISSING
from datetime import datetime, timezone
from pathlib import Path
from typing import get_args, get_origin, get_type_hints

import pytest

from parlant.adapters.db.transient import TransientDocumentDatabase
from parlant.core.agents import AgentId
from parlant.core.customers import CustomerId
from parlant.core.engines.compass.context import (
    CompassDependencies,
    EngineContext as CompassEngineContext,
    ResponseState as CompassResponseState,
)
from parlant.core.engines.engine_context import EngineContext
from parlant.core.engines.session_metadata_workspace import (
    MetadataOperationCategory,
    SessionMetadataWorkspace,
    WorkspaceLifecycleError,
    WorkspaceState,
)
from parlant.core import metadata
from parlant.core.sessions import Session, SessionDocumentStore, SessionId


def test_engine_context_requires_its_workspace_and_dependencies_and_compass_types_both() -> None:
    workspace_field = EngineContext.__dataclass_fields__["session_metadata"]
    dependencies_field = EngineContext.__dataclass_fields__["dependencies"]

    assert workspace_field.default is MISSING
    assert workspace_field.default_factory is MISSING
    assert workspace_field.kw_only
    assert dependencies_field.default is MISSING
    assert dependencies_field.default_factory is MISSING
    assert get_type_hints(EngineContext)["session_metadata"] is SessionMetadataWorkspace
    assert get_origin(CompassEngineContext) is EngineContext
    assert get_args(CompassEngineContext) == (CompassResponseState, CompassDependencies)


@pytest.mark.asyncio
async def test_two_stale_workspaces_replay_disjoint_operations_after_conflict() -> None:
    async with SessionDocumentStore(TransientDocumentDatabase()) as store:
        session = await store.create_session(CustomerId("customer"), AgentId("agent"))
        rule_workspace = SessionMetadataWorkspace(session, store)
        glossary_workspace = SessionMetadataWorkspace(session, store)

        rule_workspace.stage_rule_discovery(["rule-1"])
        glossary_workspace.stage_glossary_update(hits={"term-1": 8})

        await rule_workspace.commit()
        committed = await glossary_workspace.commit()

        assert committed.metadata_revision == 2
        assert metadata.internal_value(
            committed.metadata,
            ("compass", "discovered_rule_ids"),
        ) == ["rule-1"]
        assert metadata.internal_value(
            committed.metadata,
            ("compass", "session_terms"),
        ) == {"term-1": 8}


@pytest.mark.asyncio
async def test_external_user_write_survives_workspace_commit() -> None:
    async with SessionDocumentStore(TransientDocumentDatabase()) as store:
        session = await store.create_session(CustomerId("customer"), AgentId("agent"))
        workspace = SessionMetadataWorkspace(session, store)
        workspace.stage_rule_discovery(["rule-1"])

        await store.set_metadata(session.id, "priority", "high")
        committed = await workspace.commit()

        assert committed.metadata["priority"] == "high"
        assert metadata.internal_value(
            committed.metadata,
            ("compass", "discovered_rule_ids"),
        ) == ["rule-1"]


@pytest.mark.asyncio
async def test_clean_workspace_commit_performs_no_metadata_update() -> None:
    async with SessionDocumentStore(TransientDocumentDatabase()) as store:
        session = await store.create_session(CustomerId("customer"), AgentId("agent"))
        workspace = SessionMetadataWorkspace(session, store)

        committed = await workspace.commit()

        assert committed.metadata_revision == 0
        assert workspace.state is WorkspaceState.COMMITTED


@pytest.mark.asyncio
async def test_behavioral_rollback_retains_accounting_operation() -> None:
    async with SessionDocumentStore(TransientDocumentDatabase()) as store:
        session = await store.create_session(CustomerId("customer"), AgentId("agent"))
        workspace = SessionMetadataWorkspace(session, store)
        workspace.stage_rule_discovery(["rule-1"])
        workspace.set_internal(
            ("usage", "records", "trace-1"),
            {"requests": 1},
            category=MetadataOperationCategory.ACCOUNTING,
        )

        workspace.rollback_behavioral_state()
        committed = await workspace.commit()

        assert (
            metadata.internal_value(
                committed.metadata,
                ("compass", "discovered_rule_ids"),
                [],
            )
            == []
        )
        assert metadata.internal_value(
            committed.metadata,
            ("usage", "records", "trace-1"),
        ) == {"requests": 1}


@pytest.mark.asyncio
async def test_finalized_workspace_rejects_mutation_but_keeps_final_view() -> None:
    async with SessionDocumentStore(TransientDocumentDatabase()) as store:
        session = await store.create_session(CustomerId("customer"), AgentId("agent"))
        workspace = SessionMetadataWorkspace(session, store)
        workspace.stage_rule_discovery(["rule-1"])
        await workspace.commit()

        assert metadata.internal_value(
            workspace.view,
            ("compass", "discovered_rule_ids"),
        ) == ["rule-1"]

        with pytest.raises(WorkspaceLifecycleError):
            workspace.stage_rule_discovery(["rule-2"])


def test_workspace_view_never_exposes_mutable_nested_state() -> None:
    database = TransientDocumentDatabase()
    # This unit only exercises synchronous staging; the store is not entered or used.
    store = SessionDocumentStore(database)
    now = datetime.now(timezone.utc)
    session = Session(
        id=SessionId("session"),
        creation_utc=now,
        modified_utc=now,
        customer_id=CustomerId("customer"),
        agent_id=AgentId("agent"),
        mode="auto",
        title=None,
        agent_states=[],
        metadata={},
    )
    workspace = SessionMetadataWorkspace(session, store)
    workspace.set_internal(("compass", "state"), {"ids": ["r1"]})

    state = workspace.get_internal(("compass", "state"))
    assert isinstance(state, dict)
    state["ids"].append("r2")

    assert workspace.get_internal(("compass", "state")) == {"ids": ["r1"]}


def test_workspace_module_documents_every_declared_class_and_method() -> None:
    source_path = (
        Path(__file__).parents[4]
        / "src"
        / "parlant"
        / "core"
        / "engines"
        / "session_metadata_workspace.py"
    )
    tree = ast.parse(source_path.read_text())

    assert ast.get_docstring(tree)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        assert ast.get_docstring(node), f"{node.name} is missing a class docstring"
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert ast.get_docstring(child), (
                    f"{node.name}.{child.name} is missing a method docstring"
                )
