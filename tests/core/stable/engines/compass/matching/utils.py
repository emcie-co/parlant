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

"""Test helpers for the Compass rule-matching components.

Provides hand-built ``EngineContext`` and ``Rule`` objects so the
turn evaluators / discovery can be exercised over a known interaction
history without spinning up the full SDK/engine.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
import os
from datetime import datetime, timezone
from typing import cast

from parlant.core.agents import Agent, AgentId, Effort, MessageOutputMode
from parlant.core.common import Weight, JSONSerializable, generate_id
from parlant.core.context_variables import (
    ContextVariable,
    ContextVariableId,
    ContextVariableValue,
    ContextVariableValueId,
)
from parlant.core.customers import Customer, CustomerId
from parlant.core.emission.event_buffer import EventBuffer
from parlant.core.emissions import EmittedEvent
from parlant.core.engines.engine_context import Interaction
from parlant.core.engines.session_metadata_workspace import SessionMetadataWorkspace
from parlant.core.engines.compass.matching.rule_ranker import RuleRanker
from parlant.core.engines.compass.context import (
    CompassDependencies,
    EngineContext,
    ResponseState,
)
from parlant.core.engines.compass.hooks import CompassHooks
from parlant.core.engines.compass.optimization_policy import CompassOptimizationPolicy
from parlant.core.engines.types import Context
from parlant.core.glossary import Term, TermId
from parlant.core.rules import Rule, RuleId
from parlant.core.loggers import StdoutLogger
from parlant.core.sessions import (
    EventKind,
    EventSource,
    MetadataRevisionConflict,
    Session,
    SessionId,
    SessionStore,
)
from parlant.core.tracer import LocalTracer
from parlant.core.groups import GroupId

from tests.core.common.utils import create_event_message


# Flip to False to only assert that relevant rules pass the filter, ignoring
# whatever the ranker does with the irrelevant ones.
# The negative half of every ranking case is off by default. Turn it on with
# PARLANT_ASSERT_IRRELEVANT_RULES=1 to measure a ranker's false positives, which
# is the only way a relevance threshold can be calibrated in both directions.
ASSERT_IRRELEVANT_RULES = os.environ.get("PARLANT_ASSERT_IRRELEVANT_RULES", "") not in (
    "",
    "0",
    "false",
)


class _TestSessionMetadataStore:
    """Minimal CAS store used by hand-built Compass contexts."""

    def __init__(self, session: Session) -> None:
        self._session = session

    async def read_session(self, session_id: SessionId) -> Session:
        if session_id != self._session.id:
            raise LookupError(session_id)
        return self._session

    async def commit_session_metadata(
        self,
        session_id: SessionId,
        expected_revision: int,
        metadata: Mapping[str, JSONSerializable],
    ) -> Session:
        if session_id != self._session.id:
            raise LookupError(session_id)
        if expected_revision != self._session.metadata_revision:
            raise MetadataRevisionConflict(self._session.metadata_revision)

        self._session = replace(
            self._session,
            metadata=deepcopy(dict(metadata)),
            metadata_revision=self._session.metadata_revision + 1,
        )
        return self._session


def create_rule(
    condition: str | None = None,
    action: str | None = None,
    *,
    content: str | None = None,
    title: str | None = None,
    groups: list[GroupId] = [],
) -> Rule:
    """Build a standalone ``Rule`` (no store) for unit tests.

    ``action`` is a legacy alias: pre-flattening (condition, action) call
    sites keep their meaning with the action text carried as the rule's
    content."""
    return Rule(
        id=RuleId(generate_id()),
        creation_utc=datetime.now(timezone.utc),
        modified_utc=datetime.now(timezone.utc),
        content=content or action,
        condition=condition or None,
        weight=Weight.MEDIUM,
        enabled=True,
        groups=groups,
        metadata={},
        title=title,
    )


def create_rule_by_name(
    rules_dict: Mapping[str, Mapping[str, str]],
    name: str,
) -> Rule:
    spec = rules_dict[name]
    return create_rule(condition=spec["condition"], action=spec.get("action"))


def create_term(
    name: str,
    description: str,
    synonyms: list[str] = [],
    groups: list[GroupId] = [],
) -> Term:
    return Term(
        id=TermId("-"),
        creation_utc=datetime.now(timezone.utc),
        modified_utc=datetime.now(timezone.utc),
        name=name,
        description=description,
        synonyms=synonyms,
        groups=groups,
    )


def create_context_variable(
    name: str,
    data: JSONSerializable,
    groups: list[GroupId] = [],
) -> tuple[ContextVariable, ContextVariableValue]:
    return ContextVariable(
        id=ContextVariableId("-"),
        creation_utc=datetime.now(timezone.utc),
        modified_utc=datetime.now(timezone.utc),
        name=name,
        description="",
        tool_id=None,
        freshness_rules=None,
        groups=groups,
    ), ContextVariableValue(
        ContextVariableValueId("-"),
        modified_utc=datetime.now(timezone.utc),
        data=data,
    )


def create_staged_tool_event(data: JSONSerializable) -> EmittedEvent:
    return EmittedEvent(
        source=EventSource.AI_AGENT,
        kind=EventKind.TOOL,
        trace_id="",
        data=data,
        metadata=None,
    )


async def base_test_that_rules_are_ranked_correctly(
    ranker: RuleRanker,
    rules_dict: Mapping[str, Mapping[str, str]],
    conversation: list[tuple[EventSource, str]],
    conversation_rule_names: list[str],
    relevant_rule_names: list[str],
    irrelevant_rule_names: list[str],
    *,
    agent_instructions: str | None = None,
    context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]] = [],
    terms: Sequence[Term] = [],
    staged_events: Sequence[EmittedEvent] = [],
) -> None:
    """Rank ``conversation_rule_names`` against ``conversation`` and assert that:

    - every rule in ``relevant_rule_names`` was ranked as relevant, and
    - every rule in ``irrelevant_rule_names`` was ranked as not relevant.

    A rule that appears in neither list is a "don't care": any decision the
    ranker makes about it is accepted.
    """
    assert set(relevant_rule_names) <= set(conversation_rule_names)
    assert set(irrelevant_rule_names) <= set(conversation_rule_names)
    assert not (set(relevant_rule_names) & set(irrelevant_rule_names))

    rules_by_name = {
        name: create_rule_by_name(rules_dict, name) for name in conversation_rule_names
    }

    agent = create_agent(instructions=agent_instructions) if agent_instructions else None

    context = create_engine_context(conversation=conversation, agent=agent)
    context.state = ResponseState(
        agent_effort=context.agent.effort,
        context_variables=list(context_variables),
        glossary_terms=set(terms),
        tool_events=list(staged_events),
    )

    result = await ranker.evaluate(context, list(rules_by_name.values()))

    relevance_by_id = {
        evaluation.rule.id: evaluation.is_relevant for evaluation in result.evaluations
    }

    for name in relevant_rule_names:
        rule = rules_by_name[name]
        assert relevance_by_id.get(rule.id) is True, (
            f"expected rule {name!r} to be ranked as relevant, but it wasn't"
        )

    if ASSERT_IRRELEVANT_RULES:
        for name in irrelevant_rule_names:
            rule = rules_by_name[name]
            assert relevance_by_id.get(rule.id) is False, (
                f"expected rule {name!r} to be ranked as not relevant, but it was"
            )


def create_agent(name: str = "Test Agent", instructions: str | None = None) -> Agent:
    now = datetime.now(timezone.utc)
    return Agent(
        engine="compass",
        id=AgentId(generate_id()),
        name=name,
        instructions=instructions,
        creation_utc=now,
        modified_utc=now,
        max_steps=3,
        groups=[],
        message_output_mode=MessageOutputMode.STREAM,
        effort=Effort.MEDIUM,
    )


def create_customer(name: str = "Test Customer") -> Customer:
    creation_utc = datetime.now(timezone.utc)

    return Customer(
        id=CustomerId(generate_id()),
        creation_utc=creation_utc,
        modified_utc=creation_utc,
        name=name,
        metadata={},
        groups=[],
    )


def create_session(
    agent: Agent,
    customer: Customer,
    metadata: Mapping[str, JSONSerializable] | None = None,
) -> Session:
    creation_utc = datetime.now(timezone.utc)

    return Session(
        id=SessionId(generate_id()),
        creation_utc=creation_utc,
        modified_utc=creation_utc,
        customer_id=customer.id,
        agent_id=agent.id,
        mode="auto",
        title=None,
        agent_states=[],
        metadata=dict(metadata or {}),
    )


def create_engine_context(
    conversation: Sequence[tuple[EventSource, str]],
    *,
    agent: Agent | None = None,
    customer: Customer | None = None,
    metadata: Mapping[str, JSONSerializable] | None = None,
    hooks: CompassHooks | None = None,
) -> EngineContext:
    """Manually assemble an ``EngineContext`` carrying ``conversation`` as its
    interaction history. Candidate rules are passed directly to the
    matcher methods (evaluate/discover), not through the context.
    """
    agent = agent or create_agent()
    customer = customer or create_customer()
    session = create_session(agent, customer, metadata)
    workspace = SessionMetadataWorkspace(
        session,
        cast(SessionStore, _TestSessionMetadataStore(session)),
    )
    session = workspace.session

    tracer = LocalTracer()
    logger = StdoutLogger(tracer)

    events = [
        create_event_message(offset=i, source=source, message=message)
        for i, (source, message) in enumerate(conversation)
    ]

    return EngineContext(
        info=Context(session_id=session.id, agent_id=agent.id),
        logger=logger,
        tracer=tracer,
        agent=agent,
        customer=customer,
        session=session,
        session_event_emitter=EventBuffer(emitting_agent=agent),
        response_event_emitter=EventBuffer(emitting_agent=agent),
        interaction=Interaction(events=events),
        state=ResponseState(),
        dependencies=CompassDependencies(
            optimization_policy=CompassOptimizationPolicy(),
            hooks=hooks or CompassHooks(),
        ),
        session_metadata=workspace,
    )
