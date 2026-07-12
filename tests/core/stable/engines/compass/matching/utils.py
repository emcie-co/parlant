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

from collections.abc import Iterator, Mapping, Sequence
import asyncio
import contextvars
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

from parlant.core.agents import Agent, AgentId, CompositionMode, Effort, MessageOutputMode
from parlant.core.capabilities import Capability, CapabilityId
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
from parlant.core.engines.engine_context import EngineContext, Interaction
from parlant.core.engines.compass.matching.rule_ranker import RuleRanker
from parlant.core.engines.compass.response_state import ResponseState
from parlant.core.engines.types import Context
from parlant.core.glossary import Term, TermId
from parlant.core.rules import Rule, RuleContent, RuleId
from parlant.core.loggers import StdoutLogger
from parlant.core.sessions import EventKind, EventSource, Session, SessionId
from parlant.core.tracer import AttributeValue, LocalTracer
from parlant.core.groups import GroupId

from tests.core.common.utils import create_event_message


# Flip to False to only assert that relevant rules pass the filter, ignoring
# whatever the ranker does with the irrelevant ones.
ASSERT_IRRELEVANT_RULES = False


def _ensure_event_loop() -> None:
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


@dataclass(frozen=True)
class RecordedEvent:
    name: str
    attributes: Mapping[str, AttributeValue]
    span_id: str


@dataclass(frozen=True)
class RecordedSpan:
    name: str
    attributes: Mapping[str, AttributeValue]


class RecordingTracer(LocalTracer):
    def __init__(self) -> None:
        super().__init__()
        self.started_spans: list[RecordedSpan] = []
        self.events: list[RecordedEvent] = []
        self.recorded_attributes: dict[str, AttributeValue] = {}
        self._recorded_span_id: contextvars.ContextVar[str] = contextvars.ContextVar(
            "matching_recording_tracer_span_id",
            default="<main>",
        )

    @contextmanager
    def span(
        self,
        span_id: str,
        attributes: Mapping[str, AttributeValue] = {},
    ) -> Iterator[None]:
        self.started_spans.append(RecordedSpan(name=span_id, attributes=dict(attributes)))
        token = self._recorded_span_id.set(span_id)

        with super().span(span_id, attributes):
            try:
                yield
            finally:
                self._recorded_span_id.reset(token)

    @property
    def span_id(self) -> str:
        return self._recorded_span_id.get()

    def add_event(
        self,
        name: str,
        attributes: Mapping[str, AttributeValue] = {},
    ) -> None:
        self.events.append(
            RecordedEvent(name=name, attributes=dict(attributes), span_id=self.span_id)
        )

    def set_attribute(self, name: str, value: AttributeValue) -> None:
        super().set_attribute(name, value)
        self.recorded_attributes[name] = value

    def get_attribute(self, name: str) -> AttributeValue | None:
        return self.recorded_attributes.get(name)


def create_rule(
    condition: str,
    action: str | None = None,
    *,
    description: str | None = None,
    title: str | None = None,
    groups: list[GroupId] = [],
) -> Rule:
    """Build a standalone ``Rule`` (no store) for unit tests."""
    return Rule(
        id=RuleId(generate_id()),
        creation_utc=datetime.now(timezone.utc),
        modified_utc=datetime.now(timezone.utc),
        content=RuleContent(condition=condition, action=action, description=description),
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


def create_capability(
    title: str,
    description: str,
    *,
    id: str = "cap_-",
    signals: list[str] = [],
    groups: list[GroupId] = [],
) -> Capability:
    return Capability(
        id=CapabilityId(id),
        creation_utc=datetime.now(timezone.utc),
        title=title,
        description=description,
        signals=signals,
        groups=groups,
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
    agent_description: str | None = None,
    context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]] = [],
    terms: Sequence[Term] = [],
    capabilities: Sequence[Capability] = [],
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

    agent = create_agent(description=agent_description) if agent_description else None

    context = create_engine_context(conversation=conversation, agent=agent)
    context.state = ResponseState(
        agent_effort=context.agent.effort,
        context_variables=list(context_variables),
        glossary_terms=set(terms),
        capabilities=list(capabilities),
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


def create_agent(name: str = "Test Agent", description: str | None = None) -> Agent:
    now = datetime.now(timezone.utc)
    return Agent(
        id=AgentId(generate_id()),
        name=name,
        description=description,
        creation_utc=now,
        modified_utc=now,
        max_engine_iterations=3,
        groups=[],
        engine="compass",
        composition_mode=CompositionMode.FLUID,
        message_output_mode=MessageOutputMode.STREAM,
        effort=Effort.MEDIUM,
    )


def create_customer(name: str = "Test Customer") -> Customer:
    return Customer(
        id=CustomerId(generate_id()),
        creation_utc=datetime.now(timezone.utc),
        name=name,
        extra={},
        groups=[],
    )


def create_session(agent: Agent, customer: Customer) -> Session:
    creation_utc = datetime.now(timezone.utc)

    return Session(
        id=SessionId(generate_id()),
        creation_utc=creation_utc,
        modified_utc=creation_utc,
        customer_id=customer.id,
        agent_id=agent.id,
        mode="auto",
        title=None,
        consumption_offsets={},
        agent_states=[],
        metadata={},
    )


def create_engine_context(
    conversation: Sequence[tuple[EventSource, str]],
    *,
    agent: Agent | None = None,
    customer: Customer | None = None,
) -> EngineContext:
    """Manually assemble an ``EngineContext`` carrying ``conversation`` as its
    interaction history. Candidate rules are passed directly to the
    matcher methods (evaluate/discover), not through the context.
    """
    agent = agent or create_agent()
    customer = customer or create_customer()
    session = create_session(agent, customer)

    tracer = LocalTracer()
    logger = StdoutLogger(tracer)

    events = [
        create_event_message(offset=i, source=source, message=message)
        for i, (source, message) in enumerate(conversation)
    ]

    _ensure_event_loop()

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
        # The rule-matching components read only `interaction`; the engine's
        # response state isn't exercised here.
        state=None,
    )
