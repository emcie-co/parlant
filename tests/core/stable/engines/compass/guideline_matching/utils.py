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

"""Test helpers for the Compass guideline-matching components.

Provides hand-built ``EngineContext`` and ``Guideline`` objects so the
distiller / ranker / recaller can be exercised over a known interaction
history without spinning up the full SDK/engine.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

from parlant.core.agents import Agent, AgentId, CompositionMode, Effort, MessageOutputMode
from parlant.core.capabilities import Capability, CapabilityId
from parlant.core.common import Criticality, JSONSerializable, generate_id
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
from parlant.core.engines.compass.guideline_matching.guideline_ranker import GuidelineRanker
from parlant.core.engines.compass.response_state import ResponseState
from parlant.core.engines.types import Context
from parlant.core.glossary import Term, TermId
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.loggers import StdoutLogger
from parlant.core.sessions import EventKind, EventSource, Session, SessionId
from parlant.core.tracer import LocalTracer
from parlant.core.tags import TagId

from tests.core.common.utils import create_event_message


# Flip to False to only assert that relevant guidelines pass the filter, ignoring
# whatever the ranker does with the irrelevant ones.
ASSERT_IRRELEVANT_GUIDELINES = False


def create_guideline(
    condition: str,
    action: str | None = None,
    *,
    tags: list[TagId] = [],
) -> Guideline:
    """Build a standalone ``Guideline`` (no store) for unit tests."""
    return Guideline(
        id=GuidelineId(generate_id()),
        creation_utc=datetime.now(timezone.utc),
        modified_utc=datetime.now(timezone.utc),
        content=GuidelineContent(condition=condition, action=action),
        criticality=Criticality.MEDIUM,
        enabled=True,
        tags=tags,
        metadata={},
    )


def create_guideline_by_name(
    guidelines_dict: Mapping[str, Mapping[str, str]],
    name: str,
) -> Guideline:
    spec = guidelines_dict[name]
    return create_guideline(condition=spec["condition"], action=spec.get("action"))


def create_term(
    name: str,
    description: str,
    synonyms: list[str] = [],
    tags: list[TagId] = [],
) -> Term:
    return Term(
        id=TermId("-"),
        creation_utc=datetime.now(timezone.utc),
        modified_utc=datetime.now(timezone.utc),
        name=name,
        description=description,
        synonyms=synonyms,
        tags=tags,
    )


def create_context_variable(
    name: str,
    data: JSONSerializable,
    tags: list[TagId] = [],
) -> tuple[ContextVariable, ContextVariableValue]:
    return ContextVariable(
        id=ContextVariableId("-"),
        creation_utc=datetime.now(timezone.utc),
        modified_utc=datetime.now(timezone.utc),
        name=name,
        description="",
        tool_id=None,
        freshness_rules=None,
        tags=tags,
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
    tags: list[TagId] = [],
) -> Capability:
    return Capability(
        id=CapabilityId(id),
        creation_utc=datetime.now(timezone.utc),
        title=title,
        description=description,
        signals=signals,
        tags=tags,
    )


def create_staged_tool_event(data: JSONSerializable) -> EmittedEvent:
    return EmittedEvent(
        source=EventSource.AI_AGENT,
        kind=EventKind.TOOL,
        trace_id="",
        data=data,
        metadata=None,
    )


async def base_test_that_guidelines_are_ranked_correctly(
    ranker: GuidelineRanker,
    guidelines_dict: Mapping[str, Mapping[str, str]],
    conversation: list[tuple[EventSource, str]],
    conversation_guideline_names: list[str],
    relevant_guideline_names: list[str],
    irrelevant_guideline_names: list[str],
    *,
    agent_description: str | None = None,
    context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]] = [],
    terms: Sequence[Term] = [],
    capabilities: Sequence[Capability] = [],
    staged_events: Sequence[EmittedEvent] = [],
) -> None:
    """Rank ``conversation_guideline_names`` against ``conversation`` and assert that:

    - every guideline in ``relevant_guideline_names`` was ranked as relevant, and
    - every guideline in ``irrelevant_guideline_names`` was ranked as not relevant.

    A guideline that appears in neither list is a "don't care": any decision the
    ranker makes about it is accepted.
    """
    assert set(relevant_guideline_names) <= set(conversation_guideline_names)
    assert set(irrelevant_guideline_names) <= set(conversation_guideline_names)
    assert not (set(relevant_guideline_names) & set(irrelevant_guideline_names))

    guidelines_by_name = {
        name: create_guideline_by_name(guidelines_dict, name)
        for name in conversation_guideline_names
    }

    agent = create_agent(description=agent_description) if agent_description else None

    context = create_engine_context(conversation=conversation, agent=agent)
    context.state = ResponseState(
        context_variables=list(context_variables),
        glossary_terms=set(terms),
        capabilities=list(capabilities),
        tool_events=list(staged_events),
    )

    result = await ranker.rank(context, list(guidelines_by_name.values()))

    relevance_by_id = {
        ranked.guideline.id: ranked.is_relevant for ranked in result.ranked_guidelines
    }

    for name in relevant_guideline_names:
        guideline = guidelines_by_name[name]
        assert relevance_by_id.get(guideline.id) is True, (
            f"expected guideline {name!r} to be ranked as relevant, but it wasn't"
        )

    if ASSERT_IRRELEVANT_GUIDELINES:
        for name in irrelevant_guideline_names:
            guideline = guidelines_by_name[name]
            assert relevance_by_id.get(guideline.id) is False, (
                f"expected guideline {name!r} to be ranked as not relevant, but it was"
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
        tags=[],
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
        tags=[],
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
    interaction history. Candidate guidelines are passed directly to the
    matcher methods (distill/rank/recall), not through the context.
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
        # The guideline-matching components read only `interaction`; the engine's
        # response state isn't exercised here.
        state=None,
    )
