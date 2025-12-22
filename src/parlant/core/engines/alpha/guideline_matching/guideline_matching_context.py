from dataclasses import dataclass
from typing import Optional, Sequence

from parlant.core.agents import Agent
from parlant.core.capabilities import Capability
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.customers import Customer
from parlant.core.emissions import EmittedEvent
from parlant.core.glossary import Term
from parlant.core.journeys import Journey, JourneyId
from parlant.core.sessions import Event, Session


@dataclass(frozen=True)
class GuidelineMatchingContext:
    agent: Agent
    session: Session
    customer: Customer
    context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]]
    interaction_history: Sequence[Event]
    terms: Sequence[Term]
    capabilities: Sequence[Capability]
    staged_events: Sequence[EmittedEvent]
    active_journeys: Sequence[Journey]
    journey_paths: dict[JourneyId, list[Optional[str]]]
