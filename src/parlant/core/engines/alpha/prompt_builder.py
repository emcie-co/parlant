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

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
import json
from typing import Any, Callable, Optional, Sequence, cast

from parlant.core.agents import Agent
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.customers import Customer
from parlant.core.journeys import Journey
from parlant.core.sessions import Event, EventKind, EventSource, MessageEventData, ToolEventData
from parlant.core.glossary import Term
from parlant.core.engines.alpha.utils import (
    context_variables_to_json,
)
from parlant.core.emissions import EmittedEvent
from parlant.core.guidelines import Guideline


class BuiltInSection(Enum):
    AGENT_IDENTITY = auto()
    CUSTOMER_IDENTITY = auto()
    INTERACTION_HISTORY = auto()
    CONTEXT_VARIABLES = auto()
    GLOSSARY = auto()
    GUIDELINE_DESCRIPTIONS = auto()
    GUIDELINES = auto()
    STAGED_EVENTS = auto()
    JOURNEYS = auto()
    OBSERVATIONS = auto()


class SectionStatus(Enum):
    ACTIVE = auto()
    """The section has active information that must be taken into account"""

    PASSIVE = auto()
    """The section is inactive, but may have explicit empty-state inclusion in the prompt"""

    NONE = auto()
    """The section is not included in the prompt in any fashion"""


@dataclass(frozen=True)
class Section:
    template: str
    props: dict[str, Any]
    status: Optional[SectionStatus]


class PromptBuilder:
    def __init__(self, on_build: Optional[Callable[[str], None]] = None) -> None:
        self.sections: dict[str | BuiltInSection, Section] = {}

        self._on_build = on_build
        self._cached_results: set[str] = set()

    def _call_on_build(self, prompt: str) -> None:
        if prompt in self._cached_results:
            return

        if self._on_build:
            self._on_build(prompt)

        self._cached_results.add(prompt)

    def build(self) -> str:
        section_contents = [s.template.format(**s.props) for s in self.sections.values()]
        prompt = "\n\n".join(section_contents)

        self._call_on_build(prompt)

        return prompt

    def add_section(
        self,
        name: str | BuiltInSection,
        template: str,
        props: dict[str, Any] = {},
        status: Optional[SectionStatus] = None,
    ) -> PromptBuilder:
        if name in self.sections:
            raise ValueError(f"Section '{name}' was already added")

        self.sections[name] = Section(
            template=template,
            props=props,
            status=status,
        )

        return self

    def edit_section(
        self,
        name: str | BuiltInSection,
        editor_func: Callable[[Section], Section],
    ) -> PromptBuilder:
        if name in self.sections:
            self.sections[name] = editor_func(self.sections[name])
        return self

    def section_status(self, name: str | BuiltInSection) -> SectionStatus:
        if name in self.sections and self.sections[name].status is not None:
            return cast(SectionStatus, self.sections[name].status)
        else:
            return SectionStatus.NONE

    @staticmethod
    def adapt_event(e: Event | EmittedEvent) -> str:
        data = e.data

        if e.kind == EventKind.MESSAGE:
            message_data = cast(MessageEventData, e.data)

            if message_data.get("flagged"):
                data = {
                    "participant": message_data["participant"]["display_name"],
                    "message": "<N/A>",
                    "censored": True,
                    "reasons": message_data["tags"],
                }
            else:
                data = {
                    "participant": message_data["participant"]["display_name"],
                    "message": message_data["message"],
                }

        if e.kind == EventKind.TOOL:
            tool_data = cast(ToolEventData, e.data)

            data = {
                "tool_calls": [
                    {
                        "tool_id": tc["tool_id"],
                        "arguments": tc["arguments"],
                        "result": tc["result"]["data"],
                    }
                    for tc in tool_data["tool_calls"]
                ]
            }

        source_map: dict[EventSource, str] = {
            EventSource.CUSTOMER: "user",
            EventSource.CUSTOMER_UI: "frontend_application",
            EventSource.HUMAN_AGENT: "human_service_agent",
            EventSource.HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT: "ai_agent",
            EventSource.AI_AGENT: "ai_agent",
            EventSource.SYSTEM: "system-provided",
        }

        return json.dumps(
            {
                "event_kind": e.kind.value,
                "event_source": source_map[e.source],
                "data": data,
            }
        )

    def add_agent_identity(
        self,
        agent: Agent,
    ) -> PromptBuilder:
        if agent.description:
            self.add_section(
                name=BuiltInSection.AGENT_IDENTITY,
                template="""
You are an AI agent named {agent_name}.

The following is a description of your background and personality: ###
{agent_description}
###
""",
                props={
                    "agent_name": agent.name,
                    "agent_description": agent.description,
                },
                status=SectionStatus.ACTIVE,
            )

        return self

    def add_customer_identity(
        self,
        customer: Customer,
    ) -> PromptBuilder:
        self.add_section(
            name=BuiltInSection.CUSTOMER_IDENTITY,
            template="""
The user you're interacting with is called {customer_name}.
""",
            props={
                "customer_name": customer.name,
            },
            status=SectionStatus.ACTIVE,
        )

        return self

    def add_interaction_history(
        self,
        events: Sequence[Event],
    ) -> PromptBuilder:
        if events:
            interaction_events = [self.adapt_event(e) for e in events if e.kind != EventKind.STATUS]

            self.add_section(
                name=BuiltInSection.INTERACTION_HISTORY,
                template="""
The following is a list of events describing a back-and-forth
interaction between you and a user: ###
{interaction_events}
###
""",
                props={"interaction_events": interaction_events},
                status=SectionStatus.ACTIVE,
            )
        else:
            self.add_section(
                name=BuiltInSection.INTERACTION_HISTORY,
                template="""
Your interaction with the user has just began, and no events have been recorded yet.
Proceed with your task accordingly.
""",
                status=SectionStatus.PASSIVE,
            )

        return self

    def add_context_variables(
        self,
        variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
    ) -> PromptBuilder:
        if variables:
            context_values = context_variables_to_json(variables)

            self.add_section(
                name=BuiltInSection.CONTEXT_VARIABLES,
                template="""
The following is information that you're given about the user and context of the interaction: ###
{context_values}
###
""",
                props={"context_values": context_values},
                status=SectionStatus.ACTIVE,
            )

        return self

    def add_glossary(
        self,
        terms: Sequence[Term],
    ) -> PromptBuilder:
        if terms:
            terms_string = "\n".join(f"{i}) {repr(t)}" for i, t in enumerate(terms, start=1))

            self.add_section(
                name=BuiltInSection.GLOSSARY,
                template="""
The following is a glossary of the business.
Understanding these terms, as they apply to the business, is critical for your task.
When encountering any of these terms, prioritize the interpretation provided here over any definitions you may already know.
Please be tolerant of possible typos by the user with regards to these terms,
and let the user know if/when you assume they meant a term by their typo: ###
{terms_string}
###
""",  # noqa
                props={"terms_string": terms_string},
                status=SectionStatus.ACTIVE,
            )

        return self

    def add_staged_events(
        self,
        events: Sequence[EmittedEvent],
    ) -> PromptBuilder:
        if events:
            staged_events_as_dict = [
                self.adapt_event(e) for e in events if e.kind == EventKind.TOOL
            ]

            self.add_section(
                name=BuiltInSection.STAGED_EVENTS,
                template="""
Here are the most recent staged events for your reference.
They represent interactions with external tools that perform actions or provide information.
Prioritize their data over any other sources and use their details to complete your task: ###
{staged_events_as_dict}
###
""",
                props={"staged_events_as_dict": staged_events_as_dict},
                status=SectionStatus.ACTIVE,
            )

        return self

    def add_observations(  # Here for future reference, not currently in use
        self,
        observations: Sequence[Guideline],
    ) -> PromptBuilder:
        if observations:
            observations_string = ""
            self.add_section(
                name=BuiltInSection.OBSERVATIONS,
                template="""
The following are observations that were deemed relevant to the interaction with the customer. Use them to inform your response:
###
{observations_string}
###
""",  # noqa
                props={"observations_string": observations_string},
                status=SectionStatus.ACTIVE,
            )

        return self

    def add_journeys(
        self,
        journeys: Sequence[Journey],
    ) -> PromptBuilder:
        if journeys:
            journeys_string = "\n\n".join(
                [
                    f"""
Journey {i}: {journey.title}
{journey.description}
"""
                    for i, journey in enumerate(journeys, start=1)
                ]
            )

            self.add_section(
                name=BuiltInSection.JOURNEYS,
                template="""
The following are 'journeys' - predefined processes from the business you represent that guide customer interactions. Journeys may include step-by-step workflows, general instructions, or relevant knowledge to help you assist customers effectively.

If a conversation is already in progress along a journey path, continue with the next appropriate step. For journeys with multiple steps:
1. Identify which steps have already been completed
2. Perform only the next logical step (either by the journey's steps or by your deduction) in the sequence
3. Reserve subsequent steps for later in the conversation

Follow each journey exactly as specified. If a journey indicates multiple actions should be taken in a single step, follow those instructions. Otherwise, take only one step at a time to avoid overwhelming the customer.

Example: In a product return journey with steps to 1) verify purchase details, 2) assess return eligibility, 3) provide return instructions, and 4) process refund, if you've just confirmed the item is eligible for return (step 2 complete), your next response should only provide shipping instructions (step 3), leaving the refund processing (step 4) for after the customer has shipped the item.
###
{journeys_string}
###
""",  # noqa
                props={"journeys_string": journeys_string},
                status=SectionStatus.ACTIVE,
            )
        return self
