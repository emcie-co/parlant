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
from itertools import chain
import json
from typing import Any, Callable, Mapping, Optional, Sequence, cast

from parlant.core.agents import Agent
from parlant.core.capabilities import Capability
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.customers import Customer
from parlant.core.engines.alpha.guideline_matching.generic.common import (
    GuidelineInternalRepresentation,
)
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.sessions import Event, EventKind, EventSource, MessageEventData, ToolEventData
from parlant.core.glossary import Term
from parlant.core.engines.alpha.utils import (
    context_variables_to_json,
)
from parlant.core.emissions import EmittedEvent
from parlant.core.guidelines import Guideline, GuidelineId
from parlant.core.tools import ToolId


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
    CAPABILITIES = auto()


class SectionStatus(Enum):
    ACTIVE = auto()
    """The section has active information that must be taken into account"""

    PASSIVE = auto()
    """The section is inactive, but may have explicit empty-state inclusion in the prompt"""

    NONE = auto()
    """The section is not included in the prompt in any fashion"""


@dataclass(frozen=True)
class PromptSection:
    template: str
    props: dict[str, Any]
    status: Optional[SectionStatus]


class PromptBuilder:
    def __init__(self, on_build: Optional[Callable[[str], None]] = None) -> None:
        self.sections: dict[str | BuiltInSection, PromptSection] = {}

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

        self.sections[name] = PromptSection(
            template=template,
            props=props,
            status=status,
        )

        return self

    def edit_section(
        self,
        name: str | BuiltInSection,
        editor_func: Callable[[PromptSection], PromptSection],
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

    _INTERACTION_BODY = """
The following is a list of events describing a back-and-forth
interaction between you and a user: ###
{interaction_events}
###
"""

    _EMPTY_HISTORY = """
Your interaction with the user has just began, and no events have been recorded yet.
Proceed with your task accordingly.
"""

    def _gather_interaction_events(
        self,
        events: Sequence[Event],
        staged_events: Sequence[EmittedEvent],
    ) -> list[str]:
        combined = list(events) + list(staged_events)
        return [self.adapt_event(e) for e in combined if e.kind != EventKind.STATUS]

    def _last_agent_message_note(
        self,
        events: Sequence[Event],
    ) -> str:
        last_message_event = next(
            (e for e in reversed(events) if e.kind == EventKind.MESSAGE),
            None,
        )
        if not last_message_event or last_message_event.source != EventSource.AI_AGENT:
            return ""

        last_message = cast(MessageEventData, last_message_event.data)["message"]
        return f"\nIMPORTANT: Please note that the last message was sent by you, the AI agent (likely as a preamble). Your last message was: ###\n{last_message}\n###\n\nYou must keep that in mind when responding to the user, to continue the last message naturally (without repeating anything similar in your last message - make sure you don't repeat something like this in your next message - it was already said!)."

    def _add_history_section(
        self,
        interaction_events: list[str],
        last_event_note: str | None = None,
    ) -> None:
        template = self._INTERACTION_BODY
        props: dict[str, Any] = {"interaction_events": interaction_events}

        if last_event_note:
            template += "{last_event_note}\n"
            props["last_event_note"] = last_event_note

        self.add_section(
            name=BuiltInSection.INTERACTION_HISTORY,
            template=template,
            props=props,
            status=SectionStatus.ACTIVE,
        )

    def _add_empty_history_section(self) -> None:
        self.add_section(
            name=BuiltInSection.INTERACTION_HISTORY,
            template=self._EMPTY_HISTORY,
            status=SectionStatus.PASSIVE,
        )

    def add_interaction_history(
        self,
        events: Sequence[Event],
        staged_events: Sequence[EmittedEvent] = [],
    ) -> PromptBuilder:
        if events:
            interaction_events = self._gather_interaction_events(events, staged_events)
            self._add_history_section(interaction_events=interaction_events)
        else:
            self._add_empty_history_section()

        return self

    def add_interaction_history_in_message_generation(
        self,
        events: Sequence[Event],
        staged_events: Sequence[EmittedEvent] = [],
    ) -> PromptBuilder:
        if events:
            interaction_events = self._gather_interaction_events(events, staged_events)
            last_event_note = self._last_agent_message_note(events)
            self._add_history_section(
                interaction_events=interaction_events, last_event_note=last_event_note
            )
        else:
            self._add_empty_history_section()

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

    def add_staged_tool_events(
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

    def _create_capabilities_string(self, capabilities: Sequence[Capability]) -> str:
        return "\n\n".join(
            [
                f"""
Supported Capability {i}: {capability.title}
{capability.description}
"""
                for i, capability in enumerate(capabilities, start=1)
            ]
        )

    def add_capabilities_for_message_generation(
        self,
        capabilities: Sequence[Capability],
        extra_instructions: list[str] = [],
    ) -> PromptBuilder:
        if capabilities:
            capabilities_string = self._create_capabilities_string(capabilities)
            capabilities_instructions = """
Below are the capabilities available to you as an agent.
You may inform the customer that you can assist them using these capabilities.
If you choose to use any of them, additional details will be provided in your next response.
Always prefer adhering to guidelines, before offering capabilities - only offer capabilities if you have no other instruction that's relevant for the current stage of the interaction.
Be proactive and offer the most relevant capabilities—but only if they are likely to move the conversation forward.
If multiple capabilities are appropriate, aim to present them all to the customer.
If none of the capabilities address the current request of the customer - DO NOT MENTION THEM."""
            if extra_instructions:
                capabilities_instructions += "\n".join(extra_instructions)
            self.add_section(
                name=BuiltInSection.CAPABILITIES,
                template=capabilities_instructions
                + """
###
{capabilities_string}
###
""",
                props={"capabilities_string": capabilities_string},
                status=SectionStatus.ACTIVE,
            )
        else:
            self.add_section(
                name=BuiltInSection.CAPABILITIES,
                template="""
When evaluating guidelines, you may sometimes be given capabilities to assist the customer beyond those dictated through guidelines.
However, in this case, no capabilities relevant to the current state of the conversation were found, besides the ones potentially listed in other sections of this prompt.


""",
                props={},
                status=SectionStatus.ACTIVE,
            )

        return self

    def add_capabilities_for_guideline_matching(
        self,
        capabilities: Sequence[Capability],
    ) -> PromptBuilder:
        if capabilities:
            capabilities_string = self._create_capabilities_string(capabilities)

            self.add_section(
                name=BuiltInSection.CAPABILITIES,
                template="""
The following are the capabilities that you hold as an agent.
They may or may not effect your decision regarding the specified guidelines.
###
{capabilities_string}
###
""",
                props={"capabilities_string": capabilities_string},
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
The following are observations that were deemed relevant to the interaction with the user. Use them to inform your response:
###
{observations_string}
###
""",  # noqa
                props={"observations_string": observations_string},
                status=SectionStatus.ACTIVE,
            )

        return self

    def add_guidelines_for_message_generation(
        self,
        ordinary: Sequence[GuidelineMatch],
        tool_enabled: Mapping[GuidelineMatch, Sequence[ToolId]],
        guideline_representations: dict[GuidelineId, GuidelineInternalRepresentation],
    ) -> PromptBuilder:
        all_matches = [
            match
            for match in chain(ordinary, tool_enabled)
            if guideline_representations[match.guideline.id].action
        ]

        if not all_matches:
            self.add_section(
                name=BuiltInSection.GUIDELINE_DESCRIPTIONS,
                template="""
In formulating your reply, you are normally required to follow a number of behavioral guidelines.
However, in this case, no special behavioral guidelines were provided. Therefore, when generating revisions,
you don't need to specifically double-check if you followed or broke any guidelines.
""",
                status=SectionStatus.PASSIVE,
            )
            return self

        guidelines = []
        agent_intention_guidelines = []
        customer_dependent_guideline_indices = []

        for i, p in enumerate(all_matches, start=1):
            if guideline_representations[p.guideline.id].action:
                if cast(
                    dict[str, bool],
                    p.guideline.metadata.get("customer_dependent_action_data", dict()),
                ).get("is_customer_dependent", False):
                    customer_dependent_guideline_indices.append(i)

                if guideline_representations[p.guideline.id].condition:
                    guideline = f"Guideline #{i}) When {guideline_representations[p.guideline.id].condition}, then {guideline_representations[p.guideline.id].action}"
                else:
                    guideline = (
                        f"Guideline #{i}) {guideline_representations[p.guideline.id].action}"
                    )

                if p.rationale:
                    guideline += f"\n      - Rationale: {p.rationale}"

                if p.guideline.metadata.get("agent_intention_condition"):
                    agent_intention_guidelines.append(guideline)
                else:
                    guidelines.append(guideline)

        guideline_list = "\n".join(guidelines)
        agent_intention_guidelines_list = "\n".join(agent_intention_guidelines)

        guideline_instruction = """
When crafting your reply, you must follow the behavioral guidelines provided below, which have been identified as relevant to the current state of the interaction.
    """
        if agent_intention_guidelines_list:
            guideline_instruction += f"""
Some guidelines are tied to conditions related to you, the agent. These guidelines are considered relevant because it is likely that you intend to produce a message that will trigger the associated condition.
You should only follow these guidelines if you are actually going to produce a message that activates the condition.
- **Guidelines with agent intention condition**:
    {agent_intention_guidelines_list}

    """
        if guideline_list:
            guideline_instruction += f"""

For any other guidelines, do not disregard a guideline because you believe its 'when' condition or rationale does not apply—this filtering has already been handled.

- **Guidelines**:
    {guideline_list}

    """

        if customer_dependent_guideline_indices:
            customer_dependent_guideline_indices_str = ", ".join(
                [str(i) for i in customer_dependent_guideline_indices]
            )
            guideline_instruction += f"""
Important note - some guidelines ({customer_dependent_guideline_indices_str}) may require asking specific questions. Never skip these questions, even if you believe the customer already provided the answer. Instead, ask them to confirm their previous response.
"""
        guideline_instruction += """

You may choose not to follow a guideline only in the following cases:
    - It conflicts with a previous customer request.
    - It is clearly inappropriate given the current context of the conversation.
    - It lacks sufficient context or data to apply reliably.
    - It conflicts with an insight.
    - It depends on an agent intention condition that does not apply in the current situation (as mentioned above)
    - If a guideline offers multiple options (e.g., "do X or Y") and another more specific guideline restricts one of those options (e.g., "don’t do X"), follow both by
        choosing the permitted alternative (i.e., do Y).
In all other situations, you are expected to adhere to the guidelines.
These guidelines have already been pre-filtered based on the interaction's context and other considerations outside your scope.
    """
        self.add_section(
            name=BuiltInSection.GUIDELINE_DESCRIPTIONS,
            template=guideline_instruction,
            props={
                "guideline_list": guideline_list,
                "agent_intention_guidelines_list": agent_intention_guidelines_list,
            },
            status=SectionStatus.ACTIVE,
        )
        return self
