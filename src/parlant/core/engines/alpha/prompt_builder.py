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

from __future__ import annotations
from collections import defaultdict
from collections.abc import Set
from dataclasses import dataclass
import dataclasses
from enum import Enum, auto
from io import StringIO
from itertools import chain
import json
from typing import Any, Callable, Generic, Mapping, Optional, Sequence, TypeVar, cast

from pydantic import BaseModel
import pydantic

from parlant.core.agents import Agent
from parlant.core.capabilities import Capability
from parlant.core.common import Weight, JSONSerializable
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.customers import Customer, CustomerStore
from parlant.core.engines.alpha.rule_matching.generic.common import (
    RuleInternalRepresentation,
    internal_representation,
)
from parlant.core.engines.alpha.rule_matching.rule_match import RuleMatch
from parlant.core.sessions import (
    Event,
    EventKind,
    EventSource,
    MessageEventData,
    Session,
    ToolEventData,
)
from parlant.core.glossary import Term
from parlant.core.engines.alpha.utils import (
    context_variables_to_json,
)
from parlant.core.emissions import EmittedEvent
from parlant.core.rules import Rule, RuleId
from parlant.core.tools import Tool, ToolId

_T = TypeVar("_T")


class BuiltInSection(str, Enum):
    @staticmethod
    def _generate_next_value_(name: str, start: int, count: int, last_values: list[str]) -> str:
        return name

    AGENT_IDENTITY = auto()
    CUSTOMER_IDENTITY = auto()
    SESSION_SUMMARY = auto()
    INTERACTION_HISTORY = auto()
    CONTEXT_VARIABLES = auto()
    GLOSSARY = auto()
    RULE_DESCRIPTIONS = auto()
    RULE_INSTRUCTIONS = auto()
    RULE_LIST = auto()
    SYSTEM_WIDE_RULES = auto()
    RULES = auto()
    GUIDELINES = "RULES"
    STAGED_EVENTS = auto()
    JOURNEYS = auto()
    OBSERVATIONS = auto()
    CAPABILITIES = auto()
    TOOL_DESCRIPTIONS = auto()


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


class EventAdaptationFormat(Enum):
    JSON = auto()
    """The event is adapted into a JSON-serializable dict, and then dumped into the prompt as a JSON string"""

    ROLE_SCRIPT = auto()
    """The event is adapted into a script-like format, e.g. for messages: `"{participant}: {message}"`"""


class PromptBuilder:
    def __init__(self, on_build: Optional[Callable[[str], None]] = None) -> None:
        self.sections: dict[str | BuiltInSection, PromptSection] = {}

        self._on_build = on_build
        self._cached_results: set[str] = set()
        self._modified = False

    def _call_on_build(self, prompt: str) -> None:
        if prompt in self._cached_results:
            return

        if self._on_build:
            self._on_build(prompt)

        self._cached_results.add(prompt)

    def _prop_to_dict(self, prop: Any) -> Any:
        class CustomTypeAdapter(pydantic.BaseModel, Generic[_T]):
            obj: _T

            __pydantic_config__ = pydantic.ConfigDict(
                json_encoders={
                    JSONSerializable: lambda v: v,  # type: ignore
                }
            )

        if isinstance(prop, (str, int, float, bool)) or prop is None:
            return prop
        elif isinstance(prop, dict):
            return {k: self._prop_to_dict(v) for k, v in prop.items()}
        elif isinstance(prop, list):
            return [self._prop_to_dict(i) for i in prop]
        elif isinstance(prop, tuple):
            return tuple(self._prop_to_dict(i) for i in prop)
        elif dataclasses.is_dataclass(prop):
            return CustomTypeAdapter(obj=prop).model_dump(mode="json")["obj"]
        elif isinstance(prop, BaseModel):
            return prop.model_dump(mode="json")
        elif isinstance(prop, Enum):
            return prop.value
        else:
            raise ValueError(f"Unsupported prop type: {type(prop)}")

    @property
    def props(self, keys: list[str] | None = None) -> dict[str, dict[str, Any]]:
        result = {
            section_name if isinstance(section_name, str) else f"__{section_name.name}__": {
                k: self._prop_to_dict(v)
                for k, v in section.props.items()
                if keys is None or k in keys
            }
            for section_name, section in self.sections.items()
        }
        result["metadata"] = {"modified": self._modified}
        return result

    def build(self) -> str:
        buffer = StringIO()

        for section_name, section in self.sections.items():
            try:
                buffer.write(section.template.format(**section.props))
                buffer.write("\n\n")
            except Exception as e:
                raise ValueError(
                    f"Error formatting section {section_name} with template: {section.template} and props: {section.props}"
                ) from e

        prompt = buffer.getvalue().strip()

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
        self._modified = True
        return self

    def section_status(self, name: str | BuiltInSection) -> SectionStatus:
        if name in self.sections and self.sections[name].status is not None:
            return cast(SectionStatus, self.sections[name].status)
        else:
            return SectionStatus.NONE

    @staticmethod
    def adapt_event(
        e: Event | EmittedEvent,
        format: EventAdaptationFormat = EventAdaptationFormat.JSON,
    ) -> str:
        adapted_data: dict[str, str] = {}

        if e.kind == EventKind.MESSAGE:
            message_data = cast(MessageEventData, e.data)

            if message_data.get("flagged"):
                adapted_data = {
                    "participant": message_data["participant"]["display_name"],
                    "message": "<N/A>",
                    "censored": "Yes",
                    "reasons": str(message_data["groups"]),
                }
            else:
                adapted_data = {
                    "participant": message_data["participant"]["display_name"],
                    "message": message_data["message"],
                }

        if e.kind == EventKind.TOOL:
            tool_data = cast(ToolEventData, e.data)

            adapted_data = {
                "tool_calls": str(
                    [
                        {
                            "tool_id": tc["tool_id"],
                            "arguments": tc["arguments"],
                            "result": tc["result"]["data"],
                        }
                        for tc in tool_data["tool_calls"]
                    ]
                )
            }

        if format == EventAdaptationFormat.JSON:
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
                    "data": adapted_data,
                }
            )
        else:
            if e.kind == EventKind.TOOL:
                return f"Tool Calls: {adapted_data['tool_calls']}"

            source_map = {
                EventSource.CUSTOMER: f"User ({adapted_data['participant']})",
                EventSource.CUSTOMER_UI: "User (Sent via Frontend App)",
                EventSource.HUMAN_AGENT: f"Human Representative ({adapted_data['participant']})",
                EventSource.HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT: "Agent",
                EventSource.AI_AGENT: "Agent",
                EventSource.SYSTEM: "System",
            }

            if e.kind == EventKind.MESSAGE:
                return f"{source_map[e.source]}: {adapted_data['message']}"
            elif e.kind == EventKind.CUSTOM:
                return f"{source_map[e.source]}: {e.data}"
            else:
                raise ValueError(f"Unsupported event kind for adaptation: {e.kind}")

    def add_agent_identity(
        self,
        agent: Agent,
    ) -> PromptBuilder:
        if agent.description:
            self.add_section(
                name=BuiltInSection.AGENT_IDENTITY,
                template="""
## You are an AI agent named {agent_name}.

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
        session: Session,
    ) -> PromptBuilder:
        if customer.id == CustomerStore.GUEST_ID:
            self.add_section(
                name=BuiltInSection.CUSTOMER_IDENTITY,
                template="""\
## The user you're interacting with is not authenticated
We do not yet know their identity
    """,
                props={
                    "customer_name": customer.name,
                    "session_id": session.id,
                },
                status=SectionStatus.ACTIVE,
            )
        else:
            self.add_section(
                name=BuiltInSection.CUSTOMER_IDENTITY,
                template="""\
## The user you're interacting with is called {customer_name}.
""",
                props={
                    "customer_name": customer.name,
                    "session_id": session.id,
                },
                status=SectionStatus.ACTIVE,
            )

        return self

    def add_session_summary(
        self,
        summary: str,
    ) -> PromptBuilder:
        if not summary.strip():
            return self

        self.add_section(
            name=BuiltInSection.SESSION_SUMMARY,
            template="""\
# Session Summary

The earlier part of this session was compacted into the following summary.
Treat it as factual background context for the current interaction. It is not a new
message from the user and should not be acknowledged directly:

###
{session_summary}
###
""",
            props={
                "session_summary": summary.strip(),
            },
            status=SectionStatus.ACTIVE,
        )

        return self

    INTERACTION_HISTORY_HEADER = "# Interaction History"

    _INTERACTION_BODY_JSON = f"""\
{INTERACTION_HISTORY_HEADER}

The following is a list of events describing the most recent state of the back-and-forth
interaction between you and a user: ###
{{interaction_events}}
###
"""

    _INTERACTION_BODY_SCRIPT = f"""\
{INTERACTION_HISTORY_HEADER}

The following is a list of events describing the most recent state of the back-and-forth
interaction between you and a user:
{{interaction_events}}
"""

    _EMPTY_HISTORY = f"""
{INTERACTION_HISTORY_HEADER}

Your interaction with the user has just began, and no events have been recorded yet.
Proceed with your task accordingly.
"""

    def _gather_interaction_events(
        self,
        events: Sequence[Event],
        staged_events: Sequence[EmittedEvent],
        format: EventAdaptationFormat,
    ) -> list[str]:
        combined = list(events) + list(staged_events)
        return [self.adapt_event(e, format=format) for e in combined if e.kind != EventKind.STATUS]

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
        format: EventAdaptationFormat = EventAdaptationFormat.JSON,
    ) -> None:
        if format == EventAdaptationFormat.JSON:
            template = self._INTERACTION_BODY_JSON
            props = {"interaction_events": "\n".join(interaction_events)}
        else:
            template = self._INTERACTION_BODY_SCRIPT
            props = {"interaction_events": "\n".join(interaction_events)}

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
        format: EventAdaptationFormat = EventAdaptationFormat.JSON,
    ) -> PromptBuilder:
        if events:
            interaction_events = self._gather_interaction_events(events, staged_events, format)
            self._add_history_section(interaction_events=interaction_events, format=format)
        else:
            self._add_empty_history_section()

        return self

    def add_interaction_history_for_message_generation(
        self,
        events: Sequence[Event],
        staged_events: Sequence[EmittedEvent] = [],
        format: EventAdaptationFormat = EventAdaptationFormat.JSON,
    ) -> PromptBuilder:
        if events:
            interaction_events = self._gather_interaction_events(events, staged_events, format)
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
            # Callers pass set-derived lists, so sort to keep the rendered section
            # deterministic — it lives in cached prompt prefixes.
            terms = sorted(terms, key=lambda t: (t.name, t.id))
            terms_string = ""

            for t in terms:
                terms_string += f"### {t.name}\n\n"

                if t.synonyms:
                    terms_string += f"Synonyms: {', '.join(t.synonyms)}\n\n"

                terms_string += f"{t.description}\n\n"

            self.add_section(
                name=BuiltInSection.GLOSSARY,
                template="""
## DOMAIN GLOSSARY

The following is a glossary of our domain.

Understanding these terms, as they apply to the domain, is critical for your task.
When encountering any of these terms, prioritize the interpretation provided here over any definitions you may already know.

Please be tolerant of possible typos by the user with regards to these terms,
and let the user know if/when you assume they meant a term by their typo:

{terms_string}
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
STAGED EVENTS
-------------
Here are the most recent staged events for your reference.
They represent interactions with external tools that perform actions or provide information.
Prioritize their data over any other sources and use their details to complete your task: ###
{staged_events_as_dict}
###
""",
                props={"staged_events_as_dict": staged_events_as_dict or "[None]"},
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
Always prefer adhering to rules, before offering capabilities - only offer capabilities if you have no other instruction that's relevant for the current stage of the interaction.
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

        return self

    def add_capabilities_for_rule_matching(
        self,
        capabilities: Sequence[Capability],
    ) -> PromptBuilder:
        if capabilities:
            capabilities_string = self._create_capabilities_string(capabilities)

            self.add_section(
                name=BuiltInSection.CAPABILITIES,
                template="""
The following are the capabilities that you hold as an agent.
They may or may not effect your decision regarding the specified rules.
All relevant capabilities you have as an agent should be listed here, if you are asked to perform something that is not listed here, assume you cannot do so (that behavior is not supported)
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
        observations: Sequence[Rule],
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

    def add_capabilities_for_guideline_matching(
        self,
        capabilities: Sequence[Capability],
    ) -> PromptBuilder:
        return self.add_capabilities_for_rule_matching(capabilities)

    def add_rules_for_message_generation(
        self,
        ordinary: Sequence[RuleMatch],
        tool_enabled: Mapping[RuleMatch, Sequence[ToolId]],
        rule_representations: dict[RuleId, RuleInternalRepresentation],
    ) -> PromptBuilder:
        all_matches = [
            match
            for match in chain(ordinary, tool_enabled)
            if rule_representations[match.rule.id].action and not match.rule.weight == Weight.LOW
        ]

        if not all_matches:
            self.add_section(
                name=BuiltInSection.RULE_DESCRIPTIONS,
                template="""
In formulating your reply, you are normally required to follow a number of behavioral rules.
However, in this case, no special behavioral rules were provided. Therefore, when generating revisions,
you don't need to specifically double-check if you followed or broke any rules.
""",
                status=SectionStatus.PASSIVE,
            )
            return self

        rules = []
        agent_intention_rules = []
        customer_dependent_rule_indices = []

        for i, p in enumerate(all_matches, start=1):
            if rule_representations[p.rule.id].action:
                if cast(
                    dict[str, bool],
                    p.rule.metadata.get("customer_dependent_action_data", dict()),
                ).get("is_customer_dependent", False):
                    customer_dependent_rule_indices.append(i)

                if rule_representations[p.rule.id].condition:
                    rule = f"Rule #{i}) When {rule_representations[p.rule.id].condition}, then {rule_representations[p.rule.id].action}"
                else:
                    rule = f"Rule #{i}) {rule_representations[p.rule.id].action}"

                if rule_representations[p.rule.id].description:
                    rule += f"\n      - Description: {rule_representations[p.rule.id].description}"

                if p.rationale:
                    rule += f"\n      - Rationale: {p.rationale}"

                if p.rule.metadata.get("agent_intention_condition"):
                    agent_intention_rules.append(rule)
                else:
                    rules.append(rule)

        rule_list = "\n".join(rules)
        agent_intention_rules_list = "\n".join(agent_intention_rules)

        rule_instruction = """
## EXTREMELY IMPORTANT - RULES YOU MUST FOLLOW:

When crafting your reply, you must follow the behavioral rules provided below, which have been identified as relevant to the current state of the interaction.
    """
        if agent_intention_rules_list:
            rule_instruction += f"""
Some rules are tied to conditions related to you, the agent. These rules are considered relevant because it is likely that you intend to produce a message that will trigger the associated condition.
You should only follow these rules if you are actually going to produce a message that activates the condition.
- **Rules with agent intention condition**:
    {agent_intention_rules_list}

    """
        if rule_list:
            rule_instruction += f"""

For any other rules, do not disregard a rule because you believe its 'when' condition or rationale does not apply—this filtering has already been handled.

- **Rules**:
    {rule_list}

    """

        if customer_dependent_rule_indices:
            customer_dependent_rule_indices_str = ", ".join(
                [str(i) for i in customer_dependent_rule_indices]
            )
            rule_instruction += """
Important note - some rules ({customer_dependent_rule_indices_str}) may require asking specific questions. Never skip these questions, even if you believe the customer already provided the answer. Instead, ask them to confirm their previous response.
"""
        else:
            customer_dependent_rule_indices_str = ""

        rule_instruction += """
You may choose not to follow a rule only in the following cases:
    - It conflicts with a previous customer request.
    - It is clearly inappropriate given the current context of the conversation.
    - It lacks sufficient context or data to apply reliably.
    - It conflicts with an insight.
    - It depends on an agent intention condition that does not apply in the current situation (as mentioned above)
    - If a rule offers multiple options (e.g., "do X or Y") and another more specific rule restricts one of those options (e.g., "don’t do X"), follow both by
        choosing the permitted alternative (i.e., do Y).
In all other situations, you are expected to adhere to the rules.
These rules have already been pre-filtered based on the interaction's context and other considerations outside your scope.
    """
        self.add_section(
            name=BuiltInSection.RULE_DESCRIPTIONS,
            template=rule_instruction,
            props={
                "rule_list": rule_list,
                "agent_intention_rules_list": agent_intention_rules_list,
                "customer_dependent_rule_indices_str": customer_dependent_rule_indices_str,
            },
            status=SectionStatus.ACTIVE,
        )
        return self

    def add_rule_instructions(self) -> PromptBuilder:
        """The *explanation* half of ``add_rules_for_message_generation``:
        how and when to follow behavioral rules, without listing any. The
        matched rules themselves are added separately via
        :meth:`add_matched_rules`. Kept stable (no per-turn data) so it can
        live in cached system-level instructions."""
        self.add_section(
            name=BuiltInSection.RULE_INSTRUCTIONS,
            template="""
# DOMAIN INSTRUCTIONS

When crafting your reply, follow the behavioral instructions (provided toward the end of the prompt) to the extent that they are (still) relevant to the current state of the interaction. The instructions to keep in mind, if any, will be provided to you in a separate instruction later in the conversation.

Some instructions are tied to conditions related to you, the agent (e.g., "When you are likely/about to do something"). You should only follow these instructions if you are actually going to produce a message that activates the condition.

Some instructions may require asking specific questions and getting clear answers to them from the user. Never skip these questions, even if you believe the user already provided the answer. Instead, ask them to confirm their previous response.

You may choose not to follow an instruction only in the following cases:
    - You have already followed the instruction and the context of its application doesn't merit following it again (i.e., it would be purely repetitive).
    - It conflicts with a previous user request.
    - It is clearly inappropriate given the current context of the conversation.
    - It lacks sufficient context or data to apply reliably.
    - It conflicts with an insight.
    - It depends on an agent intention condition that does not apply in the current situation (as mentioned above)
    - If an instruction offers multiple options (e.g., "do X or Y") and another more specific instruction restricts one of those options (e.g., "don’t do X"), follow both by
        choosing the permitted alternative (i.e., do Y).

In all other situations, you are expected to follow the instructions when and as appropriate.
""",
            status=SectionStatus.ACTIVE,
        )
        return self

    def add_system_wide_rules(
        self,
        rules: Sequence[Rule],
        tools_by_rule: Mapping[RuleId, Set[tuple[ToolId, Tool]]],
    ) -> PromptBuilder:
        # Titled instructions first; within each group ordered by id, so the rendered
        # list is deterministic (cache-stable) and the cached prefix stays byte-stable.
        listed = sorted(
            rules,
            key=lambda g: (0 if g.title else 1, g.id),
        )

        if not listed:
            return self

        instruction_texts = []

        for i, g in enumerate(listed, start=1):
            if g.title:
                text = f"### Instruction #{i}: {g.title}\n\n"
            elif g.content.condition:
                if g.content.action:
                    text = f"### Instruction #{i}: When {g.content.condition}, then {g.content.action}\n\n"
                elif g.content.description or (g.id in tools_by_rule):
                    text = f"### Instruction #{i}: When {g.content.condition}...\n\n"
                elif g.id not in tools_by_rule:
                    # Pure condition rule - probably used for relationships
                    # and not meant to be included in the prompt. Skip it.
                    continue
            else:
                text = f"### Instruction #{i}: {g.content.action}\n\n"

            if g.content.description:
                text += f"{g.content.description.strip()}"
            elif g.id not in tools_by_rule:
                text += "\nDetails:\n[None]"
            else:
                text += "\nConsider using the associated tool(s)."

            if tools_by_rule.get(g.id):
                tool_names = ", ".join(t.name for _, t in tools_by_rule[g.id])
                text += f"\nAssociated tool(s): {tool_names}"

            if g.weight == Weight.HIGH:
                text += "IMPORTANT: This one is a high-criticality instruction. Pay extra attention to its details and do not violate it, even if the user asks you to.\n\n"

            instruction_texts.append(text)

        self.add_section(
            name=BuiltInSection.SYSTEM_WIDE_RULES,
            template="""
# DOMAIN INSTRUCTIONS

The following are all the behavioral instructions and policies that govern your conduct in this domain. Keep every one of them in mind as the conversation progresses.

{instruction_list}
""",
            props={"instruction_list": "\n\n".join(instruction_texts)},
            status=SectionStatus.ACTIVE,
        )
        return self

    def _is_short_tool_related_condition(self, rule: Rule) -> bool:
        return bool(
            rule.content.condition
            and (len(rule.content.condition) <= 100)
            and not rule.content.action
            and not rule.content.description
        )

    def add_matched_rules(
        self,
        ordinary: Sequence[RuleMatch],
        tool_enabled: Mapping[RuleMatch, Sequence[ToolId]],
        rules: dict[RuleId, Rule],
    ) -> PromptBuilder:
        all_matches = [
            match
            for match in chain(ordinary, tool_enabled)
            if (
                rules[match.rule.id].content.action
                or rules[match.rule.id].content.description
                or match.metadata.get("highlights")
            )
            and not match.rule.weight == Weight.LOW
        ]
        # Titled instructions first, preserving the original order within each group.
        all_matches = sorted(all_matches, key=lambda m: 0 if m.rule.title else 1)

        if not all_matches:
            return self

        rule_texts = []

        for i, p in enumerate(all_matches, start=1):
            # tool_ids = tool_enabled.get(p, [])
            _ = i
            if self._is_short_tool_related_condition(rules[p.rule.id]):
                # If the rule is short and only has a condition, and is only relevant due to tool availability, we can assume it's meant to be a short instruction related to the tool, and we can save space in the prompt by putting it in the tool description section instead of the rule list section.
                continue

            if rules[p.rule.id].title:
                rule = f'### Review the instructions under "{rules[p.rule.id].title}"'
            else:
                if rules[p.rule.id].content.condition and rules[p.rule.id].content.action:
                    rule = f"### Remember, when {rules[p.rule.id].content.condition}, then {rules[p.rule.id].content.action}"
                elif rules[p.rule.id].content.condition:
                    rule = f"### Remember, when {rules[p.rule.id].content.condition}"
                else:
                    rule = f"### Remember: {rules[p.rule.id].content.action}"

            if rules[p.rule.id].weight == Weight.HIGH:
                if highlight_points := "\n".join(
                    f"- {point.strip()}"
                    for point in cast(Sequence[str], p.metadata.get("highlights", []))
                ):
                    rule += f"\nHighlights:\n{highlight_points}"
                    rule += "\nIMPORTANT: Please go back and reason (internally) about the original content of this instruction to the letter before proceeding."
                elif description := rules[p.rule.id].content.description:
                    rule += f"\n{description.strip()}"
            elif rules[p.rule.id].weight == Weight.MEDIUM:
                if highlight_points := "\n".join(
                    f"- {point.strip()}"
                    for point in cast(Sequence[str], p.metadata.get("highlights", []))
                ):
                    rule += f"\nHighlights:\n{highlight_points}"
                elif description := rules[p.rule.id].content.description:
                    rule += "\nPlease review the original content of this instruction before proceeding."

            # TODO: Consider whether we need the rationale.
            # Since it's a step-based reminder, the rationale might be
            # overly focused on the rule itself without considering
            # other rules or the broader context...
            #
            # if p.rationale:
            #    rule += f"\n(Note: {p.rationale})"

            # TODO: Consider whether we need to mention the associated tools.
            # It may cause the model to overly obsess over running them.
            #
            # if tool_ids:
            #    tool_names = ", ".join(tool_id.tool_name for tool_id in tool_ids)
            #    rule += f"\nAssociated tool(s): {tool_names}"

            rule_texts.append(rule)

        rule_list = "\n\n".join(rule_texts)

        rule_block = """\
This is a gentle reminder to review your instructions again, particularly with respect to the following instructions.
"""

        if rule_list:
            rule_block += """
## Instruction reminders

{rule_list}
"""

        self.add_section(
            name=BuiltInSection.RULE_LIST,
            template=rule_block,
            props={
                "rule_list": rule_list,
            },
            status=SectionStatus.ACTIVE,
        )
        return self

    def add_low_criticality_rule_instructions(self, rules: Sequence[Rule]) -> PromptBuilder:
        if rules:
            # Callers pass set-derived lists, so sort (same key as
            # add_system_wide_rules) to keep the rendered section
            # deterministic — it lives in the responder's cached system prefix.
            rules = sorted(rules, key=lambda g: (0 if g.title else 1, g.id))
            texts = []
            for rule in rules:
                if rule.content.condition:
                    text = f"### When {rule.content.condition}, then {rule.content.action}"
                else:
                    text = f"### {rule.content.action}"
                if rule.content.description:
                    text += f"\nDetails:\n{rule.content.description.strip()}"
                texts.append(text)
            rule_list = "\n\n".join(texts)

            self.add_section(
                name="matched-low-criticality-rules",
                template=f"""
## DOMAIN PRINCIPLES
The following are "principles" - these are instructions that are considered less critical than the main rules, but that still may be relevant to keep in mind as general principles of behavior in this domain. They are not meant to be followed as strictly as the main rules, but they still represent important considerations to keep in mind when generating your response.

{rule_list}
""",
                status=SectionStatus.ACTIVE,
            )
        return self

    def add_tool_descriptions(
        self,
        tools: Mapping[ToolId, Tool],
        rules: Mapping[RuleMatch, Sequence[ToolId]],
    ) -> PromptBuilder:
        any_consequential = False
        consequential_note = "If a tool has a significant, real-world effect, it will be marked with CONSEQUENTIAL. In that case, be careful before running it. Read its description carefully and, when appropriate, confirm with the user before going ahead and performing its action."

        if not tools:
            return self

        rules_per_tools: dict[ToolId, set[Rule]] = defaultdict(set)

        for match, tool_ids in rules.items():
            for tool_id in tool_ids:
                if self._is_short_tool_related_condition(match.rule):
                    rules_per_tools[tool_id].add(match.rule)

        tools_for_high_criticality_rules: set[ToolId] = set()

        for match, match_tools in rules.items():
            if match.rule.weight == Weight.HIGH:
                tools_for_high_criticality_rules.update(match_tools)

        high_criticality_or_consequential_tools: set[ToolId] = set()

        for tool_id, tool in tools.items():
            if tool.consequential:
                high_criticality_or_consequential_tools.add(tool_id)
            elif tool_id in tools_for_high_criticality_rules:
                high_criticality_or_consequential_tools.add(tool_id)
            else:
                pass  # Do not include this tool in the reminder

        tool_lines = []

        for tool_id, tool in tools.items():
            if tool_id not in high_criticality_or_consequential_tools:
                continue

            if tool.consequential:
                any_consequential = True
                line = f"- {tool.name} - CONSEQUENTIAL\n"
                line += f"[Tool description reminder]: {tool.description}\n"
            else:
                line = f"- {tool.name}"

            if associated_rules := rules_per_tools.get(tool_id, set()):
                line += " (Potential relevance: "
                line += "; or ".join(f"when {g.content.condition}" for g in associated_rules)
                line += ")"

            tool_lines.append(line)

        preface = """
## TOOL REFRESHER

IMPORTANT: When running tools, first ask yourself *very carefully* in your preliminary reasoning whether it is appropriate to call **multiple tools** if there's more than one resource to check or act on, or issue **multiple calls to the same tool** - for example, if there are *different contexts* and arguments that need to be managed across different calls. This is especially important to *respect user intent when handling sensitive operations across multiple contexts* simultaneously.
"""

        if tool_lines:
            self.add_section(
                name=BuiltInSection.TOOL_DESCRIPTIONS,
                template="""\
{preface}

{consequential_note}

For this turn of the interaction, some tools have been identified as relevant to remind you about. You are not required to use any of them; use a tool only when it is actually useful for the current response. You MAY also use any other tools that you are aware of in processing your current response, other than the ones listed below, if appropriate under certain corner cases.

### Tools

{tool_list}
""",
                props={
                    "preface": preface,
                    "tool_list": "\n\n".join(tool_lines),
                    "consequential_note": consequential_note if any_consequential else "",
                },
                status=SectionStatus.ACTIVE,
            )
        else:
            self.add_section(
                name=BuiltInSection.TOOL_DESCRIPTIONS,
                template="""\
{preface}
""",
                props={
                    "preface": preface,
                },
                status=SectionStatus.ACTIVE,
            )

        return self

    def add_low_criticality_rules(
        self,
        ordinary: Sequence[RuleMatch],
        tool_enabled: Mapping[RuleMatch, Sequence[ToolId]],
        rule_representations: dict[RuleId, RuleInternalRepresentation],
    ) -> PromptBuilder:
        all_matches = [
            match
            for match in chain(ordinary, tool_enabled)
            if rule_representations[match.rule.id].action
        ]
        low_critical_matches = [m for m in all_matches if m.rule.weight == Weight.LOW]
        if low_critical_matches:
            low_criticality_rules = []
            for p in low_critical_matches:
                if rule_representations[p.rule.id].condition:
                    rule = f" - When {rule_representations[p.rule.id].condition}, then {rule_representations[p.rule.id].action}"
                else:
                    rule = f" - When always, then {rule_representations[p.rule.id].action}"
                low_criticality_rules.append(rule)
            rule_list = "\n".join(low_criticality_rules)
            template = f"""
When generating a response, consider the following general principles:
{rule_list}
Note that you may ignore a principle if it is not relevant to the specific context or if you find it inappropriate.
Later in this prompt, you will be provided with rules that have been detected as specifically relevant to the current context and that you must follow. Prioritize those context-specific over these general principles.
"""
            self.add_section(
                name="low-criticality-rules",
                template=template,
                status=SectionStatus.ACTIVE,
            )
        return self

    def add_rules_for_canrep_selection(self, rule_matches: Sequence[RuleMatch]) -> PromptBuilder:
        matches = [
            m
            for m in rule_matches
            if internal_representation(m.rule).action and not m.rule.weight == Weight.LOW
        ]
        rule_representations = {m.rule.id: internal_representation(m.rule) for m in matches}

        if matches:
            formatted_rules = "In choosing the template, there are 2 cases. 1) There is a single, clear match. 2) There are multiple candidates for a match. In the second case, you may also find that there are multiple templates that overlap with the draft message in different ways. In those cases, you will have to decide which part (which overlap) you prioritize. When doing so, your prioritization for choosing between different overlapping templates should try to maximize adherence to the following behavioral rules: \n ###\n"

            for match in [g for g in matches if internal_representation(g.rule).action]:
                formatted_rules += f"\n- When {rule_representations[match.rule.id].condition}, then {rule_representations[match.rule.id].action}."

            formatted_rules += "\n###"
        else:
            formatted_rules = ""
        self.add_section(
            name=BuiltInSection.RULE_DESCRIPTIONS,
            template=formatted_rules,
            status=SectionStatus.ACTIVE,
        )
        return self
