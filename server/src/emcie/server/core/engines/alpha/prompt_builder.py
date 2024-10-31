from __future__ import annotations
from enum import Enum, auto
from itertools import chain
import json
from typing import Any, Mapping, Optional, Sequence, cast

from emcie.common.tools import Tool
from emcie.server.core.agents import Agent
from emcie.server.core.common import generate_id
from emcie.server.core.context_variables import ContextVariable, ContextVariableValue
from emcie.server.core.sessions import Event, EventSource, MessageEventData
from emcie.server.core.engines.alpha.guideline_proposition import GuidelineProposition
from emcie.server.core.glossary import Term
from emcie.server.core.engines.alpha.utils import (
    context_variables_to_json,
    emitted_tool_events_to_dicts,
)
from emcie.server.core.emissions import EmittedEvent
from emcie.server.core.tools import ToolId


class BuiltInSection(Enum):
    AGENT_IDENTITY = auto()
    INTERACTION_HISTORY = auto()
    CONTEXT_VARIABLES = auto()
    TERMINOLOGY = auto()
    GUIDELINE_PREDICATES = auto()
    GUIDELINE_PROPOSITIONS = auto()
    TOOLS = auto()
    STAGED_EVENTS = auto()


class SectionStatus(Enum):
    ACTIVE = auto()
    """The section has active information that must be taken into account"""

    PASSIVE = auto()
    """The section is inactive, but may have explicit empty-state inclusion in the prompt"""

    NONE = auto()
    """The section is not included in the prompt in any fashion"""


class PromptBuilder:
    def __init__(self) -> None:
        self._sections: dict[str | BuiltInSection, dict[str, Any]] = {}

    def build(self) -> str:
        section_contents = [s["content"] for s in self._sections.values()]
        prompt = "\n\n".join(section_contents)
        return prompt

    def add_section(
        self,
        content: str,
        name: str | BuiltInSection | None = None,
        title: Optional[str] = None,
        status: Optional[SectionStatus] = None,
    ) -> PromptBuilder:
        while not name:
            candidate = generate_id()

            if candidate not in self._sections:
                name = candidate

        if name in self._sections:
            raise ValueError(f"Section '{name}' was already added")

        self._sections[name] = {
            "content": content.strip(),
            "title": title,
            "status": status,
        }

        return self

    def section_status(self, name: str | BuiltInSection) -> SectionStatus:
        if section := self._sections.get(name):
            return cast(SectionStatus, section["status"])
        else:
            return SectionStatus.NONE

    def add_agent_identity(
        self,
        agent: Agent,
    ) -> PromptBuilder:
        if agent.description:
            self.add_section(
                name=BuiltInSection.AGENT_IDENTITY,
                content=f"""
The following is a description of your identity: ###
{agent.description}
###
""",
                status=SectionStatus.ACTIVE,
            )

        return self

    def add_interaction_history(
        self,
        events: Sequence[Event],
    ) -> PromptBuilder:
        def adapt(e: Event) -> str:
            data = e.data

            if e.kind == "message":
                message_data = cast(MessageEventData, e.data)

                if message_data.get("flagged"):
                    data = {
                        "message": "<N/A>",
                        "censured": True,
                        "reasons": message_data["tags"],
                    }

            source_map: dict[EventSource, str] = {
                "end_user": "user",
                "end_user_ui": "frontend_application",
                "human_agent": "human_operator",
                "human_agent_on_behalf_of_ai_agent": "agent",
                "ai_agent": "agent",
            }

            return json.dumps(
                {
                    "event_kind": e.kind,
                    "source_kind": source_map[e.source],
                    **(
                        {
                            "participant": cast(MessageEventData, e.data)["participant"][
                                "display_name"
                            ]
                        }
                        if e.kind == "message"
                        else {}
                    ),
                    **(
                        {"message": cast(MessageEventData, data)["message"]}
                        if e.kind == "message"
                        else {"data": data}
                    ),
                }
            )

        if events:
            interaction_events = [adapt(e) for e in events if e.kind != "status"]

            self.add_section(
                name=BuiltInSection.INTERACTION_HISTORY,
                content=f"""
The following is a list of events describing a back-and-forth
interaction between you and a user: ###
{interaction_events}
###
""",
                status=SectionStatus.ACTIVE,
            )
        else:
            self.add_section(
                name=BuiltInSection.INTERACTION_HISTORY,
                content="""
You, an AI assistant, are now present in an online session with a user.
An interaction, addressing the user, may or may not be initiated by you now.

Here's how to decide whether to initiate the interaction:
A. If the guidelines below both apply to the context, as well as suggest that you should say something
to the user, then you should indeed initiate the interaction now.
B. Otherwise, if no reason is provided that suggests you should say something to the user,
then you should not initiate the interaction. Produce no reply in this case.
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
                content=f"""
The following is information that you're given about the user and context of the interaction: ###
{context_values}
###
""",
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
                name=BuiltInSection.TERMINOLOGY,
                content=f"""
The following is a glossary of the business. When encountering any of these terms, prioritize the interpretation provided here over any definitions you may already know.
Please be tolerant of possible typos by the user with regards to these terms,
and let the user know if/when you assume they meant a term by their typo: ###
{terms_string}
###
""",  # noqa
                status=SectionStatus.ACTIVE,
            )

        return self

    def add_guideline_predicates(
        self,
        predicates: Sequence[str],
    ) -> PromptBuilder:
        assert predicates

        predicates = "\n".join(f"{i}) {p}" for i, p in enumerate(predicates, start=1))

        self.add_section(
            name=BuiltInSection.GUIDELINE_PREDICATES,
            content=f"""
- Predicate List: ###
{predicates}
###

IMPORTANT: Please note there are exactly {len(predicates)} predicates in the list for you to check.
    """,
            status=SectionStatus.ACTIVE,
        )

        return self

    def add_guideline_propositions(
        self,
        ordinary: Sequence[GuidelineProposition],
        tool_enabled: Mapping[GuidelineProposition, Sequence[ToolId]],
        include_priority: bool = True,
        include_tool_associations: bool = False,
    ) -> PromptBuilder:
        all_propositions = list(chain(ordinary, tool_enabled))

        if all_propositions:
            guidelines = []

            for i, p in enumerate(all_propositions, start=1):
                guideline = (
                    f"{i}) When {p.guideline.content.predicate}, then {p.guideline.content.action}"
                )

                if include_priority:
                    guideline += f"\n    [Priority (1-10): {p.score}; Rationale: {p.rationale}]"

                if include_tool_associations:
                    if p in tool_enabled:
                        service_tool_names = ", ".join(
                            [f"{t_id.service_name}:{t_id.tool_name}" for t_id in tool_enabled[p]]
                        )
                        guideline += f"\n    [Associated Tools: {service_tool_names}]"

                guidelines.append(guideline)

            guideline_list = "\n".join(guidelines)

            section_preface = """
In formulating your reply, you are required to follow these behavioral guidelines,
which are applicable to the latest state of the interaction.
"""

            if include_priority:
                section_preface += """
Each guideline is accompanied by a priority score indicating its significance,
and a rationale explaining why it is applicable.
""".strip()

            if include_tool_associations:
                section_preface += """
Note also that each guideline may be associated with one or more tools that it can utilize to achieve its goal, as needed. If a guideline has associated tool(s), use your judgement, as well as the nature of that guideline and the other guidelines provided, to decide whether any tools should be utilized.
""".strip()  # noqa

            self.add_section(
                name=BuiltInSection.GUIDELINE_PROPOSITIONS,
                content=f"""
{section_preface}

Guidelines: ###
{guideline_list}
###
""",
                status=SectionStatus.ACTIVE,
            )
        else:
            self.add_section(
                name=BuiltInSection.GUIDELINE_PROPOSITIONS,
                content="""
In formulating your reply, you are normally required to follow a number of behavioral guidelines.
However, in this case, no special behavrioal guidelines were provided.
""",
                status=SectionStatus.PASSIVE,
            )

        return self

    def add_tool_definitions(self, tools: Sequence[tuple[ToolId, Tool]]) -> PromptBuilder:
        assert tools

        tool_specs = [
            {
                "name": tool_id.to_string(),
                "description": tool.description,
                "parameters": tool.parameters,
                "required_parameters": tool.required,
            }
            for tool_id, tool in tools
        ]

        self.add_section(
            name=BuiltInSection.TOOLS,
            content=f"""
The following are the tool function definitions: ###
{tool_specs}
###
""",
            status=SectionStatus.ACTIVE,
        )

        return self

    def add_staged_events(
        self,
        events: Sequence[EmittedEvent],
    ) -> PromptBuilder:
        if events:
            # FIXME: The following is a code-smell. We can't assume staged_events
            #        is necessarily only composed of tool events.
            #        Also, emitted_tool_events_to_dict() is an oddball of a function.
            staged_events_as_dict = emitted_tool_events_to_dicts(events)

            self.add_section(
                name=BuiltInSection.STAGED_EVENTS,
                content=f"""
For your information, here are some staged events that have just been emitted,
to assist you with generating your reply message while following the guidelines above: ###
{staged_events_as_dict}
###
""",
                status=SectionStatus.ACTIVE,
            )

        return self
