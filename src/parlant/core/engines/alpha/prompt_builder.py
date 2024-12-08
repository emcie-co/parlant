# Copyright 2024 Emcie Co Ltd.
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
from enum import Enum, auto
import json
from typing import Any, Optional, Sequence, cast

from parlant.core.tools import Tool
from parlant.core.agents import Agent
from parlant.core.common import generate_id
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.sessions import Event, EventSource, MessageEventData
from parlant.core.glossary import Term
from parlant.core.engines.alpha.utils import (
    context_variables_to_json,
    emitted_tool_events_to_dicts,
)
from parlant.core.emissions import EmittedEvent
from parlant.core.tools import ToolId


class BuiltInSection(Enum):
    AGENT_IDENTITY = auto()
    INTERACTION_HISTORY = auto()
    CONTEXT_VARIABLES = auto()
    GLOSSARY = auto()
    GUIDELINE_DESCRIPTIONS = auto()
    GUIDELINES = auto()
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
You are {agent.name}, an AI agent.

The following is a description of your background and personality: ###
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

            source_map: dict[EventSource, str] = {
                "customer": "user",
                "customer_ui": "frontend_application",
                "human_agent": "human_service_agent",
                "human_agent_on_behalf_of_ai_agent": "ai_agent",
                "ai_agent": "ai_agent",
                "system": "system-provided",
            }

            return json.dumps(
                {
                    "event_kind": e.kind,
                    "event_source": source_map[e.source],
                    "data": data,
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
                name=BuiltInSection.GLOSSARY,
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
The following are the tool function definitions. Generate one reply for each tool.
IMPORTANT: You must not return results for any tool that do not appear in the following list, even if you believe they might be relevant.
: ###
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
Here are some recently emitted events for your consideration.
These events represent calls to external tools that perform real-world actions or provide useful information.
Use the details they offer to assist in your task: ###
{staged_events_as_dict}
###
""",
                status=SectionStatus.ACTIVE,
            )

        return self
