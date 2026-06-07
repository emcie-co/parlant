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

from dataclasses import dataclass, field
from typing import Any, Optional, TypeAlias

from parlant.core.capabilities import Capability
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.emissions import EmittedEvent
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.alpha.tool_calling.tool_caller import ToolInsights
from parlant.core.engines.engine_context import EngineContext as _EngineContext
from parlant.core.glossary import Term
from parlant.core.guidelines import Guideline, GuidelineId
from parlant.core.journeys import Journey, JourneyId
from parlant.core.tools import Tool, ToolId


@dataclass(frozen=True)
class IterationState:
    """State of a single iteration in the response process"""

    matched_guidelines: list[GuidelineMatch]
    ruled_out: list[GuidelineMatch]
    resolved_guidelines: list[GuidelineMatch]
    tool_insights: ToolInsights
    executed_tools: list[ToolId]


@dataclass
class ResponseState:
    ordinary_guideline_matches: list[GuidelineMatch] = field(default_factory=list)
    tool_enabled_guideline_matches: dict[GuidelineMatch, list[ToolId]] = field(default_factory=dict)
    # tools the matched guidelines enabled this turn (described in the prompt)
    matched_tools: list[Tool] = field(default_factory=list)
    # all the agent's candidate tools, ranked by relevance to the conversation
    agent_tool_pool: list[Tool] = field(default_factory=list)
    # final catalog offered to the model (matched_tools ∪ top of the pool, capped, by name)
    available_tools: list[Tool] = field(default_factory=list)
    tool_ids_by_name: dict[str, ToolId] = field(default_factory=dict)  # to run a tool by its name

    # Per-turn signals the matcher precomputes (once) so its per-guideline strategy
    # selection can stay synchronous: guidelines that carry tools, and guidelines
    # that participate in a dependency relationship.
    guideline_ids_with_tools: set[GuidelineId] = field(default_factory=set)
    guideline_ids_with_dependencies: set[GuidelineId] = field(default_factory=set)

    # TODO: Remove what isn't needed
    context_variables: list[tuple[ContextVariable, ContextVariableValue]] = field(
        default_factory=list
    )
    glossary_terms: set[Term] = field(default_factory=set)
    capabilities: list[Capability] = field(default_factory=list)
    journeys: list[Journey] = field(default_factory=list)
    journey_paths: dict[JourneyId, list[Optional[str]]] = field(default_factory=dict)
    tool_events: list[EmittedEvent] = field(default_factory=list)
    tool_insights: ToolInsights = field(default_factory=ToolInsights)
    prepared_to_respond: bool = False
    message_events: list[EmittedEvent] = field(default_factory=list)
    usable_guidelines: list[Guideline] = field(default_factory=list)
    additional_canned_response_fields: dict[str, Any] = field(default_factory=dict)
    iterations: list[IterationState] = field(default_factory=list)

    @property
    def ordinary_guidelines(self) -> list[Guideline]:
        return [gp.guideline for gp in self.ordinary_guideline_matches]

    @property
    def tool_enabled_guidelines(self) -> list[Guideline]:
        return [gp.guideline for gp in self.tool_enabled_guideline_matches.keys()]

    @property
    def guidelines(self) -> list[Guideline]:
        return self.ordinary_guidelines + self.tool_enabled_guidelines


# The compass engine sees its own ResponseState typed through context.state.
EngineContext: TypeAlias = _EngineContext[ResponseState]
