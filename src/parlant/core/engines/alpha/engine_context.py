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
from dataclasses import dataclass, field
from typing import Any, Optional, TypeAlias

from parlant.core.capabilities import Capability
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.emissions import EmittedEvent
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.rule_match import RuleMatch
from parlant.core.engines.alpha.tool_calling.tool_caller import ToolInsights
from parlant.core.engines.engine_context import (
    EngineContext as _EngineContext,
    Interaction,
    InteractionMessage,
    IterationState,
)
from parlant.core.glossary import Term
from parlant.core.rules import Rule as Guideline
from parlant.core.journeys import Journey, JourneyId
from parlant.core.tools import ToolId

# Interaction / InteractionMessage / IterationState / EngineContext are now the
# shared definitions (parlant.core.engines.engine_context); re-exported here so
# existing imports keep working. The alpha engine owns only its ResponseState.
__all__ = [
    "EngineContext",
    "Interaction",
    "InteractionMessage",
    "IterationState",
    "ResponseState",
]


@dataclass(frozen=False, init=False)
class ResponseState:
    """Used to access and update the state needed for responding properly"""

    context_variables: list[tuple[ContextVariable, ContextVariableValue]]
    glossary_terms: set[Term]
    capabilities: list[Capability]
    iterations: list[IterationState]
    ordinary_guideline_matches: list[GuidelineMatch]
    tool_enabled_guideline_matches: dict[GuidelineMatch, list[ToolId]]
    journeys: list[Journey]
    journey_paths: dict[JourneyId, list[Optional[str]]]
    tool_events: list[EmittedEvent]
    tool_insights: ToolInsights
    prepared_to_respond: bool
    message_events: list[EmittedEvent]
    usable_guidelines: list[Guideline] = field(default_factory=list)
    additional_canned_response_fields: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        context_variables: list[tuple[ContextVariable, ContextVariableValue]],
        glossary_terms: set[Term],
        capabilities: list[Capability],
        iterations: list[IterationState],
        ordinary_guideline_matches: list[GuidelineMatch] | None = None,
        tool_enabled_guideline_matches: dict[GuidelineMatch, list[ToolId]] | None = None,
        journeys: list[Journey] | None = None,
        journey_paths: dict[JourneyId, list[Optional[str]]] | None = None,
        tool_events: list[EmittedEvent] | None = None,
        tool_insights: ToolInsights | None = None,
        prepared_to_respond: bool = False,
        message_events: list[EmittedEvent] | None = None,
        usable_guidelines: list[Guideline] | None = None,
        additional_canned_response_fields: dict[str, Any] | None = None,
        ordinary_rule_matches: list[RuleMatch] | None = None,
        tool_enabled_rule_matches: dict[RuleMatch, list[ToolId]] | None = None,
        usable_rules: list[Guideline] | None = None,
    ) -> None:
        def to_guideline_match(match: GuidelineMatch | RuleMatch) -> GuidelineMatch:
            if isinstance(match, RuleMatch):
                return GuidelineMatch(
                    guideline=match.rule,
                    rationale=match.rationale,
                    metadata=match.metadata,
                )
            return match

        ordinary = ordinary_guideline_matches
        if ordinary is None and ordinary_rule_matches is not None:
            ordinary = [to_guideline_match(m) for m in ordinary_rule_matches]

        tool_enabled = tool_enabled_guideline_matches
        if tool_enabled is None and tool_enabled_rule_matches is not None:
            tool_enabled = {
                to_guideline_match(match): tool_ids
                for match, tool_ids in tool_enabled_rule_matches.items()
            }

        self.context_variables = context_variables
        self.glossary_terms = glossary_terms
        self.capabilities = capabilities
        self.iterations = iterations
        self.ordinary_guideline_matches = ordinary or []
        self.tool_enabled_guideline_matches = tool_enabled or {}
        self.journeys = journeys or []
        self.journey_paths = journey_paths or {}
        self.tool_events = tool_events or []
        self.tool_insights = tool_insights or ToolInsights()
        self.prepared_to_respond = prepared_to_respond
        self.message_events = message_events or []
        self.usable_guidelines = usable_guidelines or usable_rules or []
        self.additional_canned_response_fields = additional_canned_response_fields or {}

    @property
    def ordinary_guidelines(self) -> list[Guideline]:
        return [gp.guideline for gp in self.ordinary_guideline_matches]

    @property
    def tool_enabled_guidelines(self) -> list[Guideline]:
        return [gp.guideline for gp in self.tool_enabled_guideline_matches.keys()]

    @property
    def guidelines(self) -> list[Guideline]:
        return self.ordinary_guidelines + self.tool_enabled_guidelines

    @property
    def all_events(self) -> list[EmittedEvent]:
        return self.tool_events + self.message_events


# The alpha engine sees its own ResponseState typed through context.state.
EngineContext: TypeAlias = _EngineContext[ResponseState]
