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
from parlant.core.engines.alpha.tool_calling.tool_caller import ToolInsights
from parlant.core.engines.engine_context import (
    EngineContext as _EngineContext,
    Interaction,
    InteractionMessage,
    IterationState,
)
from parlant.core.glossary import Term
from parlant.core.guidelines import Guideline
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


@dataclass(frozen=False)
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
