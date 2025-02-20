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

from abc import abstractmethod
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from parlant.core.agents import Agent
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.customers import Customer
from parlant.core.engines.alpha.tool_caller import ToolInsights
from parlant.core.engines.alpha.guideline_proposition import GuidelineProposition
from parlant.core.glossary import Term
from parlant.core.emissions import EmittedEvent, EventEmitter
from parlant.core.sessions import Event
from parlant.core.tools import ToolId
from parlant.core.nlp.generation_info import GenerationInfo


@dataclass(frozen=True)
class MessageEventComposition:
    generation_info: GenerationInfo
    events: Sequence[Optional[EmittedEvent]]


class MessageCompositionError(Exception):
    def __init__(self, message: str = "Message composition failed") -> None:
        super().__init__(message)


class MessageEventComposer:
    @abstractmethod
    async def generate_events(
        self,
        event_emitter: EventEmitter,
        agent: Agent,
        customer: Customer,
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        ordinary_guideline_propositions: Sequence[GuidelineProposition],
        tool_enabled_guideline_propositions: Mapping[GuidelineProposition, Sequence[ToolId]],
        tool_insights: ToolInsights,
        staged_events: Sequence[EmittedEvent],
    ) -> Sequence[MessageEventComposition]: ...
