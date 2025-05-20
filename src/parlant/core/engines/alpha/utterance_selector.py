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

from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass
from datetime import datetime
from itertools import chain
import re
import jinja2
import jinja2.meta
import json
import traceback
from typing import Any, Mapping, Optional, Sequence, cast
from typing_extensions import override

from parlant.core.async_utils import safe_gather
from parlant.core.contextual_correlator import ContextualCorrelator
from parlant.core.agents import Agent, CompositionMode
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.customers import Customer
from parlant.core.engines.alpha.message_event_composer import (
    MessageCompositionError,
    MessageEventComposer,
    MessageEventComposition,
)
from parlant.core.engines.alpha.message_generator import MessageGenerator
from parlant.core.engines.alpha.tool_calling.tool_caller import ToolInsights
from parlant.core.journeys import Journey
from parlant.core.utterances import Utterance, UtteranceId, UtteranceStore
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.alpha.prompt_builder import PromptBuilder, BuiltInSection, SectionStatus
from parlant.core.glossary import Term
from parlant.core.emissions import EmittedEvent, EventEmitter
from parlant.core.sessions import (
    Event,
    EventKind,
    EventSource,
    MessageEventData,
    Participant,
    ToolCall,
    ToolEventData,
)
from parlant.core.common import CancellationSuppressionLatch, DefaultBaseModel, JSONSerializable
from parlant.core.loggers import Logger
from parlant.core.shots import Shot, ShotCollection
from parlant.core.tools import ToolId

DEFAULT_NO_MATCH_UTTERANCE = "Not sure I understand. Could you please say that another way?"


class UtteranceDraftSchema(DefaultBaseModel):
    last_message_of_user: Optional[str]
    guidelines: list[str]
    insights: Optional[list[str]] = None
    message: Optional[str] = None


class UtteranceSelectionSchema(DefaultBaseModel):
    rationale: Optional[str] = None
    chosen_template_id: Optional[str] = None
    match_quality: Optional[str] = None


class UtteranceRevisionSchema(DefaultBaseModel):
    revised_utterance: str


@dataclass
class UtteranceSelectorDraftShot(Shot):
    composition_modes: list[CompositionMode]
    expected_result: UtteranceDraftSchema


@dataclass(frozen=True)
class _UtteranceSelectionResult:
    @staticmethod
    def no_match(draft: Optional[str] = None) -> _UtteranceSelectionResult:
        return _UtteranceSelectionResult(
            message=DEFAULT_NO_MATCH_UTTERANCE,
            draft=draft or "N/A",
            utterances=[],
        )

    message: str
    draft: str
    utterances: list[tuple[UtteranceId, str]]


@dataclass(frozen=True)
class UtteranceContext:
    agent: Agent
    customer: Customer
    context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]]
    interaction_history: Sequence[Event]
    terms: Sequence[Term]
    ordinary_guideline_matches: Sequence[GuidelineMatch]
    tool_enabled_guideline_matches: Mapping[GuidelineMatch, Sequence[ToolId]]
    journeys: Sequence[Journey]
    tool_insights: ToolInsights
    staged_events: Sequence[EmittedEvent]

    @property
    def guidelines(self) -> Sequence[GuidelineMatch]:
        return [*self.ordinary_guideline_matches, *self.tool_enabled_guideline_matches.keys()]


class UtteranceFieldExtractionMethod(ABC):
    @abstractmethod
    async def extract(
        self,
        utterance: str,
        field_name: str,
        context: UtteranceContext,
    ) -> tuple[bool, JSONSerializable]: ...


class StandardFieldExtraction(UtteranceFieldExtractionMethod):
    def __init__(self, logger: Logger) -> None:
        self._logger = logger

    @override
    async def extract(
        self,
        utterance: str,
        field_name: str,
        context: UtteranceContext,
    ) -> tuple[bool, JSONSerializable]:
        if field_name != "std":
            return False, None

        return True, {
            "customer": {"name": context.customer.name},
            "agent": {"name": context.agent.name},
            "variables": {
                variable.name: value.data for variable, value in context.context_variables
            },
            "missing_params": self._extract_missing_params(context.tool_insights),
            "invalid_params": self._extract_invalid_params(context.tool_insights),
        }

    def _extract_missing_params(
        self,
        tool_insights: ToolInsights,
    ) -> list[str]:
        return [missing_data.parameter for missing_data in tool_insights.missing_data]

    def _extract_invalid_params(
        self,
        tool_insights: ToolInsights,
    ) -> dict[str, str]:
        return {
            invalid_data.parameter: invalid_data.invalid_value
            for invalid_data in tool_insights.invalid_data
        }


class ToolBasedFieldExtraction(UtteranceFieldExtractionMethod):
    @override
    async def extract(
        self,
        utterance: str,
        field_name: str,
        context: UtteranceContext,
    ) -> tuple[bool, JSONSerializable]:
        tool_calls_in_order_of_importance: list[ToolCall] = []

        tool_calls_in_order_of_importance.extend(
            tc
            for e in context.staged_events
            if e.kind == EventKind.TOOL
            for tc in cast(ToolEventData, e.data)["tool_calls"]
        )

        tool_calls_in_order_of_importance.extend(
            tc
            for e in reversed(context.interaction_history)
            if e.kind == EventKind.TOOL
            for tc in cast(ToolEventData, e.data)["tool_calls"]
        )

        for tool_call in tool_calls_in_order_of_importance:
            if value := tool_call["result"]["utterance_fields"].get(field_name, None):
                return True, value

        return False, None


class UtteranceFieldExtractionSchema(DefaultBaseModel):
    field_name: Optional[str] = None
    field_value: Optional[str] = None


class GenerativeFieldExtraction(UtteranceFieldExtractionMethod):
    def __init__(
        self,
        logger: Logger,
        generator: SchematicGenerator[UtteranceFieldExtractionSchema],
    ) -> None:
        self._logger = logger
        self._generator = generator

    @override
    async def extract(
        self,
        utterance: str,
        field_name: str,
        context: UtteranceContext,
    ) -> tuple[bool, JSONSerializable]:
        if field_name != "generative":
            return False, None

        generative_fields = set(re.findall(r"\{\{(generative\.[a-zA-Z0-9_]+)\}\}", utterance))

        if not generative_fields:
            return False, None

        tasks = {
            field[len("generative.") :]: asyncio.create_task(
                self._generate_field(utterance, field, context)
            )
            for field in generative_fields
        }

        await safe_gather(*tasks.values())

        fields = {field: task.result() for field, task in tasks.items()}

        if None in fields.values():
            return False, None

        return True, fields

    async def _generate_field(
        self,
        utterance: str,
        field_name: str,
        context: UtteranceContext,
    ) -> Optional[str]:
        builder = PromptBuilder()

        builder.add_section(
            "utterance-generative-field-extraction-instructions",
            "Your only job is to extract a particular value in the most suitable way from the following context.",
        )

        builder.add_agent_identity(context.agent)
        builder.add_context_variables(context.context_variables)
        builder.add_journeys(context.journeys)
        builder.add_interaction_history(context.interaction_history)
        builder.add_glossary(context.terms)
        builder.add_staged_events(context.staged_events)

        builder.add_section(
            "utterance-generative-field-extraction-field-name",
            """\
We're now working on rendering an utterance template as a reply to the user.

The utterance template we're rendering is this: ###
{utterance}
###

We're rendering one field at a time out of this utterance.
Your job now is to take all of the context above and extract out of it the value for the field '{field_name}' within the utterance template.

Output a JSON object containing the extracted field such that it neatly renders (substituting the field variable) into the utterance template.

When applicable, if the field is substituted by a list or dict, consider rendering the value in Markdown format.

A few examples:
---------------
1) Utterance is "Hello {{{{generative.name}}}}, how may I help you today?"
Example return value: ###
{{ "field_name": "name", "field_value": "John" }}
###

2) Utterance is "Hello {{{{generative.names}}}}, how may I help you today?"
Example return value: ###
{{ "field_name": "names", "field_value": "John and Katie" }}
###

3) Utterance is "Next flights are {{{{generative.flight_list}}}}
Example return value: ###
{{ "field_name": "flight_list", "field_value": "- <FLIGHT_1>\\n- <FLIGHT_2>\\n" }}
###
""",
            props={"utterance": utterance, "field_name": field_name},
        )

        result = await self._generator.generate(builder)

        self._logger.debug(
            f"Utterance GenerativeFieldExtraction Completion:\n{result.content.model_dump_json(indent=2)}"
        )

        return result.content.field_value


class UtteranceFieldExtractor(ABC):
    def __init__(
        self,
        standard: StandardFieldExtraction,
        tool_based: ToolBasedFieldExtraction,
        generative: GenerativeFieldExtraction,
    ) -> None:
        self.methods: list[UtteranceFieldExtractionMethod] = [
            standard,
            tool_based,
            generative,
        ]

    async def extract(
        self,
        utterance: str,
        field_name: str,
        context: UtteranceContext,
    ) -> tuple[bool, JSONSerializable]:
        for method in self.methods:
            success, extracted_value = await method.extract(
                utterance,
                field_name,
                context,
            )

            if success:
                return True, extracted_value

        return False, None


class FluidUtteranceFallback(Exception):
    def __init__(self) -> None:
        pass


def _get_utterance_template_fields(template: str) -> set[str]:
    env = jinja2.Environment()
    parse_result = env.parse(template)
    return jinja2.meta.find_undeclared_variables(parse_result)


class UtteranceSelector(MessageEventComposer):
    def __init__(
        self,
        logger: Logger,
        correlator: ContextualCorrelator,
        utterance_draft_generator: SchematicGenerator[UtteranceDraftSchema],
        utterance_selection_generator: SchematicGenerator[UtteranceSelectionSchema],
        utterance_composition_generator: SchematicGenerator[UtteranceRevisionSchema],
        utterance_store: UtteranceStore,
        field_extractor: UtteranceFieldExtractor,
        message_generator: MessageGenerator,
    ) -> None:
        self._logger = logger
        self._correlator = correlator
        self._utterance_draft_generator = utterance_draft_generator
        self._utterance_selection_generator = utterance_selection_generator
        self._utterance_composition_generator = utterance_composition_generator
        self._utterance_store = utterance_store
        self._field_extractor = field_extractor
        self._message_generator = message_generator
        self._cached_utterance_fields: dict[UtteranceId, set[str]] = {}

    async def shots(
        self, composition_mode: CompositionMode
    ) -> Sequence[UtteranceSelectorDraftShot]:
        shots = await shot_collection.list()
        supported_shots = [s for s in shots if composition_mode in s.composition_modes]
        return supported_shots

    @override
    async def generate_response_message_events(
        self,
        event_emitter: EventEmitter,
        agent: Agent,
        customer: Customer,
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        ordinary_guideline_matches: Sequence[GuidelineMatch],
        tool_enabled_guideline_matches: Mapping[GuidelineMatch, Sequence[ToolId]],
        journeys: Sequence[Journey],
        tool_insights: ToolInsights,
        staged_events: Sequence[EmittedEvent],
        latch: Optional[CancellationSuppressionLatch] = None,
    ) -> Sequence[MessageEventComposition]:
        with self._logger.scope("MessageEventComposer"):
            try:
                with self._logger.scope("UtteranceSelector"):
                    with self._logger.operation("Utterance selection and rendering"):
                        return await self._do_generate_events(
                            event_emitter=event_emitter,
                            agent=agent,
                            customer=customer,
                            context_variables=context_variables,
                            interaction_history=interaction_history,
                            terms=terms,
                            ordinary_guideline_matches=ordinary_guideline_matches,
                            journeys=journeys,
                            tool_enabled_guideline_matches=tool_enabled_guideline_matches,
                            tool_insights=tool_insights,
                            staged_events=staged_events,
                            latch=latch,
                        )
            except FluidUtteranceFallback:
                return await self._message_generator.generate_response_message_events(
                    event_emitter,
                    agent,
                    customer,
                    context_variables,
                    interaction_history,
                    terms,
                    ordinary_guideline_matches,
                    tool_enabled_guideline_matches,
                    journeys,
                    tool_insights,
                    staged_events,
                    latch,
                )

    async def _get_relevant_utterances(
        self,
        context: UtteranceContext,
    ) -> list[Utterance]:
        stored_utterances = list(await self._utterance_store.list_utterances())

        utterances_by_staged_event: list[Utterance] = []

        for event in context.staged_events:
            if event.kind == EventKind.TOOL:
                event_data: dict[str, Any] = cast(dict[str, Any], event.data)
                tool_calls: list[Any] = cast(list[Any], event_data.get("tool_calls", []))
                for tool_call in tool_calls:
                    utterances_by_staged_event.extend(
                        Utterance(
                            id=Utterance.TRANSIENT_ID,
                            value=f.value,
                            fields=f.fields,
                            creation_utc=datetime.now(),
                            tags=[],
                        )
                        for f in tool_call["result"].get("utterances", [])
                    )

        all_candidates = [*stored_utterances, *utterances_by_staged_event]

        # Filter out utterances that contain references to tool-based data
        # if that data does not exist in the session's context.
        all_tool_calls = chain.from_iterable(
            [
                *(
                    cast(ToolEventData, e.data)["tool_calls"]
                    for e in context.staged_events
                    if e.kind == EventKind.TOOL
                ),
                *(
                    cast(ToolEventData, e.data)["tool_calls"]
                    for e in context.interaction_history
                    if e.kind == EventKind.TOOL
                ),
            ]
        )

        all_available_fields = list(
            chain.from_iterable(tc["result"]["utterance_fields"] for tc in all_tool_calls)
        )

        all_available_fields.extend(("std", "generative"))

        relevant_utterances = []

        for u in all_candidates:
            if u.id not in self._cached_utterance_fields:
                self._cached_utterance_fields[u.id] = _get_utterance_template_fields(u.value)

            if all(field in all_available_fields for field in self._cached_utterance_fields[u.id]):
                relevant_utterances.append(u)

        return relevant_utterances

    async def _do_generate_events(
        self,
        event_emitter: EventEmitter,
        agent: Agent,
        customer: Customer,
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        ordinary_guideline_matches: Sequence[GuidelineMatch],
        tool_enabled_guideline_matches: Mapping[GuidelineMatch, Sequence[ToolId]],
        journeys: Sequence[Journey],
        tool_insights: ToolInsights,
        staged_events: Sequence[EmittedEvent],
        latch: Optional[CancellationSuppressionLatch] = None,
    ) -> Sequence[MessageEventComposition]:
        if (
            not interaction_history
            and not ordinary_guideline_matches
            and not tool_enabled_guideline_matches
        ):
            # No interaction and no guidelines that could trigger
            # a proactive start of the interaction
            self._logger.info("Skipping response; interaction is empty and there are no guidelines")
            return []

        context = UtteranceContext(
            agent=agent,
            customer=customer,
            context_variables=context_variables,
            interaction_history=interaction_history,
            terms=terms,
            ordinary_guideline_matches=ordinary_guideline_matches,
            tool_enabled_guideline_matches=tool_enabled_guideline_matches,
            journeys=journeys,
            tool_insights=tool_insights,
            staged_events=staged_events,
        )

        utterances = await self._get_relevant_utterances(context)

        if not utterances and agent.composition_mode == CompositionMode.FLUID_UTTERANCE:
            self._logger.warning("No utterances found; falling back to fluid generation")
            raise FluidUtteranceFallback()

        last_known_event_offset = interaction_history[-1].offset if interaction_history else -1

        await event_emitter.emit_status_event(
            correlation_id=self._correlator.correlation_id,
            data={
                "acknowledged_offset": last_known_event_offset,
                "status": "typing",
                "data": {},
            },
        )

        generation_attempt_temperatures = {
            0: 0.1,
            1: 0.05,
            2: 0.2,
        }

        last_generation_exception: Exception | None = None

        for generation_attempt in range(3):
            try:
                generation_info, result = await self._generate_utterance(
                    context,
                    utterances,
                    agent.composition_mode,
                    temperature=generation_attempt_temperatures[generation_attempt],
                )

                if latch:
                    latch.enable()

                if result is not None:
                    sub_messages = result.message.split("\n\n")
                    events = []

                    while sub_messages:
                        m = sub_messages.pop(0)

                        event = await event_emitter.emit_message_event(
                            correlation_id=self._correlator.correlation_id,
                            data=MessageEventData(
                                message=m,
                                participant=Participant(id=agent.id, display_name=agent.name),
                                draft=result.draft,
                                utterances=result.utterances,
                            ),
                        )

                        events.append(event)

                        if next_message := sub_messages[0] if sub_messages else None:
                            typing_speed_in_words_per_minute = 100
                            word_count = len(next_message.split())
                            await asyncio.sleep(word_count / typing_speed_in_words_per_minute)

                    return [MessageEventComposition(generation_info, events)]
                else:
                    self._logger.debug("Skipping response; no response deemed necessary")
                    return [MessageEventComposition(generation_info, [])]
            except FluidUtteranceFallback:
                raise
            except Exception as exc:
                self._logger.warning(
                    f"Generation attempt {generation_attempt} failed: {traceback.format_exception(exc)}"
                )
                last_generation_exception = exc

        raise MessageCompositionError() from last_generation_exception

    def _get_guideline_matches_text(
        self,
        ordinary: Sequence[GuidelineMatch],
        tool_enabled: Mapping[GuidelineMatch, Sequence[ToolId]],
    ) -> str:
        all_matches = list(chain(ordinary, tool_enabled))

        if not all_matches:
            return """
In formulating your reply, you are normally required to follow a number of behavioral guidelines.
However, in this case, no special behavioral guidelines were provided.
"""
        guidelines = []

        for i, p in enumerate(all_matches, start=1):
            if p.guideline.content.action:
                guideline = f"Guideline #{i}) When {p.guideline.content.condition}, then {p.guideline.content.action}"
                guideline += f"\n    [Priority (1-10): {p.score}; Rationale: {p.rationale}]"
                guidelines.append(guideline)

        guideline_list = "\n".join(guidelines)

        return f"""
When crafting your reply, you must follow the behavioral guidelines provided below, which have been identified as relevant to the current state of the interaction.
Each guideline includes a priority score to indicate its importance and a rationale for its relevance.

You may choose not to follow a guideline only in the following cases:
    - It conflicts with a previous user request.
    - It contradicts another guideline of equal or higher priority.
    - It is clearly inappropriate given the current context of the conversation.
In all other situations, you are expected to adhere to the guidelines.
These guidelines have already been pre-filtered based on the interaction's context and other considerations outside your scope.
Never disregard a guideline, even if you believe its 'when' condition or rationale does not apply. All of the guidelines necessarily apply right now.

- **Guidelines**:
{guideline_list}
"""

    def _format_shots(
        self,
        shots: Sequence[UtteranceSelectorDraftShot],
    ) -> str:
        return "\n".join(
            f"""
Example {i} - {shot.description}: ###
{self._format_shot(shot)}
###
"""
            for i, shot in enumerate(shots, start=1)
        )

    def _format_shot(
        self,
        shot: UtteranceSelectorDraftShot,
    ) -> str:
        return f"""
- **Expected Result**:
```json
{json.dumps(shot.expected_result.model_dump(mode="json", exclude_unset=True), indent=2)}
```"""

    def _build_draft_prompt(
        self,
        agent: Agent,
        customer: Customer,
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        ordinary_guideline_matches: Sequence[GuidelineMatch],
        journeys: Sequence[Journey],
        tool_enabled_guideline_matches: Mapping[GuidelineMatch, Sequence[ToolId]],
        staged_events: Sequence[EmittedEvent],
        tool_insights: ToolInsights,
        utterances: Sequence[Utterance],
        shots: Sequence[UtteranceSelectorDraftShot],
    ) -> PromptBuilder:
        builder = PromptBuilder(
            on_build=lambda prompt: self._logger.debug(f"Utterance Draft Prompt:\n{prompt}")
        )

        builder.add_section(
            name="utterance-selector-draft-general-instructions",
            template="""
GENERAL INSTRUCTIONS
-----------------
You are an AI agent who is part of a system that interacts with a user. The current state of this interaction will be provided to you later in this message.
Your role is to generate a reply message to the current (latest) state of the interaction, based on provided guidelines, background information, and user-provided information.

Later in this prompt, you'll be provided with behavioral guidelines and other contextual information you must take into account when generating your response.

""",
            props={},
        )

        builder.add_agent_identity(agent)
        builder.add_customer_identity(customer)
        builder.add_section(
            name="utterance-selector-draft-task-description",
            template="""
TASK DESCRIPTION:
-----------------
Continue the provided interaction in a natural and human-like manner.
Your task is to produce a response to the latest state of the interaction.
Always abide by the following general principles (note these are not the "guidelines". The guidelines will be provided later):
1. GENERAL BEHAVIOR: Make your response as human-like as possible. Be concise and avoid being overly polite when not necessary.
2. AVOID REPEATING YOURSELF: When replying— avoid repeating yourself. Instead, refer the user to your previous answer, or choose a new approach altogether. If a conversation is looping, point that out to the user instead of maintaining the loop.
3. REITERATE INFORMATION FROM PREVIOUS MESSAGES IF NECESSARY: If you previously suggested a solution or shared information during the interaction, you may repeat it when relevant. Your earlier response may have been based on information that is no longer available to you, so it's important to trust that it was informed by the context at the time.
4. MAINTAIN GENERATION SECRECY: Never reveal details about the process you followed to produce your response. Do not explicitly mention the tools, context variables, guidelines, glossary, or any other internal information. Present your replies as though all relevant knowledge is inherent to you, not derived from external instructions.
""",
            props={},
        )
        if not interaction_history or all(
            [event.kind != EventKind.MESSAGE for event in interaction_history]
        ):
            builder.add_section(
                name="utterance-selector-draft-initial-message-instructions",
                template="""
The interaction with the user has just began, and no messages were sent by either party.
If told so by a guideline or some other contextual condition, send the first message. Otherwise, do not produce a reply (utterance is null).
If you decide not to emit a message, output the following:
{{
    "last_message_of_user": "<user's last message>",
    "guidelines": [<list of strings- a re-statement of all guidelines>],
    "insights": [<list of strings- up to 3 original insights>],
    "message": null
}}
Otherwise, follow the rest of this prompt to choose the content of your response.
        """,
                props={},
            )

        else:
            builder.add_section(
                name="utterance-selector-draft-ongoing-interaction-instructions",
                template="""
Since the interaction with the user is already ongoing, always produce a reply to the user's last message.
The only exception where you may not produce a reply (i.e., setting message = null) is if the user explicitly asked you not to respond to their message.
In all other cases, even if the user is indicating that the conversation is over, you must produce a reply.
                """,
                props={},
            )

        builder.add_section(
            name="utterance-selector-draft-revision-mechanism",
            template="""
RESPONSE MECHANISM
------------------
To craft an optimal response, ensure alignment with all provided guidelines based on the latest interaction state.

Before choosing your response, identify up to three key insights based on this prompt and the ongoing conversation.
These insights should include relevant user requests, applicable principles from this prompt, or conclusions drawn from the interaction.
Ensure to include any user request as an insight, whether it's explicit or implicit.
Do not add insights unless you believe that they are absolutely necessary. Prefer suggesting fewer insights, if at all.

The final output must be a JSON document detailing the message development process, including insights to abide by,


PRIORITIZING INSTRUCTIONS (GUIDELINES VS. INSIGHTS)
---------------------------------------------------
Deviating from an instruction (either guideline or insight) is acceptable only when the deviation arises from a deliberate prioritization, based on:
    - Conflicts with a higher-priority guideline (according to their priority scores).
    - Contradictions with a user request.
    - Lack of sufficient context or data.
    - Conflicts with an insight (see below).
In all other cases, even if you believe that a guideline's condition does not apply, you must follow it.
If fulfilling a guideline is not possible, explicitly justify why in your response.

Guidelines vs. Insights:
Sometimes, a guideline may conflict with an insight you've derived.
For example, if your insight suggests "the user is vegetarian," but a guideline instructs you to offer non-vegetarian dishes, prioritizing the insight would better align with the business's goals—since offering vegetarian options would clearly benefit the user.

However, remember that the guidelines reflect the explicit wishes of the business you represent. Deviating from them should only occur if doing so does not put the business at risk.
For instance, if a guideline explicitly prohibits a specific action (e.g., "never do X"), you must not perform that action, even if requested by the user or supported by an insight.

In cases of conflict, prioritize the business's values and ensure your decisions align with their overarching goals.

""",
        )
        builder.add_section(
            name="utterance-selector-draft-examples",
            template="""
EXAMPLES
-----------------
{formatted_shots}
""",
            props={
                "formatted_shots": self._format_shots(shots),
                "shots": shots,
            },
        )
        builder.add_context_variables(context_variables)
        builder.add_journeys(journeys)
        builder.add_section(
            name=BuiltInSection.GUIDELINE_DESCRIPTIONS,
            template=self._get_guideline_matches_text(
                ordinary_guideline_matches,
                tool_enabled_guideline_matches,
            ),
            props={
                "ordinary_guideline_matches": ordinary_guideline_matches,
                "tool_enabled_guideline_matches": tool_enabled_guideline_matches,
            },
            status=SectionStatus.ACTIVE
            if ordinary_guideline_matches or tool_enabled_guideline_matches
            else SectionStatus.PASSIVE,
        )
        builder.add_interaction_history(interaction_history)
        builder.add_staged_events(staged_events)

        if tool_insights.missing_data:
            builder.add_section(
                name="utterance-selector-draft-missing-data-for-tools",
                template="""
MISSING REQUIRED DATA FOR TOOL CALLS:
-------------------------------------
The following is a description of missing data that has been deemed necessary
in order to run tools. The tools would have run, if they only had this data available.
If it makes sense in the current state of the interaction, you may choose to inform the user about this missing data: ###
{formatted_missing_data}
###
""",
                props={
                    "formatted_missing_data": json.dumps(
                        [
                            {
                                "datum_name": d.parameter,
                                **({"description": d.description} if d.description else {}),
                                **({"significance": d.significance} if d.significance else {}),
                                **({"examples": d.examples} if d.examples else {}),
                            }
                            for d in tool_insights.missing_data
                        ]
                    ),
                    "missing_data": tool_insights.missing_data,
                },
            )

        if tool_insights.invalid_data:
            builder.add_section(
                name="utterance-selector-invalid-data-for-tools",
                template="""
INVALID DATA FOR TOOL CALLS:
-------------------------------------
The following is a description of invalid data that has been deemed necessary
in order to run tools. The tools would have run, if they only had this data available.
You should inform the user about this invalid data: ###
{formatted_invalid_data}
###
""",
                props={
                    "formatted_invalid_data": json.dumps(
                        [
                            {
                                "datum_name": d.parameter,
                                **({"description": d.description} if d.description else {}),
                                **({"significance": d.significance} if d.significance else {}),
                                **({"examples": d.examples} if d.examples else {}),
                            }
                            for d in tool_insights.invalid_data
                        ]
                    ),
                    "invalid_data": tool_insights.invalid_data,
                },
            )

        builder.add_section(
            name="utterance-selector-output-format",
            template="""
Produce a valid JSON object in the following format: ###

{formatted_output_format}
""",
            props={
                "formatted_output_format": self._get_draft_output_format(
                    interaction_history,
                    list(chain(ordinary_guideline_matches, tool_enabled_guideline_matches)),
                ),
                "interaction_history": interaction_history,
                "guidelines": list(
                    chain(ordinary_guideline_matches, tool_enabled_guideline_matches)
                ),
            },
        )
        return builder

    def _get_draft_output_format(
        self,
        interaction_history: Sequence[Event],
        guidelines: Sequence[GuidelineMatch],
    ) -> str:
        last_user_message = next(
            (
                event.data["message"] if not event.data.get("flagged", False) else "<N/A>"
                for event in reversed(interaction_history)
                if (
                    event.kind == EventKind.MESSAGE
                    and event.source == EventSource.CUSTOMER
                    and isinstance(event.data, dict)
                )
            ),
            "",
        )
        guidelines_list_text = ", ".join([f'"{g.guideline}"' for g in guidelines])

        return f"""
{{
    "last_message_of_user": "{last_user_message}",
    "guidelines": [{guidelines_list_text}],
    "insights": [<Up to 3 original insights to adhere to>],
    "message": "<message text>"
}}
###"""

    def _build_selection_prompt(
        self,
        context: UtteranceContext,
        draft_message: str,
        utterances: Sequence[Utterance],
    ) -> PromptBuilder:
        builder = PromptBuilder(
            on_build=lambda prompt: self._logger.debug(f"Utterance Selection Prompt:\n{prompt}")
        )

        if context.guidelines:
            formatted_guidelines = (
                "In choosing the template, try to abide by the following general guidelines: ###\n"
            )

            for g in context.guidelines:
                formatted_guidelines += (
                    f"\n- When {g.guideline.content.condition}, then {g.guideline.content.action}."
                )

            formatted_guidelines = "###"
        else:
            formatted_guidelines = ""

        formatted_utterances = "\n".join(
            [f'Template ID: {u.id} """\n{u.value}\n"""\n' for u in utterances]
        )

        builder.add_section(
            name="utterance-selector-selection-task-description",
            template="""
1. You are an AI agent who is part of a system that interacts with a user.
2. A draft reply to the user has been generated by a human operator.
3. You are presented with a number of Jinja2 reply templates to choose from. These templates have been pre-approved by business stakeholders for producing fluent customer-facing AI conversations.
4. Your role is to choose (classify) the pre-approved reply template that MOST faithfully captures the human operator's draft reply.
5. Note that there may be multiple relevant choices. Out of those, you must choose the MOST suitable one that is MOST LIKE the human operator's draft reply.
6. Keep in mind that these are Jinja 2 *templates*. Some of them refer to variables or contain procederal instructions. These will be substituted by real values and rendered later. You can assume that such substitution will be handled well to account for the data provided in the draft message! FYI, if you encounter a variable {{generative.<something>}}, that means that it will later be substituted with a dynamic, flexible, generated value based on the appropriate context. You just need to choose the most viable reply template to use, and assume it will be filled and rendered properly later.""",
        )

        builder.add_interaction_history(context.interaction_history)

        builder.add_section(
            name="utterance-selector-selection-inputs",
            template="""
{formatted_guidelines}

Pre-approved reply templates: ###
{formatted_utterances}
###

Draft reply message: ###
{draft_message}
###

Output a JSON object with a two properties:
1. "rationale": reason about the most appropriate template choice to capture the draft message's main intent
2. "chosen_template_id" containing the selected template ID.
3. "match_quality": which can be ONLY ONE OF "low", "partial", "high".
    a. "low": You couldn't find a template that even comes close
    b. "partial": You found a template that conveys at least some of the draft message's content
    c. "high": You found a template that captures the draft message in both form and function
""",
            props={
                "draft_message": draft_message,
                "utterances": utterances,
                "formatted_utterances": formatted_utterances,
                "guidelines": context.guidelines,
                "formatted_guidelines": formatted_guidelines,
                "composition_mode": context.agent.composition_mode,
            },
        )

        return builder

    async def _generate_utterance(
        self,
        context: UtteranceContext,
        utterances: Sequence[Utterance],
        composition_mode: CompositionMode,
        temperature: float,
    ) -> tuple[Mapping[str, GenerationInfo], Optional[_UtteranceSelectionResult]]:
        draft_prompt = self._build_draft_prompt(
            agent=context.agent,
            context_variables=context.context_variables,
            customer=context.customer,
            interaction_history=context.interaction_history,
            terms=context.terms,
            ordinary_guideline_matches=context.ordinary_guideline_matches,
            journeys=context.journeys,
            tool_enabled_guideline_matches=context.tool_enabled_guideline_matches,
            staged_events=context.staged_events,
            tool_insights=context.tool_insights,
            utterances=utterances,
            shots=await self.shots(context.agent.composition_mode),
        )

        draft_response = await self._utterance_draft_generator.generate(
            prompt=draft_prompt,
            hints={"temperature": temperature},
        )

        self._logger.debug(
            f"Utterance Draft Completion:\n{draft_response.content.model_dump_json(indent=2)}"
        )

        if not draft_response.content.message:
            return {"draft": draft_response.info}, None

        selection_response = await self._utterance_selection_generator.generate(
            prompt=self._build_selection_prompt(
                context=context,
                draft_message=draft_response.content.message,
                utterances=utterances,
            ),
            hints={"temperature": 0.1},
        )

        self._logger.debug(
            f"Utterance Selection Completion:\n{selection_response.content.model_dump_json(indent=2)}"
        )

        if (
            selection_response.content.match_quality not in ["partial", "high"]
            or not selection_response.content.chosen_template_id
        ):
            if composition_mode in [
                CompositionMode.STRICT_UTTERANCE,
                CompositionMode.COMPOSITED_UTTERANCE,
            ]:
                self._logger.warning(
                    "Failed to find relevant utterances. Please review utterance selection prompt and completion."
                )

                return {
                    "draft": draft_response.info,
                    "selection": selection_response.info,
                }, _UtteranceSelectionResult.no_match(draft=draft_response.content.message)
            else:
                raise FluidUtteranceFallback()

        if (
            selection_response.content.match_quality == "partial"
            and composition_mode == CompositionMode.FLUID_UTTERANCE
        ):
            raise FluidUtteranceFallback()

        utterance_id = UtteranceId(selection_response.content.chosen_template_id)

        utterance = next((u.value for u in utterances if u.id == utterance_id), None)

        if not utterance:
            self._logger.error(
                "Invalid utterance ID choice. Please review utterance selection prompt and completion."
            )

            return {
                "draft": draft_response.info,
                "selection": selection_response.info,
            }, _UtteranceSelectionResult.no_match(draft=draft_response.content.message)

        try:
            rendered_utterance = await self._render_utterance(context, utterance)
        except Exception as exc:
            self._logger.error(f"Failed to render utterance '{utterance_id}' ('{utterance}')")
            self._logger.error(f"Utterance rendering failed: {traceback.format_exception(exc)}")

            return {
                "draft": draft_response.info,
                "selection": selection_response.info,
            }, _UtteranceSelectionResult.no_match(draft=draft_response.content.message)

        match composition_mode:
            case CompositionMode.COMPOSITED_UTTERANCE:
                recomposition_generation_info, recomposed_utterance = await self._recompose(
                    context,
                    draft_response.content.message,
                    rendered_utterance,
                )

                return {
                    "draft": draft_response.info,
                    "selection": selection_response.info,
                    "composition": recomposition_generation_info,
                }, _UtteranceSelectionResult(
                    message=recomposed_utterance,
                    draft=draft_response.content.message,
                    utterances=[(utterance_id, utterance)],
                )
            case CompositionMode.STRICT_UTTERANCE | CompositionMode.FLUID_UTTERANCE:
                return {
                    "draft": draft_response.info,
                    "selection": selection_response.info,
                }, _UtteranceSelectionResult(
                    message=rendered_utterance,
                    draft=draft_response.content.message,
                    utterances=[(utterance_id, utterance)],
                )

        raise Exception("Unsupported composition mode")

    async def _render_utterance(self, context: UtteranceContext, utterance: str) -> str:
        args = {}

        for field_name in _get_utterance_template_fields(utterance):
            success, value = await self._field_extractor.extract(
                utterance,
                field_name,
                context,
            )

            if success:
                args[field_name] = value
            else:
                self._logger.error(f"Utterance field extraction: missing '{field_name}'")
                raise KeyError(f"Missing field '{field_name}' in utterance")

        return jinja2.Template(utterance).render(**args)

    async def _recompose(
        self,
        context: UtteranceContext,
        draft_message: str,
        reference_message: str,
    ) -> tuple[GenerationInfo, str]:
        builder = PromptBuilder(
            on_build=lambda prompt: self._logger.debug(f"Composition Prompt:\n{prompt}")
        )

        builder.add_agent_identity(context.agent)

        builder.add_section(
            name="utterance-selector-composition",
            template="""\
Task Description
----------------
You are given two messages:
1. Draft message
2. Style reference message

The draft message contains what should be said right now.
The style reference message teaches you what communication style to try to copy.

You must say what the draft message says, but capture the tone and style of the style reference message precisely.

Make sure NOT to add, remove, or hallucinate information nor add or remove key words (nouns, verbs) to the message.

IMPORTANT NOTE: Always try to separate points in your message by 2 newlines (\\n\\n) — even if the reference message doesn't do so. You may do this zero or multiple times in the message, as needed. Pay extra attention to this requirement. For example, here's what you should separate:
1. Answering one thing and then another thing -- Put two newlines in between
2. Answering one thing and then asking a follow-up question (e.g., Should I... / Can I... / Want me to... / etc.) -- Put two newlines in between
3. An initial acknowledgement (Sure... / Sorry... / Thanks...) or greeting (Hey... / Good day...) and actual follow-up statements -- Put two newlines in between

Draft message: ###
{draft_message}
###

Style reference message: ###
{reference_message}
###

Respond with a JSON object {{ "revised_utterance": "<message_with_points_separated_by_double_newlines>" }}
""",
            props={
                "draft_message": draft_message,
                "reference_message": reference_message,
            },
        )

        result = await self._utterance_composition_generator.generate(
            builder,
            hints={"temperature": 1},
        )

        self._logger.debug(f"Composition Completion:\n{result.content.model_dump_json(indent=2)}")

        return result.info, result.content.revised_utterance


def shot_utterance_id(number: int) -> str:
    return f"<example-only-utterance--{number}--do-not-use-in-your-completion>"


example_1_expected = UtteranceDraftSchema(
    last_message_of_user="Hi, I'd like an onion cheeseburger please.",
    guidelines=[
        "When the user chooses and orders a burger, then provide it",
        "When the user chooses specific ingredients on the burger, only provide those ingredients if we have them fresh in stock; otherwise, reject the order",
    ],
    insights=[
        "All of our cheese has expired and is currently out of stock",
        "The user is a long-time user and we should treat him with extra respect",
    ],
    message="Unfortunately we're out of cheese. Would you like anything else instead?",
)

example_1_shot = UtteranceSelectorDraftShot(
    composition_modes=[CompositionMode.FLUID_UTTERANCE],
    description="A reply where one instruction was prioritized over another",
    expected_result=example_1_expected,
)


example_2_expected = UtteranceDraftSchema(
    last_message_of_user="Hi there, can I get something to drink? What do you have on tap?",
    guidelines=["When the user asks for a drink, check the menu and offer what's on it"],
    insights=[
        "According to contextual information about the user, this is their first time here",
        "There's no menu information in my context",
    ],
    message="I'm sorry, but I'm having trouble accessing our menu at the moment. This isn't a great first impression! Can I possibly help you with anything else?",
)

example_2_shot = UtteranceSelectorDraftShot(
    composition_modes=[
        CompositionMode.STRICT_UTTERANCE,
        CompositionMode.COMPOSITED_UTTERANCE,
        CompositionMode.FLUID_UTTERANCE,
    ],
    description="Non-adherence to guideline due to missing data",
    expected_result=example_2_expected,
)


example_3_expected = UtteranceDraftSchema(
    last_message_of_user="This is not what I was asking for!",
    guidelines=[],
    insights=[
        "I should not keep repeating myself asking for clarifications, as it makes me sound robotic"
    ],
    message="I apologize for failing to assist you with your issue. If there's anything else I can do for you, please let me know.",
)

example_3_shot = UtteranceSelectorDraftShot(
    composition_modes=[
        CompositionMode.STRICT_UTTERANCE,
        CompositionMode.COMPOSITED_UTTERANCE,
        CompositionMode.FLUID_UTTERANCE,
    ],
    description="Avoiding repetitive responses—in this case, given that the previous response by the agent was 'I am sorry, could you please clarify your request?'",
    expected_result=example_3_expected,
)


example_4_expected = UtteranceDraftSchema(
    last_message_of_user=("Hey, how can I contact customer support?"),
    guidelines=[],
    insights=[
        "When I cannot help with a topic, I should tell the user I can't help with it",
    ],
    message="Unfortunately, I cannot refer you to live customer support. Is there anything else I can help you with?",
)

example_4_shot = UtteranceSelectorDraftShot(
    composition_modes=[
        CompositionMode.STRICT_UTTERANCE,
        CompositionMode.COMPOSITED_UTTERANCE,
        CompositionMode.FLUID_UTTERANCE,
    ],
    description="An insight is derived and followed on not offering to help with something you don't know about",
    expected_result=example_4_expected,
)


_baseline_shots: Sequence[UtteranceSelectorDraftShot] = [
    example_1_shot,
    example_2_shot,
    example_3_shot,
    example_4_shot,
]

shot_collection = ShotCollection[UtteranceSelectorDraftShot](_baseline_shots)
