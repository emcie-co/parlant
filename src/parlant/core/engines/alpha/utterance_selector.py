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
from parlant.core.capabilities import Capability
from parlant.core.contextual_correlator import ContextualCorrelator
from parlant.core.agents import Agent, CompositionMode
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.customers import Customer
from parlant.core.engines.alpha.guideline_matching.generic.common import (
    GuidelineInternalRepresentation,
    internal_representation,
)
from parlant.core.engines.alpha.message_event_composer import (
    MessageCompositionError,
    MessageEventComposer,
    MessageEventComposition,
)
from parlant.core.engines.alpha.message_generator import MessageGenerator
from parlant.core.engines.alpha.perceived_performance_policy import PerceivedPerformancePolicy
from parlant.core.engines.alpha.tool_calling.tool_caller import ToolInsights
from parlant.core.entity_cq import EntityQueries
from parlant.core.guidelines import GuidelineId
from parlant.core.journeys import Journey
from parlant.core.utterances import Utterance, UtteranceId, UtteranceStore
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.alpha.prompt_builder import PromptBuilder, BuiltInSection
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
    journey_state: Optional[str] = None
    insights: Optional[list[str]] = None
    response_preamble_that_was_already_sent: Optional[str] = None
    response_body: Optional[str] = None


class UtteranceSelectionSchema(DefaultBaseModel):
    tldr: Optional[str] = None
    chosen_template_id: Optional[str] = None
    match_quality: Optional[str] = None


class UtteranceFluidPreambleSchema(DefaultBaseModel):
    preamble: str


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


@dataclass
class UtteranceContext:
    event_emitter: EventEmitter
    agent: Agent
    customer: Customer
    context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]]
    interaction_history: Sequence[Event]
    terms: Sequence[Term]
    capabilities: Sequence[Capability]
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
            "glossary": {term.name: term.description for term in context.terms},
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
        def _get_field_extraction_guidelines_text(
            all_matches: Sequence[GuidelineMatch],
            guideline_representations: dict[GuidelineId, GuidelineInternalRepresentation],
        ) -> str:
            guidelines_texts = []
            for i, p in enumerate(all_matches, start=1):
                if p.guideline.content.action:
                    guideline = f"Guideline #{i}) When {guideline_representations[p.guideline.id].condition}, then {guideline_representations[p.guideline.id].action}"
                    guideline += f"\n    [Priority (1-10): {p.score}; Rationale: {p.rationale}]"
                    guidelines_texts.append(guideline)
            return "\n".join(guidelines_texts)

        builder = PromptBuilder()

        builder.add_section(
            "utterance-generative-field-extraction-instructions",
            "Your only job is to extract a particular value in the most suitable way from the following context.",
        )

        builder.add_agent_identity(context.agent)
        builder.add_customer_identity(context.customer)
        builder.add_context_variables(context.context_variables)
        builder.add_journeys(context.journeys)

        all_guideline_matches = list(
            chain(context.ordinary_guideline_matches, context.tool_enabled_guideline_matches)
        )

        guideline_representations = {
            m.guideline.id: internal_representation(m.guideline) for m in all_guideline_matches
        }

        builder.add_section(
            name=BuiltInSection.GUIDELINES,
            template="""
When crafting your reply, you must follow the behavioral guidelines provided below, which have been identified as relevant to the current state of the interaction.
Each guideline includes a priority score to indicate its importance and a rationale for its relevance.
The guidelines are not necessarily intended to aid your current task of field generation, but to support other components in the system.
{all_guideline_matches_text}
""",
            props={
                "all_guideline_matches_text": _get_field_extraction_guidelines_text(
                    all_guideline_matches, guideline_representations
                )
            },
        )
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
        utterance_fluid_preamble_generator: SchematicGenerator[UtteranceFluidPreambleSchema],
        perceived_performance_policy: PerceivedPerformancePolicy,
        utterance_store: UtteranceStore,
        field_extractor: UtteranceFieldExtractor,
        message_generator: MessageGenerator,
        entity_queries: EntityQueries,
    ) -> None:
        self._logger = logger
        self._correlator = correlator
        self._utterance_draft_generator = utterance_draft_generator
        self._utterance_selection_generator = utterance_selection_generator
        self._utterance_composition_generator = utterance_composition_generator
        self._utterance_fluid_preamble_generator = utterance_fluid_preamble_generator
        self._utterance_store = utterance_store
        self._perceived_performance_policy = perceived_performance_policy
        self._field_extractor = field_extractor
        self._message_generator = message_generator
        self._cached_utterance_fields: dict[UtteranceId, set[str]] = {}
        self._entity_queries = entity_queries

    async def shots(
        self, composition_mode: CompositionMode
    ) -> Sequence[UtteranceSelectorDraftShot]:
        shots = await shot_collection.list()
        supported_shots = [s for s in shots if composition_mode in s.composition_modes]
        return supported_shots

    @override
    async def generate_preamble(
        self,
        event_emitter: EventEmitter,
        agent: Agent,
        customer: Customer,
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        capabilities: Sequence[Capability],
        ordinary_guideline_matches: Sequence[GuidelineMatch],
        tool_enabled_guideline_matches: Mapping[GuidelineMatch, Sequence[ToolId]],
        journeys: Sequence[Journey],
        tool_insights: ToolInsights,
        staged_events: Sequence[EmittedEvent],
    ) -> Sequence[MessageEventComposition]:
        if agent.composition_mode not in [
            # TODO: Add support for fluid and strict mode (and adjust the tests accordingly)
            CompositionMode.COMPOSITED_UTTERANCE,
        ]:
            return []

        last_known_event_offset = interaction_history[-1].offset if interaction_history else -1

        await event_emitter.emit_status_event(
            correlation_id=f"{self._correlator.correlation_id}.preamble",
            data={
                "acknowledged_offset": last_known_event_offset,
                "status": "typing",
                "data": {},
            },
        )

        prompt_builder = PromptBuilder(
            on_build=lambda prompt: self._logger.debug(f"Utterance Preamble Prompt:\n{prompt}")
        )

        prompt_builder.add_agent_identity(agent)

        prompt_builder.add_section(
            name="utterance-fluid-preamble-instructions",
            template="""\
You are an AI agent that is expected to generate a preamble message for the customer.

The actual message will be sent later by a smarter agent. Your job is only to generate the right preamble in order to save time.
You must not assume anything about how to handle the interaction in any way, shape, or form, beyond just generating the right, nuanced preamble message.

Example preamble messages:
- "Hey there!"
- "Just a moment."
- "Hello."
- "Sorry to hear that."
- "Definitely."
- "Let me check that for you."
etc.

Basically, the preamble is something very short that continues the interaction naturally, without committing to any later action or response.
We leave that later response to another agent. Make sure you understand this.

You must generate the preamble message. You must produce a JSON object with a single key, "preamble", holding the preamble message as a string.

You will now be given the current state of the interaction to which you must generate the next preamble message.
""",
            props={},
        )

        prompt_builder.add_interaction_history(interaction_history)

        response = await self._utterance_fluid_preamble_generator.generate(
            prompt=prompt_builder, hints={"temperature": 0.1}
        )

        self._logger.debug(
            f"Utterance Preamble Completion:\n{response.content.model_dump_json(indent=2)}"
        )

        emitted_event = await event_emitter.emit_message_event(
            correlation_id=f"{self._correlator.correlation_id}.preamble",
            data=response.content.preamble,
        )

        return [
            MessageEventComposition(
                generation_info={"message": response.info},
                events=[emitted_event],
            )
        ]

    @override
    async def generate_response(
        self,
        event_emitter: EventEmitter,
        agent: Agent,
        customer: Customer,
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        capabilities: Sequence[Capability],
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
                            capabilities=capabilities,
                            tool_enabled_guideline_matches=tool_enabled_guideline_matches,
                            tool_insights=tool_insights,
                            staged_events=staged_events,
                            latch=latch,
                        )
            except FluidUtteranceFallback:
                return await self._message_generator.generate_response(
                    event_emitter,
                    agent,
                    customer,
                    context_variables,
                    interaction_history,
                    terms,
                    capabilities,
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
        stored_utterances = await self._entity_queries.find_utterances_for_context(
            agent_id=context.agent.id,
            journeys=context.journeys,
        )

        # Add utterances from staged tool events (transient)
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
                            queries=[],
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
        capabilities: Sequence[Capability],
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
            event_emitter=event_emitter,
            agent=agent,
            customer=customer,
            context_variables=context_variables,
            interaction_history=interaction_history,
            terms=terms,
            ordinary_guideline_matches=ordinary_guideline_matches,
            tool_enabled_guideline_matches=tool_enabled_guideline_matches,
            journeys=journeys,
            capabilities=capabilities,
            tool_insights=tool_insights,
            staged_events=staged_events,
        )

        utterances = await self._get_relevant_utterances(context)

        if not utterances and agent.composition_mode == CompositionMode.FLUID_UTTERANCE:
            self._logger.warning("No utterances found; falling back to fluid generation")
            raise FluidUtteranceFallback()

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
                    sub_messages = result.message.strip().split("\n\n")
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

                        await context.event_emitter.emit_status_event(
                            correlation_id=self._correlator.correlation_id,
                            data={
                                "acknowledged_offset": 0,
                                "status": "ready",
                                "data": {},
                            },
                        )

                        if next_message := sub_messages[0] if sub_messages else None:
                            await self._perceived_performance_policy.get_follow_up_delay()

                            await context.event_emitter.emit_status_event(
                                correlation_id=self._correlator.correlation_id,
                                data={
                                    "acknowledged_offset": 0,
                                    "status": "typing",
                                    "data": {},
                                },
                            )

                            typing_speed_in_words_per_minute = 50

                            initial_delay = 0.0

                            word_count_for_the_message_that_was_just_sent = len(m.split())

                            if word_count_for_the_message_that_was_just_sent <= 10:
                                initial_delay += 0.5
                            else:
                                initial_delay += (
                                    word_count_for_the_message_that_was_just_sent
                                    / typing_speed_in_words_per_minute
                                ) * 2

                            word_count_for_next_message = len(next_message.split())

                            if word_count_for_next_message <= 10:
                                initial_delay += 1
                            else:
                                initial_delay += 2

                            await asyncio.sleep(
                                initial_delay
                                + (word_count_for_next_message / typing_speed_in_words_per_minute)
                            )

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
        guideline_representations: dict[GuidelineId, GuidelineInternalRepresentation],
    ) -> str:
        all_matches = [
            match for match in chain(ordinary, tool_enabled) if match.guideline.content.action
        ]

        if not all_matches:
            return """
In formulating your reply, you are normally required to follow a number of behavioral guidelines.
However, in this case, no special behavioral guidelines were provided.
"""
        guidelines = []
        agent_intention_guidelines = []

        for i, p in enumerate(all_matches, start=1):
            if p.guideline.content.action:
                guideline = f"Guideline #{i}) When {guideline_representations[p.guideline.id].condition}, then {guideline_representations[p.guideline.id].action}"
                guideline += f"\n    [Priority (1-10): {p.score}; Rationale: {p.rationale}]"
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
Some guidelines are tied to condition that related to you, the agent. These guidelines are considered relevant because it is likely that you intends to output
a message that will trigger the associated condition. You should only follow these guidelines if you are actually going to output a message that activates the condition.
- **Guidelines with agent intention condition**:
{agent_intention_guidelines_list}

"""
        if guideline_list:
            guideline_instruction += f"""

For any other guidelines, do not disregard a guideline because you believe its 'when' condition or rationale does not apply—this filtering has already been handled.
- **Guidelines**:
{guideline_list}

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
        return guideline_instruction

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
        capabilities: Sequence[Capability],
        ordinary_guideline_matches: Sequence[GuidelineMatch],
        journeys: Sequence[Journey],
        tool_enabled_guideline_matches: Mapping[GuidelineMatch, Sequence[ToolId]],
        staged_events: Sequence[EmittedEvent],
        tool_insights: ToolInsights,
        utterances: Sequence[Utterance],
        shots: Sequence[UtteranceSelectorDraftShot],
    ) -> PromptBuilder:
        guideline_representations = {
            m.guideline.id: internal_representation(m.guideline)
            for m in chain(ordinary_guideline_matches, tool_enabled_guideline_matches)
        }

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
    "journey_state": "<current state of the journey(s), if any>",
    "insights": [<list of strings- up to 3 original insights>],
    "response_preamble_that_was_already_sent": null,
    "response_body": null
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
Deviating from an instruction (either guideline or insight) is acceptable only when the deviation arises from a deliberate prioritization.
Consider the following valid reasons for such deviations:
    - The instruction contradicts a customer request.
    - The instruction lacks sufficient context or data to apply reliably.
    - The instruction conflicts with an insight (see below).
    - The instruction depends on an agent intention condition that does not apply in the current situation.
    - When a guideline offers multiple options (e.g., "do X or Y") and another more specific guideline restricts one of those options (e.g., "don’t do X"),
    follow both by choosing the permitted alternative (i.e., do Y).
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
        builder.add_glossary(terms)
        builder.add_context_variables(context_variables)
        builder.add_capabilities_for_message_generation(capabilities)
        builder.add_journeys(journeys)
        builder.add_guidelines_for_message_generation(
            ordinary_guideline_matches,
            tool_enabled_guideline_matches,
            guideline_representations,
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
Produce a valid JSON object according to the following spec. Use the values provided as follows, and only replace those in <angle brackets> with appropriate values: ###

{formatted_output_format}
""",
            props={
                "formatted_output_format": self._get_draft_output_format(
                    interaction_history,
                    list(chain(ordinary_guideline_matches, tool_enabled_guideline_matches)),
                ),
                "interaction_history": interaction_history,
                "guidelines": [
                    g
                    for g in chain(ordinary_guideline_matches, tool_enabled_guideline_matches)
                    if g.guideline.content.action
                ],
                "guideline_representations": guideline_representations,
            },
        )

        return builder

    def _get_draft_output_format(
        self,
        interaction_history: Sequence[Event],
        guidelines: Sequence[GuidelineMatch],
    ) -> str:
        last_user_message_event = next(
            (
                event
                for event in reversed(interaction_history)
                if (event.kind == EventKind.MESSAGE and event.source == EventSource.CUSTOMER)
            ),
            None,
        )

        agent_preamble = ""

        if event := last_user_message_event:
            event_data = cast(MessageEventData, event.data)

            last_user_message = (
                event_data["message"]
                if not event_data.get("flagged", False)
                else "<N/A -- censored>"
            )

            agent_preamble = next(
                (
                    cast(MessageEventData, event.data)["message"]
                    for event in reversed(interaction_history)
                    if (
                        event.kind == EventKind.MESSAGE
                        and event.source == EventSource.AI_AGENT
                        and event.offset > last_user_message_event.offset
                    )
                ),
                "",
            )
        else:
            last_user_message = ""

        guidelines_list_text = ", ".join(
            [f'"{g.guideline}"' for g in guidelines if g.guideline.content.action]
        )

        return f"""
{{
    "last_message_of_user": "{last_user_message}",
    "guidelines": [{guidelines_list_text}],
    "journey_state": "<current state of the journey(s), if any>",
    "insights": [<Up to 3 original insights to adhere to>],
    "response_preamble_that_was_already_sent": "{agent_preamble}",
    "response_body": "<response message text (that would immediately follow the preamble)>"
}}
###"""

    def _build_selection_prompt(
        self,
        context: UtteranceContext,
        draft_message: str,
        utterances: Sequence[tuple[UtteranceId, str]],
    ) -> PromptBuilder:
        guideline_representations = {
            m.guideline.id: internal_representation(m.guideline)
            for m in chain(
                context.ordinary_guideline_matches, context.tool_enabled_guideline_matches
            )
        }

        builder = PromptBuilder(
            on_build=lambda prompt: self._logger.debug(f"Utterance Selection Prompt:\n{prompt}")
        )

        if context.guidelines:
            formatted_guidelines = "In choosing the template, there are 2 cases. 1) There is a single, clear match. 2) There are multiple candidates for a match. In the second care, you may also find that there are multiple templates that overlap with the draft message in different ways. In those cases, you will have to decide which part (which overlap) you prioritize. When doing so, your prioritization for choosing between different overlapping templates should try to maximize adherence to the following behavioral guidelines: ###\n"

            for match in [g for g in context.guidelines if g.guideline.content.action]:
                formatted_guidelines += f"\n- When {guideline_representations[match.guideline.id].condition}, then {guideline_representations[match.guideline.id].action}."

            formatted_guidelines += "\n###"
        else:
            formatted_guidelines = ""

        formatted_utterances = "\n".join(
            [f'Template ID: {u[0]} """\n{u[1]}\n"""' for u in utterances]
        )

        builder.add_section(
            name="utterance-selector-selection-task-description",
            template="""
1. You are an AI agent who is part of a system that interacts with a user.
2. A draft reply to the user has been generated by a human operator.
3. You are presented with a number of Jinja2 reply templates to choose from. These templates have been pre-approved by business stakeholders for producing fluent customer-facing AI conversations.
4. Your role is to choose (classify) the pre-approved reply template that MOST faithfully captures the human operator's draft reply.
5. Note that there may be multiple relevant choices. Out of those, you must choose the MOST suitable one that is MOST LIKE the human operator's draft reply.
6. In cases where there are multiple templates that provide a partial match, you may encounter different types of partial matches. Prefer templates that do not deviate from the draft message semantically, even if they only address part of the draft message. They are better than a template that would have captured multiple parts of the draft message while introducing semantic deviations. In other words, better to match fewer parts with higher semantic fidelity than to match more parts with lower semantic fidelity.
7. If there is any noticeable semantic deviation between the draft message and the template, i.e., the draft says "Do X" and the template says "Do Y" (even if Y is a sibling concept under the same category as X), you should not choose that template, even if it captures other parts of the draft message. We want to maintain true fidelity with the draft message.
8. Keep in mind that these are Jinja 2 *templates*. Some of them refer to variables or contain procedural instructions. These will be substituted by real values and rendered later. You can assume that such substitution will be handled well to account for the data provided in the draft message! FYI, if you encounter a variable {{generative.<something>}}, that means that it will later be substituted with a dynamic, flexible, generated value based on the appropriate context. You just need to choose the most viable reply template to use, and assume it will be filled and rendered properly later.""",
        )

        builder.add_interaction_history(context.interaction_history)

        builder.add_section(
            name="utterance-selector-selection-inputs",
            template="""
Pre-approved reply templates: ###
{formatted_utterances}
###

{formatted_guidelines}

Draft reply message: ###
{draft_message}
###

Output a JSON object with three properties:
1. "tldr": consider 1-3 best candidate templates for a match (in view of the draft message and the additional behavioral guidelines) and reason about the most appropriate one choice to capture the draft message's main intent while also ensuring to take the behavioral guidelines into account. Be very pithy and concise in your reasoning, like a newsline heading stating logical notes and conclusions.
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
                "guidelines": [g for g in context.guidelines if g.guideline.content.action],
                "formatted_guidelines": formatted_guidelines,
                "composition_mode": context.agent.composition_mode,
                "guideline_representations": guideline_representations,
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
        # This will be needed throughout the process for emitting status events
        last_known_event_offset = (
            context.interaction_history[-1].offset if context.interaction_history else -1
        )

        direct_draft_output_mode = (
            not utterances
            and context.agent.composition_mode == CompositionMode.COMPOSITED_UTTERANCE
        )

        # Step 1: Generate the draft message
        draft_prompt = self._build_draft_prompt(
            agent=context.agent,
            context_variables=context.context_variables,
            customer=context.customer,
            interaction_history=context.interaction_history,
            terms=context.terms,
            ordinary_guideline_matches=context.ordinary_guideline_matches,
            journeys=context.journeys,
            capabilities=context.capabilities,
            tool_enabled_guideline_matches=context.tool_enabled_guideline_matches,
            staged_events=context.staged_events,
            tool_insights=context.tool_insights,
            utterances=utterances,
            shots=await self.shots(context.agent.composition_mode),
        )

        if direct_draft_output_mode:
            await context.event_emitter.emit_status_event(
                correlation_id=self._correlator.correlation_id,
                data={
                    "acknowledged_offset": last_known_event_offset,
                    "status": "typing",
                    "data": {},
                },
            )

        draft_response = await self._utterance_draft_generator.generate(
            prompt=draft_prompt,
            hints={"temperature": temperature},
        )

        self._logger.debug(
            f"Utterance Draft Completion:\n{draft_response.content.model_dump_json(indent=2)}"
        )

        if not draft_response.content.response_body:
            return {"draft": draft_response.info}, None

        if direct_draft_output_mode:
            return {
                "draft": draft_response.info,
            }, _UtteranceSelectionResult(
                message=draft_response.content.response_body,
                draft=draft_response.content.response_body,
                utterances=[],
            )

        await context.event_emitter.emit_status_event(
            correlation_id=self._correlator.correlation_id,
            data={
                "acknowledged_offset": last_known_event_offset,
                "status": "typing",
                "data": {},
            },
        )

        # Step 2: Select the most relevant utterance templates based on the draft message
        top_relevant_utterances = await self._utterance_store.find_relevant_utterances(
            query=draft_response.content.response_body,
            available_utterances=utterances,
            max_count=10,
        )

        # Step 3: Pre-render these templates so that matching works better
        rendered_utterances = []

        for u in top_relevant_utterances:
            try:
                rendered_utterance = await self._render_utterance(context, u.value)
                rendered_utterances.append((u.id, rendered_utterance))
            except Exception as exc:
                self._logger.error(
                    f"Failed to pre-render utterance for matching '{u.id}' ('{u.value}')"
                )
                self._logger.error(f"Utterance rendering failed: {traceback.format_exception(exc)}")

        # Step 4: Try to match the draft message with one of the rendered utterances
        selection_response = await self._utterance_selection_generator.generate(
            prompt=self._build_selection_prompt(
                context=context,
                draft_message=draft_response.content.response_body,
                utterances=rendered_utterances,
            ),
            hints={"temperature": 0.1},
        )

        self._logger.debug(
            f"Utterance Selection Completion:\n{selection_response.content.model_dump_json(indent=2)}"
        )

        # Step 5: Respond based on the match quality
        if (
            selection_response.content.match_quality not in ["partial", "high"]
            or not selection_response.content.chosen_template_id
        ):
            if composition_mode == CompositionMode.STRICT_UTTERANCE:
                self._logger.warning(
                    "Failed to find relevant utterances. Please review utterance selection prompt and completion."
                )

                return {
                    "draft": draft_response.info,
                    "selection": selection_response.info,
                }, _UtteranceSelectionResult.no_match(draft=draft_response.content.response_body)
            else:
                return {
                    "draft": draft_response.info,
                    "selection": selection_response.info,
                }, _UtteranceSelectionResult(
                    message=draft_response.content.response_body,
                    draft=draft_response.content.response_body,
                    utterances=[],
                )

        if (
            selection_response.content.match_quality == "partial"
            and composition_mode == CompositionMode.FLUID_UTTERANCE
        ):
            return {
                "draft": draft_response.info,
                "selection": selection_response.info,
            }, _UtteranceSelectionResult(
                message=draft_response.content.response_body,
                draft=draft_response.content.response_body,
                utterances=[],
            )

        utterance_id = UtteranceId(selection_response.content.chosen_template_id)

        utterance = next((u.value for u in utterances if u.id == utterance_id), None)

        if not utterance:
            self._logger.error(
                "Invalid utterance ID choice. Please review utterance selection prompt and completion."
            )

            return {
                "draft": draft_response.info,
                "selection": selection_response.info,
            }, _UtteranceSelectionResult.no_match(draft=draft_response.content.response_body)

        try:
            rendered_utterance = await self._render_utterance(context, utterance)
        except Exception as exc:
            self._logger.error(f"Failed to render utterance '{utterance_id}' ('{utterance}')")
            self._logger.error(f"Utterance rendering failed: {traceback.format_exception(exc)}")

            return {
                "draft": draft_response.info,
                "selection": selection_response.info,
            }, _UtteranceSelectionResult.no_match(draft=draft_response.content.response_body)

        match composition_mode:
            case CompositionMode.COMPOSITED_UTTERANCE if (
                selection_response.content.match_quality != "high"
            ):
                recomposition_generation_info, recomposed_utterance = await self._recompose(
                    context,
                    draft_response.content.response_body,
                    rendered_utterance,
                )

                return {
                    "draft": draft_response.info,
                    "selection": selection_response.info,
                    "composition": recomposition_generation_info,
                }, _UtteranceSelectionResult(
                    message=recomposed_utterance,
                    draft=draft_response.content.response_body,
                    utterances=[(utterance_id, utterance)],
                )
            case _:
                return {
                    "draft": draft_response.info,
                    "selection": selection_response.info,
                }, _UtteranceSelectionResult(
                    message=rendered_utterance,
                    draft=draft_response.content.response_body,
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
    journey_state="Journey 1. Still need to stay in step 2 (choose ingredients for burger), as the user's choice is not available and I need to inform them about it",
    insights=[
        "All of our cheese has expired and is currently out of stock",
        "The user is a long-time user and we should treat him with extra respect",
    ],
    response_preamble_that_was_already_sent="Let me check",
    response_body="Unfortunately we're out of cheese. Would you like anything else instead?",
)

example_1_shot = UtteranceSelectorDraftShot(
    composition_modes=[CompositionMode.FLUID_UTTERANCE],
    description="A reply where one instruction was prioritized over another",
    expected_result=example_1_expected,
)


example_2_expected = UtteranceDraftSchema(
    last_message_of_user="Hi there, can I get something to drink? What do you have on tap?",
    guidelines=["When the user asks for a drink, check the menu and offer what's on it"],
    journey_state="Journey 2. Still stuck in step 1 (take initial order), as I don't have the menu yet so I can't proceed from here",
    insights=[
        "According to contextual information about the user, this is their first time here",
        "There's no menu information in my context",
    ],
    response_preamble_that_was_already_sent="Just a moment",
    response_body="I'm sorry, but I'm having trouble accessing our menu at the moment. This isn't a great first impression! Can I possibly help you with anything else?",
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
    last_message_of_user="Sure, I'll take the Pepsi",
    guidelines=[],
    journey_state="Journey 1. Moving from step 4 (offer drinks) to step 5 (order confirmation)",
    insights=[],
    response_preamble_that_was_already_sent="Great!",
    response_body="So just to confirm, you'll have a cheeseburger with onions and a Pepsi, right?",
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
    journey_state=None,
    insights=[
        "When I cannot help with a topic, I should tell the user I can't help with it",
    ],
    response_preamble_that_was_already_sent="Hello",
    response_body="Unfortunately, I cannot refer you to live customer support. Is there anything else I can help you with?",
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
