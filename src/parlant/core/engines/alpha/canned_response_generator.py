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

from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass, field as dataclass_field
from itertools import chain
from random import shuffle
import re
import jinja2
import jinja2.meta
import json
import traceback
from typing import Any, Iterable, Mapping, Optional, Sequence, cast
from typing_extensions import override

from parlant.core.async_utils import Stopwatch, safe_gather, CancellationSuppressionLatch
from parlant.core.capabilities import Capability
from parlant.core.meter import DurationHistogram, Meter
from parlant.core.tracer import Tracer
from parlant.core.agents import Agent, CompositionMode
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.customers import Customer
from parlant.core.engines.alpha.guideline_matching.generic.common import (
    GuidelineInternalRepresentation,
    internal_representation,
)
from parlant.core.engines.alpha.hooks import EngineHooks
from parlant.core.engines.alpha.engine_context import EngineContext
from parlant.core.engines.alpha.message_event_composer import (
    MessageCompositionError,
    MessageEventComposer,
    MessageEventComposition,
)
from parlant.core.engines.alpha.message_generator import MessageGenerator
from parlant.core.engines.alpha.optimization_policy import OptimizationPolicy
from parlant.core.engines.alpha.perceived_performance_policy import (
    PerceivedPerformancePolicyProvider,
)
from parlant.core.engines.alpha.tool_calling.tool_caller import ToolInsights
from parlant.core.entity_cq import EntityQueries
from parlant.core.guidelines import GuidelineId
from parlant.core.journeys import Journey
from parlant.core.tags import Tag
from parlant.core.canned_responses import CannedResponse, CannedResponseId, CannedResponseStore
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
    Session,
    ToolCall,
    ToolEventData,
)
from parlant.core.common import DefaultBaseModel, JSONSerializable
from parlant.core.loggers import Logger
from parlant.core.shots import Shot, ShotCollection
from parlant.core.tools import ToolId

DEFAULT_NO_MATCH_CANREP = "Not sure I understand. Could you please say that another way?"


class NoMatchResponseProvider(ABC):
    async def get_response(self, context: EngineContext, draft: str | None) -> CannedResponse:
        return CannedResponse.create_transient(await self.get_template(context, draft))

    @abstractmethod
    async def get_template(self, context: EngineContext, draft: str | None) -> str: ...


class BasicNoMatchResponseProvider(NoMatchResponseProvider):
    def __init__(self) -> None:
        self.template = DEFAULT_NO_MATCH_CANREP

    @override
    async def get_template(self, context: EngineContext, draft: str | None) -> str:
        return self.template


class CannedResponseDraftSchema(DefaultBaseModel):
    last_message_of_user: Optional[str]
    guidelines: list[str]
    insights: Optional[list[str]] = None
    response_preamble_that_was_already_sent: Optional[str] = None
    response_body: Optional[str] = None


class CannedResponseSelectionSchema(DefaultBaseModel):
    tldr: Optional[str] = None
    chosen_template_id: Optional[str] = None
    match_quality: Optional[str] = None


class FollowUpCannedResponseSelectionSchema(DefaultBaseModel):
    remaining_message_draft: Optional[str] = None
    unsatisfied_guidelines: Optional[str | list[str]] = None
    tldr: Optional[str] = None
    additional_response_required: Optional[bool] = False
    additional_template_id: Optional[str] = None
    match_quality: Optional[str] = None


class CannedResponsePreambleSchema(DefaultBaseModel):
    preamble: str


class CannedResponseRevisionSchema(DefaultBaseModel):
    revised_canned_response: str


@dataclass
class CannedResponseGeneratorDraftShot(Shot):
    composition_modes: list[CompositionMode]
    expected_result: CannedResponseDraftSchema


@dataclass
class FollowUpCannedResponseSelectionShot(Shot):
    description: str
    canned_responses: Mapping[str, str]
    draft: str
    last_agent_message: str
    expected_result: FollowUpCannedResponseSelectionSchema


@dataclass
class _CannedResponseRenderResult:
    response: CannedResponse
    failed: bool
    rendered_text: str | None


@dataclass(frozen=True)
class _CannedResponseSelectionResult:
    message: str
    draft: str | None
    rendered_canned_responses: Sequence[tuple[CannedResponse, str]]
    chosen_canned_responses: list[tuple[CannedResponseId, str]]


@dataclass
class CannedResponseContext:
    start_of_processing: Stopwatch
    event_emitter: EventEmitter
    agent: Agent
    customer: Customer
    session: Session
    context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]]
    interaction_history: Sequence[Event]
    terms: Sequence[Term]
    capabilities: Sequence[Capability]
    ordinary_guideline_matches: Sequence[GuidelineMatch]
    tool_enabled_guideline_matches: Mapping[GuidelineMatch, Sequence[ToolId]]
    journeys: Sequence[Journey]
    tool_insights: ToolInsights
    staged_tool_events: Sequence[EmittedEvent]
    staged_message_events: Sequence[EmittedEvent]
    additional_canned_response_fields: Mapping[str, Any] = dataclass_field(default_factory=dict)

    @property
    def guideline_matches(self) -> Sequence[GuidelineMatch]:
        return [*self.ordinary_guideline_matches, *self.tool_enabled_guideline_matches.keys()]


class CannedResponseFieldExtractionMethod(ABC):
    @abstractmethod
    async def extract(
        self,
        canned_response: str,
        field_name: str,
        context: CannedResponseContext,
    ) -> tuple[bool, JSONSerializable]: ...


class StandardFieldExtraction(CannedResponseFieldExtractionMethod):
    def __init__(self, logger: Logger) -> None:
        self._logger = logger

    @override
    async def extract(
        self,
        canned_response: str,
        field_name: str,
        context: CannedResponseContext,
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


class ToolBasedFieldExtraction(CannedResponseFieldExtractionMethod):
    @override
    async def extract(
        self,
        canned_response: str,
        field_name: str,
        context: CannedResponseContext,
    ) -> tuple[bool, JSONSerializable]:
        tool_calls_in_order_of_importance: list[ToolCall] = []

        tool_calls_in_order_of_importance.extend(
            tc
            for e in context.staged_tool_events
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
            if value := tool_call["result"].get("canned_response_fields", {}).get(field_name, None):
                return True, value

        return False, None


class AdditionalFieldExtraction(CannedResponseFieldExtractionMethod):
    """Extracts fields from additional_canned_response_fields (e.g., from guideline field providers)."""

    @override
    async def extract(
        self,
        canned_response: str,
        field_name: str,
        context: CannedResponseContext,
    ) -> tuple[bool, JSONSerializable]:
        if field_name in context.additional_canned_response_fields:
            return True, context.additional_canned_response_fields[field_name]
        return False, None


class CannedResponseFieldExtractionSchema(DefaultBaseModel):
    field_name: Optional[str] = None
    field_value: Optional[str] = None


class GenerativeFieldExtraction(CannedResponseFieldExtractionMethod):
    def __init__(
        self,
        logger: Logger,
        generator: SchematicGenerator[CannedResponseFieldExtractionSchema],
    ) -> None:
        self._logger = logger
        self._generator = generator

    @override
    async def extract(
        self,
        canned_response: str,
        field_name: str,
        context: CannedResponseContext,
    ) -> tuple[bool, JSONSerializable]:
        if field_name != "generative":
            return False, None

        generative_fields = set(re.findall(r"\{\{(generative\.[a-zA-Z0-9_]+)\}\}", canned_response))

        if not generative_fields:
            return False, None

        tasks = {
            field[len("generative.") :]: asyncio.create_task(
                self._generate_field(canned_response, field, context)
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
        canned_response: str,
        field_name: str,
        context: CannedResponseContext,
    ) -> Optional[str]:
        def _get_field_extraction_guidelines_text(
            all_matches: Sequence[GuidelineMatch],
            guideline_representations: dict[GuidelineId, GuidelineInternalRepresentation],
        ) -> str:
            guidelines_texts = []
            for i, p in enumerate(all_matches, start=1):
                if p.guideline.content.action:
                    internal_representation = guideline_representations[p.guideline.id]

                    if internal_representation.condition:
                        guideline = f"Guideline #{i}) When {guideline_representations[p.guideline.id].condition}, then {guideline_representations[p.guideline.id].action}"
                    else:
                        guideline = (
                            f"Guideline #{i}) {guideline_representations[p.guideline.id].action}"
                        )

                    guideline += f"\n    [Priority (1-10): {p.score}; Rationale: {p.rationale}]"
                    guidelines_texts.append(guideline)
            return "\n".join(guidelines_texts)

        builder = PromptBuilder()

        builder.add_section(
            "canned-response-generative-field-extraction-instructions",
            "Your only job is to extract a particular value in the most suitable way from the following context.",
        )

        builder.add_agent_identity(context.agent)
        builder.add_customer_identity(context.customer, context.session)
        builder.add_context_variables(context.context_variables)

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
        builder.add_interaction_history_for_message_generation(
            context.interaction_history,
            context.staged_message_events,
        )
        builder.add_glossary(context.terms)
        builder.add_staged_tool_events(context.staged_tool_events)

        builder.add_section(
            "canned-response-generative-field-extraction-field-name",
            """\
We're now working on rendering a canned response template as a reply to the user.

The canned response template we're rendering is this: ###
{canned_response}
###

We're rendering one field at a time out of this canned response.
Your job now is to take all of the context above and extract out of it the value for the field '{field_name}' within the canned response template.

Output a SINGLE JSON OBJECT containing the extracted field such that it neatly renders (substituting the field variable) into the canned response template.

When applicable, if the field is substituted by a list or dict, consider rendering the value in Markdown format.

A few examples:
---------------
1) Canned response is "Hello {{{{generative.name}}}}, how may I help you today?"
Example return value: ###
{{ "field_name": "name", "field_value": "John" }}
###

2) Canned response is "Hello {{{{generative.names}}}}, how may I help you today?"
Example return value: ###
{{ "field_name": "names", "field_value": "John and Katie" }}
###

3) Canned response is "Next flights are {{{{generative.flight_list}}}}
Example return value: ###
{{ "field_name": "flight_list", "field_value": "- <FLIGHT_1>\\n- <FLIGHT_2>\\n" }}
###

4) Canned response is "It seems that {{{{generative.customer_issue}}}} might be caused by a different issue."
Example return value: ###
{{ "field_name": "customer_issue", "field_value": "the red light you're seeing" }}
###

5) Canned response is "I could suggest {{{{generative.way_to_help}}}} as a potential solution."
Example return value: ###
{{ "field_name": "way_to_help", "field_value": "that you restart your router" }}
###
""",
            props={"canned_response": canned_response, "field_name": field_name},
        )

        result = await self._generator.generate(builder)

        self._logger.trace(
            f"Canned response GenerativeFieldExtraction Completion:\n{result.content.model_dump_json(indent=2)}"
        )

        return result.content.field_value


class CannedResponseFieldExtractor(ABC):
    def __init__(
        self,
        standard: StandardFieldExtraction,
        tool_based: ToolBasedFieldExtraction,
        additional: AdditionalFieldExtraction,
        generative: GenerativeFieldExtraction,
    ) -> None:
        self.methods: list[CannedResponseFieldExtractionMethod] = [
            standard,
            tool_based,
            additional,
            generative,
        ]

    async def extract(
        self,
        canned_response: str,
        field_name: str,
        context: CannedResponseContext,
    ) -> tuple[bool, JSONSerializable]:
        for method in self.methods:
            success, extracted_value = await method.extract(
                canned_response,
                field_name,
                context,
            )

            if success:
                return True, extracted_value

        return False, None


def _get_response_template_fields(template: str) -> set[str]:
    env = jinja2.Environment()
    parse_result = env.parse(template)
    return jinja2.meta.find_undeclared_variables(parse_result)


class CannedResponseGenerator(MessageEventComposer):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        hooks: EngineHooks,
        optimization_policy: OptimizationPolicy,
        canned_response_draft_generator: SchematicGenerator[CannedResponseDraftSchema],
        canned_selection_generator: SchematicGenerator[CannedResponseSelectionSchema],
        canned_response_composition_generator: SchematicGenerator[CannedResponseRevisionSchema],
        canned_response_preamble_generator: SchematicGenerator[CannedResponsePreambleSchema],
        follow_up_canned_response_generator: SchematicGenerator[
            FollowUpCannedResponseSelectionSchema
        ],
        perceived_performance_policy_provider: PerceivedPerformancePolicyProvider,
        canned_response_store: CannedResponseStore,
        field_extractor: CannedResponseFieldExtractor,
        message_generator: MessageGenerator,
        entity_queries: EntityQueries,
        no_match_provider: NoMatchResponseProvider,
    ) -> None:
        self._logger = logger
        self._tracer = tracer
        self._meter = meter

        self._hooks = hooks
        self._optimization_policy = optimization_policy
        self._canrep_draft_generator = canned_response_draft_generator
        self._canrep_selection_generator = canned_selection_generator
        self._canrep_composition_generator = canned_response_composition_generator
        self._canrep_preamble_generator = canned_response_preamble_generator
        self._follow_up_canrep_generator = follow_up_canned_response_generator
        self._canned_response_store = canned_response_store
        self._perceived_performance_policy_provider = perceived_performance_policy_provider
        self._field_extractor = field_extractor
        self._message_generator = message_generator
        self._cached_response_fields: dict[CannedResponseId, set[str]] = {}
        self._entity_queries = entity_queries
        self._no_match_provider = no_match_provider
        self._follow_ups_enabled = True
        self.candidate_similarity_threshold = 0.4

        self._define_histograms()

    def _define_histograms(self) -> None:
        def _create_histogram(name: str, description: str) -> DurationHistogram:
            return self._meter.create_duration_histogram(
                name=f"canrep.{name}",
                description=description,
            )

        self._hist_canned_response_duration = self._meter.create_duration_histogram(
            name="canrep",
            description="Duration of canned response generation in milliseconds",
        )

        self._hist_preamble_duration = _create_histogram(
            name="preamble",
            description="Duration of canned response preamble generation in milliseconds",
        )
        self._hist_preamble_render_duration = _create_histogram(
            name="preamble.render",
            description="Duration of canned response rendering in milliseconds",
        )
        self._hist_render_duration = _create_histogram(
            name="render",
            description="Duration of canned response rendering in milliseconds",
        )
        self._hist_draft_duration = _create_histogram(
            name="draft",
            description="Duration of canned response draft generation in milliseconds",
        )
        self._hist_retrieval_duration = _create_histogram(
            name="retrieval",
            description="Duration of canned response retrieval in milliseconds",
        )
        self._hist_recompose_duration = _create_histogram(
            name="recompose",
            description="Duration of canned response recomposition in milliseconds",
        )
        self._hist_selection_duration = _create_histogram(
            name="selection",
            description="Duration of canned response selection in milliseconds",
        )
        self._hist_ttfm_duration = _create_histogram(
            name="ttfm",
            description="Time to first message generated in milliseconds",
        )

    async def _resolve_composition_mode(self, context: EngineContext) -> CompositionMode:
        """Resolve effective composition mode from matched guidelines.

        Most restrictive rule: CANNED_STRICT > CANNED_COMPOSITED > CANNED_FLUID
        """
        if context.agent.composition_mode == CompositionMode.CANNED_STRICT:
            return CompositionMode.CANNED_STRICT

        restrictiveness_priority = {
            CompositionMode.CANNED_STRICT: 3,
            CompositionMode.CANNED_COMPOSITED: 2,
            CompositionMode.CANNED_FLUID: 1,
        }

        most_restrictive_mode: CompositionMode | None = None
        max_restrictiveness = 0

        # Check all matched guidelines for composition mode
        for guideline in context.state.guidelines:
            if guideline.composition_mode is not None:
                mode = guideline.composition_mode

                # Track most restrictive (only CANNED_* modes)
                if mode in restrictiveness_priority:
                    restrictiveness = restrictiveness_priority[mode]
                    if restrictiveness > max_restrictiveness:
                        most_restrictive_mode = mode
                        max_restrictiveness = restrictiveness

        # Default to agent's composition mode
        if most_restrictive_mode is None:
            most_restrictive_mode = context.agent.composition_mode

        return most_restrictive_mode

    async def draft_generation_shots(
        self, composition_mode: CompositionMode
    ) -> Sequence[CannedResponseGeneratorDraftShot]:
        shots = await draft_generation_shot_collection.list()
        supported_shots = [s for s in shots if composition_mode in s.composition_modes]
        return supported_shots

    @override
    async def generate_preamble(
        self,
        context: EngineContext,
    ) -> Sequence[MessageEventComposition]:
        with self._logger.scope("MessageEventComposer"):
            with self._logger.scope("CannedResponseGenerator"):
                async with self._hist_preamble_duration.measure():
                    return await self._do_generate_preamble(context)

    async def _do_generate_preamble(
        self,
        context: EngineContext,
    ) -> Sequence[MessageEventComposition]:
        agent = context.agent

        # Resolve effective composition mode
        composition_mode = await self._resolve_composition_mode(context)

        canrep_context = CannedResponseContext(
            start_of_processing=context.creation,
            event_emitter=context.session_event_emitter,
            agent=agent,
            customer=context.customer,
            session=context.session,
            context_variables=context.state.context_variables,
            interaction_history=context.interaction.events,
            terms=list(context.state.glossary_terms),
            ordinary_guideline_matches=context.state.ordinary_guideline_matches,
            tool_enabled_guideline_matches=context.state.tool_enabled_guideline_matches,
            journeys=context.state.journeys,
            capabilities=context.state.capabilities,
            tool_insights=context.state.tool_insights,
            staged_tool_events=context.state.tool_events,
            staged_message_events=context.state.message_events,
            additional_canned_response_fields=context.state.additional_canned_response_fields,
        )

        prompt_builder = PromptBuilder(
            on_build=lambda prompt: self._logger.trace(
                f"Canned response Preamble Prompt:\n{prompt}"
            )
        )

        prompt_builder.add_agent_identity(agent)

        preamble_responses: Sequence[CannedResponse] = []
        preamble_choices: list[str] = []

        if composition_mode != CompositionMode.CANNED_STRICT:
            preamble_choices = [
                "Hey there!",
                "Just a moment.",
                "Hello.",
                "Sorry to hear that.",
                "Definitely.",
                "Let me check that for you.",
            ]

            preamble_choices_text = "".join([f"\n- {choice}" for choice in preamble_choices])

            instructions = f"""\
Generate a brief, natural acknowledgment of the customer's most recent message. 
You must not assume anything about how to handle the interaction in any way, shape, or form, beyond just generating the right, nuanced preamble message.

This preamble should:
- Only acknowledge what the customer just said
- Do NOT ask any questions (including "how can I help you"), make commitments, or indicate next steps
- Your message may not dictate how the conversation should continue, or commit the agent to any future processes as a result. 
- Do NOT repeat or paraphrase previous messages and preambles, as that would hurt the flow of the conversation. Acknowledge the latest customer message with a simple, UNIQUE response. 
- Keep your response on the shorter side, as seen in the examples.

Example preamble messages:
{preamble_choices_text}
etc.

BAD EXAMPLES (what NOT to do):
Customer: "I need to change my flight"
WRONG: "I can help you with that" (commits to action)
WRONG: "Can you provide more details?" (asks a question)
WRONG: "Sure, I'll help you change your flight right away." (indicates next steps)
RIGHT: "Got it. Let me check that for you."

Customer: "My bag didn't arrive"
WRONG: "I'm sorry to hear that. Can you tell me your flight number?" (asks question)
WRONG: "Don't worry, we'll help you with that." (makes commitment)
RIGHT: "I see, let me look into that."

Customer: "Thanks, that's helpful"
WRONG: "You're welcome! Is there anything else I can help you with?" (asks question)
WRONG: "You're welcome! I'm here if you need anything else." (commits to future availability)
RIGHT: "Glad I could help."

Basically, the preamble is something very short that continues the interaction naturally, without committing to any later action or response.
We leave that later response to another agent. Make sure you understand this.


You must generate the preamble message. You must produce a JSON object with a single key, "preamble", holding the preamble message as a string.
"""
        else:
            preamble_responses = [
                canrep
                for canrep in await self._entity_queries.find_canned_responses_for_context(
                    agent=agent,
                    journeys=canrep_context.journeys,
                    guidelines=[m.guideline for m in canrep_context.guideline_matches],
                )
                if Tag.preamble() in canrep.tags
            ]

            async with self._hist_preamble_render_duration.measure():
                preamble_choices = [
                    str(r.rendered_text)
                    for r in await self._render_responses(canrep_context, preamble_responses)
                    if not r.failed
                ]

            if not preamble_choices:
                return []

            # LLMs are usually biased toward the last choices, so we shuffle the list.
            shuffle(preamble_choices)

            preamble_choices_text = "".join([f'\n- "{c}"' for c in preamble_choices])

            instructions = f"""\
These are the preamble messages you can choose from. You must ONLY choose one of these: ###
{preamble_choices_text}
###

Basically, the preamble is something very short that continues the interaction naturally, without committing to any later action or response.
We leave that later response to another agent. Make sure you understand this.

Instructions:
- Note that some of the choices are more generic, and some are more specific to a particular scenario.
- If you're unsure what to choose --> prefer to go with a more generic, bland choice. This should be 80% of cases.
  Examples of generic choices: "Hey there!", "Just a moment.", "Hello.", "Got it."
- If you see clear value in saying something more specific and nuanced --> then go with a more specific choice. This should be 20% or less of cases.
  Examples of specific choices: "Let me check that for you.", "Sorry to hear that.", "Thanks for your patience."

You must now choose the preamble message. You must produce a JSON object with a single key, "preamble", holding the preamble message as a string,
EXACTLY as it is given (pay attention to subtleties like punctuation and copy your choice EXACTLY as it is given above).
"""

        prompt_builder.add_section(
            name="canned-response-fluid-preamble-instructions",
            template="""\
You are an AI agent that is expected to generate a preamble message for the customer.

The actual message will be sent later by a smarter agent. Your job is only to generate the right preamble while the smarter agent generates a comprehensive response.

{composition_mode_specific_instructions}

You will now be given the current state of the interaction to which you must generate the next preamble message.
""",
            props={
                "composition_mode_specific_instructions": instructions,
                "composition_mode": composition_mode,
                "preamble_choices": preamble_choices,
            },
        )

        prompt_builder.add_interaction_history_for_message_generation(
            canrep_context.interaction_history,
            context.state.message_events,
        )

        await canrep_context.event_emitter.emit_status_event(
            trace_id=f"{self._tracer.trace_id}",
            data={
                "status": "typing",
                "data": {},
            },
        )

        canrep = await self._canrep_preamble_generator.generate(
            prompt=prompt_builder, hints={"temperature": 0.1}
        )

        self._logger.trace(
            f"Canned Response Preamble Completion:\n{canrep.content.model_dump_json(indent=2)}"
        )

        if composition_mode == CompositionMode.CANNED_STRICT:
            if canrep.content.preamble not in preamble_choices:
                self._logger.error(
                    f"Selected preamble '{canrep.content.preamble}' is not in the list of available preamble canned_responses."
                )
                return []

        if await self._hooks.call_on_preamble_generated(context, payload=canrep.content.preamble):
            # If we're in, the hook did not bail out.

            handle = await canrep_context.event_emitter.emit_message_event(
                trace_id=self._tracer.trace_id,
                data=MessageEventData(
                    message=canrep.content.preamble,
                    participant=Participant(id=agent.id, display_name=agent.name),
                    tags=[Tag.preamble()],
                ),
            )

            self._tracer.add_event("canrep.preamble_generated")

            return [
                MessageEventComposition(
                    generation_info={"message": canrep.info},
                    events=[handle.event],
                )
            ]

        return []

    @override
    async def generate_response(
        self,
        context: EngineContext,
        latch: Optional[CancellationSuppressionLatch[None]] = None,
    ) -> Sequence[MessageEventComposition]:
        with self._logger.scope("MessageEventComposer"):
            with self._logger.scope("CannedResponseGenerator"):
                async with self._hist_canned_response_duration.measure():
                    return await self._do_generate_events(
                        loaded_context=context,
                        latch=latch,
                    )

    async def _get_relevant_canned_responses(
        self,
        context: CannedResponseContext,
    ) -> list[CannedResponse]:
        stored_responses = [
            canrep
            for canrep in await self._entity_queries.find_canned_responses_for_context(
                agent=context.agent,
                journeys=context.journeys,
                guidelines=[m.guideline for m in context.guideline_matches],
            )
            if Tag.preamble() not in canrep.tags
        ]

        # Add responses from staged tool events (transient)
        responses_by_staged_event: list[CannedResponse] = []
        for event in context.staged_tool_events:
            if event.kind == EventKind.TOOL:
                event_data: dict[str, Any] = cast(dict[str, Any], event.data)
                tool_calls: list[Any] = cast(list[Any], event_data.get("tool_calls", []))
                for tool_call in tool_calls:
                    responses_by_staged_event.extend(
                        CannedResponse.create_transient(r)
                        for r in tool_call["result"].get("canned_responses", [])
                    )

        all_candidates = [*stored_responses, *responses_by_staged_event]

        # Filter out responses that contain references to tool-based data
        # if that data does not exist in the session's context.
        all_tool_calls = chain.from_iterable(
            [
                *(
                    cast(ToolEventData, e.data)["tool_calls"]
                    for e in context.staged_tool_events
                    if e.kind == EventKind.TOOL
                ),
                *(
                    cast(ToolEventData, e.data)["tool_calls"]
                    for e in context.interaction_history
                    if e.kind == EventKind.TOOL
                ),
            ]
        )

        fields_available_in_context = list(
            chain.from_iterable(
                tc["result"].get("canned_response_fields", []) for tc in all_tool_calls
            )
        )

        fields_available_in_context.extend(("std", "generative"))
        fields_available_in_context.extend(context.additional_canned_response_fields.keys())

        relevant_responses = []

        for canrep in all_candidates:
            if (
                canrep.id != CannedResponse.TRANSIENT_ID
                and canrep.id not in self._cached_response_fields
            ):
                self._cached_response_fields[canrep.id] = _get_response_template_fields(
                    canrep.value
                )

            # Conditions for a response being relevant:
            # 1. It's a transient response just generated (e.g., by a tool)
            # 2. Its relevant fields are in-context
            if canrep.id == CannedResponse.TRANSIENT_ID or all(
                field in fields_available_in_context
                for field in self._cached_response_fields[canrep.id]
            ):
                relevant_responses.append(canrep)

        return relevant_responses

    async def _do_generate_events(
        self,
        loaded_context: EngineContext,
        latch: Optional[CancellationSuppressionLatch[None]] = None,
    ) -> Sequence[MessageEventComposition]:
        # Resolve effective composition mode
        composition_mode = await self._resolve_composition_mode(loaded_context)

        is_first_message_emitted = False

        async def output_messages(
            generation_result: _CannedResponseSelectionResult,
        ) -> list[EmittedEvent]:
            nonlocal is_first_message_emitted
            emitted_events: list[EmittedEvent] = []
            if generation_result is not None:
                policy = self._perceived_performance_policy_provider.get_policy(context.agent.id)
                event_metadata = get_canrep_metadata(generation_result)

                if await policy.is_message_splitting_required(
                    loaded_context, generation_result.message
                ):
                    sub_messages = generation_result.message.strip().split("\n\n")
                else:
                    sub_messages = [generation_result.message.strip()]

                while sub_messages:
                    m = sub_messages.pop(0)

                    if await self._hooks.call_on_message_generated(loaded_context, payload=m):
                        # If we're in, the hook did not bail out.

                        handle = await event_emitter.emit_message_event(
                            trace_id=self._tracer.trace_id,
                            data=MessageEventData(
                                message=m,
                                participant=Participant(id=agent.id, display_name=agent.name),
                                draft=generation_result.draft,
                                canned_responses=generation_result.chosen_canned_responses,
                            )
                            if generation_result.draft
                            else MessageEventData(
                                message=m,
                                participant=Participant(id=agent.id, display_name=agent.name),
                            ),
                            metadata=event_metadata,
                        )
                        if not is_first_message_emitted:
                            await self._hist_ttfm_duration.record(
                                context.start_of_processing.elapsed * 1000
                            )
                            self._tracer.add_event("canrep.ttfm")
                            is_first_message_emitted = True

                        emitted_events.append(handle.event)

                        await context.event_emitter.emit_status_event(
                            trace_id=self._tracer.trace_id,
                            data={
                                "status": "ready",
                                "data": {},
                            },
                        )
                    else:
                        await context.event_emitter.emit_status_event(
                            trace_id=self._tracer.trace_id,
                            data={
                                "status": "ready",
                                "data": {},
                            },
                        )

                        return []

                    if next_message := sub_messages[0] if sub_messages else None:
                        policy = self._perceived_performance_policy_provider.get_policy(
                            context.agent.id
                        )

                        await policy.get_follow_up_delay()

                        await context.event_emitter.emit_status_event(
                            trace_id=self._tracer.trace_id,
                            data={
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
            return emitted_events

        def get_canrep_metadata(
            generation_result: _CannedResponseSelectionResult,
        ) -> Mapping[str, JSONSerializable] | None:
            if not generation_result.chosen_canned_responses:
                return None

            chosen_canrep = next(
                iter(
                    canrep
                    for canrep, _ in generation_result.rendered_canned_responses
                    if generation_result.chosen_canned_responses[0][0] == canrep.id
                ),
                None,
            )
            metadata = chosen_canrep.metadata if chosen_canrep else {}

            return metadata

        event_emitter = loaded_context.session_event_emitter
        agent = loaded_context.agent
        customer = loaded_context.customer
        session = loaded_context.session
        context_variables = loaded_context.state.context_variables
        interaction_history = loaded_context.interaction.events
        terms = list(loaded_context.state.glossary_terms)
        ordinary_guideline_matches = loaded_context.state.ordinary_guideline_matches
        journeys = loaded_context.state.journeys
        capabilities = loaded_context.state.capabilities
        tool_enabled_guideline_matches = loaded_context.state.tool_enabled_guideline_matches
        tool_insights = loaded_context.state.tool_insights
        staged_tool_events = loaded_context.state.tool_events
        staged_message_events = loaded_context.state.message_events
        additional_canned_response_fields = loaded_context.state.additional_canned_response_fields

        if (
            not interaction_history
            and not ordinary_guideline_matches
            and not tool_enabled_guideline_matches
        ):
            # No interaction and no guidelines that could trigger
            # a proactive start of the interaction
            self._logger.info("Skipping response; interaction is empty and there are no guidelines")
            return []

        context = CannedResponseContext(
            start_of_processing=loaded_context.creation,
            event_emitter=event_emitter,
            agent=agent,
            customer=customer,
            session=session,
            context_variables=context_variables,
            interaction_history=interaction_history,
            terms=terms,
            ordinary_guideline_matches=ordinary_guideline_matches,
            tool_enabled_guideline_matches=tool_enabled_guideline_matches,
            journeys=journeys,
            capabilities=capabilities,
            tool_insights=tool_insights,
            staged_tool_events=staged_tool_events,
            staged_message_events=staged_message_events,
            additional_canned_response_fields=additional_canned_response_fields,
        )

        canreps = await self._get_relevant_canned_responses(context)

        follow_up_selection_attempt_temperatures = (
            self._optimization_policy.get_message_generation_retry_temperatures(
                hints={"type": "canned-response-generation"}
            )
        )

        last_generation_exception: Exception | None = None
        generation_result: _CannedResponseSelectionResult | None = None
        generation_info: Mapping[str, GenerationInfo] = {}
        events: list[EmittedEvent] = []

        for generation_attempt in range(3):
            try:
                generation_info, generation_result = await self._generate_response(
                    loaded_context,
                    context,
                    canreps,
                    composition_mode,
                    temperature=follow_up_selection_attempt_temperatures[generation_attempt],
                )

                if latch:
                    latch.enable()

                if generation_result:
                    emitted_events = await output_messages(generation_result)
                    events += emitted_events

                    context.staged_message_events = (
                        list(context.staged_message_events) + emitted_events
                    )

                    break

            except Exception as exc:
                self._logger.warning(
                    f"Follow-up Generation attempt {generation_attempt} failed: {traceback.format_exception(exc)}"
                )

                last_generation_exception = exc

        follow_up_selection_attempt_temperatures = (
            self._optimization_policy.get_message_generation_retry_temperatures(
                hints={"type": "follow-up_canned_response-selection"}
            )
        )
        for generation_attempt in range(3):
            try:
                if generation_result and self._follow_ups_enabled:
                    (
                        follow_up_canrep_generation_info,
                        follow_up_canrep_response,
                    ) = await self.generate_follow_up_response(
                        context=context,
                        last_response_generation=generation_result,
                        temperature=follow_up_selection_attempt_temperatures[generation_attempt],
                    )

                    if follow_up_canrep_response:
                        await context.event_emitter.emit_status_event(
                            trace_id=self._tracer.trace_id,
                            data={
                                "status": "typing",
                                "data": {},
                            },
                        )

                        follow_up_delay_time = 1.5
                        await asyncio.sleep(follow_up_delay_time)

                        follow_up_response_events = await output_messages(follow_up_canrep_response)
                        events += follow_up_response_events
                        if not follow_up_response_events:
                            self._logger.debug(
                                "Skipping follow up response; no additional response deemed necessary"
                            )

                    return [
                        MessageEventComposition(
                            {**generation_info, **follow_up_canrep_generation_info}, events
                        )
                    ]

                return [MessageEventComposition({**generation_info}, events)]

            except Exception as exc:
                self._logger.warning(
                    f"Follow-up Generation attempt {generation_attempt} failed: {traceback.format_exception(exc)}"
                )
                last_generation_exception = exc

        raise MessageCompositionError() from last_generation_exception

    def enable_follow_ups(self) -> None:
        self._follow_ups_enabled = True

    def disable_follow_ups(self) -> None:
        self._follow_ups_enabled = False

    def _get_guideline_matches_text(
        self,
        ordinary: Sequence[GuidelineMatch],
        tool_enabled: Mapping[GuidelineMatch, Sequence[ToolId]],
        guideline_representations: dict[GuidelineId, GuidelineInternalRepresentation],
    ) -> str:
        all_matches = [
            match
            for match in chain(ordinary, tool_enabled)
            if internal_representation(match.guideline).action
        ]

        if not all_matches:
            return """
In formulating your reply, you are normally required to follow a number of behavioral guidelines.
However, in this case, no special behavioral guidelines were provided.
"""
        guidelines = []
        agent_intention_guidelines = []

        for i, p in enumerate(all_matches, start=1):
            internal_rep = internal_representation(p.guideline)

            if internal_rep.action:
                if internal_rep.condition:
                    guideline = f"Guideline #{i}) When {guideline_representations[p.guideline.id].condition}, then {guideline_representations[p.guideline.id].action}"
                else:
                    guideline = (
                        f"Guideline #{i}) {guideline_representations[p.guideline.id].action}"
                    )

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

    def _format_draft_shots(
        self,
        shots: Sequence[CannedResponseGeneratorDraftShot],
    ) -> str:
        return "\n".join(
            f"""
Example {i} - {shot.description}: ###
{self._format_draft_shot(shot)}
###
"""
            for i, shot in enumerate(shots, start=1)
        )

    def _format_draft_shot(
        self,
        shot: CannedResponseGeneratorDraftShot,
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
        session: Session,
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        capabilities: Sequence[Capability],
        ordinary_guideline_matches: Sequence[GuidelineMatch],
        journeys: Sequence[Journey],
        tool_enabled_guideline_matches: Mapping[GuidelineMatch, Sequence[ToolId]],
        staged_tool_events: Sequence[EmittedEvent],
        staged_message_events: Sequence[EmittedEvent],
        tool_insights: ToolInsights,
        shots: Sequence[CannedResponseGeneratorDraftShot],
    ) -> PromptBuilder:
        guideline_representations = {
            m.guideline.id: internal_representation(m.guideline)
            for m in chain(ordinary_guideline_matches, tool_enabled_guideline_matches)
        }

        builder = PromptBuilder(
            on_build=lambda prompt: self._logger.trace(f"Canned response Draft Prompt:\n{prompt}")
        )

        builder.add_section(
            name="canned-response-generator-draft-general-instructions",
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
        builder.add_customer_identity(customer, session)
        builder.add_section(
            name="canned-response-generator-draft-task-description",
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
5. RESOLUTION-AWARE MESSAGE ENDING: Do not ask the user if there is “anything else” you can help with until their current request or problem is fully resolved. Treat a request as resolved only if a) the user explicitly confirms it; b) the original question has been answered in full; or c) all stated requirements are met. If resolution is unclear, continue engaging on the current topic instead of prompting for new topics.
6. ONLY OFFER SERVICES FROM THIS PROMPT: Offer only services explicitly mentioned within this prompt (via guidelines, capabilities section, or other documented features). Never assume or infer additional services based on general knowledge. For example, if representing a pizza store, do not offer delivery unless it's specifically documented here—even if delivery is standard for pizza stores.
7. ONLY USE FACTUAL INFORMATION FROM THIS PROMPT: Use only factual information explicitly provided in this prompt. Do not supplement with external knowledge or assumptions. For example, even if you know a business's actual address, only share it if it appears in this prompt or interaction history. Treat all information outside this context as unknown. This includes not claiming to perform actions or complete processes unless those specific capabilities are documented in this prompt.
8. ACKNOWLEDGE INFORMATION GAPS: When users request information not contained in this prompt, directly acknowledge the limitation rather than improvising. State clearly that the requested information is not available to you, then offer assistance within your documented scope.
9. THIS IS NOT A ROLE PLAY: This is a real scenario and not a role-play. Your actions have real world consequences. Only respond with what is explicitly stated in this prompt.
Based on previous experience, you seem too eager to please the customer by offering services and information that is not sourced from this prompt. Be extra careful regarding the last 3 instructions.
""",
            props={},
        )

        if not interaction_history or all(
            [event.kind != EventKind.MESSAGE for event in interaction_history]
        ):
            builder.add_section(
                name="canned-response-generator-draft-initial-message-instructions",
                template="""
The interaction with the user has just began, and no messages were sent by either party.
If told so by a guideline or some other contextual condition, send the first message. Otherwise, do not produce a reply (canned response is null).
If you decide not to emit a message, output the following:
{{
    "last_message_of_user": "<user's last message>",
    "guidelines": [<list of strings- a re-statement of all guidelines>],
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
                name="canned-response-generator-draft-ongoing-interaction-instructions",
                template="""
Since the interaction with the user is already ongoing, always produce a reply to the user's last message.
The only exception where you may not produce a reply (i.e., setting message = null) is if the user, or a provided guideline, explicitly asked you not to respond.
In all other cases, even if the user is indicating that the conversation is over, you must produce a reply.
                """,
                props={},
            )

        builder.add_section(
            name="canned-response-generator-draft-revision-mechanism",
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
            name="canned-response-generator-draft-examples",
            template="""
EXAMPLES
-----------------
{formatted_shots}
""",
            props={
                "formatted_shots": self._format_draft_shots(shots),
                "shots": shots,
            },
        )
        builder.add_glossary(terms)
        builder.add_context_variables(context_variables)
        builder.add_capabilities_for_message_generation(capabilities)
        builder.add_guidelines_for_message_generation(
            ordinary_guideline_matches,
            tool_enabled_guideline_matches,
            guideline_representations,
        )
        builder.add_low_criticality_guidelines(
            ordinary_guideline_matches,
            tool_enabled_guideline_matches,
            guideline_representations,
        )
        builder.add_interaction_history_for_message_generation(
            interaction_history,
            staged_events=staged_message_events,
        )
        builder.add_staged_tool_events(staged_tool_events)

        if tool_insights.missing_data:
            builder.add_section(
                name="canned-response-generator-draft-missing-data-for-tools",
                template="""
MISSING REQUIRED DATA FOR TOOL CALLS:
-------------------------------------
The following is a description of missing data that has been deemed necessary
in order to run tools. The tools needed to run at this stage would have run if they only had this data available.
If it makes sense in the current state of the interaction, inform the user about this missing data: ###
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
                name="canned-response-generator-invalid-data-for-tools",
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
            name="canned-response-generator-output-format",
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
                    if internal_representation(g.guideline).action
                ],
                "guideline_representations": guideline_representations,
            },
        )
        builder.add_section(
            name="canned-response-generator-draft-disclaimer",
            template="""REMINDER: Only offer information and offer services that are sourced from this prompt. Never use your intrinsic knowledge to offer services or provide information.""",
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
            [f'"{g.guideline}"' for g in guidelines if internal_representation(g.guideline).action]
        )

        return f"""
{{
    "last_message_of_user": "{last_user_message}",
    "guidelines": [{guidelines_list_text}],
    "insights": [<Up to 3 original insights to adhere to>],
    "response_preamble_that_was_already_sent": "{agent_preamble}",
    "response_body": "<response message text (that would immediately follow the preamble)>"
}}
###"""

    def _build_selection_prompt(
        self,
        context: CannedResponseContext,
        draft_message: str,
        canned_responses: Sequence[tuple[CannedResponse, str]],
    ) -> PromptBuilder:
        builder = PromptBuilder(
            on_build=lambda prompt: self._logger.trace(
                f"Canned Response Selection Prompt:\n{prompt}"
            )
        )

        formatted_canreps = "\n".join(
            [f'Template ID: {canrep[0].id} """\n{canrep[1]}\n"""' for canrep in canned_responses]
        )

        builder.add_section(
            name="canned-response-generator-selection-task-description",
            template="""
1. You are an AI agent who is part of a system that interacts with a user.
2. A draft reply to the user has been generated by a human operator.
3. You are presented with a number of Jinja2 reply templates to choose from. These templates have been pre-approved by business stakeholders for producing fluent customer-facing AI conversations.
4. Your role is to choose (classify) the pre-approved reply template that MOST faithfully captures the human operator's draft reply.
5. Note that there may be multiple relevant choices. Out of those, you must choose the MOST suitable one that is MOST LIKE the human operator's draft reply.
6. In cases where there are multiple templates that provide a partial match, you may encounter different types of partial matches. Prefer templates that do not deviate from the draft message semantically, even if they only address part of the draft message. They are better than a template that would have captured multiple parts of the draft message while introducing semantic deviations. In other words, better to match fewer parts with higher semantic fidelity than to match more parts with lower semantic fidelity.
7. If there is any noticeable semantic deviation between the draft message and the template, i.e., the draft says "Do X" and the template says "Do Y" (even if Y is a sibling concept under the same category as X), you should not choose that template, even if it captures other parts of the draft message. We want to maintain true fidelity with the draft message.
8. If the deviation between the draft and the template is quantitative in nature (e.g., the draft says "5 apples" and the template says "10 apples"), you should assume that the template has it right. Don't consider this a failure, as the template will definitely contain the correct information. So as long as it's a good *qualitative match*, you can assume that the *quantitative part* will be handled correctly.
9. Keep in mind that these are Jinja 2 *templates*. Some of them refer to variables or contain procedural instructions. These will be substituted by real values and rendered later. You can assume that such substitution will be handled well to account for the data provided in the draft message! FYI, if you encounter a variable {{generative.<something>}}, that means that it will later be substituted with a dynamic, flexible, generated value based on the appropriate context. You just need to choose the most viable reply template to use, and assume it will be filled and rendered properly later.""",
        )
        builder.add_agent_identity(context.agent)
        builder.add_customer_identity(context.customer, context.session)
        builder.add_glossary(context.terms)
        builder.add_interaction_history_for_message_generation(
            context.interaction_history,
            staged_events=context.staged_message_events,
        )

        builder.add_section(
            name="canned-response-generator-selection-templates",
            template="""
Pre-approved reply templates: ###
{formatted_canned_responses}
###
""",
            props={
                "formatted_canned_responses": formatted_canreps,
            },
        )
        builder.add_guidelines_for_canrep_selection(
            list(chain(context.ordinary_guideline_matches, context.tool_enabled_guideline_matches))
        )
        builder.add_section(
            name="canned-response-generator-selection-output-format",
            template="""
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
            },
        )

        return builder

    async def _generate_response(
        self,
        loaded_context: EngineContext,
        context: CannedResponseContext,
        canned_responses: Sequence[CannedResponse],
        composition_mode: CompositionMode,
        temperature: float,
    ) -> tuple[Mapping[str, GenerationInfo], Optional[_CannedResponseSelectionResult]]:
        # This will be needed throughout the process for emitting status events
        direct_draft_output_mode = (
            not canned_responses and composition_mode != CompositionMode.CANNED_STRICT
        )

        # Step 1: Generate the draft message
        draft_prompt = self._build_draft_prompt(
            agent=context.agent,
            context_variables=context.context_variables,
            customer=context.customer,
            session=context.session,
            interaction_history=context.interaction_history,
            terms=context.terms,
            ordinary_guideline_matches=context.ordinary_guideline_matches,
            journeys=context.journeys,
            capabilities=context.capabilities,
            tool_enabled_guideline_matches=context.tool_enabled_guideline_matches,
            staged_tool_events=context.staged_tool_events,
            staged_message_events=context.staged_message_events,
            tool_insights=context.tool_insights,
            shots=await self.draft_generation_shots(composition_mode),
        )

        if direct_draft_output_mode:
            await context.event_emitter.emit_status_event(
                trace_id=self._tracer.trace_id,
                data={
                    "status": "typing",
                    "data": {},
                },
            )
        elif not canned_responses and composition_mode == CompositionMode.CANNED_STRICT:
            no_match_canrep = await self._no_match_provider.get_response(loaded_context, None)

            return {}, _CannedResponseSelectionResult(
                message=no_match_canrep.value,
                draft=None,
                rendered_canned_responses=[],
                chosen_canned_responses=[(no_match_canrep.id, no_match_canrep.value)],
            )
        else:
            await context.event_emitter.emit_status_event(
                trace_id=self._tracer.trace_id,
                data={
                    "status": "processing",
                    "data": {"stage": "Articulating"},
                },
            )

        async with self._hist_draft_duration.measure():
            draft_response = await self._canrep_draft_generator.generate(
                prompt=draft_prompt,
                hints={"temperature": temperature},
            )

        self._logger.trace(
            f"Canned Response Draft Completion:\n{draft_response.content.model_dump_json(indent=2)}"
        )

        draft_message = draft_response.content.response_body

        if not draft_message:
            return {"draft": draft_response.info}, None

        if direct_draft_output_mode:
            return {
                "draft": draft_response.info,
            }, _CannedResponseSelectionResult(
                message=draft_message,
                draft=None,
                rendered_canned_responses=[],
                chosen_canned_responses=[],
            )

        # Check if, according to the hooks, we should consider the draft
        # good enough to be sent as-is, without choosing a canned response.
        if not await self._hooks.call_on_draft_generated(loaded_context, payload=draft_message):
            # This means it's good enough to be sent as-is.
            return {
                "draft": draft_response.info,
            }, _CannedResponseSelectionResult(
                message=draft_message,
                draft=None,
                rendered_canned_responses=[],
                chosen_canned_responses=[],
            )

        await context.event_emitter.emit_status_event(
            trace_id=self._tracer.trace_id,
            data={
                "status": "typing",
                "data": {},
            },
        )

        # Step 2: Select the most relevant canned response templates based on the draft message
        async with self._hist_retrieval_duration.measure():
            relevance_scores = await self._canned_response_store.filter_relevant_canned_responses(
                query=draft_message,
                available_canned_responses=canned_responses,
                max_count=30,
            )

            relevant_canreps = set(
                r.canned_response
                for r in relevance_scores
                if r.score >= self.candidate_similarity_threshold
            )

            # Filtering based on similarity will have taken out all transient
            # ones, so we need to bring them back.
            relevant_canreps.update(
                [r for r in canned_responses if r.id == CannedResponse.TRANSIENT_ID]
            )

            relevant_canreps.update(
                await self._entity_queries.find_canned_responses_for_guidelines(
                    guidelines=[m.guideline for m in context.guideline_matches]
                )
            )

            if not relevant_canreps and composition_mode != CompositionMode.CANNED_STRICT:
                self._logger.debug(
                    "Skipping canned response selection; no relevant canned responses found"
                )

                return {
                    "draft": draft_response.info,
                }, _CannedResponseSelectionResult(
                    message=draft_message,
                    draft=None,
                    rendered_canned_responses=[],
                    chosen_canned_responses=[],
                )

        # Step 3: Pre-render these templates so that matching works better
        async with self._hist_render_duration.measure():
            rendered_canreps = [
                (r.response, str(r.rendered_text))
                for r in await self._render_responses(
                    context=context,
                    responses=relevant_canreps,
                )
                if not r.failed
            ]

        # Step 4.1: In composited mode, recompose the draft message with the style of the rendered canned responses
        if composition_mode == CompositionMode.CANNED_COMPOSITED:
            async with self._hist_recompose_duration.measure():
                recomposition_generation_info, composited_message = await self._recompose(
                    context=context,
                    draft_message=draft_message,
                    reference_messages=[canrep[1] for canrep in rendered_canreps],
                )

                return {
                    "draft": draft_response.info,
                    "composition": recomposition_generation_info,
                }, _CannedResponseSelectionResult(
                    message=composited_message,
                    draft=draft_response.content.response_body,
                    rendered_canned_responses=rendered_canreps,
                    chosen_canned_responses=[],
                )

        # Step 4.2: In non-composited mode, try to match the draft message with one of the rendered canned responses
        async with self._hist_selection_duration.measure():
            selection_response = await self._canrep_selection_generator.generate(
                prompt=self._build_selection_prompt(
                    context=context,
                    draft_message=draft_message,
                    canned_responses=rendered_canreps,
                ),
                hints={"temperature": 0.1},
            )

        self._logger.trace(
            f"Canned Response Selection Completion:\n{selection_response.content.model_dump_json(indent=2)}"
        )

        # Step 5: Respond based on the match quality

        # Step 5.1: Assuming no match or a low-quality match
        if (
            selection_response.content.match_quality not in ["partial", "high"]
            or not selection_response.content.chosen_template_id
        ):
            if composition_mode == CompositionMode.CANNED_STRICT:
                # Return a no-match message
                self._logger.warning(
                    "Failed to find relevant canned responses. Please review canned response selection prompt and completion."
                )

                no_match_canrep = await self._no_match_provider.get_response(
                    loaded_context, draft_message
                )

                return {
                    "draft": draft_response.info,
                    "selection": selection_response.info,
                }, _CannedResponseSelectionResult(
                    message=no_match_canrep.value,
                    draft=draft_response.content.response_body,
                    rendered_canned_responses=rendered_canreps,
                    chosen_canned_responses=[(no_match_canrep.id, no_match_canrep.value)],
                )
            else:
                # Return the draft message as the response
                return {
                    "draft": draft_response.info,
                    "selection": selection_response.info,
                }, _CannedResponseSelectionResult(
                    message=draft_message,
                    draft=draft_response.content.response_body,
                    rendered_canned_responses=rendered_canreps,
                    chosen_canned_responses=[],
                )

        # Step 5.2: Assuming a partial match in non-strict mode
        if (
            selection_response.content.match_quality == "partial"
            and composition_mode != CompositionMode.CANNED_STRICT
        ):
            # Return the draft message as the response
            return {
                "draft": draft_response.info,
                "selection": selection_response.info,
            }, _CannedResponseSelectionResult(
                message=draft_message,
                draft=draft_response.content.response_body,
                rendered_canned_responses=rendered_canreps,
                chosen_canned_responses=[],
            )

        # Step 5.3: Assuming a high-quality match or a partial match in strict mode
        selected_canrep_id = CannedResponseId(selection_response.content.chosen_template_id)
        rendered_canned_response = next(
            (value for canrep, value in rendered_canreps if canrep.id == selected_canrep_id),
            None,
        )

        if not rendered_canned_response:
            self._logger.error(
                "Invalid canned response ID choice. Please review canned response selection prompt and completion."
            )

            no_match_canrep = await self._no_match_provider.get_response(
                loaded_context, draft_message
            )

            return {
                "draft": draft_response.info,
                "selection": selection_response.info,
            }, _CannedResponseSelectionResult(
                message=no_match_canrep.value,
                draft=draft_response.content.response_body,
                rendered_canned_responses=rendered_canreps,
                chosen_canned_responses=[(no_match_canrep.id, no_match_canrep.value)],
            )

        return {
            "draft": draft_response.info,
            "selection": selection_response.info,
        }, _CannedResponseSelectionResult(
            message=rendered_canned_response,
            draft=draft_response.content.response_body,
            rendered_canned_responses=rendered_canreps,
            chosen_canned_responses=[(selected_canrep_id, rendered_canned_response)],
        )

    async def _render_responses(
        self,
        context: CannedResponseContext,
        responses: Iterable[CannedResponse],
    ) -> Sequence[_CannedResponseRenderResult]:
        render_tasks = [self._render_response(context, r) for r in responses]
        return await safe_gather(*render_tasks)

    async def _render_response(
        self,
        context: CannedResponseContext,
        response: CannedResponse,
    ) -> _CannedResponseRenderResult:
        faulty_field_name: str | None = None

        try:
            args = {}

            for field_name in _get_response_template_fields(response.value):
                success, value = await self._field_extractor.extract(
                    response.value,
                    field_name,
                    context,
                )

                if success:
                    args[field_name] = value
                else:
                    faulty_field_name = field_name
                    self._logger.error(f"CannedResponse field extraction: missing '{field_name}'")
                    raise KeyError(f"Missing field '{field_name}' in canned response")

            result = jinja2.Template(response.value).render(**args)

            return _CannedResponseRenderResult(
                response=response,
                failed=False,
                rendered_text=result,
            )
        except Exception as exc:
            # TODO: Once we have the extractor registry, maybe control this using
            # something like "excluded from error" extractors or field names.
            if faulty_field_name != "generative":
                self._logger.error(
                    f"Failed to pre-render canned response for matching '{response.id}' ('{response.value}')"
                )
                self._logger.error(
                    f"Canned response rendering failed: {traceback.format_exception(exc)}"
                )

            return _CannedResponseRenderResult(
                response=response,
                failed=True,
                rendered_text=None,
            )

    async def _recompose(
        self,
        context: CannedResponseContext,
        draft_message: str,
        reference_messages: list[str],
    ) -> tuple[GenerationInfo, str]:
        builder = PromptBuilder(
            on_build=lambda prompt: self._logger.trace(f"Composition Prompt:\n{prompt}")
        )

        reference_messages_text = "\n\n".join(
            [f"{i + 1}) {msg}" for i, msg in enumerate(reference_messages)]
        )

        builder.add_agent_identity(context.agent)

        builder.add_section(
            name="canned-response-generator-composition",
            template="""\
Task Description
----------------
You are given two message types:
1. A single draft message
2. One or more style reference messages

The draft message contains what should be said right now.
The style reference messages teach you what communication style to try to copy.

You must say what the draft message says, but capture the tone, style, and choice of words in the reference messages as precisely as you can.

Make sure NOT to add, remove, or hallucinate information nor add or remove key words (nouns, verbs) to the message.

IMPORTANT NOTE: Always try to separate points in your message by 2 newlines (\\n\\n) — even if the reference messages don't do so. You may do this zero or multiple times in the message, as needed. Pay extra attention to this requirement. For example, here's what you should separate:
1. Answering one thing and then another thing -- Put two newlines in between
2. Answering one thing and then asking a follow-up question (e.g., Should I... / Can I... / Want me to... / etc.) -- Put two newlines in between
3. An initial acknowledgement (Sure... / Sorry... / Thanks...) or greeting (Hey... / Good day...) and actual follow-up statements -- Put two newlines in between

Draft message: ###
{draft_message}
###

Style reference messages: ###
{reference_messages_text}
###

Respond with a JSON object {{ "revised_canned_response": "<message_with_points_separated_by_double_newlines>" }}
""",
            props={
                "draft_message": draft_message,
                "reference_messages": reference_messages,
                "reference_messages_text": reference_messages_text,
            },
        )

        result = await self._canrep_composition_generator.generate(
            builder,
            hints={"temperature": 1},
        )

        self._logger.trace(f"Composition Completion:\n{result.content.model_dump_json(indent=2)}")

        return result.info, result.content.revised_canned_response

    def _format_follow_up_generation_shot(self, shot: FollowUpCannedResponseSelectionShot) -> str:
        formatted_shot = ""

        formatted_shot += f"""
Draft: {shot.draft}
Last agent message: {shot.last_agent_message}
"""

        candidate_canreps = "\n".join(
            f"{canrep_id}) {canrep}" for canrep_id, canrep in shot.canned_responses.items()
        )
        formatted_shot += f"""
- **Candidate Templates**:
{candidate_canreps}

"""

        formatted_shot += f"""
- **Expected Result**:
```json
{json.dumps(shot.expected_result.model_dump(mode="json", exclude_unset=True), indent=2)}
```
"""

        return formatted_shot

    def _format_follow_up_generation_shots(
        self,
        shots: Sequence[FollowUpCannedResponseSelectionShot],
    ) -> str:
        return "\n".join(
            f"""
Example {i} - {shot.description}: ###
{self._format_follow_up_generation_shot(shot)}
###
    """
            for i, shot in enumerate(shots, 1)
        )

    def _build_follow_up_canned_response_prompt(
        self,
        context: CannedResponseContext,
        draft_message: str,
        canned_responses: Mapping[str, str],
        shots: Sequence[FollowUpCannedResponseSelectionShot],
    ) -> PromptBuilder:
        outputted_message: str | None = next(
            (
                cast(Mapping[str, str], e.data).get("message", None)
                for e in reversed(context.staged_message_events)
                if e.source == EventSource.AI_AGENT
            ),
            None,
        )
        if not outputted_message:
            outputted_message = next(
                (
                    cast(Mapping[str, str], e.data).get("message", None)
                    for e in reversed(context.interaction_history)
                    if e.source == EventSource.AI_AGENT and e.kind == EventKind.MESSAGE
                ),
                None,
            )

        builder = PromptBuilder(
            on_build=lambda prompt: self._logger.trace(
                f"Follow-up Canned Response Selection Prompt:\n{prompt}"
            )
        )

        formatted_canreps = "\n".join(
            [f'Template ID: "{id}" """\n{canrep}\n"""' for id, canrep in canned_responses.items()]
        )

        builder.add_section(
            name="follow-up-canned-response-generator-selection-general_instructions",
            template="""

GENERAL INSTRUCTIONS
-----------------
You are an AI agent who is part of a system that interacts with a user. The current state of this interaction will be provided to you later in this message.
A draft reply to the user has been generated by a human operator. Based on this draft, a pre-approved template response was previously selected and sent to the customer.
In certain cases, this singular template does not fully transmit the draft crafted by the human operator. In those cases, an additional template may be transmitted to cover whatever part of the draft that was not covered by the previously outputted pre-approved template.
Key Terms:
- Template: Pre-approved response patterns in Jinja2 format that have been vetted by business stakeholders for customer-facing AI conversations
- Draft message: The original response crafted by the human operator
- Behavioral guidelines: Instructions in the form of "when <X> then do <Y>" which you must follow
""",
        )

        builder.add_section(
            name="follow-up-canned-response-generator-selection-task-description",
            template="""
TASK DESCRIPTION
-----------------
Your task is to evaluate whether an additional template should be transmitted to the customer, and if necessary, choose the specific template that best captures the remainder of the human operator's draft reply.
You are provided with a number of pre-approved templates to choose from. These templates have been vetted by business stakeholders for producing fluent customer-facing AI conversations.
Perform your task as follows:
1. Identify Unsatisfied Guidelines: Document which behavioral guidelines (instructions in the form of "when <X> then do <Y>" which you must follow) aren't satisfied by the last agent's message under the key "unsatisfied_guidelines".
2. Analyze Coverage Gap: Examine the draft message and the message already outputted to the customer. Write down the parts of the draft message that are not covered by the already outputted message under the key "remaining_message_draft". If the outputted message already includes all the information from the draft, then output an empty string under the key "remaining_message_draft".
3. Evaluate Need for Additional Response: Examine whether an additional response is required, and if so, which template best captures the remaining message draft. Document your thought process under the key "tldr". Prefer brevity, use fewer words when possible.
 - Prefer outputting an additional response if a guideline that is currently unsatisfied can be satisfied by one of the available templates
 - If no guideline is unsatisfied, or no template satisfies the unsatisfied guidelines, only output an additional response if it greatly matches the remaining message draft
4. Make Decision: Decide whether an additional template can capture your chosen "remaining_message_draft". Document your decision under the key "additional_response_required".
5. Select Template (if needed): If "additional_response_required" is True, then choose the template that best captures the "remaining_message_draft". Output the ID of your chosen template under the key "additional_template_id".
6. Assess Match Quality (if a template was selected): Evaluate how well the chosen template captures the remaining message draft. Output your evaluation under the key "match_quality". You must choose one of the following options:
    a. "low": You couldn't find a template that even comes close, or any such template also adds new information that is not in the draft.
    b. "partial": You found a template that conveys at least some of the draft message's content, without adding information that is not in the draft or the active guidelines.
    c. "high": You found a template that captures the draft message in both form and function. Note that it doesn't have to be a full, exact match.

Some nuances regarding choosing the correct template:
 - Pay special attention to whether the last outputted message already captures the draft. If it does, no further response is necessary, even if another candidate canned response matches the draft.
 - There may be multiple relevant choices for the same purpose. Choose the MOST suitable one that is MOST LIKE the remaining draft
 - When multiple templates provide partial matches, prefer templates that do not deviate from the remaining message draft semantically, even if they only address part of the draft message
 - If the missing part of the draft includes multiple unrelated components that would each require different templates, prioritize the template that addresses the most critical information for customer understanding and conversation progression. Choose the component that is essential for the customer to take their next action or properly understand the agent's response.
 - If there is any noticeable semantic deviation between the draft message and a template (e.g., the draft says "Do X" and the template says "Do Y"), do not choose that template, even if it captures other parts of the remaining message draft
 - Prioritize factual accuracy. Never output a template that conveys information which contradicts the draft. Prefer outputting a different template, or even no template whatsoever.
    - For example, if the draft mentions that a certain action takes 10 minutes to be completed, prefer a template that mentions it taking less than a day to one that says that action is completed immediately. Err on the side of caution.

 """,
        )

        builder.add_section(
            name="follow-up-canned-response-generator-selection-examples",
            template="""
EXAMPLES
-----------------
{formatted_shots}
""",
            props={"formatted_shots": self._format_follow_up_generation_shots(shots)},
        )

        builder.add_agent_identity(context.agent)
        builder.add_customer_identity(context.customer, context.session)
        builder.add_interaction_history(
            context.interaction_history,
            staged_events=context.staged_message_events,
        )

        builder.add_section(
            name="follow-up-canned-response-generator-inputs",
            template="""
INPUTS
---------------
Draft message: ###
{draft}
###
Message already sent to the customer: ###
{last_agent_message}
###
Pre-approved reply templates: ###
{formatted_canned_responses}
###
""",
            props={
                "draft": draft_message,
                "last_agent_message": outputted_message or "",
                "formatted_canned_responses": formatted_canreps,
            },
        )

        builder.add_guidelines_for_canrep_selection(
            list(chain(context.ordinary_guideline_matches, context.tool_enabled_guideline_matches))
        )

        builder.add_section(
            name="follow-up-canned-response-generator-selection-output_format",
            template="""
OUTPUT FORMAT
-----------------
Output a JSON object with three properties:
{{
    "remaining_message_draft": "<str, rephrasing of the part of the draft that isn't covered by the last outputted message>"
    "unsatisfied_guidelines": "<str, restatement of all guidelines that were not satisfied by the last outputted message. Only restate the actionable part of the guideline (the one after 'then')>"
    "tldr": "<str, brief explanation of the reasoning behind whether an additional response is required, and which template best encapsulates it>",
    "additional_response_required": <bool, if False, all remaining keys should be omitted>,
    "additional_template_id": "<str, ID of the chosen template>",
    "match_quality": "<str, either "high", "partial" or "low" depending on how similar the chosen template is to the remaining message draft>",
}}
""",
            props={
                "draft": draft_message,
                "last_agent_message": outputted_message or "",
            },
        )

        return builder

    async def generate_follow_up_response(
        self,
        context: CannedResponseContext,
        last_response_generation: _CannedResponseSelectionResult,
        temperature: float,
    ) -> tuple[Mapping[str, GenerationInfo], Optional[_CannedResponseSelectionResult]]:
        selection_result: Optional[_CannedResponseSelectionResult] = None
        if (
            context.agent.composition_mode != CompositionMode.CANNED_STRICT
            or last_response_generation.draft is None
        ):
            return {}, None

        try:
            outputted_canreps_ids = [
                cid for cid, value in last_response_generation.chosen_canned_responses
            ]

            filtered_rendered_canreps: Sequence[tuple[CannedResponse, str]] = [
                (canrep, value)
                for canrep, value in last_response_generation.rendered_canned_responses
                if canrep.id not in outputted_canreps_ids
            ]  # removes outputted response/s

            chronological_id_rendered_canreps = {
                str(i): (canrep, value)
                for i, (canrep, value) in enumerate(filtered_rendered_canreps, start=1)
            }

            prompt = self._build_follow_up_canned_response_prompt(
                context=context,
                draft_message=last_response_generation.draft,
                canned_responses={
                    i: canrep for i, (cid, canrep) in chronological_id_rendered_canreps.items()
                },
                shots=follow_up_generation_shots,
            )

            response = await self._follow_up_canrep_generator.generate(
                prompt=prompt,
                hints={"temperature": temperature},
            )

            self._logger.trace(
                f"Follow-up Canned Response Draft Completion:\n{response.content.model_dump_json(indent=2)}"
            )

            if (
                response.content.additional_response_required
                and response.content.additional_template_id
            ):
                chosen_canrep = chronological_id_rendered_canreps.get(
                    response.content.additional_template_id, None
                )

                if chosen_canrep is None:
                    self._logger.warning(
                        "Follow-up canned response returned an Illegal canned response ID"
                    )

                selection_result = (
                    _CannedResponseSelectionResult(
                        message=chosen_canrep[1],
                        draft=response.content.remaining_message_draft,
                        rendered_canned_responses=filtered_rendered_canreps,
                        chosen_canned_responses=[(chosen_canrep[0].id, chosen_canrep[1])],
                    )
                    if chosen_canrep
                    else None
                )

            return ({"follow-up": response.info}, selection_result)

        except Exception as e:
            self._logger.error(f"Failed to choose follow-up canned response: {e}")
            return ({}, None)


def shot_canned_canned_response_id(number: int) -> str:
    return f"<example-only-canned-response--{number}--do-not-use-in-your-completion>"


draft_generation_example_1_expected = CannedResponseDraftSchema(
    last_message_of_user="Hi, I'd like an onion cheeseburger please.",
    guidelines=[
        "When the user chooses and orders a burger, then provide it",
        "When the user chooses specific ingredients on the burger, only provide those ingredients if we have them fresh in stock; otherwise, reject the order",
    ],
    insights=[
        "As appears in the tool results, all of our cheese has expired and is currently out of stock",
        "The user is a long-time user and we should treat him with extra respect",
    ],
    response_preamble_that_was_already_sent="Let me check",
    response_body="Unfortunately we're out of cheese. Would you like anything else instead?",
)

draft_generation_example_1_shot = CannedResponseGeneratorDraftShot(
    composition_modes=[CompositionMode.CANNED_FLUID],
    description="A reply where one instruction was prioritized over another",
    expected_result=draft_generation_example_1_expected,
)


draft_generation_example_2_expected = CannedResponseDraftSchema(
    last_message_of_user="Hi there, can I get something to drink? What do you have on tap?",
    guidelines=["When the user asks for a drink, check the menu and offer what's on it"],
    insights=[
        "According to contextual information about the user, this is their first time here",
        "There's no menu information in my context",
    ],
    response_preamble_that_was_already_sent="Just a moment",
    response_body="I'm sorry, but I'm having trouble accessing our menu at the moment. This isn't a great first impression! Can I possibly help you with anything else?",
)

draft_generation_example_2_shot = CannedResponseGeneratorDraftShot(
    composition_modes=[
        CompositionMode.CANNED_STRICT,
        CompositionMode.CANNED_COMPOSITED,
        CompositionMode.CANNED_FLUID,
    ],
    description="Non-adherence to guideline due to missing data",
    expected_result=draft_generation_example_2_expected,
)


draft_generation_example_3_expected = CannedResponseDraftSchema(
    last_message_of_user=("Hey, how can I contact customer support?"),
    guidelines=[],
    insights=[
        "When I cannot help with a topic, I should tell the user I can't help with it",
    ],
    response_preamble_that_was_already_sent="Hello",
    response_body="Unfortunately, I cannot refer you to live customer support. Is there anything else I can help you with?",
)

draft_generation_example_3_shot = CannedResponseGeneratorDraftShot(
    composition_modes=[
        CompositionMode.CANNED_STRICT,
        CompositionMode.CANNED_COMPOSITED,
        CompositionMode.CANNED_FLUID,
    ],
    description="An insight is derived and followed on not offering to help with something you don't know about",
    expected_result=draft_generation_example_3_expected,
)


_draft_generation_baseline_shots: Sequence[CannedResponseGeneratorDraftShot] = [
    draft_generation_example_1_shot,
    draft_generation_example_2_shot,
    draft_generation_example_3_shot,
]

draft_generation_shot_collection = ShotCollection[CannedResponseGeneratorDraftShot](
    _draft_generation_baseline_shots
)


follow_up_generation_example_1_expected = FollowUpCannedResponseSelectionSchema(
    remaining_message_draft="You can call a human representative at 1-800-123-1234.",
    unsatisfied_guidelines="",
    tldr="We haven't sent out our customer support number, so the draft is not fully transmitted. Template #2 has the relevant number, so we should send it to the customer.",
    additional_response_required=True,
    additional_template_id="2",
    match_quality="high",
)

follow_up_generation_example_1_shot = FollowUpCannedResponseSelectionShot(
    description="A simple example where a follow-up response is necessary",
    draft=cast(str, follow_up_generation_example_1_expected.remaining_message_draft),
    canned_responses={
        "1": "Your account status is currently set to Active. You can change your account status using this chat, or by calling a customer support representative at 1-800-123-1234.",
        "2": "Our customer support number is 1-800-123-1234. You can call a human representative at this number.",
        "3": "Sorry, I didn't catch that. Could you please ask again in a different way?",
        "4": "You can change your account status to either Active, Automatic, or Closed.",
        "5": "Our customer support line is open from 8 AM to 8 PM Monday through Friday. You can call us at 1-800-123-1234.",
    },
    last_agent_message="I can assist you with altering the status of your account, or you can call a human representative.",
    expected_result=follow_up_generation_example_1_expected,
)


follow_up_generation_example_2_expected = FollowUpCannedResponseSelectionSchema(
    remaining_message_draft="Thank you for your purchase!",
    unsatisfied_guidelines="",
    tldr="The remaining part of the draft does not contain any critical information, and no template matches it, so no further response is necessary.",
    additional_response_required=False,
)

follow_up_generation_example_2_shot = FollowUpCannedResponseSelectionShot(
    description="A simple example where a follow-up response is not necessary",
    draft=cast(str, follow_up_generation_example_2_expected.remaining_message_draft),
    canned_responses={
        "1": "The order will be shipped to you in up to 10 business days. Thank you for your purchase!",
        "2": "Domestic orders are shipped through UPS",
        "3": "I'm here to help! What can I do for you today?",
        "4": "Your purchase is complete and will be shipped to you shortly!",
        "5": "You can track your order status on our website at verygoodstore.com",
    },
    last_agent_message="The order will be shipped to you in 5-7 business days",
    expected_result=follow_up_generation_example_2_expected,
)

follow_up_generation_example_3_expected = FollowUpCannedResponseSelectionSchema(
    remaining_message_draft="Thank you for your purchase!",
    unsatisfied_guidelines="",
    tldr="Templates 1 and 4 both capture missing parts of the draft. Template 1 is more important as it mentions potential health concerns, so it should be sent out first.",
    additional_response_required=True,
    additional_template_id="1",
    match_quality="partial",
)
follow_up_generation_example_3_shot = FollowUpCannedResponseSelectionShot(
    description="An example where one response is prioritized for its importance",
    draft="Your table is booked! Since you mentioned allergies, please note that our kitchen contains peanuts. You'll be able to get a souvenir from our store after your meal.",
    canned_responses={
        "1": "Please note that all dishes may contain peanuts",
        "2": "Please inform us of any allergies you or your party have",
        "3": "Thank you for coming in!",
        "4": "Our souvenir shop is available for all diners after their meal",
        "5": "Would you like to book another table?",
    },
    last_agent_message="Your table has been booked!",
    expected_result=follow_up_generation_example_3_expected,
)

follow_up_generation_example_4_expected = FollowUpCannedResponseSelectionSchema(
    remaining_message_draft="",
    unsatisfied_guidelines="",
    tldr="The last outputted message already captures the draft. Template 1 matches the draft, but it adds no new information compared to the last outputted message.",
    additional_response_required=False,
)
follow_up_generation_example_4_shot = FollowUpCannedResponseSelectionShot(
    description="An example where the draft was already captured by the last response. Assume there's an active guideline instructing the agent to inform the customer about our returns policy.",
    draft="Unopened items can be returned for up to 30 days from the date of purchase",
    canned_responses={
        "1": "Any item can be returned for up to 30 days from the date of purchase, given that it has not been opened.",
        "2": "Please check our website for more information about our returns policy.",
        "3": "Your items will be returned to us within 30 days.",
        "4": "Sorry, I didn't catch that",
    },
    last_agent_message="Of course! You may return your items for up to a month if they have not been opened.",
    expected_result=follow_up_generation_example_4_expected,
)

follow_up_generation_shots: Sequence[FollowUpCannedResponseSelectionShot] = [
    follow_up_generation_example_1_shot,
    follow_up_generation_example_2_shot,
    follow_up_generation_example_3_shot,
    follow_up_generation_example_4_shot,
]
