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

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from parlant.core.agents import Effort
from parlant.core.common import DefaultBaseModel, JSONSerializable
from parlant.core.engines.alpha.prompt_builder import BuiltInSection, PromptBuilder, SectionStatus
from parlant.core.engines.compass.response_state import EngineContext
from parlant.core.guidelines import Guideline, GuidelineContent
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.sessions import Event, EventId, EventKind, EventSource
from parlant.core.shots import Shot, ShotCollection
from parlant.core.tracer import Tracer


@dataclass(frozen=True)
class RankedGuideline:
    guideline: Guideline
    reasoning: str
    is_relevant: bool
    score: float


@dataclass(frozen=True)
class GuidelineRankingResult:
    ranked_guidelines: Sequence[RankedGuideline]
    # Aggregated usage across every per-guideline ranking request sent this call,
    # or None when no requests were sent.
    generation_info: GenerationInfo | None


class GuidelineRankSchema(DefaultBaseModel):
    tldr: str | None = None
    score: int


@dataclass
class GuidelineRankingShot(Shot):
    interaction_events: Sequence[Event]
    # The ranker evaluates a single guideline per prompt, so each shot carries one.
    guideline: GuidelineContent
    expected_result: GuidelineRankSchema


class GuidelineRanker:
    """First-pass guideline filter.

    Before the full guideline-matching stages, a smaller language model is used to
    cheaply filter out guidelines that do not apply for sure. Each guideline is
    evaluated in its own prompt; guidelines that pass this filter go on to the
    later stages of message generation.
    """

    RELEVANCE_SCORE_THRESHOLD = 3

    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        schematic_generator: SchematicGenerator[GuidelineRankSchema],
        include_tldr: bool = False,
    ) -> None:
        self._logger = logger
        self._tracer = tracer
        self._schematic_generator = schematic_generator
        self._include_tldr = include_tldr

    async def rank(
        self,
        context: EngineContext,
        guidelines: Sequence[Guideline],
    ) -> GuidelineRankingResult:
        if not guidelines:
            return GuidelineRankingResult([], None)

        with self._tracer.span("guideline.rank"):
            if len(guidelines) > 1:
                # Warm-then-fan-out: rank the first guideline and AWAIT it so the
                # provider's (implicit) cache is populated for the shared prompt
                # prefix, then fan out the rest concurrently — they read the warm
                # cache instead of all racing a cold one. The warm matches the
                # fan-out's prefix exactly (same context/interaction), so this is
                # reliable regardless of any earlier prefill. Costs one serialized
                # request of latency; only worth it when there's more than one.
                first = await self._rank_guideline(context, guidelines[0])
                rest = await asyncio.gather(
                    *(self._rank_guideline(context, guideline) for guideline in guidelines[1:])
                )
                results = [first, *rest]
            else:
                results = [await self._rank_guideline(context, guidelines[0])]

            return GuidelineRankingResult(
                ranked_guidelines=[ranked for ranked, _ in results],
                generation_info=_aggregate_generation_info([info for _, info in results]),
            )

    async def _rank_guideline(
        self,
        context: EngineContext,
        guideline: Guideline,
    ) -> tuple[RankedGuideline, GenerationInfo]:
        prompt = self._build_prompt(context, guideline, shots=await self.shots())

        inference = await self._schematic_generator.generate(
            prompt=prompt,
            hints={"reasoning_effort": self._get_reasoning_effort(context)},
        )

        score = inference.content.score

        # A score of 1-2 means "filtered out for sure"; 3+ ("maybe" and up) passes
        # the first-pass filter on to the next stages.
        return (
            RankedGuideline(
                guideline=guideline,
                reasoning=inference.content.tldr
                or f"This guideline ranked {score * 2} out of 10 in relevance to your next response.",
                is_relevant=score >= self.RELEVANCE_SCORE_THRESHOLD,
                score=float(score) / 5.0,
            ),
            inference.info,
        )

    def _get_reasoning_effort(self, context: EngineContext) -> str:
        match context.agent.effort:
            case Effort.MIN:
                return "minimal"
            case Effort.LOW:
                return "minimal"
            case Effort.MEDIUM:
                return "minimal"
            case Effort.HIGH:
                return "low"
            case Effort.MAX:
                return "medium"

    async def prefill(self, context: EngineContext) -> None:
        """Warm the generator's cache for the ranker's shared prompt prefix.

        `rank` fans out one request per guideline concurrently; an implicitly
        cached provider (e.g. Gemini) only populates its cache once a request
        completes, so a cold concurrent fan-out would have every request miss.
        Issuing one request over the shared prefix first (and discarding it)
        warms the cache so the real fan-out reads it. The same reasoning_effort
        is used so the warmed variant matches. Best-effort: warming failures must
        not break preparation."""
        with self._tracer.span("guideline.prefill"):
            try:
                prompt = self._build_shared_prompt(context, shots=await self.shots())
                inference = await self._schematic_generator.generate(
                    prompt=prompt,
                    hints={"reasoning_effort": self._get_reasoning_effort(context)},
                )
                if inference.info.usage.input_tokens > 0:
                    self._logger.info(
                        f"{self.__class__.__name__} prefill usage:\n {inference.info}"
                    )
            except Exception as exc:
                self._logger.warning(f"Guideline ranker prefill failed (continuing): {exc}")

    async def shots(self) -> Sequence[GuidelineRankingShot]:
        return await shot_collection.list()

    def _format_shots(self, shots: Sequence[GuidelineRankingShot]) -> str:
        return "\n".join(
            f"Example #{i}: ###\n{self._format_shot(shot)}" for i, shot in enumerate(shots, start=1)
        )

    def _format_shot(self, shot: GuidelineRankingShot) -> str:
        def adapt_event(e: Event) -> JSONSerializable:
            source_map: dict[EventSource, str] = {
                EventSource.CUSTOMER: "user",
                EventSource.CUSTOMER_UI: "frontend_application",
                EventSource.HUMAN_AGENT: "human_service_agent",
                EventSource.HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT: "ai_agent",
                EventSource.AI_AGENT: "ai_agent",
                EventSource.SYSTEM: "system-provided",
            }

            return {
                "event_kind": e.kind.value,
                "event_source": source_map[e.source],
                "data": e.data,
            }

        formatted_shot = ""
        if shot.interaction_events:
            formatted_shot += f"""
- **Interaction Events**:
{json.dumps([adapt_event(e) for e in shot.interaction_events], indent=2)}

"""

        formatted_shot += f"""
- **Guideline**:
{_format_guideline(shot.guideline.condition, shot.guideline.action)}

"""

        expected_result = shot.expected_result.model_dump(mode="json")
        if not self._include_tldr or expected_result.get("tldr") is None:
            expected_result.pop("tldr", None)

        formatted_shot += f"""
- **Expected Result**:
```json
{json.dumps(expected_result, indent=2)}
```
"""

        return formatted_shot

    def _build_prompt(
        self,
        context: EngineContext,
        guideline: Guideline,
        shots: Sequence[GuidelineRankingShot],
    ) -> PromptBuilder:
        # Start from the cross-turn-stable shared prefix, then append the
        # turn-varying context, the specific guideline, and finally the
        # output-format note. The guideline is what differs across the per-guideline
        # fan-out, so everything before it stays byte-identical within a turn.
        builder = self._build_shared_prompt(context, shots)

        # Turn-varying context: changes turn to turn, so it's deliberately NOT in
        # the shared prefix `prefill` warms (it would only bust that warm).
        builder.add_context_variables(context.state.context_variables)
        builder.add_glossary(list(context.state.glossary_terms))
        builder.add_capabilities_for_guideline_matching(context.state.capabilities)
        builder.add_interaction_history(context.interaction.events)
        builder.add_staged_tool_events(context.state.tool_events)

        builder.add_section(
            name=BuiltInSection.GUIDELINES,
            template="""
- Guideline: ###
{guideline_text}
###
""",
            props={
                "guideline_text": _format_guideline(
                    guideline.content.condition, guideline.content.action
                ),
            },
            status=SectionStatus.ACTIVE,
        )

        return builder

    def _build_shared_prompt(
        self,
        context: EngineContext,
        shots: Sequence[GuidelineRankingShot],
    ) -> PromptBuilder:
        """The cross-turn-stable head of the ranker prompt (instructions, shots,
        agent + customer identity). Excludes turn-varying context, so it stays
        byte-identical across turns — the prefix `prefill` warms."""
        builder = PromptBuilder()

        builder.add_section(
            name="guideline-ranker-general-instructions",
            template="""
GENERAL INSTRUCTIONS
-----------------
In our system, a conversational multi-turn AI agent's behavior is controlled by a set of guidelines.
Each guideline is comprised of a condition and potentially an action. Whenever a condition applies - the agent is directed to take its associated action.
""",
            props={},
        )
        builder.add_section(
            name="guideline-ranker-task-description",
            template="""
Task Description
----------------
Act as a first-pass filter to screen out clearly irrelevant guidelines based on their conditions. You are NOT making the final determination - a human judge will review every condition that isn't clearly irrelevant and decide whether it actually applies.

Output an integer score from 1 to 5 indicating how relevant the guideline's condition is to the most recent state of the conversation:
- 1 - not relevant at all, at no point in the conversation
- 2 - only very loosely related to the subject of the conversation
- 3 - partially relevant: was previously relevant but not clearly right now, or only part of the condition matches
- 4 - requires further checking by the human: a partial match, but one that clearly justifies further inspection
- 5 - clearly matches

Scores of 1 and 2 mean the guideline will be filtered out for sure; 3 means maybe; 4 and 5 mean it should continue on to the next stages. When in doubt, lean towards a higher score - the human will make the final call.

You will be given the context of the customer, the interaction so far (including tool calls and results), contextual information (variables) about the customer and the current interaction session, and other information. You shall use all of that information to evaluate the relevance of the guideline to the current state of the interaction, but you should focus more on the most recent state of the interaction.

Important considerations:
1. Focus on recency: Evaluate conditions based on the latest part of the conversation, particularly the most recent customer message.
2. Semantic evaluation: Assess the actual meaning of conditions, not just keyword matching.
3. Context matters: Consider the full context and intent behind the user's message, including tool results, context variables, capabilities etc'.
4. Ignore action: The action is only provided for you to contextualize the condition. Do not make your determination based on whether the action has occured. Only evaluate the condition.
5. Match based on entire context: a guideline may be matched based on the entire context of the interaction. It may be relevant even if it has nothing to do with the latest customer message.
""",
            props={},
        )
        builder.add_section(
            name="guideline-ranker-examples",
            template="""
Examples of Guideline Ranking Evaluations:
-------------------
{formatted_shots}
""",
            props={
                "formatted_shots": self._format_shots(shots),
                "shots": shots,
            },
        )

        builder.add_agent_identity(context.agent)
        builder.add_customer_identity(context.customer, context.session)

        # Without a tldr field to capture its reasoning, the model tends to append
        # an unsolicited explanation after the JSON; tell it not to so we don't pay
        # for output tokens we discard. Kept as the final note (recency).
        # (Providers that enforce structured output - OpenAI json_object, Gemini
        # function-calling - ignore this harmlessly.)
        json_only_note = (
            ""
            if self._include_tldr
            else "\nRespond with ONLY the JSON object above - no explanation, reasoning, or other text."
        )

        builder.add_section(
            name="guideline-ranker-output-format",
            template="""
OUTPUT FORMAT
-----------------
- Evaluate the guideline by filling in the details in the following structure:
```json
{result_structure_text}
```
{json_only_note}
""",
            props={
                "result_structure_text": self._format_output(),
                "json_only_note": json_only_note,
            },
        )

        return builder

    def _format_output(self) -> str:
        result: dict[str, JSONSerializable] = {}

        if self._include_tldr:
            result["tldr"] = (
                "<A brief, one-line summary of why the guideline's condition is or "
                "isn't relevant to the most recent state of the interaction>"
            )

        result["score"] = (
            "<An integer from 1 to 5 indicating how relevant the guideline is, per the scale described above>"
        )

        return json.dumps(result, indent=4)


def _aggregate_generation_info(infos: Sequence[GenerationInfo]) -> GenerationInfo:
    # All requests use the same schema/model; sum the token cost across them.
    # Duration is the max (the requests run concurrently, so it reflects wall-clock,
    # not total work). cached_input_tokens lives in `extra` and may be absent on some
    # infos, so default each to 0 before summing.
    return GenerationInfo(
        schema_name=infos[0].schema_name,
        model=infos[0].model,
        duration=max(info.duration for info in infos),
        usage=UsageInfo(
            input_tokens=sum(info.usage.input_tokens for info in infos),
            output_tokens=sum(info.usage.output_tokens for info in infos),
            extra={
                "cached_input_tokens": sum(
                    int((info.usage.extra or {}).get("cached_input_tokens", 0)) for info in infos
                )
            },
        ),
    )


def _format_guideline(condition: str, action: str | None) -> str:
    # The action is optional and only present to contextualize the condition; omit
    # it entirely when absent rather than rendering "Action: None".
    text = f"Condition: {condition}."
    if action:
        text += f" Action: {action}"
    return text


def _make_event(e_id: str, source: EventSource, message: str) -> Event:
    return Event(
        id=EventId(e_id),
        source=source,
        kind=EventKind.MESSAGE,
        creation_utc=datetime.now(timezone.utc),
        offset=0,
        trace_id="",
        data={"message": message},
        metadata={},
        deleted=False,
    )


example_1_events = [
    _make_event(
        "11",
        EventSource.CUSTOMER,
        "My throat is dry, do you have anything that can help me with that?",
    ),
]

example_1_guideline = GuidelineContent(
    condition="The customer might be thirsty",
    action=None,
)

example_1_expected = GuidelineRankSchema(
    tldr="Dry throat suggests potential thirst",
    score=4,
)


example_2_events = [
    _make_event(
        "21",
        EventSource.CUSTOMER,
        "Can delivery arrive here before 8 PM?",
    ),
]

example_2_guideline = GuidelineContent(
    condition="The customer asks for late night (10 PM or later) delivery",
    action=None,
)

example_2_expected = GuidelineRankSchema(
    tldr="8 PM is not late night (10 PM or later)",
    score=1,
)


example_3_events = [
    _make_event(
        "31",
        EventSource.CUSTOMER,
        "I'll consult with my partner about getting Platinum",
    ),
]

example_3_guideline = GuidelineContent(
    condition="Expressed unsureness about getting a product",
    action=None,
)

example_3_expected = GuidelineRankSchema(
    tldr="The customer needs consultation before getting Platinum, which might be a product, so it's best to forward this to the human evaluator",
    score=4,
)


_baseline_shots: Sequence[GuidelineRankingShot] = [
    GuidelineRankingShot(
        description="",
        interaction_events=example_1_events,
        guideline=example_1_guideline,
        expected_result=example_1_expected,
    ),
    GuidelineRankingShot(
        description="",
        interaction_events=example_2_events,
        guideline=example_2_guideline,
        expected_result=example_2_expected,
    ),
    GuidelineRankingShot(
        description="",
        interaction_events=example_3_events,
        guideline=example_3_guideline,
        expected_result=example_3_expected,
    ),
]

shot_collection = ShotCollection[GuidelineRankingShot](_baseline_shots)
