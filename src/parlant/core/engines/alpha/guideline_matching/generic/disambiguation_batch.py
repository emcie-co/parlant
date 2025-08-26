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

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import traceback
import json
from typing import Optional
from parlant.core.common import DefaultBaseModel, JSONSerializable
from parlant.core.engines.alpha.guideline_matching.generic.common import internal_representation
from parlant.core.engines.alpha.guideline_matching.guideline_match import (
    GuidelineMatch,
)
from parlant.core.engines.alpha.guideline_matching.guideline_matcher import (
    GuidelineMatchingBatch,
    GuidelineMatchingBatchResult,
    GuidelineMatchingContext,
    GuidelineMatchingBatchError,
)
from parlant.core.engines.alpha.optimization_policy import OptimizationPolicy
from parlant.core.engines.alpha.prompt_builder import BuiltInSection, PromptBuilder, SectionStatus
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.journeys import JourneyId, JourneyStore
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.sessions import Event, EventId, EventKind, EventSource
from parlant.core.shots import Shot, ShotCollection
from parlant.core.tags import Tag


class GuidelineCheck(DefaultBaseModel):
    guideline_id: str
    tldr: str
    requires_disambiguation: bool


class DisambiguationGuidelineMatchesSchema(DefaultBaseModel):
    tldr: str
    disambiguation_requested: bool
    customer_resolved: Optional[bool] = False
    is_ambiguous: bool
    guidelines: Optional[list[GuidelineCheck]] = []
    clarification_action: Optional[str] = ""


@dataclass
class DisambiguationGuidelineMatchingShot(Shot):
    interaction_events: Sequence[Event]
    disambiguation_condition: GuidelineContent
    disambiguation_targets: Sequence[GuidelineContent]
    expected_result: DisambiguationGuidelineMatchesSchema


@dataclass
class _Guideline:
    conditions: list[str]
    action: str | None
    ids: list[GuidelineId]


class GenericDisambiguationGuidelineMatchingBatch(GuidelineMatchingBatch):
    def __init__(
        self,
        logger: Logger,
        journey_store: JourneyStore,
        optimization_policy: OptimizationPolicy,
        schematic_generator: SchematicGenerator[DisambiguationGuidelineMatchesSchema],
        disambiguation_guideline: Guideline,
        disambiguation_targets: Sequence[Guideline],
        context: GuidelineMatchingContext,
    ) -> None:
        self._logger = logger
        self._journey_store = journey_store
        self._optimization_policy = optimization_policy
        self._schematic_generator = schematic_generator
        self._disambiguation_guideline = disambiguation_guideline
        self._disambiguation_targets = disambiguation_targets
        self._context = context

    async def _get_disambiguation_targets(
        self,
        disambiguation_targets: Sequence[Guideline],
    ) -> dict[str, _Guideline]:
        journey_to_conditions = defaultdict(list)
        guidelines_targets = []
        for g in disambiguation_targets:
            for t in g.tags:
                if journey_id := Tag.extract_journey_id(t):
                    journey_to_conditions[journey_id].append(g)
                else:
                    guidelines_targets.append(g)
                    continue
            if not g.tags:
                guidelines_targets.append(g)

        guidelines = {}
        i = 1
        for journey_id, conditions in journey_to_conditions.items():
            journey = await self._journey_store.read_journey(JourneyId(journey_id))
            guidelines[str(i)] = _Guideline(
                conditions=[g.content.condition for g in conditions],
                action=journey.title,
                ids=[g.id for g in conditions],
            )
            i += 1
        for g in guidelines_targets:
            guidelines[str(i)] = _Guideline(
                conditions=[internal_representation(g).condition],
                action=internal_representation(g).action,
                ids=[g.id],
            )
            i += 1
        return guidelines

    async def process(self) -> GuidelineMatchingBatchResult:
        disambiguation_targets_guidelines = await self._get_disambiguation_targets(
            self._disambiguation_targets
        )

        with self._logger.operation(str(self._disambiguation_guideline)):
            prompt = self._build_prompt(
                shots=await self.shots(),
                disambiguation_targets_guidelines=disambiguation_targets_guidelines,
            )

            generation_attempt_temperatures = (
                self._optimization_policy.get_guideline_matching_batch_retry_temperatures(
                    hints={"type": self.__class__.__name__}
                )
            )

            last_generation_exception: Exception | None = None

            for generation_attempt in range(3):
                try:
                    inference = await self._schematic_generator.generate(
                        prompt=prompt,
                        hints={"temperature": generation_attempt_temperatures[generation_attempt]},
                    )
                    self._logger.trace(
                        f"Completion:\n{inference.content.model_dump_json(indent=2)}"
                    )

                    metadata: dict[str, JSONSerializable] = {}

                    if inference.content.is_ambiguous:
                        guidelines: list[str] = []
                        for g in inference.content.guidelines or []:
                            if g.requires_disambiguation:
                                guidelines.extend(
                                    disambiguation_targets_guidelines[g.guideline_id].ids
                                )

                        disambiguation_data: JSONSerializable = {
                            "targets": guidelines,
                            "enriched_action": inference.content.clarification_action or "",
                        }

                        metadata["disambiguation"] = disambiguation_data

                        self._logger.debug(
                            f"Disambiguation activated: {inference.content.model_dump_json(indent=2)}"
                        )

                    matches = [
                        GuidelineMatch(
                            guideline=self._disambiguation_guideline,
                            score=10 if inference.content.is_ambiguous else 1,
                            rationale=f'''Disambiguation rationale: "{inference.content.tldr}"''',
                            metadata=metadata,
                        )
                    ]

                    return GuidelineMatchingBatchResult(
                        matches=matches,
                        generation_info=inference.info,
                    )

                except Exception as exc:
                    self._logger.warning(
                        f"Attempt {generation_attempt} failed: {traceback.format_exception(exc)}"
                    )

                    last_generation_exception = exc

            raise GuidelineMatchingBatchError() from last_generation_exception

    async def shots(self) -> Sequence[DisambiguationGuidelineMatchingShot]:
        return await shot_collection.list()

    def _format_shots(
        self,
        shots: Sequence[DisambiguationGuidelineMatchingShot],
    ) -> str:
        return "\n".join(
            f"""
Example {i} - {shot.description}: ###
{self._format_shot(shot)}
###
"""
            for i, shot in enumerate(shots, start=1)
        )

    def _format_shot(self, shot: DisambiguationGuidelineMatchingShot) -> str:
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
        if shot.disambiguation_condition:
            formatted_shot += f"""
- **Disambiguation Condition:**
{shot.disambiguation_condition.condition}

"""
        if shot.disambiguation_targets:
            formatted_guidelines = "\n".join(
                f"{i}) Condition: {g.condition}. Action: {g.action}"
                for i, g in enumerate(shot.disambiguation_targets, start=1)
            )
            formatted_shot += f"""
- **Guidelines**:
{formatted_guidelines}

"""

        formatted_shot += f"""
- **Expected Result**:
```json
{json.dumps(shot.expected_result.model_dump(mode="json", exclude_unset=True), indent=2)}
```
"""

        return formatted_shot

    def _build_prompt(
        self,
        disambiguation_targets_guidelines: dict[str, _Guideline],
        shots: Sequence[DisambiguationGuidelineMatchingShot],
    ) -> PromptBuilder:
        disambiguation_condition_internal = internal_representation(self._disambiguation_guideline)

        disambiguation_targets_text = "\n".join(
            f"{id}) Condition: {', '.join(g.conditions) if len(g.conditions) > 1 else g.conditions[0]}. "
            f"Action: {g.action}"
            for id, g in disambiguation_targets_guidelines.items()
        )
        builder = PromptBuilder(on_build=lambda prompt: self._logger.trace(f"Prompt:\n{prompt}"))

        builder.add_section(
            name="guideline-disambiguation-evaluator-general-instructions",
            template="""
GENERAL INSTRUCTIONS
-----------------
In our system, the behavior of a conversational AI agent is guided by "guidelines". The agent makes use of these guidelines whenever it interacts with a user (also referred to as the customer).
Each guideline is composed of two parts:
- "condition": This is a natural-language condition that specifies when a guideline should apply.
          We look at each conversation at its most recent state, and we test against this
          condition to understand if we should have this guideline participate in generating
          the next response to the user.
- "action": This is a natural-language instruction that should be followed by the agent
          whenever the "condition" part of the guideline applies to the conversation at its latest state.
          Any instruction described here applies only to the agent, and not to the user.


Task Description
----------------
Sometimes a customer expresses that they’ve experienced something or want to proceed with something, but there are multiple possible ways to go, and it’s as-yet unclear what exactly they intend.
In such cases, we need to identify the potential options and ask the customer which one they mean.

Your task is to determine whether the customer’s intention is currently ambiguous and, if so, what the possible interpretations or directions are.
You’ll be given a disambiguation condition — one that, if true, signals a potential ambiguity — and a list of related guidelines, each representing a possible path the customer might want to follow.

If you identify an ambiguity, return the relevant guidelines that represent the available options.
Then, formulate a response in the format:
"Ask the customer whether they want to do X, Y, or Z..."
This response should clearly present the options to help resolve the ambiguity in the customer's request.

Notes:
- Base your evaluation on the customer's most recent message.
- If you determine that there is indeed an ambiguity - then, when one of the guidelines might be relevant - include it. We prefer to let the customer choose of all plausible options.
- Some guidelines may turn out to be irrelevant based on the interaction. For example, due to earlier parts of the conversation or because the user's status (provided in the interaction history or
as a context variable) rules them out. In such cases, the ambiguity may already be resolved (only one or none option is relevant) and note that no clarification is needed in such cases.

- Notice that if you've already asked for disambiguation from the customer you need to **pay extra attention** to if we need to re-ask for clarification or the user responded and it was resolved.
- **Accept brief customer responses as valid clarifications**: Customers often communicate with very short responses (single words or phrases like "return", "replace", "yes", "no"). If the customer's brief
 response clearly indicates their choice among the previously presented options, consider the ambiguity resolved even if their answer is not in complete sentences.

distinction is between:
- Pending clarification (customer hasn't answered yet) - re-disambiguate (disambiguation_requested = true, customer_resolved=false, is_ambiguous = true)
- Clarification provided (customer has answered) - don't re-disambiguate the same issue (disambiguation_requested = true, customer_resolved=true, is_ambiguous = false)
- New ambiguity (different unclear intent emerges) - do disambiguate (is_ambiguous = true)

- Focus on current context: If the customer has changed the subject or moved on to a different topic in their most recent message, do not disambiguate previous unresolved issues.
Always prioritize the customer's current request and intent over past ambiguities.
""",
            props={},
        )
        builder.add_section(
            name="guideline-ambiguity-evaluations-examples",
            template="""
Examples of Guidelines Ambiguity Evaluation:
-------------------
{formatted_shots}
""",
            props={
                "formatted_shots": self._format_shots(shots),
                "shots": shots,
            },
        )
        builder.add_agent_identity(self._context.agent)
        builder.add_context_variables(self._context.context_variables)
        builder.add_glossary(self._context.terms)
        builder.add_capabilities_for_guideline_matching(self._context.capabilities)
        builder.add_interaction_history(self._context.interaction_history)
        builder.add_staged_tool_events(self._context.staged_events)
        builder.add_section(
            name=BuiltInSection.GUIDELINES,
            template="""
- Disambiguation Condition: ###
{disambiguation_condition}
###
- Guidelines List: ###
{disambiguation_targets_text}
###
""",
            props={
                "disambiguation_targets_text": disambiguation_targets_text,
                "disambiguation_condition": disambiguation_condition_internal.condition,
            },
            status=SectionStatus.ACTIVE,
        )
        builder.add_section(
            name="guideline-disambiguation-evaluation-output-format",
            template="""

OUTPUT FORMAT
-----------------
- Specify the evaluation of disambiguation by filling in the details in the following list as instructed:
```json
{{
    {result_structure_text}
}}
```
""",
            props={
                "result_structure_text": self._format_of_guideline_check_json_description(
                    disambiguation_targets_guidelines
                ),
            },
        )

        return builder

    def _format_of_guideline_check_json_description(
        self, disambiguation_targets_guidelines: dict[str, _Guideline]
    ) -> str:
        result = {
            "tldr": "<str, Briefly state the customer's most recent intent based on their LATEST input, and explain why there is or isn't an ambiguity.>",
            "disambiguation_requested": "<BOOL. Based on the interaction, whether a clarification was asked by the agent. If so, is_ambiguous will be true only if customer has not answered OR customer changed request OR there is a new ambiguity to resolve>",
            "customer_resolved": "<BOOL. include if disambiguation_requested=true, whether the latest ambiguity that was requested was resolved by the user",
            "is_ambiguous": "<BOOL>",
            "guidelines (include only if is_ambiguous is True)": [
                {
                    "guideline_id": i,
                    "tldr": "<str. Brief explanation of is this guideline needs disambiguation>",
                    "requires_disambiguation": "<BOOL>",
                }
                for i in disambiguation_targets_guidelines.keys()
            ],
            "clarification_action": "<include only if is_ambiguous is True. An action of the form ask the user whether they want to...>",
        }
        return json.dumps(result, indent=4)


def _make_event(e_id: str, source: EventSource, message: str) -> Event:
    return Event(
        id=EventId(e_id),
        source=source,
        kind=EventKind.MESSAGE,
        creation_utc=datetime.now(timezone.utc),
        offset=0,
        correlation_id="",
        data={"message": message},
        deleted=False,
    )


example_1_events = [
    _make_event(
        "11",
        EventSource.CUSTOMER,
        "I received the wrong item in my order.",
    ),
]

example_1_disambiguation_targets = [
    GuidelineContent(
        condition="The customer asks to return an item for a refund",
        action="refund the order",
    ),
    GuidelineContent(
        condition="The customer asks to replace an item",
        action="Send the correct item and ask the customer to return the one they received",
    ),
]

example_1_disambiguation_condition = GuidelineContent(
    condition="The customer received a wrong or damaged item",
    action="-",
)

example_1_expected = DisambiguationGuidelineMatchesSchema(
    tldr="The customer claimed to receive the wrong item; may want to either replace it or get a refund.",
    disambiguation_requested=False,
    is_ambiguous=True,
    guidelines=[
        GuidelineCheck(
            guideline_id="1",
            tldr="may want to refund the wrong item",
            requires_disambiguation=True,
        ),
        GuidelineCheck(
            guideline_id="2",
            tldr="may want to replace the wrong item",
            requires_disambiguation=True,
        ),
    ],
    clarification_action="ask the customer whether they’d prefer a replacement or a refund.",
)


example_2_events = [
    _make_event(
        "11",
        EventSource.CUSTOMER,
        "Hey, can you book me an appointment? I need a prescription",
    ),
]

example_2__disambiguation_targets = [
    GuidelineContent(
        condition="The customer asks to book an appointment with a doctor",
        action="book the appointment",
    ),
    GuidelineContent(
        condition="The customer asks to book a session with a psychologist",
        action="book the appointment",
    ),
    GuidelineContent(
        condition="The customer asks to book an online appointment to a medical consultation or a session with a psychologist",
        action="book the appointment online",
    ),
]

example_2_disambiguation_condition = GuidelineContent(
    condition="The customer wants to book an appointment, but it’s unclear whether it’s with a doctor or a psychologist, and whether it should be online or in-person.",
    action="-",
)

example_2_expected = DisambiguationGuidelineMatchesSchema(
    tldr="The customer asks to book an appointment but didn't specify the type. Since they mention needing a prescription, it likely relates to a medical consultation, not psychological.",
    disambiguation_requested=False,
    is_ambiguous=True,
    guidelines=[
        GuidelineCheck(
            guideline_id="1",
            tldr="the appointment is with a doctor",
            requires_disambiguation=True,
        ),
        GuidelineCheck(
            guideline_id="2",
            tldr="psychologist is not relevant",
            requires_disambiguation=False,
        ),
        GuidelineCheck(
            guideline_id="3",
            tldr="online appointment can be relevant",
            requires_disambiguation=True,
        ),
    ],
    clarification_action="Ask the customer if they prefer an online or in person doctor’s appointment",
)


example_3_events = [
    _make_event(
        "11",
        EventSource.CUSTOMER,
        "Hey, can you book me an online appointment? I need a prescription",
    ),
]

example_3__disambiguation_targets = [
    GuidelineContent(
        condition="The customer asks to book an appointment with a doctor",
        action="book the appointment",
    ),
    GuidelineContent(
        condition="The customer asks to book a session with a psychologist",
        action="book the appointment",
    ),
    GuidelineContent(
        condition="The customer asks to book an online appointment to a medical consultation or a session with a psychologist",
        action="book the appointment online",
    ),
]

example_3_disambiguation_condition = GuidelineContent(
    condition="The customer asked to book an appointment, but it’s unclear whether it’s with a doctor or a psychologist, and whether it should be online or in-person.",
    action="-",
)

example_3_expected = DisambiguationGuidelineMatchesSchema(
    tldr="The customer requests an online appointment and mentions needing a prescription, which suggests a medical consultation",
    disambiguation_requested=False,
    is_ambiguous=False,
)


example_4_events = [
    _make_event(
        "11",
        EventSource.CUSTOMER,
        "Hey, are you offering in-person sessions these days, or is everything online?",
    ),
    _make_event(
        "15",
        EventSource.AI_AGENT,
        "I'm sorry, but due to the current situation, we aren't holding in-person meetings. However, we do offer online sessions if needed",
    ),
    _make_event(
        "20",
        EventSource.CUSTOMER,
        "Got it. I’ll need an appointment — my throat is hurting.",
    ),
]

example_4__disambiguation_targets = [
    GuidelineContent(
        condition="The customer asks to book an appointment with a doctor",
        action="book the appointment",
    ),
    GuidelineContent(
        condition="The customer asks to book a session with a psychologist",
        action="book the appointment",
    ),
    GuidelineContent(
        condition="The customer asks to book an online appointment to a medical consultation or a session with a psychologist",
        action="book the appointment online",
    ),
]

example_4_disambiguation_condition = GuidelineContent(
    condition="The customer wants to book an appointment, but it’s unclear whether it’s with a doctor or a psychologist, and whether it should be online or in-person.",
    action="-",
)

example_4_expected = DisambiguationGuidelineMatchesSchema(
    tldr="The customer asks to book an appointment. Online sessions are not available. Since they mention hurting throat, it likely relates to a medical consultation, not a psychologist.",
    disambiguation_requested=False,
    is_ambiguous=False,
)


example_5_events = [
    _make_event(
        "11",
        EventSource.CUSTOMER,
        "Hey, can you book me an appointment? I need a prescription",
    ),
    _make_event(
        "14",
        EventSource.AI_AGENT,
        "You can have a doctor’s session either in person or online. Which do you prefer?",
    ),
    _make_event(
        "17",
        EventSource.CUSTOMER,
        "I can do it online. Also, I need to book an appointment for my daughter.",
    ),
]

example_5_disambiguation_targets = [
    GuidelineContent(
        condition="The customer asks to book an appointment with a doctor",
        action="book the appointment",
    ),
    GuidelineContent(
        condition="The customer asks to book a session with a psychologist",
        action="book the appointment",
    ),
    GuidelineContent(
        condition="The customer asks to book an online appointment to a medical consultation or a session with a psychologist",
        action="book the appointment online",
    ),
]

example_5_disambiguation_condition = GuidelineContent(
    condition="The customer wants to book an appointment, but it’s unclear whether it’s with a doctor or a psychologist, and whether it should be online or in-person.",
    action="-",
)

example_5_expected = DisambiguationGuidelineMatchesSchema(
    tldr="Based on latest message, there is a new request which is again ambiguous. Need to clarify whether it's with a doctor or a psychologist, and whether it should be online or in person",
    disambiguation_requested=False,
    is_ambiguous=True,
    guidelines=[
        GuidelineCheck(
            guideline_id="1",
            tldr="the appointment may be with a doctor",
            requires_disambiguation=True,
        ),
        GuidelineCheck(
            guideline_id="2",
            tldr="psychologist may be relevant",
            requires_disambiguation=True,
        ),
        GuidelineCheck(
            guideline_id="3",
            tldr="online appointment can be relevant",
            requires_disambiguation=True,
        ),
    ],
    clarification_action="Ask the customer if they need a doctor or psychologist appointment and if they prefer an online or in person session for their daughter",
)


example_6_events = [
    _make_event(
        "11",
        EventSource.CUSTOMER,
        "Hey, can you book me an appointment? I need a prescription. And also I need to a session with a psychologist with my wife in your office.",
    ),
]

example_6__disambiguation_targets = [
    GuidelineContent(
        condition="The customer asks to book an appointment with a doctor",
        action="book the appointment",
    ),
    GuidelineContent(
        condition="The customer asks to book a session with a psychologist",
        action="book the appointment",
    ),
    GuidelineContent(
        condition="The customer asks to book an online appointment to a medical consultation or a session with a psychologist",
        action="book the appointment online",
    ),
    GuidelineContent(
        condition="The customer asks to book an in person appointment to a medical consultation or a session with a psychologist",
        action="book the in person appointment",
    ),
]

example_6_disambiguation_condition = GuidelineContent(
    condition="The customer wants to book an appointment, but it’s unclear whether it should be online or in-person. They say prescription so they need a doctor.",
    action="-",
)

example_6_expected = DisambiguationGuidelineMatchesSchema(
    tldr="The customer asked to book two appointments. For the first appointment there is an ambiguity between doctor or psychologist, and online or in-person. The second one is clear.",
    disambiguation_requested=False,
    is_ambiguous=True,
    guidelines=[
        GuidelineCheck(
            guideline_id="1",
            tldr="need a doctor",
            requires_disambiguation=False,
        ),
        GuidelineCheck(
            guideline_id="2",
            tldr="psychologist can't be relevant",
            requires_disambiguation=False,
        ),
        GuidelineCheck(
            guideline_id="3",
            tldr="online appointment can be relevant",
            requires_disambiguation=True,
        ),
        GuidelineCheck(
            guideline_id="4",
            tldr="in person appointment can be relevant",
            requires_disambiguation=True,
        ),
    ],
    clarification_action="Ask the customer if they prefer an online or in person session",
)


example_7_events = [
    _make_event(
        "1",
        EventSource.CUSTOMER,
        "I received the wrong item in my order. This isn't what I ordered at all.",
    ),
    _make_event(
        "2",
        EventSource.AI_AGENT,
        "I'm sorry to hear you received the wrong item. Would you prefer a replacement of the correct item or a refund?",
    ),
    _make_event(
        "3",
        EventSource.CUSTOMER,
        "replace",
    ),
]

example_7_disambiguation_targets = [
    GuidelineContent(
        condition="The customer asks to return an item for a refund",
        action="refund the order",
    ),
    GuidelineContent(
        condition="The customer asks to replace an item",
        action="Send the correct item and ask the customer to return the one they received",
    ),
]

example_7_disambiguation_condition = GuidelineContent(
    condition="The customer received a wrong or damaged item",
    action="-",
)

example_7_expected = DisambiguationGuidelineMatchesSchema(
    tldr="The customer received a wrong item and was asked whether they wanted a replacement or refund. They responded with 'replace', which clearly indicates their choice and resolves the ambiguity.",
    disambiguation_requested=True,
    customer_resolved=True,
    is_ambiguous=False,
)

example_8_events = [
    _make_event(
        "1",
        EventSource.CUSTOMER,
        "I received the wrong item in my order. This isn't what I ordered at all.",
    ),
    _make_event(
        "2",
        EventSource.AI_AGENT,
        "I'm sorry to hear you received the wrong item. Would you prefer a replacement of the correct item or a refund?",
    ),
    _make_event(
        "3",
        EventSource.CUSTOMER,
        "I need to think.",
    ),
]

example_8_disambiguation_targets = [
    GuidelineContent(
        condition="The customer asks to return an item for a refund",
        action="refund the order",
    ),
    GuidelineContent(
        condition="The customer asks to replace an item",
        action="Send the correct item and ask the customer to return the one they received",
    ),
]

example_8_disambiguation_condition = GuidelineContent(
    condition="The customer received a wrong or damaged item",
    action="-",
)

example_8_expected = DisambiguationGuidelineMatchesSchema(
    tldr="The customer received a wrong item and clarification was asked. The customer only said that they need to think so ambiguity still apply",
    disambiguation_requested=True,
    customer_resolved=False,
    is_ambiguous=True,
    guidelines=[
        GuidelineCheck(
            guideline_id="1",
            tldr="may want to refund the wrong item",
            requires_disambiguation=True,
        ),
        GuidelineCheck(
            guideline_id="2",
            tldr="may want to replace the wrong item",
            requires_disambiguation=True,
        ),
    ],
    clarification_action="ask the customer whether they’d prefer a replacement or a refund.",
)

_baseline_shots: Sequence[DisambiguationGuidelineMatchingShot] = [
    DisambiguationGuidelineMatchingShot(
        description="Disambiguation example",
        interaction_events=example_1_events,
        disambiguation_targets=example_1_disambiguation_targets,
        disambiguation_condition=example_1_disambiguation_condition,
        expected_result=example_1_expected,
    ),
    DisambiguationGuidelineMatchingShot(
        description="Disambiguation example when not all guidelines are relevant",
        interaction_events=example_2_events,
        disambiguation_targets=example_2__disambiguation_targets,
        disambiguation_condition=example_2_disambiguation_condition,
        expected_result=example_2_expected,
    ),
    DisambiguationGuidelineMatchingShot(
        description="Non disambiguation example",
        interaction_events=example_3_events,
        disambiguation_targets=example_3__disambiguation_targets,
        disambiguation_condition=example_3_disambiguation_condition,
        expected_result=example_3_expected,
    ),
    DisambiguationGuidelineMatchingShot(
        description="Disambiguation resolves based on the interaction",
        interaction_events=example_4_events,
        disambiguation_targets=example_4__disambiguation_targets,
        disambiguation_condition=example_4_disambiguation_condition,
        expected_result=example_4_expected,
    ),
    DisambiguationGuidelineMatchingShot(
        description="New ambiguous request",
        interaction_events=example_5_events,
        disambiguation_targets=example_5_disambiguation_targets,
        disambiguation_condition=example_5_disambiguation_condition,
        expected_result=example_5_expected,
    ),
    DisambiguationGuidelineMatchingShot(
        description="Several requests, one needs disambiguation",
        interaction_events=example_6_events,
        disambiguation_targets=example_6__disambiguation_targets,
        disambiguation_condition=example_6_disambiguation_condition,
        expected_result=example_6_expected,
    ),
    DisambiguationGuidelineMatchingShot(
        description="Disambiguation applied and clarified",
        interaction_events=example_7_events,
        disambiguation_targets=example_7_disambiguation_targets,
        disambiguation_condition=example_7_disambiguation_condition,
        expected_result=example_7_expected,
    ),
    DisambiguationGuidelineMatchingShot(
        description="Disambiguation applied and customer did not respond",
        interaction_events=example_8_events,
        disambiguation_targets=example_8_disambiguation_targets,
        disambiguation_condition=example_8_disambiguation_condition,
        expected_result=example_8_expected,
    ),
]

shot_collection = ShotCollection[DisambiguationGuidelineMatchingShot](_baseline_shots)
