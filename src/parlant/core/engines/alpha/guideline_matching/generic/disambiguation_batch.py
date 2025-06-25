from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from typing import Optional
from parlant.core.common import DefaultBaseModel, JSONSerializable
from parlant.core.engines.alpha.guideline_matching.generic.common import internal_representation
from parlant.core.engines.alpha.guideline_matching.guideline_match import (
    GuidelineMatch,
    PreviouslyAppliedType,
)
from parlant.core.engines.alpha.guideline_matching.guideline_matcher import (
    GuidelineMatchingBatch,
    GuidelineMatchingBatchResult,
    GuidelineMatchingContext,
)
from parlant.core.engines.alpha.prompt_builder import BuiltInSection, PromptBuilder, SectionStatus
from parlant.core.guidelines import Guideline, GuidelineContent
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.sessions import Event, EventId, EventKind, EventSource
from parlant.core.shots import Shot, ShotCollection


class GuidelineCheck(DefaultBaseModel):
    guideline_id: str
    short_evaluation: str
    requires_disambiguation: bool


class DisambiguationGuidelineMatchesSchema(DefaultBaseModel):
    rationale: str
    is_ambiguous: bool
    guidelines: Optional[list[GuidelineCheck]] = []
    clarification_action: Optional[str] = ""


@dataclass
class DisambiguationGuidelineMatchingShot(Shot):
    interaction_events: Sequence[Event]
    guidelines: Sequence[GuidelineContent]
    guideline_head: GuidelineContent
    expected_result: DisambiguationGuidelineMatchesSchema


# TODO - when adding the new clarification guideline, add it with customer dependent flag


class GenericDisambiguationGuidelineMatchingBatch(GuidelineMatchingBatch):
    def __init__(
        self,
        logger: Logger,
        schematic_generator: SchematicGenerator[DisambiguationGuidelineMatchesSchema],
        disambiguation_guideline: Guideline,
        disambiguation_targets: Sequence[Guideline],
        context: GuidelineMatchingContext,
    ) -> None:
        self._logger = logger
        self._schematic_generator = schematic_generator
        self._guideline_head = disambiguation_guideline
        self._disambiguation_targets = {g.id: g for g in disambiguation_targets}

        self._guideline_ids = {
            str(i): id for i, id in enumerate(self._disambiguation_targets.keys(), start=1)
        }
        self._context = context

    async def process(self) -> GuidelineMatchingBatchResult:
        prompt = self._build_prompt(shots=await self.shots())

        with self._logger.operation("DisambiguationGuidelineMatchingBatch:"):
            inference = await self._schematic_generator.generate(
                prompt=prompt,
                hints={"temperature": 0.15},
            )
            self._logger.debug(f"Completion:\n{inference.content.model_dump_json(indent=2)}")

        metadata: dict[str, JSONSerializable] = {}

        if inference.content.is_ambiguous:
            guidelines: list[str] = [
                self._guideline_ids[g.guideline_id]
                for g in inference.content.guidelines or []
                if g.requires_disambiguation
            ]

            disambiguation_data: JSONSerializable = {
                "targets": guidelines,
                "enriched_action": inference.content.clarification_action or "",
            }

            metadata["disambiguation"] = disambiguation_data

        matches = [
            GuidelineMatch(
                guideline=self._guideline_head,
                score=10 if inference.content.is_ambiguous else 1,
                rationale=f'''Not previously applied matcher rationale: "{inference.content.rationale}"''',
                guideline_previously_applied=PreviouslyAppliedType.NO,
                metadata=metadata,
            )
        ]

        return GuidelineMatchingBatchResult(
            matches=matches,
            generation_info=inference.info,
        )

    async def shots(self) -> Sequence[DisambiguationGuidelineMatchingShot]:
        return await shot_collection.list()

    def _format_shots(self, shots: Sequence[DisambiguationGuidelineMatchingShot]) -> str:
        return "\n".join(
            f"Example #{i}: ###\n{self._format_shot(shot)}" for i, shot in enumerate(shots, start=1)
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
        if shot.guideline_head:
            guideline_head = (
                f"Condition {shot.guideline_head.condition}. Action: {shot.guideline_head.action}"
            )
            formatted_shot += f"""
- **Guideline Head**:
{guideline_head}

"""
        if shot.guidelines:
            formatted_guidelines = "\n".join(
                f"{i}) Condition {g.condition}. Action: {g.action}"
                for i, g in enumerate(shot.guidelines, start=1)
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
        shots: Sequence[DisambiguationGuidelineMatchingShot],
    ) -> PromptBuilder:
        guideline_head_representation = internal_representation(self._guideline_head)
        guideline_targets_representations = {
            g.id: internal_representation(g) for g in self._disambiguation_targets.values()
        }

        guidelines_text = "\n".join(
            f"{i}) Condition: {guideline_targets_representations[id].condition}. Action: {guideline_targets_representations[id].action}"
            for i, id in self._guideline_ids.items()
        )

        guideline_head_text = f"Condition {guideline_head_representation.condition}. Action: {guideline_head_representation.action}"
        builder = PromptBuilder(on_build=lambda prompt: self._logger.debug(f"Prompt:\n{prompt}"))

        builder.add_section(
            name="guideline-disambiguation-evaluator-general-instructions",
            template="""
GENERAL INSTRUCTIONS
-----------------
In our system, the behavior of a conversational AI agent is guided by "guidelines". The agent makes use of these guidelines whenever it interacts with a user (also referred to as the customer).
Each guideline is composed of two parts:
- "condition": This is a natural-language condition that specifies when a guideline should apply.
          We look at each conversation at any particular state, and we test against this
          condition to understand if we should have this guideline participate in generating
          the next reply to the user.
- "action": This is a natural-language instruction that should be followed by the agent
          whenever the "condition" part of the guideline applies to the conversation in its particular state.
          Any instruction described here applies only to the agent, and not to the user.


Task Description
----------------
Sometimes a customer expresses that they’ve experienced something or want to proceed with something, but there are multiple possible ways to go, and it’s unclear what exactly they intend. 
In such cases, we need to identify the potential options and ask the customer which one they mean.

Your task is to determine whether the customer’s request is ambiguous and, if so, what the possible interpretations or directions are. 
You’ll be given a guideline head — a condition that, if true, signals a potential ambiguity — and a list of related guidelines, each representing a possible path the customer might want to follow.

If you identify an ambiguity, return the relevant guidelines that represent the available options. 
Then, formulate a response in the format:
"Ask the customer whether they want to do X, Y, or Z..."
This response should clearly present the options to help resolve the ambiguity in the customer's request.

Notes:
- If a guideline might be relevant - include it. We prefer to let the customer choose of all plausible options.
- Base your evaluation on the customer's most recent request.
- Some guidelines may turn out to be irrelevant based on the interaction—for example, due to earlier parts of the conversation or because the user's status (provided in the interaction history or 
as a context variable) rules them out. In such cases, the ambiguity may already be resolved (only one or none option is relevant) and no clarification is needed.
- If during the interaction, the agent asked for clarification but the customer hasn't asked yet, do not consider it as there is a disambiguation, unless a new disambiguation arises.




""",
            props={},
        )
        builder.add_section(
            name="guideline-ambiguity-evaluations-examples",
            template="""
Examples of Guidelines Ambiguity Evaluations:
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
        builder.add_interaction_history(self._context.interaction_history)
        builder.add_staged_events(self._context.staged_events)
        builder.add_section(
            name=BuiltInSection.GUIDELINES,
            template="""
- Guidelines Head: ###
{guideline_head_text}
###
- Guidelines List: ###
{guidelines_text}
###
""",
            props={"guidelines_text": guidelines_text, "guideline_head_text": guideline_head_text},
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
                "result_structure_text": self._format_of_guideline_check_json_description(),
            },
        )

        return builder

    def _format_of_guideline_check_json_description(self) -> str:
        result = {
            "rationale": "<str, Explanation for why there is or isn't an ambiguity>",
            "is_ambiguous": "<BOOL>",
            "guidelines (include only if is_ambiguous is true)": [
                {
                    "guideline_id": i,
                    "short_evaluation": "<str. Brief explanation of is this guideline needs disambiguation>",
                    "requires_disambiguation": "<BOOL>",
                }
                for i in self._guideline_ids.keys()
            ],
            "clarification_action": "<include only if is_ambiguous is true. An action of the form ask the user whether they want to...>",
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

example_1_guidelines = [
    GuidelineContent(
        condition="The customer asks to return an item for a refund",
        action="refund the order",
    ),
    GuidelineContent(
        condition="The customer asks to replace an item",
        action="Send the correct item and ask the customer to return the one they received",
    ),
]

example_1_guideline_head = GuidelineContent(
    condition="The customer received a wrong or damaged item",
    action="-",
)

example_1_expected = DisambiguationGuidelineMatchesSchema(
    rationale="The customer received the wrong item and may want to either replace it or get a refund.",
    is_ambiguous=True,
    guidelines=[
        GuidelineCheck(
            guideline_id="1",
            short_evaluation="may want to refund the wrong item",
            requires_disambiguation=True,
        ),
        GuidelineCheck(
            guideline_id="2",
            short_evaluation="may want to replace the wrong item",
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

example_2_guidelines = [
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

example_2_guideline_head = GuidelineContent(
    condition="The customer wants to book an appointment, but it’s unclear whether it’s with a doctor or a psychologist, and whether it should be online or in-person.",
    action="-",
)

example_2_expected = DisambiguationGuidelineMatchesSchema(
    rationale="The customer asks to book an appointment but didn't specify the type. Since they mention needing a prescription, it likely relates to a medical consultation, not psychological.",
    is_ambiguous=True,
    guidelines=[
        GuidelineCheck(
            guideline_id="1",
            short_evaluation="the appointment is with a doctor",
            requires_disambiguation=True,
        ),
        GuidelineCheck(
            guideline_id="2",
            short_evaluation="psychologist is not relevant",
            requires_disambiguation=False,
        ),
        GuidelineCheck(
            guideline_id="3",
            short_evaluation="online appointment can be relevant",
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

example_3_guidelines = [
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

example_3_guideline_head = GuidelineContent(
    condition="The customer wants to book an appointment, but it’s unclear whether it’s with a doctor or a psychologist, and whether it should be online or in-person.",
    action="-",
)

example_3_expected = DisambiguationGuidelineMatchesSchema(
    rationale="The customer requests an online appointment and mentions needing a prescription, which suggests a medical consultation",
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

example_4_guidelines = [
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

example_4_guideline_head = GuidelineContent(
    condition="The customer wants to book an appointment, but it’s unclear whether it’s with a doctor or a psychologist, and whether it should be online or in-person.",
    action="-",
)

example_4_expected = DisambiguationGuidelineMatchesSchema(
    rationale="The customer asks to book an appointment. Online sessions are not available. Since they mention hurting throat, it likely relates to a medical consultation, not a psychologist.",
    is_ambiguous=False,
)

example_5_events = [
    _make_event(
        "11",
        EventSource.CUSTOMER,
        "I received the wrong item in my order",
    ),
    _make_event(
        "14",
        EventSource.AI_AGENT,
        "I'm sorry to hear that. We can either offer you a return with a refund or send you the correct item instead. What would you prefer?",
    ),
    _make_event(
        "22",
        EventSource.CUSTOMER,
        "I'm not sure yet. Let me think for a moment",
    ),
]

example_5_guidelines = [
    GuidelineContent(
        condition="The customer asks to return an item for a refund",
        action="refund the order",
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

example_5_guideline_head = GuidelineContent(
    condition="The customer received a wrong or damaged item",
    action="-",
)

example_5_expected = DisambiguationGuidelineMatchesSchema(
    rationale="There is a new disambiguated request. Need to clarify whether it's with a doctor or a psychologist, and whether it should be online or in person",
    is_ambiguous=True,
    guidelines=[
        GuidelineCheck(
            guideline_id="1",
            short_evaluation="the appointment may be with a doctor",
            requires_disambiguation=True,
        ),
        GuidelineCheck(
            guideline_id="2",
            short_evaluation="psychologist may be relevant",
            requires_disambiguation=True,
        ),
        GuidelineCheck(
            guideline_id="3",
            short_evaluation="online appointment can be relevant",
            requires_disambiguation=True,
        ),
    ],
    clarification_action="Ask the customer if they need a doctor or psychologist appointment and if they prefer an online or in person session",
)


example_6_events = [
    _make_event(
        "11",
        EventSource.CUSTOMER,
        "Hey, can you book me an appointment? I need a prescription. And also I need to a session with a psychologist with my wife in your office.",
    ),
]

example_6_guidelines = [
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

example_6_guideline_head = GuidelineContent(
    condition="The customer wants to book an appointment, but it’s unclear whether it should be online or in-person. They say prescription so they need a doctor.",
    action="-",
)

example_6_expected = DisambiguationGuidelineMatchesSchema(
    rationale="For the first appointment there is an ambiguity. Need to clarify a doctor or a psychologist, and should it be online or in person",
    is_ambiguous=True,
    guidelines=[
        GuidelineCheck(
            guideline_id="1",
            short_evaluation="need a doctor",
            requires_disambiguation=False,
        ),
        GuidelineCheck(
            guideline_id="2",
            short_evaluation="psychologist can't be relevant",
            requires_disambiguation=False,
        ),
        GuidelineCheck(
            guideline_id="3",
            short_evaluation="online appointment can be relevant",
            requires_disambiguation=True,
        ),
        GuidelineCheck(
            guideline_id="4",
            short_evaluation="in person appointment can be relevant",
            requires_disambiguation=True,
        ),
    ],
    clarification_action="Ask the customer if they prefer an online or in person session",
)

_baseline_shots: Sequence[DisambiguationGuidelineMatchingShot] = [
    DisambiguationGuidelineMatchingShot(
        description="Disambiguation example",
        interaction_events=example_1_events,
        guidelines=example_1_guidelines,
        guideline_head=example_1_guideline_head,
        expected_result=example_1_expected,
    ),
    DisambiguationGuidelineMatchingShot(
        description="Disambiguation example when not all guidelines are relevant",
        interaction_events=example_2_events,
        guidelines=example_2_guidelines,
        guideline_head=example_2_guideline_head,
        expected_result=example_2_expected,
    ),
    DisambiguationGuidelineMatchingShot(
        description="Non disambiguation example",
        interaction_events=example_3_events,
        guidelines=example_3_guidelines,
        guideline_head=example_3_guideline_head,
        expected_result=example_3_expected,
    ),
    DisambiguationGuidelineMatchingShot(
        description="Disambiguation resolves based on the interaction",
        interaction_events=example_4_events,
        guidelines=example_4_guidelines,
        guideline_head=example_4_guideline_head,
        expected_result=example_4_expected,
    ),
    DisambiguationGuidelineMatchingShot(
        description="New ambiguous request",
        interaction_events=example_5_events,
        guidelines=example_5_guidelines,
        guideline_head=example_5_guideline_head,
        expected_result=example_5_expected,
    ),
    DisambiguationGuidelineMatchingShot(
        description="Several requests, one needs disambiguation",
        interaction_events=example_6_events,
        guidelines=example_6_guidelines,
        guideline_head=example_6_guideline_head,
        expected_result=example_6_expected,
    ),
]

shot_collection = ShotCollection[DisambiguationGuidelineMatchingShot](_baseline_shots)
