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
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence

from parlant.core.capabilities import Capability
from parlant.core.common import DefaultBaseModel, JSONSerializable
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.emissions import EmittedEvent
from parlant.core.engines.alpha.prompt_builder import BuiltInSection, PromptBuilder, SectionStatus
from parlant.core.engines.alpha.tool_calling.common import get_tool_spec
from parlant.core.engines.compass.response_state import EngineContext
from parlant.core.glossary import Term
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.sessions import Event, EventId, EventKind, EventSource
from parlant.core.shots import Shot, ShotCollection
from parlant.core.tools import Tool, ToolId

# A guideline distilled from a journey may carry the tools attached to its
# tool-using steps. Tools are provided per guideline, keyed by guideline id.
GuidelineTools = Mapping[GuidelineId, Sequence[tuple[ToolId, Tool]]]


@dataclass(frozen=True)
class DistilledGuideline:
    guideline: Guideline
    reasoning: str
    is_relevant: bool
    distilled_action: Optional[str]


@dataclass(frozen=True)
class GuidelineDistillationResult:
    distilled_guidelines: Sequence[DistilledGuideline]


class GuidelineDistillSchema(DefaultBaseModel):
    reasoning: str
    is_relevant: bool
    distilled_action: Optional[str] = None


@dataclass
class GuidelineDistillationShot(Shot):
    interaction_events: Sequence[Event]
    # The distiller evaluates a single guideline per prompt, so each shot carries one.
    guideline: GuidelineContent
    expected_result: GuidelineDistillSchema


class GuidelineDistiller:
    """Extracts the next relevant action out of a (potentially verbose) guideline.

    A guideline's action may describe many things to do across different situations, or
    spell out a checklist. The distiller evaluates whether the guideline currently
    applies and, if so, extracts only the part of the action that is relevant to the
    next agent response. Each guideline is evaluated in its own prompt.
    """

    def __init__(
        self,
        logger: Logger,
        schematic_generator: SchematicGenerator[GuidelineDistillSchema],
    ) -> None:
        self._logger = logger
        self._schematic_generator = schematic_generator

    async def distill(
        self,
        context: EngineContext,
        guidelines: Sequence[Guideline],
        *,
        tools: GuidelineTools = {},
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]] = [],
        terms: Sequence[Term] = [],
        capabilities: Sequence[Capability] = [],
        staged_events: Sequence[EmittedEvent] = [],
    ) -> GuidelineDistillationResult:
        if not guidelines:
            return GuidelineDistillationResult([])

        distilled_guidelines = await asyncio.gather(
            *(
                self._distill_guideline(
                    context,
                    guideline,
                    tools=tools.get(guideline.id, []),
                    context_variables=context_variables,
                    terms=terms,
                    capabilities=capabilities,
                    staged_events=staged_events,
                )
                for guideline in guidelines
            )
        )

        return GuidelineDistillationResult(list(distilled_guidelines))

    async def _distill_guideline(
        self,
        context: EngineContext,
        guideline: Guideline,
        *,
        tools: Sequence[tuple[ToolId, Tool]],
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        terms: Sequence[Term],
        capabilities: Sequence[Capability],
        staged_events: Sequence[EmittedEvent],
    ) -> DistilledGuideline:
        prompt = self._build_prompt(
            context,
            guideline,
            shots=await self.shots(),
            tools=tools,
            context_variables=context_variables,
            terms=terms,
            capabilities=capabilities,
            staged_events=staged_events,
        )

        inference = await self._schematic_generator.generate(prompt=prompt)

        self._logger.trace(f"Completion:\n{inference.content.model_dump_json(indent=2)}")

        return DistilledGuideline(
            guideline=guideline,
            reasoning=inference.content.reasoning,
            is_relevant=inference.content.is_relevant,
            distilled_action=inference.content.distilled_action,
        )

    async def shots(self) -> Sequence[GuidelineDistillationShot]:
        return await shot_collection.list()

    def _format_shots(self, shots: Sequence[GuidelineDistillationShot]) -> str:
        return "\n".join(
            f"Example #{i}: ###\n{self._format_shot(shot)}" for i, shot in enumerate(shots, start=1)
        )

    def _format_shot(self, shot: GuidelineDistillationShot) -> str:
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
        if expected_result.get("distilled_action") is None:
            expected_result.pop("distilled_action", None)

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
        shots: Sequence[GuidelineDistillationShot],
        *,
        tools: Sequence[tuple[ToolId, Tool]],
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        terms: Sequence[Term],
        capabilities: Sequence[Capability],
        staged_events: Sequence[EmittedEvent],
    ) -> PromptBuilder:
        builder = PromptBuilder(on_build=lambda prompt: self._logger.trace(f"Prompt:\n{prompt}"))

        builder.add_section(
            name="guideline-distiller-general-instructions",
            template="""
GENERAL INSTRUCTIONS
-----------------
In our system, the behavior of a conversational AI agent is guided by "guidelines". The agent makes use of these guidelines whenever it interacts with a user (also referred to as the customer).
Each guideline is composed of two parts:
- "condition": A natural-language condition that specifies when the guideline should apply.
          We examine the conversation in its current state and test this condition
          to determine whether the guideline should inform the next reply to the user.
- "action": A natural-language instruction that the agent should follow whenever the "condition"
          part of the guideline applies to the conversation in its particular state.
          Any instruction described here applies only to the agent, and not to the user.
""",
            props={},
        )
        builder.add_section(
            name="guideline-distiller-task-description",
            template="""
Task Description
----------------
Your task is twofold. First, evaluate whether the provided guideline applies to the most recent state of the interaction between yourself (an AI agent) and a user. Second, if it does apply, determine how its action should be carried out right now and distill it into the single, specific action the agent should take in its very next response.

Determining applicability:
A guideline applies in either of these cases:
1. Its condition is relevant to the latest part of the conversation, and in particular to the most recent customer message; or
2. Its condition applied earlier and the agent is still in the middle of carrying out the action. Many actions span several steps, so a guideline remains applicable until its action has been fully carried out. For example, for the action "when the customer wants a drink, ask which drink and then which size", if the customer asked for a drink and the agent has only asked and received an answer for the first question, the guideline still applies - the agent has yet to ask about the size.

Evaluate the actual meaning of the condition, not just keyword matches, and take the full context into account - including context variables, glossary terms, capabilities, and tool results. Do not consider the guideline applicable based solely on earlier parts of the conversation if the topic has since shifted and its action is not still in progress, even if the previous topic remains unresolved. If the conversation moves from a broader issue to a related sub-issue, the guideline remains applicable as long as it is relevant to that sub-issue; once the discussion has clearly moved on to an entirely different topic, it no longer applies.
Record your applicability decision in the "is_relevant" field. If the guideline does not apply, set "is_relevant" to false and omit "distilled_action" entirely - there is nothing more to do.

Distilling the action:
If the guideline applies, recognize how its action should be carried out right now. A guideline's action is often broad: it may describe several different things to do across different situations, spell out a checklist, or be phrased generally enough that it could be applied in more than one way. Recognizing how the action currently applies means either:
1. Picking the part of the action that is relevant to the current state of the conversation, when the action covers multiple situations; or
2. Choosing how best to apply the action given the specific context of the conversation, when the action is more general.

Some of the action may already have been taken, in full or in part, earlier in the conversation. In that case, output only the part of the action that still needs to be taken now. If the action was already fully taken and its condition has not arisen again for a new reason, there is nothing left to take. If the condition has arisen again for a new reason (a new or subtly different context), the action should be taken again for that new occurrence. Be conservative about repeating actions that deliver static, one-time information (e.g. "send our address"): only repeat them if the condition genuinely arose again.

If a flow has returned to an earlier stage (e.g. the customer corrects something they said earlier), don't just take the step that literally follows the changed point. Skip any later steps whose information you already have and that is still valid, and jump forward to the next step that genuinely still needs doing.

Prioritize choosing a single action. Even when the guideline lists many things to do, return only the one that is appropriate to take next - do not bundle together everything the guideline mentions. It is fine for a single action to ask for several details at once when they naturally belong together, but it should never overwhelm the customer with unrelated requests or with steps that aren't yet relevant. Record the chosen action in the "distilled_action" field.

The exact format of your response will be provided later in this prompt.
""",
            props={},
        )
        builder.add_section(
            name="guideline-distiller-examples",
            template="""
Examples of Guideline Distillations:
-------------------
{formatted_shots}
""",
            props={
                "formatted_shots": self._format_shots(shots),
                "shots": shots,
            },
        )

        builder.add_agent_identity(context.agent)
        builder.add_context_variables(context_variables)
        builder.add_glossary(terms)
        builder.add_capabilities_for_guideline_matching(capabilities)
        builder.add_customer_identity(context.customer, context.session)
        builder.add_interaction_history(context.interaction.events)
        builder.add_staged_tool_events(staged_events)

        builder.add_section(
            name=BuiltInSection.GUIDELINES,
            template="""
- Guideline: ###
{guideline_text}
###
""",
            props={
                "guideline_text": _format_guideline(
                    guideline.content.condition, guideline.content.action, tools
                ),
            },
            status=SectionStatus.ACTIVE,
        )

        builder.add_section(
            name="guideline-distiller-output-format",
            template="""
OUTPUT FORMAT
-----------------
- Evaluate the guideline by filling in the details in the following structure:
```json
{result_structure_text}
```
""",
            props={
                "result_structure_text": self._format_output(),
            },
        )

        return builder

    def _format_output(self) -> str:
        result: dict[str, JSONSerializable] = {
            "reasoning": (
                "<A brief explanation of whether the guideline currently applies and, "
                "if so, which part of its action is relevant to the next agent response>"
            ),
            "is_relevant": "<BOOL indicating whether the guideline currently applies>",
            "distilled_action": (
                "<Include only if is_relevant=True. The specific action that should be "
                "taken next, extracted from the guideline's action. Omit this field "
                "entirely if is_relevant=False>"
            ),
        }

        return json.dumps(result, indent=4)


def _readable_tool_spec(tool_id: ToolId, tool: Tool) -> dict[str, JSONSerializable]:
    # ``get_tool_spec`` renders each parameter as a JSON *string*; left as-is, the
    # surrounding ``json.dumps`` would re-escape it into an unreadable nested string.
    # Parse the parameter specs back into objects so they render as clean nested JSON.
    spec = get_tool_spec(tool_id, tool)
    for key in ("optional_arguments", "required_parameters"):
        params = spec.get(key)
        if isinstance(params, dict):
            spec[key] = {name: json.loads(value) for name, value in params.items()}
    return spec


def _format_guideline(
    condition: str,
    action: Optional[str],
    tools: Sequence[tuple[ToolId, Tool]] = (),
) -> str:
    # The action is optional and only present to contextualize the condition; omit it
    # entirely when absent rather than rendering "Action: None".
    text = f"Condition: {condition}."
    if action:
        text += f" Action: {action}"
    if tools:
        # Surface the tools attached to the action (description + arguments), so the
        # distiller knows what each tool does and can name it as the next step.
        tools_text = json.dumps(
            [_readable_tool_spec(tool_id, tool) for tool_id, tool in tools], indent=2
        )
        text += (
            "\nThe action may be carried out (in full or in part) using the following tools. "
            "When the next step is to run one of these tools, the distilled action should "
            f"say so explicitly:\n{tools_text}"
        )
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


# Shot 1: a journey expressed as a single guideline. The action describes the entire
# multi-step journey, and the distiller must collapse it into just the next step given
# that the journey is already in progress.
example_1_events = [
    _make_event("11", EventSource.CUSTOMER, "Hi, I'd like to book a flight please."),
    _make_event(
        "23",
        EventSource.AI_AGENT,
        "I'd be happy to help you book a flight! Could you please tell me your source and destination airports?",
    ),
    _make_event(
        "34",
        EventSource.CUSTOMER,
        "I want to fly from JFK in New York to LAX in Los Angeles.",
    ),
]

example_1_guideline = GuidelineContent(
    condition="the customer wants to book a flight",
    action=(
        "Ask for the source and destination airports, then for the dates of the departure "
        "and return flight, then whether they want economy or business class, then for the "
        "name of the traveler, and finally book the flight using the book_flight tool."
    ),
)

example_1_expected = GuidelineDistillSchema(
    reasoning=(
        "The customer wants to book a flight and has already provided the source and "
        "destination airports, so the journey is in progress. The next step in the action "
        "is to ask for the departure and return dates."
    ),
    is_relevant=True,
    distilled_action="Ask for the dates of the departure and return flight.",
)


# Shot 2: the guideline's condition does not apply to the current state of the
# conversation at all.
example_2_events = [
    _make_event(
        "11",
        EventSource.CUSTOMER,
        "Hi, I'm planning a trip to Italy next month. What can I do there?",
    ),
    _make_event(
        "23",
        EventSource.AI_AGENT,
        "That sounds exciting! Do you prefer exploring cities or enjoying scenic landscapes?",
    ),
    _make_event(
        "34",
        EventSource.CUSTOMER,
        "Actually I'm also wondering — do I need any special visas or documents as an American citizen?",
    ),
]

example_2_guideline = GuidelineContent(
    condition="The customer is looking for flight or accommodation booking assistance",
    action="Provide links or suggestions for flight aggregators and hotel booking platforms.",
)

example_2_expected = GuidelineDistillSchema(
    reasoning=(
        "The customer is asking about visas and travel documents, not about booking flights "
        "or accommodation, so the condition does not apply to the current state of the "
        "conversation."
    ),
    is_relevant=False,
)


# Shot 3: the guideline applies and its action is a single, concrete instruction that
# should be taken as is.
example_3_events = [
    _make_event(
        "11",
        EventSource.CUSTOMER,
        "Hi there, what is the S&P 500 trading at right now?",
    ),
]

example_3_guideline = GuidelineContent(
    condition="the customer asks about the value of a stock",
    action="provide the price using the 'check_stock_price' tool",
)

example_3_expected = GuidelineDistillSchema(
    reasoning=(
        "The customer is asking about the value of the S&P 500, so the guideline applies. "
        "Its action is a single concrete instruction, so it should be taken as is."
    ),
    is_relevant=True,
    distilled_action="Provide the price of the S&P 500 using the 'check_stock_price' tool.",
)


# Shot 4: the guideline previously applied and its (static) action was already taken;
# there is no new reason to retake it, so no further action is necessary.
example_4_events = [
    _make_event("11", EventSource.CUSTOMER, "Hi, I need help changing the email on my account."),
    _make_event(
        "23",
        EventSource.AI_AGENT,
        "Sure! Could you please provide your account ID so I can verify your identity?",
    ),
    _make_event("34", EventSource.CUSTOMER, "It's ACC12345."),
    _make_event("56", EventSource.AI_AGENT, "Thanks! I've updated your email."),
    _make_event(
        "88",
        EventSource.CUSTOMER,
        "Also, can you check the last payment on my account?",
    ),
]

example_4_guideline = GuidelineContent(
    condition="The customer is asking for account-related help",
    action="Ask for their account ID to verify their identity",
)

example_4_expected = GuidelineDistillSchema(
    reasoning=(
        "The customer is still asking for account-related help, so the guideline applies. "
        "However, they already provided their account ID earlier and it remains valid for "
        "this request, so the action has already been taken and there is no new reason to "
        "ask for it again."
    ),
    is_relevant=True,
    distilled_action="No further action necessary.",
)


# Shot 5: the guideline previously applied and is now triggered again for a new reason;
# the action should be retaken in a way that is specific to the current context.
example_5_events = [
    _make_event(
        "11",
        EventSource.CUSTOMER,
        "I'm planning a trip next month. Any ideas on where to go?",
    ),
    _make_event(
        "23",
        EventSource.AI_AGENT,
        "That sounds exciting! What kind of activities do you enjoy — relaxing on the beach, hiking, museums, food tours?",
    ),
    _make_event("34", EventSource.CUSTOMER, "I love hiking and exploring local food scenes."),
    _make_event(
        "56",
        EventSource.AI_AGENT,
        "Great! You might enjoy a trip to the Pacific Northwest — plenty of trails and great food in Portland and Seattle.",
    ),
    _make_event("88", EventSource.CUSTOMER, "What about a winter trip in Europe?"),
]

example_5_guideline = GuidelineContent(
    condition="The customer wants recommendations for a trip",
    action="Ask for their preferred activities and recommend accordingly",
)

example_5_expected = GuidelineDistillSchema(
    reasoning=(
        "The customer raised a new trip — a winter trip to Europe — so the condition arose "
        "again for a new reason and the action should be reapplied. Their preferred "
        "activities for this new trip aren't known yet, so the next step is to ask about "
        "them for the Europe trip specifically."
    ),
    is_relevant=True,
    distilled_action="Ask the customer what activities they'd enjoy on their winter trip to Europe.",
)


# Shot 6: the flow has returned to an earlier step (the customer changed a detail they
# gave earlier). The later detail they already provided is still valid, so the next step
# is not the one that literally follows the changed step - it's the next step that still
# needs doing.
example_6_events = [
    _make_event("11", EventSource.CUSTOMER, "I'd like to book a home cleaning."),
    _make_event("21", EventSource.AI_AGENT, "Sure! What date would you like?"),
    _make_event("31", EventSource.CUSTOMER, "Next Tuesday."),
    _make_event("41", EventSource.AI_AGENT, "Got it. What's the address?"),
    _make_event("51", EventSource.CUSTOMER, "42 Oak Street."),
    _make_event("61", EventSource.AI_AGENT, "And how many rooms need cleaning?"),
    _make_event(
        "71",
        EventSource.CUSTOMER,
        "Actually, can we make it next Wednesday instead of Tuesday?",
    ),
]

example_6_guideline = GuidelineContent(
    condition="the customer wants to book a home cleaning",
    action=(
        "Ask for the desired date, then for the home address, then for the number of rooms, "
        "and finally confirm the details and book the cleaning."
    ),
)

example_6_expected = GuidelineDistillSchema(
    reasoning=(
        "The customer changed the date, returning to an earlier step of the action. The home "
        "address they already gave is still valid, so there's no need to ask for it again. The "
        "next step that still needs doing is asking how many rooms need cleaning."
    ),
    is_relevant=True,
    distilled_action="Ask the customer how many rooms need cleaning.",
)


_baseline_shots: Sequence[GuidelineDistillationShot] = [
    GuidelineDistillationShot(
        description="",
        interaction_events=example_1_events,
        guideline=example_1_guideline,
        expected_result=example_1_expected,
    ),
    GuidelineDistillationShot(
        description="",
        interaction_events=example_2_events,
        guideline=example_2_guideline,
        expected_result=example_2_expected,
    ),
    GuidelineDistillationShot(
        description="",
        interaction_events=example_3_events,
        guideline=example_3_guideline,
        expected_result=example_3_expected,
    ),
    GuidelineDistillationShot(
        description="",
        interaction_events=example_4_events,
        guideline=example_4_guideline,
        expected_result=example_4_expected,
    ),
    GuidelineDistillationShot(
        description="",
        interaction_events=example_5_events,
        guideline=example_5_guideline,
        expected_result=example_5_expected,
    ),
    GuidelineDistillationShot(
        description="",
        interaction_events=example_6_events,
        guideline=example_6_guideline,
        expected_result=example_6_expected,
    ),
]

shot_collection = ShotCollection[GuidelineDistillationShot](_baseline_shots)
