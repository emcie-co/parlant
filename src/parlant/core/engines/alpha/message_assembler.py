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

from dataclasses import dataclass
from itertools import chain
import json
import traceback
from typing import Any, Mapping, Optional, Sequence

from parlant.core.contextual_correlator import ContextualCorrelator
from parlant.core.agents import Agent, CompositionMode
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.customers import Customer
from parlant.core.engines.alpha.message_event_composer import (
    MessageCompositionError,
    MessageEventComposer,
    MessageEventComposition,
)
from parlant.core.engines.alpha.tool_caller import ToolInsights
from parlant.core.fragments import Fragment, FragmentId, FragmentStore
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.engines.alpha.guideline_proposition import GuidelineProposition
from parlant.core.engines.alpha.prompt_builder import PromptBuilder, BuiltInSection, SectionStatus
from parlant.core.glossary import Term
from parlant.core.emissions import EmittedEvent, EventEmitter
from parlant.core.sessions import Event, MessageEventData, Participant
from parlant.core.common import DefaultBaseModel
from parlant.core.loggers import Logger
from parlant.core.shots import Shot, ShotCollection
from parlant.core.tools import ToolId


class ContextEvaluation(DefaultBaseModel):
    most_recent_user_inquiries_or_needs: Optional[str] = None
    parts_of_the_context_i_have_here_if_any_with_specific_information_on_how_to_address_these_needs: Optional[
        str
    ] = None
    topics_for_which_i_have_sufficient_information_and_can_therefore_help_with: Optional[str] = None
    what_i_do_not_have_enough_information_to_help_with_with_based_on_the_provided_information_that_i_have: Optional[
        str
    ] = None
    was_i_given_specific_information_here_on_how_to_address_some_of_these_specific_needs: Optional[
        bool
    ] = None
    should_i_tell_the_user_i_cannot_help_with_some_of_those_needs: Optional[bool] = None


class MaterializedFragmentField(DefaultBaseModel):
    field_name: str
    have_sufficient_data_in_context: Optional[bool] = None
    value: Optional[str] = None


class MaterializedFragment(DefaultBaseModel):
    fragment_id: str
    raw_content: str
    fields: Optional[list[MaterializedFragmentField]] = None
    justification: str


class Revision(DefaultBaseModel):
    revision_number: int
    insights_about_the_user: Optional[str] = None
    selected_content_fragments: list[MaterializedFragment]
    sequenced_rendered_content_fragments: list[str]
    composited_fragment_sequence: str
    instructions_followed: Optional[list[str]] = None
    instructions_broken: Optional[list[str]] = None
    is_practically_repeating_yourself: Optional[bool] = None
    followed_all_instructions: Optional[bool] = None
    instructions_broken_due_to_missing_data: Optional[bool] = None
    missing_data_rationale: Optional[str] = None
    instructions_broken_only_due_to_prioritization: Optional[bool] = None
    prioritization_rationale: Optional[str] = None


class InstructionEvaluation(DefaultBaseModel):
    number: int
    instruction: str
    evaluation: str
    data_available: str
    do_i_have_fragments_in_the_bank_for_fulfilling_this_instruction: bool
    if_i_do_not_have_fragments_for_fulfilling_then_do_i_at_least_have_fragments_to_explain_that_i_cannot_help: Optional[
        bool
    ] = None


class AssembledMessageSchema(DefaultBaseModel):
    last_message_of_user: Optional[str]
    produced_reply: Optional[bool] = None
    produced_reply_rationale: Optional[str] = None
    guidelines: list[str]
    context_evaluation: Optional[ContextEvaluation] = None
    insights: Optional[list[str]] = None
    evaluation_for_each_instruction: Optional[list[InstructionEvaluation]] = None
    revisions: list[Revision]


@dataclass
class MessageAssemblerShot(Shot):
    composition_modes: list[CompositionMode]
    expected_result: AssembledMessageSchema


@dataclass(frozen=True)
class _MessageAssemblyGenerationResult:
    message: str
    fragments: dict[FragmentId, str]


class MessageAssembler(MessageEventComposer):
    def __init__(
        self,
        logger: Logger,
        correlator: ContextualCorrelator,
        schematic_generator: SchematicGenerator[AssembledMessageSchema],
        fragment_store: FragmentStore,
    ) -> None:
        self._logger = logger
        self._correlator = correlator
        self._schematic_generator = schematic_generator
        self._fragment_store = fragment_store

    async def shots(self, composition_mode: CompositionMode) -> Sequence[MessageAssemblerShot]:
        shots = await shot_collection.list()
        supported_shots = [s for s in shots if composition_mode in s.composition_modes]
        return supported_shots

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
    ) -> Sequence[MessageEventComposition]:
        with self._logger.scope("MessageEventComposer"):
            with self._logger.scope("Assembly"):
                with self._logger.operation("Message generation"):
                    return await self._do_generate_events(
                        event_emitter,
                        agent,
                        customer,
                        context_variables,
                        interaction_history,
                        terms,
                        ordinary_guideline_propositions,
                        tool_enabled_guideline_propositions,
                        tool_insights,
                        staged_events,
                    )

    async def _do_generate_events(
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
    ) -> Sequence[MessageEventComposition]:
        if (
            not interaction_history
            and not ordinary_guideline_propositions
            and not tool_enabled_guideline_propositions
        ):
            # No interaction and no guidelines that could trigger
            # a proactive start of the interaction
            self._logger.info("Skipping response; interaction is empty and there are no guidelines")
            return []

        fragments = await self._fragment_store.list_fragments()

        prompt = self._build_prompt(
            agent=agent,
            context_variables=context_variables,
            customer=customer,
            interaction_history=interaction_history,
            terms=terms,
            ordinary_guideline_propositions=ordinary_guideline_propositions,
            tool_enabled_guideline_propositions=tool_enabled_guideline_propositions,
            staged_events=staged_events,
            tool_insights=tool_insights,
            fragments=fragments,
            shots=await self.shots(agent.composition_mode),
        )

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
                generation_info, assembly_result = await self._generate_response_message(
                    prompt,
                    fragments,
                    agent.composition_mode,
                    temperature=generation_attempt_temperatures[generation_attempt],
                    final_attempt=(generation_attempt + 1) == len(generation_attempt_temperatures),
                )

                if assembly_result is not None:
                    event = await event_emitter.emit_message_event(
                        correlation_id=self._correlator.correlation_id,
                        data=MessageEventData(
                            message=assembly_result.message,
                            participant=Participant(id=agent.id, display_name=agent.name),
                            fragments={
                                id: value for id, value in assembly_result.fragments.items()
                            },
                        ),
                    )

                    return [MessageEventComposition(generation_info, [event])]
                else:
                    self._logger.debug("Skipping response; no response deemed necessary")
                    return [MessageEventComposition(generation_info, [])]
            except Exception as exc:
                self._logger.warning(
                    f"Generation attempt {generation_attempt} failed: {traceback.format_exception(exc)}"
                )
                last_generation_exception = exc

        raise MessageCompositionError() from last_generation_exception

    def _get_fragment_bank_text(
        self,
        fragments: Sequence[Fragment],
    ) -> tuple[str, list[str]]:
        template = """
In formulating your reply, you must rely on the following bank of fragments.
Each fragment contains content, which may or may not refer to "fragment fields" using curly braces.
For example, in the fragment 'I can help you with {{something}}', there is one fragment field called 'something'.
For your references, some fragment may include some examples for how to fill out their fragment fields properly.

Note: If you do not have fragments for fulfilling any instruction, you should at least try to
explain to the user that cannot help (even if only because you don't have the necessary fragments).
Only attempt to say something like this if you do at least have fragments in the bank that help
you explain this situation (the very fact you cannot help).

FRAGMENT BANK:
--------------
{rendered_fragments}
"""

        rendered_fragments = []

        for fragment in fragments:
            fragment_dict: dict[str, Any] = {"fragment_id": fragment.id, "value": fragment.value}

            if fragment.fields:
                fragment_dict["fields"] = {}

                for field in fragment.fields:
                    field_description = field.description

                    if field.examples:
                        examples = []

                        for i, example in enumerate(field.examples, start=1):
                            examples.append(f"{i}) {example}")

                        field_description += f" -- Example Extractions (only use these for reference on how to properly extract values in the right format): {'; '.join(examples)}"

                    fragment_dict["fields"][field.name] = field_description

            rendered_fragments.append(str(fragment_dict))

        template

        return template, rendered_fragments

    def _get_guideline_propositions_text(
        self,
        ordinary: Sequence[GuidelineProposition],
        tool_enabled: Mapping[GuidelineProposition, Sequence[ToolId]],
    ) -> str:
        all_propositions = list(chain(ordinary, tool_enabled))

        if not all_propositions:
            return """
In formulating your reply, you are normally required to follow a number of behavioral guidelines.
However, in this case, no special behavioral guidelines were provided. Therefore, when generating revisions,
you don't need to specifically double-check if you followed or broke any guidelines.
"""
        guidelines = []

        for i, p in enumerate(all_propositions, start=1):
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
Do not disregard a guideline because you believe its 'when' condition or rationale does not apply—this filtering has already been handled.

- **Guidelines**:
{guideline_list}
"""

    def _format_shots(
        self,
        shots: Sequence[MessageAssemblerShot],
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
        shot: MessageAssemblerShot,
    ) -> str:
        return f"""
- **Expected Result**:
```json
{json.dumps(shot.expected_result.model_dump(mode="json", exclude_unset=True), indent=2)}
```"""

    def _build_prompt(
        self,
        agent: Agent,
        customer: Customer,
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        ordinary_guideline_propositions: Sequence[GuidelineProposition],
        tool_enabled_guideline_propositions: Mapping[GuidelineProposition, Sequence[ToolId]],
        staged_events: Sequence[EmittedEvent],
        tool_insights: ToolInsights,
        fragments: Sequence[Fragment],
        shots: Sequence[MessageAssemblerShot],
    ) -> PromptBuilder:
        can_suggest_fragments = agent.composition_mode == "fluid_assembly"

        builder = PromptBuilder(on_build=lambda prompt: self._logger.debug(f"Prompt:\n{prompt}"))

        builder.add_section(
            name="message-assembler-general-instructions",
            template="""
GENERAL INSTRUCTIONS
-----------------
You are an AI agent who is part of a system that interacts with a customer, also referred to as 'the user'. The current state of this interaction will be provided to you later in this message.
You role is to generate a reply message to the current (latest) state of the interaction, based on provided guidelines and background information.

Later in this prompt, you'll be provided with behavioral guidelines and other contextual information you must take into account when generating your response.

""",
            props={},
        )

        builder.add_agent_identity(agent)
        builder.add_section(
            name="message-assembler-task-description",
            template="""
TASK DESCRIPTION:
-----------------
Continue the provided interaction in a natural and human-like manner.
Your task is to produce a response to the latest state of the interaction.
Always abide by the following general principles (note these are not the "guidelines". The guidelines will be provided later):
1. GENERAL BEHAVIOR: Make your response as human-like as possible. Be concise and avoid being overly polite when not necessary.
2. AVOID REPEATING YOURSELF: When replying— avoid repeating yourself. Instead, refer the customer to your previous answer, or choose a new approach altogether. If a conversation is looping, point that out to the customer instead of maintaining the loop.
3. DO NOT HALLUCINATE: Do not state factual information that you do not know or are not sure about. If the customer requests information you're unsure about, state that this information is not available to you.
4. ONLY OFFER SERVICES AND INFORMATION PROVIDED IN THIS PROMPT: Do not output information or offer services based on your intrinsic knowledge - you must only represent the business according to the information provided in this prompt.
5. REITERATE INFORMATION FROM PREVIOUS MESSAGES IF NECESSARY: If you previously suggested a solution or shared information during the interaction, you may repeat it when relevant. Your earlier response may have been based on information that is no longer available to you, so it’s important to trust that it was informed by the context at the time.
6. MAINTAIN GENERATION SECRECY: Never reveal details about the process you followed to produce your response. Do not explicitly mention the tools, context variables, guidelines, glossary, or any other internal information. Present your replies as though all relevant knowledge is inherent to you, not derived from external instructions.
7. OUTPUT FORMAT: In your generated reply to the customer, use markdown format when applicable.
""",
            props={},
        )
        if not interaction_history or all(
            [event.kind != "message" for event in interaction_history]
        ):
            builder.add_section(
                name="message-assembler-initial-message-instructions",
                template="""
The interaction with the customer has just began, and no messages were sent by either party.
If told so by a guideline or some other contextual condition, send the first message. Otherwise, do not produce a reply.
If you decide not to emit a message, output the following:
{{
    "last_message_of_customer": None,
    "produced_reply": false,
    "guidelines": <list of strings- a re-statement of all guidelines>,
    "context_evaluation": None,
    "insights": [<list of strings- up to 3 original insights>],
    "produced_reply_rationale": "<a few words to justify why a reply was NOT produced here>",
    "revisions": []
}}
Otherwise, follow the rest of this prompt to choose the content of your response.
        """,
                props={},
            )

        else:
            builder.add_section(
                name="message-assembler-ongoing-interaction-instructions",
                template="""
Since the interaction with the customer is already ongoing, always produce a reply to the customer's last message.
The only exception where you may not produce a reply is if the customer explicitly asked you not to respond to their message.
In all other cases, even if the customer is indicating that the conversation is over, you must produce a reply.
                """,
                props={},
            )

        if can_suggest_fragments:
            fragment_instruction = """
Prefer to use fragments from the bank in generating the revision's content.
If no viable fragments exist in the bank, you may suggest new fragments.
For suggested fragments, use the special ID "<auto>".
"""
        else:
            fragment_instruction = "You can ONLY USE FRAGMENTS FROM THE FRAGMENT BANK in generating the revision's content."

        builder.add_section(
            name="message-assembler-revision-mechanism",
            template="""
REVISION MECHANISM
-----------------
To craft an optimal response, you must produce incremental revisions of your reply, ensuring alignment with all provided guidelines based on the latest interaction state.
Each critique during the revision process should be unique to avoid redundancy.

Your final reply must comply with the outlined guidelines and the instructions in this prompt.

Before drafting replies and revisions, identify up to three key insights based on this prompt and the ongoing conversation.
These insights should include relevant customer requests, applicable principles from this prompt, or conclusions drawn from the interaction.
Ensure to include any customer request as an insight, whether it's explicit or implicit.
Do not add insights unless you believe that they are absolutely necessary. Prefer suggesting fewer insights, if at all.
When revising, indicate whether each guideline and insight is satisfied in the suggested reply.

Also note that the content of each revision is to be made up using FRAGMENTS.

How to use fragments:
    - {fragment_instruction}
    - When listing fragments from the bank, must be displayed EXACTLY AS-IS FROM THE BANK.
    - Some fragments have "fragment fields" that you need to fill out by extracting relevant information from the content.
    - If you don't have sufficient data in the context to fill out a fragment field, you should explicitly give it a null value.
    - The composited content may contain VERY SPECIFIC EDITS to the final sequencing, such as capitalization fixes and connective punctuation marks between fragments.

The final output must be a JSON document detailing the message development process, including:
    - Insights to abide by,
    - If and how each instruction (guidelines and insights) was adhered to,
    - Instances where one instruction was prioritized over another,
    - Situations where guidelines and insights were unmet due to insufficient context or data,
    - Justifications for all decisions made during the revision process.
    - A marking for whether the suggested response repeats previous messages. If the response is repetitive, continue revising until it is sufficiently unique.

Do not exceed 5 revisions. If you reach the 5th revision, stop there.


PRIORITIZING INSTRUCTIONS (GUIDELINES VS. INSIGHTS)
-----------------
Deviating from an instruction (either guideline or insight) is acceptable only when the deviation arises from a deliberate prioritization, based on:
    - Conflicts with a higher-priority guideline (according to their priority scores).
    - Contradictions with a customer request.
    - Lack of sufficient context or data.
    - Conflicts with an insight (see below).
In all other cases, even if you believe that a guideline's condition does not apply, you must follow it.
If fulfilling a guideline is not possible, explicitly justify why in your response.

Guidelines vs. Insights:
Sometimes, a guideline may conflict with an insight you've derived.
For example, if your insight suggests "the customer is vegetarian," but a guideline instructs you to offer non-vegetarian dishes, prioritizing the insight would better align with the business's goals—since offering vegetarian options would clearly benefit the customer.

However, remember that the guidelines reflect the explicit wishes of the business you represent. Deviating from them should only occur if doing so does not put the business at risk.
For instance, if a guideline explicitly prohibits a specific action (e.g., "never do X"), you must not perform that action, even if requested by the customer or supported by an insight.

In cases of conflict, prioritize the business's values and ensure your decisions align with their overarching goals.

""",
            props={"fragment_instruction": fragment_instruction},
        )
        builder.add_section(
            name="message-assembler-examples",
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
        builder.add_glossary(terms)
        fragment_bank_template, fragment_bank_rendered_fragments = self._get_fragment_bank_text(
            fragments
        )
        builder.add_section(
            name="message-assembler-fragment-bank",
            template=fragment_bank_template,
            props={
                "fragments": fragments,
                "rendered_fragments": fragment_bank_rendered_fragments,
            },
        )
        builder.add_section(
            name=BuiltInSection.GUIDELINE_DESCRIPTIONS,
            template=self._get_guideline_propositions_text(
                ordinary_guideline_propositions,
                tool_enabled_guideline_propositions,
            ),
            props={
                "ordinary_guideline_propositions": ordinary_guideline_propositions,
                "tool_enabled_guideline_propositions": tool_enabled_guideline_propositions,
            },
            status=SectionStatus.ACTIVE
            if ordinary_guideline_propositions or tool_enabled_guideline_propositions
            else SectionStatus.PASSIVE,
        )
        builder.add_interaction_history(interaction_history)
        builder.add_staged_events(staged_events)

        if tool_insights.missing_data:
            builder.add_section(
                name="message-assembler-missing-data-for-tools",
                template="""
MISSING DATA FOR TOOL REQUIRED CALLS:
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

        builder.add_section(
            name="message-assembler-output-format",
            template="""
Produce a valid JSON object in the following format: ###

{formatted_output_format}
""",
            props={
                "formatted_output_format": self._get_output_format(
                    interaction_history,
                    list(
                        chain(ordinary_guideline_propositions, tool_enabled_guideline_propositions)
                    ),
                    can_suggest_fragments,
                ),
                "interaction_history": interaction_history,
                "guidelines": list(
                    chain(ordinary_guideline_propositions, tool_enabled_guideline_propositions)
                ),
                "can_suggest_fragments": can_suggest_fragments,
            },
        )

        return builder

    def _get_output_format(
        self,
        interaction_history: Sequence[Event],
        guidelines: Sequence[GuidelineProposition],
        allow_suggestions: bool,
    ) -> str:
        last_customer_message = next(
            (
                event.data["message"] if not event.data.get("flagged", False) else "<N/A>"
                for event in reversed(interaction_history)
                if (
                    event.kind == "message"
                    and event.source == "customer"
                    and isinstance(event.data, dict)
                )
            ),
            "",
        )
        guidelines_list_text = ", ".join([f'"{g.guideline}"' for g in guidelines])
        guidelines_output_format = "\n".join(
            [
                f"""
        {{
            "number": {i},
            "instruction": "{g.guideline.content.action}",
            "evaluation": "<your evaluation of how the guideline should be followed>",
            "data_available": "<explanation whether you are provided with the required data to follow this guideline now>",
            "do_i_have_fragments_in_the_bank_for_fulfilling_this_instruction": <BOOL>,
            "if_i_do_not_have_fragments_for_fulfilling_then_do_i_at_least_have_fragments_to_explain_that_i_cannot_help": <BOOL; optional, only if the previous is false>
        }},"""
                for i, g in enumerate(guidelines, start=1)
            ]
        )

        if len(guidelines) == 0:
            insights_output_format = """
            {{
                "number": 1,
                "instruction": "<Insight #1, if it exists>",
                "evaluation": "<your evaluation of how the insight should be followed>",
                "data_available": "<explanation whether you are provided with the required data to follow this insight now>",
                "do_i_have_fragments_in_the_bank_for_fulfilling_this_instruction": <BOOL>,
                "if_i_do_not_have_fragments_for_fulfilling_then_do_i_at_least_have_fragments_to_explain_that_i_cannot_help": <BOOL>
            }},
            <Additional entries for all insights>
        """
        else:
            insights_output_format = """
            <Additional entries for all insights>
"""

        return f"""
{{
    "last_message_of_user": "{last_customer_message}",
    "produced_reply": "<BOOL, should be true unless the user explicitly asked you not to respond>",
    "produced_reply_rationale": "<str, optional. required only if produced_reply is false>",
    "guidelines": [{guidelines_list_text}],
    "context_evaluation": {{
        "most_recent_user_inquiries_or_needs": "<fill out accordingly>",
        "parts_of_the_context_i_have_here_if_any_with_specific_information_on_how_to_address_these_needs": "<fill out accordingly>",
        "topics_for_which_i_have_sufficient_information_and_can_therefore_help_with": "<fill out accordingly>",
        "what_i_do_not_have_enough_information_to_help_with_with_based_on_the_provided_information_that_i_have": "<fill out accordingly>",
        "was_i_given_specific_information_here_on_how_to_address_some_of_these_specific_needs": <BOOL>,
        "should_i_tell_the_user_i_cannot_help_with_some_of_those_needs": <BOOL>
    }},
    "insights": [<Up to 3 original insights to adhere to>],
    "evaluation_for_each_instruction": [
{guidelines_output_format}
{insights_output_format}
    ],
    "revisions": [
    {{
        "revision_number": 1,
        "insights_about_the_user": "<insights based on your fragment selection and what you know about the user>",
        "selected_content_fragments": [
            {{
                "fragment_id": "<chosen fragment_id from bank>{' or <auto> if you suggested this fragment yourself' if allow_suggestions else ''}",
                "raw_content": "<raw fragment content>",
                "fields": [{{
                        "field_name": "<fragment field name from this fragment id>",
                        "have_sufficient_data_in_context": <BOOL whether you have enough data in context to fill out this fragment field's value>,
                        "value": "<fragment field value>"
                    }}
                }}],
                "justification": "<brief justification for choosing this fragment here>"
            }},
            ...
        ],
        "sequenced_rendered_content_fragments": <a raw sequenced list of the chosen fragments with fragment fields replaced by their values, ith capitalization or puncutation fixes as needed. DO NOT ADD ANY WORDS HERE, ONLY PUNCTUATION MARKS ARE ACCEPTABLE AT THIS STAGE.>,
        "composited_fragment_sequence": "<a composited version of the sequenced rendering, with ONLY GRAMMATICAL (NON-SEMANTIC) EDITS to make them blend together correctly>",
        "instructions_followed": <list of guidelines and insights that were followed>,
        "instructions_broken": <list of guidelines and insights that were broken>,
        "is_practically_repeating_yourself": <BOOL, indicating whether "content" is a repeat of a previous message by the agent>,
        "followed_all_instructions": <BOOL, whether all guidelines and insights followed>,
        "instructions_broken_due_to_missing_data": <BOOL, optional. Necessary only if instructions_broken_only_due_to_prioritization is true>,
        "missing_data_rationale": <STR, optional. Necessary only if instructions_broken_due_to_missing_data is true>,
        "instructions_broken_only_due_to_prioritization": <BOOL, optional. Necessary only if followed_all_instructions is true>,
        "prioritization_rationale": <STR, optional. Necessary only if instructions_broken_only_due_to_prioritization is true>
    }},
    ...
    ]
}}
###"""

    async def _generate_response_message(
        self,
        prompt: PromptBuilder,
        fragments: Sequence[Fragment],
        composition_mode: CompositionMode,
        temperature: float,
        final_attempt: bool,
    ) -> tuple[GenerationInfo, Optional[_MessageAssemblyGenerationResult]]:
        message_event_response = await self._schematic_generator.generate(
            prompt=prompt,
            hints={"temperature": temperature},
        )

        self._logger.debug(
            f"Completion:\n{message_event_response.content.model_dump_json(indent=2)}"
        )

        if message_event_response.content.produced_reply is False:
            self._logger.debug("Produced no reply")
            return message_event_response.info, None

        if first_correct_revision := next(
            (
                r
                for r in message_event_response.content.revisions
                if not r.is_practically_repeating_yourself
                and (
                    r.followed_all_instructions
                    or r.instructions_broken_only_due_to_prioritization
                    or r.instructions_broken_due_to_missing_data
                )
            ),
            None,
        ):
            # Sometimes the LLM continues generating revisions even after
            # it generated a correct one. Those next revisions tend to be
            # faulty, as they do not handle prioritization well. This is a workaround.
            final_revision = first_correct_revision
        else:
            final_revision = message_event_response.content.revisions[-1]

        if (
            not final_revision.followed_all_instructions
            and not final_revision.instructions_broken_only_due_to_prioritization
        ) or final_revision.is_practically_repeating_yourself:
            if message_event_response.content.evaluation_for_each_instruction and all(
                e.do_i_have_fragments_in_the_bank_for_fulfilling_this_instruction
                or e.if_i_do_not_have_fragments_for_fulfilling_then_do_i_at_least_have_fragments_to_explain_that_i_cannot_help
                for e in message_event_response.content.evaluation_for_each_instruction
            ):
                pass
            else:
                self._logger.warning(
                    f"Conceding despite problematic message generation (review completion): {final_revision.composited_fragment_sequence}"
                )

        if (
            composition_mode in ["strict_assembly", "composited_assembly"]
            and not final_revision.selected_content_fragments
        ):
            self._logger.warning(
                "No relevant fragments in the bank to generate a sensible response"
            )
            return message_event_response.info, None

        if len(final_revision.selected_content_fragments) != len(
            final_revision.sequenced_rendered_content_fragments
        ):
            self._logger.error(
                "Selected list of content fragments diverges from list of rendered fragments"
            )

        used_fragments = {}

        for index, materialized_fragment in enumerate(final_revision.selected_content_fragments):
            if materialized_fragment.fragment_id == "<auto>":
                used_fragments[Fragment.TRANSIENT_ID] = materialized_fragment.raw_content
                continue

            fragment = next(
                (
                    fragment
                    for fragment in fragments
                    if fragment.id == materialized_fragment.fragment_id
                ),
                None,
            )

            if not fragment:
                self._logger.error(
                    f"Invalid fragment selection. ID={materialized_fragment.fragment_id}; Value={materialized_fragment.raw_content}; Fields={materialized_fragment.fields}"
                )
                used_fragments[Fragment.INVALID_ID] = materialized_fragment.raw_content
                continue

            if index < len(final_revision.sequenced_rendered_content_fragments):
                used_fragments[fragment.id] = fragment.value
            else:
                self._logger.error(
                    f"Invalid fragment index. ID={materialized_fragment.fragment_id}; Index={index}"
                )
                used_fragments[fragment.id] = "<error: index mismatch>"

        match composition_mode:
            case "fluid_assembly" | "composited_assembly":
                return message_event_response.info, _MessageAssemblyGenerationResult(
                    message=str(final_revision.composited_fragment_sequence),
                    fragments=used_fragments,
                )
            case "strict_assembly":
                return message_event_response.info, _MessageAssemblyGenerationResult(
                    message="".join(final_revision.sequenced_rendered_content_fragments),
                    fragments=used_fragments,
                )

        raise Exception("Unsupported composition mode")


example_1_expected = AssembledMessageSchema(
    last_message_of_user="Hi, I'd like to know the schedule for the next trains to Boston, please.",
    produced_reply=True,
    guidelines=[
        "When the customer asks for train schedules, provide them accurately and concisely."
    ],
    context_evaluation=ContextEvaluation(
        most_recent_user_inquiries_or_needs="Knowing the schedule for the next trains to Boston",
        parts_of_the_context_i_have_here_if_any_with_specific_information_on_how_to_address_these_needs="The interaction history contains a tool call with the train schedule for Boston",
        topics_for_which_i_have_sufficient_information_and_can_therefore_help_with="I can provide the schedule directly from the tool call's result",
        what_i_do_not_have_enough_information_to_help_with_with_based_on_the_provided_information_that_i_have="I am not given the current time so I can't say what trains are *next*",
        was_i_given_specific_information_here_on_how_to_address_some_of_these_specific_needs=True,
        should_i_tell_the_user_i_cannot_help_with_some_of_those_needs=True,
    ),
    insights=[
        "Use markdown format when applicable.",
        "Provide the train schedule without specifying which trains are *next*.",
    ],
    evaluation_for_each_instruction=[
        InstructionEvaluation(
            number=1,
            instruction="When the user asks for train schedules, provide them accurately and concisely.",
            evaluation="The user requested train schedules, so I need to respond with accurate timing information.",
            data_available="Yes, the train schedule data is available.",
            do_i_have_fragments_in_the_bank_for_fulfilling_this_instruction=True,
            if_i_do_not_have_fragments_for_fulfilling_then_do_i_at_least_have_fragments_to_explain_that_i_cannot_help=False,
        ),
        InstructionEvaluation(
            number=2,
            instruction="Use markdown format when applicable.",
            evaluation="Markdown formatting makes the schedule clearer and more readable.",
            data_available="Not specifically needed, but markdown format can be applied to any response.",
            do_i_have_fragments_in_the_bank_for_fulfilling_this_instruction=True,
            if_i_do_not_have_fragments_for_fulfilling_then_do_i_at_least_have_fragments_to_explain_that_i_cannot_help=False,
        ),
        InstructionEvaluation(
            number=3,
            instruction="Provide the train schedule without specifying which trains are *next*.",
            evaluation="I don't want to mislead the user so, while I can provide the schedule, I should be clear that I don't know which trains are next",
            data_available="I have the schedule itself, so I can conform to this instruction.",
            do_i_have_fragments_in_the_bank_for_fulfilling_this_instruction=True,
            if_i_do_not_have_fragments_for_fulfilling_then_do_i_at_least_have_fragments_to_explain_that_i_cannot_help=False,
        ),
    ],
    revisions=[
        Revision(
            revision_number=1,
            selected_content_fragments=[
                MaterializedFragment(
                    fragment_id="<example-id-for-few-shots--do-not-use-this-in-output>",
                    raw_content="Here's the relevant train schedule:\n{schedule_markdown}",
                    fields=[
                        MaterializedFragmentField(
                            field_name="schedule_markdown",
                            have_sufficient_data_in_context=True,
                            value="Train 101 departs at 10:00 AM and arrives at 12:30 PM.\n"
                            "Train 205 departs at 1:00 PM and arrives at 3:45 PM.",
                        )
                    ],
                    justification="Render the train schedule",
                )
            ],
            sequenced_rendered_content_fragments=[
                "Here's the relevant train schedule:\n"
                "Train 101 departs at 10:00 AM and arrives at 12:30 PM.\n"
                "Train 205 departs at 1:00 PM and arrives at 3:45 PM."
            ],
            composited_fragment_sequence=(
                "Here's the relevant train schedule:\n"
                "Train 101 departs at 10:00 AM and arrives at 12:30 PM.\n"
                "Train 205 departs at 1:00 PM and arrives at 3:45 PM."
            ),
            instructions_followed=[
                "#1; When the user asks for train schedules, provide them accurately and concisely."
            ],
            instructions_broken=[
                "#2; Did not use markdown format when applicable.",
                "#3; Was not clear enough that I don't know which trains are next because I don't have the time",
            ],
            is_practically_repeating_yourself=False,
            followed_all_instructions=False,
            instructions_broken_due_to_missing_data=False,
            instructions_broken_only_due_to_prioritization=False,
        ),
        Revision(
            revision_number=2,
            selected_content_fragments=[
                MaterializedFragment(
                    fragment_id="<example-id-for-few-shots--do-not-use-this-in-output>",
                    raw_content="Here's the relevant train schedule:\n{schedule_markdown}",
                    fields=[
                        MaterializedFragmentField(
                            field_name="schedule_markdown",
                            have_sufficient_data_in_context=True,
                            value="""\
| Train | Departure | Arrival |
|-------|-----------|---------|
| 101   | 10:00 AM  | 12:30 PM |
| 205   | 1:00 PM   | 3:45 PM  |""",
                        )
                    ],
                    justification="Render the train schedule",
                )
            ],
            sequenced_rendered_content_fragments=[
                """\
Here's the relevant train schedule:
| Train | Departure | Arrival |
|-------|-----------|---------|
| 101   | 10:00 AM  | 12:30 PM |
| 205   | 1:00 PM   | 3:45 PM  |"""
            ],
            composited_fragment_sequence=(
                """\
Here's the relevant train schedule:

| Train | Departure | Arrival  |
|-------|-----------|----------|
| 101   | 10:00 AM  | 12:30 PM |
| 205   | 1:00 PM   | 3:45 PM  |"""
            ),
            instructions_followed=[
                "#1; When the user asks for train schedules, provide them accurately and concisely.",
                "#2; Use markdown format when applicable.",
                "#3; Clearly stated that I can't guarantee which trains are next as I don't have the time.",
            ],
            instructions_broken=[],
            is_practically_repeating_yourself=False,
            followed_all_instructions=True,
        ),
    ],
)

example_1_shot = MessageAssemblerShot(
    composition_modes=["strict_assembly", "composited_assembly", "fluid_assembly"],
    description="A reply that took critique in a few revisions to get right",
    expected_result=example_1_expected,
)


example_2_expected = AssembledMessageSchema(
    last_message_of_user="<user’s last message in the interaction>",
    guidelines=[
        "When the user chooses and orders a burger, then provide it",
        "When the user chooses specific ingredients on the burger, only provide those ingredients if we have them fresh in stock; otherwise, reject the order",
    ],
    context_evaluation=ContextEvaluation(
        most_recent_user_inquiries_or_needs="<most recent user inquiries or need>",
        parts_of_the_context_i_have_here_if_any_with_specific_information_on_how_to_address_these_needs="<relevant contextual quotes>",
        was_i_given_specific_information_here_on_how_to_address_some_of_these_specific_needs=True,
        should_i_tell_the_user_i_cannot_help_with_some_of_those_needs=False,
        topics_for_which_i_have_sufficient_information_and_can_therefore_help_with="<what you can help with>",
        what_i_do_not_have_enough_information_to_help_with_with_based_on_the_provided_information_that_i_have=None,
    ),
    insights=[],
    evaluation_for_each_instruction=[
        InstructionEvaluation(
            number=1,
            instruction="When the user chooses and orders a burger, then provide it",
            evaluation="This guideline currently applies, so I need to provide the user with a burger.",
            data_available="The burger choice is available in the interaction",
            do_i_have_fragments_in_the_bank_for_fulfilling_this_instruction=True,
            if_i_do_not_have_fragments_for_fulfilling_then_do_i_at_least_have_fragments_to_explain_that_i_cannot_help=True,
        ),
        InstructionEvaluation(
            number=2,
            instruction="When the user chooses specific ingredients on the burger, only provide those ingredients if we have them fresh in stock; otherwise, reject the order.",
            evaluation="The user chose cheese on the burger, but all of the cheese we currently have is expired",
            data_available="The relevant stock availability is given in the tool calls' data. Our cheese has expired.",
            do_i_have_fragments_in_the_bank_for_fulfilling_this_instruction=True,
            if_i_do_not_have_fragments_for_fulfilling_then_do_i_at_least_have_fragments_to_explain_that_i_cannot_help=True,
        ),
    ],
    revisions=[
        Revision(
            revision_number=1,
            insights_about_the_user="The user is a long-time customer and we should treat him with extra respect",
            selected_content_fragments=[
                MaterializedFragment(
                    fragment_id="<example-id-for-few-shots--do-not-use-this-in-output>",
                    raw_content="I'd be happy",
                    justification="Manners",
                ),
                MaterializedFragment(
                    fragment_id="<auto>",
                    raw_content="to",
                    justification="Linking",
                ),
                MaterializedFragment(
                    fragment_id="<example-id-for-few-shots--do-not-use-this-in-output>",
                    raw_content="prepare your burger",
                    justification="User request",
                ),
                MaterializedFragment(
                    fragment_id="<auto>",
                    raw_content="as soon as we",
                    justification="Linking",
                ),
                MaterializedFragment(
                    fragment_id="<example-id-for-few-shots--do-not-use-this-in-output>",
                    raw_content="Restock {something}",
                    fields=[
                        MaterializedFragmentField(
                            field_name="something",
                            have_sufficient_data_in_context=True,
                            value="Requested toppings",
                        )
                    ],
                    justification="Requested toppings aren't in stock",
                ),
            ],
            sequenced_rendered_content_fragments=[
                "I'd be happy ",
                "to ",
                "prepare your burger ",
                "as soon as we ",
                "restock the requested toppings.",
            ],
            composited_fragment_sequence=(
                "I'd be happy to prepare your burger as soon as we restock the requested toppings."
            ),
            instructions_followed=[
                "#2; upheld food quality and did not go on to preparing the burger without fresh toppings."
            ],
            instructions_broken=[
                "#1; did not provide the burger with requested toppings immediately due to the unavailability of fresh ingredients."
            ],
            is_practically_repeating_yourself=False,
            followed_all_instructions=False,
            instructions_broken_only_due_to_prioritization=True,
            prioritization_rationale=(
                "Given the higher priority score of guideline 2, maintaining food quality "
                "standards before serving the burger is prioritized over immediate service."
            ),
            instructions_broken_due_to_missing_data=False,
        )
    ],
)

example_2_shot = MessageAssemblerShot(
    composition_modes=["fluid_assembly"],
    description="A reply where one instruction was prioritized over another",
    expected_result=example_2_expected,
)


example_3_expected = AssembledMessageSchema(
    last_message_of_user="Hi there, can I get something to drink? What do you have on tap?",
    guidelines=["When the user asks for a drink, check the menu and offer what's on it"],
    context_evaluation=ContextEvaluation(
        most_recent_user_inquiries_or_needs="Knowing what drinks we have on tap",
        parts_of_the_context_i_have_here_if_any_with_specific_information_on_how_to_address_these_needs="None",
        was_i_given_specific_information_here_on_how_to_address_some_of_these_specific_needs=False,
        should_i_tell_the_user_i_cannot_help_with_some_of_those_needs=True,
        topics_for_which_i_have_sufficient_information_and_can_therefore_help_with=None,
        what_i_do_not_have_enough_information_to_help_with_with_based_on_the_provided_information_that_i_have="I was not given any contextual information (including tool calls) about what drinks we have at all",
    ),
    insights=[
        "Do not state factual information that you do not know, don't have access to, or are not sure about."
    ],
    evaluation_for_each_instruction=[
        InstructionEvaluation(
            number=1,
            instruction="When the user asks for a drink, check the menu and offer what's on it",
            evaluation="The user did ask for a drink, so I should check the menu to see what's available.",
            data_available="No, I don't have the menu info in the interaction or tool calls",
            do_i_have_fragments_in_the_bank_for_fulfilling_this_instruction=False,
            if_i_do_not_have_fragments_for_fulfilling_then_do_i_at_least_have_fragments_to_explain_that_i_cannot_help=True,
        ),
        InstructionEvaluation(
            number=2,
            instruction="Do not state factual information that you do not know or are not sure about",
            evaluation="There's no information about what we have on tap, so I should not offer any specific option.",
            data_available="No, the list of available drinks is not available to me",
            do_i_have_fragments_in_the_bank_for_fulfilling_this_instruction=True,
            if_i_do_not_have_fragments_for_fulfilling_then_do_i_at_least_have_fragments_to_explain_that_i_cannot_help=False,
        ),
    ],
    revisions=[
        Revision(
            revision_number=1,
            insights_about_the_user="According to contextual information about the user, this is their first time here",
            selected_content_fragments=[
                MaterializedFragment(
                    fragment_id="<example-id-for-few-shots--do-not-use-this-in-output>",
                    raw_content="I'm sorry",
                    justification="Apologize for not having the required info",
                ),
                MaterializedFragment(
                    fragment_id="<example-id-for-few-shots--do-not-use-this-in-output>",
                    raw_content="but",
                    justification="Linking",
                ),
                MaterializedFragment(
                    fragment_id="<example-id-for-few-shots--do-not-use-this-in-output>",
                    raw_content="I'm having trouble accessing {something} at the moment",
                    fields=[
                        MaterializedFragmentField(
                            field_name="something",
                            have_sufficient_data_in_context=True,
                            value="Our menu",
                        )
                    ],
                    justification="Lacking menu information in context (note that I can still fill out this fragment field accordingly)",
                ),
            ],
            sequenced_rendered_content_fragments=[
                "I'm sorry, ",
                "but ",
                "I'm having trouble accessing our menu at the moment.",
            ],
            composited_fragment_sequence="I'm sorry, but I'm having trouble accessing our menu at the moment.",
            instructions_followed=[
                "#2; Do not state factual information that you do not know or are not sure about"
            ],
            instructions_broken=[
                "#1; Lacking menu data in the context prevented me from providing the client with drink information."
            ],
            is_practically_repeating_yourself=False,
            followed_all_instructions=False,
            missing_data_rationale="Menu data was missing",
            instructions_broken_due_to_missing_data=True,
            instructions_broken_only_due_to_prioritization=False,
        )
    ],
)

example_3_shot = MessageAssemblerShot(
    composition_modes=["strict_assembly", "composited_assembly", "fluid_assembly"],
    description="Non-Adherence Due to Missing Data",
    expected_result=example_3_expected,
)


example_4_expected = AssembledMessageSchema(
    last_message_of_user="This is not what I was asking for",
    guidelines=[],
    context_evaluation=ContextEvaluation(
        most_recent_user_inquiries_or_needs="At this point it appears that I do not understand what the user is asking",
    ),
    insights=["I should not keep repeating myself as it makes me sound robotic"],
    evaluation_for_each_instruction=[
        InstructionEvaluation(
            number=1,
            instruction="I should not keep repeating myself as it makes me sound robotic",
            evaluation="If I keep repeating myself in asking for clarifications, it makes me sound robotic and unempathetic as if I'm not really tuned into the user's vibe",
            data_available="None needed",
            do_i_have_fragments_in_the_bank_for_fulfilling_this_instruction=True,
            if_i_do_not_have_fragments_for_fulfilling_then_do_i_at_least_have_fragments_to_explain_that_i_cannot_help=False,
        )
    ],
    revisions=[
        Revision(
            revision_number=1,
            selected_content_fragments=[
                MaterializedFragment(
                    fragment_id="<example-id-for-few-shots--do-not-use-this-in-output>",
                    raw_content="I apologize for {something}",
                    fields=[
                        MaterializedFragmentField(
                            field_name="something",
                            have_sufficient_data_in_context=True,
                            value="the confusion",
                        )
                    ],
                    justification="User is upset",
                ),
                MaterializedFragment(
                    fragment_id="<auto>",
                    raw_content="Could you please explain what I'm missing?",
                    justification="I can't see what I did wrong",
                ),
            ],
            sequenced_rendered_content_fragments=[
                "I apologize for the confusion. ",
                "Could you please explain what I'm missing?",
            ],
            composited_fragment_sequence="I apologize for the confusion. Could you please explain what I'm missing?",
            instructions_followed=[],
            instructions_broken=[
                "#1; I've already apologized and asked for clarifications, and I shouldn't repeat myself"
            ],
            is_practically_repeating_yourself=True,
            followed_all_instructions=False,
        ),
        Revision(
            revision_number=2,
            selected_content_fragments=[
                MaterializedFragment(
                    fragment_id="<example-id-for-few-shots--do-not-use-this-in-output>",
                    raw_content="I apologize for {something}",
                    fields=[
                        MaterializedFragmentField(
                            field_name="something",
                            have_sufficient_data_in_context=True,
                            value="Failing to assist you with your issue",
                        )
                    ],
                    justification="I've failed to understand and help the user",
                ),
                MaterializedFragment(
                    fragment_id="<example-id-for-few-shots--do-not-use-this-in-output>",
                    raw_content="If there's anything else I can do for you, please let me know",
                    justification="I don't want to keep repeating myself asking for clarifications",
                ),
            ],
            sequenced_rendered_content_fragments=[
                "I apologize for failing to assist you with your issue. ",
                "If there's anything else I can do for you, please let me know.",
            ],
            composited_fragment_sequence=(
                "I apologize for failing to assist you with your issue. "
                "If there's anything else I can do for you, please let me know."
            ),
            instructions_followed=[
                "#1; I broke of out of the self-repeating loop by admitting that I can't seem to help"
            ],
            instructions_broken=[],
            is_practically_repeating_yourself=False,
            followed_all_instructions=True,
        ),
    ],
)

example_4_shot = MessageAssemblerShot(
    composition_modes=["strict_assembly", "composited_assembly", "fluid_assembly"],
    description="Avoiding repetitive responses—in this case, given that the previous response by the agent was 'I am sorry, could you please clarify your request?'",
    expected_result=example_4_expected,
)


example_5_expected = AssembledMessageSchema(
    last_message_of_user=(
        "How much money do I have in my account, and how do you know it? Is there some service you use to check "
        "my balance? Can I access it too?"
    ),
    guidelines=["When you need the balance of a user, then use the 'check_balance' tool."],
    context_evaluation=ContextEvaluation(
        most_recent_user_inquiries_or_needs="Know how much money they have in their account; Knowing how and what I use to know how much money they have",
        parts_of_the_context_i_have_here_if_any_with_specific_information_on_how_to_address_these_needs="I know how much money they have based on a tool call's result",
        was_i_given_specific_information_here_on_how_to_address_some_of_these_specific_needs=True,
        should_i_tell_the_user_i_cannot_help_with_some_of_those_needs=False,
        topics_for_which_i_have_sufficient_information_and_can_therefore_help_with="Telling them how much is in their account",
        what_i_do_not_have_enough_information_to_help_with_with_based_on_the_provided_information_that_i_have="I should not expose my internal process, despite their request",
    ),
    insights=["Never reveal details about the process you followed to produce your response"],
    evaluation_for_each_instruction=[
        InstructionEvaluation(
            number=1,
            instruction="use the 'check_balance' tool",
            evaluation="There's already a staged tool call with this tool, so no further action is required.",
            data_available="Yes, I know that the user's balance is 1,000$",
            do_i_have_fragments_in_the_bank_for_fulfilling_this_instruction=True,
            if_i_do_not_have_fragments_for_fulfilling_then_do_i_at_least_have_fragments_to_explain_that_i_cannot_help=True,
        ),
        InstructionEvaluation(
            number=1,
            instruction="Never reveal details about the process you followed to produce your response",
            evaluation="The reply must not reveal details about how I know the client's balance",
            data_available="Not needed",
            do_i_have_fragments_in_the_bank_for_fulfilling_this_instruction=True,
            if_i_do_not_have_fragments_for_fulfilling_then_do_i_at_least_have_fragments_to_explain_that_i_cannot_help=False,
        ),
    ],
    revisions=[
        Revision(
            revision_number=1,
            selected_content_fragments=[
                MaterializedFragment(
                    fragment_id="<example-id-for-few-shots--do-not-use-this-in-output>",
                    raw_content="Your balance is {balance}",
                    fields=[
                        MaterializedFragmentField(
                            field_name="balance",
                            have_sufficient_data_in_context=True,
                            value="$1,000",
                        )
                    ],
                    justification="User requested this information",
                ),
                MaterializedFragment(
                    fragment_id="<example-id-for-few-shots--do-not-use-this-in-output>",
                    raw_content="however",
                    justification="Linking",
                ),
                MaterializedFragment(
                    fragment_id="<example-id-for-few-shots--do-not-use-this-in-output>",
                    raw_content="I'm unable to disclose details about the specific services I use.",
                    justification="I should not reveal my thought process",
                ),
            ],
            sequenced_rendered_content_fragments=[
                "Your balance is $1,000. ",
                "However, ",
                "I'm unable to disclose details about the specific services I use.",
            ],
            composited_fragment_sequence=(
                "Your balance is $1,000. However, I’m unable to disclose details about the specific services I use."
            ),
            instructions_followed=[
                "#1; use the 'check_balance' tool",
                "#2; Never reveal details about the process you followed to produce your response",
            ],
            instructions_broken=[],
            is_practically_repeating_yourself=False,
            followed_all_instructions=True,
        )
    ],
)

example_5_shot = MessageAssemblerShot(
    composition_modes=["strict_assembly", "composited_assembly", "fluid_assembly"],
    description="Not exposing thought process: Assume a tool call for 'check_balance' with a returned value of 1,000$ is staged",
    expected_result=example_5_expected,
)


example_6_expected = AssembledMessageSchema(
    last_message_of_user=("Hey, how can I contact customer support?"),
    guidelines=[],
    context_evaluation=ContextEvaluation(
        most_recent_user_inquiries_or_needs="The user wants to know how to contact customer support",
        parts_of_the_context_i_have_here_if_any_with_specific_information_on_how_to_address_these_needs="The system has given me no information on contacting customer support",
        topics_for_which_i_have_sufficient_information_and_can_therefore_help_with="None in this case; I'm not authorized to offer help beyond my configured capabilities",
        what_i_do_not_have_enough_information_to_help_with_with_based_on_the_provided_information_that_i_have="I cannot help with contacting customer support",
        was_i_given_specific_information_here_on_how_to_address_some_of_these_specific_needs=False,
        should_i_tell_the_user_i_cannot_help_with_some_of_those_needs=True,
    ),
    insights=["When I cannot help with a topic, I should tell the user I can't help with it"],
    evaluation_for_each_instruction=[
        InstructionEvaluation(
            number=1,
            instruction="When I cannot help with a topic, I should tell the user I can't help with it",
            evaluation="Indeed, no information on contacting customer support is provided in my context",
            data_available="Not needed",
            do_i_have_fragments_in_the_bank_for_fulfilling_this_instruction=False,
            if_i_do_not_have_fragments_for_fulfilling_then_do_i_at_least_have_fragments_to_explain_that_i_cannot_help=True,
        ),
    ],
    revisions=[
        Revision(
            revision_number=1,
            selected_content_fragments=[
                MaterializedFragment(
                    fragment_id="<example-id-for-few-shots--do-not-use-this-in-output>",
                    raw_content="Could you please provide more details on {something}",
                    fields=[
                        MaterializedFragmentField(
                            field_name="something",
                            have_sufficient_data_in_context=True,
                            value="What you would need from customer support?",
                        )
                    ],
                    justification="User requested this information",
                ),
                MaterializedFragment(
                    fragment_id="<example-id-for-few-shots--do-not-use-this-in-output>",
                    raw_content="maybe",
                    justification="Linking",
                ),
                MaterializedFragment(
                    fragment_id="<example-id-for-few-shots--do-not-use-this-in-output>",
                    raw_content="I could help you",
                    justification="Offer to help",
                ),
            ],
            sequenced_rendered_content_fragments=[
                "Could you please provide more details on what you would need from customer support? ",
                "Maybe ",
                "I could help you.",
            ],
            composited_fragment_sequence=(
                "Could you please provide more details on what you would need from customer support? Maybe I could help you."
            ),
            instructions_followed=[],
            instructions_broken=[
                "#1; Instead of saying I can't help, I asked for more details from the user",
            ],
            is_practically_repeating_yourself=False,
            followed_all_instructions=False,
        ),
        Revision(
            revision_number=2,
            selected_content_fragments=[
                MaterializedFragment(
                    fragment_id="<example-id-for-few-shots--do-not-use-this-in-output>",
                    raw_content="unfortunately",
                    justification="Manners",
                ),
                MaterializedFragment(
                    fragment_id="<example-id-for-few-shots--do-not-use-this-in-output>",
                    raw_content="I cannot help you with {something} as I do not have enough information about it.",
                    fields=[
                        MaterializedFragmentField(
                            field_name="something",
                            have_sufficient_data_in_context=True,
                            value="This topic",
                        )
                    ],
                    justification="I cannot help with this topic",
                ),
                MaterializedFragment(
                    fragment_id="<example-id-for-few-shots--do-not-use-this-in-output>",
                    raw_content="Is there anything else I can assist you with?",
                    justification="Offer to help",
                ),
            ],
            sequenced_rendered_content_fragments=[
                "Unfortunately, ",
                "I cannot help you with this topic as I do not have enough information about it. ",
                "Is there anything else I can assist you with?",
            ],
            composited_fragment_sequence=(
                "Unfortunately I cannot help you with this topic as I do not have enough information about it. Is there anything else I can assist you with?"
            ),
            instructions_followed=[
                "#1; I adhered to the instruction by clearly stating that I cannot help with this topic",
            ],
            instructions_broken=[],
            is_practically_repeating_yourself=False,
            followed_all_instructions=True,
        ),
    ],
)

example_6_shot = MessageAssemblerShot(
    composition_modes=["strict_assembly", "composited_assembly", "fluid_assembly"],
    description="An insight is derived and followed on not offering to help with something you don't know about",
    expected_result=example_6_expected,
)


example_7_expected = AssembledMessageSchema(
    last_message_of_user="I don't have any android devices, and I do not want to buy a ticket at the moment. Now, what flights are there from New York to Los Angeles tomorrow?",
    guidelines=[
        "When asked anything about plane tickets, suggest completing the order on our android app",
        "When asked about first-class tickets, mention that shorter flights do not offer a complementary meal",
    ],
    context_evaluation=ContextEvaluation(
        most_recent_user_inquiries_or_needs="Knowing what flights there are from NY to LA tomorrow",
        parts_of_the_context_i_have_here_if_any_with_specific_information_on_how_to_address_these_needs="Today's date is [...] and I can see the relevant flight schedule in a staged tool call",
        was_i_given_specific_information_here_on_how_to_address_some_of_these_specific_needs=True,
        should_i_tell_the_user_i_cannot_help_with_some_of_those_needs=False,
        topics_for_which_i_have_sufficient_information_and_can_therefore_help_with="I know the date today, and I have the relevant flight schedule",
        what_i_do_not_have_enough_information_to_help_with_with_based_on_the_provided_information_that_i_have=None,
    ),
    insights=[
        "In your generated reply to the user, use markdown format when applicable.",
        "The user does not have an android device and does not want to buy anything",
    ],
    evaluation_for_each_instruction=[
        InstructionEvaluation(
            number=1,
            instruction="When asked anything about plane tickets, suggest completing the order on our android app",
            evaluation="I should suggest completing the order on our android app",
            data_available="Yes, I know that the name of our android app is BestPlaneTickets",
            do_i_have_fragments_in_the_bank_for_fulfilling_this_instruction=True,
            if_i_do_not_have_fragments_for_fulfilling_then_do_i_at_least_have_fragments_to_explain_that_i_cannot_help=True,
        ),
        InstructionEvaluation(
            number=2,
            instruction="When asked about first-class tickets, mention that shorter flights do not offer a complementary meal",
            evaluation="Evaluating whether the 'when' condition applied is not my role. I should therefore just mention that shorter flights do not offer a complementary meal",
            data_available="not needed",
            do_i_have_fragments_in_the_bank_for_fulfilling_this_instruction=True,
            if_i_do_not_have_fragments_for_fulfilling_then_do_i_at_least_have_fragments_to_explain_that_i_cannot_help=False,
        ),
        InstructionEvaluation(
            number=3,
            instruction="In your generated reply to the user, use markdown format when applicable",
            evaluation="I need to output a message in markdown format",
            data_available="Not needed",
            do_i_have_fragments_in_the_bank_for_fulfilling_this_instruction=True,
            if_i_do_not_have_fragments_for_fulfilling_then_do_i_at_least_have_fragments_to_explain_that_i_cannot_help=False,
        ),
        InstructionEvaluation(
            number=4,
            instruction="The user does not have an android device and does not want to buy anything",
            evaluation="A guideline should not override a user's request, so I should not suggest products requiring an android device",
            data_available="Not needed",
            do_i_have_fragments_in_the_bank_for_fulfilling_this_instruction=True,
            if_i_do_not_have_fragments_for_fulfilling_then_do_i_at_least_have_fragments_to_explain_that_i_cannot_help=False,
        ),
    ],
    revisions=[
        Revision(
            revision_number=1,
            selected_content_fragments=[
                MaterializedFragment(
                    fragment_id="<example-id-for-few-shots--do-not-use-this-in-output>",
                    raw_content="""\
Here are the flights from {origin} to {destination} {when}:
| Option | Departure Airport | Departure Time | Arrival Airport   |
|--------|-------------------|----------------|-------------------|
{schedule_rows}""",
                    fields=[
                        MaterializedFragmentField(
                            field_name="origin",
                            have_sufficient_data_in_context=True,
                            value="New York",
                        ),
                        MaterializedFragmentField(
                            field_name="destination",
                            have_sufficient_data_in_context=True,
                            value="Los Angeles",
                        ),
                        MaterializedFragmentField(
                            field_name="when",
                            have_sufficient_data_in_context=True,
                            value="Tomorrow",
                        ),
                        MaterializedFragmentField(
                            field_name="schedule_rows",
                            have_sufficient_data_in_context=True,
                            value="""\
| Option | Departure Airport | Departure Time | Arrival Airport   |
|--------|-------------------|----------------|-------------------|
| 1      | Newark (EWR)      | 10:00 AM       | Los Angeles (LAX) |
| 2      | JFK               | 3:30 PM        | Los Angeles (LAX) |""",
                        ),
                    ],
                    justification="User asks to depart from New York to Los Angeles tomorrow",
                ),
                MaterializedFragment(
                    fragment_id="<example-id-for-few-shots--do-not-use-this-in-output>",
                    raw_content="While some of these flights are quite long, please note that we do not offer complementary meals on short flights.",
                    justification="Important to keep in mind",
                ),
            ],
            sequenced_rendered_content_fragments=[
                """\
Here are the flights from New York to Los Angeles tomorrow:
| Option | Departure Airport | Departure Time | Arrival Airport   |
|--------|-------------------|----------------|-------------------|
| Option | Departure Airport | Departure Time | Arrival Airport   |
|--------|-------------------|----------------|-------------------|
| 1      | Newark (EWR)      | 10:00 AM       | Los Angeles (LAX) |
| 2      | JFK               | 3:30 PM        | Los Angeles (LAX) |""",
                "While some of these flights are quite long, please note that we do not offer complementary meals on short flights.",
            ],
            composited_fragment_sequence=(
                """
                Here are the flights from New York to Los Angeles tomorrow.

                | Option | Departure Airport | Departure Time | Arrival Airport   |
                |--------|-------------------|----------------|-------------------|
                | 1      | Newark (EWR)      | 10:00 AM       | Los Angeles (LAX) |
                | 2      | JFK               | 3:30 PM        | Los Angeles (LAX) |

                While some of these flights are quite long, please note that we do not offer complementary meals on short flights."""
            ),
            instructions_followed=[
                "#2; When asked about first-class tickets, mention that shorter flights do not offer a complementary meal",
                "#3; In your generated reply to the user, use markdown format when applicable.",
                "#4; The user does not have an android device and does not want to buy anything",
            ],
            instructions_broken=[
                "#1; When asked anything about plane tickets, suggest completing the order on our android app."
            ],
            is_practically_repeating_yourself=False,
            followed_all_instructions=False,
            instructions_broken_only_due_to_prioritization=True,
            prioritization_rationale=(
                "Instructions #1 and #3 contradict each other, and user requests take precedent "
                "over guidelines, so instruction #1 was prioritized."
            ),
            instructions_broken_due_to_missing_data=False,
        )
    ],
)

example_7_shot = MessageAssemblerShot(
    composition_modes=["strict_assembly", "composited_assembly", "fluid_assembly"],
    description="Applying Insight—assuming the agent is provided with a list of outgoing flights from a tool call",
    expected_result=example_7_expected,
)

_baseline_shots: Sequence[MessageAssemblerShot] = [
    example_1_shot,
    example_2_shot,
    example_3_shot,
    example_4_shot,
    example_5_shot,
    example_6_shot,
    example_7_shot,
]

shot_collection = ShotCollection[MessageAssemblerShot](_baseline_shots)
