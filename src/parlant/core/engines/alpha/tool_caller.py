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

from collections import defaultdict
from dataclasses import dataclass, asdict, field
from itertools import chain
import json
import time
import traceback
from typing import Any, Mapping, NewType, Optional, Sequence

from parlant.core import async_utils
from parlant.core.agents import Agent
from parlant.core.common import JSONSerializable, generate_id, DefaultBaseModel
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.emissions import EmittedEvent
from parlant.core.engines.alpha.guideline_proposition import GuidelineProposition
from parlant.core.engines.alpha.prompt_builder import PromptBuilder, BuiltInSection, SectionStatus
from parlant.core.glossary import Term
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.sessions import Event, ToolResult
from parlant.core.shots import Shot, ShotCollection
from parlant.core.tools import Tool, ToolContext, ToolParameterDescriptor, ToolParameterOptions
from parlant.core.tools import ToolId, ToolService

ToolCallId = NewType("ToolCallId", str)
ToolResultId = NewType("ToolResultId", str)


class ArgumentEvaluation(DefaultBaseModel):
    parameter_name: str
    acceptable_source_for_this_argument_according_to_its_tool_definition: str
    evaluate_is_it_provided_by_an_acceptable_source: str
    evaluate_was_it_already_provided_and_should_it_be_provided_again: str
    evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided: str
    is_optional: bool
    has_default_value_if_not_provided_by_acceptable_source: Optional[bool] = None
    is_missing: bool
    value_as_string: Optional[str] = None


class ToolCallEvaluation(DefaultBaseModel):
    applicability_rationale: str
    applicability_score: int
    argument_evaluations: Optional[list[ArgumentEvaluation]] = None
    same_call_is_already_staged: bool
    comparison_with_rejected_tools_including_references_to_subtleties: str
    relevant_subtleties: str
    a_rejected_tool_would_have_been_a_better_fit_if_it_werent_already_rejected: bool
    potentially_better_rejected_tool_name: Optional[str] = None
    potentially_better_rejected_tool_rationale: Optional[str] = None
    the_better_rejected_tool_should_clearly_be_run_in_tandem_with_the_candidate_tool: Optional[
        bool
    ] = None
    # These 3 ARQs are for cases we've observed where many optional arguments are missing
    # such that the model would be possibly biased to say the tool shouldn't run.
    are_optional_arguments_missing: bool
    are_non_optional_arguments_missing: bool
    allowed_to_run_without_optional_arguments_even_if_they_are_missing: bool
    should_run: bool


class ToolCallInferenceSchema(DefaultBaseModel):
    last_customer_message: Optional[str] = None
    most_recent_customer_inquiry_or_need: Optional[str] = None
    most_recent_customer_inquiry_or_need_was_already_resolved: Optional[bool] = None
    name: str
    subtleties_to_be_aware_of: str
    tool_calls_for_candidate_tool: list[ToolCallEvaluation]


@dataclass
class ToolCallerInferenceShot(Shot):
    expected_result: ToolCallInferenceSchema


@dataclass(frozen=True)
class ToolCall:
    id: ToolCallId
    tool_id: ToolId
    arguments: Mapping[str, JSONSerializable]

    def __eq__(self, value: object) -> bool:
        if isinstance(value, ToolCall):
            return bool(self.tool_id == value.tool_id and self.arguments == value.arguments)
        return False


@dataclass(frozen=True)
class ToolCallResult:
    id: ToolResultId
    tool_call: ToolCall
    result: ToolResult


@dataclass(frozen=True)
class MissingToolData:
    parameter: str
    significance: Optional[str] = field(default=None)
    description: Optional[str] = field(default=None)
    examples: Optional[Sequence[str]] = field(default=None)


@dataclass(frozen=True)
class ToolInsights:
    missing_data: Sequence[MissingToolData] = field(default_factory=list)


@dataclass(frozen=True)
class InferenceToolCallsResult:
    total_duration: float
    batch_count: int
    batch_generations: Sequence[GenerationInfo]
    batches: Sequence[Sequence[ToolCall]]
    insights: ToolInsights


class ToolCaller:
    def __init__(
        self,
        logger: Logger,
        service_registry: ServiceRegistry,
        schematic_generator: SchematicGenerator[ToolCallInferenceSchema],
    ) -> None:
        self._service_registry = service_registry
        self._logger = logger
        self._schematic_generator = schematic_generator

    async def infer_tool_calls(
        self,
        agent: Agent,
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        ordinary_guideline_propositions: Sequence[GuidelineProposition],
        tool_enabled_guideline_propositions: Mapping[GuidelineProposition, Sequence[ToolId]],
        staged_events: Sequence[EmittedEvent],
    ) -> InferenceToolCallsResult:
        with self._logger.scope("ToolCaller"):
            return await self._do_infer_tool_calls(
                agent,
                context_variables,
                interaction_history,
                terms,
                ordinary_guideline_propositions,
                tool_enabled_guideline_propositions,
                staged_events,
            )

    async def _do_infer_tool_calls(
        self,
        agent: Agent,
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        ordinary_guideline_propositions: Sequence[GuidelineProposition],
        tool_enabled_guideline_propositions: Mapping[GuidelineProposition, Sequence[ToolId]],
        staged_events: Sequence[EmittedEvent],
    ) -> InferenceToolCallsResult:
        if not tool_enabled_guideline_propositions:
            return InferenceToolCallsResult(
                total_duration=0.0,
                batch_count=0,
                batch_generations=[],
                batches=[],
                insights=ToolInsights(),
            )

        batches: dict[tuple[ToolId, Tool], list[GuidelineProposition]] = defaultdict(list)
        services: dict[str, ToolService] = {}

        for guideline_proposition, tool_ids in tool_enabled_guideline_propositions.items():
            for tool_id in tool_ids:
                if tool_id.service_name not in services:
                    services[tool_id.service_name] = await self._service_registry.read_tool_service(
                        tool_id.service_name
                    )

                tool = await services[tool_id.service_name].read_tool(tool_id.tool_name)

                batches[(tool_id, tool)].append(guideline_proposition)

        t_start = time.time()

        with self._logger.operation(f"Evaluation: {len(batches)} tools"):
            batch_tasks = [
                self._infer_calls_for_single_tool(
                    agent=agent,
                    context_variables=context_variables,
                    interaction_history=interaction_history,
                    terms=terms,
                    ordinary_guideline_propositions=ordinary_guideline_propositions,
                    candidate_descriptor=(tool_id, tool, props),
                    reference_tools=[
                        tool_descriptor
                        for tool_descriptor in batches
                        if tool_descriptor != (tool_id, tool)
                    ],
                    staged_events=staged_events,
                )
                for (tool_id, tool), props in batches.items()
            ]

            batch_results = list(await async_utils.safe_gather(*batch_tasks))
            batch_generations = [generation for generation, _, _ in batch_results]
            tool_call_batches = [tool_calls for _, tool_calls, _ in batch_results]

        t_end = time.time()

        total_missing_data: list[MissingToolData] = []

        for _, _, missing_data_for_single_tool in batch_results:
            for missing_data_for_single_call in missing_data_for_single_tool:
                total_missing_data.append(missing_data_for_single_call)

        return InferenceToolCallsResult(
            total_duration=t_end - t_start,
            batch_count=len(batches),
            batch_generations=batch_generations,
            batches=tool_call_batches,
            insights=ToolInsights(missing_data=total_missing_data),
        )

    async def _infer_calls_for_single_tool(
        self,
        agent: Agent,
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        ordinary_guideline_propositions: Sequence[GuidelineProposition],
        candidate_descriptor: tuple[ToolId, Tool, list[GuidelineProposition]],
        reference_tools: Sequence[tuple[ToolId, Tool]],
        staged_events: Sequence[EmittedEvent],
    ) -> tuple[GenerationInfo, list[ToolCall], list[MissingToolData]]:
        inference_prompt = self._build_tool_call_inference_prompt(
            agent,
            context_variables,
            interaction_history,
            terms,
            ordinary_guideline_propositions,
            candidate_descriptor,
            reference_tools,
            staged_events,
            await self.shots(),
        )

        tool_id, tool, _ = candidate_descriptor

        with self._logger.operation(f"Evaluation: {tool_id}"):
            generation_info, inference_output = await self._run_inference(inference_prompt)

        tool_calls = []
        missing_data = []

        for tc in inference_output:
            if (
                tc.applicability_score >= 6
                and not tc.same_call_is_already_staged
                and (
                    not tc.a_rejected_tool_would_have_been_a_better_fit_if_it_werent_already_rejected
                    or tc.the_better_rejected_tool_should_clearly_be_run_in_tandem_with_the_candidate_tool
                )
            ):
                if tc.should_run and all(
                    not evaluation.is_missing
                    for evaluation in tc.argument_evaluations or []
                    if evaluation.parameter_name in candidate_descriptor[1].required
                ):
                    self._logger.debug(
                        f"Inference::Completion::Activated:\n{tc.model_dump_json(indent=2)}"
                    )

                    tool_calls.append(
                        ToolCall(
                            id=ToolCallId(generate_id()),
                            tool_id=tool_id,
                            arguments={
                                evaluation.parameter_name: evaluation.value_as_string
                                for evaluation in tc.argument_evaluations or []
                            },
                        )
                    )
                elif tc.applicability_score >= 8:
                    for evaluation in tc.argument_evaluations or []:
                        if evaluation.parameter_name not in tool.parameters:
                            self._logger.error(
                                f"Inference::Completion: Argument {evaluation.parameter_name} not found in tool parameters"
                            )
                            continue

                        _, tool_options = tool.parameters[evaluation.parameter_name]

                        if (
                            evaluation.is_missing
                            and not evaluation.is_optional
                            and not tool_options.hidden
                        ):
                            missing_data.append(
                                MissingToolData(
                                    parameter=evaluation.parameter_name,
                                    significance=tool_options.significance,
                                )
                            )

            else:
                self._logger.debug(
                    f"Inference::Completion::Skipped:\n{tc.model_dump_json(indent=2)}"
                )

        return generation_info, tool_calls, missing_data

    async def execute_tool_calls(
        self,
        context: ToolContext,
        tool_calls: Sequence[ToolCall],
    ) -> Sequence[ToolCallResult]:
        with self._logger.scope("ToolCaller"):
            with self._logger.operation("Execution"):
                tool_results = await async_utils.safe_gather(
                    *(
                        self._run_tool(
                            context=context,
                            tool_call=tool_call,
                            tool_id=tool_call.tool_id,
                        )
                        for tool_call in tool_calls
                    )
                )

                return tool_results

    def _get_glossary_text(
        self,
        terms: Sequence[Term],
    ) -> str:
        terms_string = "\n".join(f"{i}) {repr(t)}" for i, t in enumerate(terms, start=1))

        return f"""
The following is a glossary of the business.
In some cases, a glossary term directly overrides "common knowledge" or the most prevalent definition of that same term (or object).
Therefore, when encountering any of these terms, prioritize the interpretation provided in the glossary over any definitions you may already know.
Please be tolerant of possible typos by the user with regards to these terms,and let the user know if/when you assume they meant a term by their typo: ###
{terms_string}
###
"""  # noqa

    async def shots(self) -> Sequence[ToolCallerInferenceShot]:
        return await shot_collection.list()

    def _format_shots(
        self,
        shots: Sequence[ToolCallerInferenceShot],
    ) -> str:
        return "\n".join(
            f"""
Example #{i}: ###
{self._format_shot(shot)}
###
"""
            for i, shot in enumerate(shots, start=1)
        )

    def _format_shot(
        self,
        shot: ToolCallerInferenceShot,
    ) -> str:
        return f"""
- **Context**:
{shot.description}

- **Expected Result**:
```json
{json.dumps(shot.expected_result.model_dump(mode="json", exclude_unset=True), indent=2)}
```"""

    def _build_tool_call_inference_prompt(
        self,
        agent: Agent,
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_event_list: Sequence[Event],
        terms: Sequence[Term],
        ordinary_guideline_propositions: Sequence[GuidelineProposition],
        batch: tuple[ToolId, Tool, list[GuidelineProposition]],
        reference_tools: Sequence[tuple[ToolId, Tool]],
        staged_events: Sequence[EmittedEvent],
        shots: Sequence[ToolCallerInferenceShot],
    ) -> PromptBuilder:
        staged_calls = self._get_staged_calls(staged_events)

        builder = PromptBuilder(on_build=lambda prompt: self._logger.debug(f"Prompt:\n{prompt}"))

        builder.add_section(
            name="tool-caller-general-instructions",
            template="""
GENERAL INSTRUCTIONS
-----------------
You are part of a system of AI agents which interact with a customer on the behalf of a business.
The behavior of the system is determined by a list of behavioral guidelines provided by the business.
Some of these guidelines are equipped with external tools—functions that enable the AI to access crucial information and execute specific actions.
Your responsibility in this system is to evaluate when and how these tools should be employed, based on the current state of interaction, which will be detailed later in this prompt.

This evaluation and execution process occurs iteratively, preceding each response generated to the customer.
Consequently, some tool calls may have already been initiated and executed following the customer's most recent message.
Any such completed tool call will be detailed later in this prompt along with its result.
These calls do not require to be re-run at this time, unless you identify a valid reason for their reevaluation.

""",
            props={},
        )
        builder.add_agent_identity(agent)
        builder.add_section(
            name="tool-caller-task-description",
            template="""
-----------------
TASK DESCRIPTION
-----------------
Your task is to review the provided tool and, based on your most recent interaction with the customer, decide whether to use it.
For the provided tool, assign a score from 1 to 10 to indicate its usefulness at this time, where a higher score indicates that the tool call should execute.
For any tool with a score of 5 or higher, provide the arguments for activation, following the format in its description.

While doing so, take the following instructions into account:

1. You may suggest tool that don’t directly address the customer’s latest interaction but can advance the conversation to a more useful state based on function definitions.
2. Each tool may be called multiple times with different arguments.
3. Avoid calling a tool with the same arguments more than once, unless clearly justified by the interaction.
4. Ensure each tool call relies only on the immediate context and staged calls, without requiring other tools not yet invoked, to avoid dependencies.
5. Use the "should_run" argument to indicate whether a tool should be executed, meaning it has a high applicability score and either (a) has not been staged with the same arguments, or (b) was staged but needs to be re-executed.
6. If a tool needs to be applied multiple times (each with different arguments), you may include it in the output multiple times.

Produce a valid JSON object according to the following format:
```json
{{
    "last_customer_message": "<REPEAT THE LAST USER MESSAGE IN THE INTERACTION>",
    "most_recent_customer_inquiry_or_need": "<customer's inquiry or need>",
    "most_recent_customer_inquiry_or_need_was_already_resolved": <BOOL>,
    "name": "<TOOL NAME>",
    "subtleties_to_be_aware_of": "<NOTE ANY SIGNIFICANT SUBTLETIES TO BE AWARE OF WHEN RUNNING THIS TOOL IN OUR AGENT'S CONTEXT>",
    "tool_calls_for_candidate_tool": [
        {{
            "applicability_rationale": "<A FEW WORDS THAT EXPLAIN WHETHER AND HOW THE TOOL NEEDS TO BE CALLED>",
            "applicability_score": <INTEGER FROM 1 TO 10>,
            "argument_evaluations": [<EVALUATIONS FOR THE ARGUMENTS. CAN BE DROPPED ONLY IF THE TOOL APPLICABILITY IS UNDER 6>],
            "same_call_is_already_staged": <BOOL>,
            "comparison_with_rejected_tools_including_references_to_subtleties": "<A VERY BRIEF OVERVIEW OF HOW THIS CALL FARES AGAINST OTHER TOOLS IN APPLICABILITY>",
            "relevant_subtleties": "<IF SUBTLETIES FOUND, REFER TO THE RELEVANT ONES HERE>",
            "a_rejected_tool_would_have_been_a_better_fit_if_it_werent_already_rejected": <BOOL>,
            "potentially_better_rejected_tool_name": "<IF CANDIDATE TOOL IS A WORSE FIT THAN A REJECTED TOOL, THIS IS THE NAME OF THAT REJECTED TOOL>",
            "potentially_better_rejected_tool_rationale": "<IF CANDIDATE TOOL IS A WORSE FIT THAN A REJECTED TOOL, THIS EXPLAINS WHY>",
            "the_better_rejected_tool_should_clearly_be_run_in_tandem_with_the_candidate_tool": <BOOL>,
            "are_optional_arguments_missing": <BOOL>,
            "are_non_optional_arguments_missing": <BOOL>,
            "allowed_to_run_without_optional_arguments_even_if_they_are_missing": <BOOL-ALWAYS TRUE>,
            "should_run": <BOOL-WHETHER THE TOOL IS APPLICABLE, NOT YET STAGED, AND ALL REQUIRED PARAMS ARE PROVIDED>
        }}
        ...
    ]
}}
```

where the tool provided to you under appears at least once in "tool_calls_for_candidate_tool", whether you decide to use it or not.
The exact format of your output will be provided to you at the end of this prompt.

The following examples show correct outputs for various hypothetical situations.
Only the responses are provided, without the interaction history or tool descriptions, though these can be inferred from the responses.

""",
            props={},
        )
        builder.add_section(
            name="tool-caller-examples",
            template="""
EXAMPLES
-----------------
{formatted_shots}
""",
            props={"formatted_shots": self._format_shots(shots), "shots": shots},
        )
        builder.add_context_variables(context_variables)
        if terms:
            builder.add_section(
                name=BuiltInSection.GLOSSARY,
                template=self._get_glossary_text(terms),
                props={"terms": terms},
                status=SectionStatus.ACTIVE,
            )
        builder.add_interaction_history(interaction_event_list)

        builder.add_section(
            name=BuiltInSection.GUIDELINE_DESCRIPTIONS,
            template=self._add_guideline_propositions_section(
                ordinary_guideline_propositions,
                (batch[0], batch[2]),
            ),
            props={
                "ordinary_guideline_propositions": ordinary_guideline_propositions,
                "tool_id_propositions": (batch[0], batch[2]),
            },
        )
        tool_definitions_template, tool_definitions_props = self._add_tool_definitions_section(
            candidate_tool=(batch[0], batch[1]),
            reference_tools=reference_tools,
        )
        builder.add_section(
            name="tool-caller-tool-definitions",
            template=tool_definitions_template,
            props={
                **tool_definitions_props,
                "candidate_tool": (batch[0], batch[1]),
                "reference_tools": reference_tools,
            },
        )
        if staged_calls:
            builder.add_section(
                name="tool-caller-staged-tool-calls",
                template="""
STAGED TOOL CALLS
-----------------
The following is a list of tool calls staged after the interaction’s latest state. Use this information to avoid redundant calls and to guide your response.

Reminder: If a tool is already staged with the exact same arguments, set "same_call_is_already_staged" to true.
You may still choose to re-run the tool call, but only if there is a specific reason for it to be executed multiple times.

The staged tool calls are:
{staged_calls}
###
""",
                props={"staged_calls": staged_calls},
            )
        else:
            builder.add_section(
                name="tool-caller-empty-staged-tool-calls",
                template="""
STAGED TOOL CALLS
-----------------
There are no staged tool calls at this time.
""",
                props={},
            )

        builder.add_section(
            name="tool-caller-output-format",
            template="""
OUTPUT FORMAT
-----------------
Given the tool, your output should adhere to the following format:
```json
{{
    "last_customer_message": "<REPEAT THE LAST USER MESSAGE IN THE INTERACTION>",
    "most_recent_customer_inquiry_or_need": "<customer's inquiry or need>",
    "most_recent_customer_inquiry_or_need_was_already_resolved": <BOOL>,
    "name": "{service_name}:{tool_name}",
    "subtleties_to_be_aware_of": "<NOTE ANY SIGNIFICANT SUBTLETIES TO BE AWARE OF WHEN RUNNING THIS TOOL IN OUR AGENT'S CONTEXT>",
    "tool_calls_for_candidate_tool": [
        {{
            "applicability_rationale": "<A FEW WORDS THAT EXPLAIN WHETHER, HOW, AND TO WHAT EXTENT THE TOOL NEEDS TO BE CALLED AT THIS POINT>",
            "applicability_score": <INTEGER FROM 1 TO 10>,
            "argument_evaluations": [<EVALUATIONS FOR THE ARGUMENTS. CAN BE DROPPED ONLY IF THE TOOL APPLICABILITY IS UNDER 6>],
            "same_call_is_already_staged": <BOOL>,
            "comparison_with_rejected_tools_including_references_to_subtleties": "<A VERY BRIEF OVERVIEW OF HOW THIS CALL FARES AGAINST OTHER TOOLS IN APPLICABILITY>",
            "relevant_subtleties": "<IF SUBTLETIES FOUND, REFER TO THE RELEVANT ONES HERE>",
            "a_rejected_tool_would_have_been_a_better_fit_if_it_werent_already_rejected": <BOOL>,
            "potentially_better_rejected_tool_name": "<IF CANDIDATE TOOL IS A WORSE FIT THAN A REJECTED TOOL, THIS IS THE NAME OF THAT REJECTED TOOL>",
            "potentially_better_rejected_tool_rationale": "<IF CANDIDATE TOOL IS A WORSE FIT THAN A REJECTED TOOL, THIS EXPLAINS WHY>",
            "the_better_rejected_tool_should_clearly_be_run_in_tandem_with_the_candidate_tool": <BOOL>,
            "are_optional_arguments_missing": <BOOL>,
            "are_non_optional_arguments_missing": <BOOL>,
            "allowed_to_run_without_optional_arguments_even_if_they_are_missing": <BOOL-ALWAYS TRUE>,
            "should_run": <BOOL-WHETHER THE TOOL IS APPLICABLE, NOT YET STAGED, AND ALL REQUIRED PARAMS ARE PROVIDED>
        }}
    ]
}}
```

However, note that you may choose to have multiple entries in 'tool_calls_for_candidate_tool' if you wish to call the candidate tool multiple times with different arguments.
""",
            props={
                "service_name": batch[0].service_name,
                "tool_name": batch[0].tool_name,
            },
        )

        return builder

    def _add_tool_definitions_section(
        self,
        candidate_tool: tuple[ToolId, Tool],
        reference_tools: Sequence[tuple[ToolId, Tool]],
    ) -> tuple[str, dict[str, Any]]:
        def _get_param_spec(spec: tuple[ToolParameterDescriptor, ToolParameterOptions]) -> str:
            descriptor, options = spec

            result: dict[str, Any] = {"schema": {"type": descriptor["type"]}}

            if descriptor["type"] == "array":
                result["schema"]["items"] = {"type": descriptor["item_type"]}

                if enum := descriptor.get("enum"):
                    result["schema"]["items"]["enum"] = enum
            else:
                if enum := descriptor.get("enum"):
                    result["schema"]["enum"] = enum

            if options.description:
                result["description"] = options.description
            elif description := descriptor.get("description"):
                result["description"] = description

            if examples := descriptor.get("examples"):
                result["extraction_examples__only_for_reference"] = examples

            match options.source:
                case "any":
                    result["acceptable_source"] = (
                        "This argument can be extracted in the best way you think"
                    )
                case "context":
                    result["acceptable_source"] = (
                        "This argument can be extracted only from the context given in this prompt"
                    )
                case "customer":
                    result["acceptable_source"] = (
                        "This argument must be provided by the customer, and NEVER automatically guessed by you"
                    )

            return json.dumps(result)

        def _get_tool_spec(t_id: ToolId, t: Tool) -> dict[str, Any]:
            return {
                "tool_name": t_id.to_string(),
                "description": t.description,
                "optional_arguments": {
                    name: _get_param_spec(spec)
                    for name, spec in t.parameters.items()
                    if name not in t.required
                },
                "required_parameters": {
                    name: _get_param_spec(spec)
                    for name, spec in t.parameters.items()
                    if name in t.required
                },
            }

        candidate_tool_spec = _get_tool_spec(candidate_tool[0], candidate_tool[1])
        if not reference_tools:
            return (
                """
The following is the tool function definition.
IMPORTANT: You must not return results for any tool other than this one, even if you believe they might be relevant:
###
{candidate_tool_spec}
###
""",
                {"candidate_tool_spec": candidate_tool_spec},
            )

        else:
            reference_tool_specs = [
                _get_tool_spec(tool_id, tool) for tool_id, tool in reference_tools
            ]
            return (
                """
You are provided with multiple tools, categorized as follows:
- Candidate Tool: The tool under your evaluation.
- Rejected Tools: A list of additional tools that have been considered already and deemed irrelevant for an unspecified reason

Your task is to evaluate the necessity and usage of the Candidate Tool ONLY.
- Use the Rejected Tools as a contextual benchmark to decide whether the Candidate Tool should be run.
The rejected tools may have been rejected for any reason whatsoever, which you are not privy to.
If the Candidate Tool seems even less relevant than any of the Rejected Tools, then it should not be run at all.
DO NOT RUN the Candidate Tool as a "FALLBACK", "LAST RESORT", or "LAST VIABLE CHOICE" if another tool that actually seems more appropriate was nonetheless rejected for some reason.
Remember that other tools were rejected while taking your (agent's) description and glossary into full consideration. Nothing was overlooked.
However, if the Candidate Tool truly offers a unique advantage or capability over all other Rejected Tools,
given the agent's description and glossary, then do choose to use it and provide its arguments.
Finally, focus solely on evaluating the Candidate Tool; do not evaluate any other tool.

Rejected tools: ###
{reference_tool_specs}
###

Candidate tool: ###
{candidate_tool_spec}
###
""",
                {
                    "candidate_tool_spec": candidate_tool_spec,
                    "reference_tool_specs": reference_tool_specs,
                },
            )

    def _add_guideline_propositions_section(
        self,
        ordinary_guideline_propositions: Sequence[GuidelineProposition],
        tool_id_propositions: tuple[ToolId, list[GuidelineProposition]],
    ) -> str:
        all_propositions = list(chain(ordinary_guideline_propositions, tool_id_propositions[1]))

        if all_propositions:
            guidelines = []

            for i, p in enumerate(all_propositions, start=1):
                guideline = (
                    f"{i}) When {p.guideline.content.condition}, then {p.guideline.content.action}"
                )
                guidelines.append(guideline)

            guideline_list = "\n".join(guidelines)
        return f"""
GUIDELINES
---------------------
The following guidelines have been identified as relevant to the current state of interaction with the customer.
Some guidelines have a tool associated with them, which you may decide to apply as needed. Use these guidelines to understand the context for the provided tool.

Guidelines:
###
{guideline_list}
\n    Associated Tool: {tool_id_propositions[0].service_name}:{tool_id_propositions[0].tool_name}"
###
"""

    def _get_staged_calls(
        self,
        emitted_events: Sequence[EmittedEvent],
    ) -> Optional[str]:
        staged_calls = [PromptBuilder.adapt_event(e) for e in emitted_events if e.kind == "tool"]

        if not staged_calls:
            return None

        return json.dumps(staged_calls)

    async def _run_inference(
        self,
        prompt: PromptBuilder,
    ) -> tuple[GenerationInfo, Sequence[ToolCallEvaluation]]:
        inference = await self._schematic_generator.generate(
            prompt=prompt,
            hints={"temperature": 0.05},
        )

        self._logger.debug(f"Inference::Completion:\n{inference.content.model_dump_json(indent=2)}")

        return inference.info, inference.content.tool_calls_for_candidate_tool

    async def _run_tool(
        self,
        context: ToolContext,
        tool_call: ToolCall,
        tool_id: ToolId,
    ) -> ToolCallResult:
        try:
            self._logger.debug(
                f"Execution::Invocation: ({tool_call.tool_id.to_string()}/{tool_call.id})"
                + (f"\n{json.dumps(tool_call.arguments, indent=2)}" if tool_call.arguments else "")
            )

            try:
                service = await self._service_registry.read_tool_service(tool_id.service_name)

                result = await service.call_tool(
                    tool_id.tool_name,
                    context,
                    tool_call.arguments,
                )

                self._logger.debug(
                    f"Execution::Result: Tool call succeeded ({tool_call.tool_id.to_string()}/{tool_call.id})\n{json.dumps(asdict(result), indent=2)}"
                )
            except Exception as exc:
                self._logger.error(
                    f"Execution::Result: Tool call failed ({tool_id.to_string()}/{tool_call.id})\n{traceback.format_exception(exc)}"
                )
                raise

            return ToolCallResult(
                id=ToolResultId(generate_id()),
                tool_call=tool_call,
                result={
                    "data": result.data,
                    "metadata": result.metadata,
                    "control": result.control,
                },
            )
        except Exception as e:
            self._logger.error(
                f"Execution::Error: ToolId: {tool_call.tool_id.to_string()}', "
                f"Arguments:\n{json.dumps(tool_call.arguments, indent=2)}"
                + "\nTraceback:\n"
                + "\n".join(traceback.format_exception(e)),
            )

            return ToolCallResult(
                id=ToolResultId(generate_id()),
                tool_call=tool_call,
                result={
                    "data": "Tool call error",
                    "metadata": {"error_details": str(e)},
                    "control": {},
                },
            )


_baseline_shots: Sequence[ToolCallerInferenceShot] = [
    ToolCallerInferenceShot(
        description="the id of the customer is 12345, and check_balance(12345) is already listed as a staged tool call",
        expected_result=ToolCallInferenceSchema(
            last_customer_message="Do I have enough money in my account to get a taxi from New York to Newark?",
            most_recent_customer_inquiry_or_need=(
                "Checking customer's balance, comparing it to the price of a taxi from New York to Newark, "
                "and report the result to the customer"
            ),
            most_recent_customer_inquiry_or_need_was_already_resolved=False,
            name="check_balance",
            subtleties_to_be_aware_of="check_balance(12345) is already staged",
            tool_calls_for_candidate_tool=[
                ToolCallEvaluation(
                    applicability_rationale="We need the client's current balance to respond to their question",
                    applicability_score=9,
                    argument_evaluations=[
                        ArgumentEvaluation(
                            parameter_name="customer_id",
                            acceptable_source_for_this_argument_according_to_its_tool_definition="<INFER THIS BASED ON TOOL DEFINITION>",
                            evaluate_is_it_provided_by_an_acceptable_source="The customer ID is given by a context variable",
                            evaluate_was_it_already_provided_and_should_it_be_provided_again="No need to provide it again as the customer's ID is unique and doesn't change",
                            evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided="It would be extremely problematic, but I don't need to guess here since I have it",
                            is_missing=False,
                            is_optional=False,
                            value_as_string="12345",
                        )
                    ],
                    same_call_is_already_staged=True,
                    comparison_with_rejected_tools_including_references_to_subtleties=(
                        "There are no tools in the list of rejected tools"
                    ),
                    relevant_subtleties="check_balance(12345) is already staged",
                    a_rejected_tool_would_have_been_a_better_fit_if_it_werent_already_rejected=False,
                    are_optional_arguments_missing=False,
                    are_non_optional_arguments_missing=False,
                    allowed_to_run_without_optional_arguments_even_if_they_are_missing=True,
                    should_run=False,
                )
            ],
        ),
    ),
    ToolCallerInferenceShot(
        description="the id of the customer is 12345, and check_balance(12345) is listed as the only staged tool call",
        expected_result=ToolCallInferenceSchema(
            last_customer_message="Do I have enough money in my account to get a taxi from New York to Newark?",
            most_recent_customer_inquiry_or_need=(
                "Checking customer's balance, comparing it to the price of a taxi from New York to Newark, "
                "and report the result to the customer"
            ),
            most_recent_customer_inquiry_or_need_was_already_resolved=False,
            name="ping_supervisor",
            subtleties_to_be_aware_of="no subtleties were detected",
            tool_calls_for_candidate_tool=[
                ToolCallEvaluation(
                    applicability_rationale="There is no reason to notify the supervisor of anything",
                    applicability_score=1,
                    same_call_is_already_staged=False,
                    comparison_with_rejected_tools_including_references_to_subtleties="There are no tools in the list of rejected tools",
                    relevant_subtleties="no subtleties were detected",
                    a_rejected_tool_would_have_been_a_better_fit_if_it_werent_already_rejected=False,
                    are_optional_arguments_missing=False,
                    are_non_optional_arguments_missing=False,
                    allowed_to_run_without_optional_arguments_even_if_they_are_missing=True,
                    should_run=False,
                )
            ],
        ),
    ),
    ToolCallerInferenceShot(
        description=(
            "the id of the customer is 12345, and check_balance(12345) is the only staged tool call; "
            "some irrelevant reference tools exist"
        ),
        expected_result=ToolCallInferenceSchema(
            last_customer_message="Do I have enough money in my account to get a taxi from New York to Newark?",
            most_recent_customer_inquiry_or_need=(
                "Checking customer's balance, comparing it to the price of a taxi from New York to Newark, "
                "and report the result to the customer"
            ),
            most_recent_customer_inquiry_or_need_was_already_resolved=False,
            name="check_ride_price",
            subtleties_to_be_aware_of="no subtleties were detected",
            tool_calls_for_candidate_tool=[
                ToolCallEvaluation(
                    applicability_rationale="We need to know the price of a ride from New York to Newark to respond to the customer",
                    applicability_score=9,
                    argument_evaluations=[
                        ArgumentEvaluation(
                            parameter_name="origin",
                            acceptable_source_for_this_argument_according_to_its_tool_definition="<INFER THIS BASED ON TOOL DEFINITION>",
                            evaluate_is_it_provided_by_an_acceptable_source="Yes, the customer mentioned New York as the origin for their ride",
                            evaluate_was_it_already_provided_and_should_it_be_provided_again="The customer already specifically provided it",
                            evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided="It would be extremely problematic, but I don't need to guess here since the customer provided it",
                            is_missing=False,
                            is_optional=False,
                            value_as_string="New York",
                        ),
                        ArgumentEvaluation(
                            parameter_name="destination",
                            acceptable_source_for_this_argument_according_to_its_tool_definition="<INFER THIS BASED ON TOOL DEFINITION>",
                            evaluate_is_it_provided_by_an_acceptable_source="Yes, the customer mentioned Newark as the destination for their ride",
                            evaluate_was_it_already_provided_and_should_it_be_provided_again="The customer already specifically provided it",
                            evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided="It would be extremely problematic, but I don't need to guess here since the customer provided it",
                            is_missing=False,
                            is_optional=False,
                            value_as_string="Newark",
                        ),
                    ],
                    same_call_is_already_staged=False,
                    comparison_with_rejected_tools_including_references_to_subtleties=(
                        "None of the available reference tools are deemed more suitable for the candidate tool’s application"
                    ),
                    relevant_subtleties="no subtleties were detected",
                    a_rejected_tool_would_have_been_a_better_fit_if_it_werent_already_rejected=False,
                    are_optional_arguments_missing=False,
                    are_non_optional_arguments_missing=False,
                    allowed_to_run_without_optional_arguments_even_if_they_are_missing=True,
                    should_run=True,
                )
            ],
        ),
    ),
    ToolCallerInferenceShot(
        description=(
            "the candidate tool is check_calories(<product_name>): returns the number of calories in a product; "
            "one reference tool is check_stock()"
        ),
        expected_result=ToolCallInferenceSchema(
            last_customer_message="Which pizza has more calories, the classic margherita or the deep dish?",
            most_recent_customer_inquiry_or_need=(
                "Checking the number of calories in two types of pizza and replying with which one has more"
            ),
            most_recent_customer_inquiry_or_need_was_already_resolved=False,
            name="check_calories",
            subtleties_to_be_aware_of="two products need to be checked for calories - margherita and deep dish",
            tool_calls_for_candidate_tool=[
                ToolCallEvaluation(
                    applicability_rationale="We need to check how many calories are in the margherita pizza",
                    applicability_score=9,
                    argument_evaluations=[
                        ArgumentEvaluation(
                            parameter_name="product_name",
                            acceptable_source_for_this_argument_according_to_its_tool_definition="<INFER THIS BASED ON TOOL DEFINITION>",
                            evaluate_is_it_provided_by_an_acceptable_source="The first product the customer specified is a margherita",
                            evaluate_was_it_already_provided_and_should_it_be_provided_again="The customer already specifically provided it",
                            evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided="It would be absurd to provide unsolicited information on some random product, but I don't need to guess here since the customer provided it",
                            is_missing=False,
                            is_optional=False,
                            value_as_string="Margherita",
                        ),
                    ],
                    same_call_is_already_staged=False,
                    comparison_with_rejected_tools_including_references_to_subtleties=(
                        "None of the available reference tools are deemed more suitable for the candidate tool’s application"
                    ),
                    relevant_subtleties="two products need to be checked for calories - begin with margherita",
                    a_rejected_tool_would_have_been_a_better_fit_if_it_werent_already_rejected=False,
                    are_optional_arguments_missing=False,
                    are_non_optional_arguments_missing=False,
                    allowed_to_run_without_optional_arguments_even_if_they_are_missing=True,
                    should_run=True,
                ),
                ToolCallEvaluation(
                    applicability_rationale="We need to check how many calories are in the deep dish pizza",
                    applicability_score=9,
                    argument_evaluations=[
                        ArgumentEvaluation(
                            parameter_name="product_name",
                            acceptable_source_for_this_argument_according_to_its_tool_definition="<INFER THIS BASED ON TOOL DEFINITION>",
                            evaluate_is_it_provided_by_an_acceptable_source="The second product the customer specified is the deep dish",
                            evaluate_was_it_already_provided_and_should_it_be_provided_again="The customer already specifically provided it",
                            evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided="It would be absurd to provide unsolicited information on some random product, but I don't need to guess here since the customer provided it",
                            is_missing=False,
                            is_optional=False,
                            value_as_string="Deep Dish",
                        ),
                    ],
                    same_call_is_already_staged=False,
                    comparison_with_rejected_tools_including_references_to_subtleties=(
                        "None of the available reference tools are deemed more suitable for the candidate tool’s application"
                    ),
                    relevant_subtleties="two products need to be checked for calories - now check deep dish",
                    a_rejected_tool_would_have_been_a_better_fit_if_it_werent_already_rejected=False,
                    are_optional_arguments_missing=False,
                    are_non_optional_arguments_missing=False,
                    allowed_to_run_without_optional_arguments_even_if_they_are_missing=True,
                    should_run=True,
                ),
            ],
        ),
    ),
    ToolCallerInferenceShot(
        description=(
            "the candidate tool is check_vehicle_price(model: str), and reference tool is check_motorcycle_price(model: str)"
        ),
        expected_result=ToolCallInferenceSchema(
            last_customer_message="What's your price for a Harley-Davidson Street Glide?",
            most_recent_customer_inquiry_or_need="Checking the price of a Harley-Davidson Street Glide motorcycle",
            most_recent_customer_inquiry_or_need_was_already_resolved=False,
            name="check_motorcycle_price",
            subtleties_to_be_aware_of="Both the candidate and referenc tool could apply - we need to choose the one that applies best",
            tool_calls_for_candidate_tool=[
                ToolCallEvaluation(
                    applicability_rationale="we need to check for the price of a specific motorcycle model",
                    applicability_score=9,
                    argument_evaluations=[
                        ArgumentEvaluation(
                            parameter_name="model",
                            acceptable_source_for_this_argument_according_to_its_tool_definition="<INFER THIS BASED ON TOOL DEFINITION>",
                            evaluate_is_it_provided_by_an_acceptable_source="Yes; the customer asked about a specific model",
                            evaluate_was_it_already_provided_and_should_it_be_provided_again="The customer asked about a specific model",
                            evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided="It would be absurd to provide unsolicited information on some random model, but I don't need to guess here since the customer provided it",
                            is_missing=False,
                            is_optional=False,
                            value_as_string="Harley-Davidson Street Glide",
                        )
                    ],
                    same_call_is_already_staged=False,
                    comparison_with_rejected_tools_including_references_to_subtleties=(
                        "candidate tool is more specialized for this use case than the rejected tools"
                    ),
                    relevant_subtleties="Both the candidate and referenc tool could apply - we need to choose the one that applies best",
                    a_rejected_tool_would_have_been_a_better_fit_if_it_werent_already_rejected=False,
                    potentially_better_rejected_tool_name="check_motorcycle_price",
                    potentially_better_rejected_tool_rationale=(
                        "the only reference tool is less relevant than the candidate tool, "
                        "since the candidate tool is designed specifically for motorcycle models, "
                        "and not just general vehicles."
                    ),
                    the_better_rejected_tool_should_clearly_be_run_in_tandem_with_the_candidate_tool=False,
                    are_optional_arguments_missing=False,
                    are_non_optional_arguments_missing=False,
                    allowed_to_run_without_optional_arguments_even_if_they_are_missing=True,
                    should_run=True,
                )
            ],
        ),
    ),
    ToolCallerInferenceShot(
        description=(
            "the candidate tool is check_motorcycle_price(model: str), and one reference tool is check_vehicle_price(model: str)"
        ),
        expected_result=ToolCallInferenceSchema(
            last_customer_message="What's your price for a Harley-Davidson Street Glide?",
            most_recent_customer_inquiry_or_need="Checking the price of a Harley-Davidson Street Glide motorcycle",
            most_recent_customer_inquiry_or_need_was_already_resolved=False,
            name="check_vehicle_price",
            subtleties_to_be_aware_of="no subtleties were detected",
            tool_calls_for_candidate_tool=[
                ToolCallEvaluation(
                    applicability_rationale="we need to check for the price of a specific vehicle - a Harley-Davidson Street Glide",
                    applicability_score=8,
                    argument_evaluations=[
                        ArgumentEvaluation(
                            parameter_name="model",
                            acceptable_source_for_this_argument_according_to_its_tool_definition="<INFER THIS BASED ON TOOL DEFINITION>",
                            evaluate_is_it_provided_by_an_acceptable_source="Yes; the customer asked about a specific model",
                            evaluate_was_it_already_provided_and_should_it_be_provided_again="The customer asked about a specific model",
                            evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided="It would be absurd to provide unsolicited information on some random model, but I don't need to guess here since the customer provided it",
                            is_missing=False,
                            is_optional=False,
                            value_as_string="Harley-Davidson Street Glide",
                        )
                    ],
                    same_call_is_already_staged=False,
                    comparison_with_rejected_tools_including_references_to_subtleties="not as good a fit as check_motorcycle_price",
                    relevant_subtleties="no subtleties were detected",
                    a_rejected_tool_would_have_been_a_better_fit_if_it_werent_already_rejected=True,
                    potentially_better_rejected_tool_name="check_motorcycle_price",
                    potentially_better_rejected_tool_rationale=(
                        "check_motorcycle_price applies specifically for motorcycles, "
                        "which is better fitting for this case compared to the more general check_vehicle_price"
                    ),
                    the_better_rejected_tool_should_clearly_be_run_in_tandem_with_the_candidate_tool=False,
                    are_optional_arguments_missing=False,
                    are_non_optional_arguments_missing=False,
                    allowed_to_run_without_optional_arguments_even_if_they_are_missing=True,
                    should_run=False,
                )
            ],
        ),
    ),
    ToolCallerInferenceShot(
        description=(
            "the candidate tool is check_temperature(location: str), and reference tool is check_indoor_temperature(room: str)"
        ),
        expected_result=ToolCallInferenceSchema(
            last_customer_message="What's the temperature in the living room right now?",
            most_recent_customer_inquiry_or_need="Checking the current temperature in the living room",
            most_recent_customer_inquiry_or_need_was_already_resolved=False,
            name="check_temperature",
            subtleties_to_be_aware_of="no subtleties were detected",
            tool_calls_for_candidate_tool=[
                ToolCallEvaluation(
                    applicability_rationale="need to check the current temperature in the living room",
                    applicability_score=8,
                    argument_evaluations=[
                        ArgumentEvaluation(
                            parameter_name="location",
                            acceptable_source_for_this_argument_according_to_its_tool_definition="<INFER THIS BASED ON TOOL DEFINITION>",
                            evaluate_is_it_provided_by_an_acceptable_source="Yes; the customer asked about the living room",
                            evaluate_was_it_already_provided_and_should_it_be_provided_again="The customer asked about a specific location",
                            evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided="It would be absurd to provide unsolicited information on some random room, but I don't need to guess here since the customer provided it",
                            is_missing=False,
                            is_optional=False,
                            value_as_string="living room",
                        )
                    ],
                    same_call_is_already_staged=False,
                    comparison_with_rejected_tools_including_references_to_subtleties="check_indoor_temperature is a better fit for this usecase, as it's more specific",
                    relevant_subtleties="no subtleties were detected",
                    a_rejected_tool_would_have_been_a_better_fit_if_it_werent_already_rejected=True,
                    potentially_better_rejected_tool_name="check_indoor_temperature",
                    potentially_better_rejected_tool_rationale=(
                        "check_temperature is a more general case of check_indoor_temperature. "
                        "Here, since the customer inquired about the temperature of a specific room, the check_indoor_temperature is more fitting."
                    ),
                    the_better_rejected_tool_should_clearly_be_run_in_tandem_with_the_candidate_tool=False,
                    are_optional_arguments_missing=False,
                    are_non_optional_arguments_missing=False,
                    allowed_to_run_without_optional_arguments_even_if_they_are_missing=True,
                    should_run=False,
                )
            ],
        ),
    ),
    ToolCallerInferenceShot(
        description=(
            "the candidate tool is search_product(query: str), and reference tool is "
            "search_electronics(query: str, specifications: dict)"
        ),
        expected_result=ToolCallInferenceSchema(
            last_customer_message="I'm looking for a gaming laptop with at least 16GB RAM and an RTX 3080",
            most_recent_customer_inquiry_or_need="Searching for a gaming laptop with specific technical requirements",
            most_recent_customer_inquiry_or_need_was_already_resolved=False,
            name="search_product",
            subtleties_to_be_aware_of="A gaming laptop is strictly speaking a product, but more specifically it's an electronic product",
            tool_calls_for_candidate_tool=[
                ToolCallEvaluation(
                    applicability_rationale="need to search for a product with specific technical requirements",
                    applicability_score=6,
                    argument_evaluations=[
                        ArgumentEvaluation(
                            parameter_name="query",
                            acceptable_source_for_this_argument_according_to_its_tool_definition="<INFER THIS BASED ON TOOL DEFINITION>",
                            evaluate_is_it_provided_by_an_acceptable_source="Yes; the customer mentioned their specific requirements",
                            evaluate_was_it_already_provided_and_should_it_be_provided_again="The customer mentioned specific requirements, which is enough for me to construct a query",
                            evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided="It would be absurd to provide unsolicited information on some random product, but I don't need to guess here since the customer provided their requirements",
                            is_missing=False,
                            is_optional=False,
                            value_as_string="gaming laptop, RTX 3080, 16GB RAM",
                        )
                    ],
                    same_call_is_already_staged=False,
                    comparison_with_rejected_tools_including_references_to_subtleties="not as good a fit as search_electronics",
                    relevant_subtleties="While laptops are a kind of product, they are specifically a type of electronics product",
                    a_rejected_tool_would_have_been_a_better_fit_if_it_werent_already_rejected=True,
                    potentially_better_rejected_tool_name="search_electronics",
                    potentially_better_rejected_tool_rationale=(
                        "search_electronics is more appropriate as it allows for structured "
                        "specification of technical requirements rather than relying on text search, "
                        "which will provide more accurate results for electronic products"
                    ),
                    the_better_rejected_tool_should_clearly_be_run_in_tandem_with_the_candidate_tool=False,
                    are_optional_arguments_missing=False,
                    are_non_optional_arguments_missing=False,
                    allowed_to_run_without_optional_arguments_even_if_they_are_missing=True,
                    should_run=False,
                )
            ],
        ),
    ),
    ToolCallerInferenceShot(
        description=("the candidate tool is schedule_appointment(date: str)"),
        expected_result=ToolCallInferenceSchema(
            last_customer_message="I want to schedule an appointment please",
            most_recent_customer_inquiry_or_need="The customer wishes to schedule an appointment",
            most_recent_customer_inquiry_or_need_was_already_resolved=False,
            name="schedule_appointment",
            subtleties_to_be_aware_of="The candidate tool has a date argument",
            tool_calls_for_candidate_tool=[
                ToolCallEvaluation(
                    applicability_rationale="The customer specifically wants to schedule an appointment, and there are no better reference tools",
                    applicability_score=10,
                    argument_evaluations=[
                        ArgumentEvaluation(
                            parameter_name="date",
                            acceptable_source_for_this_argument_according_to_its_tool_definition="<INFER THIS BASED ON TOOL DEFINITION>",
                            evaluate_is_it_provided_by_an_acceptable_source="No; the customer hasn't provided a date, and I cannot guess it or infer when they'd be available",
                            evaluate_was_it_already_provided_and_should_it_be_provided_again="The customer hasn't specified it yet",
                            evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided="It is very problematic to just guess when the customer would be available for an appointment",
                            is_missing=True,
                            is_optional=False,
                            value_as_string=None,
                        )
                    ],
                    same_call_is_already_staged=False,
                    relevant_subtleties="This is the right tool to run, but we lack information for the date argument",
                    comparison_with_rejected_tools_including_references_to_subtleties="There are no tools in the list of rejected tools",
                    a_rejected_tool_would_have_been_a_better_fit_if_it_werent_already_rejected=False,
                    are_optional_arguments_missing=False,
                    are_non_optional_arguments_missing=False,
                    allowed_to_run_without_optional_arguments_even_if_they_are_missing=True,
                    should_run=False,
                )
            ],
        ),
    ),
]


shot_collection = ShotCollection[ToolCallerInferenceShot](_baseline_shots)
