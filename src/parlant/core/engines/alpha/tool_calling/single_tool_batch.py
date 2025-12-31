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

from dataclasses import dataclass
from enum import Enum
from itertools import chain
import ast
import json
import traceback
from typing import Any, Literal, Optional, Sequence, TypeAlias, cast
from pydantic import field_validator
from typing_extensions import override

from parlant.core.agents import Agent
from parlant.core.common import DefaultBaseModel, generate_id
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.emissions import EmittedEvent
from parlant.core.engines.alpha.guideline_matching.generic.common import internal_representation
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.alpha.optimization_policy import OptimizationPolicy
from parlant.core.engines.alpha.prompt_builder import BuiltInSection, PromptBuilder, SectionStatus
from parlant.core.engines.alpha.tool_calling.tool_caller import (
    ToolCallEvaluation,
    MissingToolData,
    InvalidToolData,
    ToolCall,
    ToolCallBatch,
    ToolCallBatchError,
    ToolCallBatchResult,
    ToolCallContext,
    ToolCallId,
    ToolInsights,
    measure_tool_call_batch,
)
from parlant.core.glossary import Term
from parlant.core.journeys import Journey
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.sessions import Event, EventKind, ToolEventData
from parlant.core.shots import Shot, ShotCollection
from parlant.core.tools import Tool, ToolId, ToolParameterDescriptor, ToolParameterOptions


class ValidationStatus(Enum):
    VALID = "valid"
    INVALID = "invalid"
    MISSING = "missing"


class NonConsequentialToolCallEvaluation(DefaultBaseModel):
    args: Optional[dict[str, str | None]] = None

    @field_validator("args", mode="before")
    @classmethod
    def stringify_values(cls, args: Optional[dict[str, Any]]) -> Optional[dict[str, str | None]]:
        return {k: str(v) if v is not None else None for k, v in args.items()} if args else None


class NonConsequentialToolBatchSchema(DefaultBaseModel):
    reasoning_tldr: str | None = None
    should_run: bool | None = False
    calls: list[NonConsequentialToolCallEvaluation] = []


@dataclass
class NonConsequentialSingleToolBatchShot(Shot):
    expected_result: NonConsequentialToolBatchSchema


class SingleToolBatchArgumentEvaluation(DefaultBaseModel):
    parameter_name: str
    acceptable_source_for_this_argument_according_to_its_tool_definition: str
    evaluate_is_it_provided_by_an_acceptable_source: str
    evaluate_was_it_already_provided_and_should_it_be_provided_again: str
    evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided: str
    is_optional: Optional[bool] = False
    has_default_value_if_not_provided_by_acceptable_source: Optional[bool] = None
    valid_invalid_or_missing: ValidationStatus
    value_as_string: Optional[str] = None


class SingleToolBatchToolCallEvaluation(DefaultBaseModel):
    applicability_rationale: str
    is_applicable: bool
    argument_evaluations: Optional[list[SingleToolBatchArgumentEvaluation]] = None
    same_call_is_already_staged: bool
    comparison_with_rejected_tools_including_references_to_subtleties: Optional[str] = None
    relevant_subtleties: str
    a_rejected_tool_would_have_been_a_better_fit_if_it_werent_already_rejected: Optional[bool] = (
        None
    )
    potentially_better_rejected_tool_name: Optional[str] = None
    potentially_better_rejected_tool_rationale: Optional[str] = None
    the_better_rejected_tool_should_clearly_be_run_in_tandem_with_the_candidate_tool: Optional[
        bool
    ] = None
    # These 3 ARQs are for cases we've observed where many optional arguments are missing
    # such that the model would be possibly biased to say the tool shouldn't run.
    are_optional_arguments_missing: Optional[bool] = None
    are_non_optional_arguments_missing: Optional[bool] = None
    allowed_to_run_without_optional_arguments_even_if_they_are_missing: Optional[bool] = None


class SingleToolBatchSchema(DefaultBaseModel):
    last_customer_message: Optional[str] = None
    most_recent_customer_inquiry_or_need: Optional[str] = None
    most_recent_customer_inquiry_or_need_was_already_resolved: Optional[bool] = None
    name: str
    subtleties_to_be_aware_of: str
    tool_calls_for_candidate_tool: list[SingleToolBatchToolCallEvaluation]


SingleToolCallFeature: TypeAlias = Literal["has_reference_tools", "has_optional_arguments"]

_SECTION_NAMES = {
    "general-instructions": "tool-caller-general-instructions",
    "task-description": "tool-caller-task-description",
    "examples": "tool-caller-examples",
    "tool-definitions": "tool-caller-tool-definitions",
    "tool-definition": "tool-caller-tool-definition",
    "staged-tool-calls": "tool-caller-staged-tool-calls",
    "empty-staged-tool-calls": "tool-caller-empty-staged-tool-calls",
    "output-format": "tool-caller-output-format",
}

_CONSEQUENTIAL_SUFFIX = "_consequential"
_NON_CONSEQUENTIAL_SUFFIX = "_non_consequential"


@dataclass
class SingleToolBatchShot(Shot):
    feature_set: list[SingleToolCallFeature]
    expected_result: SingleToolBatchSchema


class SingleToolBatch(ToolCallBatch):
    def __init__(
        self,
        logger: Logger,
        meter: Meter,
        optimization_policy: OptimizationPolicy,
        service_registry: ServiceRegistry,
        consequential_schema_generator: SchematicGenerator[SingleToolBatchSchema],
        non_consequential_schema_generator: SchematicGenerator[NonConsequentialToolBatchSchema],
        candidate_tool: tuple[ToolId, Tool, Sequence[GuidelineMatch]],
        context: ToolCallContext,
    ) -> None:
        self._logger = logger
        self._meter = meter

        self._optimization_policy = optimization_policy
        self._service_registry = service_registry
        self._consequential_schema_generator = consequential_schema_generator
        self._non_consequential_schema_generator = non_consequential_schema_generator
        self._context = context
        self._candidate_tool = candidate_tool

    def _is_tool_already_staged(self, tool_id: ToolId) -> bool:
        for event in self._context.staged_events:
            if event.kind == EventKind.TOOL:
                tool_calls = cast(ToolEventData, event.data).get("tool_calls", [])

                for tc in tool_calls:
                    if tc.get("tool_id") == tool_id.to_string():
                        return True
        return False

    def _is_tool_call_already_staged(self, tool_id: ToolId, args: dict[str, Any] | None) -> bool:
        for event in self._context.staged_events:
            if event.kind == EventKind.TOOL:
                tool_calls = cast(ToolEventData, event.data).get("tool_calls", [])

                for tc in tool_calls:
                    if tc.get("tool_id") == tool_id.to_string():
                        if not args:
                            return not tc.get("arguments")
                        else:
                            return tc.get("arguments") == args
        return False

    @override
    async def process(self) -> ToolCallBatchResult:
        tool_id, tool, _ = self._candidate_tool

        # Optimization 1: auto-approve non-consequential tools with no parameters
        # if they are not already staged
        if (
            not tool.consequential
            and not tool.parameters
            and not self._is_tool_already_staged(tool_id)
        ):
            self._logger.debug(
                f"Inference::Completion::Activated: {tool_id.to_string()}: "
                "Auto-approved non-consequential tool with no parameters"
            )

            return ToolCallBatchResult(
                generation_info=GenerationInfo(
                    schema_name="SingleToolBatchSchema",
                    model="auto-approved",
                    duration=0.0,
                    usage=UsageInfo(input_tokens=0, output_tokens=0, extra={}),
                ),
                tool_calls=[
                    ToolCall(
                        id=ToolCallId(generate_id()),
                        tool_id=tool_id,
                        arguments={},
                    )
                ],
                insights=ToolInsights(
                    evaluations=[(tool_id, ToolCallEvaluation.NEEDS_TO_RUN)],
                    missing_data=[],
                    invalid_data=[],
                ),
            )

        # Optimization 2: use simplified mode for non-consequential tools WITH parameters
        if not tool.consequential:
            async with measure_tool_call_batch(self._meter, self):
                return await self._infer_calls_for_non_consequential_tool(
                    agent=self._context.agent,
                    context_variables=self._context.context_variables,
                    interaction_history=self._context.interaction_history,
                    terms=self._context.terms,
                    candidate_descriptor=self._candidate_tool,
                    staged_events=self._context.staged_events,
                )

        # Full inference path (for consequential tools)
        async with measure_tool_call_batch(self._meter, self):
            (
                generation_info,
                inference_output,
                execution_status,
                missing_data,
                invalid_data,
            ) = await self._infer_calls_for_consequential_tool(
                agent=self._context.agent,
                context_variables=self._context.context_variables,
                interaction_history=self._context.interaction_history,
                terms=self._context.terms,
                ordinary_guideline_matches=self._context.ordinary_guideline_matches,
                journeys=self._context.journeys,
                candidate_descriptor=self._candidate_tool,
                reference_tools=[],
                staged_events=self._context.staged_events,
            )

            return ToolCallBatchResult(
                generation_info=generation_info,
                tool_calls=inference_output,
                insights=ToolInsights(
                    evaluations=execution_status,
                    missing_data=missing_data,
                    invalid_data=invalid_data,
                ),
            )

    async def _validate_argument_value(
        self,
        parameter: tuple[ToolParameterDescriptor, ToolParameterOptions],
        value: str,
    ) -> bool:
        """Currently validate only parameters with enum values"""
        descriptor = parameter[0]
        if "enum" in descriptor:
            if descriptor["type"] == "string":
                return value in descriptor["enum"]
            if descriptor["type"] == "array":
                return all(v in descriptor["enum"] for v in ast.literal_eval(value))
        return True

    async def _infer_calls_for_consequential_tool(
        self,
        agent: Agent,
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        ordinary_guideline_matches: Sequence[GuidelineMatch],
        journeys: Sequence[Journey],
        candidate_descriptor: tuple[ToolId, Tool, Sequence[GuidelineMatch]],
        reference_tools: Sequence[tuple[ToolId, Tool]],
        staged_events: Sequence[EmittedEvent],
    ) -> tuple[
        GenerationInfo,
        list[ToolCall],
        list[tuple[ToolId, ToolCallEvaluation]],
        list[MissingToolData],
        list[InvalidToolData],
    ]:
        inference_prompt = self._build_consequential_tool_prompt(
            agent,
            context_variables,
            interaction_history,
            terms,
            ordinary_guideline_matches,
            journeys,
            candidate_descriptor,
            reference_tools,
            staged_events,
            self._get_shot_collection_for_tools(await self.shots(), bool(reference_tools)),
        )

        # Send the tool call inference prompt to the LLM
        generation_attempt_temperatures = (
            self._optimization_policy.get_tool_calling_batch_retry_temperatures()
        )

        last_generation_exception: Exception | None = None

        for generation_attempt in range(3):
            try:
                generation_info, inference_output = await self._run_consequential_tool_inference(
                    prompt=inference_prompt,
                    tool_id=candidate_descriptor[0],
                    temperature=generation_attempt_temperatures[generation_attempt],
                )

                # Evaluate the tool calls
                (
                    tool_calls,
                    evaluations,
                    missing_data,
                    invalid_data,
                ) = await self._evaluate_consequential_tool_calls(
                    inference_output, candidate_descriptor
                )

                return generation_info, tool_calls, evaluations, missing_data, invalid_data

            except Exception as exc:
                self._logger.warning(
                    f"SingleToolBatch attempt {generation_attempt} failed: {traceback.format_exception(exc)}"
                )

                last_generation_exception = exc

        raise ToolCallBatchError() from last_generation_exception

    async def _evaluate_consequential_tool_calls(
        self,
        inference_output: Sequence[SingleToolBatchToolCallEvaluation],
        candidate_descriptor: tuple[ToolId, Tool, Sequence[GuidelineMatch]],
    ) -> tuple[
        list[ToolCall],
        list[tuple[ToolId, ToolCallEvaluation]],
        list[MissingToolData],
        list[InvalidToolData],
    ]:
        tool = candidate_descriptor[1]
        tool_calls = []
        evaluations = []
        missing_data: list[MissingToolData] = []
        invalid_data: list[InvalidToolData] = []
        tool_id, tool, _ = candidate_descriptor

        for tc in inference_output:
            entries = {
                e.parameter_name: e
                for e in tc.argument_evaluations or []
                if e.parameter_name in tool.parameters
            }

            # First - check validity of all parameters with provided values
            all_values_valid = True

            for evaluation in entries.values():
                descriptor, options = tool.parameters[evaluation.parameter_name]

                if evaluation.value_as_string and not await self._validate_argument_value(
                    tool.parameters[evaluation.parameter_name],
                    evaluation.value_as_string,
                ):
                    all_values_valid = False

                    self._logger.warning(
                        f'Inference::Completion::InvalidArgument: {tool_id.to_string()}: {evaluation.parameter_name}="{evaluation.value_as_string}"'
                    )

                    if not options.hidden:
                        invalid_data.append(
                            InvalidToolData(
                                parameter=options.display_name or evaluation.parameter_name,
                                invalid_value=evaluation.value_as_string,
                                significance=options.significance,
                                description=descriptor.get("description"),
                                precedence=options.precedence,
                                choices=descriptor.get("enum", None),
                            )
                        )

                    evaluations.append((tool_id, ToolCallEvaluation.CANNOT_RUN))

            if (
                tc.is_applicable
                and not tc.same_call_is_already_staged
                and (
                    not tc.a_rejected_tool_would_have_been_a_better_fit_if_it_werent_already_rejected
                    or tc.the_better_rejected_tool_should_clearly_be_run_in_tandem_with_the_candidate_tool
                )
            ):
                if all(
                    not evaluation.valid_invalid_or_missing == ValidationStatus.MISSING
                    for evaluation in tc.argument_evaluations or []
                    if evaluation.parameter_name in tool.required
                ):
                    arguments = {}

                    if tool.parameters:  # We check this because sometimes LLMs hallucinate placeholders for no-param tools
                        for evaluation in tc.argument_evaluations or []:
                            if evaluation.valid_invalid_or_missing == ValidationStatus.MISSING:
                                continue

                            # Note that if LLM provided 'None' for a required parameter with a default - it will get 'None' as value
                            arguments[evaluation.parameter_name] = evaluation.value_as_string

                    if all_values_valid:
                        self._logger.debug(
                            f"Inference::Completion::Activated: {tool_id.to_string()}:\n{tc.model_dump_json(indent=2)}"
                        )

                        tool_calls.append(
                            ToolCall(
                                id=ToolCallId(generate_id()),
                                tool_id=tool_id,
                                arguments=arguments,
                            )
                        )

                        evaluations.append((tool_id, ToolCallEvaluation.NEEDS_TO_RUN))
                else:
                    has_missing_arguments = False

                    for evaluation in tc.argument_evaluations or []:
                        if evaluation.parameter_name not in tool.parameters:
                            self._logger.warning(
                                f"Inference::Completion: Unknown argument '{evaluation.parameter_name}' not found in tool parameters"
                            )
                            continue

                        tool_descriptor, tool_options = tool.parameters[evaluation.parameter_name]

                        if (
                            evaluation.valid_invalid_or_missing == ValidationStatus.MISSING
                            and not evaluation.is_optional
                        ):
                            display_name = tool_options.display_name or evaluation.parameter_name

                            if not tool_options.hidden:
                                if display_name not in [p.parameter for p in missing_data]:
                                    missing_data.append(
                                        MissingToolData(
                                            parameter=display_name,
                                            significance=tool_options.significance,
                                            description=tool_descriptor.get("description"),
                                            precedence=tool_options.precedence,
                                            choices=tool_descriptor.get("enum", None),
                                        )
                                    )
                            else:
                                self._logger.warning(
                                    f"Inference::Completion: Hidden argument '{evaluation.parameter_name}' is missing"
                                )

                            has_missing_arguments = True

                    for parameter_name in tool.parameters:
                        if parameter_name not in entries:
                            self._logger.warning(
                                f"Inference::Completion: Argument '{parameter_name}' is missing"
                            )

                            if not tool_options.hidden:
                                if display_name not in [p.parameter for p in invalid_data]:
                                    missing_data.append(
                                        MissingToolData(
                                            parameter=display_name,
                                            significance=tool_options.significance,
                                            description=tool_descriptor.get("description"),
                                            precedence=tool_options.precedence,
                                            choices=tool_descriptor.get("enum", None),
                                        )
                                    )
                            else:
                                self._logger.warning(
                                    f"Inference::Completion: Hidden argument '{evaluation.parameter_name}' is missing"
                                )

                        has_missing_arguments = True

                    if has_missing_arguments:
                        evaluations.append((tool_id, ToolCallEvaluation.CANNOT_RUN))

                    self._logger.debug(
                        f"Inference::Completion::Rejected: Missing arguments for {tool_id.to_string()}\n{tc.model_dump_json(indent=2)}"
                    )

            else:
                self._logger.debug(
                    f"Inference::Completion::Skipped: {tool_id.to_string()}\n{tc.model_dump_json(indent=2)}"
                )

                evaluations.append((tool_id, ToolCallEvaluation.DATA_ALREADY_IN_CONTEXT))

        return tool_calls, evaluations, missing_data, invalid_data

    def _get_shot_collection_for_tools(
        self, shots: Sequence[SingleToolBatchShot], has_reference_tools: bool
    ) -> Sequence[SingleToolBatchShot]:
        shot_collection: Sequence[SingleToolBatchShot] = [
            shot
            for shot in shots
            if not shot.feature_set
            or ("has_reference_tools" in shot.feature_set) == has_reference_tools
        ]
        return shot_collection

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

    async def shots(self) -> Sequence[SingleToolBatchShot]:
        return await consequential_shot_collection.list()

    def _format_shots(
        self,
        shots: Sequence[SingleToolBatchShot | NonConsequentialSingleToolBatchShot],
    ) -> str:
        def format_shot(
            shot: SingleToolBatchShot | NonConsequentialSingleToolBatchShot,
        ) -> str:
            return f"""
- **Context**:
{shot.description}

- **Expected Result**:
```json
{json.dumps(shot.expected_result.model_dump(mode="json", exclude_unset=True), indent=2)}
```"""

        return "\n".join(
            f"""
Example #{i}: ###
{format_shot(shot)}
###
"""
            for i, shot in enumerate(shots, start=1)
        )

    def _build_consequential_tool_prompt(
        self,
        agent: Agent,
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_event_list: Sequence[Event],
        terms: Sequence[Term],
        ordinary_guideline_matches: Sequence[GuidelineMatch],
        journeys: Sequence[Journey],
        batch: tuple[ToolId, Tool, Sequence[GuidelineMatch]],
        reference_tools: Sequence[tuple[ToolId, Tool]],
        staged_events: Sequence[EmittedEvent],
        shots: Sequence[SingleToolBatchShot],
    ) -> PromptBuilder:
        staged_calls = self._get_staged_calls(staged_events)

        builder = PromptBuilder(on_build=lambda prompt: self._logger.trace(f"Prompt:\n{prompt}"))

        builder.add_section(
            name=_SECTION_NAMES["general-instructions"] + _CONSEQUENTIAL_SUFFIX,
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
            name=_SECTION_NAMES["task-description"] + _CONSEQUENTIAL_SUFFIX,
            template="""
-----------------
TASK DESCRIPTION
-----------------
Your task is to review the provided tool and, based on your most recent interaction with the customer, decide whether it is applicable.
Indicate the tool applicability with a boolean value: true if the tool is useful at this point, or false if it is not.
For any tool marked as true, include the available arguments for activation.
Note that a tool may be considered applicable even if not all of its required arguments are available. In such cases, provide the parameters that are currently available,
following the format specified in its description.

While doing so, take the following instructions into account:

1. You may suggest tool that don't directly address the customer's latest interaction but can advance the conversation to a more useful state based on function definitions.
2. Each tool may be called multiple times with different arguments.
3. Avoid calling a tool with the same arguments more than once, unless clearly justified by the interaction.
4. Ensure each tool call relies only on the immediate context and staged calls, without requiring other tools not yet invoked, to avoid dependencies.
5. If a tool needs to be applied multiple times (each with different arguments), you may include it in the output multiple times.

The exact format of your output will be provided to you at the end of this prompt.

The following examples show correct outputs for various hypothetical situations.
Only the responses are provided, without the interaction history or tool descriptions, though these can be inferred from the responses.

""",
            props={},
        )
        builder.add_section(
            name=_SECTION_NAMES["examples"] + _CONSEQUENTIAL_SUFFIX,
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
            template=self._add_guideline_matches_section(
                ordinary_guideline_matches,
                (batch[0], batch[2]),
            ),
            props={
                "ordinary_guideline_matches": ordinary_guideline_matches,
                "tool_id_propositions": (batch[0], batch[2]),
            },
        )
        tool_definitions_template, tool_definitions_props = self._add_tool_definitions_section(
            candidate_tool=(batch[0], batch[1]),
            reference_tools=reference_tools,
        )
        builder.add_section(
            name=_SECTION_NAMES["tool-definitions"] + _CONSEQUENTIAL_SUFFIX,
            template=tool_definitions_template,
            props={
                **tool_definitions_props,
                "candidate_tool": (batch[0], batch[1]),
                "reference_tools": reference_tools,
            },
        )
        if staged_calls:
            builder.add_section(
                name=_SECTION_NAMES["staged-tool-calls"] + _CONSEQUENTIAL_SUFFIX,
                template="""
STAGED TOOL CALLS
-----------------
The following is a list of tool calls staged after the interaction's latest state. Use this information to avoid redundant calls and to guide your response.

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
                name=_SECTION_NAMES["empty-staged-tool-calls"] + _CONSEQUENTIAL_SUFFIX,
                template="""
STAGED TOOL CALLS
-----------------
There are no staged tool calls at this time.
""",
                props={},
            )

        builder.add_section(
            name=_SECTION_NAMES["output-format"] + _CONSEQUENTIAL_SUFFIX,
            template="""
OUTPUT FORMAT
-----------------
Given the tool, your output should adhere to the following format:
```json
{{
    "last_customer_message": "<REPEAT THE LAST USER MESSAGE IN THE INTERACTION>",
    "most_recent_customer_inquiry_or_need": "<CUSTOMER'S INQUIRY OR NEED>",
    "most_recent_customer_inquiry_or_need_was_already_resolved": <BOOL>,
    "name": "{service_name}:{tool_name}",
    "subtleties_to_be_aware_of": "<NOTE ANY SIGNIFICANT SUBTLETIES TO BE AWARE OF WHEN RUNNING THIS TOOL IN OUR AGENT'S CONTEXT>",
    "tool_calls_for_candidate_tool": [{tool_calls_for_candidate_tool_json_description}
    ]
}}
```

However, note that you may choose to have multiple entries in 'tool_calls_for_candidate_tool' if you wish to call the candidate tool multiple times with different arguments.
""",
            props={
                "service_name": batch[0].service_name,
                "tool_name": batch[0].tool_name,
                "candidate_tool": batch[1],
                "has_reference_tools": bool(reference_tools),
                "tool_calls_for_candidate_tool_json_description": self._format_tool_calls_for_candidate_tool_json_description(
                    candidate_tool=batch[1], has_reference_tools=bool(reference_tools)
                ),
            },
        )

        return builder

    def _format_tool_calls_for_candidate_tool_json_description(
        self, candidate_tool: Tool, has_reference_tools: bool
    ) -> str:
        optional_arguments = [
            name for name in candidate_tool.parameters if name not in candidate_tool.required
        ]
        result = """
        {
            "applicability_rationale": "<A FEW WORDS THAT EXPLAIN WHETHER, HOW, AND TO WHAT EXTENT THE TOOL NEEDS TO BE CALLED AT THIS POINT>",
            "is_applicable": <BOOL>,"""
        result += """
            "argument_evaluations": [
                {
                    "parameter_name": "<PARAMETER NAME>",
                    "acceptable_source_for_this_argument_according_to_its_tool_definition": "<REPEAT THE ACCEPTABLE SOURCE FOR THE ARGUMENT FROM TOOL DEFINITION>",
                    "evaluate_is_it_provided_by_an_acceptable_source": "<BRIEFLY EVALUATE IF THE SOURCE FOR THE VALUE MATCHES THE ACCEPTABLE SOURCE>",
                    "evaluate_was_it_already_provided_and_should_it_be_provided_again": "<BRIEFLY EVALUATE IF THE PARAMETER VALUE WAS PROVIDED AND SHOULD BE PROVIDED AGAIN>",
                    "evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided": "<BRIEFLY EVALUATE IF IT'S A PROBLEM TO GUESS THE VALUE>","""
        if optional_arguments:
            result += """
                    "is_optional": <BOOL>,"""

        result += """
                    "valid_invalid_or_missing": "<STR: EITHER 'missing', 'invalid' OR 'valid' DEPENDING IF THE VALUE IS MISSING, PROVIDED BUT NOT FOUND IN ENUM LIST, OR PROVIDED AND FOUND IN ENUM LIST (OR DOESN'T HAVE ENUM LIST)>",
                    "value_as_string": "<PARAMETER VALUE>,"
                }
            ],"""

        result += """
            "same_call_is_already_staged": <BOOL>,
            "relevant_subtleties": "<IF SUBTLETIES FOUND, REFER TO THE RELEVANT ONES HERE>", """

        if has_reference_tools:
            result += """
            "comparison_with_rejected_tools_including_references_to_subtleties": "<A VERY BRIEF OVERVIEW OF HOW THIS CALL FARES AGAINST OTHER TOOLS IN APPLICABILITY>",
            "a_rejected_tool_would_have_been_a_better_fit_if_it_werent_already_rejected": <BOOL>,
            "potentially_better_rejected_tool_name": "<IF CANDIDATE TOOL IS A WORSE FIT THAN A REJECTED TOOL, THIS IS THE NAME OF THAT REJECTED TOOL>",
            "potentially_better_rejected_tool_rationale": "<IF CANDIDATE TOOL IS A WORSE FIT THAN A REJECTED TOOL, THIS EXPLAINS WHY>",
            "the_better_rejected_tool_should_clearly_be_run_in_tandem_with_the_candidate_tool": <BOOL>,"""

        if optional_arguments:
            result += """
            "are_optional_arguments_missing": <BOOL>,
            "are_non_optional_arguments_missing": <BOOL>,
            "allowed_to_run_without_optional_arguments_even_if_they_are_missing": <BOOL-ALWAYS TRUE>,"""

        result += """
        }"""
        return result

    def _add_tool_definitions_section(
        self,
        candidate_tool: tuple[ToolId, Tool],
        reference_tools: Sequence[tuple[ToolId, Tool]],
    ) -> tuple[str, dict[str, Any]]:
        def _format_type(descriptor_type: str) -> str:
            """Return the type-specific format suffix for the given descriptor type."""
            if descriptor_type == "datetime":
                return f"{descriptor_type}: year-month-day hour:minute:second"
            if descriptor_type == "date":
                return f"{descriptor_type}: year-month-day"
            if descriptor_type == "timedelta":
                return f"{descriptor_type}: hours:minutes:seconds"
            return descriptor_type

        def _get_param_spec(spec: tuple[ToolParameterDescriptor, ToolParameterOptions]) -> str:
            descriptor, options = spec

            result: dict[str, Any] = {"schema": {"type": _format_type(descriptor["type"])}}

            if descriptor["type"] == "array":
                result["schema"]["items"] = {"type": _format_type(descriptor["item_type"])}

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
                        "This argument can be extracted in the best way you think (context, tool results, customer input, etc.)"
                    )
                case "context":
                    result["acceptable_source"] = (
                        "This argument can be extracted only from the context given in this prompt (tool results, interaction, variables, etc.)"
                    )
                case "customer":
                    result["acceptable_source"] = (
                        "This argument must be provided by the customer in the interaction itself, and NEVER automatically guessed by you"
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

    def _add_guideline_matches_section(
        self,
        ordinary_guideline_matches: Sequence[GuidelineMatch],
        tool_id_propositions: tuple[ToolId, Sequence[GuidelineMatch]],
    ) -> str:
        all_matches = [
            match
            for match in list(set(chain(ordinary_guideline_matches, tool_id_propositions[1])))
            if internal_representation(match.guideline).action
        ]

        guideline_list = ""
        if all_matches:
            guidelines = []

            for i, p in enumerate(all_matches, start=1):
                rep = internal_representation(p.guideline)
                if rep.condition:
                    guideline = f"{i}) When {rep.condition}, then {rep.action}"
                else:
                    guideline = f"{i}) {rep.action}"
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
        staged_calls = [
            PromptBuilder.adapt_event(e) for e in emitted_events if e.kind == EventKind.TOOL
        ]

        if not staged_calls:
            return None

        return json.dumps(staged_calls)

    async def _run_consequential_tool_inference(
        self,
        prompt: PromptBuilder,
        tool_id: ToolId,
        temperature: float,
    ) -> tuple[GenerationInfo, Sequence[SingleToolBatchToolCallEvaluation]]:
        inference = await self._consequential_schema_generator.generate(
            prompt=prompt,
            hints={"temperature": temperature},
        )
        self._logger.trace(
            f"Inference::Completion: {tool_id.to_string()}\n{inference.content.model_dump_json(indent=2)}"
        )

        return inference.info, inference.content.tool_calls_for_candidate_tool

    async def _infer_calls_for_non_consequential_tool(
        self,
        agent: Agent,
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        candidate_descriptor: tuple[ToolId, Tool, Sequence[GuidelineMatch]],
        staged_events: Sequence[EmittedEvent],
    ) -> ToolCallBatchResult:
        prompt = self._build_non_consequential_tool_prompt(
            agent=agent,
            context_variables=context_variables,
            interaction_history=interaction_history,
            ordinary_guideline_matches=self._context.ordinary_guideline_matches,
            terms=terms,
            candidate_descriptor=candidate_descriptor,
            staged_events=staged_events,
            shots=await non_consequential_shot_collection.list(),
        )

        generation_attempt_temperatures = (
            self._optimization_policy.get_tool_calling_batch_retry_temperatures()
        )

        last_exception: Exception | None = None

        for attempt in range(3):
            try:
                generation_info, output = await self._run_non_consequential_tool_inference(
                    prompt=prompt,
                    tool_id=candidate_descriptor[0],
                    temperature=generation_attempt_temperatures[attempt],
                )

                tool_calls, evaluations, missing_data, invalid_data = (
                    self._evaluate_non_consequential_tool_calls(output, candidate_descriptor)
                )

                return ToolCallBatchResult(
                    generation_info=generation_info,
                    tool_calls=tool_calls,
                    insights=ToolInsights(
                        evaluations=evaluations,
                        missing_data=missing_data,
                        invalid_data=invalid_data,
                    ),
                )

            except Exception as exc:
                self._logger.warning(
                    f"SingleToolBatch attempt {attempt} failed: {traceback.format_exception(exc)}"
                )
                last_exception = exc

        raise ToolCallBatchError() from last_exception

    def _build_non_consequential_tool_prompt(
        self,
        agent: Agent,
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        ordinary_guideline_matches: Sequence[GuidelineMatch],
        candidate_descriptor: tuple[ToolId, Tool, Sequence[GuidelineMatch]],
        staged_events: Sequence[EmittedEvent],
        shots: Sequence[NonConsequentialSingleToolBatchShot],
    ) -> PromptBuilder:
        staged_calls = self._get_staged_calls(staged_events)
        tool_id, tool, _ = candidate_descriptor

        builder = PromptBuilder(on_build=lambda prompt: self._logger.trace(f"Prompt:\n{prompt}"))

        # Minimal instructions
        builder.add_section(
            name=_SECTION_NAMES["general-instructions"] + _NON_CONSEQUENTIAL_SUFFIX,
            template=f"""\
GENERAL INSTRUCTIONS
-----------------
You are part of a system of AI agents which interact with a customer on the behalf of a business.

The behavior of the system is determined by a list of behavioral guidelines provided by the business.

Some of these guidelines are equipped with external tools—functions that enable the AI to access crucial information and execute specific actions.

Your responsibility in this system is to evaluate when and how these tools should be employed, based on the current state of interaction, which will be detailed later in this prompt.

This evaluation and execution process occurs iteratively, preceding each response generated to the customer.

{'''Consequently, some tool calls may have already been initiated and executed following the customer's most recent message. Any such completed tool call will be detailed later in this prompt along with its result, under "staged calls". These specific calls do not require to be re-run at this time, unless you identify a valid reason for their reevaluation.''' if staged_calls else ""}

**Task Instructions:**
- CASE 1: The tool is clearly relevant to the customer's current request and all **required** parameter values can be determined or inferred contextually. In this case, mark "should_run": true and provide the call parameters in "calls" - for one or more of the calls you've determined we must now run, as the case may be. 
    If an **optional** parameter value cannot be determined or inferred contextually, you may still create the tool call, inserting **null** as value for that parameter
{"- CASE 2: If the same call is already staged, do not call the tool again - and do not create a tool call - and mark it as should_run: false, explaining this decision in your reasoning" if staged_calls else ""}
- CASE 3: If the tool *should* run in principle, except that a **required** parameter value cannot be determined or inferred contextually, do not create a tool call - but still mark it as should_run: true. For the missing parameters, insert "<<__missing__>>" as value (as a string, regardless of the parameter type).
- CASE 4: The tool should not be called at this time because it's not particularly relevant to the situation - in this case, mark "should_run": false and do not create any tool calls

Make sure to note the case number you're following in your "reasoning_tldr" field.

Parameter values extraction:
Extract parameter values from the conversation or prompt context. 
You're free to infer the parameters from context, in the best way you think - e.g., if someone says "next Friday", you can convert that to an actual date string if you know the date - and so forth
Use inference and contextual understanding to derive accurate values. It's better to make reasonable inferences about parameter values from the user's request than to not run the tool at all because a parameter wasn't explicitly specified.
You do have to adhere to the format of each parameter - never pass a parameter value that doesn't match the expected format.
""",
            props={},
        )

        builder.add_section(
            name=_SECTION_NAMES["examples"] + _NON_CONSEQUENTIAL_SUFFIX,
            template="""
The following examples show correct outputs for various hypothetical situations.
Only the responses are provided, without the interaction history or tool descriptions, though these can be inferred from the responses.


EXAMPLES
-----------------
{formatted_shots}
""",
            props={"formatted_shots": self._format_shots(shots), "shots": shots},
        )

        builder.add_agent_identity(agent)

        parameters_info = {}
        for name, spec in tool.parameters.items():
            descriptor, options = spec
            param_info: dict[str, Any] = {
                "type": descriptor.get("type", "string"),
            }
            if description := options.description or descriptor.get("description"):
                param_info["description"] = description
            if enum := descriptor.get("enum"):
                param_info["enum"] = enum
            parameters_info[name] = param_info

        tool_notes = ""
        parameter_types = {p.get("type", "") for _, p in parameters_info.items()}
        if "datetime" in parameter_types:
            tool_notes += "\nNote: If a parameter's type is datetime, return the parameter value in the format 'year-month-day hour:minute:second'."
        if "date" in parameter_types:
            tool_notes += "\nNote: If a parameter's type is date, return the parameter value in the format 'year-month-day'."
        if "timedelta" in parameter_types:
            tool_notes += "\nNote: If a parameter's type is timedelta, return the parameter value in the format 'hours:minutes:seconds'."

        builder.add_section(
            name=_SECTION_NAMES["tool-definition"] + _NON_CONSEQUENTIAL_SUFFIX,
            template="""
TOOL TO EVALUATE:
-----------------
Name: {tool_name}
Description: {tool_description}
Parameters: {parameters_json}
Required parameters: {required_params}
Optional parameters: {optional_params}
{notes}
""",
            props={
                "tool_name": f"{tool_id.service_name}:{tool_id.tool_name}",
                "tool_description": tool.description,
                "parameters_json": json.dumps(parameters_info, indent=2),
                "required_params": list(tool.required),
                "optional_params": list(set(tool.parameters) - set(tool.required)),
                "notes": tool_notes.strip(),
            },
        )

        if staged_calls:
            builder.add_section(
                name=_SECTION_NAMES["staged-tool-calls"] + _NON_CONSEQUENTIAL_SUFFIX,
                template="""
**ALREADY STAGED CALLS:**
{staged_calls}
Do not call the tool again with the same arguments.
""",
                props={"staged_calls": staged_calls},
            )

        arg_formats = []

        for param_name, spec in tool.parameters.items():
            descriptor, _ = spec

            missing_str = "None" if param_name not in tool.required else '"<<__missing__>>"'
            prefix, suffix = ('["', '", ...]') if descriptor.get("type") == "array" else ('"', '"')

            if choices := descriptor.get("enum"):
                choices_str = ", ".join(f'"{choice}"' for choice in choices)
                arg_formats.append(
                    f'"{param_name}": {prefix}<Select the most appropriate value for this parameter from ({choices_str}) — if it can not be inferred from the interaction set it to {missing_str}>{suffix}'
                )
            else:
                arg_formats.append(
                    f'"{param_name}": {prefix}<Extract the value from the interaction ALWAYS IN STRINGIFIED FORM. If it can not be inferred set it to {missing_str}>{suffix}'
                )

        builder.add_section(
            name="Note-about-context" + _NON_CONSEQUENTIAL_SUFFIX,
            template="""
CONTEXT OF INTERACTION:
-----------------------
Consider ALL following information when deciding how and with what parameters to run the tool:
""",
        )

        builder.add_context_variables(context_variables)

        if terms:
            builder.add_section(
                name=BuiltInSection.GLOSSARY,
                template=self._get_glossary_text(terms),
                props={"terms": terms},
                status=SectionStatus.ACTIVE,
            )

        builder.add_interaction_history(interaction_history)

        builder.add_section(
            name=BuiltInSection.GUIDELINE_DESCRIPTIONS,
            template=self._add_guideline_matches_section(
                ordinary_guideline_matches,
                (candidate_descriptor[0], candidate_descriptor[2]),
            ),
            props={
                "ordinary_guideline_matches": ordinary_guideline_matches,
                "tool_id_propositions": (candidate_descriptor[0], candidate_descriptor[2]),
            },
        )

        arg_formats_str = ",\n                ".join(arg_formats)

        builder.add_section(
            name=_SECTION_NAMES["output-format"] + _NON_CONSEQUENTIAL_SUFFIX,
            template=f"""\
OUTPUT FORMAT:
```json
{{{{
    "reasoning_tldr": "<BRIEFLY EXPLAIN - IN A FEW WORDS - YOUR REASONING FOR RUNNING {tool_id.to_string()}>",
    "should_run": <true if tool should be called (at least one call), false otherwise>,
    "calls": [
        {{{{
            "args":
            {{{{
                {arg_formats_str}
            }}}}
        }}}},
        ...
    ]
}}}}
```
""",
            props={
                "tool_name": f"{tool_id.service_name}:{tool_id.tool_name}",
                "tool_parameters": tool.parameters,
            },
        )

        return builder

    async def _run_non_consequential_tool_inference(
        self,
        tool_id: ToolId,
        prompt: PromptBuilder,
        temperature: float,
    ) -> tuple[GenerationInfo, Sequence[NonConsequentialToolCallEvaluation]]:
        inference = await self._non_consequential_schema_generator.generate(
            prompt=prompt,
            hints={"temperature": temperature},
        )
        self._logger.trace(
            f"Inference::Completion: {tool_id.to_string()}\n{inference.content.model_dump_json(indent=2)}"
        )

        return inference.info, inference.content.calls

    def _evaluate_non_consequential_tool_calls(
        self,
        output: Sequence[NonConsequentialToolCallEvaluation],
        candidate_descriptor: tuple[ToolId, Tool, Sequence[GuidelineMatch]],
    ) -> tuple[
        list[ToolCall],
        list[tuple[ToolId, ToolCallEvaluation]],
        list[MissingToolData],
        list[InvalidToolData],
    ]:
        tool_id, tool, _ = candidate_descriptor
        tool_calls: list[ToolCall] = []
        evaluations: list[tuple[ToolId, ToolCallEvaluation]] = []
        missing_data: list[MissingToolData] = []
        invalid_data: list[InvalidToolData] = []

        for tc in output:
            if not self._is_tool_call_already_staged(tool_id, tc.args):
                arguments: dict[str, str | None] = {}

                for param_name, param_value in (tc.args or {}).items():
                    if param_name in tool.parameters:
                        arguments[param_name] = param_value

                # Check if all required parameters are present
                missing_required = [r for r in tool.required if r not in arguments] + [
                    name
                    for name, value in arguments.items()
                    if value == "<<__missing__>>" or value == "['<<__missing__>>']"
                ]

                if not missing_required:
                    tool_calls.append(
                        ToolCall(
                            id=ToolCallId(generate_id()),
                            tool_id=tool_id,
                            arguments=arguments,
                        )
                    )

                    self._logger.debug(
                        f"Inference::Completion::Activated: {tool_id.to_string()}:\n{tc.model_dump_json(indent=2)}"
                    )

                    evaluations.append((tool_id, ToolCallEvaluation.NEEDS_TO_RUN))
                else:
                    self._logger.debug(
                        f"Inference::Completion::Rejected: Missing arguments for {tool_id.to_string()}\n{tc.model_dump_json(indent=2)}"
                    )

                    for parameter_name in missing_required:
                        descriptor, options = tool.parameters[parameter_name]

                        if not options.hidden:
                            self._logger.warning(
                                f"Inference::Completion: Argument '{parameter_name}' is missing"
                            )

                            missing_data.append(
                                MissingToolData(
                                    parameter=options.display_name or parameter_name,
                                    significance=options.significance,
                                    description=descriptor.get("description"),
                                    precedence=options.precedence,
                                    choices=descriptor.get("enum", None),
                                )
                            )
                        else:
                            self._logger.warning(
                                f"Inference::Completion: Hidden argument '{parameter_name}' is missing"
                            )

                    evaluations.append((tool_id, ToolCallEvaluation.CANNOT_RUN))
            else:
                self._logger.debug(
                    f"Inference::Completion::Skipped: {tool_id.to_string()}\n{tc.model_dump_json(indent=2)}"
                )

                evaluations.append((tool_id, ToolCallEvaluation.DATA_ALREADY_IN_CONTEXT))

        return tool_calls, evaluations, missing_data, invalid_data

    def __repr__(self) -> str:
        return (
            f"SingleToolBatchEngine({self._candidate_tool[0].to_string()}, "
            f"consequential={self._candidate_tool[1].consequential})"
        )


example_1_shot = SingleToolBatchShot(
    description="the id of the customer is 12345, and check_balance(12345) is already listed as a staged tool call",
    feature_set=[],
    expected_result=SingleToolBatchSchema(
        last_customer_message="Do I have enough money in my account to get a taxi from New York to Newark?",
        most_recent_customer_inquiry_or_need=(
            "Checking customer's balance, comparing it to the price of a taxi from New York to Newark, "
            "and report the result to the customer"
        ),
        most_recent_customer_inquiry_or_need_was_already_resolved=False,
        name="check_balance",
        subtleties_to_be_aware_of="check_balance(12345) is already staged",
        tool_calls_for_candidate_tool=[
            SingleToolBatchToolCallEvaluation(
                applicability_rationale="We need the client's current balance to respond to their question",
                is_applicable=True,
                argument_evaluations=[
                    SingleToolBatchArgumentEvaluation(
                        parameter_name="customer_id",
                        acceptable_source_for_this_argument_according_to_its_tool_definition="<INFER THIS BASED ON TOOL DEFINITION>",
                        evaluate_is_it_provided_by_an_acceptable_source="The customer ID is given by a context variable",
                        evaluate_was_it_already_provided_and_should_it_be_provided_again="No need to provide it again as the customer's ID is unique and doesn't change",
                        evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided="It would be extremely problematic, but I don't need to guess here since I have it",
                        is_optional=False,
                        valid_invalid_or_missing=ValidationStatus.VALID,
                        value_as_string="12345",
                    )
                ],
                same_call_is_already_staged=True,
                relevant_subtleties="check_balance(12345) is already staged",
                are_optional_arguments_missing=False,
                are_non_optional_arguments_missing=False,
                allowed_to_run_without_optional_arguments_even_if_they_are_missing=True,
            )
        ],
    ),
)

example_2_shot = SingleToolBatchShot(
    description="the id of the customer is 12345, and check_balance(12345) is listed as the only staged tool call",
    feature_set=[],
    expected_result=SingleToolBatchSchema(
        last_customer_message="Do I have enough money in my account to get a taxi from New York to Newark?",
        most_recent_customer_inquiry_or_need=(
            "Checking customer's balance, comparing it to the price of a taxi from New York to Newark, "
            "and report the result to the customer"
        ),
        most_recent_customer_inquiry_or_need_was_already_resolved=False,
        name="ping_supervisor",
        subtleties_to_be_aware_of="no subtleties were detected",
        tool_calls_for_candidate_tool=[
            SingleToolBatchToolCallEvaluation(
                applicability_rationale="There is no reason to notify the supervisor of anything",
                is_applicable=False,
                same_call_is_already_staged=False,
                relevant_subtleties="no subtleties were detected",
                are_optional_arguments_missing=False,
                are_non_optional_arguments_missing=False,
                allowed_to_run_without_optional_arguments_even_if_they_are_missing=True,
            )
        ],
    ),
)

example_3_shot = SingleToolBatchShot(
    description=(
        "the id of the customer is 12345, and check_balance(12345) is the only staged tool call; "
        "some irrelevant reference tools exist"
    ),
    feature_set=["has_reference_tools"],
    expected_result=SingleToolBatchSchema(
        last_customer_message="Do I have enough money in my account to get a taxi from New York to Newark?",
        most_recent_customer_inquiry_or_need=(
            "Checking customer's balance, comparing it to the price of a taxi from New York to Newark, "
            "and report the result to the customer"
        ),
        most_recent_customer_inquiry_or_need_was_already_resolved=False,
        name="check_ride_price",
        subtleties_to_be_aware_of="no subtleties were detected",
        tool_calls_for_candidate_tool=[
            SingleToolBatchToolCallEvaluation(
                applicability_rationale="We need to know the price of a ride from New York to Newark to respond to the customer",
                is_applicable=True,
                argument_evaluations=[
                    SingleToolBatchArgumentEvaluation(
                        parameter_name="origin",
                        acceptable_source_for_this_argument_according_to_its_tool_definition="<INFER THIS BASED ON TOOL DEFINITION>",
                        evaluate_is_it_provided_by_an_acceptable_source="Yes, the customer mentioned New York as the origin for their ride",
                        evaluate_was_it_already_provided_and_should_it_be_provided_again="The customer already specifically provided it",
                        evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided="It would be extremely problematic, but I don't need to guess here since the customer provided it",
                        is_optional=False,
                        valid_invalid_or_missing=ValidationStatus.VALID,
                        value_as_string="New York",
                    ),
                    SingleToolBatchArgumentEvaluation(
                        parameter_name="destination",
                        acceptable_source_for_this_argument_according_to_its_tool_definition="<INFER THIS BASED ON TOOL DEFINITION>",
                        evaluate_is_it_provided_by_an_acceptable_source="Yes, the customer mentioned Newark as the destination for their ride",
                        evaluate_was_it_already_provided_and_should_it_be_provided_again="The customer already specifically provided it",
                        evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided="It would be extremely problematic, but I don't need to guess here since the customer provided it",
                        is_optional=False,
                        valid_invalid_or_missing=ValidationStatus.VALID,
                        value_as_string="Newark",
                    ),
                ],
                same_call_is_already_staged=False,
                relevant_subtleties="no subtleties were detected",
                comparison_with_rejected_tools_including_references_to_subtleties=(
                    "None of the available reference tools are deemed more suitable for the candidate tool’s application"
                ),
                a_rejected_tool_would_have_been_a_better_fit_if_it_werent_already_rejected=False,
                are_optional_arguments_missing=False,
                are_non_optional_arguments_missing=False,
                allowed_to_run_without_optional_arguments_even_if_they_are_missing=True,
            )
        ],
    ),
)

example_4_shot = SingleToolBatchShot(
    description=(
        "the candidate tool is check_calories(<product_name>): returns the number of calories in a product; "
        "one reference tool is check_stock()"
    ),
    feature_set=["has_reference_tools"],
    expected_result=SingleToolBatchSchema(
        last_customer_message="Which pizza has more calories, the classic margherita or the deep dish?",
        most_recent_customer_inquiry_or_need=(
            "Checking the number of calories in two types of pizza and replying with which one has more"
        ),
        most_recent_customer_inquiry_or_need_was_already_resolved=False,
        name="check_calories",
        subtleties_to_be_aware_of="two products need to be checked for calories - margherita and deep dish",
        tool_calls_for_candidate_tool=[
            SingleToolBatchToolCallEvaluation(
                applicability_rationale="We need to check how many calories are in the margherita pizza",
                is_applicable=True,
                argument_evaluations=[
                    SingleToolBatchArgumentEvaluation(
                        parameter_name="product_name",
                        acceptable_source_for_this_argument_according_to_its_tool_definition="<INFER THIS BASED ON TOOL DEFINITION>",
                        evaluate_is_it_provided_by_an_acceptable_source="The first product the customer specified is a margherita",
                        evaluate_was_it_already_provided_and_should_it_be_provided_again="The customer already specifically provided it",
                        evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided="It would be absurd to provide unsolicited information on some random product, but I don't need to guess here since the customer provided it",
                        is_optional=False,
                        valid_invalid_or_missing=ValidationStatus.VALID,
                        value_as_string="Margherita",
                    ),
                ],
                same_call_is_already_staged=False,
                relevant_subtleties="two products need to be checked for calories - begin with margherita",
                comparison_with_rejected_tools_including_references_to_subtleties=(
                    "None of the available reference tools are deemed more suitable for the candidate tool’s application"
                ),
                a_rejected_tool_would_have_been_a_better_fit_if_it_werent_already_rejected=False,
                are_optional_arguments_missing=False,
                are_non_optional_arguments_missing=False,
                allowed_to_run_without_optional_arguments_even_if_they_are_missing=True,
            ),
            SingleToolBatchToolCallEvaluation(
                applicability_rationale="We need to check how many calories are in the deep dish pizza",
                is_applicable=True,
                argument_evaluations=[
                    SingleToolBatchArgumentEvaluation(
                        parameter_name="product_name",
                        acceptable_source_for_this_argument_according_to_its_tool_definition="<INFER THIS BASED ON TOOL DEFINITION>",
                        evaluate_is_it_provided_by_an_acceptable_source="The second product the customer specified is the deep dish",
                        evaluate_was_it_already_provided_and_should_it_be_provided_again="The customer already specifically provided it",
                        evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided="It would be absurd to provide unsolicited information on some random product, but I don't need to guess here since the customer provided it",
                        is_optional=False,
                        valid_invalid_or_missing=ValidationStatus.VALID,
                        value_as_string="Deep Dish",
                    ),
                ],
                same_call_is_already_staged=False,
                relevant_subtleties="two products need to be checked for calories - now check deep dish",
                comparison_with_rejected_tools_including_references_to_subtleties=(
                    "None of the available reference tools are deemed more suitable for the candidate tool’s application"
                ),
                a_rejected_tool_would_have_been_a_better_fit_if_it_werent_already_rejected=False,
                are_optional_arguments_missing=False,
                are_non_optional_arguments_missing=False,
                allowed_to_run_without_optional_arguments_even_if_they_are_missing=True,
            ),
        ],
    ),
)

example_5_shot = SingleToolBatchShot(
    description=(
        "the candidate tool is check_vehicle_price(model: str), and reference tool is check_motorcycle_price(model: str)"
    ),
    feature_set=["has_reference_tools"],
    expected_result=SingleToolBatchSchema(
        last_customer_message="What's your price for a Harley-Davidson Street Glide?",
        most_recent_customer_inquiry_or_need="Checking the price of a Harley-Davidson Street Glide motorcycle",
        most_recent_customer_inquiry_or_need_was_already_resolved=False,
        name="check_motorcycle_price",
        subtleties_to_be_aware_of="Both the candidate and reference tool could apply - we need to choose the one that applies best",
        tool_calls_for_candidate_tool=[
            SingleToolBatchToolCallEvaluation(
                applicability_rationale="we need to check for the price of a specific motorcycle model",
                is_applicable=True,
                argument_evaluations=[
                    SingleToolBatchArgumentEvaluation(
                        parameter_name="model",
                        acceptable_source_for_this_argument_according_to_its_tool_definition="<INFER THIS BASED ON TOOL DEFINITION>",
                        evaluate_is_it_provided_by_an_acceptable_source="Yes; the customer asked about a specific model",
                        evaluate_was_it_already_provided_and_should_it_be_provided_again="The customer asked about a specific model",
                        evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided="It would be absurd to provide unsolicited information on some random model, but I don't need to guess here since the customer provided it",
                        is_optional=False,
                        valid_invalid_or_missing=ValidationStatus.VALID,
                        value_as_string="Harley-Davidson Street Glide",
                    )
                ],
                same_call_is_already_staged=False,
                relevant_subtleties="Both the candidate and reference tool could apply - we need to choose the one that applies best",
                comparison_with_rejected_tools_including_references_to_subtleties=(
                    "candidate tool is more specialized for this use case than the rejected tools"
                ),
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
            )
        ],
    ),
)

example_6_shot = SingleToolBatchShot(
    description=(
        "the candidate tool is check_motorcycle_price(model: str), and one reference tool is check_vehicle_price(model: str)"
    ),
    feature_set=["has_reference_tools"],
    expected_result=SingleToolBatchSchema(
        last_customer_message="What's your price for a Harley-Davidson Street Glide?",
        most_recent_customer_inquiry_or_need="Checking the price of a Harley-Davidson Street Glide motorcycle",
        most_recent_customer_inquiry_or_need_was_already_resolved=False,
        name="check_vehicle_price",
        subtleties_to_be_aware_of="no subtleties were detected",
        tool_calls_for_candidate_tool=[
            SingleToolBatchToolCallEvaluation(
                applicability_rationale="we need to check for the price of a specific vehicle - a Harley-Davidson Street Glide",
                is_applicable=True,
                argument_evaluations=[
                    SingleToolBatchArgumentEvaluation(
                        parameter_name="model",
                        acceptable_source_for_this_argument_according_to_its_tool_definition="<INFER THIS BASED ON TOOL DEFINITION>",
                        evaluate_is_it_provided_by_an_acceptable_source="Yes; the customer asked about a specific model",
                        evaluate_was_it_already_provided_and_should_it_be_provided_again="The customer asked about a specific model",
                        evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided="It would be absurd to provide unsolicited information on some random model, but I don't need to guess here since the customer provided it",
                        is_optional=False,
                        valid_invalid_or_missing=ValidationStatus.VALID,
                        value_as_string="Harley-Davidson Street Glide",
                    )
                ],
                same_call_is_already_staged=False,
                relevant_subtleties="no subtleties were detected",
                comparison_with_rejected_tools_including_references_to_subtleties="not as good a fit as check_motorcycle_price",
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
            )
        ],
    ),
)

example_7_shot = SingleToolBatchShot(
    description=(
        "the candidate tool is check_temperature(location: str), and reference tool is check_indoor_temperature(room: str)"
    ),
    feature_set=["has_reference_tools"],
    expected_result=SingleToolBatchSchema(
        last_customer_message="What's the temperature in the living room right now?",
        most_recent_customer_inquiry_or_need="Checking the current temperature in the living room",
        most_recent_customer_inquiry_or_need_was_already_resolved=False,
        name="check_temperature",
        subtleties_to_be_aware_of="no subtleties were detected",
        tool_calls_for_candidate_tool=[
            SingleToolBatchToolCallEvaluation(
                applicability_rationale="need to check the current temperature in the living room",
                is_applicable=True,
                argument_evaluations=[
                    SingleToolBatchArgumentEvaluation(
                        parameter_name="location",
                        acceptable_source_for_this_argument_according_to_its_tool_definition="<INFER THIS BASED ON TOOL DEFINITION>",
                        evaluate_is_it_provided_by_an_acceptable_source="Yes; the customer asked about the living room",
                        evaluate_was_it_already_provided_and_should_it_be_provided_again="The customer asked about a specific location",
                        evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided="It would be absurd to provide unsolicited information on some random room, but I don't need to guess here since the customer provided it",
                        is_optional=False,
                        valid_invalid_or_missing=ValidationStatus.VALID,
                        value_as_string="living room",
                    )
                ],
                same_call_is_already_staged=False,
                relevant_subtleties="no subtleties were detected",
                comparison_with_rejected_tools_including_references_to_subtleties="check_indoor_temperature is a better fit for this use case, as it's more specific",
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
            )
        ],
    ),
)


example_8_shot = SingleToolBatchShot(
    description=(
        "the candidate tool is search_product(query: str), and reference tool is "
        "search_electronics(query: str, specifications: dict)"
    ),
    feature_set=["has_reference_tools"],
    expected_result=SingleToolBatchSchema(
        last_customer_message="I'm looking for a gaming laptop with at least 16GB RAM and an RTX 3080",
        most_recent_customer_inquiry_or_need="Searching for a gaming laptop with specific technical requirements",
        most_recent_customer_inquiry_or_need_was_already_resolved=False,
        name="search_product",
        subtleties_to_be_aware_of="A gaming laptop is strictly speaking a product, but more specifically it's an electronic product",
        tool_calls_for_candidate_tool=[
            SingleToolBatchToolCallEvaluation(
                applicability_rationale="need to search for a product with specific technical requirements",
                is_applicable=True,
                argument_evaluations=[
                    SingleToolBatchArgumentEvaluation(
                        parameter_name="query",
                        acceptable_source_for_this_argument_according_to_its_tool_definition="<INFER THIS BASED ON TOOL DEFINITION>",
                        evaluate_is_it_provided_by_an_acceptable_source="Yes; the customer mentioned their specific requirements",
                        evaluate_was_it_already_provided_and_should_it_be_provided_again="The customer mentioned specific requirements, which is enough for me to construct a query",
                        evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided="It would be absurd to provide unsolicited information on some random product, but I don't need to guess here since the customer provided their requirements",
                        is_optional=False,
                        valid_invalid_or_missing=ValidationStatus.VALID,
                        value_as_string="gaming laptop, RTX 3080, 16GB RAM",
                    )
                ],
                same_call_is_already_staged=False,
                relevant_subtleties="While laptops are a kind of product, they are specifically a type of electronics product",
                comparison_with_rejected_tools_including_references_to_subtleties="not as good a fit as search_electronics",
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
            )
        ],
    ),
)


example_9_shot = SingleToolBatchShot(
    description=("the candidate tool is schedule_appointment(date: str)"),
    feature_set=[],
    expected_result=SingleToolBatchSchema(
        last_customer_message="I want to schedule an appointment please",
        most_recent_customer_inquiry_or_need="The customer wishes to schedule an appointment",
        most_recent_customer_inquiry_or_need_was_already_resolved=False,
        name="schedule_appointment",
        subtleties_to_be_aware_of="The candidate tool has a date argument",
        tool_calls_for_candidate_tool=[
            SingleToolBatchToolCallEvaluation(
                applicability_rationale="The customer specifically wants to schedule an appointment, and there are no better reference tools",
                is_applicable=True,
                argument_evaluations=[
                    SingleToolBatchArgumentEvaluation(
                        parameter_name="date",
                        acceptable_source_for_this_argument_according_to_its_tool_definition="<INFER THIS BASED ON TOOL DEFINITION>",
                        evaluate_is_it_provided_by_an_acceptable_source="No; the customer hasn't provided a date, and I cannot guess it or infer when they'd be available",
                        evaluate_was_it_already_provided_and_should_it_be_provided_again="The customer hasn't specified it yet",
                        evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided="It is very problematic to just guess when the customer would be available for an appointment",
                        is_optional=False,
                        valid_invalid_or_missing=ValidationStatus.MISSING,
                        value_as_string=None,
                    )
                ],
                same_call_is_already_staged=False,
                relevant_subtleties="This is the right tool to run, but we lack information for the date argument",
                are_optional_arguments_missing=False,
                are_non_optional_arguments_missing=False,
                allowed_to_run_without_optional_arguments_even_if_they_are_missing=True,
            )
        ],
    ),
)

example_10_shot = SingleToolBatchShot(
    description="the candidate tool is check_products_availability(products: list[str])",
    feature_set=[],
    expected_result=SingleToolBatchSchema(
        last_customer_message="Hey can I buy a laptop and a mouse please?",
        most_recent_customer_inquiry_or_need=(
            "The customer wants to purchase a laptop and a mouse and we need to check if those products are available"
        ),
        most_recent_customer_inquiry_or_need_was_already_resolved=False,
        name="check_products_availability",
        subtleties_to_be_aware_of="Before the customer can make a purchase, we need to check the availability of laptops and mice. The 'products' parameter is a list, so the tool should be called once with both products in the list.",
        tool_calls_for_candidate_tool=[
            SingleToolBatchToolCallEvaluation(
                applicability_rationale="The tool is applicable because the customer is inquiring about purchasing specific products and the tool checks the availability of a list of products.",
                is_applicable=True,
                argument_evaluations=[
                    SingleToolBatchArgumentEvaluation(
                        parameter_name="products",
                        acceptable_source_for_this_argument_according_to_its_tool_definition="<INFER THIS BASED ON TOOL DEFINITION>",
                        evaluate_is_it_provided_by_an_acceptable_source="Yes, the product names 'laptop' and 'mouse' were provided in the customer's message so should be passed as list.",
                        evaluate_was_it_already_provided_and_should_it_be_provided_again="It was provided in customer's message and should not be provided again.",
                        evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided="Yes, guessing product names can result in incorrect availability checks.",
                        is_optional=False,
                        valid_invalid_or_missing=ValidationStatus.VALID,
                        value_as_string='["laptop", "mouse"]',
                    )
                ],
                same_call_is_already_staged=False,
                relevant_subtleties="We should run this tool.",
                comparison_with_rejected_tools_including_references_to_subtleties="There are no tools in the list of rejected tools",
                a_rejected_tool_would_have_been_a_better_fit_if_it_werent_already_rejected=False,
                are_optional_arguments_missing=False,
                are_non_optional_arguments_missing=False,
                allowed_to_run_without_optional_arguments_even_if_they_are_missing=True,
            )
        ],
    ),
)

example_11_shot = SingleToolBatchShot(
    description="the candidate tool is book_flight(passenger_name: str, origin: str, destination: str, departure_date: str, return_date:str)",
    feature_set=[],
    expected_result=SingleToolBatchSchema(
        last_customer_message="Hey can I book a flight to Bangkok?",
        most_recent_customer_inquiry_or_need=("The customer wants to book a flight to Bangkok"),
        most_recent_customer_inquiry_or_need_was_already_resolved=False,
        name="book_flight",
        subtleties_to_be_aware_of="The customer clearly wants to book a flight but has not provided many of the required details for booking like origin anf departure date.",
        tool_calls_for_candidate_tool=[
            SingleToolBatchToolCallEvaluation(
                applicability_rationale="The customer explicitly asked to book a flight and mentioned the destination. Although multiple required details are missing, the customer's intent is clear, so this tool should be applied.",
                is_applicable=True,
                argument_evaluations=[
                    SingleToolBatchArgumentEvaluation(
                        parameter_name="passenger_name",
                        acceptable_source_for_this_argument_according_to_its_tool_definition="<INFER THIS BASED ON TOOL DEFINITION>",
                        evaluate_is_it_provided_by_an_acceptable_source="No, the customer has not provided a name and there is no prior context.",
                        evaluate_was_it_already_provided_and_should_it_be_provided_again="It has not been provided.",
                        evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided="Yes, using an incorrect or placeholder name could result in booking errors.",
                        is_optional=False,
                        valid_invalid_or_missing=ValidationStatus.MISSING,
                        value_as_string=None,
                    ),
                    SingleToolBatchArgumentEvaluation(
                        parameter_name="origin",
                        acceptable_source_for_this_argument_according_to_its_tool_definition="<INFER THIS BASED ON TOOL DEFINITION>",
                        evaluate_is_it_provided_by_an_acceptable_source="No, the customer did not mention the departure location.",
                        evaluate_was_it_already_provided_and_should_it_be_provided_again="It has not been provided.",
                        evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided="Yes, guessing the origin can result in incorrect flight details.",
                        is_optional=False,
                        valid_invalid_or_missing=ValidationStatus.MISSING,
                        value_as_string=None,
                    ),
                    SingleToolBatchArgumentEvaluation(
                        parameter_name="destination",
                        acceptable_source_for_this_argument_according_to_its_tool_definition="<INFER THIS BASED ON TOOL DEFINITION>",
                        evaluate_is_it_provided_by_an_acceptable_source="Yes, the customer specifically mentioned Bangkok.",
                        evaluate_was_it_already_provided_and_should_it_be_provided_again="Yes, it was included in the customer's message and should not be asked again.",
                        evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided="Yes, guessing the destination could lead to incorrect booking",
                        is_optional=False,
                        valid_invalid_or_missing=ValidationStatus.VALID,
                        value_as_string="Bangkok",
                    ),
                    SingleToolBatchArgumentEvaluation(
                        parameter_name="departure_date",
                        acceptable_source_for_this_argument_according_to_its_tool_definition="<INFER THIS BASED ON TOOL DEFINITION>",
                        evaluate_is_it_provided_by_an_acceptable_source="No, the customer did not mention a departure date.",
                        evaluate_was_it_already_provided_and_should_it_be_provided_again="It has not been provided.",
                        evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided="Yes, guessing a date could lead to incorrect or undesired bookings.",
                        is_optional=False,
                        valid_invalid_or_missing=ValidationStatus.MISSING,
                        value_as_string=None,
                    ),
                    SingleToolBatchArgumentEvaluation(
                        parameter_name="return_date",
                        acceptable_source_for_this_argument_according_to_its_tool_definition="<INFER THIS BASED ON TOOL DEFINITION>",
                        evaluate_is_it_provided_by_an_acceptable_source="No, the customer did not mention a return date.",
                        evaluate_was_it_already_provided_and_should_it_be_provided_again="It has not been provided.",
                        evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided="Yes, assuming a return date can misrepresent the customer's intent",
                        is_optional=False,
                        valid_invalid_or_missing=ValidationStatus.MISSING,
                        value_as_string=None,
                    ),
                ],
                same_call_is_already_staged=False,
                relevant_subtleties="We should run this tool as it aligns with customer's inquiry while requesting the necessary missing booking information.",
                are_optional_arguments_missing=False,
                are_non_optional_arguments_missing=True,
                allowed_to_run_without_optional_arguments_even_if_they_are_missing=True,
            )
        ],
    ),
)

example_12_shot = SingleToolBatchShot(
    description=(
        "the candidate tool is book_flight(origin:str, destination: str) and there are no better reference tools, origin and destination are enum that can get only these values: 'New York', 'London', 'Paris'."
        "the customer wants to book a flight from Tel-Aviv to Singapore."
    ),
    feature_set=[],
    expected_result=SingleToolBatchSchema(
        last_customer_message="I want to book a flight from Tel-Aviv to Singapore",
        most_recent_customer_inquiry_or_need="The customer want to book a flight",
        most_recent_customer_inquiry_or_need_was_already_resolved=False,
        name="book_flight",
        subtleties_to_be_aware_of="The customer specified a flight origin and destination that may be invalid in the schema's enum, but their values are still important and should be filled in the output",
        tool_calls_for_candidate_tool=[
            SingleToolBatchToolCallEvaluation(
                applicability_rationale="The customer specifically wants to book a flight and provided the origin and destination, and there are no better reference tools",
                is_applicable=True,
                argument_evaluations=[
                    SingleToolBatchArgumentEvaluation(
                        parameter_name="origin",
                        acceptable_source_for_this_argument_according_to_its_tool_definition="<INFER THIS BASED ON TOOL DEFINITION>",
                        evaluate_is_it_provided_by_an_acceptable_source="Yes; the customer has explicitly provided an origin, which is an acceptable source but not in the enum, so regardless of validity considerations its value is extracted into the relevant field",
                        evaluate_was_it_already_provided_and_should_it_be_provided_again="Yes, the customer has explicitly provided an origin, so it should be extracted and filled into the matching output field even if not a valid enum value",
                        evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided="It is very problematic to guess the origin the customer wants to fly from",
                        is_optional=False,
                        valid_invalid_or_missing=ValidationStatus.INVALID,
                        value_as_string="Tel-Aviv",
                    ),
                    SingleToolBatchArgumentEvaluation(
                        parameter_name="destination",
                        acceptable_source_for_this_argument_according_to_its_tool_definition="<INFER THIS BASED ON TOOL DEFINITION>",
                        evaluate_is_it_provided_by_an_acceptable_source="Yes; the customer has explicitly provided a destination, which is an acceptable source but not in the enum, so regardless of validity considerations its value is extracted into the relevant field",
                        evaluate_was_it_already_provided_and_should_it_be_provided_again="Yes, the customer has explicitly provided a destination, so it should be extracted and filled into the matching output field even if not a valid enum value",
                        evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided="It is very problematic to guess the destination the customer wants to fly to",
                        is_optional=False,
                        valid_invalid_or_missing=ValidationStatus.INVALID,
                        value_as_string="Singapore",
                    ),
                ],
                same_call_is_already_staged=False,
                relevant_subtleties="This is the right tool to run although a parameter may be invalid. This parameter value, however, still needs to be extracted from the customer's message and provided in the output",
                comparison_with_rejected_tools_including_references_to_subtleties="There are no tools in the list of rejected tools",
                a_rejected_tool_would_have_been_a_better_fit_if_it_werent_already_rejected=False,
                are_optional_arguments_missing=False,
                are_non_optional_arguments_missing=False,
                allowed_to_run_without_optional_arguments_even_if_they_are_missing=True,
            )
        ],
    ),
)

_baseline_consequential_shots: Sequence[SingleToolBatchShot] = [
    example_1_shot,
    example_2_shot,
    example_3_shot,
    example_4_shot,
    example_5_shot,
    example_6_shot,
    example_7_shot,
    example_8_shot,
    example_9_shot,
    example_10_shot,
    example_11_shot,
    example_12_shot,
]

_baseline_non_consequential_shots: Sequence[NonConsequentialSingleToolBatchShot] = [
    NonConsequentialSingleToolBatchShot(
        description="Tool should be called",
        expected_result=NonConsequentialToolBatchSchema(
            name="get_weather",
            should_run=True,
            reasoning_tldr="CASE 1. The user asked about the weather in Paris. There is a required argument which is Paris",
            calls=[NonConsequentialToolCallEvaluation(args={"city": "Paris"})],
        ),
    ),
    NonConsequentialSingleToolBatchShot(
        description="Tool should NOT be called",
        expected_result=NonConsequentialToolBatchSchema(
            name="get_weather",
            should_run=False,
            reasoning_tldr="CASE 4. The user only greeted me and did not inquire about the weather",
            calls=[],
        ),
    ),
    NonConsequentialSingleToolBatchShot(
        description="Multiple calls needed",
        expected_result=NonConsequentialToolBatchSchema(
            name="get_weather",
            should_run=True,
            reasoning_tldr="The user asked to compare the weather in Paris and London - all required params are available for all calls. CASE 1 for both calls.",
            calls=[
                NonConsequentialToolCallEvaluation(args={"city": "Paris"}),
                NonConsequentialToolCallEvaluation(args={"city": "London"}),
            ],
        ),
    ),
    NonConsequentialSingleToolBatchShot(
        description="Missing required parameter but in principle the tool should run",
        expected_result=NonConsequentialToolBatchSchema(
            name="get_weather",
            should_run=True,
            reasoning_tldr="CASE 3: The user asked about the weather but did not specify a city and I don't know their location. Should mark the required argument 'city' as missing",
            calls=[
                NonConsequentialToolCallEvaluation(args={"city": "<<__missing__>>"}),
            ],
        ),
    ),
    NonConsequentialSingleToolBatchShot(
        description="Tool call is already staged, so should not run",
        expected_result=NonConsequentialToolBatchSchema(
            name="get_weather",
            should_run=False,
            reasoning_tldr="CASE 2. The user asked to compare the weather in Paris and London - all required params are available but ALL of those calls are already staged",
            calls=[],
        ),
    ),
    NonConsequentialSingleToolBatchShot(
        description="The tool - get_weather(city : str, temp_unit: Optional[str]). temp_unit is missing",
        expected_result=NonConsequentialToolBatchSchema(
            name="get_weather",
            should_run=True,
            reasoning_tldr="The user asked about the weather in Tel Aviv. The parameter 'city' is required and available. 'temp_unit' is missing but it is an optional argument, so CASE 1 holds for this call",
            calls=[
                NonConsequentialToolCallEvaluation(
                    args={
                        "city": "Tel Aviv",
                        "temp_unit": None,
                    }
                ),
            ],
        ),
    ),
]

consequential_shot_collection = ShotCollection[SingleToolBatchShot](_baseline_consequential_shots)
non_consequential_shot_collection = ShotCollection[NonConsequentialSingleToolBatchShot](
    _baseline_non_consequential_shots
)
