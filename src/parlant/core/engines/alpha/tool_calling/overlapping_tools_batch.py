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

import ast
from enum import Enum
import json
import traceback
from typing import Any, Optional, Sequence
from parlant.core.agents import Agent
from parlant.core.common import DefaultBaseModel, generate_id
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.emissions import EmittedEvent
from parlant.core.engines.alpha.guideline_matching.generic.common import internal_representation
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.alpha.optimization_policy import OptimizationPolicy
from parlant.core.engines.alpha.prompt_builder import BuiltInSection, PromptBuilder, SectionStatus
from parlant.core.glossary import Term
from parlant.core.journeys import Journey
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.sessions import Event, EventKind
from parlant.core.shots import Shot, ShotCollection
from dataclasses import dataclass
from parlant.core.engines.alpha.tool_calling.tool_caller import (
    ToolCallEvaluation,
    InvalidToolData,
    MissingToolData,
    ToolCall,
    ToolCallBatch,
    ToolCallBatchError,
    ToolCallBatchResult,
    ToolCallContext,
    ToolCallId,
    ToolInsights,
)
from parlant.core.tools import Tool, ToolId, ToolParameterDescriptor, ToolParameterOptions


class ValidationStatus(Enum):
    VALID = "valid"
    INVALID = "invalid"
    MISSING = "missing"


class OverlappingToolsBatchArgumentEvaluation(DefaultBaseModel):
    parameter_name: str
    acceptable_source_for_this_argument_according_to_its_tool_definition: str
    evaluate_is_it_provided_by_an_acceptable_source: str
    evaluate_was_it_already_provided_and_should_it_be_provided_again: str
    evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided: str
    is_optional: Optional[bool] = False
    has_default_value_if_not_provided_by_acceptable_source: Optional[bool] = None
    valid_invalid_or_missing: ValidationStatus
    value_as_string: Optional[str] = None


class OverlappingToolsBatchToolCallEvaluation(DefaultBaseModel):
    argument_evaluations: Optional[list[OverlappingToolsBatchArgumentEvaluation]] = None
    same_call_is_already_staged: bool


class OverlappingToolsBatchToolEvaluation(DefaultBaseModel):
    name: str
    subtleties_to_be_aware_of: str
    applicability_rationale: str
    potentially_alternative_tools: str
    comparison_with_alternative_tools_including_references_to_subtleties: str
    alternative_tool_should_run_instead_of_this_tool: bool
    is_applicable: bool
    calls: Optional[list[OverlappingToolsBatchToolCallEvaluation]] = None


class OverlappingToolsBatchSchema(DefaultBaseModel):
    last_customer_message: Optional[str] = None
    most_recent_customer_inquiry_or_need: Optional[str] = None
    most_recent_customer_inquiry_or_need_was_already_resolved: Optional[bool] = None
    tools_evaluation: list[OverlappingToolsBatchToolEvaluation]


@dataclass
class OverlappingToolsBatchShot(Shot):
    expected_result: OverlappingToolsBatchSchema


class OverlappingToolsBatch(ToolCallBatch):
    def __init__(
        self,
        logger: Logger,
        optimization_policy: OptimizationPolicy,
        service_registry: ServiceRegistry,
        schematic_generator: SchematicGenerator[OverlappingToolsBatchSchema],
        overlapping_tools_batch: Sequence[tuple[ToolId, Tool, Sequence[GuidelineMatch]]],
        context: ToolCallContext,
    ) -> None:
        self._logger = logger
        self._optimization_policy = optimization_policy
        self._service_registry = service_registry
        self._schematic_generator = schematic_generator
        self._context = context
        self._overlapping_tools_batch = overlapping_tools_batch

    async def process(self) -> ToolCallBatchResult:
        with self._logger.operation("OverlappingToolsBatch"):
            (
                generation_info,
                inference_output,
                evaluations,
                missing_data,
                invalid_data,
            ) = await self._infer_calls_for_overlapping_tools(
                agent=self._context.agent,
                context_variables=self._context.context_variables,
                interaction_history=self._context.interaction_history,
                terms=self._context.terms,
                ordinary_guideline_matches=self._context.ordinary_guideline_matches,
                journeys=self._context.journeys,
                overlapping_tools_batch=self._overlapping_tools_batch,
                staged_events=self._context.staged_events,
            )
            return ToolCallBatchResult(
                generation_info=generation_info,
                tool_calls=inference_output,
                insights=ToolInsights(
                    evaluations=evaluations,
                    missing_data=missing_data,
                    invalid_data=invalid_data,
                ),
            )

    async def _infer_calls_for_overlapping_tools(
        self,
        agent: Agent,
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        ordinary_guideline_matches: Sequence[GuidelineMatch],
        journeys: Sequence[Journey],
        overlapping_tools_batch: Sequence[tuple[ToolId, Tool, Sequence[GuidelineMatch]]],
        staged_events: Sequence[EmittedEvent],
    ) -> tuple[
        GenerationInfo,
        list[ToolCall],
        list[tuple[ToolId, ToolCallEvaluation]],
        list[MissingToolData],
        list[InvalidToolData],
    ]:
        inference_prompt = self._build_tool_call_inference_prompt(
            agent,
            context_variables,
            interaction_history,
            terms,
            ordinary_guideline_matches,
            journeys,
            overlapping_tools_batch,
            staged_events,
            await self.shots(),
        )

        generation_attempt_temperatures = (
            self._optimization_policy.get_tool_calling_batch_retry_temperatures()
        )

        last_generation_exception: Exception | None = None

        for generation_attempt in range(3):
            try:
                # Send the tool call inference prompt to the LLM
                generation_info, inference_output = await self._run_inference(
                    prompt=inference_prompt,
                    temperature=generation_attempt_temperatures[generation_attempt],
                )

                # Evaluate the tool calls
                (
                    tool_calls,
                    evaluations,
                    missing_data,
                    invalid_data,
                ) = await self._evaluate_tool_calls_parameters(
                    inference_output, overlapping_tools_batch
                )

                return generation_info, tool_calls, evaluations, missing_data, invalid_data

            except Exception as exc:
                self._logger.warning(
                    f"OverlappingToolBatch attempt {generation_attempt} failed: {traceback.format_exception(exc)}"
                )

                last_generation_exception = exc

        raise ToolCallBatchError() from last_generation_exception

    def _get_tool_descriptor(
        self,
        tool_name: str,
        overlapping_tools_batch: Sequence[tuple[ToolId, Tool, Sequence[GuidelineMatch]]],
    ) -> Optional[tuple[ToolId, Tool]]:
        for tool_id, tool, _ in overlapping_tools_batch:
            if tool_name == tool_id.to_string():
                return tool_id, tool
        return None

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

    async def _evaluate_tool_calls_parameters(
        self,
        inference_output: Sequence[OverlappingToolsBatchToolEvaluation],
        overlapping_tools_batch: Sequence[tuple[ToolId, Tool, Sequence[GuidelineMatch]]],
    ) -> tuple[
        list[ToolCall],
        list[tuple[ToolId, ToolCallEvaluation]],
        list[MissingToolData],
        list[InvalidToolData],
    ]:
        tool_calls = []
        evaluations: list[tuple[ToolId, ToolCallEvaluation]] = []  # FIXME: handle evaluations
        missing_data = []
        invalid_data = []

        for tool_inference in inference_output:
            tool_name = tool_inference.name
            result = self._get_tool_descriptor(tool_name, overlapping_tools_batch)
            # First - check validity of all parameters with provided values
            if (
                result
                and tool_inference.is_applicable
                and not tool_inference.alternative_tool_should_run_instead_of_this_tool
                and tool_inference.calls
            ):
                tool_id, tool = result
                for tc in tool_inference.calls:
                    all_values_valid = True
                    for evaluation in tc.argument_evaluations or []:
                        tool_id, tool = result
                        descriptor, options = tool.parameters[evaluation.parameter_name]

                        if evaluation.value_as_string and not await self._validate_argument_value(
                            tool.parameters[evaluation.parameter_name],
                            evaluation.value_as_string,
                        ):
                            all_values_valid = False
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

                for tc in tool_inference.calls:
                    if not tc.same_call_is_already_staged:
                        if all(
                            not evaluation.valid_invalid_or_missing == ValidationStatus.MISSING
                            for evaluation in tc.argument_evaluations or []
                            if evaluation.parameter_name in tool.required
                        ):
                            self._logger.debug(
                                f"Inference::Completion::Activated: {tool_id.to_string()}\n{tc.model_dump_json(indent=2)}"
                            )

                            arguments = {}

                            if tool.parameters:  # We check this because sometimes LLMs hallucinate placeholders for no-param tools
                                for evaluation in tc.argument_evaluations or []:
                                    if (
                                        evaluation.valid_invalid_or_missing
                                        == ValidationStatus.MISSING
                                    ):
                                        continue

                                    # Note that if LLM provided 'None' for a required parameter with a default - it will get 'None' as value
                                    arguments[evaluation.parameter_name] = (
                                        evaluation.value_as_string
                                    )
                            if all_values_valid:
                                tool_calls.append(
                                    ToolCall(
                                        id=ToolCallId(generate_id()),
                                        tool_id=tool_id,
                                        arguments=arguments,
                                    )
                                )
                        else:
                            for evaluation in tc.argument_evaluations or []:
                                if evaluation.parameter_name not in tool.parameters:
                                    self._logger.error(
                                        f"Inference::Completion: Argument {evaluation.parameter_name} not found in tool parameters"
                                    )
                                    continue

                                tool_descriptor, tool_options = tool.parameters[
                                    evaluation.parameter_name
                                ]

                                if (
                                    evaluation.valid_invalid_or_missing == ValidationStatus.MISSING
                                    and not evaluation.is_optional
                                    and not tool_options.hidden
                                ):
                                    missing_data.append(
                                        MissingToolData(
                                            parameter=tool_options.display_name
                                            or evaluation.parameter_name,
                                            significance=tool_options.significance,
                                            description=tool_descriptor.get("description"),
                                            precedence=tool_options.precedence,
                                        )
                                    )

                    else:
                        self._logger.debug(
                            f"Inference::Completion::Skipped: {tool_id.to_string()}\n{tc.model_dump_json(indent=2)}"
                        )

        return tool_calls, evaluations, missing_data, invalid_data

    async def shots(self) -> Sequence[OverlappingToolsBatchShot]:
        return await shot_collection.list()

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

    def _format_shots(
        self,
        shots: Sequence[OverlappingToolsBatchShot],
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
        shot: OverlappingToolsBatchShot,
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
        ordinary_guideline_matches: Sequence[GuidelineMatch],
        journeys: Sequence[Journey],
        batch: Sequence[tuple[ToolId, Tool, Sequence[GuidelineMatch]]],
        staged_events: Sequence[EmittedEvent],
        shots: Sequence[OverlappingToolsBatchShot],
    ) -> PromptBuilder:
        staged_calls = self._get_staged_calls(staged_events)

        builder = PromptBuilder(on_build=lambda prompt: self._logger.trace(f"Prompt:\n{prompt}"))

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
Your task is to review a batch of provided tools and, based on your most recent interaction with the customer, decide whether to use them.
The provided tools have been grouped together due to overlapping functionality or shared parameters.
You should prefer to run the tool that is the best fit for the current context. Specifically, the one that is most relevant and tailored to the user's inquiry or need.
For each tool in the batch, indicate the tool applicability with a boolean value: true if the tool is useful at this point, or false if it is not.
For any tool marked as true, include the available arguments for activation.
Note that a tool may be considered applicable even if not all of its required arguments are available. In such cases, provide the parameters that are currently available,
following the format specified in its description.

While doing so, take the following instructions into account:

1. You may suggest tool that don’t directly address the customer’s latest interaction but can advance the conversation to a more useful state based on function definitions.
2. You may choose to call more than one tool, specifically when handling more than one requirement.
3. Each tool may be called multiple times with different arguments.
4. Avoid calling a tool with the SAME arguments more than once, unless clearly justified by the interaction.
5. Ensure each tool call relies only on the immediate context and staged calls, without requiring other tools not yet invoked, to avoid dependencies.
6. If a tool needs to be applied multiple times (each with different arguments), you may include it in the output multiple times.
7. When multiple tools can perform the same task and yield the same result, avoid selecting more than one. Choose the tool that best matches the user's specific request.

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
            template=self._add_guideline_matches_section(
                ordinary_guideline_matches,
                batch,
            ),
            props={
                "ordinary_guideline_matches": ordinary_guideline_matches,
                "batch": batch,
            },
        )
        tool_descriptors = [(tool_descriptor[0], tool_descriptor[1]) for tool_descriptor in batch]
        tool_definitions_template, tool_definitions_props = self._add_tool_definitions_section(
            tool_descriptors=tool_descriptors,
        )
        builder.add_section(
            name="tool-caller-tool-definitions",
            template=tool_definitions_template,
            props={
                **tool_definitions_props,
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
Given these tools, your output should adhere to the following format:
```json
{{
    "last_customer_message": "<REPEAT THE LAST USER MESSAGE IN THE INTERACTION>",
    "most_recent_customer_inquiry_or_need": "<CUSTOMER'S INQUIRY OR NEED>",
    "most_recent_customer_inquiry_or_need_was_already_resolved": <BOOL>,
    "tools_evaluation": [
        {{
            "name": "<TOOL NAME>",
            "subtleties_to_be_aware_of": "<NOTE ANY SIGNIFICANT SUBTLETIES TO BE AWARE OF WHEN RUNNING THIS TOOL IN OUR AGENT'S CONTEXT>",
            "applicability_rationale": "<A FEW WORDS THAT EXPLAIN WHETHER, HOW, AND TO WHAT EXTENT THE TOOL NEEDS TO BE CALLED AT THIS POINT>",
            "potentially_alternative_tools": "<NAME(S) OF THE TOOL(S) IF ANY THAT CAN ALTERNATIVELY BE RUN INSTEAD OF THIS TOOL>",
            "comparison_with_alternative_tools_including_references_to_subtleties": "<A VERY BRIEF OVERVIEW OF HOW THIS CALL FARES AGAINST THE ALTERNATIVE TOOLS IN APPLICABILITY>",
            "alternative_tool_should_run_instead_of_this_tool": <BOOL>,
            "is_applicable": <BOOL>,
            "calls": [
                {{
                    "argument_evaluations": [
                        {{
                            "parameter_name": "<PARAMETER NAME>",
                            "acceptable_source_for_this_argument_according_to_its_tool_definition": "<REPEAT THE ACCEPTABLE SOURCE FOR THE ARGUMENT FROM TOOL DEFINITION>",
                            "evaluate_is_it_provided_by_an_acceptable_source": "<BRIEFLY EVALUATE IF THE SOURCE FOR THE VALUE MATCHES THE ACCEPTABLE SOURCE>",
                            "evaluate_was_it_already_provided_and_should_it_be_provided_again": "<BRIEFLY EVALUATE IF THE PARAMETER VALUE WAS PROVIDED AND SHOULD BE PROVIDED AGAIN>",
                            "evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided": "<BRIEFLY EVALUATE IF IT'S A PROBLEM TO GUESS THE VALUE>",
                            "is_optional": <BOOL>,
                            "valid_invalid_or_missing" : "<STR: EITHER 'missing', 'invalid' OR 'valid' DEPENDING IF THE VALUE IS MISSING, PROVIDED BUT NOT FOUND IN ENUM LIST, OR PROVIDED AND FOUND IN ENUM LIST (OR DOESN'T HAVE ENUM LIST)>"
                            "value_as_string": "<PARAMETER VALUE>"
                        }},
                        ...
                    ],
                    "same_call_is_already_staged": <BOOL>,
                }},
                ...
            ],
        }},
        ...
    ]

}}
```

You need to have tools_evaluation for each tool in the tools batch. Also, note that you may choose to have multiple entries in 'calls' if you wish to call the tool multiple times with different arguments.
""",
            props={},
        )
        return builder

    def _add_tool_definitions_section(
        self,
        tool_descriptors: Sequence[tuple[ToolId, Tool]],
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

        tools_specs = [_get_tool_spec(tool_id, tool) for tool_id, tool in tool_descriptors]
        return (
            """
You are provided with multiple tools, which are active candidates for execution.
Your task is to evaluate the necessity and applicability of each tool. Note that those tools may serve similar purpose. if so, choose the one that best fits the current context.
Use the more specific tool when it fits. Fallback to the generic tool when the specific one doesn't apply.
In certain cases it will be necessary to use more than one tool. However, if you decide to activate multiple tools that address the SAME PURPOSE or produce SIMILAR RESULTS,
you must provide a clear and strong justification for doing so.


Tools: ###
{tools_specs}
###

""",
            {
                "tools_specs": tools_specs,
            },
        )

    def _add_guideline_matches_section(
        self,
        ordinary_guideline_matches: Sequence[GuidelineMatch],
        tools_propositions: Sequence[tuple[ToolId, Tool, Sequence[GuidelineMatch]]],
    ) -> str:
        ordinary_guidelines_list = ""
        if ordinary_guideline_matches:
            ordinary_guidelines_list = "\n".join(
                [
                    f"{i}) When {internal_representation(p.guideline).condition}, then {internal_representation(p.guideline).action}"
                    for i, p in enumerate(ordinary_guideline_matches, start=1)
                    if internal_representation(p.guideline).action
                ]
            )

        if tools_propositions:
            tools_guidelines: list[str] = []
            for id, _, guidelines in tools_propositions:
                tool_guidelines: list[str] = []
                for i, p in enumerate(guidelines, start=1):
                    if internal_representation(p.guideline).action:
                        guideline = f"{i}) When {internal_representation(p.guideline).condition}, then {internal_representation(p.guideline).action}"
                        tool_guidelines.append(guideline)
                if tool_guidelines:
                    tools_guidelines.append("\n".join(tool_guidelines))
                    tools_guidelines.append(f"Associated Tool: {id.service_name}:{id.tool_name}")
            tools_guidelines_list = "\n".join(tools_guidelines)
        guidelines_list = ordinary_guidelines_list + tools_guidelines_list
        return f"""
GUIDELINES
---------------------
The following guidelines have been identified as relevant to the current state of interaction with the customer.
Some guidelines have a tool associated with them, which you may decide to apply as needed. Use these guidelines to understand the context for the provided tools.

Guidelines:
###
{guidelines_list}
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

    async def _run_inference(
        self,
        prompt: PromptBuilder,
        temperature: float,
    ) -> tuple[GenerationInfo, Sequence[OverlappingToolsBatchToolEvaluation]]:
        inference = await self._schematic_generator.generate(
            prompt=prompt,
            hints={"temperature": temperature},
        )

        self._logger.trace(f"Inference::Completion:\n{inference.content.model_dump_json(indent=2)}")

        return inference.info, inference.content.tools_evaluation


example_1_shot = OverlappingToolsBatchShot(
    description=(
        "the candidate tools are check_vehicle_price(model: str), check_motorcycle_price(model: str)"
    ),
    expected_result=OverlappingToolsBatchSchema(
        last_customer_message="What's your price for a Harley-Davidson Street Glide?",
        most_recent_customer_inquiry_or_need="Checking the price of a Harley-Davidson Street Glide motorcycle",
        most_recent_customer_inquiry_or_need_was_already_resolved=False,
        tools_evaluation=[
            OverlappingToolsBatchToolEvaluation(
                name="check_vehicle_price",
                subtleties_to_be_aware_of="Harley-Davidson Street Glide is a vehicle, but more specifically a motorcycle. "
                "While this general vehicle pricing tool could apply, it is less tailored to the specific type of vehicle.",
                applicability_rationale="we need to check for the price of a specific motorcycle model",
                potentially_alternative_tools="check_motorcycle_price",
                comparison_with_alternative_tools_including_references_to_subtleties="Harley-Davidson Street Glide is a vehicle and specifically a motorcycle."
                " check_motorcycle_price is specifically designed for that category. Choosing the more specific tool ensures better alignment with the product type.",
                alternative_tool_should_run_instead_of_this_tool=True,
                is_applicable=False,
            ),
            OverlappingToolsBatchToolEvaluation(
                name="check_motorcycle_price",
                subtleties_to_be_aware_of="Harley-Davidson Street Glide is a type of motorcycle.",
                applicability_rationale="we need to check for the price of a specific motorcycle model",
                potentially_alternative_tools="check_vehicle_price",
                comparison_with_alternative_tools_including_references_to_subtleties="This tool is specifically intended for motorcycle models, which makes it a more suitable choice "
                "than a general vehicle pricing tool. Specificity to product type improves accuracy and relevance.",
                alternative_tool_should_run_instead_of_this_tool=False,
                is_applicable=True,
                calls=[
                    OverlappingToolsBatchToolCallEvaluation(
                        argument_evaluations=[
                            OverlappingToolsBatchArgumentEvaluation(
                                parameter_name="model",
                                acceptable_source_for_this_argument_according_to_its_tool_definition="<INFER THIS BASED ON TOOL DEFINITION>",
                                evaluate_is_it_provided_by_an_acceptable_source="Yes; the customer asked about a specific model",
                                evaluate_was_it_already_provided_and_should_it_be_provided_again="The customer asked about a specific model",
                                evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided="It would be absurd to provide unsolicited information on some random model, but I don't need to guess here since the customer provided it",
                                valid_invalid_or_missing=ValidationStatus.VALID,
                                is_optional=False,
                                value_as_string="Harley-Davidson Street Glide",
                            )
                        ],
                        same_call_is_already_staged=False,
                    )
                ],
            ),
        ],
    ),
)


example_2_shot = OverlappingToolsBatchShot(
    description=(
        "the candidate tools are check_temperature(location: str), check_indoor_temperature(room: str)"
    ),
    expected_result=OverlappingToolsBatchSchema(
        last_customer_message="What's the temperatures in the living room right now? And what is the temperature outside?",
        most_recent_customer_inquiry_or_need="Checking the current temperature in the living room and outside",
        most_recent_customer_inquiry_or_need_was_already_resolved=False,
        tools_evaluation=[
            OverlappingToolsBatchToolEvaluation(
                name="check_temperature",
                subtleties_to_be_aware_of="The user is asking about both indoor and outdoor temperatures. "
                "This tool is suitable for checking outdoor temperature, but for indoor queries like the living room, a more specific tool exists and should be used instead.",
                applicability_rationale="The user is asking for the temperature outside, which matches the purpose of this tool.",
                potentially_alternative_tools="check_indoor_temperature",
                comparison_with_alternative_tools_including_references_to_subtleties="This tool is suitable for general or outdoor temperature queries. "
                "While it could technically be used for any temperature check, it's better to use a tool specifically designed for indoor readings—like check_indoor_temperature for the living room.",
                alternative_tool_should_run_instead_of_this_tool=False,
                is_applicable=True,
                calls=[
                    OverlappingToolsBatchToolCallEvaluation(
                        argument_evaluations=[
                            OverlappingToolsBatchArgumentEvaluation(
                                parameter_name="location",
                                acceptable_source_for_this_argument_according_to_its_tool_definition="<INFER THIS BASED ON TOOL DEFINITION>",
                                evaluate_is_it_provided_by_an_acceptable_source="Yes, the user asked about the temperature outside, which implies a general outdoor location (e.g., 'outside')",
                                evaluate_was_it_already_provided_and_should_it_be_provided_again="The customer asked about a specific location",
                                evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided="It would be absurd to provide information on some random place, but I don't need to guess here since the customer provided it",
                                valid_invalid_or_missing=ValidationStatus.VALID,
                                is_optional=False,
                                value_as_string="outside",
                            )
                        ],
                        same_call_is_already_staged=False,
                    )
                ],
            ),
            OverlappingToolsBatchToolEvaluation(
                name="check_indoor_temperature",
                subtleties_to_be_aware_of="The user is asking about both indoor and outdoor temperatures. "
                "While a general temperature tool exists, this tool is specifically designed for indoor use and should be preferred for queries like the living room.",
                applicability_rationale="We need to check the temperature in an indoor room—the living room.",
                potentially_alternative_tools="check_indoor_temperature",
                comparison_with_alternative_tools_including_references_to_subtleties="We are checking the temperature in the living room, which is an indoor space. "
                "Even though check_temperature could return a value, check_indoor_temperature is more specific and should be used when available. "
                "For the outdoor part of the question, we already used check_temperature.",
                alternative_tool_should_run_instead_of_this_tool=False,
                is_applicable=True,
                calls=[
                    OverlappingToolsBatchToolCallEvaluation(
                        argument_evaluations=[
                            OverlappingToolsBatchArgumentEvaluation(
                                parameter_name="location",
                                acceptable_source_for_this_argument_according_to_its_tool_definition="<INFER THIS BASED ON TOOL DEFINITION>",
                                evaluate_is_it_provided_by_an_acceptable_source="Yes; the customer asked about a specific location",
                                evaluate_was_it_already_provided_and_should_it_be_provided_again="The customer asked about a specific location",
                                evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided="It would be absurd to provide unsolicited information on some random room, but I don't need to guess here since the customer provided it",
                                valid_invalid_or_missing=ValidationStatus.VALID,
                                is_optional=False,
                                value_as_string="living room",
                            )
                        ],
                        same_call_is_already_staged=False,
                    )
                ],
            ),
        ],
    ),
)


_baseline_shots: Sequence[OverlappingToolsBatchShot] = [
    example_1_shot,
    example_2_shot,
]


shot_collection = ShotCollection[OverlappingToolsBatchShot](_baseline_shots)
