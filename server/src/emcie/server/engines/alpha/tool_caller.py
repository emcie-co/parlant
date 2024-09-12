import asyncio
from dataclasses import dataclass, asdict
from itertools import chain
import json
import traceback
from typing import Any, Mapping, NewType, Optional, Sequence


from emcie.common.tools import Tool, ToolContext
from emcie.server.core.agents import Agent
from emcie.server.core.common import JSONSerializable, generate_id
from emcie.server.core.context_variables import ContextVariable, ContextVariableValue
from emcie.server.core.sessions import Event, ToolResult
from emcie.server.core.tools import ToolService
from emcie.server.core.terminology import Term
from emcie.server.engines.alpha.guideline_proposition import GuidelineProposition
from emcie.server.engines.alpha.prompt_builder import PromptBuilder
from emcie.server.engines.alpha.utils import emitted_tool_events_to_dicts
from emcie.server.engines.event_emitter import EmittedEvent
from emcie.server.llm.schematic_generators import SchematicGenerator
from emcie.server.base_models import DefaultBaseModel
from emcie.server.logger import Logger

ToolCallId = NewType("ToolCallId", str)
ToolResultId = NewType("ToolResultId", str)


class ToolCallEvaluation(DefaultBaseModel):
    name: str
    rationale: str
    applicability_score: int
    should_run: bool
    arguments: Mapping[str, Any]
    same_call_is_already_staged: bool


class ToolCallEvaluationsSchema(DefaultBaseModel):
    last_user_message: Optional[str] = None
    most_recent_user_inquiry_or_need: Optional[str] = None
    most_recent_user_inquiry_or_need_was_already_resolved: Optional[bool] = False
    tool_call_evaluations: list[ToolCallEvaluation]


@dataclass(frozen=True)
class ToolCall:
    id: ToolCallId
    name: str
    arguments: Mapping[str, JSONSerializable]


@dataclass(frozen=True)
class ToolCallResult:
    id: ToolResultId
    tool_call: ToolCall
    result: ToolResult


class ToolCaller:
    def __init__(
        self,
        logger: Logger,
        tool_service: ToolService,
        schematic_generator: SchematicGenerator[ToolCallEvaluationsSchema],
    ) -> None:
        self._tool_service = tool_service
        self.logger = logger
        self._schematic_generator = schematic_generator

    async def infer_tool_calls(
        self,
        agents: Sequence[Agent],
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        ordinary_guideline_propositions: Sequence[GuidelineProposition],
        tool_enabled_guideline_propositions: Mapping[GuidelineProposition, Sequence[Tool]],
        staged_events: Sequence[EmittedEvent],
    ) -> Sequence[ToolCall]:
        inference_prompt = self._format_tool_call_inference_prompt(
            agents,
            context_variables,
            interaction_history,
            terms,
            ordinary_guideline_propositions,
            tool_enabled_guideline_propositions,
            staged_events,
        )

        with self.logger.operation("Tool classification"):
            inference_output = await self._run_inference(inference_prompt)

        tool_calls_that_need_to_run = [
            c for c in inference_output if not c.same_call_is_already_staged
        ]

        return [
            ToolCall(
                id=ToolCallId(generate_id()),
                name=tc.name,
                arguments=tc.arguments,
            )
            for tc in tool_calls_that_need_to_run
            if tc.should_run and tc.applicability_score >= 7
        ]

    async def execute_tool_calls(
        self,
        context: ToolContext,
        tool_calls: Sequence[ToolCall],
        tools: Sequence[Tool],
    ) -> Sequence[ToolCallResult]:
        tools_by_name = {t.name: t for t in tools}

        with self.logger.operation("Tool calls"):
            tool_results = await asyncio.gather(
                *(
                    self._run_tool(
                        context=context,
                        tool_call=tool_call,
                        tool=tools_by_name[tool_call.name],
                    )
                    for tool_call in tool_calls
                )
            )

            return tool_results

    def _format_tool_call_inference_prompt(
        self,
        agents: Sequence[Agent],
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_event_list: Sequence[Event],
        terms: Sequence[Term],
        ordinary_guideline_propositions: Sequence[GuidelineProposition],
        tool_enabled_guideline_propositions: Mapping[GuidelineProposition, Sequence[Tool]],
        staged_events: Sequence[EmittedEvent],
    ) -> str:
        assert len(agents) == 1

        staged_calls = self._get_staged_calls(staged_events)
        tools = list(chain(*tool_enabled_guideline_propositions.values()))

        builder = PromptBuilder()

        builder.add_agent_identity(agents[0])
        builder.add_interaction_history(interaction_event_list)
        builder.add_context_variables(context_variables)
        builder.add_terminology(terms)

        builder.add_section(
            """
Before generating your next response, you are highly encouraged to use tools that are provided to you,
in order to generate a high-quality, well-informed response.

Your task at this point is to evaluate what, if any, tools should be run in order to facilitate
an informed continuation of your interaction with the user.

You will also receive a list of tool calls that have already been staged, under the heading STAGED TOOL CALLS.
If you find what you need in those staged calls, then you don't need to call the corresponding tool(s) again
with the same arguments, as the staged calls' data is extremely fresh.
"""  # noqa
        )

        builder.add_guideline_propositions(
            ordinary_guideline_propositions,
            tool_enabled_guideline_propositions,
            include_priority=False,
            include_tool_associations=True,
        )

        builder.add_tool_definitions(tools)

        builder.add_section(
            f"""
Before generating your next response, you must now decide whether to use any of the tools provided.
Here are the principles by which you can decide whether to use tools:

1. Determine the tools that need to be called based on the provided instructions and the interaction's latest state.
2. It is permissible to propose tools that do not directly answer the interaction's latest state but can compute based on the function definitions to promote a more advanced state for answering.
3. A tool may be called multiple times with different arguments, but preferably not with the same ones unless the situation clearly merits it.
4. If a tool is not to be called at all, it must still be mentioned in the results!
5. Avoid calling a tool that has already been called with the same arguments!
6. If the data required for a tool call is already given in previous invoked tool results, then the same tool call should not run.
7. Ensure that each proposed tool call relies solely on the immediate context and the staged calls, without depending on other tools yet to be invoked. This prevents the proposal of interconnected tools unless their dependencies are already satisfied by previous calls.

Produce a valid JSON object according to the following format:

{{
    "tool_call_evaluations": [
        {{
            "name": "<TOOL NAME>",
            "rationale": "<A FEW WORDS THAT EXPLAIN WHETHER AND WHY THE TOOL NEEDS TO BE CALLED>",
            "applicability_score": <INTEGER FROM 1 TO 10>,
            "should_run": <BOOLEAN>,
            "arguments": <ARGUMENTS FOR THE TOOL>,
            "same_call_is_already_staged": <BOOLEAN>
        }},
        ...,
        {{
            "name": "<TOOL NAME>",
            "rationale": "<A FEW WORDS THAT EXPLAIN WHETHER AND WHY THE TOOL NEEDS TO BE CALLED>",
            "applicability_score": <INTEGER FROM 1 TO 10>,
            "should_run": <BOOLEAN>,
            "arguments": <ARGUMENTS FOR THE TOOL>,
            "same_call_is_already_staged": <BOOLEAN>
        }}
    ]
}}

Here's a hypothetical example, for your reference:

{{
    "last_user_message": "<REPEAT THE LAST USER MESSAGE IN THE INTERACTION>",
    "most_recent_user_inquiry_or_need": "<...>",
    "most_recent_user_inquiry_or_need_was_already_resolved": <BOOL>,
    "tool_call_evaluations": [
        {{
            "name": "transfer_money",
            "rationale": "Jack owes John $5",
            "applicability_score": 9,
            "should_run": true,
            "arguments": {{
                "from": "Jack",
                "to": "John",
                "amount": 5.00
            }}
            "same_call_is_already_staged": false
        }},
        {{
            "name": "loan_money",
            "rationale": "There's no obvious reason for Jack to loan any money to anyone",
            "applicability_score": 2,
            "should_run": false,
            "same_call_is_already_staged": false
        }},
        {{
            "name": "get_account_information",
            "rationale": "Bob's account information is already given in the list of staged tool calls",
            "applicability_score": 9,
            "should_run": false,
            "same_call_is_already_staged": true
        }}
    ]
}}


Note that the `tool_call_evaluations` list can be empty if no tools need to be called.
"""  # noqa
        )

        if staged_calls:
            builder.add_section(
                f"""
STAGED TOOL CALLS
-----------------
The following is a list of staged tool calls after the interaction's latest state.
You can use this information to avoid redundant calls and inform your response.
If the same tool is already staged with the exact same arguments, then you must
set "same_call_is_already_staged" to true, because its information is fresh here!: ###
{staged_calls}
###
"""
            )
        else:
            builder.add_section(
                """
STAGED TOOL CALLS
-----------------
There are no staged tool calls at this moment.
###
"""
            )

        return builder.build()

    def _get_staged_calls(
        self,
        emitted_events: Sequence[EmittedEvent],
    ) -> Optional[str]:
        staged_calls = list(
            chain(*[e["data"] for e in emitted_tool_events_to_dicts(emitted_events)])
        )

        if not staged_calls:
            return None

        return json.dumps(
            [
                {
                    "name": invocation["tool_name"],
                    "arguments": invocation["arguments"],
                    "result": invocation["result"],
                }
                for invocation in staged_calls
            ]
        )

    async def _run_inference(
        self,
        prompt: str,
    ) -> Sequence[ToolCallEvaluation]:
        self.logger.debug(f"Tool call inference prompt: {prompt}")

        tool_call_evaluation = await self._schematic_generator.generate(
            prompt=prompt, hints={"temperature": 0.3}
        )

        self.logger.debug(
            f"Tool call request results: {json.dumps([t.model_dump() for t in tool_call_evaluation.content.tool_call_evaluations], indent=2),}"
        )
        return tool_call_evaluation.content.tool_call_evaluations

    async def _run_tool(
        self,
        context: ToolContext,
        tool_call: ToolCall,
        tool: Tool,
    ) -> ToolCallResult:
        try:
            self.logger.debug(f"Tool call executing: {tool_call.name}/{tool_call.id}")
            result = await self._tool_service.call_tool(
                tool.id,
                context,
                tool_call.arguments,
            )
            self.logger.debug(
                f"Tool call returned: {tool_call.name}/{tool_call.id}: {json.dumps(asdict(result), indent=2)}"
            )

            return ToolCallResult(
                id=ToolResultId(generate_id()),
                tool_call=tool_call,
                result={"data": result.data, "metadata": result.metadata},
            )
        except Exception as e:
            self.logger.error(
                f"Tool execution error (tool='{tool_call.name}', "
                "arguments={tool_call.arguments}): " + "\n".join(traceback.format_exception(e)),
            )

            return ToolCallResult(
                id=ToolResultId(generate_id()),
                tool_call=tool_call,
                result={"data": "Tool call error", "metadata": {"error_details": str(e)}},
            )
