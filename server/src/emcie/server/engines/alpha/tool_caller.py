import asyncio
from dataclasses import dataclass
from itertools import chain
import json
import traceback
import jsonfinder  # type: ignore
from typing import Any, Mapping, NewType, Optional, Sequence, TypedDict

from loguru import logger

from emcie.server.core.agents import Agent
from emcie.server.core.common import JSONSerializable, generate_id
from emcie.server.core.context_variables import ContextVariable, ContextVariableValue
from emcie.server.core.sessions import Event
from emcie.server.core.tools import Tool, ToolParameter, ToolService
from emcie.server.core.terminology import Term
from emcie.server.engines.alpha.guideline_proposition import GuidelineProposition
from emcie.server.engines.alpha.prompt_builder import PromptBuilder
from emcie.server.engines.alpha.utils import (
    duration_logger,
    make_llm_client,
    produced_tool_events_to_dicts,
)
from emcie.server.engines.common import (
    ProducedEvent,
)

ToolCallId = NewType("ToolCallId", str)
ToolResultId = NewType("ToolResultId", str)


@dataclass(frozen=True)
class ToolCall:
    id: ToolCallId
    name: str
    parameters: dict[str, ToolParameter]


@dataclass(frozen=True)
class ToolResult:
    id: ToolResultId
    tool_call: ToolCall
    result: JSONSerializable


class ToolCaller:
    class ToolCallRequest(TypedDict):
        name: str
        rationale: str
        applicability_score: int
        should_run: bool
        parameters: dict[str, Any]
        same_call_is_already_staged: bool

    def __init__(
        self,
        tool_service: ToolService,
    ) -> None:
        self._tool_service = tool_service
        self._llm_client = make_llm_client("openai")

    async def infer_tool_calls(
        self,
        agents: Sequence[Agent],
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        ordinary_guideline_propositions: Sequence[GuidelineProposition],
        tool_enabled_guideline_propositions: Mapping[GuidelineProposition, Sequence[Tool]],
        staged_events: Sequence[ProducedEvent],
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

        with duration_logger("Tool classification"):
            inference_output = await self._run_inference(inference_prompt)

        tool_calls_that_need_to_run = [
            c for c in inference_output if not c["same_call_is_already_staged"]
        ]

        return [
            ToolCall(
                id=ToolCallId(generate_id()),
                name=tc["name"],
                parameters=tc["parameters"],
            )
            for tc in tool_calls_that_need_to_run
            if tc["should_run"] and tc["applicability_score"] >= 7
        ]

    async def execute_tool_calls(
        self,
        tool_calls: Sequence[ToolCall],
        tools: Sequence[Tool],
    ) -> Sequence[ToolResult]:
        tools_by_name = {t.name: t for t in tools}

        with duration_logger("Tool calls"):
            tool_results = await asyncio.gather(
                *(
                    self._run_tool(
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
        staged_events: Sequence[ProducedEvent],
    ) -> str:
        assert len(agents) == 1

        staged_function_calls = self._get_invoked_functions(staged_events)
        tools = list(chain(*tool_enabled_guideline_propositions.values()))

        builder = PromptBuilder()

        builder.add_agent_identity(agents[0])
        builder.add_interaction_history(interaction_event_list)
        builder.add_context_variables(context_variables)
        builder.add_terminology(terms)

        builder.add_section(
            """
Before generating your next response, you are highly encouraged to use tools that are provided to you, in order to generate a high-quality, well-informed response.
"""  # noqa
        )

        builder.add_guideline_propositions(
            ordinary_guideline_propositions,
            tool_enabled_guideline_propositions,
            include_priority=False,
            include_tool_associations=True,
        )

        builder.add_tool_definitions(tools)

        if staged_function_calls:
            builder.add_section(
                f"""
The following is a list of staged tool calls after the interaction's latest state.
You can use this information to avoid redundant calls and inform your response.
For example, if the data you need already exists in one of these calls, then you DO NOT
need to ask for that exact tool call (with the same parameters) to be run again,
because its information is fresh here!: ###
{staged_function_calls}
###
"""
            )

        builder.add_section(
            f"""
Before generating your next response, you must now decide whether to use any of the tools provided.
Here are the principles by which you can decide whether to use tools:

1. Determine the tool functions that need to be called based on the provided instructions and the interaction's latest state.
2. It is permissible to propose tool functions that do not directly answer the interaction's latest state but can compute based on the function definitions to promote a more advanced state for answering.
3. A tool function may be called multiple times with different parameters.
4. If a tool function is not called at all, it must still be mentioned in the results!
5. Avoid calling a tool function that has already been called with the same parameters!
6. If the data required for a tool call is already given in previous invoked tool results, then the same tool call should not run.
7. Ensure that each function proposed for invocation relies solely on the immediate context and previously invoked functions, without depending on other functions yet to be invoked. This prevents the proposal of interconnected functions unless their dependencies are already satisfied by previous calls.

Produce a valid JSON object according to the following format:

{{
    "tool_call_specifications": [
        {{
            "name": "<FUNCTION NAME>",
            "rationale": "<A FEW WORDS THAT EXPLAIN WHETHER AND WHY THE TOOL FUNCTION NEEDS TO BE CALLED>",
            "applicability_score": <INTEGER FROM 1 TO 10>,
            "should_run": <BOOLEAN>,
            "parameters": <PARAMETERS FOR THE TOOL FUNCTION>,
            "same_call_is_already_staged": <BOOLEAN>
        }},
        ...,
        {{
            "name": "<FUNCTION NAME>",
            "rationale": "<A FEW WORDS THAT EXPLAIN WHETHER AND WHY THE TOOL FUNCTION NEEDS TO BE CALLED>",
            "applicability_score": <INTEGER FROM 1 TO 10>,
            "should_run": <BOOLEAN>,
            "parameters": <PARAMETERS FOR THE TOOL FUNCTION>,
            "same_call_is_already_staged": <BOOLEAN>
        }}
    ]
}}

Here's a hypothetical example, for your reference:

{{
    "tool_call_specifications": [
        {{
            "name": "transfer_money",
            "rationale": "Jack owes John $5",
            "applicability_score": 9,
            "should_run": true,
            "parameters": {{
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
            "rationale": "The account information is already given in the list of staged tool calls",
            "applicability_score": 9,
            "should_run": false,
            "same_call_is_already_staged": true
        }}
    ]
}}


Note that the `tool_call_specifications` list can be empty if no functions need to be called.
"""  # noqa
        )

        return builder.build()

    def _get_invoked_functions(
        self,
        produced_events: Sequence[ProducedEvent],
    ) -> Optional[str]:
        ordered_function_invocations = list(
            chain(*[e["data"] for e in produced_tool_events_to_dicts(produced_events)])
        )

        if not ordered_function_invocations:
            return None

        return json.dumps(
            [
                {
                    "function_name": invocation["tool_name"],
                    "parameters": invocation["parameters"],
                    "result": invocation["result"],
                }
                for invocation in ordered_function_invocations
            ]
        )

    async def _run_inference(
        self,
        prompt: str,
    ) -> Sequence[ToolCallRequest]:
        response = await self._llm_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            temperature=0.3,
        )

        content = response.choices[0].message.content or ""
        json_content = jsonfinder.only_json(content)[2]

        logger.debug(
            f"Tool call request results: {json.dumps(
                json_content['tool_call_specifications'], indent=2
                )}"
        )
        return json_content["tool_call_specifications"]  # type: ignore

    async def _run_tool(
        self,
        tool_call: ToolCall,
        tool: Tool,
    ) -> ToolResult:
        try:
            logger.debug(f"Tool call executing: {tool_call.name}/{tool_call.id}")
            result = await self._tool_service.execute_tool(tool.id, tool_call.parameters)
            logger.debug(f"Tool call returned: {tool_call.name}/{tool_call.id}: {result}")

            return ToolResult(
                id=ToolResultId(generate_id()),
                tool_call=tool_call,
                result=result,
            )
        except Exception as e:
            logger.error(
                f"Tool execution error (tool='{tool_call.name}', "
                "parameters={tool_call.parameters}): "
                + "\n".join(traceback.format_exception(e)),
            )

            return ToolResult(
                id=ToolResultId(generate_id()),
                tool_call=tool_call,
                result=f"Tool call error: {e}",
            )
