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

from __future__ import annotations

import json
import traceback
from typing import Any, Optional, Sequence

from parlant.core.engines.alpha.optimization_policy import OptimizationPolicy
from parlant.core.guidelines import GuidelineContent
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.common import DefaultBaseModel
from parlant.core.services.indexing.common import EvaluationError, ProgressReport
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.tools import Tool, ToolId, ToolParameterDescriptor, ToolParameterOptions


class GuidelineActionProposition(DefaultBaseModel):
    content: GuidelineContent
    rationale: str


class GuidelineActionPropositionSchema(DefaultBaseModel):
    rationale: str
    action: str


class GuidelineActionProposer:
    def __init__(
        self,
        logger: Logger,
        optimization_policy: OptimizationPolicy,
        schematic_generator: SchematicGenerator[GuidelineActionPropositionSchema],
        service_registry: ServiceRegistry,
    ) -> None:
        self._logger = logger
        self._optimization_policy = optimization_policy

        self._schematic_generator = schematic_generator
        self._service_registry = service_registry

    async def propose_action(
        self,
        guideline: GuidelineContent,
        tool_ids: Sequence[ToolId],
        progress_report: Optional[ProgressReport] = None,
    ) -> Optional[GuidelineActionProposition]:
        if not tool_ids or guideline.action:
            return None

        if progress_report:
            await progress_report.stretch(1)

        with self._logger.scope("GuidelineActionProposer"):
            generation_attempt_temperatures = (
                self._optimization_policy.get_guideline_proposition_retry_temperatures(
                    hints={"type": self.__class__.__name__}
                )
            )

            last_generation_exception: Exception | None = None

            for generation_attempt in range(3):
                try:
                    tools: list[Tool] = []
                    for tid in tool_ids:
                        service = await self._service_registry.read_tool_service(tid.service_name)
                        tool = await service.read_tool(tid.tool_name)
                        tools.append(tool)

                    proposition = await self._generate_action(
                        guideline,
                        tools,
                        tool_ids,
                        generation_attempt_temperatures[generation_attempt],
                    )

                    if progress_report:
                        await progress_report.increment(1)

                    return GuidelineActionProposition(
                        content=GuidelineContent(
                            condition=guideline.condition,
                            action=proposition.action,
                        ),
                        rationale=proposition.rationale,
                    )
                except Exception as exc:
                    self._logger.warning(
                        f"GuidelineActionProposition attempt {generation_attempt} failed: {traceback.format_exception(exc)}"
                    )

                    last_generation_exception = exc

            raise EvaluationError() from last_generation_exception

    def _add_tool_definitions_section(
        self,
        tool: tuple[ToolId, Tool],
    ) -> dict[str, Any]:
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

        return _get_tool_spec(tool[0], tool[1])

    async def _build_prompt(
        self,
        guideline: GuidelineContent,
        tools: Sequence[Tool],
        tool_ids: Sequence[ToolId],
    ) -> PromptBuilder:
        builder = PromptBuilder()

        builder.add_section(
            name="guideline-action-proposer-general-instructions",
            template="""
In our system, the behavior of a conversational AI agent is guided by "guidelines". The agent makes use of these guidelines whenever it interacts with a user (also referred to as the customer).
Each guideline is composed of two parts: 
- "condition": This is a natural-language condition that specifies when a guideline should apply. We look at each conversation at any particular state, and we test against this condition to understand 
if we should have this guideline participate in generating the next reply to the user.
- "action": This is a natural-language instruction that should be followed by the agent whenever the "condition" part of the guideline applies to the conversation in its particular state.
Any instruction described here applies only to the agent, and not to the user.
Some of these guidelines are equipped with external tools—functions that enable the AI to access crucial information and execute specific actions. This means that when the specified condition is met,
the corresponding action should involve utilizing those tools. 

Your task is given a guideline's condition and a tool description (or a list of tools) to provide an action that shortly and concisely describe an action that aligns with the tool purpose.
You will receive a tool description that includes the tool signature, a description of the tool (if exists), and the types and descriptions of its parameters.
If available, use the tool description to incorporate any relevant information that may inform how the tool should be used.
Note that the tool name and description may be uninformative, so you may need to infer the tool's purpose from its parameters.

""",
        )
        builder.add_section(
            name="guideline-action-proposer-example",
            template="""
Examples:
1. 
Condition: Asked to get the weather forecast for a city  
Tool description:
{{
    "tool_name": "local:get_weather",
    "description": "Get the current weather and forecast for a specific city",
    "optional_arguments": {{
        "unit": {{"schema": {{"type": "string"}}, "description": "Temperature unit: Celsius or Fahrenheit"}}
    }},
    "required_parameters": {{
        "city": {{"schema": {{"type": "string"}}, "description": "The city to get the weather for"}}
    }}
}}
Action: Provide current weather and forecast

2.  
Condition: Asked to send an email  
Tool description:
{{
    "tool_name": "local:send_email",
    "description": "Send an email to a recipient",
    "optional_arguments": {{}},
    "required_parameters": {{
        "to": {{"schema": {{"type": "string"}}, "description": "Recipient email address"}},
        "subject": {{"schema": {{"type": "string"}}, "description": "Subject of the email"}},
        "body": {{"schema": {{"type": "string"}}, "description": "Content of the email"}}
    }}
}}
Action: Send the specified email

3.  
Condition: A recurring invoice has failed to process due to an expired payment method.  
Tool description:
{{
    "tool_name": "local:send_payment_failure_notification",
    "description": "Notify the user that a payment attempt failed",
    "required_parameters": {{
        "user_id": {{"schema": {{"type": "string"}}, "description": "The ID of the user to notify"}},
        "invoice_id": {{"schema": {{"type": "string"}}, "description": "The invoice that failed to process"}}
    }}
}}
Action: Notify the user that payment for their invoice could not be processed.

4.  
Condition: A scheduled backup did not complete within its expected time window.  
Tool descriptions:
[
  {{
    "tool_name": "local:check_backup_status",
    "description": "Check the current or last-known status of a backup job",
    "required_parameters": {{
        "job_id": {{"schema": {{"type": "string"}}, "description": "Identifier of the backup job"}}
    }}
  }},
  {{
    "tool_name": "local:send_alert",
    "description": "Send an alert to system administrators",
    "required_parameters": {{
        "message": {{"schema": {{"type": "string"}}, "description": "The alert message"}},
        "recipients": {{"schema": {{"type": "array", "items": {{"type": "string"}}}}, "description": "List of recipient user IDs"}}
    }}
  }}
]
Action: Check the status of the backup job and alert administrators if it failed.

5.  
Condition: A weather alert has been issued for a location where outdoor company events are scheduled.  
Tool descriptions:
[
  {{
    "tool_name": "local:check_weather_alerts",
    "description": "Get current weather alerts for a region",
    "required_parameters": {{
        "location": {{"schema": {{"type": "string"}}, "description": "City or region to check"}}
    }}
  }},
  {{
    "tool_name": "local:reschedule_event",
    "description": "Reschedule or cancel an event based on external factors",
    "required_parameters": {{
        "event_id": {{"schema": {{"type": "string"}}, "description": "ID of the event"}},
        "reason": {{"schema": {{"type": "string"}}, "description": "Reason for rescheduling"}}
    }}
  }}
]
Action: Check for severe weather and reschedule outdoor events if necessary.

--------------------------------------------------------------------------------
""",
        )

        builder.add_section(
            name="guideline-action-proposer-guideline",
            template="""
Guideline Condition:
--------------------------
{condition}
""",
            props={"condition": guideline.condition},
        )

        tools_text = "\n".join(
            f"- {tid.to_string()}: {self._add_tool_definitions_section((tid, tool))}"
            for tid, tool in zip(tool_ids, tools)
        )
        builder.add_section(
            name="guideline-action-proposer-tools",
            template="""

Relevant Tools:
--------------
{tools_text}
""",
            props={"tools_text": tools_text},
        )

        builder.add_section(
            name="guideline-action-proposer-output-format",
            template="""
Expected output (JSON):
```json
{{
    "rationale": "<RATIONALE>"
    "action": "<SINGLE-LINE-INSTRUCTION>",
}}
```
""",
        )

        return builder

    async def _generate_action(
        self,
        guideline: GuidelineContent,
        tools: Sequence[Tool],
        tool_ids: Sequence[ToolId],
        temperature: float,
    ) -> GuidelineActionPropositionSchema:
        prompt = await self._build_prompt(guideline, tools, tool_ids)

        response = await self._schematic_generator.generate(
            prompt=prompt,
            hints={"temperature": temperature},
        )

        return response.content
