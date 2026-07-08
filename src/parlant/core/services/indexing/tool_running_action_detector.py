from dataclasses import dataclass
import json
import traceback
from typing import Any, Optional, Sequence

import httpx

from parlant.core.common import DefaultBaseModel
from parlant.core.engines.alpha.optimization_policy import OptimizationPolicy
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.rules import RuleContent
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.services.indexing.common import EvaluationError, ProgressReport
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.shots import Shot, ShotCollection
from parlant.core.tools import Tool, ToolId, ToolParameterDescriptor, ToolParameterOptions
from parlant.core.store_provider import StoreProvider, StoreProviderHints


class ToolRunningActionProposition(DefaultBaseModel):
    is_tool_running_only: bool


class ToolRunningActionSchema(DefaultBaseModel):
    action: str
    rationale: str
    is_tool_running_only: bool


@dataclass
class ToolRunningActionShot(Shot):
    rule: RuleContent
    expected_result: ToolRunningActionSchema


class ToolRunningActionDetector:
    def __init__(
        self,
        logger: Logger,
        optimization_policy: OptimizationPolicy,
        schematic_generator: SchematicGenerator[ToolRunningActionSchema],
        store_provider: StoreProvider,
    ) -> None:
        self._logger = logger
        self._store_provider = store_provider
        self._optimization_policy = optimization_policy

        self._schematic_generator = schematic_generator

    @property
    def _service_registry(self) -> ServiceRegistry:
        return self._store_provider.get_store(
            ServiceRegistry, StoreProviderHints(call_site="engine")
        )

    async def detect_if_tool_running(
        self,
        rule: RuleContent,
        tool_ids: Sequence[ToolId],
        progress_report: Optional[ProgressReport] = None,
    ) -> ToolRunningActionProposition:
        if not tool_ids:
            return ToolRunningActionProposition(
                is_tool_running_only=False,
            )

        if progress_report:
            await progress_report.stretch(1)

        tools = await self._load_tools(tool_ids)

        with self._logger.scope("ToolRunningActionDetector"):
            generation_attempt_temperatures = (
                self._optimization_policy.get_rule_proposition_retry_temperatures(
                    hints={"type": self.__class__.__name__}
                )
            )

            last_generation_exception: Exception | None = None

            for generation_attempt in range(3):
                try:
                    result = await self._generate_tool_running(
                        rule,
                        tools,
                        temperature=generation_attempt_temperatures[generation_attempt],
                    )

                    if progress_report:
                        await progress_report.increment(1)

                    return ToolRunningActionProposition(
                        is_tool_running_only=result.is_tool_running_only,
                    )

                except Exception as exc:
                    self._logger.warning(
                        f"ToolRunningActionDetector attempt {generation_attempt} failed: {traceback.format_exception(exc)}"
                    )

                    last_generation_exception = exc

            raise EvaluationError() from last_generation_exception

    async def _load_tools(self, tool_ids: Sequence[ToolId]) -> dict[ToolId, Tool]:
        tools: dict[ToolId, Tool] = {}

        for tid in tool_ids:
            service = await self._service_registry.read_tool_service(tid.service_name)
            try:
                tools[tid] = await service.read_tool(tid.tool_name)
            except httpx.HTTPError as exc:
                service_url = getattr(service, "url", "unknown")
                raise EvaluationError(
                    f"Could not read tool '{tid.tool_name}' from tool service "
                    f"'{tid.service_name}' at {service_url} while evaluating a rule. "
                    "Make sure the tool service is running and reachable before SDK startup evaluations run."
                ) from exc

        return tools

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
        rule: RuleContent,
        tools: dict[ToolId, Tool],
        shots: Sequence[ToolRunningActionShot],
    ) -> PromptBuilder:
        builder = PromptBuilder()

        builder.add_section(
            name="tool-running-action-detector-general-instructions",
            template="""
GENERAL INSTRUCTIONS
-----------------
In our system, the behavior of a conversational AI agent is guided by "rules". The agent makes use of these rules whenever it interacts with a user (also referred to as the customer).
Each rule is composed of two parts:
- "condition": This is a natural-language condition that specifies when a rule should apply. We test against this condition to determine whether this rule should be applied when generating the agent's next reply.
- "action": This is a natural-language instruction that should be followed by the agent whenever the "condition" part of the rule applies to the conversation in its particular state.
Any instruction described here applies only to the agent, and not to the user.

Some of these rules are equipped with external tools — functions that enable the AI to access crucial information and execute specific actions. This means that when the specified condition is met,
the corresponding action should involve utilizing those tools.

""",
        )

        builder.add_section(
            name="tool-running-action-detector-task-description",
            template="""
TASK DESCRIPTION
-----------------
Your task is to determine whether a rule’s action involves only running one or more tools, without requiring any communication to the user.
You will be provided with an action description and a list of associated tools. Your job is to decide whether the action is tool only.

Examples:
- If the action is "check the customer balance", and the tool "check_balance" is associated, this is a tool-only action.
- If the action is "notify the customer with their balance" and the tool "check_balance" is associated, then it involves both running a tool and
sending a message to the user, so it is not tool-only.

Even when multiple tools are involved, the action should be considered tool-only as long as there is no instruction to communicate with the user.
If the action includes multiple steps or instructions, you should evaluate each one individually. The action is tool-only only if all steps involve
running tools without requiring any user facing communication.
""",
        )
        builder.add_section(
            name="tool-running-action-shots",
            template="""
EXAMPLES
-----------
{shots_text}""",
            props={"shots_text": self._format_shots(shots)},
        )
        builder.add_section(
            name="tool-running-action-detector-rule",
            template="""
RULE
-----------
condition: {condition}
action: {action}
""",
            props={"condition": rule.condition, "action": rule.action},
        )
        tools_text = "\n".join(
            f"- {i}: {self._add_tool_definitions_section((tid, tools[tid]))}"
            for i, tid in enumerate(tools, start=1)
        )
        builder.add_section(
            name="tool-running-action-detector-tools",
            template="""

Relevant Tools:
--------------
{tools_text}
""",
            props={"tools_text": tools_text},
        )

        builder.add_section(
            name="rule-action-proposer-output-format",
            template="""OUTPUT FORMAT
-----------
Use the following format to evaluate whether the rule has a customer dependent action:
Expected output (JSON):
```json
{{
  "action": "{action}",
  "rationale": "<str, a few words that explains whether it tool running only>"
  "is_tool_running_only": "<BOOL>",
}}
```
""",
            props={"action": rule.action},
        )

        return builder

    async def _generate_tool_running(
        self,
        rule: RuleContent,
        tools: dict[ToolId, Tool],
        temperature: float,
    ) -> ToolRunningActionSchema:
        prompt = await self._build_prompt(rule, tools, _baseline_shots)

        response = await self._schematic_generator.generate(
            prompt=prompt,
            hints={"temperature": temperature},
        )

        return response.content

    def _format_shots(
        self,
        shots: Sequence[ToolRunningActionShot],
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
        shot: ToolRunningActionShot,
    ) -> str:
        return f"""
- **Context**:
{shot.description}

- **Expected Result**:
```json
{json.dumps(shot.expected_result.model_dump(mode="json", exclude_unset=True), indent=2)}
```"""


example_1_rule = RuleContent(
    condition="the customer wishes to reset their password",
    action="reset the customer’s password and confirm the reset by email",
)
example_1_shot = ToolRunningActionShot(
    description="tool available:  reset_password(acount_number: int)",
    rule=example_1_rule,
    expected_result=ToolRunningActionSchema(
        action=example_1_rule.action,
        rationale="Need to confirm with the customer that the reset was sent by mail",
        is_tool_running_only=False,
    ),
)

example_2_rule = RuleContent(
    condition="the customer wishes to reset their password",
    action="reset the customer’s password and confirm the reset by email",
)
example_2_shot = ToolRunningActionShot(
    description="tool available: reset_password(acount_number: int) and send_email_confirmation(email_address: str)",
    rule=example_2_rule,
    expected_result=ToolRunningActionSchema(
        action=example_2_rule.action,
        rationale="need to reset with a tool and confirm also with a tool",
        is_tool_running_only=True,
    ),
)

_baseline_shots: Sequence[ToolRunningActionShot] = [
    example_1_shot,
    example_2_shot,
]

shot_collection = ShotCollection[ToolRunningActionShot](_baseline_shots)
