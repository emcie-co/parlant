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

from dataclasses import dataclass
import json
import traceback
from typing import Optional, Sequence
from parlant.core.common import DefaultBaseModel
from parlant.core.engines.alpha.optimization_policy import OptimizationPolicy
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.guidelines import GuidelineContent
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.services.indexing.common import EvaluationError, ProgressReport
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.shots import Shot, ShotCollection


class CustomerDependentActionProposition(DefaultBaseModel):
    is_customer_dependent: bool
    customer_action: Optional[str] = ""
    agent_action: Optional[str] = ""


class CustomerDependentActionSchema(DefaultBaseModel):
    action: str
    is_customer_dependent: bool
    customer_action: Optional[str] = ""
    agent_action: Optional[str] = ""


@dataclass
class CustomerDependentActionShot(Shot):
    guideline: GuidelineContent
    expected_result: CustomerDependentActionSchema


class CustomerDependentActionDetector:
    def __init__(
        self,
        logger: Logger,
        optimization_policy: OptimizationPolicy,
        schematic_generator: SchematicGenerator[CustomerDependentActionSchema],
        service_registry: ServiceRegistry,
    ) -> None:
        self._logger = logger
        self._optimization_policy = optimization_policy

        self._schematic_generator = schematic_generator
        self._service_registry = service_registry

    async def detect_if_customer_dependent(
        self,
        guideline: GuidelineContent,
        progress_report: Optional[ProgressReport] = None,
    ) -> CustomerDependentActionProposition:
        if progress_report:
            await progress_report.stretch(1)

        with self._logger.scope("CustomerDependentActionDetector"):
            generation_attempt_temperatures = (
                self._optimization_policy.get_guideline_proposition_retry_temperatures(
                    hints={"type": self.__class__.__name__}
                )
            )

            last_generation_exception: Exception | None = None

            for generation_attempt in range(3):
                try:
                    proposition = await self._generate_customer_dependent(
                        guideline, temperature=generation_attempt_temperatures[generation_attempt]
                    )

                    if progress_report:
                        await progress_report.increment(1)

                    return CustomerDependentActionProposition(
                        is_customer_dependent=proposition.is_customer_dependent,
                        customer_action=proposition.customer_action,
                        agent_action=proposition.agent_action,
                    )

                except Exception as exc:
                    self._logger.warning(
                        f"CustomerDependentActionDetector attempt {generation_attempt} failed: {traceback.format_exception(exc)}"
                    )

                    last_generation_exception = exc

            raise EvaluationError() from last_generation_exception

    async def _build_prompt(
        self, guideline: GuidelineContent, shots: Sequence[CustomerDependentActionShot]
    ) -> PromptBuilder:
        builder = PromptBuilder()

        builder.add_section(
            name="customer-dependent-action-detector-general-instructions",
            template="""
GENERAL INSTRUCTIONS
-----------------
In our system, the behavior of a conversational AI agent is guided by "guidelines". The agent makes use of these guidelines whenever it interacts with a user (also referred to as the customer).
Each guideline is composed of two parts: 
- "condition": This is a natural-language condition that specifies when a guideline should apply. We test against this condition to determine whether this guideline should be applied when generating the agent's next reply.
- "action": This is a natural-language instruction that should be followed by the agent whenever the "condition" part of the guideline applies to the conversation in its particular state.
Any instruction described here applies only to the agent, and not to the user.

While an action can only instruct the agent to do something, it may require something from the customer to be considered completed.
For example, the action "get the customer's account number" requires the customer to provide their account number for it to be considered completed.
""",
        )

        builder.add_section(
            name="customer-dependent-action-detector-task-description",
            template="""
TASK DESCRIPTION
-----------------
Your task is to determine whether a given guideline’s action requires something from the customer in order for the action to be considered complete.

Actions that require input or behavior from the customer are called customer-dependent actions.

Later in this prompt, you will be provided with a single guideline. The guideline’s condition is included for context, but your decision should be based only on the action.

Ask yourself: what must happen for this action to be considered complete? Is it something the agent alone must do, or does it also rely on a response or action from the customer?

Edge Cases to Consider:
 - If the action includes multiple steps (e.g., “offer assistance to the customer and ask them for their account number”), then the entire action is considered customer dependent if any part of it depends on the customer.
 - If the action tells the agent to ask the customer a question, it is generally considered customer dependent, since the question expects an answer in order to complete the action. Exception: If the question is clearly a pleasantry or rhetorical (e.g., “what’s up with you?” in a casual exchange), and not meant to gather meaningful information, the action is not considered customer dependent.


If you determine the action is customer dependent, you must also split it into:
 - the portion that depends solely on the agent (agent_action)
 - the portion that depends on the customer (customer_action). 

Your decision will be used to asses whether this guideline was completed at different stages of the conversation. You should split the action such that it is considered complete if and only if both the agent and customer portions were completed.
For example, the customer dependent action "ask the customer for their age" should be split into the agent_action "the agent asked the customer for their age" and the customer_action "the customer provided their age"
""",
        )
        builder.add_section(
            name="customer-dependent-action-shots",
            template="""
EXAMPLES
-----------
{shots_text}""",
            props={"shots_text": self._format_shots(shots)},
        )
        builder.add_section(
            name="customer-dependent-action-detector-guideline",
            template="""
GUIDELINE
-----------
condition: {condition}
action: {action}
""",
            props={"condition": guideline.condition, "action": guideline.action},
        )

        builder.add_section(
            name="guideline-action-proposer-output-format",
            template="""OUTPUT FORMAT
-----------
Use the following format to evaluate whether the guideline has a customer dependent action:
Expected output (JSON):
```json
{{
  "action": "{action}",
  "is_customer_dependent": "<BOOL>",
  "customer_action": "<STR, the portion of the action that applies to the customer. Can be omitted if is_customer_dependent is false>",
  "agent_action": "<STR, the portion of the action that applies to the agent. Can be omitted necessary if is_customer_dependent is false>"
}}
```
""",
            props={"action": guideline.action},
        )

        return builder

    async def _generate_customer_dependent(
        self,
        guideline: GuidelineContent,
        temperature: float,
    ) -> CustomerDependentActionSchema:
        prompt = await self._build_prompt(guideline, _baseline_shots)

        response = await self._schematic_generator.generate(
            prompt=prompt,
            hints={"temperature": temperature},
        )

        return response.content

    def _format_shots(self, shots: Sequence[CustomerDependentActionShot]) -> str:
        return "\n".join(
            [
                f"""Example {i}: {shot.description}
Guideline:
    Condition: {shot.guideline.condition}
    Action: {shot.guideline.action}

Expected Response:
{json.dumps(shot.expected_result.model_dump(mode="json", exclude_unset=True), indent=2)}
###
"""
                for i, shot in enumerate(shots, start=1)
            ]
        )


example_1_guideline = GuidelineContent(
    condition="the customer wishes to submit an order",
    action="ask for their account number and shipping address. Inform them that it would take 3-5 business days.",
)
example_1_shot = CustomerDependentActionShot(
    description="A guideline with a customer dependent action",
    guideline=example_1_guideline,
    expected_result=CustomerDependentActionSchema(
        action=example_1_guideline.action or "",
        is_customer_dependent=True,
        customer_action="The customer provided both their account number and shipping address",
        agent_action="The agent asks for the customer's account number and shipping address, and informs them that it would take 3-5 business days.",
    ),
)

example_2_guideline = GuidelineContent(
    condition='asked "whats up dog"', action='reply with "nothing much, what\'s up with you?"'
)
example_2_shot = CustomerDependentActionShot(
    description="A guideline whose action involves a question, but is not customer dependent",
    guideline=example_2_guideline,
    expected_result=CustomerDependentActionSchema(
        action=example_2_guideline.action or "", is_customer_dependent=False
    ),
)

_baseline_shots: Sequence[CustomerDependentActionShot] = [
    example_1_shot,
    example_2_shot,
]

shot_collection = ShotCollection[CustomerDependentActionShot](_baseline_shots)
