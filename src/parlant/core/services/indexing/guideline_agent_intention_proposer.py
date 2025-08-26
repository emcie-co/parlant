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


class AgentIntentionProposition(DefaultBaseModel):
    is_agent_intention: bool
    rewritten_condition: Optional[str] = ""


class AgentIntentionProposerSchema(DefaultBaseModel):
    condition: str
    is_agent_intention: bool
    rewritten_condition: Optional[str] = ""


@dataclass
class AgentIntentionProposerShot(Shot):
    guideline: GuidelineContent
    expected_result: AgentIntentionProposerSchema


class AgentIntentionProposer:
    def __init__(
        self,
        logger: Logger,
        optimization_policy: OptimizationPolicy,
        schematic_generator: SchematicGenerator[AgentIntentionProposerSchema],
        service_registry: ServiceRegistry,
    ) -> None:
        self._logger = logger
        self._optimization_policy = optimization_policy

        self._schematic_generator = schematic_generator
        self._service_registry = service_registry

    async def propose_agent_intention(
        self,
        guideline: GuidelineContent,
        progress_report: Optional[ProgressReport] = None,
    ) -> AgentIntentionProposition:
        if progress_report:
            await progress_report.stretch(1)

        with self._logger.scope("AgentIntentionProposer"):
            generation_attempt_temperatures = (
                self._optimization_policy.get_guideline_proposition_retry_temperatures(
                    hints={"type": self.__class__.__name__}
                )
            )

            last_generation_exception: Exception | None = None

            for generation_attempt in range(3):
                try:
                    proposition = await self._generate_agent_intention(
                        guideline, generation_attempt_temperatures[generation_attempt]
                    )

                    if progress_report:
                        await progress_report.increment(1)

                    return AgentIntentionProposition(
                        is_agent_intention=proposition.is_agent_intention,
                        rewritten_condition=proposition.rewritten_condition,
                    )
                except Exception as exc:
                    self._logger.warning(
                        f"AgentIntentionProposer attempt {generation_attempt} failed: {traceback.format_exception(exc)}"
                    )

                    last_generation_exception = exc

            raise EvaluationError() from last_generation_exception

    async def _build_prompt(
        self, guideline: GuidelineContent, shots: Sequence[AgentIntentionProposerShot]
    ) -> PromptBuilder:
        builder = PromptBuilder()

        builder.add_section(
            name="agent-intention-general-instructions",
            template="""
GENERAL INSTRUCTIONS
-----------------
In our system, the behavior of a conversational AI agent is guided by "guidelines". You make use of these guidelines whenever it interacts with a user (also referred to as the customer).
Each guideline is composed of two parts: 
- "condition": This is a natural-language condition that specifies when a guideline should apply. We test against this condition to determine whether this guideline should be applied when generating your next reply.
- "action": This is a natural-language instruction that should be followed by you whenever the "condition" part of the guideline applies to the conversation in its particular state.
Any instruction described here applies only to you, and not to the user.

""",
        )

        builder.add_section(
            name="agent-intention-task-description",
            template="""
TASK DESCRIPTION
-----------------
Your task is to determine whether a guideline condition reflects your intention. That is, whether it describes something you are doing or is about to do (e.g., "You discusses a patient's 
medical record" or "You explain the conditions and terms"). Note: If the condition refers to something you have already done, it should not be considered an agent intention.

If the condition reflects agent intention, rephrase it to describe what you are likely to do next, using the following format:
"You are likely to (do something)."

For example:
Original: "You discusses a patient's medical record"
Rewritten: "You are likely to discuss a patient's medical record"

Why this matters:
Although the original condition can be written in present tense, guideline matching happens before you reply. So we need the condition to reflect your probable upcoming behavior, based on the customer's latest message.




""",
        )
        builder.add_section(
            name="agent-intention-shots",
            template="""
EXAMPLES
-----------
{shots_text}""",
            props={"shots_text": self._format_shots(shots)},
        )
        builder.add_section(
            name="agent-intention-guideline",
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
  "condition": "{condition}",
  "is_agent_intention": "<BOOL>",
  "rewritten_condition": "<STR, include it is_agent_intention is True. Rewrite the condition in the format of "You are likely to (do something)" >",
}}
```
""",
            props={"condition": guideline.condition},
        )

        return builder

    async def _generate_agent_intention(
        self,
        guideline: GuidelineContent,
        temperature: float,
    ) -> AgentIntentionProposerSchema:
        prompt = await self._build_prompt(guideline, _baseline_shots)

        response = await self._schematic_generator.generate(
            prompt=prompt,
            hints={"temperature": temperature},
        )
        if not response.content:
            self._logger.warning("Completion:\nNo checks generated! This shouldn't happen.")

        return response.content

    def _format_shots(self, shots: Sequence[AgentIntentionProposerShot]) -> str:
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
    condition="You discuss a patient's medical record",
    action="Do not send any personal information",
)
example_1_shot = AgentIntentionProposerShot(
    description="",
    guideline=example_1_guideline,
    expected_result=AgentIntentionProposerSchema(
        condition=example_1_guideline.condition,
        is_agent_intention=True,
        rewritten_condition="You are likely to discuss a patient's medical record",
    ),
)

example_2_guideline = GuidelineContent(
    condition="You intend to interpret a contract or legal term",
    action="Add a disclaimer clarifying that the response is not legal advice",
)
example_2_shot = AgentIntentionProposerShot(
    description="",
    guideline=example_2_guideline,
    expected_result=AgentIntentionProposerSchema(
        condition=example_2_guideline.condition,
        is_agent_intention=True,
        rewritten_condition="You are likely to interpret a contract or legal term",
    ),
)

example_3_guideline = GuidelineContent(
    condition="You just confirmed that the order will be shipped to the customer",
    action="provide the package's tracking information",
)
example_3_shot = AgentIntentionProposerShot(
    description="",
    guideline=example_3_guideline,
    expected_result=AgentIntentionProposerSchema(
        condition=example_3_guideline.condition,
        is_agent_intention=False,
    ),
)

example_4_guideline = GuidelineContent(
    condition="You are likely to interpret a contract or legal term",
    action="Add a disclaimer clarifying that the response is not legal advice",
)
example_4_shot = AgentIntentionProposerShot(
    description="",
    guideline=example_4_guideline,
    expected_result=AgentIntentionProposerSchema(
        condition=example_4_guideline.condition,
        is_agent_intention=True,
        rewritten_condition="You are likely to interpret a contract or legal term",
    ),
)

example_5_guideline = GuidelineContent(
    condition="The customer is asking about the opening hours",
    action="Provide our opening hours as described on out website",
)
example_5_shot = AgentIntentionProposerShot(
    description="",
    guideline=example_5_guideline,
    expected_result=AgentIntentionProposerSchema(
        condition=example_5_guideline.condition,
        is_agent_intention=False,
    ),
)
_baseline_shots: Sequence[AgentIntentionProposerShot] = [
    example_1_shot,
    example_2_shot,
    example_3_shot,
    example_4_shot,
    example_5_shot,
]

shot_collection = ShotCollection[AgentIntentionProposerShot](_baseline_shots)
