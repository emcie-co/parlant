from dataclasses import dataclass
from typing import Optional, Sequence
from parlant.core.common import DefaultBaseModel
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.guidelines import GuidelineContent
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.services.indexing.common import ProgressReport
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.shots import Shot, ShotCollection


class CustomerDependentActionProposition(DefaultBaseModel):
    is_customer_dependent: bool
    customer_action: Optional[str] = ""
    agent_action: Optional[str] = ""


class CustomerDependentActionSchema(
    DefaultBaseModel
):  # TODO register everywhere where guideline_is_continuous (or whatever the name is) is registered
    action: str
    is_customer_dependent: bool
    customer_action: Optional[str] = ""
    agent_action: Optional[str] = ""


@dataclass
class CustomerDependentActionShot(Shot):
    guideline: GuidelineContent
    is_customer_dependent: bool
    customer_action: Optional[str] = ""
    agent_action: Optional[str] = ""


class CustomerDependentActionDetector:
    def __init__(
        self,
        logger: Logger,
        schematic_generator: SchematicGenerator[CustomerDependentActionSchema],
        service_registry: ServiceRegistry,
    ) -> None:
        self._logger = logger
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
            proposition = await self._generate_customer_dependent(guideline)

        if progress_report:
            await progress_report.increment(1)

        return CustomerDependentActionProposition(
            is_customer_dependent=proposition.is_customer_dependent,
            customer_action=proposition.customer_action,
            agent_action=proposition.agent_action,
        )

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
- "condition": This is a natural-language condition that specifies when a guideline should apply. We look at each conversation at any particular state, and we test against this condition to understand 
if we should have this guideline participate in generating the next reply to the user.
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
Your role is to evaluate whether a given guideline has an action which requires something from the customer for the action to be completed.
Actions that require something from the customer are called "customer dependent actions".
Later in this prompt, you will be provided a single guideline. The guideline's condition is provided to you for context, but your decision should depend only on its action.
Ask yourself - what should be done for the action to be considered completed? Is it only something from the agent, or does it require something from the customer as well?

Two important edge cases you might encounter are:
 - Guidelines with an action that involves multiple steps (e.g., having the action "offer assistance to the customer and ask them for their account number) are considered customer dependent if at least one of the sub actions described are customer dependent.
 - When the action requires the agent to ask the customer a question, it is considered customer dependent, since we require the customer to answer for the action to be actually considered complete. One exception to this rule is an action that instructs the agent to ask a question as a pleasantry or without expecting an informative answer in return.


If you deem the action to be customer dependent, you are also required to split it into its portion that depends solely on the agent, and its portion that depends on the customer.

""",
        )
        builder.add_section(
            name="customer-dependent-action-shots",
            template=self._format_shots(shots),  # TODO I was here
        )
        builder.add_section(
            name="customer-dependent-action-detector-guideline",
            template="""
GUIDELINE
-----------
condition: {condition}
action: {action}
+""",
            props={"condition": guideline.condition, "action": guideline.action},
        )

        builder.add_section(
            name="guideline-action-proposer-output-format",
            template="""
OUTPUT FORMAT
-----------
Use the following format to evaluate wether the guideline has a customer dependent action:
Expected output (JSON):
```json
{{
  "action": "{action}",
  "is_customer_dependent": "<BOOL>",
  "customer_action": "<STR, the portion of the action that applies to the customer. Only necessary if is_customer_dependent is true>",
    "agent_action": "<STR, the portion of the action that applies to the agent. Only necessary if is_customer_dependent is true>"
}}
```
""",
            props={"action": guideline.action},
        )

        with open("customer dependent action detector prompt.txt", "w") as f:
            f.write(builder.build())

        return builder

    async def _generate_customer_dependent(
        self,
        guideline: GuidelineContent,
    ) -> CustomerDependentActionSchema:
        prompt = await self._build_prompt(guideline, _baseline_shots)

        response = await self._schematic_generator.generate(
            prompt=prompt,
            hints={"temperature": 0.0},
        )

        return response.content

    def _format_shots(self, shots: Sequence[CustomerDependentActionShot]) -> str:
        return "\n".join(
            [
                f"""
Example {i}: {shot.description}
"""
                for i, shot in enumerate(shots, start=1)
            ]
        )


example_1_shot = CustomerDependentActionShot(
    description="A guideline with a customer dependent action",
    guideline=GuidelineContent(
        condition="the customer wishes to submit an order",
        action="ask for their account number and shipping address. Inform them that it would take 3-5 business days.",
    ),
    is_customer_dependent=True,
    customer_action="provide their account number and shipping address",
    agent_action="ask for the customer's account number and shipping address, and inform them that it would take 3-5 business days.",
)

example_2_shot = CustomerDependentActionShot(
    description="A guideline whose action involves a question, but is not customer dependent",
    guideline=GuidelineContent(
        condition="asked 'whats up dog'", action="reply with 'nothing much, what's up with you?'"
    ),
    is_customer_dependent=False,
)

_baseline_shots: Sequence[CustomerDependentActionShot] = [
    example_1_shot,
    example_2_shot,
]

shot_collection = ShotCollection[CustomerDependentActionShot](_baseline_shots)
