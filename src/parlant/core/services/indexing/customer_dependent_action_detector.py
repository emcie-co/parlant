from typing import Optional
from parlant.core.common import DefaultBaseModel
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.guidelines import GuidelineContent
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.services.indexing.common import ProgressReport
from parlant.core.services.tools.service_registry import ServiceRegistry


class CustomerDependentActionProposition(DefaultBaseModel):
    is_customer_dependent: bool
    customer_action: Optional[str] = ""
    agent_action: Optional[str] = ""


class CustomerDependentActionSchema(
    DefaultBaseModel
):  # TODO register everywhere where guideline_is_continuous (or whatever the name is) is registered
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

    async def _build_prompt(  # TODO write, add shots
        self,
        guideline: GuidelineContent,
    ) -> PromptBuilder:
        builder = PromptBuilder()

        builder.add_section(
            name="guideline-continuous-proposer-general-instructions",
            template="""
In our system, the behavior of a conversational AI agent is guided by "guidelines". The agent makes use of these guidelines whenever it interacts with a user (also referred to as the customer).
Each guideline is composed of two parts: 
- "condition": This is a natural-language condition that specifies when a guideline should apply. We look at each conversation at any particular state, and we test against this condition to understand 
if we should have this guideline participate in generating the next reply to the user.
- "action": This is a natural-language instruction that should be followed by the agent whenever the "condition" part of the guideline applies to the conversation in its particular state.
Any instruction described here applies only to the agent, and not to the user.

...
""",
        )

        builder.add_section(
            name="guideline-continuous-proposer-guideline",
            template="""
Guideline
-----------
condition: {condition}
action: {action}
+""",
            props={"condition": guideline.condition, "action": guideline.action},
        )

        builder.add_section(
            name="guideline-action-proposer-output-format",
            template="""
Use the following format to evaluate wether the guideline has a customer dependent action:
Expected output (JSON):
```json
{{
  "is_customer_dependent": "<BOOL>",
  "customer_action": "<STR, the portion of the action that applies to the customer>",
    "agent_action": "<STR, the portion of the action that applies to the agent>"
}}
```
""",
        )

        return builder

    async def _generate_customer_dependent(
        self,
        guideline: GuidelineContent,
    ) -> CustomerDependentActionSchema:
        prompt = await self._build_prompt(guideline)

        response = await self._schematic_generator.generate(
            prompt=prompt,
            hints={"temperature": 0.0},
        )

        return response.content
