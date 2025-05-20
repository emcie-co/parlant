from typing import Optional
from parlant.core.common import DefaultBaseModel
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.guidelines import GuidelineContent
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.services.indexing.common import ProgressReport
from parlant.core.services.tools.service_registry import ServiceRegistry


class GuidelineContinuousProposition(DefaultBaseModel):
    is_continuous: bool


class GuidelineContinuousPropositionSchema(DefaultBaseModel):
    reason: str
    is_continuous: bool


class GuidelineContinuousProposer:
    def __init__(
        self,
        logger: Logger,
        schematic_generator: SchematicGenerator[GuidelineContinuousPropositionSchema],
        service_registry: ServiceRegistry,
    ) -> None:
        self._logger = logger
        self._schematic_generator = schematic_generator
        self._service_registry = service_registry

    async def propose_continuous(
        self,
        guideline: GuidelineContent,
        progress_report: Optional[ProgressReport] = None,
    ) -> GuidelineContinuousProposition:
        if progress_report:
            await progress_report.stretch(1)

        with self._logger.scope("GuidelineContinuousProposer"):
            proposition = await self._generate_continuous(guideline)

        if progress_report:
            await progress_report.increment(1)

        return GuidelineContinuousProposition(
            is_continuous=proposition.is_continuous,
        )

    async def _build_prompt(
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

A condition typically no longer applies if its corresponding action has already been executed. However, for actions that involve continuous behavior (e.g., "do not ask the user for their age",
or guidelines involving the language the agent should use) there’s no clear point at which the action is considered complete, since it involves behavior that must be consistently maintained. 
These guidelines will be called ‘continuous’.

Your task is to evaluate if a given guideline is continuous.
""",
        )

        builder.add_section(
            name="guideline-continuous-proposer-notes",
            template="""
Note that:
    1. If a guideline's condition has multiple requirements, mark it as continuous if at least one of them is continuous. Actions like "tell the customer they are pretty and ensure all communications are polite and supportive." 
    should be marked as continuous, since 'ensure all communications are polite and supportive' is continuous.
    2. Actions that forbid certain behaviors are generally considered continuous, as they must be consistently upheld throughout the conversation. Unlike tasks with an end point, 
    forbidden actions remain active throughout to ensure ongoing compliance.
    3. Guidelines that only require you to say a specific thing are generally not continuous. Once you said the required thing - the guideline is fulfilled.
    4. Some guidelines may involve actions that unfold over multiple steps and require several responses to complete. These actions might require ongoing interaction with the user throughout the conversation. 
    However, if the steps can be fully completed at some point in the exchange, the guideline should NOT be considered continuous — since the action, once fulfilled, does not need to be repeated.
""",
        )

        builder.add_section(
            name="guideline-continuous-proposer-examples",
            template="""
Examples of continuous guidelines:
    - Guideline that prohibits certain behavior (e.g., "do not ask the user their age").
        This must be upheld throughout the interaction, not just once.
    - Guideline that involves the agent's style, tone, or language (e.g., "speak in a friendly tone").
        The agent must maintain this across the whole conversation.
Examples of non continuous guidelines:
    - Guide the user through some process. (e.g., "help the user with the account setup process")
        This involves several steps that need to be completed, but once the process finished, the guideline is fulfilled and doesn't need to be repeated.

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
Use the following format to evaluate wether the guideline is continuous
Expected output (JSON):
```json
{{
  "reason": "<SHORT RATIONAL>",
  "is_continuous": "<BOOL>"
}}
```
""",
        )

        return builder

    async def _generate_continuous(
        self,
        guideline: GuidelineContent,
    ) -> GuidelineContinuousPropositionSchema:
        prompt = await self._build_prompt(guideline)

        response = await self._schematic_generator.generate(
            prompt=prompt,
            hints={"temperature": 0.0},
        )

        return response.content
