from abc import ABC, abstractmethod
import asyncio
from datetime import datetime, timezone
from enum import Enum, auto
from itertools import chain
from typing import Sequence
from more_itertools import chunked
from tenacity import retry, stop_after_attempt, wait_fixed

from emcie.server.base_models import DefaultBaseModel
from emcie.server.core.common import UniqueId, generate_id
from emcie.server.core.generation.schematic_generators import BaseSchematicGenerator
from emcie.server.core.guidelines import GuidelineContent
from emcie.server.logger import Logger

LLM_RETRY_WAIT_TIME_SECONDS = 3.5
LLM_MAX_RETRIES = 100
EVALUATION_BATCH_SIZE = 5


class ContradictionKind(Enum):
    HIERARCHICAL = auto()
    PARALLEL = auto()
    TEMPORAL = auto()
    CONTEXTUAL = auto()

    def _describe(self) -> str:
        return {
            ContradictionKind.HIERARCHICAL: "Hierarchical Contradiction",
            ContradictionKind.PARALLEL: "Parallel Contradiction",
            ContradictionKind.TEMPORAL: "Temporal Contradiction",
            ContradictionKind.CONTEXTUAL: "Contextual Contradiction",
        }[self]


class ContradictionTestSchema(DefaultBaseModel):
    compared_guideline_id: UniqueId
    severity_level: int
    rationale: str


class ContradictionTestsSchema(DefaultBaseModel):
    contradictions: list[ContradictionTestSchema]


class ContradictionTest(DefaultBaseModel):
    kind: ContradictionKind
    guideline_a: GuidelineContent
    guideline_b: GuidelineContent
    severity: int
    rationale: str
    creation_utc: datetime


class ContradictionEvaluatorBase(ABC):
    def __init__(
        self,
        logger: Logger,
        contradiction_kind: ContradictionKind,
        schematic_generator: BaseSchematicGenerator[ContradictionTestsSchema],
    ) -> None:
        self.logger = logger
        self._schematic_generator = schematic_generator

        self.contradiction_kind = contradiction_kind

    async def evaluate(
        self,
        guidelines_to_evaluate: Sequence[GuidelineContent],
        comparison_guidelines: Sequence[GuidelineContent] = [],
    ) -> Sequence[ContradictionTest]:
        comparison_guidelines_list = list(comparison_guidelines)
        guidelines_to_evaluate_list = list(guidelines_to_evaluate)
        tasks = []

        for i, guideline_to_evaluate in enumerate(guidelines_to_evaluate):
            filtered_existing_guidelines = [
                g for g in guidelines_to_evaluate_list[i + 1 :] + comparison_guidelines_list
            ]
            guideline_batches = chunked(filtered_existing_guidelines, EVALUATION_BATCH_SIZE)
            tasks.extend(
                [
                    asyncio.create_task(
                        self._process_proposed_guideline(guideline_to_evaluate, batch)
                    )
                    for batch in guideline_batches
                ]
            )
        with self.logger.operation(
            f"Evaluating {self.contradiction_kind} for {len(tasks)} "
            f"batches (batch size={EVALUATION_BATCH_SIZE})",
        ):
            contradictions = list(chain.from_iterable(await asyncio.gather(*tasks)))

        return contradictions

    async def _process_proposed_guideline(
        self,
        guideline_to_evaluate: GuidelineContent,
        comparison_guidelines: Sequence[GuidelineContent],
    ) -> Sequence[ContradictionTest]:
        indexed_comparison_guidelines = {generate_id(): c for c in comparison_guidelines}
        prompt = self._format_contradiction_prompt(
            guideline_to_evaluate,
            indexed_comparison_guidelines,
        )
        contradictions = await self._generate_contradictions(
            guideline_to_evaluate, indexed_comparison_guidelines, prompt
        )
        return contradictions

    @abstractmethod
    def _format_contradiction_type_definition(self) -> str: ...

    @abstractmethod
    def _format_contradiction_type_examples(self) -> str: ...

    def _format_contradiction_prompt(
        self,
        guideline_to_evaluate: GuidelineContent,
        comparison_guidelines: dict[UniqueId, GuidelineContent],
    ) -> str:
        comparison_guidelines_string = "\n\t".join(
            f"{i}) {{id: {id}, guideline: When {comparison_guidelines[id].predicate}, then {comparison_guidelines[id].action}}}"
            for i, id in enumerate(comparison_guidelines, start=1)
        )
        guideline_to_evaluate_string = f"guideline: When {guideline_to_evaluate.predicate}, then {guideline_to_evaluate.action}}}"
        result_structure = [
            {
                "compared_guideline_id": id,
                "severity_level": "<Severity Level (1-10): Indicates the intensity of the "
                "contradiction arising from overlapping conditions>",
                "rationale": "<Concise explanation of why the Guideline A and the "
                f"Guideline B exhibit a {self.contradiction_kind._describe()}>",
            }
            for id in comparison_guidelines
        ]

        return f"""
### Definition of {self.contradiction_kind._describe()}:

{self._format_contradiction_type_definition()}

**Objective**: Evaluate potential {self.contradiction_kind._describe()}s between guideline A and the comparison guideline set.

**Task Description**:
1. **Input**:
   - Guideline Comparison Set: ###
    {comparison_guidelines_string}
   ###
   - Guideline A: ###
   {guideline_to_evaluate_string}
   ###

2. **Process**:
   - Compare each of the {len(comparison_guidelines)} guidelines in the Guideline Comparison Set with the Guideline A.
   - Determine if there is a {self.contradiction_kind._describe()}, where Guideline A directly contradicts a guideline from the Guideline Comparison Set when both guidelines are based on the same input.
   - If no contradiction is detected, set the severity_level to 1 to indicate minimal or no contradiction.


3. **Output**:
   - A list of results, each item detailing a potential contradiction, structured as follows:
     ```json
     {{
         "contradictions":
            {result_structure}
     }}
     ```

### Examples of Evaluations:

{self._format_contradiction_type_examples()}
"""  # noqa

    @retry(wait=wait_fixed(LLM_RETRY_WAIT_TIME_SECONDS), stop=stop_after_attempt(LLM_MAX_RETRIES))
    async def _generate_contradictions(
        self,
        guideline_to_evaluate: GuidelineContent,
        indexed_compared_guidelines: dict[UniqueId, GuidelineContent],
        prompt: str,
    ) -> Sequence[ContradictionTest]:
        contradiction_tests_response = await self._schematic_generator.generate(
            prompt=prompt,
            hints={"temperature": 0.0},
        )

        contradictions = [
            ContradictionTest(
                kind=self.contradiction_kind,
                guideline_a=guideline_to_evaluate,
                guideline_b=indexed_compared_guidelines[c.compared_guideline_id],
                severity=c.severity_level,
                rationale=c.rationale,
                creation_utc=datetime.now(timezone.utc),
            )
            for c in contradiction_tests_response.content.contradictions
        ]

        return contradictions


class HierarchicalContradictionEvaluator(ContradictionEvaluatorBase):
    def __init__(
        self,
        logger: Logger,
        schematic_generator: BaseSchematicGenerator[ContradictionTestsSchema],
    ) -> None:
        super().__init__(logger, ContradictionKind.HIERARCHICAL, schematic_generator)

    def _format_contradiction_type_definition(self) -> str:
        return """
Hierarchical Coherence Contradiction arises when there are multiple layers of guidelines, with one being more specific or detailed than the other.
This type of Contradiction occurs when the application of a general guideline is contradicted by a more specific guideline under certain conditions, leading to inconsistencies in decision-making.
"""  # noqa

    def _format_contradiction_type_examples(self) -> str:
        return f"""
#### Example #1:
- **Guideline A**: {{"id": 4, "guideline": "When a customer orders a high-demand item, Then ship immediately, regardless of loyalty level."}}
- **Guideline B**: {{"id": 3, "guideline": "When a customer orders any item, Then prioritize shipping based on customer loyalty level."}}
- **Expected Result**:
     ```json
     {{
         "contradictions": [
             {{
                 "compared_guideline_id": "3",
                 "severity_level": 9,
                 "rationale": "Shipping high-demand items immediately contradicts the policy of prioritizing shipments based on loyalty."
             }}
         ]
     }}
     ```

#### Example #2:
- **Guideline A**: {{"id": 2, "guideline": "When an employee excels in a critical project, Then offer additional rewards beyond standard metrics."}}
- **Guideline B**: {{"id": 1, "guideline": "When an employee qualifies for any reward, Then distribute rewards based on standard performance metrics."}}
- **Expected Result**:
     ```json
     {{
        "contradictions": [
             {{
                 "compared_guideline_id": "1",
                 "severity_level": 8,
                 "rationale": "Offering extra rewards for specific projects directly conflicts with the uniform application of standard performance metrics."
             }}
         ]
     }}
     ```

#### Example #3:
- **Guideline A**: {{"id": 6, "guideline": "When a customer subscribes to any plan during a promotional period, Then offer an additional 5% discount on the subscription fee."}}
- **Guideline B**: {{"id": 5, "guideline": "When a customer subscribes to a yearly plan, Then offer a 10% discount on the subscription fee."}}
- **Expected Result**:
     ```json
     {{
        "contradictions": [
             {{
                 "compared_guideline_id": "5",
                 "severity_level": 1,
                 "rationale": "The policies to offer discounts for yearly subscriptions and additional discounts during promotional periods complement each other rather than contradict. Both discounts can be applied simultaneously without undermining one another, enhancing the overall attractiveness of the subscription offers during promotions."
             }}
         ]
     }}
     ```

#### Example #4:
- **Guideline A**: {{"id": 8, "guideline": "When a software update includes major changes affecting user interfaces, Then delay deployment for additional user training."}}
- **Guideline B**: {{"id": 7, "guideline": "When there is a software update, Then deploy it within 48 hours."}}
- **Expected Result**:
     ```json
     {{
        "contradictions": [
             {{
                 "compared_guideline_id": "7",
                 "severity_level": 9,
                 "rationale": "The need for additional training on major UI changes necessitates delaying rapid deployments, causing a conflict with established update protocols."
             }}
         ]
     }}
     ```
"""  # noqa


class ParallelContradictionEvaluator(ContradictionEvaluatorBase):
    def __init__(
        self,
        logger: Logger,
        schematic_generator: BaseSchematicGenerator[ContradictionTestsSchema],
    ) -> None:
        super().__init__(logger, ContradictionKind.PARALLEL, schematic_generator)

    def _format_contradiction_type_definition(self) -> str:
        return """
Parallel Contradiction occurs when two guidelines of equal specificity lead to contradictory actions.
This happens when conditions for both guidelines are met simultaneously, without a clear resolution mechanism to prioritize one over the other.
"""  # noqa

    def _format_contradiction_type_examples(self) -> str:
        return f"""
#### Example #1:
- **Guideline A**: {{"id": 2, "guideline": "When the returned item is a special order, Then do not offer refunds."}}
- **Guideline B**: {{"id": 1, "guideline": "When a customer returns an item within 30 days, Then issue a full refund."}}
- **Expected Result**:
     ```json
     {{
        "contradictions": [
             {{
                "compared_guideline_id": "1",
                "severity_level": 9,
                "rationale": "Refund policy conflict: special orders returned within 30 days challenge the standard refund guideline, causing potential customer confusion."
             }}
         ]
     }}
     ```

#### Example #2:
- **Guideline A**: {{"id": 4, "guideline": "When multiple projects are nearing deadlines at the same time, Then distribute resources equally among projects."}}
- **Guideline B**: {{"id": 3, "guideline": "When a project deadline is imminent, Then allocate all available resources to complete the project."}}
- **Expected Result**:
     ```json
     {{
        "contradictions": [
             {{
                "compared_guideline_id": "3",
                "severity_level": 8,
                "rationale": "Resource allocation conflict: The need to focus resources on imminent deadlines clashes with equal distribution policies during multiple simultaneous project deadlines."
             }}
         ]
     }}
     ```

#### Example #3:
- **Guideline A**: {{"id": 6, "guideline": "When team collaboration is essential, Then require standard working hours for all team members."}}
- **Guideline B**: {{"id": 5, "guideline": "When an employee requests flexible working hours, Then approve to support work-life balance."}}
- **Expected Result**:
     ```json
     {{
        "contradictions": [
             {{
                 "compared_guideline_id": "5",
                 "severity_level": 7,
                 "rationale": "Flexible vs. standard hours conflict: Approving flexible hours contradicts the necessity for standard hours required for effective team collaboration."
             }}
         ]
     }}
     ```

#### Example #4:
- **Guideline A**: {{"id": 8, "guideline": "When a customer asks about compatibility with other products, Then offer guidance on compatible products and configurations."}}
- **Guideline B**: {{"id": 7, "guideline": "When a customer inquires about product features, Then provide detailed information and recommendations based on their needs."}}
- **Expected Result**:
     ```json
     {{
        "contradictions": [
             {{
                "compared_guideline_id": "7",
                "severity_level": 1,
                "rationale": "These guidelines complement each other by addressing different customer needs: detailed product information and specific compatibility advice."
             }}
         ]
     }}
     ```
"""  # noqa


class TemporalContradictionEvaluator(ContradictionEvaluatorBase):
    def __init__(
        self,
        logger: Logger,
        schematic_generator: BaseSchematicGenerator[ContradictionTestsSchema],
    ) -> None:
        super().__init__(logger, ContradictionKind.TEMPORAL, schematic_generator)

    def _format_contradiction_type_definition(self) -> str:
        return """
Temporal Contradiction occurs when guidelines dependent on timing or sequence overlap in a way that leads to contradictions.
This arises from a lack of clear prioritization or differentiation between actions required at the same time.
"""  # noqa

    def _format_contradiction_type_examples(self) -> str:
        return f"""
#### Example #1:
- **Guideline A**: {{"id": 2, "guideline": "When it is the end-of-year sale period, Then apply no discounts."}}
- **Guideline B**: {{"id": 1, "guideline": "When it is the holiday season, Then apply discounts."}}
- **Expected Result**:
     ```json
     {{
        "contradictions": [
             {{
                "compared_guideline_id": "1",
                "severity_level": 9,
                "rationale": "Applying discounts during the holiday season directly contradicts withholding them during overlapping end-of-year sales, creating inconsistent pricing strategies."
             }}
         ]
     }}
     ```

#### Example #2:
- **Guideline A**: {{"id": 4, "guideline": "When a promotional campaign is active, Then maintain standard pricing to maximize campaign impact."}}
- **Guideline B**: {{"id": 3, "guideline": "When a product reaches its expiration date, Then mark it down for quick sale."}}
- **Expected Result**:
     ```json
     {{
        "contradictions": [
             {{
                "compared_guideline_id": "3",
                "severity_level": 8,
                "rationale": "Reducing prices for expiring products conflicts with maintaining standard pricing during promotional campaigns, causing pricing conflicts."
             }}
         ]
     }}
     ```

#### Example #3:
- **Guideline A**: {{"id": 6, "guideline": "When a major sales event is planned, Then ensure maximum operational capacity."}}
- **Guideline B**: {{"id": 5, "guideline": "When severe weather conditions are forecasted, Then activate emergency protocols and limit business operations."}}
- **Expected Result**:
     ```json
     {{
        "contradictions": [
             {{
                "compared_guideline_id": "5",
                "severity_level": 9,
                "rationale": "Emergency protocol for severe weather contradicts the need for maximum capacity during major sales events, challenging management decisions."
             }}
         ]
     }}
     ```

#### Example #4:
- **Guideline A**: {{"id": 8, "guideline": "When a new product launch is scheduled, Then prepare customer service for increased inquiries."}}
- **Guideline B**: {{"id": 7, "guideline": "When customer service receives high call volumes, Then deploy additional staff to handle the influx."}}
- **Expected Result**:
     ```json
     {{
        "contradictions": [
             {{
                "compared_guideline_id": "7",
                "severity_level": 1,
                "rationale": "Both guidelines support enhancing customer service under different circumstances, effectively complementing each other without conflict."
             }}
         ]
     }}
     ```
"""  # noqa


class ContextualContradictionEvaluator(ContradictionEvaluatorBase):
    def __init__(
        self,
        logger: Logger,
        schematic_generator: BaseSchematicGenerator[ContradictionTestsSchema],
    ) -> None:
        super().__init__(logger, ContradictionKind.CONTEXTUAL, schematic_generator)

    def _format_contradiction_type_definition(self) -> str:
        return """
Contextual Contradiction occurs when external conditions or operational contexts lead to contradictory actions.
These conflicts arise from different but potentially overlapping circumstances requiring actions that are valid under each specific context yet oppose each other.
"""  # noqa

    def _format_contradiction_type_examples(self) -> str:
        return f"""
#### Example #1:
- **Guideline A**: {{"id": 2, "guideline": "When operational costs need to be minimized, Then restrict free shipping."}}
- **Guideline B**: {{"id": 1, "guideline": "When operating in urban areas, Then offer free shipping."}}
- **Expected Result**:
     ```json
     {{
         "contextual_contradictions": [
             {{
                "compared_guideline_id": "1",
                "severity_level": 9,
                "rationale": "Offering free shipping in urban areas directly conflicts with initiatives to minimize operational costs, leading to contradictory shipping policies when both conditions apply."
             }}
         ]
     }}
     ```

#### Example #2:
- **Guideline A**: {{"id": 4, "guideline": "When cost considerations drive decisions, Then continue using less expensive, traditional materials."}}
- **Guideline B**: {{"id": 3, "guideline": "When customer surveys indicate a preference for environmentally friendly products, Then shift production to eco-friendly materials."}}
- **Expected Result**:
     ```json
     {{
        "contradictions": [
             {{
                "compared_guideline_id": "3",
                "severity_level": 8,
                "rationale": "The preference for eco-friendly products conflicts with cost-driven decisions to use cheaper materials, creating a strategic dilemma between sustainability and cost efficiency."
             }}
         ]
     }}
     ```

#### Example #3:
- **Guideline A**: {{"id": 6, "guideline": "When internal strategy targets mass market appeal, Then increase production of lower-cost items."}}
- **Guideline B**: {{"id": 5, "guideline": "When market data shows customer preference for high-end products, Then focus on premium product lines."}}
- **Expected Result**:
     ```json
     {{
        "contradictions": [
             {{
                "compared_guideline_id": "5",
                "severity_level": 9,
                "rationale": "Targeting premium product lines based on customer preferences contradicts strategies to enhance mass market appeal with lower-cost items, presenting a strategic conflict."
             }}
         ]
     }}
     ```

#### Example #4:
- **Guideline A**: {{"id": 8, "guideline": "When a new software update is released, Then send notifications to existing customers to encourage updates."}}
- **Guideline B**: {{"id": 7, "guideline": "When a technology product is released, Then launch a marketing campaign to promote the new product."}}
- **Expected Result**:
     ```json
     {{
        "contradictions": [
             {{
                "compared_guideline_id": "7",
                "severity_level": 1,
                "rationale": "Marketing new products and notifying existing customers about updates serve complementary purposes without conflicting, effectively targeting different customer segments."
             }}
         ]
     }}
     ```
"""  # noqa


class CoherenceChecker:
    def __init__(
        self,
        logger: Logger,
        schematic_generator: BaseSchematicGenerator[ContradictionTestsSchema],
    ) -> None:
        self.logger = logger

        self.hierarchical_contradiction_evaluator = HierarchicalContradictionEvaluator(
            logger,
            schematic_generator,
        )
        self.parallel_contradiction_evaluator = ParallelContradictionEvaluator(
            logger,
            schematic_generator,
        )
        self.temporal_contradiction_evaluator = TemporalContradictionEvaluator(
            logger,
            schematic_generator,
        )
        self.contextual_contradiction_evaluator = ContextualContradictionEvaluator(
            logger,
            schematic_generator,
        )

    async def evaluate_coherence(
        self,
        guidelines_to_evaluate: Sequence[GuidelineContent],
        comparison_guidelines: Sequence[GuidelineContent] = [],
    ) -> Sequence[ContradictionTest]:
        hierarchical_contradictions_task = self.hierarchical_contradiction_evaluator.evaluate(
            guidelines_to_evaluate,
            comparison_guidelines,
        )
        parallel_contradictions_task = self.parallel_contradiction_evaluator.evaluate(
            guidelines_to_evaluate,
            comparison_guidelines,
        )
        temporal_contradictions_task = self.temporal_contradiction_evaluator.evaluate(
            guidelines_to_evaluate,
            comparison_guidelines,
        )
        contextual_contradictions_task = self.contextual_contradiction_evaluator.evaluate(
            guidelines_to_evaluate,
            comparison_guidelines,
        )
        combined_contradictions = list(
            chain.from_iterable(
                await asyncio.gather(
                    hierarchical_contradictions_task,
                    parallel_contradictions_task,
                    temporal_contradictions_task,
                    contextual_contradictions_task,
                )
            )
        )
        return combined_contradictions
