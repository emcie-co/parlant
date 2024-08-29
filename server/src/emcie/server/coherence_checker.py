from abc import ABC, abstractmethod
import asyncio
from datetime import datetime, timezone
from enum import Enum
from itertools import chain
import json
from typing import NewType, Sequence
from more_itertools import chunked
from tenacity import retry, stop_after_attempt, wait_fixed

from emcie.server.base_models import DefaultBaseModel
from emcie.server.core.guidelines import Guideline, GuidelineId
from emcie.server.engines.alpha.utils import make_llm_client
from emcie.server.logger import Logger

CoherenceContradictionId = NewType("CoherenceContradictionId", str)

LLM_RETRY_WAIT_TIME_SECONDS = 3.5
LLM_MAX_RETRIES = 100
EVALUATION_BATCH_SIZE = 5


class ContradictionType(Enum):
    HIERARCHICAL = "Hierarchical Contradiction"
    PARALLEL = "Parallel Contradiction"
    TEMPORAL = "Temporal Contradiction"
    CONTEXTUAL = "Contextual Contradiction"


class ContradictionTest(DefaultBaseModel):
    contradiction_type: ContradictionType
    existing_guideline_id: GuidelineId
    proposed_guideline_id: GuidelineId
    severity: int
    rationale: str
    creation_utc: datetime


def _filter_unique_contradictions(
    contradictions: Sequence[ContradictionTest],
) -> Sequence[ContradictionTest]:
    """
    Filter unique contradictions based on the combination of existing and proposed guidelines.

    Args:
        contradictions: Sequence of Contradiction objects to filter.

    Returns:
        Sequence of unique Contradiction objects.
    """

    def _generate_key(g1_id: GuidelineId, g2_id: GuidelineId) -> tuple[GuidelineId, GuidelineId]:
        _sorted_guidelines_tuple = tuple(sorted((g1_id, g2_id)))
        return _sorted_guidelines_tuple  # type: ignore

    seen_keys = set()
    unique_contradictions = []
    for contradiction in contradictions:
        key = _generate_key(
            contradiction.existing_guideline_id, contradiction.proposed_guideline_id
        )
        if key not in seen_keys:
            seen_keys.add(key)
            unique_contradictions.append(contradiction)
    return unique_contradictions


class ContradictionEvaluator(ABC):
    @abstractmethod
    async def evaluate(
        self,
        proposed_guidelines: Sequence[Guideline],
        existing_guidelines: Sequence[Guideline] = [],
    ) -> Sequence[ContradictionTest]: ...


class ContradictionEvaluatorBase(ABC):
    def __init__(
        self,
        logger: Logger,
        contradiction_type: ContradictionType,
    ) -> None:
        self.logger = logger

        self._llm_client = make_llm_client("openai")
        self.contradiction_type = contradiction_type
        self.contradiction_response_outcome_key = (
            f'{self.contradiction_type.value.replace(" ", "_").lower()}s'
        )

    async def evaluate(
        self,
        proposed_guidelines: Sequence[Guideline],
        existing_guidelines: Sequence[Guideline] = [],
    ) -> Sequence[ContradictionTest]:
        existing_guideline_list = list(existing_guidelines)
        proposed_guidelines_list = list(proposed_guidelines)
        tasks = []

        for i, proposed_guideline in enumerate(proposed_guidelines):
            filtered_existing_guidelines = [
                g for g in proposed_guidelines_list[i + 1 :] + existing_guideline_list
            ]
            guideline_batches = chunked(filtered_existing_guidelines, EVALUATION_BATCH_SIZE)
            tasks.extend(
                [
                    asyncio.create_task(self._process_proposed_guideline(proposed_guideline, batch))
                    for batch in guideline_batches
                ]
            )
        with self.logger.operation(
            f"Evaluating {self.contradiction_type} for {len(tasks)} "
            f"batches (batch size={EVALUATION_BATCH_SIZE})",
        ):
            contradictions = list(chain.from_iterable(await asyncio.gather(*tasks)))

        distinct_contradictions = _filter_unique_contradictions(contradictions)
        return distinct_contradictions

    async def _process_proposed_guideline(
        self,
        proposed_guideline: Guideline,
        existing_guidelines: Sequence[Guideline],
    ) -> Sequence[ContradictionTest]:
        prompt = self._format_contradiction_prompt(
            proposed_guideline,
            list(existing_guidelines),
        )
        contradictions = await self._generate_contradictions(prompt)
        return contradictions

    @abstractmethod
    def _format_contradiction_type_definition(self) -> str: ...

    @abstractmethod
    def _format_contradiction_type_examples(self) -> str: ...

    def _format_contradiction_prompt(
        self,
        proposed_guideline: Guideline,
        existing_guidelines: list[Guideline],
    ) -> str:
        existing_guidelines_string = "\n\t".join(
            f"{i}) {{id: {g.id}, guideline: When {g.predicate}, then {g.content}}}"
            for i, g in enumerate(existing_guidelines, start=1)
        )
        proposed_guideline_string = (
            f"{{id: {proposed_guideline.id}, "
            f"guideline: When {proposed_guideline.predicate}, then {proposed_guideline.content}}}"
        )
        result_structure = [
            {
                "compared_guideline_id": g.id,
                "guideline_under_test_id": proposed_guideline.id,
                "severity_level": "<Severity Level (1-10): Indicates the intensity of the "
                "contradiction arising from overlapping conditions>",
                "rationale": "<Concise explanation of why the Guideline Under Test and the "
                f"compared guideline exhibit a {self.contradiction_type.value}>",
            }
            for g in existing_guidelines
        ]

        return f"""
### Definition of {self.contradiction_type.value}:

{self._format_contradiction_type_definition()}

**Objective**: Evaluate potential {self.contradiction_type.value}s between the set of existing guidelines and the proposed guideline.

**Task Description**:
1. **Input**:
   - Guideline Comparison Set: ###
    {existing_guidelines_string}
   ###
   - Guideline Under Test: ###
   {proposed_guideline_string}
   ###

2. **Process**:
   - Compare each of the {len(existing_guidelines)} guidelines in the Guideline Comparison Set with the Guideline Under Test.
   - Determine if there is a {self.contradiction_type.value}, where the Guideline Under Test is more specific and directly contradicts a more general guideline from the Guideline Comparison Set.
   - If no contradiction is detected, set the severity_level to 1 to indicate minimal or no contradiction.


3. **Output**:
   - A list of results, each item detailing a potential contradiction, structured as follows:
     ```json
     {{
         "{self.contradiction_response_outcome_key}":
            {result_structure}
     }}
     ```

### Examples of Evaluations:

{self._format_contradiction_type_examples()}
"""  # noqa

    @retry(wait=wait_fixed(LLM_RETRY_WAIT_TIME_SECONDS), stop=stop_after_attempt(LLM_MAX_RETRIES))
    async def _generate_contradictions(
        self,
        prompt: str,
    ) -> Sequence[ContradictionTest]:
        response = await self._llm_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o",
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""

        json_content = json.loads(content)[self.contradiction_response_outcome_key]

        contradictions = [
            ContradictionTest(
                contradiction_type=self.contradiction_type,
                existing_guideline_id=json_contradiction["compared_guideline_id"],
                proposed_guideline_id=json_contradiction["guideline_under_test_id"],
                severity=json_contradiction["severity_level"],
                rationale=json_contradiction["rationale"],
                creation_utc=datetime.now(timezone.utc),
            )
            for json_contradiction in json_content
        ]

        return contradictions


class HierarchicalContradictionEvaluator(ContradictionEvaluatorBase):
    def __init__(
        self,
        logger: Logger,
    ) -> None:
        super().__init__(logger, ContradictionType.HIERARCHICAL)

    def _format_contradiction_type_definition(self) -> str:
        return """
Hierarchical Coherence Contradiction arises when there are multiple layers of guidelines, with one being more specific or detailed than the other.
This type of Contradiction occurs when the application of a general guideline is contradicted by a more specific guideline under certain conditions, leading to inconsistencies in decision-making.
"""  # noqa

    def _format_contradiction_type_examples(self) -> str:
        return f"""
#### Example #1:
- **Compared Guideline**: {{"id": 3, "guideline": "When a customer orders any item, Then prioritize shipping based on customer loyalty level."}}
- **Guideline Under Test**: {{"id": 4, "guideline": "When a customer orders a high-demand item, Then ship immediately, regardless of loyalty level."}}
- **Expected Result**:
     ```json
     {{
         "{self.contradiction_response_outcome_key}": [
             {{
                 "compared_guideline_id": "3",
                 "guideline_under_test_id": "4",
                 "severity_level": 9,
                 "rationale": "Shipping high-demand items immediately contradicts the policy of prioritizing shipments based on loyalty."
             }}
         ]
     }}
     ```

#### Example #2:
- **Compared Guideline**: {{"id": 1, "guideline": "When an employee qualifies for any reward, Then distribute rewards based on standard performance metrics."}}
- **Guideline Under Test**: {{"id": 2, "guideline": "When an employee excels in a critical project, Then offer additional rewards beyond standard metrics."}}
- **Expected Result**:
     ```json
     {{
        "{self.contradiction_response_outcome_key}": [
             {{
                 "compared_guideline_id": "1",
                 "guideline_under_test_id": "2",
                 "severity_level": 8,
                 "rationale": "Offering extra rewards for specific projects directly conflicts with the uniform application of standard performance metrics."
             }}
         ]
     }}
     ```

#### Example #3:
- **Compared Guideline**: {{"id": 5, "guideline": "When a customer subscribes to a yearly plan, Then offer a 10% discount on the subscription fee."}}
- **Guideline Under Test**: {{"id": 6, "guideline": "When a customer subscribes to any plan during a promotional period, Then offer an additional 5% discount on the subscription fee."}}
- **Expected Result**:
     ```json
     {{
        "{self.contradiction_response_outcome_key}": [
             {{
                 "compared_guideline_id": "5",
                 "guideline_under_test_id": "6",
                 "severity_level": 1,
                 "rationale": "The policies to offer discounts for yearly subscriptions and additional discounts during promotional periods complement each other rather than contradict. Both discounts can be applied simultaneously without undermining one another, enhancing the overall attractiveness of the subscription offers during promotions."
             }}
         ]
     }}
     ```

#### Example #4:
- **Compared Guideline**: {{"id": 7, "guideline": "When there is a software update, Then deploy it within 48 hours."}}
- **Guideline Under Test**: {{"id": 8, "guideline": "When a software update includes major changes affecting user interfaces, Then delay deployment for additional user training."}}
- **Expected Result**:
     ```json
     {{
        "{self.contradiction_response_outcome_key}": [
             {{
                 "compared_guideline_id": "7",
                 "guideline_under_test_id": "8",
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
    ) -> None:
        super().__init__(logger, ContradictionType.PARALLEL)

    def _format_contradiction_type_definition(self) -> str:
        return """
Parallel Contradiction occurs when two guidelines of equal specificity lead to contradictory actions.
This happens when conditions for both guidelines are met simultaneously, without a clear resolution mechanism to prioritize one over the other.
"""  # noqa

    def _format_contradiction_type_examples(self) -> str:
        return f"""
#### Example #1:
- **Compared Guideline**: {{"id": 1, "guideline": "When a customer returns an item within 30 days, Then issue a full refund."}}
- **Guideline Under Test**: {{"id": 2, "guideline": "When the returned item is a special order, Then do not offer refunds."}}
- **Expected Result**:
     ```json
     {{
        "{self.contradiction_response_outcome_key}": [
             {{
                 "compared_guideline_id": "1",
                 "guideline_under_test_id": "2",
                 "severity_level": 9,
                 "rationale": "Refund policy conflict: special orders returned within 30 days challenge the standard refund guideline, causing potential customer confusion."
             }}
         ]
     }}
     ```

#### Example #2:
- **Compared Guideline**: {{"id": 3, "guideline": "When a project deadline is imminent, Then allocate all available resources to complete the project."}}
- **Guideline Under Test**: {{"id": 4, "guideline": "When multiple projects are nearing deadlines at the same time, Then distribute resources equally among projects."}}
- **Expected Result**:
     ```json
     {{
        "{self.contradiction_response_outcome_key}": [
             {{
                 "compared_guideline_id": "3",
                 "guideline_under_test_id": "4",
                 "severity_level": 8,
                 "rationale": "Resource allocation conflict: The need to focus resources on imminent deadlines clashes with equal distribution policies during multiple simultaneous project deadlines."
             }}
         ]
     }}
     ```

#### Example #3:
- **Compared Guideline**: {{"id": 5, "guideline": "When an employee requests flexible working hours, Then approve to support work-life balance."}}
- **Guideline Under Test**: {{"id": 6, "guideline": "When team collaboration is essential, Then require standard working hours for all team members."}}
- **Expected Result**:
     ```json
     {{
        "{self.contradiction_response_outcome_key}": [
             {{
                 "compared_guideline_id": "5",
                 "guideline_under_test_id": "6",
                 "severity_level": 7,
                 "rationale": "Flexible vs. standard hours conflict: Approving flexible hours contradicts the necessity for standard hours required for effective team collaboration."
             }}
         ]
     }}
     ```

#### Example #4:
- **Compared Guideline**: {{"id": 7, "guideline": "When a customer inquires about product features, Then provide detailed information and recommendations based on their needs."}}
- **Guideline Under Test**: {{"id": 8, "guideline": "When a customer asks about compatibility with other products, Then offer guidance on compatible products and configurations."}}
- **Expected Result**:
     ```json
     {{
        "{self.contradiction_response_outcome_key}": [
             {{
                 "compared_guideline_id": "7",
                 "guideline_under_test_id": "8",
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
    ) -> None:
        super().__init__(logger, ContradictionType.TEMPORAL)

    def _format_contradiction_type_definition(self) -> str:
        return """
Temporal Contradiction occurs when guidelines dependent on timing or sequence overlap in a way that leads to contradictions.
This arises from a lack of clear prioritization or differentiation between actions required at the same time.
"""  # noqa

    def _format_contradiction_type_examples(self) -> str:
        return f"""
#### Example #1:
- **Compared Guideline**: {{"id": 1, "guideline": "When it is the holiday season, Then apply discounts."}}
- **Guideline Under Test**: {{"id": 2, "guideline": "When it is the end-of-year sale period, Then apply no discounts."}}
- **Expected Result**:
     ```json
     {{
        "{self.contradiction_response_outcome_key}": [
             {{
                 "compared_guideline_id": "1",
                 "guideline_under_test_id": "2",
                 "severity_level": 9,
                 "rationale": "Applying discounts during the holiday season directly contradicts withholding them during overlapping end-of-year sales, creating inconsistent pricing strategies."
             }}
         ]
     }}
     ```

#### Example #2:
- **Compared Guideline**: {{"id": 3, "guideline": "When a product reaches its expiration date, Then mark it down for quick sale."}}
- **Guideline Under Test**: {{"id": 4, "guideline": "When a promotional campaign is active, Then maintain standard pricing to maximize campaign impact."}}
- **Expected Result**:
     ```json
     {{
        "{self.contradiction_response_outcome_key}": [
             {{
                 "compared_guideline_id": "3",
                 "guideline_under_test_id": "4",
                 "severity_level": 8,
                 "rationale": "Reducing prices for expiring products conflicts with maintaining standard pricing during promotional campaigns, causing pricing conflicts."
             }}
         ]
     }}
     ```

#### Example #3:
- **Compared Guideline**: {{"id": 5, "guideline": "When severe weather conditions are forecasted, Then activate emergency protocols and limit business operations."}}
- **Guideline Under Test**: {{"id": 6, "guideline": "When a major sales event is planned, Then ensure maximum operational capacity."}}
- **Expected Result**:
     ```json
     {{
        "{self.contradiction_response_outcome_key}": [
             {{
                 "compared_guideline_id": "5",
                 "guideline_under_test_id": "6",
                 "severity_level": 9,
                 "rationale": "Emergency protocol for severe weather contradicts the need for maximum capacity during major sales events, challenging management decisions."
             }}
         ]
     }}
     ```

#### Example #4:
- **Compared Guideline**: {{"id": 7, "guideline": "When customer service receives high call volumes, Then deploy additional staff to handle the influx."}}
- **Guideline Under Test**: {{"id": 8, "guideline": "When a new product launch is scheduled, Then prepare customer service for increased inquiries."}}
- **Expected Result**:
     ```json
     {{
        "{self.contradiction_response_outcome_key}": [
             {{
                 "compared_guideline_id": "7",
                 "guideline_under_test_id": "8",
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
    ) -> None:
        super().__init__(logger, ContradictionType.CONTEXTUAL)

    def _format_contradiction_type_definition(self) -> str:
        return """
Contextual Contradiction occurs when external conditions or operational contexts lead to contradictory actions.
These conflicts arise from different but potentially overlapping circumstances requiring actions that are valid under each specific context yet oppose each other.
"""  # noqa

    def _format_contradiction_type_examples(self) -> str:
        return f"""
#### Example #1:
- **Compared Guideline**: {{"id": 1, "guideline": "When operating in urban areas, Then offer free shipping."}}
- **Guideline Under Test**: {{"id": 2, "guideline": "When operational costs need to be minimized, Then restrict free shipping."}}
- **Expected Result**:
     ```json
     {{
         "contextual_contradictions": [
             {{
                 "compared_guideline_id": "1",
                 "guideline_under_test_id": "2",
                 "severity_level": 9,
                 "rationale": "Offering free shipping in urban areas directly conflicts with initiatives to minimize operational costs, leading to contradictory shipping policies when both conditions apply."
             }}
         ]
     }}
     ```

#### Example #2:
- **Compared Guideline**: {{"id": 3, "guideline": "When customer surveys indicate a preference for environmentally friendly products, Then shift production to eco-friendly materials."}}
- **Guideline Under Test**: {{"id": 4, "guideline": "When cost considerations drive decisions, Then continue using less expensive, traditional materials."}}
- **Expected Result**:
     ```json
     {{
        "{self.contradiction_response_outcome_key}": [
             {{
                 "compared_guideline_id": "3",
                 "guideline_under_test_id": "4",
                 "severity_level": 8,
                 "rationale": "The preference for eco-friendly products conflicts with cost-driven decisions to use cheaper materials, creating a strategic dilemma between sustainability and cost efficiency."
             }}
         ]
     }}
     ```

#### Example #3:
- **Compared Guideline**: {{"id": 5, "guideline": "When market data shows customer preference for high-end products, Then focus on premium product lines."}}
- **Guideline Under Test**: {{"id": 6, "guideline": "When internal strategy targets mass market appeal, Then increase production of lower-cost items."}}
- **Expected Result**:
     ```json
     {{
        "{self.contradiction_response_outcome_key}": [
             {{
                 "compared_guideline_id": "5",
                 "guideline_under_test_id": "6",
                 "severity_level": 9,
                 "rationale": "Targeting premium product lines based on customer preferences contradicts strategies to enhance mass market appeal with lower-cost items, presenting a strategic conflict."
             }}
         ]
     }}
     ```

#### Example #4:
- **Compared Guideline**: {{"id": 7, "guideline": "When a technology product is released, Then launch a marketing campaign to promote the new product."}}
- **Guideline Under Test**: {{"id": 8, "guideline": "When a new software update is released, Then send notifications to existing customers to encourage updates."}}
- **Expected Result**:
     ```json
     {{
        "{self.contradiction_response_outcome_key}": [
             {{
                 "compared_guideline_id": "7",
                 "guideline_under_test_id": "8",
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
    ) -> None:
        self.logger = logger

        self._llm_client = make_llm_client("openai")
        self.hierarchical_contradiction_evaluator = HierarchicalContradictionEvaluator(logger)
        self.parallel_contradiction_evaluator = ParallelContradictionEvaluator(logger)
        self.temporal_contradiction_evaluator = TemporalContradictionEvaluator(logger)
        self.contextual_contradiction_evaluator = ContextualContradictionEvaluator(logger)

    async def evaluate_coherence(
        self,
        proposed_guidelines: Sequence[Guideline],
        existing_guidelines: Sequence[Guideline],
    ) -> Sequence[ContradictionTest]:
        hierarchical_contradictions_task = self.hierarchical_contradiction_evaluator.evaluate(
            proposed_guidelines,
            existing_guidelines,
        )
        parallel_contradictions_task = self.parallel_contradiction_evaluator.evaluate(
            proposed_guidelines,
            existing_guidelines,
        )
        temporal_contradictions_task = self.temporal_contradiction_evaluator.evaluate(
            proposed_guidelines,
            existing_guidelines,
        )
        contextual_contradictions_task = self.contextual_contradiction_evaluator.evaluate(
            proposed_guidelines,
            existing_guidelines,
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
