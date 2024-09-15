from dataclasses import dataclass
from lagom import Container
from pytest import fixture, mark

from emcie.server.core.agents import AgentId, AgentStore
from emcie.server.core.generation.schematic_generators import SchematicGenerator
from emcie.server.core.guidelines import GuidelineContent
from emcie.server.indexing.coherence_checker import (
    CoherenceChecker,
    ContextualContradictionEvaluator,
    ContradictionKind,
    ContradictionTestsSchema,
    HierarchicalContradictionEvaluator,
    ParallelContradictionEvaluator,
    TemporalContradictionEvaluator,
)
from emcie.server.logger import Logger

from tests.test_utilities import SyncAwaiter, nlp_test


@fixture
def agent_id(
    container: Container,
    sync_await: SyncAwaiter,
) -> AgentId:
    store = container[AgentStore]
    agent = sync_await(store.create_agent(name="test-agent"))
    return agent.id


@dataclass
class _TestContext:
    sync_await: SyncAwaiter
    container: Container
    agent_id: AgentId


@fixture
def context(
    sync_await: SyncAwaiter,
    container: Container,
    agent_id: AgentId,
) -> _TestContext:
    return _TestContext(sync_await, container, agent_id)


@fixture
def guidelines_with_contradictions() -> list[GuidelineContent]:
    guidelines: list[GuidelineContent] = []

    for guideline_params in [
        {
            "predicate": "A VIP customer requests a specific feature that aligns with their business needs but is not on the current product roadmap",  # noqa
            "action": "Escalate the request to product management for special consideration",
        },
        {
            "predicate": "Any customer requests a feature not available in the current version",
            "action": "Inform them about the product roadmap and upcoming features",
        },
        {
            "predicate": "Any customer reports a technical issue",
            "action": "Queue the issue for resolution according to standard support protocols",
        },
        {
            "predicate": "The issue reported affects a critical operational feature for multiple clients",  # noqa
            "action": "Escalate immediately to the highest priority for resolution",
        },
        {
            "predicate": "Receiving feedback on a new feature",
            "action": "encourage users to adopt and adapt to the change as part of ongoing product",  # noqa
        },
    ]:
        guidelines.append(
            GuidelineContent(
                predicate=guideline_params["predicate"], action=guideline_params["action"]
            )
        )

    return guidelines


@fixture
def guidelines_without_contradictions() -> list[GuidelineContent]:
    guidelines: list[GuidelineContent] = []

    for guideline_params in [
        {
            "predicate": "A customer inquires about upgrading their service package",
            "action": "Provide information on available upgrade options and benefits",
        },
        {
            "predicate": "A customer needs assistance with understanding their billing statements",
            "action": "Guide them through the billing details and explain any charges",
        },
        {
            "predicate": "A customer expresses satisfaction with the service",
            "action": "encourage them to leave a review or testimonial",
        },
        {
            "predicate": "A customer refers another potential client",
            "action": "initiate the referral rewards process",
        },
        {
            "predicate": "A customer asks about the security of their data",
            "action": "Provide detailed information about the company’s security measures and certifications",  # noqa
        },
        {
            "predicate": "A customer inquires about compliance with specific regulations",
            "action": "Direct them to documentation detailing the company’s compliance with those regulations",  # noqa
        },
        {
            "predicate": "A customer requests faster support response times",
            "action": "Explain the standard response times and efforts to improve them",
        },
        {
            "predicate": "A customer compliments the service on social media",
            "action": "Thank them publicly and encourage them to share more about their positive experience",  # noqa
        },
        {
            "predicate": "A customer asks about the security of their data",
            "action": "Provide detailed information about the company’s security measures and certifications",  # noqa
        },
        {
            "predicate": "A customer inquires about compliance with specific regulations",
            "action": "Direct them to documentation detailing the company’s compliance with those regulations",  # noqa
        },
    ]:
        guidelines.append(
            GuidelineContent(
                predicate=guideline_params["predicate"], action=guideline_params["action"]
            )
        )

    return guidelines


@mark.parametrize(
    (
        "guideline_a_definition",
        "guideline_b_definition",
    ),
    [
        (
            {
                "predicate": "A VIP customer requests a specific feature that aligns with their business needs but is not on the current product roadmap",  # noqa
                "action": "Escalate the request to product management for special consideration",
            },
            {
                "predicate": "Any customer requests a feature not available in the current version",
                "action": "Inform them about the product roadmap and upcoming features",
            },
        ),
        (
            {
                "predicate": "Any customer reports a technical issue",
                "action": "Queue the issue for resolution according to standard support protocols",
            },
            {
                "predicate": "The issue reported affects a critical operational feature for multiple clients",  # noqa
                "action": "Escalate immediately to the highest priority for resolution",
            },
        ),
    ],
)
def test_that_hierarchical_evaluator_detects_contradictions(
    context: _TestContext,
    guideline_a_definition: dict[str, str],
    guideline_b_definition: dict[str, str],
) -> None:
    guideline_a = GuidelineContent(
        predicate=guideline_a_definition["predicate"], action=guideline_a_definition["action"]
    )

    guideline_b = GuidelineContent(
        predicate=guideline_b_definition["predicate"], action=guideline_b_definition["action"]
    )

    hierarchical_contradiction_evaluator = HierarchicalContradictionEvaluator(
        context.container[Logger],
        context.container[SchematicGenerator[ContradictionTestsSchema]],
    )

    contradiction_results = list(
        context.sync_await(
            hierarchical_contradiction_evaluator.evaluate(
                [guideline_a, guideline_b],
            )
        )
    )

    assert len(contradiction_results) == 1
    contradiction = contradiction_results[0]

    assert ContradictionKind(contradiction.kind) == ContradictionKind.HIERARCHICAL
    assert contradiction.severity >= 5

    assert context.sync_await(
        nlp_test(
            context.container[Logger],
            f"Here is an explanation of what {hierarchical_contradiction_evaluator.contradiction_kind._describe()} type is:"  # noqa
            f"{hierarchical_contradiction_evaluator._format_contradiction_type_definition()}"
            "Here are two behavioral guidelines:"
            "a semantic contradiction test was conducted, regarding the following two behavioral guidelines:"  # noqa
            f"1. {contradiction.guideline_a}\n"
            f"2. {contradiction.guideline_b}\n"
            "Here is the output explanation generated by the contradiction test:"
            f"{contradiction.rationale}",
            "A contradiction has been found and an explanation is provided as to why it is a "
            "hierarchical contradiction",
        )
    )


@mark.parametrize(
    (
        "guideline_a_definition",
        "guideline_b_definition",
    ),
    [
        (
            {
                "predicate": "A customer inquires about upgrading their service package",
                "action": "Provide information on available upgrade options and benefits",
            },
            {
                "predicate": "A customer needs assistance with understanding their billing statements",  # noqa
                "action": "Guide them through the billing details and explain any charges",
            },
        ),
        (
            {
                "predicate": "A customer expresses satisfaction with the service",
                "action": "encourage them to leave a review or testimonial",
            },
            {
                "predicate": "A customer refers another potential client",
                "action": "initiate the referral rewards process",
            },
        ),
    ],
)
def test_that_hierarchical_evaluator_does_not_produce_false_positives(
    context: _TestContext,
    guideline_a_definition: dict[str, str],
    guideline_b_definition: dict[str, str],
) -> None:
    guideline_a = GuidelineContent(
        predicate=guideline_a_definition["predicate"], action=guideline_a_definition["action"]
    )

    guideline_b = GuidelineContent(
        predicate=guideline_b_definition["predicate"], action=guideline_b_definition["action"]
    )

    hierarchical_contradiction_evaluator = HierarchicalContradictionEvaluator(
        context.container[Logger],
        context.container[SchematicGenerator[ContradictionTestsSchema]],
    )

    contradiction_results = list(
        context.sync_await(
            hierarchical_contradiction_evaluator.evaluate(
                [guideline_a, guideline_b],
            )
        )
    )

    assert len(contradiction_results) == 1
    contradiction = contradiction_results[0]

    assert ContradictionKind(contradiction.kind) == ContradictionKind.HIERARCHICAL
    assert contradiction.severity < 5

    assert context.sync_await(
        nlp_test(
            context.container[Logger],
            f"Here is an explanation of what {hierarchical_contradiction_evaluator.contradiction_kind._describe()} type is:"  # noqa
            f"{hierarchical_contradiction_evaluator._format_contradiction_type_definition()}"
            "Here are two behavioral guidelines:"
            "a semantic contradiction test was conducted, regarding the following two behavioral guidelines:"  # noqa
            f"1. {contradiction.guideline_a}\n"
            f"2. {contradiction.guideline_b}\n"
            "Here is the output explanation generated by the contradiction test:"
            f"{contradiction.rationale}",
            "No contradiction has been found between the two behavioral guidelines",
        )
    )


@mark.parametrize(
    (
        "guideline_a_definition",
        "guideline_b_definition",
    ),
    [
        (
            {
                "predicate": "A customer exceeds their data storage limit",
                "action": "Prompt them to upgrade their subscription plan",
            },
            {
                "predicate": "Promoting customer retention and satisfaction",
                "action": "Offer a temporary data limit extension without requiring an upgrade",
            },
        ),
        (
            {
                "predicate": "Receiving feedback on a new feature",
                "action": "encourage users to adopt and adapt to the change as part of ongoing product",  # noqa
            },
            {
                "predicate": "Users express significant resistance to a new feature",
                "action": "Roll back or offer the option to revert to previous settings",
            },
        ),
    ],
)
def test_that_parallel_evaluator_detects_contradictions(
    context: _TestContext,
    guideline_a_definition: dict[str, str],
    guideline_b_definition: dict[str, str],
) -> None:
    guideline_a = GuidelineContent(
        predicate=guideline_a_definition["predicate"], action=guideline_a_definition["action"]
    )

    guideline_b = GuidelineContent(
        predicate=guideline_b_definition["predicate"], action=guideline_b_definition["action"]
    )

    parallel_contradiction_evaluator = ParallelContradictionEvaluator(
        context.container[Logger],
        context.container[SchematicGenerator[ContradictionTestsSchema]],
    )
    contradiction_results = list(
        context.sync_await(
            parallel_contradiction_evaluator.evaluate(
                [guideline_a, guideline_b],
                [],
            )
        )
    )

    assert len(contradiction_results) == 1
    contradiction = contradiction_results[0]

    assert ContradictionKind(contradiction.kind) == ContradictionKind.PARALLEL
    assert contradiction.severity >= 5

    assert context.sync_await(
        nlp_test(
            context.container[Logger],
            f"Here is an explanation of what {parallel_contradiction_evaluator.contradiction_kind._describe()} type is:"  # noqa
            f"{parallel_contradiction_evaluator._format_contradiction_type_definition()}"
            "Here are two behavioral guidelines:"
            "a semantic contradiction test was conducted, regarding the following two behavioral guidelines:"  # noqa
            f"1. {contradiction.guideline_a}\n"
            f"2. {contradiction.guideline_b}\n"
            "Here is the output explanation generated by the contradiction test:"
            f"{contradiction.rationale}",
            "A contradiction has been found and an explanation is provided as to why it is a "
            "parallel contradiction",
        )
    )


@mark.parametrize(
    (
        "guideline_a_definition",
        "guideline_b_definition",
    ),
    [
        (
            {
                "predicate": "A customer asks about the security of their data",
                "action": "Provide detailed information about the company’s security measures and certifications",  # noqa
            },
            {
                "predicate": "A customer inquires about compliance with specific regulations",
                "action": "Direct them to documentation detailing the company’s compliance with those regulations",  # noqa
            },  # noqa
        ),
        (
            {
                "predicate": "A customer requests faster support response times",
                "action": "Explain the standard response times and efforts to improve them",
            },
            {
                "predicate": "A customer compliments the service on social media",
                "action": "Thank them publicly and encourage them to share more about their positive experience",  # noqa
            },
        ),
    ],
)
def test_that_parallel_evaluator_does_not_produce_false_positives(
    context: _TestContext,
    guideline_a_definition: dict[str, str],
    guideline_b_definition: dict[str, str],
) -> None:
    guideline_a = GuidelineContent(
        predicate=guideline_a_definition["predicate"], action=guideline_a_definition["action"]
    )

    guideline_b = GuidelineContent(
        predicate=guideline_b_definition["predicate"], action=guideline_b_definition["action"]
    )

    parallel_contradiction_evaluator = ParallelContradictionEvaluator(
        context.container[Logger],
        context.container[SchematicGenerator[ContradictionTestsSchema]],
    )
    contradiction_results = list(
        context.sync_await(
            parallel_contradiction_evaluator.evaluate(
                [guideline_a, guideline_b],
                [],
            )
        )
    )

    assert len(contradiction_results) == 1
    contradiction = contradiction_results[0]

    assert ContradictionKind(contradiction.kind) == ContradictionKind.PARALLEL
    assert contradiction.severity < 5

    assert context.sync_await(
        nlp_test(
            context.container[Logger],
            f"Here is an explanation of what {parallel_contradiction_evaluator.contradiction_kind._describe()} type is:"  # noqa
            f"{parallel_contradiction_evaluator._format_contradiction_type_definition()}"
            "Here are two behavioral guidelines:"
            "a semantic contradiction test was conducted, regarding the following two behavioral guidelines:"  # noqa
            f"1. {contradiction.guideline_a}\n"
            f"2. {contradiction.guideline_b}\n"
            "Here is the output explanation generated by the contradiction test:"
            f"{contradiction.rationale}",
            "No contradiction has been found between the two behavioral guidelines",
        )
    )


@mark.parametrize(
    (
        "guideline_a_definition",
        "guideline_b_definition",
    ),
    [
        (
            {
                "predicate": "A new software update is scheduled for release",
                "action": "Roll it out to all users to ensure everyone has the latest version",
            },
            {
                "predicate": "Key clients are in the middle of a critical project",
                "action": "Delay software updates to avoid disrupting their operations",
            },
        ),
        (
            {
                "predicate": "The financial quarter ends",
                "action": "Finalize all pending transactions and close the books",
            },
            {
                "predicate": "A new financial regulation is implemented at the end of the quarter",
                "action": "re-evaluate all transactions from that quarter before closing the books",  # noqa
            },
        ),
    ],
)
def test_that_temporal_evaluator_detects_contradictions(
    context: _TestContext,
    guideline_a_definition: dict[str, str],
    guideline_b_definition: dict[str, str],
) -> None:
    guideline_a = GuidelineContent(
        predicate=guideline_a_definition["predicate"], action=guideline_a_definition["action"]
    )

    guideline_b = GuidelineContent(
        predicate=guideline_b_definition["predicate"], action=guideline_b_definition["action"]
    )

    temporal_contradiction_evaluator = TemporalContradictionEvaluator(
        context.container[Logger],
        context.container[SchematicGenerator[ContradictionTestsSchema]],
    )
    contradiction_results = list(
        context.sync_await(
            temporal_contradiction_evaluator.evaluate(
                [guideline_a, guideline_b],
                [],
            )
        )
    )

    assert len(contradiction_results) == 1
    contradiction = contradiction_results[0]

    assert ContradictionKind(contradiction.kind) == ContradictionKind.TEMPORAL
    assert contradiction.severity >= 5

    assert context.sync_await(
        nlp_test(
            context.container[Logger],
            f"Here is an explanation of what {temporal_contradiction_evaluator.contradiction_kind._describe()} type is:\n"  # noqa
            f"{temporal_contradiction_evaluator._format_contradiction_type_definition()}"
            "Here are two behavioral guidelines:"
            "a semantic contradiction test was conducted, regarding the following two behavioral guidelines:"  # noqa
            f"1. {contradiction.guideline_a}\n"
            f"2. {contradiction.guideline_b}\n"
            "Here is the output explanation generated by the contradiction test:"
            f"{contradiction.rationale}",
            "A contradiction has been found and an explanation is provided as to why it is a "
            "temporal contradiction",
        )
    )


@mark.parametrize(
    (
        "guideline_a_definition",
        "guideline_b_definition",
    ),
    [
        (
            {
                "predicate": "A customer asks about the security of their data",
                "action": "Provide detailed information about the company’s security measures and certifications",  # noqa
            },
            {
                "predicate": "A customer inquires about compliance with specific regulations",
                "action": "Direct them to documentation detailing the company’s compliance with those regulations",  # noqa
            },  # noqa
        ),
        (
            {
                "predicate": "A customer requests faster support response times",
                "action": "Explain the standard response times and efforts to improve them",
            },
            {
                "predicate": "A customer compliments the service on social media",
                "action": "Thank them publicly and encourage them to share more about their positive experience",  # noqa
            },
        ),
    ],
)
def test_that_temporal_evaluator_does_not_produce_false_positives(
    context: _TestContext,
    guideline_a_definition: dict[str, str],
    guideline_b_definition: dict[str, str],
) -> None:
    guideline_a = GuidelineContent(
        predicate=guideline_a_definition["predicate"], action=guideline_a_definition["action"]
    )

    guideline_b = GuidelineContent(
        predicate=guideline_b_definition["predicate"], action=guideline_b_definition["action"]
    )

    temporal_contradiction_evaluator = TemporalContradictionEvaluator(
        context.container[Logger],
        context.container[SchematicGenerator[ContradictionTestsSchema]],
    )
    contradiction_results = list(
        context.sync_await(
            temporal_contradiction_evaluator.evaluate(
                [guideline_a, guideline_b],
                [],
            )
        )
    )

    assert len(contradiction_results) == 1
    contradiction = contradiction_results[0]

    assert ContradictionKind(contradiction.kind) == ContradictionKind.TEMPORAL
    assert contradiction.severity < 5

    assert context.sync_await(
        nlp_test(
            context.container[Logger],
            f"Here is an explanation of what {temporal_contradiction_evaluator.contradiction_kind._describe()} type is:"  # noqa
            f"{temporal_contradiction_evaluator._format_contradiction_type_definition()}"
            "Here are two behavioral guidelines:"
            "a semantic contradiction test was conducted, regarding the following two behavioral guidelines:"  # noqa
            f"1. {contradiction.guideline_a}\n"
            f"2. {contradiction.guideline_b}\n"
            "Here is the output explanation generated by the contradiction test:"
            f"{contradiction.rationale}",
            "No contradiction has been found between the two behavioral guidelines",
        )
    )


@mark.parametrize(
    (
        "guideline_a_definition",
        "guideline_b_definition",
    ),
    [
        (
            {
                "predicate": "A customer is located in a region with strict data sovereignty laws",
                "action": "Store and process all customer data locally as required by law",
            },
            {
                "predicate": "The company's policy is to centralize data processing in a single, cost-effective location",  # noqa
                "action": "Consolidate data handling to enhance efficiency",
            },
        ),
        (
            {
                "predicate": "A customer's contract is up for renewal during a market downturn",
                "action": "Offer discounts and incentives to ensure renewal",
            },
            {
                "predicate": "The company’s financial performance targets require maximizing revenue",  # noqa
                "action": "avoid discounts and push for higher-priced contracts",
            },
        ),
    ],
)
def test_that_contextual_evaluator_detects_contradictions(
    context: _TestContext,
    guideline_a_definition: dict[str, str],
    guideline_b_definition: dict[str, str],
) -> None:
    guideline_a = GuidelineContent(
        predicate=guideline_a_definition["predicate"], action=guideline_a_definition["action"]
    )

    guideline_b = GuidelineContent(
        predicate=guideline_b_definition["predicate"], action=guideline_b_definition["action"]
    )

    contextual_contradiction_evaluator = ContextualContradictionEvaluator(
        context.container[Logger],
        context.container[SchematicGenerator[ContradictionTestsSchema]],
    )
    contradiction_results = list(
        context.sync_await(
            contextual_contradiction_evaluator.evaluate(
                [guideline_a, guideline_b],
                [],
            )
        )
    )

    assert len(contradiction_results) == 1
    contradiction = contradiction_results[0]

    assert ContradictionKind(contradiction.kind) == ContradictionKind.CONTEXTUAL
    assert contradiction.severity >= 5

    assert context.sync_await(
        nlp_test(
            context.container[Logger],
            f"Here is an explanation of what {contextual_contradiction_evaluator.contradiction_kind._describe()} type is:"  # noqa
            f"{contextual_contradiction_evaluator._format_contradiction_type_definition()}"
            "Here are two behavioral guidelines:"
            "a semantic contradiction test was conducted, regarding the following two behavioral guidelines:"  # noqa
            f"1. {contradiction.guideline_a}\n"
            f"2. {contradiction.guideline_b}\n"
            "Here is the output explanation generated by the contradiction test:"
            f"{contradiction.rationale}",
            "A contradiction has been found and an explanation is provided as to why it is a "
            "contextual contradiction",
        )
    )


@mark.parametrize(
    (
        "guideline_a_definition",
        "guideline_b_definition",
    ),
    [
        (
            {
                "predicate": "A customer asks about the security of their data",
                "action": "Provide detailed information about the company’s security measures and certifications",  # noqa
            },
            {
                "predicate": "A customer inquires about compliance with specific regulations",
                "action": "Direct them to documentation detailing the company’s compliance with those regulations",  # noqa
            },
        ),
        (
            {
                "predicate": "A customer requests faster support response times",
                "action": "Explain the standard response times and efforts to improve them",
            },
            {
                "predicate": "A customer compliments the service on social media",
                "action": "Thank them publicly and encourage them to share more about their positive experience",  # noqa
            },
        ),
    ],
)
def test_that_contextual_evaluator_does_not_produce_false_positives(
    context: _TestContext,
    guideline_a_definition: dict[str, str],
    guideline_b_definition: dict[str, str],
) -> None:
    guideline_a = GuidelineContent(
        predicate=guideline_a_definition["predicate"], action=guideline_a_definition["action"]
    )

    guideline_b = GuidelineContent(
        predicate=guideline_b_definition["predicate"], action=guideline_b_definition["action"]
    )

    contextual_contradiction_evaluator = ContextualContradictionEvaluator(
        context.container[Logger],
        context.container[SchematicGenerator[ContradictionTestsSchema]],
    )
    contradiction_results = list(
        context.sync_await(
            contextual_contradiction_evaluator.evaluate(
                [guideline_a, guideline_b],
                [],
            )
        )
    )

    assert len(contradiction_results) == 1
    contradiction = contradiction_results[0]

    assert ContradictionKind(contradiction.kind) == ContradictionKind.CONTEXTUAL
    assert contradiction.severity < 5

    assert context.sync_await(
        nlp_test(
            context.container[Logger],
            "Here are two behavioral guidelines:"
            "a semantic contradiction test was conducted, regarding the following two behavioral guidelines:"  # noqa
            f"1. {contradiction.guideline_a}\n"
            f"2. {contradiction.guideline_b}\n"
            "Here is the output explanation generated by the contradiction test:"
            f"{contradiction.rationale}",
            "No contradiction has been found between the two behavioral guidelines",
        )
    )


def test_that_coherence_check_does_not_produce_false_positives(
    context: _TestContext,
    sync_await: SyncAwaiter,
    guidelines_without_contradictions: list[GuidelineContent],
) -> None:
    coherence_checker = CoherenceChecker(
        context.container[Logger],
        context.container[SchematicGenerator[ContradictionTestsSchema]],
    )

    contradiction_results = sync_await(
        coherence_checker.evaluate_coherence(guidelines_without_contradictions, [])
    )

    assert len(list(filter(lambda c: c.severity >= 6, contradiction_results))) == 0


def test_that_coherence_check_produces_multiple_contradictions(
    context: _TestContext,
    sync_await: SyncAwaiter,
    guidelines_with_contradictions: list[GuidelineContent],
) -> None:
    coherence_checker = CoherenceChecker(
        context.container[Logger],
        context.container[SchematicGenerator[ContradictionTestsSchema]],
    )

    contradiction_results = list(
        sync_await(coherence_checker.evaluate_coherence(guidelines_with_contradictions, []))
    )

    n = len(guidelines_with_contradictions)
    pairs_per_evaluator = n * (n - 1) / 2
    num_contradiction_evaluators = 4

    assert len(contradiction_results) == num_contradiction_evaluators * pairs_per_evaluator


def test_that_existing_guidelines_are_not_evaluated_as_proposed_guidelines(
    context: _TestContext,
) -> None:
    guideline_to_evaluate = GuidelineContent(
        predicate="A VIP customer requests a specific feature that aligns with their business needs but is not on the current product roadmap",
        action="Escalate the request to product management for special consideration",
    )

    first_guideline_to_compare = GuidelineContent(
        predicate="Any customer requests a feature not available in the current version",
        action="Inform them about the product roadmap and upcoming features",
    )

    second_guideline_to_compare = GuidelineContent(
        predicate="A customer with low ranking requests a specific feature that does not aligns the current product roadmap",
        action="Inform them about the current roadmap and advise them to inquire again in one year.",
    )

    coherence_checker = CoherenceChecker(
        context.container[Logger],
        context.container[SchematicGenerator[ContradictionTestsSchema]],
    )
    contradiction_results = list(
        context.sync_await(
            coherence_checker.evaluate_coherence(
                [guideline_to_evaluate], [first_guideline_to_compare, second_guideline_to_compare]
            )
        )
    )

    # Hierarchical and contradiction between each existing guideline, but not between them both.
    assert contradiction_results[0].guideline_a == guideline_to_evaluate
    assert contradiction_results[1].guideline_a == guideline_to_evaluate
    assert contradiction_results[0].guideline_b == first_guideline_to_compare
    assert contradiction_results[1].guideline_b == second_guideline_to_compare
