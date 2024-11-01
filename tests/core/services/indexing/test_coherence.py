from datetime import datetime, timezone
from dataclasses import dataclass
from lagom import Container
from pytest import fixture, mark

from emcie.server.core.agents import Agent, AgentId, AgentStore

from emcie.server.core.guidelines import GuidelineContent
from emcie.server.core.glossary import GlossaryStore

from emcie.server.core.services.indexing.coherence_checker import (
    CoherenceChecker,
    IncoherenceKind,
    IncoherenceTest,
)

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


async def incoherence_nlp_test(
    agent: Agent, glossary_store: GlossaryStore, incoherence: IncoherenceTest
) -> bool:
    action_contradiction_test_result = await nlp_test_action_contradiction(
        agent, glossary_store, incoherence
    )
    predicate_entailment_test_result = await nlp_test_predicate_entailment(
        agent, glossary_store, incoherence
    )
    return action_contradiction_test_result and predicate_entailment_test_result


async def nlp_test_action_contradiction(
    agent: Agent, glossary_store: GlossaryStore, incoherence: IncoherenceTest
) -> bool:
    guideline_a_text = f"""{{when: "{incoherence.guideline_a.predicate}", then: "{incoherence.guideline_a.action}"}}"""
    guideline_b_text = f"""{{when: "{incoherence.guideline_b.predicate}", then: "{incoherence.guideline_b.action}"}}"""
    terms = await glossary_store.find_relevant_terms(
        agent.id,
        query=guideline_a_text + guideline_b_text,
    )
    context = f"""Two guidelines are said to have contradicting 'then' statements if applying both of their 'then' statements would result in a contradiction or an illogical action.

The following two guidelines were found to have contradicting 'then' statements:
{guideline_a_text}
{guideline_b_text}

The rationale for marking these 'then' statements as contradicting is: 
{incoherence.actions_contradiction_rationale}

The following is a description of the agent these guidelines apply to:
{agent.description}

The following is a glossary that applies to this agent:
{terms}"""
    predicate = "The provided rationale correctly explains why action contradiction is fulfilled between the two guidelines, given this agent and its glossary."
    return await nlp_test(context, predicate)


async def nlp_test_predicate_entailment(
    agent: Agent, glossary_store: GlossaryStore, incoherence: IncoherenceTest
) -> bool:
    guideline_a_text = f"""{{when: "{incoherence.guideline_a.predicate}", then: {incoherence.guideline_a.action}"}}"""
    guideline_b_text = f"""{{when: "{incoherence.guideline_b.predicate}", then: {incoherence.guideline_b.action}"}}"""
    terms = await glossary_store.find_relevant_terms(
        agent.id,
        query=guideline_a_text + guideline_b_text,
    )
    entailment_found_text = (
        "found" if incoherence.IncoherenceKind == IncoherenceKind.STRICT else "not found"
    )
    context = f"""Two guidelines should be marked as having entailing 'when' statements if the 'when' statement of one guideline entails the 'when' statement of the other, or vice-versa.

Such an entailment was {entailment_found_text} between these two guidelines:

{guideline_a_text}
{guideline_b_text}

The rationale given for this decision is: 
{incoherence.predicates_entailment_rationale}

The following is a description of the agent these guidelines apply to:
{agent.description}

The following is a glossary that applies to this agent:
{terms}"""
    predicate = f"The provided rationale correctly explains why these guidelines were {entailment_found_text} to have entailing 'when' statements, given this agent and its glossary."
    return await nlp_test(context, predicate)


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
                "action": "Inform them that upcoming features are added only according to the roadmap",
            },
        ),
        (
            {
                "predicate": "Any customer reports a technical issue",
                "action": "Queue the issue for resolution according to standard support protocols",
            },
            {
                "predicate": "An issue which affects a critical operational feature for multiple clients is reported",  # noqa
                "action": "Escalate immediately to the highest priority for resolution",
            },
        ),
    ],
)
def test_that_contradicting_actions_with_hierarchical_predicates_are_detected(
    context: _TestContext,
    agent: Agent,
    guideline_a_definition: dict[str, str],
    guideline_b_definition: dict[str, str],
) -> None:
    coherence_checker = context.container[CoherenceChecker]
    guideline_a = GuidelineContent(
        predicate=guideline_a_definition["predicate"], action=guideline_a_definition["action"]
    )

    guideline_b = GuidelineContent(
        predicate=guideline_b_definition["predicate"], action=guideline_b_definition["action"]
    )

    incoherence_results = list(
        context.sync_await(
            coherence_checker.propose_incoherencies(
                agent,
                [guideline_a, guideline_b],
            )
        )
    )

    assert len(incoherence_results) == 1

    correct_guidelines_option_1 = (incoherence_results[0].guideline_a == guideline_a) and (
        incoherence_results[0].guideline_b == guideline_b
    )
    correct_guidelines_option_2 = (incoherence_results[0].guideline_b == guideline_a) and (
        incoherence_results[0].guideline_a == guideline_b
    )
    assert correct_guidelines_option_1 or correct_guidelines_option_2
    assert incoherence_results[0].IncoherenceKind == IncoherenceKind.STRICT
    assert context.sync_await(
        incoherence_nlp_test(agent, context.container[GlossaryStore], incoherence_results[0])
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
                "predicate": "A user expresses dissatisfaction with our new design",
                "action": "encourage users to adopt and adapt to the change as part of ongoing product",
            },
            {
                "predicate": "A user requests a feature that is no longer available",
                "action": "Roll back or offer the option to revert to previous settings",
            },
        ),
    ],
)
def test_that_contingent_incoherencies_are_detected(
    context: _TestContext,
    agent: Agent,
    guideline_a_definition: dict[str, str],
    guideline_b_definition: dict[str, str],
) -> None:
    coherence_checker = context.container[CoherenceChecker]
    guideline_a = GuidelineContent(
        predicate=guideline_a_definition["predicate"], action=guideline_a_definition["action"]
    )

    guideline_b = GuidelineContent(
        predicate=guideline_b_definition["predicate"], action=guideline_b_definition["action"]
    )

    incoherence_results = list(
        context.sync_await(
            coherence_checker.propose_incoherencies(
                agent,
                [guideline_a, guideline_b],
            )
        )
    )

    assert len(incoherence_results) == 1

    correct_guidelines_option_1 = (incoherence_results[0].guideline_a == guideline_a) and (
        incoherence_results[0].guideline_b == guideline_b
    )
    correct_guidelines_option_2 = (incoherence_results[0].guideline_b == guideline_a) and (
        incoherence_results[0].guideline_a == guideline_b
    )
    assert correct_guidelines_option_1 or correct_guidelines_option_2
    assert incoherence_results[0].IncoherenceKind == IncoherenceKind.CONTINGENT

    assert context.sync_await(
        incoherence_nlp_test(agent, context.container[GlossaryStore], incoherence_results[0])
    )


@mark.parametrize(
    (
        "guideline_a_definition",
        "guideline_b_definition",
    ),
    [
        (
            {
                "predicate": "A new software update is scheduled for release during the latter half of the year",
                "action": "Roll it out to all users to ensure everyone has the latest version",
            },
            {
                "predicate": "A software update is about to release and the month is November",
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
def test_that_temporal_contradictions_are_detected_as_incoherencies(
    context: _TestContext,
    agent: Agent,
    guideline_a_definition: dict[str, str],
    guideline_b_definition: dict[str, str],
) -> None:
    coherence_checker = context.container[CoherenceChecker]
    guideline_a = GuidelineContent(
        predicate=guideline_a_definition["predicate"], action=guideline_a_definition["action"]
    )

    guideline_b = GuidelineContent(
        predicate=guideline_b_definition["predicate"], action=guideline_b_definition["action"]
    )

    incoherence_results = list(
        context.sync_await(
            coherence_checker.propose_incoherencies(
                agent,
                [guideline_a, guideline_b],
            )
        )
    )

    assert len(incoherence_results) == 1

    correct_guidelines_option_1 = (incoherence_results[0].guideline_a == guideline_a) and (
        incoherence_results[0].guideline_b == guideline_b
    )
    correct_guidelines_option_2 = (incoherence_results[0].guideline_b == guideline_a) and (
        incoherence_results[0].guideline_a == guideline_b
    )
    assert correct_guidelines_option_1 or correct_guidelines_option_2
    assert incoherence_results[0].IncoherenceKind == IncoherenceKind.STRICT
    assert context.sync_await(
        incoherence_nlp_test(agent, context.container[GlossaryStore], incoherence_results[0])
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
                "predicate": "The company’s financial performance targets require maximizing revenue",
                "action": "avoid discounts and push for higher-priced contracts",
            },
        ),
    ],
)
def test_that_contextual_contradictions_are_detected_as_contingent_incoherence(
    context: _TestContext,
    agent: Agent,
    guideline_a_definition: dict[str, str],
    guideline_b_definition: dict[str, str],
) -> None:
    coherence_checker = context.container[CoherenceChecker]
    guideline_a = GuidelineContent(
        predicate=guideline_a_definition["predicate"], action=guideline_a_definition["action"]
    )

    guideline_b = GuidelineContent(
        predicate=guideline_b_definition["predicate"], action=guideline_b_definition["action"]
    )

    incoherence_results = list(
        context.sync_await(
            coherence_checker.propose_incoherencies(
                agent,
                [guideline_a, guideline_b],
            )
        )
    )

    assert len(incoherence_results) == 1

    correct_guidelines_option_1 = (incoherence_results[0].guideline_a == guideline_a) and (
        incoherence_results[0].guideline_b == guideline_b
    )
    correct_guidelines_option_2 = (incoherence_results[0].guideline_b == guideline_a) and (
        incoherence_results[0].guideline_a == guideline_b
    )
    assert correct_guidelines_option_1 or correct_guidelines_option_2
    assert incoherence_results[0].IncoherenceKind == IncoherenceKind.CONTINGENT
    assert context.sync_await(
        incoherence_nlp_test(agent, context.container[GlossaryStore], incoherence_results[0])
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
def test_that_non_contradicting_guidelines_arent_false_positives(
    context: _TestContext,
    agent: Agent,
    guideline_a_definition: dict[str, str],
    guideline_b_definition: dict[str, str],
) -> None:
    coherence_checker = context.container[CoherenceChecker]
    guideline_a = GuidelineContent(
        predicate=guideline_a_definition["predicate"], action=guideline_a_definition["action"]
    )

    guideline_b = GuidelineContent(
        predicate=guideline_b_definition["predicate"], action=guideline_b_definition["action"]
    )

    incoherence_results = list(
        context.sync_await(
            coherence_checker.propose_incoherencies(
                agent,
                [guideline_a, guideline_b],
            )
        )
    )

    assert len(incoherence_results) == 0


def test_that_suggestive_predicates_with_contradicting_actions_are_detected_as_contingent_incoherencies(
    context: _TestContext,
    agent: Agent,
) -> None:
    coherence_checker = context.container[CoherenceChecker]
    guideline_a = GuidelineContent(
        predicate="Recommending pizza toppings",
        action="Only recommend mushrooms as they are healthy",
    )

    guideline_b = GuidelineContent(
        predicate="Asked for our pizza topping selection",
        action="list the possible toppings and recommend olives",
    )

    incoherence_results = list(
        context.sync_await(
            coherence_checker.propose_incoherencies(
                agent,
                [guideline_a, guideline_b],
            )
        )
    )

    assert len(incoherence_results) == 1

    correct_guidelines_option_1 = (incoherence_results[0].guideline_a == guideline_a) and (
        incoherence_results[0].guideline_b == guideline_b
    )
    correct_guidelines_option_2 = (incoherence_results[0].guideline_b == guideline_a) and (
        incoherence_results[0].guideline_a == guideline_b
    )
    assert correct_guidelines_option_1 or correct_guidelines_option_2
    assert incoherence_results[0].IncoherenceKind == IncoherenceKind.CONTINGENT
    assert context.sync_await(
        incoherence_nlp_test(agent, context.container[GlossaryStore], incoherence_results[0])
    )


def test_that_logically_contradicting_response_actions_are_detected_as_incoherencies(
    context: _TestContext,
    agent: Agent,
) -> None:
    coherence_checker = context.container[CoherenceChecker]
    guideline_a = GuidelineContent(
        predicate="Recommending pizza toppings", action="Recommend tomatoes"
    )

    guideline_b = GuidelineContent(
        predicate="asked about our toppings while inventory indicates that we are almost out of tomatoes",
        action="mention that we are out of tomatoes",
    )

    incoherence_results = list(
        context.sync_await(
            coherence_checker.propose_incoherencies(
                agent,
                [guideline_a, guideline_b],
            )
        )
    )

    assert len(incoherence_results) == 1

    correct_guidelines_option_1 = (incoherence_results[0].guideline_a == guideline_a) and (
        incoherence_results[0].guideline_b == guideline_b
    )
    correct_guidelines_option_2 = (incoherence_results[0].guideline_b == guideline_a) and (
        incoherence_results[0].guideline_a == guideline_b
    )
    assert correct_guidelines_option_1 or correct_guidelines_option_2
    assert incoherence_results[0].IncoherenceKind == IncoherenceKind.CONTINGENT
    assert context.sync_await(
        incoherence_nlp_test(agent, context.container[GlossaryStore], incoherence_results[0])
    )


def test_that_entailing_predicates_with_unrelated_actions_arent_false_positives(
    context: _TestContext,
    agent: Agent,
) -> None:
    coherence_checker = context.container[CoherenceChecker]
    guideline_a = GuidelineContent(
        predicate="ordering tickets for a movie",
        action="check if the customer is eligible for a discount",
    )

    guideline_b = GuidelineContent(
        predicate="buying tickets for rated R movies",
        action="ask the customer for identification",
    )

    incoherence_results = list(
        context.sync_await(
            coherence_checker.propose_incoherencies(
                agent,
                [guideline_a, guideline_b],
            )
        )
    )

    assert len(incoherence_results) == 0


def test_that_contradicting_actions_that_are_contextualized_by_their_predicates_are_detected(
    context: _TestContext,
    agent: Agent,
) -> None:
    coherence_checker = context.container[CoherenceChecker]
    guideline_a = GuidelineContent(
        predicate="asked to schedule an appointment for the weekend",
        action="alert the customer about our weekend hours and schedule the appointment",
    )

    guideline_b = GuidelineContent(
        predicate="asked for an appointment with a physician",
        action="schedule a physician appointment for the next available Monday",
    )

    incoherence_results = list(
        context.sync_await(
            coherence_checker.propose_incoherencies(
                agent,
                [guideline_a, guideline_b],
            )
        )
    )

    assert len(incoherence_results) == 1

    correct_guidelines_option_1 = (incoherence_results[0].guideline_a == guideline_a) and (
        incoherence_results[0].guideline_b == guideline_b
    )
    correct_guidelines_option_2 = (incoherence_results[0].guideline_b == guideline_a) and (
        incoherence_results[0].guideline_a == guideline_b
    )
    assert correct_guidelines_option_1 or correct_guidelines_option_2
    assert incoherence_results[0].IncoherenceKind == IncoherenceKind.CONTINGENT
    assert context.sync_await(
        incoherence_nlp_test(agent, context.container[GlossaryStore], incoherence_results[0])
    )


def test_that_many_coherent_guidelines_arent_detected_as_false_positive(
    context: _TestContext,
    agent: Agent,
) -> None:
    coherent_guidelines: list[GuidelineContent] = []

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
        coherent_guidelines.append(
            GuidelineContent(
                predicate=guideline_params["predicate"], action=guideline_params["action"]
            )
        )

    coherence_checker = context.container[CoherenceChecker]

    incoherence_results = list(
        context.sync_await(
            coherence_checker.propose_incoherencies(
                agent,
                coherent_guidelines,
            )
        )
    )

    assert len(incoherence_results) == 0


def test_that_guidelines_with_many_incoherencies_are_detected(
    context: _TestContext,
    agent: Agent,
) -> None:
    guidelines_with_incoherencies: list[GuidelineContent] = []

    for guideline_params in [
        {
            "predicate": "Asked for UPS shipping",
            "action": "Send UPS a request and inform the client that the order would arrive in 5 business days",
        },
        {
            "predicate": "Asked for express shipping",
            "action": "register the order and inform the client that the order would arrive tomorrow",
        },
        {
            "predicate": "Asked for delayed shipping",
            "action": "Send UPS the request and inform the client that the order would arrive in the next calendar year",
        },
        {
            "predicate": "The client stops responding mid-order",
            "action": "save the client's order in their cart and notify them that it has not been shipped yet",
        },
        {
            "predicate": "The client refuses to give their address",
            "action": "cancel the order and notify the client",
        },
    ]:
        guidelines_with_incoherencies.append(
            GuidelineContent(
                predicate=guideline_params["predicate"], action=guideline_params["action"]
            )
        )

    coherence_checker = context.container[CoherenceChecker]

    incoherence_results = list(
        context.sync_await(
            coherence_checker.propose_incoherencies(
                agent,
                guidelines_with_incoherencies,
            )
        )
    )

    n = len(guidelines_with_incoherencies)
    assert len(incoherence_results) == n * (n - 1) // 2

    # Tests that there's exactly 1 case of incoherence per pair of guidelines
    expected_pairs = {
        (g1, g2)
        for i, g1 in enumerate(guidelines_with_incoherencies)
        for g2 in guidelines_with_incoherencies[i + 1 :]
    }

    for c in incoherence_results:
        assert (c.guideline_a, c.guideline_b) in expected_pairs
        assert c.IncoherenceKind == IncoherenceKind.CONTINGENT
        expected_pairs.remove((c.guideline_a, c.guideline_b))

    assert len(expected_pairs) == 0


def test_that_contradictory_next_message_commands_are_detected_as_incoherencies(
    context: _TestContext,
    agent: Agent,
) -> None:
    coherence_checker = context.container[CoherenceChecker]
    guidelines_to_evaluate = [
        GuidelineContent(
            predicate="a document is being discussed",
            action="provide the full document to the user",
        ),
        GuidelineContent(
            predicate="a client asks to summarize a document",
            action="provide a summary of the document in 100 words or less",
        ),
        GuidelineContent(
            predicate="the client asks for a summary of another user's medical document",
            action="refuse to share the document or its summary",
        ),
    ]

    incoherence_results = list(
        context.sync_await(
            coherence_checker.propose_incoherencies(
                agent,
                guidelines_to_evaluate,
            )
        )
    )

    assert len(incoherence_results) == 3
    incoherencies_to_detect = {
        (guideline_a, guideline_b)
        for i, guideline_a in enumerate(guidelines_to_evaluate)
        for guideline_b in guidelines_to_evaluate[i + 1 :]
    }
    for c in incoherence_results:
        assert (c.guideline_a, c.guideline_b) in incoherencies_to_detect
        assert c.IncoherenceKind == IncoherenceKind.STRICT
        incoherencies_to_detect.remove((c.guideline_a, c.guideline_b))
    assert len(incoherencies_to_detect) == 0

    for incoherence in incoherence_results:
        assert context.sync_await(
            incoherence_nlp_test(agent, context.container[GlossaryStore], incoherence)
        )


def test_that_existing_guidelines_are_not_checked_against_each_other(
    context: _TestContext,
    agent: Agent,
) -> None:
    coherence_checker = context.container[CoherenceChecker]
    guideline_to_evaluate = GuidelineContent(
        predicate="the user is dissatisfied",
        action="apologize and suggest to forward the request to management",
    )

    first_guideline_to_compare = GuidelineContent(
        predicate="a client asks to summarize a document",
        action="provide a summary of the document",
    )

    second_guideline_to_compare = GuidelineContent(
        predicate="the client asks for a summary of another user's medical document",
        action="refuse to share the document or its summary",
    )

    incoherence_results = list(
        context.sync_await(
            coherence_checker.propose_incoherencies(
                agent,
                [guideline_to_evaluate],
                [first_guideline_to_compare, second_guideline_to_compare],
            )
        )
    )

    assert len(incoherence_results) == 0


def test_that_a_terminology_based_incoherency_is_detected(
    context: _TestContext,
    agent: Agent,
) -> None:
    glossary_store = context.container[GlossaryStore]

    context.sync_await(
        glossary_store.create_term(
            term_set=agent.id,
            name="PAP",
            description="Pineapple pizza - a pizza topped with pineapples",
            synonyms=["PP", "pap"],
        )
    )

    coherence_checker = context.container[CoherenceChecker]
    guideline_a = GuidelineContent(
        predicate="the client asks our recommendation",
        action="add one pap to the order",
    )

    guideline_b = GuidelineContent(
        predicate="the client asks for a specific pizza topping",
        action="Add the pizza to the order unless a fruit-based topping is requested",
    )

    incoherence_results = list(
        context.sync_await(
            coherence_checker.propose_incoherencies(
                agent,
                [guideline_a, guideline_b],
            )
        )
    )

    assert len(incoherence_results) == 1

    correct_guidelines_option_1 = (incoherence_results[0].guideline_a == guideline_a) and (
        incoherence_results[0].guideline_b == guideline_b
    )
    correct_guidelines_option_2 = (incoherence_results[0].guideline_b == guideline_a) and (
        incoherence_results[0].guideline_a == guideline_b
    )
    assert correct_guidelines_option_1 or correct_guidelines_option_2
    assert incoherence_results[0].IncoherenceKind == IncoherenceKind.CONTINGENT

    assert context.sync_await(
        incoherence_nlp_test(agent, context.container[GlossaryStore], incoherence_results[0])
    )


def test_that_an_agent_description_based_incoherency_is_detected(
    context: _TestContext,
) -> None:
    agent = Agent(
        id=AgentId("sparkling-water-agent"),
        name="sparkling-water-agent",
        description="You are a helpful AI assistant for a sparkling water company. Our company sells sparkling water, but never sparkling juices. The philosophy of our company dictates that juices should never be carbonated.",
        creation_utc=datetime.now(timezone.utc),
        max_engine_iterations=3,
    )

    coherence_checker = context.container[CoherenceChecker]
    guideline_a = GuidelineContent(
        predicate="the client asks for our recommendation",
        action="Recommend a product according to our company's philosophy",
    )

    guideline_b = GuidelineContent(
        predicate="the client asks for a recommended sweetened soda",
        action="suggest sparkling orange juice",
    )
    incoherence_results = list(
        context.sync_await(
            coherence_checker.propose_incoherencies(
                agent,
                [guideline_a, guideline_b],
            )
        )
    )

    assert len(incoherence_results) == 1

    correct_guidelines_option_1 = (incoherence_results[0].guideline_a == guideline_a) and (
        incoherence_results[0].guideline_b == guideline_b
    )
    correct_guidelines_option_2 = (incoherence_results[0].guideline_b == guideline_a) and (
        incoherence_results[0].guideline_a == guideline_b
    )
    assert correct_guidelines_option_1 or correct_guidelines_option_2
    assert incoherence_results[0].IncoherenceKind == IncoherenceKind.STRICT
    assert context.sync_await(
        nlp_test_action_contradiction(
            agent, context.container[GlossaryStore], incoherence_results[0]
        )
    )


def test_that_many_guidelines_which_are_all_contradictory_are_detected(
    context: _TestContext,
    agent: Agent,
    n: int = 7,
) -> None:
    coherence_checker = context.container[CoherenceChecker]

    contradictory_guidelines = [
        GuidelineContent(
            predicate="a client asks for the price of a television",
            action=f"reply that the price is {i*10}$",
        )
        for i in range(n)
    ]
    incoherence_results = list(
        context.sync_await(
            coherence_checker.propose_incoherencies(
                agent,
                contradictory_guidelines,
            )
        )
    )
    incoherencies_n = (n * (n - 1)) // 2

    assert len(incoherence_results) == incoherencies_n
    for c in incoherence_results:
        assert c.IncoherenceKind == IncoherenceKind.STRICT


def test_that_misspelled_contradicting_actions_are_detected_as_incoherencies(  # Same as test_that_logically_contradicting_actions_are_detected_as_incoherencies but with typos
    context: _TestContext,
    agent: Agent,
) -> None:
    coherence_checker = context.container[CoherenceChecker]
    guideline_a = GuidelineContent(predicate="Recommending pizza tops", action="Recommend tomatos")

    guideline_b = GuidelineContent(
        predicate="asked about our toppings while inventory indicates that we are almost out of tomatoes",
        action="mention that we are oout of tomatoes",
    )

    incoherence_results = list(
        context.sync_await(
            coherence_checker.propose_incoherencies(
                agent,
                [guideline_a, guideline_b],
            )
        )
    )

    assert len(incoherence_results) == 1

    correct_guidelines_option_1 = (incoherence_results[0].guideline_a == guideline_a) and (
        incoherence_results[0].guideline_b == guideline_b
    )
    correct_guidelines_option_2 = (incoherence_results[0].guideline_b == guideline_a) and (
        incoherence_results[0].guideline_a == guideline_b
    )
    assert correct_guidelines_option_1 or correct_guidelines_option_2
    assert incoherence_results[0].IncoherenceKind == IncoherenceKind.CONTINGENT
    assert context.sync_await(
        incoherence_nlp_test(agent, context.container[GlossaryStore], incoherence_results[0])
    )


def test_that_seemingly_contradictory_but_actually_complementary_actions_are_not_false_positives(
    context: _TestContext,
    agent: Agent,
) -> None:
    coherence_checker = context.container[CoherenceChecker]
    guideline_a = GuidelineContent(
        predicate="the user is a returning customer", action="add a 5% discount to the order"
    )

    guideline_b = GuidelineContent(
        predicate="the user is a very frequent customer",
        action="add a 10% discount to the order",
    )

    incoherence_results = list(
        context.sync_await(
            coherence_checker.propose_incoherencies(
                agent,
                [guideline_a, guideline_b],
            )
        )
    )

    assert len(incoherence_results) == 1
    assert incoherence_results[0].IncoherenceKind == IncoherenceKind.STRICT
    assert context.sync_await(
        incoherence_nlp_test(agent, context.container[GlossaryStore], incoherence_results[0])
    )
