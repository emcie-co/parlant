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

from datetime import datetime, timezone

from parlant.core.agents import Agent, AgentId
from parlant.core.entity_cq import EntityQueries
from parlant.core.guidelines import GuidelineContent
from parlant.core.glossary import GlossaryStore
from parlant.core.services.indexing.coherence_checker import (
    CoherenceChecker,
    IncoherenceKind,
    IncoherenceTest,
)

from parlant.core.tags import Tag
from tests.core.common.utils import ContextOfTest
from tests.test_utilities import nlp_test


async def incoherence_nlp_test(
    context: ContextOfTest,
    agent: Agent,
    incoherence: IncoherenceTest,
) -> bool:
    action_contradiction_test_result = await nlp_test_action_contradiction(
        context=context,
        agent=agent,
        incoherence=incoherence,
    )

    condition_entailment_test_result = await nlp_test_condition_entailment(
        context=context,
        agent=agent,
        incoherence=incoherence,
    )

    return action_contradiction_test_result and condition_entailment_test_result


async def nlp_test_action_contradiction(
    context: ContextOfTest,
    agent: Agent,
    incoherence: IncoherenceTest,
) -> bool:
    entity_queries = context.container[EntityQueries]

    guideline_a_text = f"""{{when: "{incoherence.guideline_a.condition}", then: "{incoherence.guideline_a.action}"}}"""
    guideline_b_text = f"""{{when: "{incoherence.guideline_b.condition}", then: "{incoherence.guideline_b.action}"}}"""

    terms = await entity_queries.find_glossary_terms_for_context(
        agent_id=agent.id,
        query=guideline_a_text + guideline_b_text,
    )
    test_context = f"""Two guidelines are said to have contradicting 'then' statements if applying both of their 'then' statements would result in a contradiction or an illogical action.

The following two guidelines were found to have contradicting 'then' statements:
{guideline_a_text}
{guideline_b_text}

The rationale for marking these 'then' statements as contradicting is:
{incoherence.actions_contradiction_rationale}

The following is a description of the agent these guidelines apply to:
{agent.description}

The following is a glossary that applies to this agent:
{terms}"""
    condition = "The provided rationale correctly explains why action contradiction is fulfilled between the two guidelines, given this agent and its glossary."
    return await nlp_test(test_context, condition)


async def nlp_test_condition_entailment(
    context: ContextOfTest, agent: Agent, incoherence: IncoherenceTest
) -> bool:
    entity_queries = context.container[EntityQueries]

    guideline_a_text = f"""{{when: "{incoherence.guideline_a.condition}", then: {incoherence.guideline_a.action}"}}"""
    guideline_b_text = f"""{{when: "{incoherence.guideline_b.condition}", then: {incoherence.guideline_b.action}"}}"""

    terms = await entity_queries.find_glossary_terms_for_context(
        agent_id=agent.id,
        query=guideline_a_text + guideline_b_text,
    )
    entailment_found_text = (
        "found" if incoherence.IncoherenceKind == IncoherenceKind.STRICT else "not found"
    )
    test_context = f"""Two guidelines should be marked as having entailing 'when' statements if the 'when' statement of one guideline entails the 'when' statement of the other, or vice-versa.

Such an entailment was {entailment_found_text} between these two guidelines:

{guideline_a_text}
{guideline_b_text}

The rationale given for this decision is:
{incoherence.conditions_entailment_rationale}

The following is a description of the agent these guidelines apply to:
{agent.description}

The following is a glossary that applies to this agent:
{terms}"""
    condition = f"The provided rationale correctly explains why these guidelines were {entailment_found_text} to have entailing 'when' statements, given this agent and its glossary."
    return await nlp_test(test_context, condition)


def base_test_that_contradicting_actions_with_hierarchical_conditions_are_detected(
    context: ContextOfTest,
    agent: Agent,
    guideline_a_definition: dict[str, str],
    guideline_b_definition: dict[str, str],
) -> None:
    coherence_checker = context.container[CoherenceChecker]
    guideline_a = GuidelineContent(
        condition=guideline_a_definition["condition"], action=guideline_a_definition["action"]
    )

    guideline_b = GuidelineContent(
        condition=guideline_b_definition["condition"], action=guideline_b_definition["action"]
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
    assert context.sync_await(incoherence_nlp_test(context, agent, incoherence_results[0]))


def test_that_contradicting_actions_with_hierarchical_conditions_are_detected_parametrized_1(
    context: ContextOfTest,
    agent: Agent,
) -> None:
    guideline_a_definition: dict[str, str] = {
        "condition": "A VIP customer requests a specific feature that aligns with their business needs but is not on the current product roadmap",  # noqa
        "action": "Escalate the request to product management for special consideration",
    }
    guideline_b_definition: dict[str, str] = {
        "condition": "Any customer requests a feature not available in the current version",
        "action": "Inform them that upcoming features are added only according to the roadmap",
    }
    base_test_that_contradicting_actions_with_hierarchical_conditions_are_detected(
        context, agent, guideline_a_definition, guideline_b_definition
    )


def test_that_contradicting_actions_with_hierarchical_conditions_are_detected_parametrized_2(
    context: ContextOfTest,
    agent: Agent,
) -> None:
    guideline_a_definition: dict[str, str] = {
        "condition": "Any customer reports a technical issue",
        "action": "Queue the issue for resolution according to standard support protocols",
    }
    guideline_b_definition: dict[str, str] = {
        "condition": "An issue which affects a critical operational feature for multiple clients is reported",  # noqa
        "action": "Escalate immediately to the highest priority for resolution",
    }
    base_test_that_contradicting_actions_with_hierarchical_conditions_are_detected(
        context, agent, guideline_a_definition, guideline_b_definition
    )


def base_test_that_contingent_incoherencies_are_detected(
    context: ContextOfTest,
    agent: Agent,
    guideline_a_definition: dict[str, str],
    guideline_b_definition: dict[str, str],
) -> None:
    coherence_checker = context.container[CoherenceChecker]
    guideline_a = GuidelineContent(
        condition=guideline_a_definition["condition"], action=guideline_a_definition["action"]
    )

    guideline_b = GuidelineContent(
        condition=guideline_b_definition["condition"], action=guideline_b_definition["action"]
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

    assert context.sync_await(incoherence_nlp_test(context, agent, incoherence_results[0]))


def test_that_contingent_incoherencies_are_detected_parametrized_1(
    context: ContextOfTest,
    agent: Agent,
) -> None:
    guideline_a_definition: dict[str, str] = {
        "condition": "A customer exceeds their data storage limit",
        "action": "Prompt them to upgrade their subscription plan",
    }

    guideline_b_definition: dict[str, str] = {
        "condition": "Promoting customer retention and satisfaction",
        "action": "Offer a temporary data limit extension without requiring an upgrade",
    }
    base_test_that_contingent_incoherencies_are_detected(
        context, agent, guideline_a_definition, guideline_b_definition
    )


def test_that_contingent_incoherencies_are_detected_parametrized_2(
    context: ContextOfTest,
    agent: Agent,
) -> None:
    guideline_a_definition: dict[str, str] = {
        "condition": "A customer expresses dissatisfaction with our new design",
        "action": "encourage customers to adopt and adapt to the change as part of ongoing product",
    }

    guideline_b_definition: dict[str, str] = {
        "condition": "A customer requests a feature that is no longer available",
        "action": "Roll back or offer the option to revert to previous settings",
    }
    base_test_that_contingent_incoherencies_are_detected(
        context, agent, guideline_a_definition, guideline_b_definition
    )


def base_test_that_temporal_contradictions_are_detected_as_incoherencies(
    context: ContextOfTest,
    agent: Agent,
    guideline_a_definition: dict[str, str],
    guideline_b_definition: dict[str, str],
) -> None:
    coherence_checker = context.container[CoherenceChecker]
    guideline_a = GuidelineContent(
        condition=guideline_a_definition["condition"], action=guideline_a_definition["action"]
    )

    guideline_b = GuidelineContent(
        condition=guideline_b_definition["condition"], action=guideline_b_definition["action"]
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
    assert context.sync_await(incoherence_nlp_test(context, agent, incoherence_results[0]))


def test_that_temporal_contradictions_are_detected_as_incoherencies_parametrized_1(
    context: ContextOfTest,
    agent: Agent,
) -> None:
    guideline_a_definition: dict[str, str] = {
        "condition": "A new software update is scheduled for release during the latter half of the year",
        "action": "Roll it out to all customers to ensure everyone has the latest version",
    }
    guideline_b_definition: dict[str, str] = {
        "condition": "A software update is about to release and the month is November",
        "action": "Delay software updates to avoid disrupting their operations",
    }
    base_test_that_temporal_contradictions_are_detected_as_incoherencies(
        context, agent, guideline_a_definition, guideline_b_definition
    )


def test_that_temporal_contradictions_are_detected_as_incoherencies_parametrized_2(
    context: ContextOfTest,
    agent: Agent,
) -> None:
    guideline_a_definition: dict[str, str] = {
        "condition": "The financial quarter ends",
        "action": "Finalize all pending transactions and close the books",
    }
    guideline_b_definition: dict[str, str] = {
        "condition": "A new financial regulation is implemented at the end of the quarter",
        "action": "re-evaluate all transactions from that quarter before closing the books",  # noqa
    }
    base_test_that_temporal_contradictions_are_detected_as_incoherencies(
        context, agent, guideline_a_definition, guideline_b_definition
    )


def base_test_that_contextual_contradictions_are_detected_as_contingent_incoherence(
    context: ContextOfTest,
    agent: Agent,
    guideline_a_definition: dict[str, str],
    guideline_b_definition: dict[str, str],
) -> None:
    coherence_checker = context.container[CoherenceChecker]
    guideline_a = GuidelineContent(
        condition=guideline_a_definition["condition"], action=guideline_a_definition["action"]
    )

    guideline_b = GuidelineContent(
        condition=guideline_b_definition["condition"], action=guideline_b_definition["action"]
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
    assert context.sync_await(incoherence_nlp_test(context, agent, incoherence_results[0]))


def test_that_contextual_contradictions_are_detected_as_contingent_incoherence_parametrized_1(
    context: ContextOfTest,
    agent: Agent,
) -> None:
    guideline_a_definition: dict[str, str] = {
        "condition": "A customer is located in a region with strict data sovereignty laws",
        "action": "Store and process all customer data locally as required by law",
    }
    guideline_b_definition: dict[str, str] = {
        "condition": "The company's policy is to centralize data processing in a single, cost-effective location",  # noqa
        "action": "Consolidate data handling to enhance efficiency",
    }
    base_test_that_contextual_contradictions_are_detected_as_contingent_incoherence(
        context, agent, guideline_a_definition, guideline_b_definition
    )


def test_that_contextual_contradictions_are_detected_as_contingent_incoherence_parametrized_2(
    context: ContextOfTest,
    agent: Agent,
) -> None:
    guideline_a_definition: dict[str, str] = {
        "condition": "A customer's contract is up for renewal during a market downturn",
        "action": "Offer discounts and incentives to ensure renewal",
    }

    guideline_b_definition: dict[str, str] = {
        "condition": "The company’s financial performance targets require maximizing revenue",
        "action": "avoid discounts and push for higher-priced contracts",
    }
    base_test_that_contextual_contradictions_are_detected_as_contingent_incoherence(
        context, agent, guideline_a_definition, guideline_b_definition
    )


def base_test_that_non_contradicting_guidelines_arent_false_positives(
    context: ContextOfTest,
    agent: Agent,
    guideline_a_definition: dict[str, str],
    guideline_b_definition: dict[str, str],
) -> None:
    coherence_checker = context.container[CoherenceChecker]
    guideline_a = GuidelineContent(
        condition=guideline_a_definition["condition"], action=guideline_a_definition["action"]
    )

    guideline_b = GuidelineContent(
        condition=guideline_b_definition["condition"], action=guideline_b_definition["action"]
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


def test_that_non_contradicting_guidelines_arent_false_positives_parametrized_1(
    context: ContextOfTest,
    agent: Agent,
) -> None:
    guideline_a_definition: dict[str, str] = {
        "condition": "A customer asks about the security of their data",
        "action": "Provide detailed information about the company’s security measures and certifications",  # noqa
    }
    guideline_b_definition: dict[str, str] = {
        "condition": "A customer inquires about compliance with specific regulations",
        "action": "Direct them to documentation detailing the company’s compliance with those regulations",  # noqa
    }
    base_test_that_non_contradicting_guidelines_arent_false_positives(
        context, agent, guideline_a_definition, guideline_b_definition
    )


def test_that_non_contradicting_guidelines_arent_false_positives_parametrized_2(
    context: ContextOfTest,
    agent: Agent,
) -> None:
    guideline_a_definition: dict[str, str] = {
        "condition": "A customer requests faster support response times",
        "action": "Explain the standard response times and efforts to improve them",
    }
    guideline_b_definition: dict[str, str] = {
        "condition": "A customer compliments the service on social media",
        "action": "Thank them publicly and encourage them to share more about their positive experience",  # noqa
    }
    base_test_that_non_contradicting_guidelines_arent_false_positives(
        context, agent, guideline_a_definition, guideline_b_definition
    )


def test_that_suggestive_conditions_with_contradicting_actions_are_detected_as_contingent_incoherencies(
    context: ContextOfTest,
    agent: Agent,
) -> None:
    coherence_checker = context.container[CoherenceChecker]
    guideline_a = GuidelineContent(
        condition="Recommending pizza toppings",
        action="Only recommend mushrooms as they are healthy",
    )

    guideline_b = GuidelineContent(
        condition="Asked for our pizza topping selection",
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
    assert context.sync_await(incoherence_nlp_test(context, agent, incoherence_results[0]))


def test_that_logically_contradicting_response_actions_are_detected_as_incoherencies(
    context: ContextOfTest,
    agent: Agent,
) -> None:
    coherence_checker = context.container[CoherenceChecker]
    guideline_a = GuidelineContent(
        condition="Recommending pizza toppings", action="Recommend tomatoes"
    )

    guideline_b = GuidelineContent(
        condition="asked about our toppings while inventory indicates that we are almost out of tomatoes",
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
    assert context.sync_await(incoherence_nlp_test(context, agent, incoherence_results[0]))


def test_that_entailing_conditions_with_unrelated_actions_arent_false_positives(
    context: ContextOfTest,
    agent: Agent,
) -> None:
    coherence_checker = context.container[CoherenceChecker]
    guideline_a = GuidelineContent(
        condition="ordering tickets for a movie",
        action="check if the customer is eligible for a discount",
    )

    guideline_b = GuidelineContent(
        condition="buying tickets for rated R movies",
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


def test_that_contradicting_actions_that_are_contextualized_by_their_conditions_are_detected(
    context: ContextOfTest,
    agent: Agent,
) -> None:
    coherence_checker = context.container[CoherenceChecker]
    guideline_a = GuidelineContent(
        condition="asked to schedule an appointment for the weekend",
        action="alert the customer about our weekend hours and schedule the appointment",
    )

    guideline_b = GuidelineContent(
        condition="asked for an appointment with a physician",
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
    assert context.sync_await(incoherence_nlp_test(context, agent, incoherence_results[0]))


def test_that_many_coherent_guidelines_arent_detected_as_false_positive(
    context: ContextOfTest,
    agent: Agent,
) -> None:
    coherent_guidelines: list[GuidelineContent] = []

    for guideline_params in [
        {
            "condition": "A customer inquires about upgrading their service package",
            "action": "Provide information on available upgrade options and benefits",
        },
        {
            "condition": "A customer needs assistance with understanding their billing statements",
            "action": "Guide them through the billing details and explain any charges",
        },
        {
            "condition": "A customer expresses satisfaction with the service",
            "action": "encourage them to leave a review or testimonial",
        },
        {
            "condition": "A customer refers another potential client",
            "action": "initiate the referral rewards process",
        },
        {
            "condition": "A customer asks about the security of their data",
            "action": "Provide detailed information about the company’s security measures and certifications",  # noqa
        },
        {
            "condition": "A customer inquires about compliance with specific regulations",
            "action": "Direct them to documentation detailing the company’s compliance with those regulations",  # noqa
        },
        {
            "condition": "A customer requests faster support response times",
            "action": "Explain the standard response times and efforts to improve them",
        },
        {
            "condition": "A customer compliments the service on social media",
            "action": "Thank them publicly and encourage them to share more about their positive experience",  # noqa
        },
        {
            "condition": "A customer asks about the security of their data",
            "action": "Provide detailed information about the company’s security measures and certifications",  # noqa
        },
        {
            "condition": "A customer inquires about compliance with specific regulations",
            "action": "Direct them to documentation detailing the company’s compliance with those regulations",  # noqa
        },
    ]:
        coherent_guidelines.append(
            GuidelineContent(
                condition=guideline_params["condition"], action=guideline_params["action"]
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


def test_that_existing_guidelines_are_not_checked_against_each_other(
    context: ContextOfTest,
    agent: Agent,
) -> None:
    coherence_checker = context.container[CoherenceChecker]
    guideline_to_evaluate = GuidelineContent(
        condition="the customer is dissatisfied",
        action="apologize and suggest to forward the request to management",
    )

    first_guideline_to_compare = GuidelineContent(
        condition="a client asks to summarize a document",
        action="provide a summary of the document",
    )

    second_guideline_to_compare = GuidelineContent(
        condition="the client asks for a summary of another customer's medical document",
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


def test_that_a_glossary_based_incoherency_is_detected(
    context: ContextOfTest,
    agent: Agent,
) -> None:
    glossary_store = context.container[GlossaryStore]

    term = context.sync_await(
        glossary_store.create_term(
            name="PAP",
            description="Pineapple pizza - a pizza topped with pineapples",
            synonyms=["PP", "pap"],
        )
    )

    context.sync_await(
        glossary_store.upsert_tag(
            term_id=term.id,
            tag_id=Tag.for_agent_id(agent.id),
        )
    )

    coherence_checker = context.container[CoherenceChecker]
    guideline_a = GuidelineContent(
        condition="the client asks our recommendation",
        action="add one pap to the order",
    )

    guideline_b = GuidelineContent(
        condition="the client asks for a specific pizza topping",
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

    assert context.sync_await(incoherence_nlp_test(context, agent, incoherence_results[0]))


def test_that_an_agent_description_based_incoherency_is_detected(
    context: ContextOfTest,
) -> None:
    agent = Agent(
        id=AgentId("sparkling-water-agent"),
        name="sparkling-water-agent",
        description="You are a helpful AI assistant for a sparkling water company. Our company sells sparkling water, but never sparkling juices. The philosophy of our company dictates that juices should never be carbonated.",
        creation_utc=datetime.now(timezone.utc),
        max_engine_iterations=3,
        tags=[],
    )

    coherence_checker = context.container[CoherenceChecker]
    guideline_a = GuidelineContent(
        condition="the client asks for our recommendation",
        action="Recommend a product according to our company's philosophy",
    )

    guideline_b = GuidelineContent(
        condition="the client asks for a recommended sweetened soda",
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
    assert context.sync_await(nlp_test_action_contradiction(context, agent, incoherence_results[0]))


def test_that_many_guidelines_which_are_all_contradictory_are_detected(
    context: ContextOfTest,
    agent: Agent,
    n: int = 7,
) -> None:
    coherence_checker = context.container[CoherenceChecker]

    contradictory_guidelines = [
        GuidelineContent(
            condition="a client asks for the price of a television",
            action=f"reply that the price is {i * 10}$",
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
    context: ContextOfTest,
    agent: Agent,
) -> None:
    coherence_checker = context.container[CoherenceChecker]
    guideline_a = GuidelineContent(condition="Recommending pizza tops", action="Recommend tomatos")

    guideline_b = GuidelineContent(
        condition="asked about our toppings while inventory indicates that we are almost out of tomatoes",
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
    assert context.sync_await(incoherence_nlp_test(context, agent, incoherence_results[0]))


def test_that_seemingly_contradictory_but_actually_complementary_actions_are_not_false_positives(
    context: ContextOfTest,
    agent: Agent,
) -> None:
    coherence_checker = context.container[CoherenceChecker]
    guideline_a = GuidelineContent(
        condition="the customer is a returning customer", action="add a 5% discount to the order"
    )

    guideline_b = GuidelineContent(
        condition="the customer is a very frequent customer",
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
    assert context.sync_await(incoherence_nlp_test(context, agent, incoherence_results[0]))
