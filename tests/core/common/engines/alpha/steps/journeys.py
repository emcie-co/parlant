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

from collections.abc import Sequence
from typing import cast
from pytest_bdd import given, parsers

from parlant.core.entity_cq import EntityCommands
from parlant.core.journeys import Journey, JourneyStore
from parlant.core.guidelines import Guideline, GuidelineId, GuidelineStore

from parlant.core.relationships import (
    RelationshipEntity,
    RelationshipEntityKind,
    RelationshipKind,
    RelationshipStore,
)
from parlant.core.sessions import AgentState, SessionId, SessionStore, SessionUpdateParams
from parlant.core.tags import Tag
from parlant.core.tools import LocalToolService, ToolId
from tests.core.common.engines.alpha.steps.tools import TOOLS
from tests.core.common.engines.alpha.utils import step
from tests.core.common.utils import ContextOfTest


@step(
    given,
    parsers.parse(
        'a journey titled "{journey_title}" to {journey_description} when {a_condition_holds}'
    ),
)
def given_a_journey_to_when(
    context: ContextOfTest,
    journey_title: str,
    journey_description: str,
    a_condition_holds: str,
) -> None:
    guideline_store = context.container[GuidelineStore]
    journey_store = context.container[JourneyStore]

    conditioning_guideline: Guideline = context.sync_await(
        guideline_store.create_guideline(condition=a_condition_holds, action=None)
    )

    journey = context.sync_await(
        journey_store.create_journey(
            conditions=[conditioning_guideline.id],
            title=journey_title,
            description=journey_description,
        )
    )

    context.journeys[journey.title] = journey


@step(
    given,
    parsers.parse('the journey called "{journey_title}"'),
)
def given_the_journey_called(
    context: ContextOfTest,
    journey_title: str,
) -> Journey:
    journey_store = context.container[JourneyStore]
    guideline_store = context.container[GuidelineStore]
    relationship_store = context.container[RelationshipStore]
    local_tool_service = context.container[LocalToolService]

    def create_reset_password_journey() -> Journey:
        conditions = [
            "the customer wants to reset their password",
            "the customer can't remember their password",
        ]

        condition_guidelines: Sequence[Guideline] = [
            context.sync_await(
                guideline_store.create_guideline(
                    condition=condition,
                    action=None,
                    metadata={},
                )
            )
            for condition in conditions
        ]

        journey = context.sync_await(
            journey_store.create_journey(
                title="reset password journey",
                description="",
                conditions=[c.id for c in condition_guidelines],
                tags=[],
            )
        )

        for c in condition_guidelines:
            context.sync_await(
                guideline_store.upsert_tag(
                    guideline_id=c.id,
                    tag_id=Tag.for_journey_id(journey_id=journey.id),
                )
            )

        node1 = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action="ask for their account name",
                tools=[],
            )
        )
        context.sync_await(
            journey_store.set_node_metadata(
                node1.id,
                "customer_dependent_action_data",
                {
                    "is_customer_dependent": True,
                    "customer_action": "",
                    "agent_action": "",
                },
            )
        )
        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=journey.root_id,
                target=node1.id,
                condition="The customer has not provided their account number",
            )
        )

        node2 = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action="Ask for their email address or phone number",
                tools=[],
            )
        )

        context.sync_await(
            journey_store.set_node_metadata(
                node2.id,
                "customer_dependent_action_data",
                {
                    "is_customer_dependent": True,
                    "customer_action": "",
                    "agent_action": "",
                },
            )
        )
        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=node1.id,
                target=node2.id,
                condition="The customer provided their account number",
            )
        )
        node3 = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action="Wish them a good day",
                tools=[],
            )
        )
        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=node2.id,
                target=node3.id,
                condition="The customer provided their email address or phone number",
            )
        )

        tool = context.sync_await(local_tool_service.create_tool(**TOOLS["reset_password"]))
        node4 = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action="Use the reset_password tool with the provided information",
                tools=[ToolId("local", tool.name)],
            )
        )
        context.sync_await(
            journey_store.set_node_metadata(
                node4.id,
                "tool_running_only",
                True,
            )
        )
        context.sync_await(
            relationship_store.create_relationship(
                source=RelationshipEntity(
                    id=ToolId("local", tool.name),
                    kind=RelationshipEntityKind.TOOL,
                ),
                target=RelationshipEntity(
                    id=Tag.for_journey_node_id(node4.id),
                    kind=RelationshipEntityKind.TAG,
                ),
                kind=RelationshipKind.REEVALUATION,
            )
        )

        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=node3.id,
                target=node4.id,
                condition="The customer wished you a good day in return",
            )
        )

        node5 = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action="Report the result to the customer",
                tools=[],
            )
        )
        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=node4.id,
                target=node5.id,
                condition="reset_password tool returned that the password was successfully reset",
            )
        )

        node6 = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action="Apologize to the customer and report that the password cannot be reset at this times",
                tools=[],
            )
        )

        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=node3.id,
                target=node6.id,
                condition="The customer did not immediately wish you a good day in return",
            )
        )
        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=node4.id,
                target=node6.id,
                condition="reset_password tool returned that the password was not successfully reset, or otherwise failed",
            )
        )

        return journey

    def create_book_flight_journey() -> Journey:
        conditions = [
            "the customer wants to book a flight",
        ]

        condition_guidelines: Sequence[Guideline] = [
            context.sync_await(
                guideline_store.create_guideline(
                    condition=condition,
                    action=None,
                    metadata={},
                )
            )
            for condition in conditions
        ]

        journey = context.sync_await(
            journey_store.create_journey(
                title="book flight journey",
                description="",
                conditions=[c.id for c in condition_guidelines],
                tags=[],
            )
        )

        for c in condition_guidelines:
            context.sync_await(
                guideline_store.upsert_tag(
                    guideline_id=c.id,
                    tag_id=Tag.for_journey_id(journey_id=journey.id),
                )
            )

        node1 = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action="ask for the source and destination airport",
                tools=[],
            )
        )
        context.sync_await(
            journey_store.set_node_metadata(
                node1.id,
                "customer_dependent_action_data",
                {
                    "is_customer_dependent": True,
                    "customer_action": "",
                    "agent_action": "",
                },
            )
        )

        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=journey.root_id,
                target=node1.id,
                condition="",
            )
        )

        node2 = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action="ask for the dates of the departure and return flight",
                tools=[],
            )
        )
        context.sync_await(
            journey_store.set_node_metadata(
                node2.id,
                "customer_dependent_action_data",
                {
                    "is_customer_dependent": True,
                    "customer_action": "",
                    "agent_action": "",
                },
            )
        )

        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=node1.id,
                target=node2.id,
                condition="",
            )
        )

        node3 = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action="ask whether they want economy or business class",
                tools=[],
            )
        )
        context.sync_await(
            journey_store.set_node_metadata(
                node3.id,
                "customer_dependent_action_data",
                {
                    "is_customer_dependent": True,
                    "customer_action": "",
                    "agent_action": "",
                },
            )
        )

        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=node2.id,
                target=node3.id,
                condition="",
            )
        )

        node4 = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action="ask for the name of the traveler",
                tools=[],
            )
        )
        context.sync_await(
            journey_store.set_node_metadata(
                node4.id,
                "customer_dependent_action_data",
                {
                    "is_customer_dependent": True,
                    "customer_action": "",
                    "agent_action": "",
                },
            )
        )

        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=node3.id,
                target=node4.id,
                condition="",
            )
        )

        tool = context.sync_await(local_tool_service.create_tool(**TOOLS["book_flight"]))

        node5 = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action="book the flight using book_flight tool and the provided details",
                tools=[ToolId("local", tool.name)],
            )
        )

        context.sync_await(
            journey_store.set_node_metadata(
                node5.id,
                "tool_running_only",
                True,
            )
        )
        context.sync_await(
            relationship_store.create_relationship(
                source=RelationshipEntity(
                    id=ToolId("local", tool.name),
                    kind=RelationshipEntityKind.TOOL,
                ),
                target=RelationshipEntity(
                    id=Tag.for_journey_node_id(node5.id),
                    kind=RelationshipEntityKind.TAG,
                ),
                kind=RelationshipKind.REEVALUATION,
            )
        )

        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=node4.id,
                target=node5.id,
                condition="",
            )
        )

        return journey

    def create_book_taxi_journey() -> Journey:
        conditions = [
            "the customer wants to book a taxi ride",
        ]

        condition_guidelines: Sequence[Guideline] = [
            context.sync_await(
                guideline_store.create_guideline(
                    condition=condition,
                    action=None,
                    metadata={},
                )
            )
            for condition in conditions
        ]

        journey = context.sync_await(
            journey_store.create_journey(
                title="book taxi ride journey",
                description="",
                conditions=[c.id for c in condition_guidelines],
                tags=[],
            )
        )

        for c in condition_guidelines:
            context.sync_await(
                guideline_store.upsert_tag(
                    guideline_id=c.id,
                    tag_id=Tag.for_journey_id(journey_id=journey.id),
                )
            )

        node1 = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action="Ask for the pickup location",
                tools=[],
            )
        )
        context.sync_await(
            journey_store.set_node_metadata(
                node1.id,
                "customer_dependent_action_data",
                {
                    "is_customer_dependent": True,
                    "customer_action": "",
                    "agent_action": "",
                },
            )
        )

        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=journey.root_id,
                target=node1.id,
                condition="",
            )
        )

        node2 = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action="Ask for the drop-off location",
                tools=[],
            )
        )
        context.sync_await(
            journey_store.set_node_metadata(
                node2.id,
                "customer_dependent_action_data",
                {
                    "is_customer_dependent": True,
                    "customer_action": "",
                    "agent_action": "",
                },
            )
        )

        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=node1.id,
                target=node2.id,
                condition="",
            )
        )

        node3 = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action="Ask for the desired pickup time",
                tools=[],
            )
        )
        context.sync_await(
            journey_store.set_node_metadata(
                node3.id,
                "customer_dependent_action_data",
                {
                    "is_customer_dependent": True,
                    "customer_action": "",
                    "agent_action": "",
                },
            )
        )

        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=node2.id,
                target=node3.id,
                condition="",
            )
        )

        node4 = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action="Confirm all details with the customer before booking",
                tools=[],
            )
        )
        context.sync_await(
            journey_store.set_node_metadata(
                node4.id,
                "customer_dependent_action_data",
                {
                    "is_customer_dependent": True,
                    "customer_action": "",
                    "agent_action": "",
                },
            )
        )

        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=node3.id,
                target=node4.id,
                condition="",
            )
        )

        return journey

    def create_place_food_order_journey() -> Journey:
        conditions = [
            "the customer wants to order food",
        ]

        condition_guidelines: Sequence[Guideline] = [
            context.sync_await(
                guideline_store.create_guideline(
                    condition=condition,
                    action=None,
                    metadata={},
                )
            )
            for condition in conditions
        ]

        journey = context.sync_await(
            journey_store.create_journey(
                title="place food order journey",
                description="",
                conditions=[c.id for c in condition_guidelines],
                tags=[],
            )
        )

        for c in condition_guidelines:
            context.sync_await(
                guideline_store.upsert_tag(
                    guideline_id=c.id,
                    tag_id=Tag.for_journey_id(journey_id=journey.id),
                )
            )

        node1 = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action="Ask if they’d like a salad or a sandwich",
                tools=[],
            )
        )
        context.sync_await(
            journey_store.set_node_metadata(
                node1.id,
                "customer_dependent_action_data",
                {
                    "is_customer_dependent": True,
                    "customer_action": "",
                    "agent_action": "",
                },
            )
        )

        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=journey.root_id,
                target=node1.id,
                condition="",
            )
        )

        node2 = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action="ask what kind of bread they’d like",
                tools=[],
            )
        )
        context.sync_await(
            journey_store.set_node_metadata(
                node2.id,
                "customer_dependent_action_data",
                {
                    "is_customer_dependent": True,
                    "customer_action": "",
                    "agent_action": "",
                },
            )
        )

        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=node1.id,
                target=node2.id,
                condition="they choose a sandwich",
            )
        )

        node3 = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action="ask what main filling they’d like from: Peanut butter, jam or pesto",
                tools=[],
            )
        )
        context.sync_await(
            journey_store.set_node_metadata(
                node3.id,
                "customer_dependent_action_data",
                {
                    "is_customer_dependent": True,
                    "customer_action": "",
                    "agent_action": "",
                },
            )
        )

        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=node2.id,
                target=node3.id,
                condition="",
            )
        )

        node4 = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action="ask if they want any extras",
                tools=[],
            )
        )
        context.sync_await(
            journey_store.set_node_metadata(
                node4.id,
                "customer_dependent_action_data",
                {
                    "is_customer_dependent": True,
                    "customer_action": "",
                    "agent_action": "",
                },
            )
        )

        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=node3.id,
                target=node4.id,
                condition="",
            )
        )

        node5 = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action="ask what base greens they want",
                tools=[],
            )
        )
        context.sync_await(
            journey_store.set_node_metadata(
                node5.id,
                "customer_dependent_action_data",
                {
                    "is_customer_dependent": True,
                    "customer_action": "",
                    "agent_action": "",
                },
            )
        )

        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=node1.id,
                target=node5.id,
                condition="they choose a salad",
            )
        )

        node6 = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action="what toppings they’d like",
                tools=[],
            )
        )
        context.sync_await(
            journey_store.set_node_metadata(
                node6.id,
                "customer_dependent_action_data",
                {
                    "is_customer_dependent": True,
                    "customer_action": "",
                    "agent_action": "",
                },
            )
        )
        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=node5.id,
                target=node6.id,
                condition="",
            )
        )

        node7 = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action="what kind of dressing they prefer",
                tools=[],
            )
        )
        context.sync_await(
            journey_store.set_node_metadata(
                node7.id,
                "customer_dependent_action_data",
                {
                    "is_customer_dependent": True,
                    "customer_action": "",
                    "agent_action": "",
                },
            )
        )

        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=node6.id,
                target=node7.id,
                condition="",
            )
        )

        node8 = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action="Confirm the full order before placing it",
                tools=[],
            )
        )
        context.sync_await(
            journey_store.set_node_metadata(
                node8.id,
                "customer_dependent_action_data",
                {
                    "is_customer_dependent": True,
                    "customer_action": "",
                    "agent_action": "",
                },
            )
        )
        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=node7.id,
                target=node8.id,
                condition="",
            )
        )
        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=node4.id,
                target=node8.id,
                condition="",
            )
        )

        return journey

    def create_decrease_spending_journey() -> Journey:
        conditions = [
            "the customer asks about decreasing their spending",
        ]

        condition_guidelines: Sequence[Guideline] = [
            context.sync_await(
                guideline_store.create_guideline(
                    condition=condition,
                    action=None,
                    metadata={},
                )
            )
            for condition in conditions
        ]

        journey = context.sync_await(
            journey_store.create_journey(
                title="decrease spending journey",
                description="",
                conditions=[c.id for c in condition_guidelines],
                tags=[],
            )
        )

        for c in condition_guidelines:
            context.sync_await(
                guideline_store.upsert_tag(
                    guideline_id=c.id,
                    tag_id=Tag.for_journey_id(journey_id=journey.id),
                )
            )

        node1 = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action="ask for the customer's account number",
                tools=[],
            )
        )
        context.sync_await(
            journey_store.set_node_metadata(
                node1.id,
                "customer_dependent_action_data",
                {
                    "is_customer_dependent": True,
                    "customer_action": "",
                    "agent_action": "",
                },
            )
        )

        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=journey.root_id,
                target=node1.id,
                condition="",
            )
        )

        node2 = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action="Ask for the customer's full name",
                tools=[],
            )
        )
        context.sync_await(
            journey_store.set_node_metadata(
                node2.id,
                "customer_dependent_action_data",
                {
                    "is_customer_dependent": True,
                    "customer_action": "",
                    "agent_action": "",
                },
            )
        )

        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=node1.id,
                target=node2.id,
                condition="",
            )
        )

        node3 = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action="suggest all relevant capabilities available in this prompt",
                tools=[],
            )
        )

        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=node2.id,
                target=node3.id,
                condition="",
            )
        )

        node4 = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action="inform the customer that you cannot help them with their request",
                tools=[],
            )
        )
        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=node3.id,
                target=node4.id,
                condition="No relevant capability is available",
            )
        )

        node5 = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action="Ask the customer if they need any further help",
                tools=[],
            )
        )
        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=node4.id,
                target=node5.id,
                condition="",
            )
        )
        return journey

    def create_request_loan_journey() -> Journey:
        conditions = [
            "the customer is interested in applying for a loan",
        ]

        condition_guidelines: Sequence[Guideline] = [
            context.sync_await(
                guideline_store.create_guideline(
                    condition=condition,
                    action=None,
                    metadata={},
                )
            )
            for condition in conditions
        ]

        journey = context.sync_await(
            journey_store.create_journey(
                title="Loan Application Request",
                description="",
                conditions=[c.id for c in condition_guidelines],
                tags=[],
            )
        )

        for c in condition_guidelines:
            context.sync_await(
                guideline_store.upsert_tag(
                    guideline_id=c.id,
                    tag_id=Tag.for_journey_id(journey_id=journey.id),
                )
            )

        node1 = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action="ask what type of loan the customer is interested in",
                tools=[],
            )
        )
        context.sync_await(
            journey_store.set_node_metadata(
                node1.id,
                "customer_dependent_action_data",
                {
                    "is_customer_dependent": True,
                    "customer_action": "",
                    "agent_action": "",
                },
            )
        )

        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=journey.root_id,
                target=node1.id,
                condition="",
            )
        )

        node2 = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action="Ask for the loan amount",
                tools=[],
            )
        )
        context.sync_await(
            journey_store.set_node_metadata(
                node2.id,
                "customer_dependent_action_data",
                {
                    "is_customer_dependent": True,
                    "customer_action": "",
                    "agent_action": "",
                },
            )
        )

        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=node1.id,
                target=node2.id,
                condition="the customer requested a personal loan",
            )
        )

        node3 = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action="Ask for the purpose of the loan",
                tools=[],
            )
        )
        context.sync_await(
            journey_store.set_node_metadata(
                node3.id,
                "customer_dependent_action_data",
                {
                    "is_customer_dependent": True,
                    "customer_action": "",
                    "agent_action": "",
                },
            )
        )

        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=node2.id,
                target=node3.id,
                condition=None,
            )
        )

        node4 = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action="Ask for account number for validation",
                tools=[],
            )
        )
        context.sync_await(
            journey_store.set_node_metadata(
                node3.id,
                "customer_dependent_action_data",
                {
                    "is_customer_dependent": True,
                    "customer_action": "",
                    "agent_action": "",
                },
            )
        )

        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=node3.id,
                target=node4.id,
                condition=None,
            )
        )
        tool = context.sync_await(local_tool_service.create_tool(**TOOLS["check_eligibility"]))

        node5 = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action="Validate the customer's eligibility",
                tools=[ToolId("local", tool.name)],
            )
        )
        context.sync_await(
            journey_store.set_node_metadata(
                node5.id,
                "customer_dependent_action_data",
                {
                    "is_customer_dependent": False,
                    "customer_action": "",
                    "agent_action": "",
                },
            )
        )

        context.sync_await(
            journey_store.set_node_metadata(
                node5.id,
                "tool_running_only",
                True,
            )
        )

        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=node4.id,
                target=node5.id,
                condition=None,
            )
        )

        node6 = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action="Confirm eligibility with terms and ask to proceed with application",
                tools=[],
            )
        )
        context.sync_await(
            journey_store.set_node_metadata(
                node6.id,
                "customer_dependent_action_data",
                {
                    "is_customer_dependent": True,
                    "customer_action": "",
                    "agent_action": "",
                },
            )
        )
        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=node5.id,
                target=node6.id,
                condition="If the account is eligible",
            )
        )

        node6 = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action="Confirm eligibility with terms and ask to proceed with application",
                tools=[],
            )
        )
        context.sync_await(
            journey_store.set_node_metadata(
                node6.id,
                "customer_dependent_action_data",
                {
                    "is_customer_dependent": True,
                    "customer_action": "",
                    "agent_action": "",
                },
            )
        )
        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=node5.id,
                target=node6.id,
                condition="If the account is eligible",
            )
        )

        node7 = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action="Explain the denied request for a loan due to ineligibility",
                tools=[],
            )
        )
        context.sync_await(
            journey_store.set_node_metadata(
                node7.id,
                "customer_dependent_action_data",
                {
                    "is_customer_dependent": False,
                    "customer_action": "",
                    "agent_action": "",
                },
            )
        )

        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=node5.id,
                target=node7.id,
                condition="Account is not eligible",
            )
        )

        return journey

    JOURNEYS = {
        "Reset Password Journey": create_reset_password_journey,
        "Book Flight": create_book_flight_journey,
        "Book Taxi Ride": create_book_taxi_journey,
        "Place Food Order": create_place_food_order_journey,
        "Decrease Spending Journey": create_decrease_spending_journey,
        "Request Loan Journey": create_request_loan_journey,
    }

    create_journey_func = JOURNEYS[journey_title]
    journey = create_journey_func()
    context.journeys[journey_title] = journey

    return journey


@step(
    given,
    parsers.parse('a journey path "{journey_path}" for the journey "{journey_title}"'),
)
def given_a_journey_path_for_the_journey(
    context: ContextOfTest,
    journey_path: str,
    journey_title: str,
    session_id: SessionId,
) -> None:
    session_store = context.container[SessionStore]
    entity_commands = context.container[EntityCommands]

    session = context.sync_await(session_store.read_session(session_id))

    path = journey_path.strip("[]").split(", ")
    guideline_path = [cast(GuidelineId | None, p) for p in path]

    journey = context.journeys[journey_title]

    context.sync_await(
        entity_commands.update_session(
            session_id=session.id,
            params=SessionUpdateParams(
                agent_states=list(session.agent_states)
                + [
                    AgentState(
                        correlation_id="<main>",
                        applied_guideline_ids=[],
                        journey_paths={journey.id: guideline_path},
                    )
                ]
            ),
        )
    )


# todo - add a version with description?
@step(
    given,
    parsers.parse('a journey "{journey_title}"'),
)
def given_a_journey_titled(
    context: ContextOfTest,
    journey_title: str,
) -> Journey:
    journey_store = context.container[JourneyStore]

    journey = context.sync_await(
        journey_store.create_journey(
            title=journey_title,
            description="",
            conditions=[],
            tags=[],
        )
    )

    context.journeys[journey_title] = journey

    return journey


@step(
    given,
    parsers.parse('the journey "{journey_title}" is triggered when {condition}'),
)
def given_the_journey_is_triggered_when(
    context: ContextOfTest,
    journey_title: str,
    condition: str,
) -> Journey:
    journey_store = context.container[JourneyStore]
    guideline_store = context.container[GuidelineStore]

    journey = context.journeys[journey_title]

    guideline_condition = context.sync_await(
        guideline_store.create_guideline(
            condition=condition,
            action=None,
            metadata={},
        )
    )
    context.sync_await(
        journey_store.add_condition(
            journey_id=journey.id,
            condition=guideline_condition.id,
        )
    )

    context.sync_await(
        guideline_store.upsert_tag(
            guideline_id=guideline_condition.id,
            tag_id=Tag.for_journey_id(journey_id=journey.id),
        )
    )

    context.journeys[journey_title] = journey

    return journey


@step(
    given,
    parsers.parse('a node "{node_name}" to {action} in "{journey_title}" journey'),
)
def given_a_node_with_an_action_in_journey(
    context: ContextOfTest,
    node_name: str,
    action: str,
    journey_title: str,
) -> None:
    journey_store = context.container[JourneyStore]

    journey = context.journeys[journey_title]

    node = context.sync_await(
        journey_store.create_node(
            journey_id=journey.id,
            action=action,
            tools=[],
        )
    )

    context.nodes[node_name] = node


@step(
    given,
    parsers.parse('a terminating node "{node_name}" in "{journey_title}" journey'),
)
def given_a_terminating_node(
    context: ContextOfTest,
    node_name: str,
    journey_title: str,
) -> None:
    journey_store = context.container[JourneyStore]

    journey = context.journeys[journey_title]

    node = context.sync_await(
        journey_store.create_node(
            journey_id=journey.id,
            action=None,
            tools=[],
        )
    )

    context.nodes[node_name] = node


@step(
    given,
    parsers.parse('the node "{node_name}" uses the tool "{tool_name}"'),
)
def given_the_node_uses_the_tool(
    context: ContextOfTest,
    tool_name: str,
    node_name: str,
) -> None:
    journey_store = context.container[JourneyStore]
    local_tool_service = context.container[LocalToolService]

    node = context.nodes[node_name]

    tool = context.sync_await(local_tool_service.create_tool(**TOOLS[tool_name]))

    new_node_tools = list(node.tools) + [ToolId("local", tool.name)]
    context.sync_await(journey_store.update_node(node_id=node.id, params={"tools": new_node_tools}))

    context.nodes[node_name] = node


@step(
    given,
    parsers.parse("the node {node_name} requires customer input"),
)
def given_the_node_requires_customer_input(
    context: ContextOfTest,
    node_name: str,
) -> None:
    journey_store = context.container[JourneyStore]

    node = context.nodes[node_name]

    context.sync_await(
        journey_store.set_node_metadata(
            node.id,
            "customer_dependent_action_data",
            {
                "is_customer_dependent": True,
                "customer_action": "",
                "agent_action": "",
            },
        )
    )


@step(
    given,
    parsers.parse("the node {node_name} is tool running only"),
)
def given_the_node_is_tool_running_only(
    context: ContextOfTest,
    node_name: str,
) -> None:
    journey_store = context.container[JourneyStore]
    relationship_store = context.container[RelationshipStore]

    node = context.nodes[node_name]

    context.sync_await(
        journey_store.set_node_metadata(
            node.id,
            "tool_running_only",
            True,
        )
    )

    for tool_id in node.tools:  # Assume all associated tools were added
        context.sync_await(
            relationship_store.create_relationship(
                source=RelationshipEntity(
                    id=tool_id,
                    kind=RelationshipEntityKind.TOOL,
                ),
                target=RelationshipEntity(
                    id=Tag.for_journey_node_id(node.id),
                    kind=RelationshipEntityKind.TAG,
                ),
                kind=RelationshipKind.REEVALUATION,
            )
        )


@step(
    given,
    parsers.parse(
        'a transition from the root to {node_name} when {condition} in "{journey_title}" journey'
    ),
)
def given_a_transition_from_to_the_node_when_in_journey(
    context: ContextOfTest,
    node_name: str,
    condition: str,
    journey_title: str,
) -> None:
    journey_store = context.container[JourneyStore]
    journey = context.journeys[journey_title]
    node = context.nodes[node_name]

    context.sync_await(
        journey_store.create_edge(
            journey_id=journey.id,
            source=journey.root_id,
            target=node.id,
            condition=condition,
        )
    )


@step(
    given,
    parsers.parse('a transition from the root to "{node_name}" in "{journey_title}" journey'),
)
def given_a_transition_from_root_to_the_node_in_journey(
    context: ContextOfTest,
    node_name: str,
    journey_title: str,
) -> None:
    journey_store = context.container[JourneyStore]
    journey = context.journeys[journey_title]
    node = context.nodes[node_name]

    context.sync_await(
        journey_store.create_edge(
            journey_id=journey.id,
            source=JourneyStore.ROOT_NODE_ID,
            target=node.id,
            condition=None,
        )
    )


@step(
    given,
    parsers.parse(
        'a transition from "{node_name1}" to "{node_name2}" when {condition} in "{journey_title}" journey'
    ),
)
def given_a_transition_from_to_when_in_journey(
    context: ContextOfTest,
    node_name1: str,
    node_name2: str,
    condition: str,
    journey_title: str,
) -> None:
    journey_store = context.container[JourneyStore]
    journey = context.journeys[journey_title]

    node1 = context.nodes[node_name1]
    node2 = context.nodes[node_name2]

    context.sync_await(
        journey_store.create_edge(
            journey_id=journey.id,
            source=node1.id,
            target=node2.id,
            condition=condition,
        )
    )


@step(
    given,
    parsers.parse(
        'a transition from "{node_name1}"" to "{node_name2}" in "{journey_title}" journey'
    ),
)
def given_a_transition_from_to_in_journey(
    context: ContextOfTest,
    node_name1: str,
    node_name2: str,
    journey_title: str,
) -> None:
    journey_store = context.container[JourneyStore]
    journey = context.journeys[journey_title]

    node1 = context.nodes[node_name1]
    node2 = context.nodes[node_name2]

    context.sync_await(
        journey_store.create_edge(
            journey_id=journey.id,
            source=node1.id,
            target=node2.id,
            condition=None,
        )
    )
