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

# from parlant.core.engines.alpha.journey_guideline_projection import JourneyGuidelineProjection
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
    # journey_guideline_projection = context.container[JourneyGuidelineProjection]
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
                source=JourneyStore.ROOT_NODE_ID,
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
        none_node = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action=None,
                tools=[],
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

        context.sync_await(
            journey_store.create_edge(
                journey_id=journey.id,
                source=node3.id,
                target=none_node.id,
                condition=None,
            )
        )

        node5 = context.sync_await(
            journey_store.create_node(
                journey_id=journey.id,
                action="Report the result to the customer",
                tools=[],
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
                source=node4.id,
                target=node5.id,
                condition="reset_password tool returned that the password was successfully reset",
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

        # guidelines = context.sync_await(
        #     journey_guideline_projection.project_journey_to_guidelines(journey_id=journey.id)
        # )

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
                source=JourneyStore.ROOT_NODE_ID,
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
                source=JourneyStore.ROOT_NODE_ID,
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

        # guidelines = context.sync_await(
        #     journey_guideline_projection.project_journey_to_guidelines(journey_id=journey.id)
        # )

        return journey

    def create_place_food_order_journey() -> Journey:
        conditions = [
            "the customer wants to order food ",
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
                source=JourneyStore.ROOT_NODE_ID,
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

        # guidelines = context.sync_await(
        #     journey_guideline_projection.project_journey_to_guidelines(journey_id=journey.id)
        # )

        return journey

    JOURNEYS = {
        "Reset Password Journey": create_reset_password_journey,
        "Book Flight": create_book_flight_journey,
        "Book Taxi Ride": create_book_taxi_journey,
        "Place Food Order": create_place_food_order_journey,
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
    guideline_path = [cast(GuidelineId, p) for p in path]

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
