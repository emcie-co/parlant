# Copyright 2026 Emcie Co Ltd.
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

from itertools import chain
from typing import Mapping, Optional, Sequence, cast

from cachetools import TTLCache

from parlant.core import async_utils
from parlant.core.agents import Agent, AgentId, AgentStore
from parlant.core.capabilities import Capability, CapabilityStore
from parlant.core.common import JSONSerializable
from parlant.core.context_variables import (
    ContextVariable,
    ContextVariableId,
    ContextVariableStore,
    ContextVariableValue,
)
from parlant.core.customers import Customer, CustomerId, CustomerStore
from parlant.core.engines.alpha.tool_calling.tool_caller import ToolCallEvaluation, ToolInsights
from parlant.core.journey_guideline_projection import (
    JourneyGuidelineProjection,
    extract_node_id_from_journey_node_guideline_id,
)
from parlant.core.guidelines import (
    Guideline,
    GuidelineId,
    GuidelineStore,
)
from parlant.core.journeys import Journey, JourneyId, JourneyNodeId, JourneyStore
from parlant.core.relationships import (
    RelationshipKind,
    RelationshipEntityKind,
    RelationshipStore,
)
from parlant.core.guideline_tool_associations import (
    GuidelineToolAssociation,
    GuidelineToolAssociationStore,
)
from parlant.core.glossary import GlossaryStore, Term
from parlant.core.app_modules.sessions import SessionUpdateParamsModel
from parlant.core.playbooks import Playbook, PlaybookId, PlaybookStore
from parlant.core.sessions import (
    SessionId,
    Session,
    SessionStore,
    Event,
)
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.tags import Tag, TagId
from parlant.core.tools import ToolId, ToolService
from parlant.core.canned_responses import CannedResponse, CannedResponseStore


class EntityQueries:
    def __init__(
        self,
        agent_store: AgentStore,
        session_store: SessionStore,
        guideline_store: GuidelineStore,
        customer_store: CustomerStore,
        context_variable_store: ContextVariableStore,
        relationship_store: RelationshipStore,
        guideline_tool_association_store: GuidelineToolAssociationStore,
        glossary_store: GlossaryStore,
        journey_store: JourneyStore,
        service_registry: ServiceRegistry,
        canned_response_store: CannedResponseStore,
        capability_store: CapabilityStore,
        journey_guideline_projection: JourneyGuidelineProjection,
        playbook_store: PlaybookStore,
    ) -> None:
        self._agent_store = agent_store
        self._session_store = session_store
        self._guideline_store = guideline_store
        self._customer_store = customer_store
        self._context_variable_store = context_variable_store
        self._relationship_store = relationship_store
        self._guideline_tool_association_store = guideline_tool_association_store
        self._glossary_store = glossary_store
        self._journey_store = journey_store
        self._capability_store = capability_store
        self._service_registry = service_registry
        self._canned_response_store = canned_response_store
        self._journey_guideline_projection = journey_guideline_projection
        self._playbook_store = playbook_store

        self.guideline_and_journeys_it_depends_on = TTLCache[GuidelineId, list[Journey]](
            maxsize=1024, ttl=120
        )

    async def read_agent(
        self,
        agent_id: AgentId,
    ) -> Agent:
        return await self._agent_store.read_agent(agent_id)

    async def read_session(
        self,
        session_id: SessionId,
    ) -> Session:
        return await self._session_store.read_session(session_id)

    async def read_customer(
        self,
        customer_id: CustomerId,
    ) -> Customer:
        return await self._customer_store.read_customer(customer_id)

    async def _resolve_playbook_chain(
        self,
        playbook_id: PlaybookId,
    ) -> list[Playbook]:
        """Returns playbook chain from child to root (most specific first)."""
        playbook_chain: list[Playbook] = []
        current_id: Optional[PlaybookId] = playbook_id
        seen: set[PlaybookId] = set()

        while current_id and current_id not in seen:
            seen.add(current_id)
            playbook = await self._playbook_store.read_playbook(current_id)
            playbook_chain.append(playbook)
            current_id = playbook.parent_id

        return playbook_chain

    def _get_playbook_tags(self, playbook_chain: list[Playbook]) -> list[TagId]:
        """Returns list of playbook tags for the chain."""
        return [Tag.for_playbook_id(pb.id) for pb in playbook_chain]

    def _get_disabled_rule_ids(
        self, playbook_chain: list[Playbook], rule_type: str
    ) -> set[str]:
        """Returns set of disabled rule IDs for a given rule type from the entire playbook chain.

        Args:
            playbook_chain: List of playbooks from child to root.
            rule_type: The rule type prefix (e.g., "guideline", "term", "journey").

        Returns:
            Set of rule IDs that are disabled at any level of the chain.
        """
        if not playbook_chain:
            return set()

        prefix = f"{rule_type}:"
        prefix_len = len(prefix)
        disabled_ids: set[str] = set()

        # Collect disabled rules from ALL playbooks in the chain
        for playbook in playbook_chain:
            for rule_ref in playbook.disabled_rules:
                if rule_ref.startswith(prefix):
                    disabled_ids.add(rule_ref[prefix_len:])

        return disabled_ids

    def _get_disabled_guideline_ids(self, playbook_chain: list[Playbook]) -> set[str]:
        """Returns set of disabled guideline IDs from the entire playbook chain."""
        return self._get_disabled_rule_ids(playbook_chain, "guideline")

    async def find_guidelines_for_context(
        self,
        agent_id: AgentId,
        journeys: Sequence[Journey],
    ) -> Sequence[Guideline]:
        agent_guidelines = await self._guideline_store.list_guidelines(
            tags=[Tag.for_agent_id(agent_id)],
        )
        global_guidelines = await self._guideline_store.list_guidelines(tags=[])

        agent = await self._agent_store.read_agent(agent_id)
        guidelines_for_agent_tags = await self._guideline_store.list_guidelines(
            tags=[tag for tag in agent.tags]
        )

        guidelines_for_journeys = await self._guideline_store.list_guidelines(
            tags=[Tag.for_journey_id(journey.id) for journey in journeys]
        )

        tasks = [
            self._journey_guideline_projection.project_journey_to_guidelines(journey.id)
            for journey in journeys
            if journey.conditions  # If a journey has no conditions, it indicates that the journey cannot be activated.
        ]
        projected_journey_guidelines = await async_utils.safe_gather(*tasks)

        # Collect disabled guideline IDs from playbook chain AND agent
        disabled_guideline_ids: set[str] = set()

        # Add agent's own disabled rules
        for rule_ref in agent.disabled_rules:
            if rule_ref.startswith("guideline:"):
                disabled_guideline_ids.add(rule_ref[10:])

        # Playbook guidelines (only if agent has a playbook assigned)
        playbook_guidelines: list[Guideline] = []

        if agent.playbook_id:
            playbook_chain = await self._resolve_playbook_chain(agent.playbook_id)
            playbook_tags = self._get_playbook_tags(playbook_chain)
            # Add disabled rules from entire playbook chain
            disabled_guideline_ids.update(self._get_disabled_guideline_ids(playbook_chain))

            if playbook_tags:
                all_playbook_guidelines = await self._guideline_store.list_guidelines(
                    tags=playbook_tags
                )
                # Filter out disabled guidelines
                playbook_guidelines = [
                    g for g in all_playbook_guidelines if g.id not in disabled_guideline_ids
                ]

        # Filter global guidelines by disabled rules too
        filtered_global_guidelines = [
            g for g in global_guidelines if g.id not in disabled_guideline_ids
        ]

        all_guidelines = set(
            chain(
                agent_guidelines,
                filtered_global_guidelines,
                guidelines_for_agent_tags,
                guidelines_for_journeys,
                playbook_guidelines,
                *projected_journey_guidelines,
            )
        )

        return list(all_guidelines)

    async def find_journey_related_guidelines(
        self,
        journey: Journey,
    ) -> Sequence[GuidelineId]:
        """Return guidelines that are dependent or derived on the specified journey."""
        iterated_relationships = set()

        guideline_ids = set()

        relationships = set(
            await self._relationship_store.list_relationships(
                kind=RelationshipKind.DEPENDENCY,
                indirect=False,
                target_id=Tag.for_journey_id(journey.id),
            )
        )

        while relationships:
            r = relationships.pop()

            if r in iterated_relationships:
                continue

            if r.source.kind == RelationshipEntityKind.GUIDELINE:
                guideline_ids.add(cast(GuidelineId, r.source.id))

            new_relationships = await self._relationship_store.list_relationships(
                kind=RelationshipKind.DEPENDENCY,
                indirect=False,
                target_id=r.source.id,
            )
            if new_relationships:
                relationships.update(
                    [rel for rel in new_relationships if rel not in iterated_relationships]
                )

            iterated_relationships.add(r)

        for id in guideline_ids:
            journeys = self.guideline_and_journeys_it_depends_on.get(id, [])
            journeys.append(journey)

            self.guideline_and_journeys_it_depends_on[id] = journeys

        guideline_ids.update(
            g.id
            for g in await self._journey_guideline_projection.project_journey_to_guidelines(
                journey.id
            )
        )

        return list(guideline_ids)

    async def find_context_variables_for_context(
        self,
        agent_id: AgentId,
    ) -> Sequence[ContextVariable]:
        agent_context_variables = await self._context_variable_store.list_variables(
            tags=[Tag.for_agent_id(agent_id)],
        )
        global_context_variables = await self._context_variable_store.list_variables(tags=[])
        agent = await self._agent_store.read_agent(agent_id)
        context_variables_for_agent_tags = await self._context_variable_store.list_variables(
            tags=[tag for tag in agent.tags]
        )

        # Playbook context variables (only if agent has a playbook assigned)
        playbook_context_variables: list[ContextVariable] = []
        if agent.playbook_id:
            playbook_chain = await self._resolve_playbook_chain(agent.playbook_id)
            playbook_tags = self._get_playbook_tags(playbook_chain)
            if playbook_tags:
                playbook_context_variables = list(
                    await self._context_variable_store.list_variables(tags=playbook_tags)
                )

        all_context_variables = set(
            chain(
                agent_context_variables,
                global_context_variables,
                context_variables_for_agent_tags,
                playbook_context_variables,
            )
        )
        return list(all_context_variables)

    async def read_context_variable_value(
        self,
        variable_id: ContextVariableId,
        key: str,
    ) -> Optional[ContextVariableValue]:
        return await self._context_variable_store.read_value(variable_id, key)

    async def find_events(
        self,
        session_id: SessionId,
    ) -> Sequence[Event]:
        return await self._session_store.list_events(session_id)

    async def find_guideline_tool_associations(
        self,
    ) -> Sequence[GuidelineToolAssociation]:
        return await self._guideline_tool_association_store.list_associations()

    async def find_journey_node_tool_associations(
        self,
        node_id: JourneyNodeId,
    ) -> Sequence[ToolId]:
        return (await self._journey_store.read_node(node_id=node_id)).tools

    async def find_capabilities_for_agent(
        self,
        agent_id: AgentId,
        query: str,
        max_count: int,
    ) -> Sequence[Capability]:
        agent_capabilities = await self._capability_store.list_capabilities(
            tags=[Tag.for_agent_id(agent_id)],
        )
        global_capabilities = await self._capability_store.list_capabilities(tags=[])
        agent = await self._agent_store.read_agent(agent_id)
        capabilities_for_agent_tags = await self._capability_store.list_capabilities(
            tags=[tag for tag in agent.tags]
        )

        # Playbook capabilities (only if agent has a playbook assigned)
        playbook_capabilities: list[Capability] = []
        if agent.playbook_id:
            playbook_chain = await self._resolve_playbook_chain(agent.playbook_id)
            playbook_tags = self._get_playbook_tags(playbook_chain)
            if playbook_tags:
                playbook_capabilities = list(
                    await self._capability_store.list_capabilities(tags=playbook_tags)
                )

        all_capabilities = set(
            chain(
                agent_capabilities,
                global_capabilities,
                capabilities_for_agent_tags,
                playbook_capabilities,
            )
        )

        result = await self._capability_store.find_relevant_capabilities(
            query,
            list(all_capabilities),
            max_count=max_count,
        )

        return result

    async def find_glossary_terms_for_context(
        self,
        agent_id: AgentId,
        query: str,
    ) -> Sequence[Term]:
        agent_terms = await self._glossary_store.list_terms(
            tags=[Tag.for_agent_id(agent_id)],
        )
        global_terms = await self._glossary_store.list_terms(tags=[])
        agent = await self._agent_store.read_agent(agent_id)
        glossary_for_agent_tags = await self._glossary_store.list_terms(
            tags=[tag for tag in agent.tags]
        )

        # Playbook terms (only if agent has a playbook assigned)
        playbook_terms: list[Term] = []
        if agent.playbook_id:
            playbook_chain = await self._resolve_playbook_chain(agent.playbook_id)
            playbook_tags = self._get_playbook_tags(playbook_chain)
            if playbook_tags:
                playbook_terms = list(await self._glossary_store.list_terms(tags=playbook_tags))

        all_terms = set(chain(agent_terms, global_terms, glossary_for_agent_tags, playbook_terms))

        return await self._glossary_store.find_relevant_terms(query, list(all_terms))

    async def read_tool_service(
        self,
        service_name: str,
    ) -> ToolService:
        return await self._service_registry.read_tool_service(service_name)

    async def finds_journeys_for_context(
        self,
        agent_id: AgentId,
    ) -> Sequence[Journey]:
        agent_journeys = await self._journey_store.list_journeys(
            tags=[Tag.for_agent_id(agent_id)],
        )
        global_journeys = await self._journey_store.list_journeys(tags=[])

        agent = await self._agent_store.read_agent(agent_id)
        journeys_for_agent_tags = (
            await self._journey_store.list_journeys(tags=[tag for tag in agent.tags])
            if agent.tags
            else []
        )

        # Playbook journeys (only if agent has a playbook assigned)
        playbook_journeys: list[Journey] = []
        if agent.playbook_id:
            playbook_chain = await self._resolve_playbook_chain(agent.playbook_id)
            playbook_tags = self._get_playbook_tags(playbook_chain)
            if playbook_tags:
                playbook_journeys = list(
                    await self._journey_store.list_journeys(tags=playbook_tags)
                )

        return list(
            set(chain(agent_journeys, global_journeys, journeys_for_agent_tags, playbook_journeys))
        )

    async def sort_journeys_by_contextual_relevance(
        self,
        available_journeys: Sequence[Journey],
        query: str,
    ) -> Sequence[Journey]:
        return await self._journey_store.find_relevant_journeys(
            query=query,
            available_journeys=available_journeys,
            max_journeys=len(available_journeys),
        )

    async def find_canned_responses_for_context(
        self,
        agent: Agent,
        journeys: Sequence[Journey],
        guidelines: Sequence[Guideline],
    ) -> Sequence[CannedResponse]:
        agent_canreps = await self._canned_response_store.list_canned_responses(
            tags=[Tag.for_agent_id(agent.id)],
        )
        global_canreps = await self._canned_response_store.list_canned_responses(tags=[])

        canreps_for_agent_tags = await self._canned_response_store.list_canned_responses(
            tags=[tag for tag in agent.tags]
        )

        journey_canreps = await self._canned_response_store.list_canned_responses(
            tags=[Tag.for_journey_id(journey.id) for journey in journeys]
        )

        guideline_canreps = await self.find_canned_responses_for_guidelines(guidelines)

        # Playbook canned responses (only if agent has a playbook assigned)
        playbook_canreps: list[CannedResponse] = []
        if agent.playbook_id:
            playbook_chain = await self._resolve_playbook_chain(agent.playbook_id)
            playbook_tags = self._get_playbook_tags(playbook_chain)
            if playbook_tags:
                playbook_canreps = list(
                    await self._canned_response_store.list_canned_responses(tags=playbook_tags)
                )

        all_canreps = set(
            chain(
                agent_canreps,
                global_canreps,
                canreps_for_agent_tags,
                journey_canreps,
                guideline_canreps,
                playbook_canreps,
            )
        )

        return list(all_canreps)

    async def find_canned_responses_for_guidelines(
        self,
        guidelines: Sequence[Guideline],
    ) -> Sequence[CannedResponse]:
        tags = []

        for g in guidelines:
            if g.id.startswith("journey_node:"):
                tags.append(
                    Tag.for_journey_node_id(extract_node_id_from_journey_node_guideline_id(g.id))
                )

            else:
                tags.append(Tag.for_guideline_id(g.id))

        return await self._canned_response_store.list_canned_responses(tags=tags)

    async def find_guidelines_that_need_reevaluation(
        self,
        available_guidelines: dict[GuidelineId, Guideline],
        active_journeys: Sequence[Journey],
        tool_insights: ToolInsights,
    ) -> Sequence[Guideline]:
        """Find guidelines that need reevaluation based on the tool calls made."""

        if not tool_insights.evaluations:
            return []

        executed_tool_ids = [
            tid for tid, e in tool_insights.evaluations if e == ToolCallEvaluation.NEEDS_TO_RUN
        ]

        active_journeys_mapping = {journey.id: journey for journey in active_journeys}
        guidelines: list[Guideline] = []

        tasks = [
            self._relationship_store.list_relationships(
                kind=RelationshipKind.REEVALUATION,
                indirect=False,
                target_id=tool_id,
            )
            for tool_id in set(tid for tid, _ in tool_insights.evaluations)
        ]

        reevaluation_relationships = list(
            chain.from_iterable(await async_utils.safe_gather(*tasks))
        )

        for relationship in reevaluation_relationships:
            # See if the relationship's guideline is within
            # the set of guidelines we were given to check.
            guideline_to_reevaluate = next(
                (
                    g
                    for gid, g in available_guidelines.items()
                    if gid.startswith(relationship.source.id)
                ),
                None,
            )

            if not guideline_to_reevaluate:
                # Couldn't find this relationship's guideline
                # in the set of available guidelines to check.
                continue

            the_id_of_the_tool_related_to_the_guideline_to_reevaluate = relationship.target.id

            # At this point we know that one of the guidelines given to us
            # has a reevaluation relationship with one of the relevant tools.

            if guideline_to_reevaluate.metadata.get("journey_node"):
                # We found a journey node that has a reevaluation relationship with one of the tools.
                #
                # This journey node is by definition a tool node.
                #
                # Now, this actually means we need to reevaluate the entire journey,
                # so we'll need to add all of its projected guidelines to the list.

                # The only exception to this rule here is if the tool was deliberately skipped
                # because the context already existed in the session.

                # FIXME: Strictly speaking, we should only reevaluate the journey if the tool
                # was called ON BEHALF OF THE JOURNEY NODE — since it could have been called
                # for some other reason, e.g. due to an unrelated guideline.

                tool_should_be_considered_as_having_been_called = all(
                    e
                    in [ToolCallEvaluation.DATA_ALREADY_IN_CONTEXT, ToolCallEvaluation.NEEDS_TO_RUN]
                    for tool_id, e in tool_insights.evaluations
                    if tool_id == the_id_of_the_tool_related_to_the_guideline_to_reevaluate
                )

                if tool_should_be_considered_as_having_been_called:
                    journey_id = cast(
                        JourneyId,
                        cast(
                            Mapping[str, JSONSerializable],
                            guideline_to_reevaluate.metadata["journey_node"],
                        ).get("journey_id"),
                    )

                    if journey_id in active_journeys_mapping:
                        projected_journey_guidelines = (
                            await self._journey_guideline_projection.project_journey_to_guidelines(
                                journey_id
                            )
                        )

                        guidelines.extend(projected_journey_guidelines)
            else:
                # For normal guidelines, we only reevaluate them if their related
                # tool WAS JUST executed -- not if it was skipped.
                if the_id_of_the_tool_related_to_the_guideline_to_reevaluate in executed_tool_ids:
                    guidelines.append(guideline_to_reevaluate)

        return list(set(guidelines))


class EntityCommands:
    def __init__(
        self,
        session_store: SessionStore,
        context_variable_store: ContextVariableStore,
    ) -> None:
        self._session_store = session_store
        self._context_variable_store = context_variable_store

    async def update_session(
        self,
        session_id: SessionId,
        params: SessionUpdateParamsModel,
    ) -> None:
        await self._session_store.update_session(session_id, params)

    async def update_context_variable_value(
        self,
        variable_id: ContextVariableId,
        key: str,
        data: JSONSerializable,
    ) -> ContextVariableValue:
        return await self._context_variable_store.update_value(variable_id, key, data)
