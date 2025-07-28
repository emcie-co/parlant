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

from itertools import chain
from typing import Optional, Sequence, cast

from cachetools import TTLCache

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
from parlant.core.guidelines import (
    Guideline,
    GuidelineId,
    GuidelineStore,
)
from parlant.core.journeys import Journey, JourneyStore
from parlant.core.relationships import (
    GuidelineRelationshipKind,
    RelationshipEntityKind,
    RelationshipStore,
)
from parlant.core.guideline_tool_associations import (
    GuidelineToolAssociation,
    GuidelineToolAssociationStore,
)
from parlant.core.glossary import GlossaryStore, Term
from parlant.core.sessions import (
    SessionId,
    Session,
    SessionStore,
    Event,
    MessageGenerationInspection,
    PreparationIteration,
    SessionUpdateParams,
)
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.tags import Tag
from parlant.core.tools import ToolService
from parlant.core.utterances import Utterance, UtteranceStore


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
        utterance_store: UtteranceStore,
        capability_store: CapabilityStore,
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
        self._utterance_store = utterance_store

        self.find_journeys_on_which_this_guideline_depends = TTLCache[GuidelineId, list[Journey]](
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

        all_guidelines = set(
            chain(
                agent_guidelines,
                global_guidelines,
                guidelines_for_agent_tags,
                guidelines_for_journeys,
            )
        )

        return list(all_guidelines)

    async def find_journey_scoped_guidelines(
        self,
        journey: Journey,
    ) -> Sequence[GuidelineId]:
        """Return guidelines that are dependent on the specified journey."""
        iterated_relationships = set()

        guideline_ids = set()

        relationships = set(
            await self._relationship_store.list_relationships(
                kind=GuidelineRelationshipKind.DEPENDENCY,
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
                kind=GuidelineRelationshipKind.DEPENDENCY,
                indirect=False,
                target_id=r.source.id,
            )
            if new_relationships:
                relationships.update(
                    [rel for rel in new_relationships if rel not in iterated_relationships]
                )

            iterated_relationships.add(r)

        for id in guideline_ids:
            journeys = self.find_journeys_on_which_this_guideline_depends.get(id, [])
            journeys.append(journey)

            self.find_journeys_on_which_this_guideline_depends[id] = journeys

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

        all_context_variables = set(
            chain(
                agent_context_variables,
                global_context_variables,
                context_variables_for_agent_tags,
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

        all_capabilities = set(
            chain(
                agent_capabilities,
                global_capabilities,
                capabilities_for_agent_tags,
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

        all_terms = set(chain(agent_terms, global_terms, glossary_for_agent_tags))

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

        return list(set(chain(agent_journeys, global_journeys, journeys_for_agent_tags)))

    async def find_relevant_journeys_for_context(
        self,
        available_journeys: Sequence[Journey],
        query: str,
    ) -> Sequence[Journey]:
        return await self._journey_store.find_relevant_journeys(
            query=query,
            available_journeys=available_journeys,
            max_journeys=len(available_journeys),
        )

    async def find_utterances_for_context(
        self,
        agent_id: AgentId,
        journeys: Sequence[Journey],
    ) -> Sequence[Utterance]:
        agent_utterances = await self._utterance_store.list_utterances(
            tags=[Tag.for_agent_id(agent_id)],
        )
        global_utterances = await self._utterance_store.list_utterances(tags=[])

        agent = await self._agent_store.read_agent(agent_id)
        utterances_for_agent_tags = await self._utterance_store.list_utterances(
            tags=[tag for tag in agent.tags]
        )

        journey_utterances = await self._utterance_store.list_utterances(
            tags=[Tag.for_journey_id(journey.id) for journey in journeys]
        )

        all_utterances = set(
            chain(
                agent_utterances,
                global_utterances,
                utterances_for_agent_tags,
                journey_utterances,
            )
        )

        return list(all_utterances)


class EntityCommands:
    def __init__(
        self,
        session_store: SessionStore,
        context_variable_store: ContextVariableStore,
    ) -> None:
        self._session_store = session_store
        self._context_variable_store = context_variable_store

    async def create_inspection(
        self,
        session_id: SessionId,
        correlation_id: str,
        message_generations: Sequence[MessageGenerationInspection],
        preparation_iterations: Sequence[PreparationIteration],
    ) -> None:
        await self._session_store.create_inspection(
            session_id=session_id,
            correlation_id=correlation_id,
            preparation_iterations=preparation_iterations,
            message_generations=message_generations,
        )

    async def update_session(
        self,
        session_id: SessionId,
        params: SessionUpdateParams,
    ) -> None:
        await self._session_store.update_session(session_id, params)

    async def update_context_variable_value(
        self,
        variable_id: ContextVariableId,
        key: str,
        data: JSONSerializable,
    ) -> ContextVariableValue:
        return await self._context_variable_store.update_value(variable_id, key, data)
