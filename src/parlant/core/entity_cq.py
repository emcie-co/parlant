from itertools import chain
from typing import Optional, Sequence

from parlant.core.agents import Agent, AgentId, AgentStore
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
from parlant.core.guideline_connections import GuidelineConnection, GuidelineConnectionStore
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
from parlant.core.tags import Tag, TagId
from parlant.core.tools import ToolService


class EntityQueries:
    def __init__(
        self,
        agent_store: AgentStore,
        session_store: SessionStore,
        guideline_store: GuidelineStore,
        customer_store: CustomerStore,
        context_variable_store: ContextVariableStore,
        guideline_connection_store: GuidelineConnectionStore,
        guideline_tool_association_store: GuidelineToolAssociationStore,
        glossary_store: GlossaryStore,
        service_registry: ServiceRegistry,
    ) -> None:
        self._agent_store = agent_store
        self._session_store = session_store
        self._guideline_store = guideline_store
        self._customer_store = customer_store
        self._context_variable_store = context_variable_store
        self._guideline_connection_store = guideline_connection_store
        self._guideline_tool_association_store = guideline_tool_association_store
        self._glossary_store = glossary_store
        self._service_registry = service_registry

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

    async def find_guidelines_for_agent(
        self,
        agent_id: AgentId,
    ) -> Sequence[Guideline]:
        agent_guidelines = await self._guideline_store.list_guidelines(
            tags=[Tag.for_agent_id(agent_id)],
        )
        global_guidelines = await self._guideline_store.list_guidelines(tags=[])

        agent = await self._agent_store.read_agent(agent_id)
        guidelines_for_agent_tags = await self._guideline_store.list_guidelines(
            tags=[tag for tag in agent.tags]
        )

        all_guidelines = set(chain(agent_guidelines, global_guidelines, guidelines_for_agent_tags))
        return list(all_guidelines)

    async def find_context_variables_for_agent(
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
                agent_context_variables, global_context_variables, context_variables_for_agent_tags
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

    async def find_guideline_connections(
        self,
        source: Optional[GuidelineId] = None,
        target: Optional[GuidelineId] = None,
        indirect: bool = False,
    ) -> Sequence[GuidelineConnection]:
        return await self._guideline_connection_store.list_connections(
            source=source,
            target=target,
            indirect=indirect,
        )

    async def find_guideline_tool_associations(
        self,
    ) -> Sequence[GuidelineToolAssociation]:
        return await self._guideline_tool_association_store.list_associations()

    async def find_relevant_glossary_terms(
        self,
        query: str,
        tags: Sequence[TagId],
    ) -> Sequence[Term]:
        return await self._glossary_store.find_relevant_terms(query, tags)

    async def read_tool_service(
        self,
        service_name: str,
    ) -> ToolService:
        return await self._service_registry.read_tool_service(service_name)


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
