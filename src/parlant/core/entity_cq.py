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

from dataclasses import dataclass
from itertools import chain
from typing import Awaitable, Callable, Optional, Sequence, TypeVar, cast


from parlant.core import async_utils
from parlant.core.agents import Agent, AgentId, AgentStore
from parlant.core.common import JSONSerializable
from parlant.core.context_variables import (
    ContextVariable,
    ContextVariableId,
    ContextVariableStore,
    ContextVariableValue,
)
from parlant.core.customers import Customer, CustomerId, CustomerStore
from parlant.core.rules import (
    Rule,
    RuleStore,
)
from parlant.core.relationships import (
    RelationshipStore,
)
from parlant.core.rule_tool_associations import (
    RuleToolAssociation,
    RuleToolAssociationStore,
)
from parlant.core.glossary import GlossaryStore, Term
from parlant.core.loggers import Logger
from parlant.core.app_modules.sessions import SessionUpdateParamsModel
from parlant.core.sessions import (
    SessionId,
    Session,
    SessionStore,
    Event,
)
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.groups import GroupIds, GroupId
from parlant.core.tools import ToolService
from parlant.core.canned_responses import (
    CannedResponse,
    CannedResponseRelevanceResult,
    CannedResponseStore,
)
from parlant.core.store_provider import StoreProvider, StoreProviderHints


_TEntity = TypeVar("_TEntity")


@dataclass(frozen=True)
class TurnCannedResponses:
    """The canned responses in scope for a turn, split by precedence tier:
    ``rule_scoped`` are attached to rules that matched this turn; ``agent_wide``
    are the agent's own responses that are not tied to any matched rule."""

    rule_scoped: Sequence[CannedResponse]
    agent_wide: Sequence[CannedResponse]


def _take_unranked(
    turn: TurnCannedResponses,
    tool_provided: Sequence[CannedResponse],
    max_count: int,
    logger: Logger,
) -> Sequence[CannedResponseRelevanceResult]:
    """Every eligible response, in precedence order, capped at ``max_count``.

    Nothing is ranked, so the score carries no relevance signal: tool-provided
    responses keep their 1.0, and the rest are marked 0.0 to say "unscored"."""
    eligible = [
        *((response, 1.0) for response in tool_provided),
        *((response, 0.0) for response in turn.rule_scoped),
        *((response, 0.0) for response in turn.agent_wide),
    ]

    if (left_out := len(eligible) - max_count) > 0:
        logger.warning(
            f"Canned responses: {len(eligible)} are in scope for this turn but "
            f"only {max_count} fit the unranked cap, so {left_out} were left "
            "out. Raise Responder.max_unranked_canned_responses, or scope some "
            "of these responses to rules so they only compete when relevant."
        )

    return [
        CannedResponseRelevanceResult(canned_response=response, score=score)
        for response, score in eligible[:max_count]
    ]


async def select_turn_canned_responses(
    store: CannedResponseStore,
    query: str,
    turn: TurnCannedResponses,
    max_count: int,
    logger: Logger,
    tool_provided: Sequence[CannedResponse] = (),
    rank: bool = True,
) -> Sequence[CannedResponseRelevanceResult]:
    """Pick up to ``max_count`` canned responses for a turn, in precedence
    order: ``tool_provided`` responses first (literal texts a tool or
    retriever returned for THIS turn — already maximally relevant, so they
    skip relevance ranking and fill the budget outright), then rule-scoped
    responses (ranked by relevance to ``query``), then agent-wide responses
    for whatever slots remain.

    With ``rank=False`` nothing is ranked and ``query`` goes unused: every
    eligible response is taken in the same precedence order, up to
    ``max_count``. That path issues no store query and embeds nothing, which
    is what low-effort turns want."""
    if max_count <= 0:
        return []

    if not rank:
        return _take_unranked(turn, tool_provided, max_count, logger)

    tool_hits: list[CannedResponseRelevanceResult] = [
        CannedResponseRelevanceResult(canned_response=response, score=1.0)
        for response in tool_provided[:max_count]
    ]

    remaining_for_rules = max_count - len(tool_hits)

    rule_hits: Sequence[CannedResponseRelevanceResult] = (
        await store.filter_relevant_canned_responses(
            query=query,
            available_canned_responses=turn.rule_scoped,
            max_count=remaining_for_rules,
        )
        if remaining_for_rules > 0 and turn.rule_scoped
        else []
    )

    remaining = max_count - len(tool_hits) - len(rule_hits)

    agent_hits: Sequence[CannedResponseRelevanceResult] = (
        await store.filter_relevant_canned_responses(
            query=query,
            available_canned_responses=turn.agent_wide,
            max_count=remaining,
        )
        if remaining > 0 and turn.agent_wide
        else []
    )

    return [*tool_hits, *rule_hits, *agent_hits]


class EntityQueries:
    def __init__(
        self,
        store_provider: StoreProvider,
    ) -> None:
        self._store_provider = store_provider

    @property
    def _agent_store(self) -> AgentStore:
        return self._store_provider.get_store(AgentStore, StoreProviderHints(call_site="engine"))

    @property
    def _session_store(self) -> SessionStore:
        return self._store_provider.get_store(SessionStore, StoreProviderHints(call_site="engine"))

    @property
    def session_store(self) -> SessionStore:
        """Return the authoritative session store used by engine queries."""

        return self._session_store

    @property
    def _rule_store(self) -> RuleStore:
        return self._store_provider.get_store(RuleStore, StoreProviderHints(call_site="engine"))

    @property
    def _customer_store(self) -> CustomerStore:
        return self._store_provider.get_store(CustomerStore, StoreProviderHints(call_site="engine"))

    @property
    def _context_variable_store(self) -> ContextVariableStore:
        return self._store_provider.get_store(
            ContextVariableStore, StoreProviderHints(call_site="engine")
        )

    @property
    def _relationship_store(self) -> RelationshipStore:
        return self._store_provider.get_store(
            RelationshipStore, StoreProviderHints(call_site="engine")
        )

    @property
    def _rule_tool_association_store(self) -> RuleToolAssociationStore:
        return self._store_provider.get_store(
            RuleToolAssociationStore, StoreProviderHints(call_site="engine")
        )

    @property
    def _glossary_store(self) -> GlossaryStore:
        return self._store_provider.get_store(GlossaryStore, StoreProviderHints(call_site="engine"))

    @property
    def _service_registry(self) -> ServiceRegistry:
        return self._store_provider.get_store(
            ServiceRegistry, StoreProviderHints(call_site="engine")
        )

    @property
    def _canned_response_store(self) -> CannedResponseStore:
        return self._store_provider.get_store(
            CannedResponseStore, StoreProviderHints(call_site="engine")
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

    async def _list_for_agent_groups(
        self,
        groups: Sequence[GroupId],
        list_by_groups: Callable[[Sequence[GroupId]], Awaitable[Sequence[_TEntity]]],
    ) -> Sequence[_TEntity]:
        """The group-scoped read, skipped when the agent belongs to no group.

        An empty group list means *global* to every store, and each caller has
        already read the global entities alongside this one — so asking would
        only repeat that query and union the same rows back in.
        """
        if not groups:
            return []

        return await list_by_groups(list(groups))

    async def find_rules_for_context(
        self,
        agent_id: AgentId,
    ) -> Sequence[Rule]:
        agent_rules, global_rules, agent = await async_utils.safe_gather(
            self._rule_store.list_rules(groups=[GroupIds.for_agent_id(agent_id)]),
            self._rule_store.list_rules(groups=[]),
            self._agent_store.read_agent(agent_id),
        )

        rules_for_agent_groups = await self._list_for_agent_groups(
            agent.groups,
            lambda groups: self._rule_store.list_rules(groups=groups),
        )

        return list(set(chain(agent_rules, global_rules, rules_for_agent_groups)))

    async def find_context_variables_for_context(
        self,
        agent_id: AgentId,
    ) -> Sequence[ContextVariable]:
        (
            agent_context_variables,
            global_context_variables,
            agent,
        ) = await async_utils.safe_gather(
            self._context_variable_store.list_variables(
                groups=[GroupIds.for_agent_id(agent_id)],
            ),
            self._context_variable_store.list_variables(groups=[]),
            self._agent_store.read_agent(agent_id),
        )

        context_variables_for_agent_groups = await self._list_for_agent_groups(
            agent.groups,
            lambda groups: self._context_variable_store.list_variables(groups=groups),
        )

        all_context_variables = set(
            chain(
                agent_context_variables,
                global_context_variables,
                context_variables_for_agent_groups,
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

    async def find_rule_tool_associations(
        self,
    ) -> Sequence[RuleToolAssociation]:
        return await self._rule_tool_association_store.list_associations()

    async def find_glossary_terms_for_context(
        self,
        agent_id: AgentId,
        query: str,
        max_terms: int = 20,
    ) -> Sequence[Term]:
        agent_terms, global_terms, agent = await async_utils.safe_gather(
            self._glossary_store.list_terms(groups=[GroupIds.for_agent_id(agent_id)]),
            self._glossary_store.list_terms(groups=[]),
            self._agent_store.read_agent(agent_id),
        )

        glossary_for_agent_groups = await self._list_for_agent_groups(
            agent.groups,
            lambda groups: self._glossary_store.list_terms(groups=groups),
        )

        all_terms = set(chain(agent_terms, global_terms, glossary_for_agent_groups))

        return await self._glossary_store.find_relevant_terms(
            query, list(all_terms), max_terms=max_terms
        )

    async def list_glossary_terms_for_context(
        self,
        agent_id: AgentId,
    ) -> Sequence[Term]:
        agent_terms, global_terms, agent = await async_utils.safe_gather(
            self._glossary_store.list_terms(groups=[GroupIds.for_agent_id(agent_id)]),
            self._glossary_store.list_terms(groups=[]),
            self._agent_store.read_agent(agent_id),
        )

        glossary_for_agent_groups = await self._list_for_agent_groups(
            agent.groups,
            lambda groups: self._glossary_store.list_terms(groups=groups),
        )

        return list(set(chain(agent_terms, global_terms, glossary_for_agent_groups)))

    async def read_tool_service(
        self,
        service_name: str,
    ) -> ToolService:
        return await self._service_registry.read_tool_service(service_name)

    def _rule_group_ids(self, rules: Sequence[Rule]) -> list[GroupId]:
        groups: list[GroupId] = []

        for g in rules:
            groups.append(GroupIds.for_rule_id(g.id))

        return groups

    async def find_canned_responses_for_turn(
        self,
        rules: Sequence[Rule],
        agent_id: AgentId,
    ) -> TurnCannedResponses:
        """Canned responses in scope for a Compass turn, split by precedence tier:
        those attached to the turn's matched rules, and the agent's own agent-wide
        responses that aren't already attached to a matched rule.

        A response attached to a matched rule counts as rule-scoped even when it is
        also filed under the agent (deduplicated toward the higher-precedence tier).
        """
        rule_groups = self._rule_group_ids(rules)

        # An empty group list would match *global* (ungrouped) responses, which are
        # not in scope here — so only query when the turn actually matched rules.
        rule_scoped, agent_scoped = await async_utils.safe_gather(
            self._canned_response_store.list_canned_responses(groups=rule_groups)
            if rule_groups
            else async_utils.completed_task(cast(Sequence[CannedResponse], [])),
            self._canned_response_store.list_canned_responses(
                groups=[GroupIds.for_agent_id(agent_id)]
            ),
        )

        rule_scoped_ids = {c.id for c in rule_scoped}
        agent_wide = [c for c in agent_scoped if c.id not in rule_scoped_ids]

        return TurnCannedResponses(rule_scoped=rule_scoped, agent_wide=agent_wide)


class EntityCommands:
    def __init__(
        self,
        store_provider: StoreProvider,
    ) -> None:
        self._store_provider = store_provider

    @property
    def _session_store(self) -> SessionStore:
        return self._store_provider.get_store(SessionStore, StoreProviderHints(call_site="engine"))

    @property
    def _context_variable_store(self) -> ContextVariableStore:
        return self._store_provider.get_store(
            ContextVariableStore, StoreProviderHints(call_site="engine")
        )

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

    async def upsert_session_labels(
        self,
        session_id: SessionId,
        labels: set[str],
    ) -> Session:
        """Upserts labels to a session."""
        return await self._session_store.upsert_labels(session_id, labels)
