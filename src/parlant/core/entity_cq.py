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

from collections import defaultdict
from itertools import chain
from typing import Mapping, Optional, Sequence, cast

from cachetools import TTLCache

import warnings

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
from parlant.core.engines.alpha.canned_response_source import (
    GLOBAL_CANNED_RESPONSE_SOURCE,
    CannedResponseLookup,
    CannedResponseSource,
    CannedResponseSourceKind,
)
from parlant.core.engines.alpha.tool_calling.tool_caller import ToolCallEvaluation, ToolInsights
from parlant.core.journey_rule_projection import (
    JourneyRuleProjection,
    extract_node_id_from_journey_node_rule_id,
)
from parlant.core.rules import (
    Rule,
    RuleId,
    RuleStore,
)
from parlant.core.journeys import Journey, JourneyId, JourneyNodeId, JourneyStore
from parlant.core.relationships import (
    RelationshipKind,
    RelationshipEntityKind,
    RelationshipStore,
)
from parlant.core.rule_tool_associations import (
    RuleToolAssociation,
    RuleToolAssociationStore,
)
from parlant.core.glossary import GlossaryStore, Term
from parlant.core.app_modules.sessions import SessionUpdateParamsModel
from parlant.core.sessions import (
    SessionId,
    Session,
    SessionStore,
    Event,
)
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.groups import GroupIds, GroupId
from parlant.core.tools import ToolId, ToolService
from parlant.core.canned_responses import CannedResponse, CannedResponseId, CannedResponseStore
from parlant.core.store_provider import StoreProvider, StoreProviderHints


class EntityQueries:
    def __init__(
        self,
        journey_rule_projection: JourneyRuleProjection,
        store_provider: StoreProvider,
    ) -> None:
        self._journey_rule_projection = journey_rule_projection
        self._store_provider = store_provider
        self.rule_and_journeys_it_depends_on = TTLCache[RuleId, list[Journey]](
            maxsize=1024, ttl=120
        )

    @property
    def _agent_store(self) -> AgentStore:
        return self._store_provider.get_store(AgentStore, StoreProviderHints(call_site="engine"))

    @property
    def _session_store(self) -> SessionStore:
        return self._store_provider.get_store(SessionStore, StoreProviderHints(call_site="engine"))

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
    def _journey_store(self) -> JourneyStore:
        return self._store_provider.get_store(JourneyStore, StoreProviderHints(call_site="engine"))

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

    @property
    def _capability_store(self) -> CapabilityStore:
        return self._store_provider.get_store(
            CapabilityStore, StoreProviderHints(call_site="engine")
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

    async def find_rules_for_context(
        self,
        agent_id: AgentId,
        journeys: Sequence[Journey],
    ) -> Sequence[Rule]:
        agent = await self._agent_store.read_agent(agent_id)

        async def _empty_rules() -> list[Rule]:
            return []

        projectable_journeys = []
        for journey in journeys:
            if not journey.triggers:
                continue
            if journey.node_properties is None:
                warnings.warn(
                    f"Skipping journey '{journey.title}' (id={journey.id}) for not having node_properties"
                )
                continue
            projectable_journeys.append(journey)

        (
            agent_rules,
            global_rules,
            rules_for_agent_groups,
            rules_for_journeys,
        ) = await async_utils.safe_gather(
            self._rule_store.list_rules(groups=[GroupIds.for_agent_id(agent_id)]),
            self._rule_store.list_rules(groups=[]),
            (
                self._rule_store.list_rules(groups=list(agent.groups))
                if agent.groups
                else _empty_rules()
            ),
            (
                self._rule_store.list_rules(
                    groups=[GroupIds.for_journey_id(journey.id) for journey in journeys]
                )
                if journeys
                else _empty_rules()
            ),
        )
        projection_tasks = [
            self._journey_rule_projection.project_journey_to_rules(journey.id)
            for journey in projectable_journeys
        ]
        projected_journey_rules = (
            await async_utils.safe_gather(*projection_tasks) if projection_tasks else []
        )

        all_rules = set(
            chain(
                agent_rules,
                global_rules,
                rules_for_agent_groups,
                rules_for_journeys,
                *projected_journey_rules,
            )
        )

        return list(all_rules)

    async def find_journey_related_rules(
        self,
        journey: Journey,
    ) -> Sequence[RuleId]:
        """Return rules that are dependent or derived on the specified journey."""
        iterated_relationships = set()

        rule_ids = set()

        relationships = set(
            await self._relationship_store.list_relationships(
                kind=RelationshipKind.DEPENDENCY,
                indirect=False,
                target_id=GroupIds.for_journey_id(journey.id),
            )
        )

        while relationships:
            r = relationships.pop()

            if r in iterated_relationships:
                continue

            if r.source.kind == RelationshipEntityKind.RULE:
                rule_ids.add(cast(RuleId, r.source.id))

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

        for id in rule_ids:
            journeys = self.rule_and_journeys_it_depends_on.get(id, [])
            journeys.append(journey)

            self.rule_and_journeys_it_depends_on[id] = journeys

        rule_ids.update(
            g.id for g in await self._journey_rule_projection.project_journey_to_rules(journey.id)
        )

        return list(rule_ids)

    async def find_context_variables_for_context(
        self,
        agent_id: AgentId,
    ) -> Sequence[ContextVariable]:
        agent = await self._agent_store.read_agent(agent_id)

        async def _empty_variables() -> list[ContextVariable]:
            return []

        results = await async_utils.safe_gather(
            self._context_variable_store.list_variables(groups=[GroupIds.for_agent_id(agent_id)]),
            self._context_variable_store.list_variables(groups=[]),
            (
                self._context_variable_store.list_variables(groups=list(agent.groups))
                if agent.groups
                else _empty_variables()
            ),
        )

        all_context_variables = set(chain(*results))
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
        agent = await self._agent_store.read_agent(agent_id)

        async def _empty_capabilities() -> list[Capability]:
            return []

        results = await async_utils.safe_gather(
            self._capability_store.list_capabilities(groups=[GroupIds.for_agent_id(agent_id)]),
            self._capability_store.list_capabilities(groups=[]),
            (
                self._capability_store.list_capabilities(groups=list(agent.groups))
                if agent.groups
                else _empty_capabilities()
            ),
        )

        all_capabilities = set(chain(*results))

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
        max_terms: int = 20,
    ) -> Sequence[Term]:
        agent = await self._agent_store.read_agent(agent_id)

        async def _empty_terms() -> list[Term]:
            return []

        results = await async_utils.safe_gather(
            self._glossary_store.list_terms(groups=[GroupIds.for_agent_id(agent_id)]),
            self._glossary_store.list_terms(groups=[]),
            (
                self._glossary_store.list_terms(groups=list(agent.groups))
                if agent.groups
                else _empty_terms()
            ),
        )

        all_terms = set(chain(*results))

        return await self._glossary_store.find_relevant_terms(
            query, list(all_terms), max_terms=max_terms
        )

    async def list_glossary_terms_for_context(
        self,
        agent_id: AgentId,
    ) -> Sequence[Term]:
        agent_terms = await self._glossary_store.list_terms(
            groups=[GroupIds.for_agent_id(agent_id)],
        )
        global_terms = await self._glossary_store.list_terms(groups=[])
        agent = await self._agent_store.read_agent(agent_id)
        glossary_for_agent_groups = await self._glossary_store.list_terms(
            groups=[group for group in agent.groups]
        )

        return list(set(chain(agent_terms, global_terms, glossary_for_agent_groups)))

    async def read_tool_service(
        self,
        service_name: str,
    ) -> ToolService:
        return await self._service_registry.read_tool_service(service_name)

    async def finds_journeys_for_context(
        self,
        agent_id: AgentId,
    ) -> Sequence[Journey]:
        agent = await self._agent_store.read_agent(agent_id)

        async def _empty_journeys() -> list[Journey]:
            return []

        results = await async_utils.safe_gather(
            self._journey_store.list_journeys(groups=[GroupIds.for_agent_id(agent_id)]),
            self._journey_store.list_journeys(groups=[]),
            (
                self._journey_store.list_journeys(groups=list(agent.groups))
                if agent.groups
                else _empty_journeys()
            ),
        )

        return list(set(chain(*results)))

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
        rules: Sequence[Rule],
    ) -> CannedResponseLookup:
        agent_canreps = await self._canned_response_store.list_canned_responses(
            groups=[GroupIds.for_agent_id(agent.id)],
        )
        global_canreps = await self._canned_response_store.list_canned_responses(groups=[])

        canreps_for_agent_groups = await self._canned_response_store.list_canned_responses(
            groups=[group for group in agent.groups]
        )

        journey_canreps = await self._canned_response_store.list_canned_responses(
            groups=[GroupIds.for_journey_id(journey.id) for journey in journeys]
        )

        rule_canreps = await self.find_canned_responses_for_rules(rules)

        sources: dict[CannedResponseId, set[CannedResponseSource]] = defaultdict(set)

        agent_source = CannedResponseSource(kind=CannedResponseSourceKind.AGENT, id=agent.id)
        for c in agent_canreps:
            sources[c.id].add(agent_source)

        for c in global_canreps:
            sources[c.id].add(GLOBAL_CANNED_RESPONSE_SOURCE)

        agent_group_set = set(agent.groups)
        for c in canreps_for_agent_groups:
            for t in c.groups:
                if t in agent_group_set:
                    sources[c.id].add(
                        CannedResponseSource(kind=CannedResponseSourceKind.AGENT_TAG, id=t)
                    )

        journey_group_to_journey_id = {GroupIds.for_journey_id(j.id): j.id for j in journeys}
        for c in journey_canreps:
            for t in c.groups:
                if t in journey_group_to_journey_id:
                    sources[c.id].add(
                        CannedResponseSource(
                            kind=CannedResponseSourceKind.JOURNEY,
                            id=journey_group_to_journey_id[t],
                        )
                    )

        rule_group_to_source: dict[GroupId, CannedResponseSource] = {}
        for g in rules:
            if g.id.startswith("journey_node:"):
                node_id = extract_node_id_from_journey_node_rule_id(g.id)
                rule_group_to_source[GroupIds.for_journey_node_id(node_id)] = CannedResponseSource(
                    kind=CannedResponseSourceKind.JOURNEY_NODE,
                    id=node_id,
                )
            else:
                rule_group_to_source[GroupIds.for_rule_id(g.id)] = CannedResponseSource(
                    kind=CannedResponseSourceKind.GUIDELINE,
                    id=g.id,
                )
        for c in rule_canreps:
            for t in c.groups:
                if t in rule_group_to_source:
                    sources[c.id].add(rule_group_to_source[t])

        all_canreps = set(
            chain(
                agent_canreps,
                global_canreps,
                canreps_for_agent_groups,
                journey_canreps,
                rule_canreps,
            )
        )

        return CannedResponseLookup(
            canned_responses=list(all_canreps),
            sources={cid: list(s) for cid, s in sources.items()},
        )

    async def find_canned_responses_for_rules(
        self,
        rules: Sequence[Rule],
    ) -> Sequence[CannedResponse]:
        groups = []

        for g in rules:
            if g.id.startswith("journey_node:"):
                groups.append(
                    GroupIds.for_journey_node_id(extract_node_id_from_journey_node_rule_id(g.id))
                )

            else:
                groups.append(GroupIds.for_rule_id(g.id))

        return await self._canned_response_store.list_canned_responses(groups=groups)

    async def find_rules_that_need_reevaluation(
        self,
        available_rules: dict[RuleId, Rule],
        active_journeys: Sequence[Journey],
        tool_insights: ToolInsights,
    ) -> Sequence[Rule]:
        """Find rules that need reevaluation based on the tool calls made."""

        if not tool_insights.evaluations:
            return []

        executed_tool_ids = {
            tid
            for tid, e in tool_insights.evaluations.items()
            if any(value == ToolCallEvaluation.NEEDS_TO_RUN for value in e.values())
        }

        active_journeys_mapping = {journey.id: journey for journey in active_journeys}
        rules: list[Rule] = []

        tasks = [
            self._relationship_store.list_relationships(
                kind=RelationshipKind.REEVALUATION,
                indirect=False,
                target_id=tool_id,
            )
            for tool_id in tool_insights.evaluations
        ]

        reevaluation_relationships = list(
            chain.from_iterable(await async_utils.safe_gather(*tasks))
        )

        for relationship in reevaluation_relationships:
            matched_rules: list[Rule] = []

            # Check by rule ID prefix (existing behavior for RULE and
            # journey-node TAG sources).
            by_id = [
                g for gid, g in available_rules.items() if gid.startswith(relationship.source.id)
            ]
            matched_rules.extend(by_id)

            # For TAG sources that didn't match by ID prefix, check by group
            # membership so that custom groups can trigger reevaluation for all
            # rules that carry that group.
            if not by_id and relationship.source.kind.is_group:
                by_tag = [g for g in available_rules.values() if relationship.source.id in g.groups]
                matched_rules.extend(by_tag)

            for rule_to_reevaluate in matched_rules:
                the_id_of_the_tool_related_to_the_rule_to_reevaluate = relationship.target.id

                # At this point we know that one of the rules given to us
                # has a reevaluation relationship with one of the relevant tools.

                if rule_to_reevaluate.metadata.get("journey_node"):
                    # We found a journey node that has a reevaluation relationship with one of the tools.
                    #
                    # This journey node is by definition a tool node.
                    #
                    # Now, this actually means we need to reevaluate the entire journey,
                    # so we'll need to add all of its projected rules to the list.

                    # The only exception to this rule here is if the tool was deliberately skipped
                    # because the context already existed in the session.

                    # FIXME: Strictly speaking, we should only reevaluate the journey if the tool
                    # was called ON BEHALF OF THE JOURNEY NODE — since it could have been called
                    # for some other reason, e.g. due to an unrelated rule.

                    tc_evals_for_tool = tool_insights.evaluations.get(
                        cast(ToolId, the_id_of_the_tool_related_to_the_rule_to_reevaluate),
                        {},
                    )
                    tool_should_be_considered_as_having_been_called = all(
                        e
                        in [
                            ToolCallEvaluation.DATA_ALREADY_IN_CONTEXT,
                            ToolCallEvaluation.NEEDS_TO_RUN,
                        ]
                        for e in tc_evals_for_tool.values()
                    )

                    if tool_should_be_considered_as_having_been_called:
                        journey_id = cast(
                            JourneyId,
                            cast(
                                Mapping[str, JSONSerializable],
                                rule_to_reevaluate.metadata["journey_node"],
                            ).get("journey_id"),
                        )

                        if journey_id in active_journeys_mapping:
                            projected_journey_rules = (
                                await self._journey_rule_projection.project_journey_to_rules(
                                    journey_id
                                )
                            )

                            rules.extend(projected_journey_rules)
                else:
                    # For normal rules, we only reevaluate them if their related
                    # tool WAS JUST executed -- not if it was skipped.
                    if the_id_of_the_tool_related_to_the_rule_to_reevaluate in executed_tool_ids:
                        rules.append(rule_to_reevaluate)

        return list(set(rules))


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
