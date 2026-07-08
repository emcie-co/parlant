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

import asyncio
import traceback
from dataclasses import replace
from typing import Optional, Sequence, cast

from parlant.core import async_utils
from parlant.core.agents import AgentStore
from parlant.core.background_tasks import BackgroundTaskService
from parlant.core.common import JSONSerializable, xxh3_checksum
from parlant.core.evaluations import (
    Evaluation,
    EvaluationStatus,
    EvaluationId,
    EvaluationUpdateParams,
    RulePayload,
    InvoiceData,
    InvoiceJourneyData,
    JourneyPayload,
    Invoice,
    InvoiceRuleData,
    EvaluationStore,
    PayloadDescriptor,
    PayloadKind,
    PayloadOperation,
)
from parlant.core.rules import (
    Rule,
    RuleContent,
    RuleStore,
    compose_rule_query,
)
from parlant.core.journey_rule_projection import (
    JourneyRuleProjection,
    extract_link_id_from_journey_node_rule_id,
    extract_node_id_from_journey_node_rule_id,
)
from parlant.core.journeys import Journey, JourneyId, JourneyNodeId, JourneyStore
from parlant.core.services.indexing.common import EvaluationError, ProgressReport
from parlant.core.services.indexing.customer_dependent_action_detector import (
    CustomerDependentActionDetector,
    CustomerDependentActionProposition,
)
from parlant.core.services.indexing.rule_action_proposer import (
    RuleActionProposer,
    RuleActionProposition,
)
from parlant.core.services.indexing.rule_agent_intention_proposer import (
    AgentIntentionProposer,
    AgentIntentionProposition,
)
from parlant.core.services.indexing.rule_continuous_proposer import (
    RuleContinuousProposer,
    RuleContinuousProposition,
)
from parlant.core.services.indexing.rule_signal_proposer import (
    RuleSignalProposer,
    RuleSignalProposition,
)
from parlant.core.services.indexing.rule_title_proposer import (
    RuleTitleProposer,
    RuleTitleProposition,
)
from parlant.core.loggers import Logger
from parlant.core.entity_cq import EntityQueries
from parlant.core.services.indexing.journey_reachable_nodes_evaluation import (
    JourneyReachableNodesEvaluator,
    ReachableNodesEvaluation,
)
from parlant.core.services.indexing.relative_action_proposer import (
    RelativeActionProposer,
    RelativeActionProposition,
)
from parlant.core.store_provider import StoreProvider, StoreProviderHints
from parlant.core.services.indexing.tool_running_action_detector import (
    ToolRunningActionDetector,
    ToolRunningActionProposition,
)


class EvaluationValidationError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class RuleEvaluator:
    # Glossary terms fetched per evaluated rule (top-k by the rule's own
    # query) - the terms the rule depends on to be interpreted correctly.
    _TERMS_PER_RULE = 10

    def __init__(
        self,
        logger: Logger,
        entity_queries: EntityQueries,
        rule_action_proposer: RuleActionProposer,
        rule_continuous_proposer: RuleContinuousProposer,
        customer_dependent_action_detector: CustomerDependentActionDetector,
        agent_intention_proposer: AgentIntentionProposer,
        tool_running_action_detector: ToolRunningActionDetector,
        rule_signal_proposer: RuleSignalProposer,
        rule_title_proposer: RuleTitleProposer,
    ) -> None:
        self._logger = logger
        self._entity_queries = entity_queries
        self._rule_action_proposer = rule_action_proposer
        self._rule_continuous_proposer = rule_continuous_proposer
        self._customer_dependent_action_detector = customer_dependent_action_detector
        self._agent_intention_proposer = agent_intention_proposer
        self._tool_running_action_detector = tool_running_action_detector
        self._rule_signal_proposer = rule_signal_proposer
        self._rule_title_proposer = rule_title_proposer

    def _build_invoice_data(
        self,
        action_propositions: Sequence[Optional[RuleActionProposition]],
        continuous_propositions: Sequence[Optional[RuleContinuousProposition]],
        customer_dependant_action_detections: Sequence[
            Optional[CustomerDependentActionProposition]
        ],
        agent_intention_propositions: Sequence[Optional[AgentIntentionProposition]],
        tool_running_action_propositions: Sequence[Optional[ToolRunningActionProposition]],
        signal_propositions: Sequence[Optional[RuleSignalProposition]],
        title_propositions: Sequence[Optional[RuleTitleProposition]],
    ) -> Sequence[InvoiceRuleData]:
        results = []
        for (
            payload_action,
            payload_continuous,
            payload_customer_dependent,
            agent_intention,
            tool_running_action,
            signal_proposition,
            title_proposition,
        ) in zip(
            action_propositions,
            continuous_propositions,
            customer_dependant_action_detections,
            agent_intention_propositions,
            tool_running_action_propositions,
            signal_propositions,
            title_propositions,
        ):
            properties_prop: dict[str, JSONSerializable] = {
                **{
                    "continuous": payload_continuous.is_continuous if payload_continuous else None,
                    "customer_dependent_action_data": payload_customer_dependent.model_dump()
                    if payload_customer_dependent
                    else None,
                    "agent_intention_condition": agent_intention.rewritten_condition
                    if agent_intention
                    and agent_intention.rewritten_condition
                    and agent_intention.is_agent_intention
                    else None,
                    "internal_action": payload_action.content.action if payload_action else None,
                },
                **(
                    {"tool_running_only": tool_running_action.is_tool_running_only}
                    if tool_running_action
                    else {}
                ),
            }

            invoice_data = InvoiceRuleData(
                properties_proposition=properties_prop,
                signals_proposition=signal_proposition.signals if signal_proposition else None,
                anti_signals_proposition=signal_proposition.anti_signals
                if signal_proposition
                else None,
                title_proposition=title_proposition.title if title_proposition else None,
            )

            results.append(invoice_data)

        return results

    async def evaluate(
        self,
        payloads: Sequence[RulePayload],
        progress_report: Optional[ProgressReport] = None,
    ) -> Sequence[InvoiceRuleData]:
        action_propositions = await self._propose_actions(
            payloads,
            progress_report,
        )

        continuous_propositions = await self._propose_continuous(
            payloads,
            action_propositions,
            progress_report,
        )

        customer_dependant_action_detections = await self._detect_customer_dependant_actions(
            payloads, action_propositions, progress_report
        )

        agent_intention_propositions = await self._propose_agent_intention(
            payloads, progress_report
        )

        tool_running_action_propositions = await self._detect_tool_running_actions(
            payloads, progress_report
        )

        signal_propositions = await self._propose_signals(
            payloads,
            action_propositions,
            progress_report,
        )

        title_propositions = await self._propose_titles(
            payloads,
            action_propositions,
            progress_report,
        )

        return self._build_invoice_data(
            action_propositions,
            continuous_propositions,
            customer_dependant_action_detections,
            agent_intention_propositions,
            tool_running_action_propositions,
            signal_propositions,
            title_propositions,
        )

    async def _propose_actions(
        self,
        payloads: Sequence[RulePayload],
        progress_report: Optional[ProgressReport] = None,
    ) -> Sequence[Optional[RuleActionProposition]]:
        tasks: list[asyncio.Task[Optional[RuleActionProposition]]] = []
        indices: list[int] = []

        for i, p in enumerate(payloads):
            if p.action_proposition:
                indices.append(i)
                tasks.append(
                    asyncio.create_task(
                        self._rule_action_proposer.propose_action(
                            rule=p.content,
                            tool_ids=p.tool_ids or [],
                            progress_report=progress_report,
                        )
                    )
                )

        sparse_results = await async_utils.safe_gather(*tasks)
        results: list[Optional[RuleActionProposition]] = [None] * len(payloads)
        for i, res in zip(indices, sparse_results):
            results[i] = res

        return results

    async def _detect_customer_dependant_actions(
        self,
        payloads: Sequence[RulePayload],
        proposed_actions: Sequence[Optional[RuleActionProposition]],
        progress_report: Optional[ProgressReport] = None,
    ) -> Sequence[Optional[CustomerDependentActionProposition]]:
        tasks: list[asyncio.Task[CustomerDependentActionProposition]] = []
        indices: list[int] = []
        for i, (p, action_prop) in enumerate(zip(payloads, proposed_actions)):
            if not p.properties_proposition and not p.journey_node_proposition:
                continue
            action_to_use = (
                action_prop.content.action if action_prop is not None else p.content.action
            )
            rule_content = RuleContent(
                condition=p.content.condition,
                action=action_to_use,
            )
            indices.append(i)
            tasks.append(
                asyncio.create_task(
                    self._customer_dependent_action_detector.detect_if_customer_dependent(
                        rule=rule_content,
                        progress_report=progress_report,
                    )
                )
            )
        sparse_results = await async_utils.safe_gather(*tasks)
        results: list[Optional[CustomerDependentActionProposition]] = [None] * len(payloads)
        for i, res in zip(indices, sparse_results):
            results[i] = res
        return results

    async def _propose_continuous(
        self,
        payloads: Sequence[RulePayload],
        proposed_actions: Sequence[Optional[RuleActionProposition]],
        progress_report: Optional[ProgressReport] = None,
    ) -> Sequence[Optional[RuleContinuousProposition]]:
        tasks: list[asyncio.Task[RuleContinuousProposition]] = []
        indices: list[int] = []

        for i, (p, action_prop) in enumerate(zip(payloads, proposed_actions)):
            if not p.properties_proposition:
                continue

            action_to_use = (
                action_prop.content.action if action_prop is not None else p.content.action
            )
            rule_content = RuleContent(
                condition=p.content.condition,
                action=action_to_use,
            )

            indices.append(i)
            tasks.append(
                asyncio.create_task(
                    self._rule_continuous_proposer.propose_continuous(
                        rule=rule_content,
                        progress_report=progress_report,
                    )
                )
            )

        sparse_results = await async_utils.safe_gather(*tasks)
        results: list[Optional[RuleContinuousProposition]] = [None] * len(payloads)
        for i, res in zip(indices, sparse_results):
            results[i] = res
        return results

    async def _propose_agent_intention(
        self,
        payloads: Sequence[RulePayload],
        progress_report: Optional[ProgressReport] = None,
    ) -> Sequence[Optional[AgentIntentionProposition]]:
        tasks: list[asyncio.Task[AgentIntentionProposition]] = []
        indices: list[int] = []

        for i, p in enumerate(payloads):
            if not p.properties_proposition:
                continue

            rule_content = RuleContent(
                condition=p.content.condition,
                action=p.content.action,
            )

            indices.append(i)
            tasks.append(
                asyncio.create_task(
                    self._agent_intention_proposer.propose_agent_intention(
                        rule=rule_content,
                        progress_report=progress_report,
                    )
                )
            )

        sparse_results = await async_utils.safe_gather(*tasks)
        results: list[Optional[AgentIntentionProposition]] = [None] * len(payloads)
        for i, res in zip(indices, sparse_results):
            results[i] = res
        return results

    async def _detect_tool_running_actions(
        self,
        payloads: Sequence[RulePayload],
        progress_report: Optional[ProgressReport] = None,
    ) -> Sequence[Optional[ToolRunningActionProposition]]:
        tasks: list[asyncio.Task[ToolRunningActionProposition]] = []
        indices: list[int] = []

        for i, p in enumerate(payloads):
            if not p.journey_node_proposition:
                continue

            tasks.append(
                asyncio.create_task(
                    self._tool_running_action_detector.detect_if_tool_running(
                        rule=p.content,
                        tool_ids=p.tool_ids,
                        progress_report=progress_report,
                    )
                )
            )
            indices.append(i)

        sparse_results = await async_utils.safe_gather(*tasks)
        results: list[Optional[ToolRunningActionProposition]] = [None] * len(payloads)

        for i, res in zip(indices, sparse_results):
            results[i] = res

        return results

    async def _propose_signals(
        self,
        payloads: Sequence[RulePayload],
        proposed_actions: Sequence[Optional[RuleActionProposition]],
        progress_report: Optional[ProgressReport] = None,
    ) -> Sequence[Optional[RuleSignalProposition]]:
        tasks: list[asyncio.Task[RuleSignalProposition]] = []
        indices: list[int] = []

        for i, (p, action_prop) in enumerate(zip(payloads, proposed_actions)):
            if not p.signal_proposition:
                continue

            action_to_use = (
                action_prop.content.action if action_prop is not None else p.content.action
            )
            rule_content = RuleContent(
                condition=p.content.condition,
                action=action_to_use,
                description=p.content.description,
            )

            agent = await self._entity_queries.read_agent(p.agent_id) if p.agent_id else None
            # The terms this rule depends on for correct interpretation -
            # ranked by the rule's own query, not the agent's whole glossary.
            glossary_terms = (
                list(
                    await self._entity_queries.find_glossary_terms_for_context(
                        p.agent_id,
                        query=compose_rule_query(
                            title=p.title,
                            condition=rule_content.condition,
                            action=rule_content.action,
                            description=rule_content.description,
                        ),
                        max_terms=self._TERMS_PER_RULE,
                    )
                )
                if p.agent_id
                else []
            )

            tasks.append(
                asyncio.create_task(
                    self._rule_signal_proposer.propose_signals(
                        rule=rule_content,
                        title=p.title or "Untitled Rule",
                        agent=agent,
                        glossary_terms=glossary_terms,
                        progress_report=progress_report,
                    )
                )
            )
            indices.append(i)

        sparse_results = await async_utils.safe_gather(*tasks)
        results: list[Optional[RuleSignalProposition]] = [None] * len(payloads)

        for i, res in zip(indices, sparse_results):
            results[i] = res

        return results

    async def _propose_titles(
        self,
        payloads: Sequence[RulePayload],
        proposed_actions: Sequence[Optional[RuleActionProposition]],
        progress_report: Optional[ProgressReport] = None,
    ) -> Sequence[Optional[RuleTitleProposition]]:
        tasks: list[asyncio.Task[RuleTitleProposition]] = []
        indices: list[int] = []

        for i, (p, action_prop) in enumerate(zip(payloads, proposed_actions)):
            if not p.title_proposition:
                continue

            action_to_use = (
                action_prop.content.action if action_prop is not None else p.content.action
            )
            rule_content = RuleContent(
                condition=p.content.condition,
                action=action_to_use,
                description=p.content.description,
            )

            agent = await self._entity_queries.read_agent(p.agent_id) if p.agent_id else None
            # The terms this rule depends on for correct interpretation -
            # ranked by the rule's own query, not the agent's whole glossary.
            glossary_terms = (
                list(
                    await self._entity_queries.find_glossary_terms_for_context(
                        p.agent_id,
                        query=compose_rule_query(
                            title=p.title,
                            condition=rule_content.condition,
                            action=rule_content.action,
                            description=rule_content.description,
                        ),
                        max_terms=self._TERMS_PER_RULE,
                    )
                )
                if p.agent_id
                else []
            )

            tasks.append(
                asyncio.create_task(
                    self._rule_title_proposer.propose_title(
                        rule=rule_content,
                        agent=agent,
                        glossary_terms=glossary_terms,
                        progress_report=progress_report,
                    )
                )
            )
            indices.append(i)

        sparse_results = await async_utils.safe_gather(*tasks)
        results: list[Optional[RuleTitleProposition]] = [None] * len(payloads)

        for i, res in zip(indices, sparse_results):
            results[i] = res

        return results


class JourneyEvaluator:
    def __init__(
        self,
        logger: Logger,
        journey_rule_projection: JourneyRuleProjection,
        rule_evaluator: RuleEvaluator,
        relative_action_proposer: RelativeActionProposer,
        journey_reachable_node_evaluator: JourneyReachableNodesEvaluator,
        store_provider: StoreProvider,
    ) -> None:
        self._logger = logger
        self._store_provider = store_provider
        self._journey_rule_projection = journey_rule_projection
        self._rule_evaluator = rule_evaluator
        self._journey_reachable_node_evaluator = journey_reachable_node_evaluator

        self._relative_action_proposer = relative_action_proposer

    @property
    def _rule_store(self) -> RuleStore:
        return self._store_provider.get_store(RuleStore, StoreProviderHints(call_site="engine"))

    @property
    def _journey_store(self) -> JourneyStore:
        return self._store_provider.get_store(JourneyStore, StoreProviderHints(call_site="engine"))

    async def _build_invoice_data(
        self,
        relative_action_propositions: Sequence[RelativeActionProposition],
        reachable_nodes_evaluations: Sequence[ReachableNodesEvaluation],
        journey_projections: dict[JourneyId, tuple[Journey, Sequence[Rule], tuple[Rule]]],
    ) -> Sequence[InvoiceJourneyData]:
        # Build index-to-key mappings. Use node_id:link_id as key to distinguish
        # the same sub-journey node linked multiple times from the same parent.
        def _evaluation_key(g: Rule) -> JourneyNodeId:
            node_id = extract_node_id_from_journey_node_rule_id(g.id)
            link_id = extract_link_id_from_journey_node_rule_id(g.id)
            if link_id:
                return JourneyNodeId(f"{node_id}:{link_id}")
            return node_id

        index_to_eval_keys: dict[JourneyId, dict[str, JourneyNodeId]] = {
            journey_id: {
                str(
                    cast(dict[str, JSONSerializable], g.metadata["journey_node"])["index"]
                ): _evaluation_key(g)
                for g in journey_projections[journey_id][1]
            }
            for journey_id in journey_projections
        }

        result = []

        for action_proposition, reachable_node_evaluation, journey_id in zip(
            relative_action_propositions, reachable_nodes_evaluations, journey_projections.keys()
        ):
            node_properties_proposition: dict[JourneyNodeId, dict[str, JSONSerializable]] = {}

            # Add rule evaluation properties for each node
            _, step_rules, __ = journey_projections[journey_id]
            for rule in step_rules:
                eval_key = _evaluation_key(rule)

                if eval_key not in node_properties_proposition:
                    node_properties_proposition[eval_key] = {}

                # Extract rule evaluation metadata
                for key, value in cast(
                    dict[str, JSONSerializable], rule.metadata.get("rule_evaluation", {})
                ).items():
                    node_properties_proposition[eval_key][key] = value

            for a in action_proposition.actions:
                eval_key = index_to_eval_keys[journey_id][a.index]
                if eval_key not in node_properties_proposition:
                    node_properties_proposition[eval_key] = {}
                node_properties_proposition[eval_key]["internal_action"] = a.rewritten_actions

            for index, r in reachable_node_evaluation.node_to_reachable_follow_ups.items():
                eval_key = index_to_eval_keys[journey_id][index]
                if eval_key not in node_properties_proposition:
                    node_properties_proposition[eval_key] = {}
                if "journey_node" not in node_properties_proposition[eval_key]:
                    node_properties_proposition[eval_key]["journey_node"] = {}
                node_properties_proposition[eval_key]["journey_node"] = {
                    **cast(
                        dict[str, JSONSerializable],
                        node_properties_proposition[eval_key]["journey_node"],
                    ),
                    "reachable_follow_ups": [{"condition": c, "path": p} for c, p in r],
                }

            invoice_data = InvoiceJourneyData(
                node_properties_proposition=node_properties_proposition,
                edge_properties_proposition={},
            )

            result.append(invoice_data)

        return result

    async def evaluate(
        self,
        payloads: Sequence[JourneyPayload],
        progress_report: Optional[ProgressReport] = None,
    ) -> Sequence[InvoiceJourneyData]:
        journeys: dict[JourneyId, Journey] = {
            j.id: j
            for j in await async_utils.safe_gather(
                *[
                    self._journey_store.read_journey(journey_id=payload.journey_id)
                    for payload in payloads
                ]
            )
        }

        journey_triggers = [
            await async_utils.safe_gather(
                *[self._rule_store.read_rule(rule_id=trigger) for trigger in journey.triggers]
            )
            for journey in journeys.values()
        ]

        journey_projections = {
            payload.journey_id: (journeys[payload.journey_id], projection, triggers)
            for payload, projection, triggers in zip(
                payloads,
                await async_utils.safe_gather(
                    *[
                        self._journey_rule_projection.project_journey_to_rules(
                            journey_id=payload.journey_id
                        )
                        for payload in payloads
                    ]
                ),
                journey_triggers,
            )
        }

        # Evaluate rules to get metadata for journey nodes
        journey_projections_with_metadata = await self._add_rule_metadata_to_projections(
            journey_projections, progress_report
        )

        relative_action_propositions = await self._propose_relative_actions(
            journey_projections_with_metadata,
            progress_report,
        )

        reachable_nodes_evaluations = await self._evaluate_reachable_nodes(
            journey_projections_with_metadata,
            progress_report,
        )

        invoices = await self._build_invoice_data(
            relative_action_propositions,
            reachable_nodes_evaluations,
            journey_projections_with_metadata,
        )

        return invoices

    async def _add_rule_metadata_to_projections(
        self,
        journey_projections: dict[JourneyId, tuple[Journey, Sequence[Rule], tuple[Rule]]],
        progress_report: Optional[ProgressReport] = None,
    ) -> dict[JourneyId, tuple[Journey, Sequence[Rule], tuple[Rule]]]:
        """Add rule evaluation metadata to journey node rules."""
        rule_payloads: list[RulePayload] = []
        journey_to_node_rules: dict[JourneyId, dict[JourneyNodeId, Rule]] = {}

        # Collect all nodes and create payloads
        for journey_id, (
            journey,
            step_rules,
            journey_triggers,
        ) in journey_projections.items():
            journey_to_node_rules[journey_id] = {}
            for rule in step_rules:
                node_id = extract_node_id_from_journey_node_rule_id(rule.id)

                if node_id != JourneyStore.END_NODE_ID:
                    node = await self._journey_store.read_node(node_id=node_id)

                # Store the rule by node_id for later mapping
                journey_to_node_rules[journey_id][node_id] = rule

                # Create RulePayload for journey node rules
                rule_payload = RulePayload(
                    content=rule.content,
                    tool_ids=node.tools,
                    operation=PayloadOperation.ADD,
                    action_proposition=True,
                    properties_proposition=False,
                    journey_node_proposition=True,
                )
                rule_payloads.append(rule_payload)

        if not rule_payloads:
            return journey_projections

        # Evaluate each rule payload individually using async gather
        rule_evaluation_tasks = [
            self._rule_evaluator.evaluate(
                [payload],  # Pass each payload as a single-item list
                progress_report=progress_report,
            )
            for payload in rule_payloads
        ]

        rule_evaluation_results = await async_utils.safe_gather(*rule_evaluation_tasks)

        rule_evaluations = [result[0] for result in rule_evaluation_results]

        # Add metadata back to the rules
        updated_projections: dict[JourneyId, tuple[Journey, Sequence[Rule], tuple[Rule]]] = {}

        # Create a mapping from rule payloads to evaluations
        evaluation_index = 0
        for journey_id, (
            journey,
            step_rules,
            journey_triggers,
        ) in journey_projections.items():
            updated_step_rules: list[Rule] = []
            node_rules = journey_to_node_rules[journey_id]

            for rule in step_rules:
                node_id = extract_node_id_from_journey_node_rule_id(rule.id)

                if node_id in node_rules and evaluation_index < len(rule_evaluations):
                    evaluation_data = rule_evaluations[evaluation_index]
                    evaluation_index += 1

                    updated_metadata = {
                        **(
                            evaluation_data.properties_proposition
                            if evaluation_data.properties_proposition
                            else {}
                        ),
                        **rule.metadata,
                        **(
                            {"rule_evaluation": evaluation_data.properties_proposition}
                            if evaluation_data.properties_proposition
                            else {}
                        ),
                    }

                    updated_rule = replace(
                        rule,
                        metadata=updated_metadata,
                    )
                    updated_step_rules.append(updated_rule)
                else:
                    updated_step_rules.append(rule)

            updated_projections[journey_id] = (journey, updated_step_rules, journey_triggers)

        return updated_projections

    async def _propose_relative_actions(
        self,
        journey_projections: dict[JourneyId, tuple[Journey, Sequence[Rule], tuple[Rule]]],
        progress_report: Optional[ProgressReport] = None,
    ) -> Sequence[RelativeActionProposition]:
        tasks: list[asyncio.Task[RelativeActionProposition]] = []

        for journey_id, (
            journey,
            step_rules,
            journey_triggers,
        ) in journey_projections.items():
            if not step_rules:
                continue

            tasks.append(
                asyncio.create_task(
                    self._relative_action_proposer.propose_relative_action(
                        examined_journey=journey,
                        step_rules=step_rules,
                        journey_triggers=journey_triggers,
                        progress_report=progress_report,
                    )
                )
            )

        sparse_results = list(await async_utils.safe_gather(*tasks))

        return sparse_results

    async def _evaluate_reachable_nodes(
        self,
        journey_projections: dict[JourneyId, tuple[Journey, Sequence[Rule], tuple[Rule]]],
        progress_report: Optional[ProgressReport] = None,
    ) -> Sequence[ReachableNodesEvaluation]:
        tasks: list[asyncio.Task[ReachableNodesEvaluation]] = []

        for journey_id, (
            journey,
            step_rules,
            journey_triggers,
        ) in journey_projections.items():
            if not step_rules:
                continue

            tasks.append(
                asyncio.create_task(
                    self._journey_reachable_node_evaluator.evaluate_reachable_follow_ups(
                        node_rules=step_rules,
                        progress_report=progress_report,
                    )
                )
            )

        sparse_results = list(await async_utils.safe_gather(*tasks))

        return sparse_results


class EvaluationService:
    def __init__(
        self,
        logger: Logger,
        background_task_service: BackgroundTaskService,
        entity_queries: EntityQueries,
        journey_rule_projection: JourneyRuleProjection,
        rule_action_proposer: RuleActionProposer,
        rule_continuous_proposer: RuleContinuousProposer,
        customer_dependent_action_detector: CustomerDependentActionDetector,
        agent_intention_proposer: AgentIntentionProposer,
        tool_running_action_detector: ToolRunningActionDetector,
        rule_signal_proposer: RuleSignalProposer,
        rule_title_proposer: RuleTitleProposer,
        relative_action_proposer: RelativeActionProposer,
        journey_reachable_node_evaluator: JourneyReachableNodesEvaluator,
        store_provider: StoreProvider,
    ) -> None:
        self._logger = logger
        self._store_provider = store_provider
        self._background_task_service = background_task_service
        self._entity_queries = entity_queries
        self._rule_evaluator = RuleEvaluator(
            logger=logger,
            entity_queries=entity_queries,
            rule_action_proposer=rule_action_proposer,
            rule_continuous_proposer=rule_continuous_proposer,
            customer_dependent_action_detector=customer_dependent_action_detector,
            agent_intention_proposer=agent_intention_proposer,
            tool_running_action_detector=tool_running_action_detector,
            rule_signal_proposer=rule_signal_proposer,
            rule_title_proposer=rule_title_proposer,
        )

        self._journey_evaluator = JourneyEvaluator(
            logger=logger,
            journey_rule_projection=journey_rule_projection,
            rule_evaluator=self._rule_evaluator,
            relative_action_proposer=relative_action_proposer,
            journey_reachable_node_evaluator=journey_reachable_node_evaluator,
            store_provider=store_provider,
        )

    @property
    def _agent_store(self) -> AgentStore:
        return self._store_provider.get_store(AgentStore, StoreProviderHints(call_site="engine"))

    def _get_evaluation_store(self, hints: StoreProviderHints) -> EvaluationStore:
        return self._store_provider.get_store(EvaluationStore, hints)

    async def validate_payloads(
        self,
        payload_descriptors: Sequence[PayloadDescriptor],
    ) -> None:
        if not payload_descriptors:
            raise EvaluationValidationError("No payloads provided for the evaluation task.")

    async def create_evaluation_task(
        self,
        payload_descriptors: Sequence[PayloadDescriptor],
        hints: StoreProviderHints = {},
    ) -> EvaluationId:
        await self.validate_payloads(payload_descriptors)

        evaluation = await self._get_evaluation_store(hints).create_evaluation(
            payload_descriptors,
        )

        await self._background_task_service.start(
            self.run_evaluation(evaluation, hints=hints),
            tag=f"evaluation({evaluation.id})",
        )

        return evaluation.id

    async def read_evaluation(
        self,
        evaluation_id: EvaluationId,
        hints: StoreProviderHints = {},
    ) -> Evaluation:
        return await self._get_evaluation_store(hints).read_evaluation(evaluation_id)

    async def list_evaluations(
        self,
        hints: StoreProviderHints = {},
    ) -> Sequence[Evaluation]:
        return await self._get_evaluation_store(hints).list_evaluations()

    async def update_evaluation(
        self,
        evaluation_id: EvaluationId,
        params: EvaluationUpdateParams,
        hints: StoreProviderHints = {},
    ) -> Evaluation:
        return await self._get_evaluation_store(hints).update_evaluation(
            evaluation_id=evaluation_id,
            params=params,
        )

    async def run_evaluation(
        self,
        evaluation: Evaluation,
        hints: StoreProviderHints = {},
    ) -> None:
        evaluation_store = self._get_evaluation_store(hints)

        async def _update_progress(percentage: float) -> None:
            await evaluation_store.update_evaluation(
                evaluation_id=evaluation.id,
                params={"progress": percentage},
            )

        progress_report = ProgressReport(_update_progress)

        try:
            await evaluation_store.update_evaluation(
                evaluation_id=evaluation.id,
                params={"status": EvaluationStatus.RUNNING},
            )

            evaluation_invoices = list(evaluation.invoices)

            rule_evaluation_data, journey_evaluation_data = await async_utils.safe_gather(
                self._rule_evaluator.evaluate(
                    payloads=[
                        cast(RulePayload, invoice.payload)
                        for invoice in evaluation_invoices
                        if invoice.kind == PayloadKind.RULE
                    ],
                    progress_report=progress_report,
                ),
                self._journey_evaluator.evaluate(
                    payloads=[
                        cast(JourneyPayload, invoice.payload)
                        for invoice in evaluation_invoices
                        if invoice.kind == PayloadKind.JOURNEY
                    ],
                    progress_report=progress_report,
                ),
            )

            evaluation_data: Sequence[InvoiceData] = list(rule_evaluation_data) + list(
                journey_evaluation_data
            )

            invoices: list[Invoice] = []
            for i, result in enumerate(evaluation_data):
                invoice_checksum = xxh3_checksum(str(evaluation.invoices[i].payload))
                state_version = str(hash("Temporarily"))

                invoices.append(
                    Invoice(
                        kind=evaluation.invoices[i].kind,
                        payload=evaluation.invoices[i].payload,
                        checksum=invoice_checksum,
                        state_version=state_version,
                        approved=True,
                        data=result,
                        error=None,
                    )
                )

            await evaluation_store.update_evaluation(
                evaluation_id=evaluation.id,
                params={"invoices": invoices},
            )

            self._logger.trace(f"evaluation task '{evaluation.id}' completed")

            await evaluation_store.update_evaluation(
                evaluation_id=evaluation.id,
                params={"status": EvaluationStatus.COMPLETED},
            )

        except Exception as exc:
            logger_level = "info" if isinstance(exc, EvaluationError) else "error"
            getattr(self._logger, logger_level)(
                f"Evaluation task '{evaluation.id}' failed due to the following error: '{str(exc)}'"
            )

            await evaluation_store.update_evaluation(
                evaluation_id=evaluation.id,
                params={
                    "status": EvaluationStatus.FAILED,
                    "error": str(exc)
                    if isinstance(exc, EvaluationError)
                    else str(exc) + str(traceback.format_exception(exc)),
                },
            )

            raise
