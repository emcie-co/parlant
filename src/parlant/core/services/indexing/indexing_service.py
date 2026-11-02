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
from typing import Optional, Sequence

from parlant.core import async_utils
from parlant.core.background_tasks import BackgroundTaskService
from parlant.core.common import xxh3_checksum
from parlant.core.entity_cq import EntityQueries
from parlant.core.glossary import Term
from parlant.core.indexing import (
    IndexingStatus,
    IndexingStore,
    IndexingTask,
    IndexingTaskId,
    IndexingTaskUpdateParams,
    Invoice,
    InvoiceData,
    InvoiceRuleData,
    PayloadDescriptor,
    RulePayload,
)
from parlant.core.loggers import Logger
from parlant.core.rules import compose_rule_query
from parlant.core.services.indexing.common import IndexingError, ProgressReport
from parlant.core.services.indexing.rule_relativity_proposer import (
    RuleRelativityProposer,
    RuleRelativityProposition,
)
from parlant.core.services.indexing.rule_signal_proposer import (
    RuleSignalProposer,
    RuleSignalProposition,
)
from parlant.core.services.indexing.rule_title_proposer import (
    RuleTitleProposer,
    RuleTitleProposition,
)
from parlant.core.store_provider import StoreProvider, StoreProviderHints


class IndexingValidationError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class RuleIndexingProcessor:
    # Glossary terms fetched per indexed rule (top-k by the rule's own
    # query) - the terms the rule depends on to be interpreted correctly.
    _TERMS_PER_RULE = 10

    def __init__(
        self,
        logger: Logger,
        entity_queries: EntityQueries,
        rule_signal_proposer: RuleSignalProposer,
        rule_title_proposer: RuleTitleProposer,
        rule_relativity_proposer: RuleRelativityProposer,
    ) -> None:
        self._logger = logger
        self._entity_queries = entity_queries
        self._rule_signal_proposer = rule_signal_proposer
        self._rule_title_proposer = rule_title_proposer
        self._rule_relativity_proposer = rule_relativity_proposer

    def _build_invoice_data(
        self,
        relativity_propositions: Sequence[Optional[RuleRelativityProposition]],
        signal_propositions: Sequence[Optional[RuleSignalProposition]],
        title_propositions: Sequence[Optional[RuleTitleProposition]],
    ) -> Sequence[InvoiceRuleData]:
        return [
            InvoiceRuleData(
                references_other_policies_proposition=(
                    relativity.references_other_policies if relativity else None
                ),
                signals_proposition=signals.signals if signals else None,
                anti_signals_proposition=signals.anti_signals if signals else None,
                title_proposition=title.title if title else None,
            )
            for relativity, signals, title in zip(
                relativity_propositions,
                signal_propositions,
                title_propositions,
            )
        ]

    async def process(
        self,
        payloads: Sequence[RulePayload],
        progress_report: Optional[ProgressReport] = None,
    ) -> Sequence[InvoiceRuleData]:
        relativity_propositions = await self._propose_relativity(payloads, progress_report)
        signal_propositions = await self._propose_signals(payloads, progress_report)
        title_propositions = await self._propose_titles(payloads, progress_report)

        return self._build_invoice_data(
            relativity_propositions,
            signal_propositions,
            title_propositions,
        )

    async def _propose_relativity(
        self,
        payloads: Sequence[RulePayload],
        progress_report: Optional[ProgressReport] = None,
    ) -> Sequence[Optional[RuleRelativityProposition]]:
        tasks: list[asyncio.Task[RuleRelativityProposition]] = []
        indices: list[int] = []

        for i, payload in enumerate(payloads):
            if not payload.relativity_proposition:
                continue

            tasks.append(
                asyncio.create_task(
                    self._rule_relativity_proposer.propose_relativity(
                        content=payload.content,
                        condition=payload.condition,
                        title=payload.title,
                        agent_id=payload.agent_id,
                        progress_report=progress_report,
                    )
                )
            )
            indices.append(i)

        sparse_results = await async_utils.safe_gather(*tasks)
        results: list[Optional[RuleRelativityProposition]] = [None] * len(payloads)
        for i, result in zip(indices, sparse_results):
            results[i] = result
        return results

    async def _propose_signals(
        self,
        payloads: Sequence[RulePayload],
        progress_report: Optional[ProgressReport] = None,
    ) -> Sequence[Optional[RuleSignalProposition]]:
        tasks: list[asyncio.Task[RuleSignalProposition]] = []
        indices: list[int] = []

        for i, payload in enumerate(payloads):
            if not payload.signal_proposition:
                continue

            agent = (
                await self._entity_queries.read_agent(payload.agent_id)
                if payload.agent_id
                else None
            )
            glossary_terms = await self._find_relevant_glossary_terms(payload)

            tasks.append(
                asyncio.create_task(
                    self._rule_signal_proposer.propose_signals(
                        content=payload.content,
                        condition=payload.condition,
                        title=payload.title or "Untitled Rule",
                        agent=agent,
                        glossary_terms=glossary_terms,
                        agent_id=payload.agent_id,
                        progress_report=progress_report,
                    )
                )
            )
            indices.append(i)

        sparse_results = await async_utils.safe_gather(*tasks)
        results: list[Optional[RuleSignalProposition]] = [None] * len(payloads)
        for i, result in zip(indices, sparse_results):
            results[i] = result
        return results

    async def _propose_titles(
        self,
        payloads: Sequence[RulePayload],
        progress_report: Optional[ProgressReport] = None,
    ) -> Sequence[Optional[RuleTitleProposition]]:
        tasks: list[asyncio.Task[RuleTitleProposition]] = []
        indices: list[int] = []

        for i, payload in enumerate(payloads):
            if not payload.title_proposition:
                continue

            agent = (
                await self._entity_queries.read_agent(payload.agent_id)
                if payload.agent_id
                else None
            )
            glossary_terms = await self._find_relevant_glossary_terms(payload)

            tasks.append(
                asyncio.create_task(
                    self._rule_title_proposer.propose_title(
                        content=payload.content,
                        condition=payload.condition,
                        agent=agent,
                        glossary_terms=glossary_terms,
                        agent_id=payload.agent_id,
                        progress_report=progress_report,
                    )
                )
            )
            indices.append(i)

        sparse_results = await async_utils.safe_gather(*tasks)
        results: list[Optional[RuleTitleProposition]] = [None] * len(payloads)
        for i, result in zip(indices, sparse_results):
            results[i] = result
        return results

    async def _find_relevant_glossary_terms(self, payload: RulePayload) -> list[Term]:
        if not payload.agent_id:
            return []

        return list(
            await self._entity_queries.find_glossary_terms_for_context(
                payload.agent_id,
                query=compose_rule_query(
                    title=payload.title,
                    condition=payload.condition,
                    content=payload.content,
                ),
                max_terms=self._TERMS_PER_RULE,
            )
        )


GuidelineIndexingProcessor = RuleIndexingProcessor


class IndexingService:
    def __init__(
        self,
        logger: Logger,
        background_task_service: BackgroundTaskService,
        entity_queries: EntityQueries,
        rule_signal_proposer: RuleSignalProposer,
        rule_title_proposer: RuleTitleProposer,
        rule_relativity_proposer: RuleRelativityProposer,
        store_provider: StoreProvider,
    ) -> None:
        self._logger = logger
        self._store_provider = store_provider
        self._background_task_service = background_task_service
        self._rule_indexing_processor = RuleIndexingProcessor(
            logger=logger,
            entity_queries=entity_queries,
            rule_signal_proposer=rule_signal_proposer,
            rule_title_proposer=rule_title_proposer,
            rule_relativity_proposer=rule_relativity_proposer,
        )

    def _get_indexing_store(self, hints: StoreProviderHints) -> IndexingStore:
        return self._store_provider.get_store(IndexingStore, hints)

    async def validate_payloads(
        self,
        payload_descriptors: Sequence[PayloadDescriptor],
    ) -> None:
        if not payload_descriptors:
            raise IndexingValidationError("No payloads provided for the indexing task.")

    async def create_indexing_task(
        self,
        payload_descriptors: Sequence[PayloadDescriptor],
        hints: StoreProviderHints = {},
    ) -> IndexingTaskId:
        await self.validate_payloads(payload_descriptors)

        indexing_task = await self._get_indexing_store(hints).create_indexing_task(
            payload_descriptors,
        )

        await self._background_task_service.start(
            self.run_indexing_task(indexing_task, hints=hints),
            tag=f"indexing({indexing_task.id})",
        )

        return indexing_task.id

    async def read_indexing_task(
        self,
        indexing_task_id: IndexingTaskId,
        hints: StoreProviderHints = {},
    ) -> IndexingTask:
        return await self._get_indexing_store(hints).read_indexing_task(indexing_task_id)

    async def list_indexing_tasks(
        self,
        hints: StoreProviderHints = {},
    ) -> Sequence[IndexingTask]:
        return await self._get_indexing_store(hints).list_indexing_tasks()

    async def update_indexing_task(
        self,
        indexing_task_id: IndexingTaskId,
        params: IndexingTaskUpdateParams,
        hints: StoreProviderHints = {},
    ) -> IndexingTask:
        return await self._get_indexing_store(hints).update_indexing_task(
            indexing_task_id=indexing_task_id,
            params=params,
        )

    async def run_indexing_task(
        self,
        indexing_task: IndexingTask,
        hints: StoreProviderHints = {},
    ) -> None:
        indexing_store = self._get_indexing_store(hints)

        async def _update_progress(percentage: float) -> None:
            await indexing_store.update_indexing_task(
                indexing_task_id=indexing_task.id,
                params={"progress": percentage},
            )

        progress_report = ProgressReport(_update_progress)

        try:
            await indexing_store.update_indexing_task(
                indexing_task_id=indexing_task.id,
                params={"status": IndexingStatus.RUNNING},
            )

            indexing_data: Sequence[InvoiceData] = await self._rule_indexing_processor.process(
                payloads=[invoice.payload for invoice in indexing_task.invoices],
                progress_report=progress_report,
            )

            invoices = [
                Invoice(
                    kind=source.kind,
                    payload=source.payload,
                    checksum=xxh3_checksum(str(source.payload)),
                    state_version=str(hash("Temporarily")),
                    approved=True,
                    data=result,
                    error=None,
                )
                for source, result in zip(indexing_task.invoices, indexing_data)
            ]

            await indexing_store.update_indexing_task(
                indexing_task_id=indexing_task.id,
                params={"invoices": invoices},
            )

            self._logger.trace(f"indexing task '{indexing_task.id}' completed")

            await indexing_store.update_indexing_task(
                indexing_task_id=indexing_task.id,
                params={"status": IndexingStatus.COMPLETED},
            )

        except Exception as exc:
            logger_level = "info" if isinstance(exc, IndexingError) else "error"
            getattr(self._logger, logger_level)(
                f"Indexing task '{indexing_task.id}' failed due to the following error: '{str(exc)}'"
            )

            await indexing_store.update_indexing_task(
                indexing_task_id=indexing_task.id,
                params={
                    "status": IndexingStatus.FAILED,
                    "error": str(exc)
                    if isinstance(exc, IndexingError)
                    else str(exc) + str(traceback.format_exception(exc)),
                },
            )
