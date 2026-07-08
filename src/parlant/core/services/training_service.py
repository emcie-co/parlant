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

"""Background training of the rule recaller's per-policy discriminants.

A training run re-derives every policy's discriminant from the current inventory
(see :meth:`RuleRecaller.retrain`). It is re-runnable, so jobs are tracked
in memory rather than persisted: the SDK drives a progress bar off the same
:class:`ProgressReport`, and the API exposes a job whose progress is polled via
``GET /train/{job_id}``.
"""

import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from parlant.core.agents import AgentId, AgentStore
from parlant.core.background_tasks import BackgroundTaskService
from parlant.core.common import ItemNotFoundError, UniqueId, generate_id
from parlant.core.engines.compass.matching.rule_recaller import RuleRecaller
from parlant.core.entity_cq import EntityQueries
from parlant.core.loggers import Logger
from parlant.core.services.indexing.common import ProgressReport


class TrainingStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TrainingJob:
    id: UniqueId
    status: TrainingStatus
    percentage: float
    agent_ids: Optional[Sequence[AgentId]] = None
    error: Optional[str] = None


class TrainingService:
    def __init__(
        self,
        recaller: RuleRecaller,
        agent_store: AgentStore,
        entity_queries: EntityQueries,
        background_task_service: BackgroundTaskService,
        logger: Logger,
    ) -> None:
        self._recaller = recaller
        self._agent_store = agent_store
        self._entity_queries = entity_queries
        self._background_task_service = background_task_service
        self._logger = logger
        self._jobs: dict[UniqueId, TrainingJob] = {}

    async def create_training_task(
        self,
        agent_ids: Optional[Sequence[AgentId]] = None,
    ) -> UniqueId:
        # Validate up front so a typo'd agent id fails the request rather than
        # silently training nothing.
        for agent_id in agent_ids or []:
            await self._agent_store.read_agent(agent_id)

        job_id = generate_id()
        self._jobs[job_id] = TrainingJob(
            id=job_id,
            status=TrainingStatus.PENDING,
            percentage=0.0,
            agent_ids=list(agent_ids) if agent_ids else None,
        )
        await self._background_task_service.start(self._run(job_id), tag=f"train({job_id})")
        return job_id

    async def read_training_job(self, job_id: UniqueId) -> TrainingJob:
        if job_id not in self._jobs:
            raise ItemNotFoundError(job_id, "Training job not found")
        return self._jobs[job_id]

    async def train(
        self,
        agent_ids: Optional[Sequence[AgentId]] = None,
        progress_report: Optional[ProgressReport] = None,
    ) -> None:
        """Train per-agent discriminant frames. With no ``agent_ids`` (or an empty
        list), train every agent; otherwise train only the named ones. Each agent's
        policies only compete within that agent, so frames are never shared. The SDK
        calls this directly on startup; the API path goes through a job. A shared
        ``progress_report`` accumulates across agents."""
        if agent_ids:
            target_ids = list(agent_ids)
        else:
            target_ids = [agent.id for agent in await self._agent_store.list_agents()]

        for agent_id in target_ids:
            await self.train_agent(agent_id, progress_report)

    async def train_agent(
        self,
        agent_id: AgentId,
        progress_report: Optional[ProgressReport] = None,
    ) -> None:
        """Train a single agent's discriminant frame over its own rule space."""
        rules = await self._entity_queries.find_rules_for_context(agent_id, [])
        await self._recaller.retrain(agent_id, rules, progress_report)

    async def _run(self, job_id: UniqueId) -> None:
        job = self._jobs[job_id]
        job.status = TrainingStatus.RUNNING

        async def on_progress(percentage: float) -> None:
            job.percentage = percentage

        try:
            await self.train(job.agent_ids, ProgressReport(on_progress))
            job.status = TrainingStatus.COMPLETED
            job.percentage = 100.0
        except Exception as exc:
            self._logger.error(
                f"Training job '{job_id}' failed: {exc}\n\n"
                f"{''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}"
            )
            job.status = TrainingStatus.FAILED
            job.error = str(exc)
