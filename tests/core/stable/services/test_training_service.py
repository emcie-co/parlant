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
from collections.abc import Sequence
from typing import Optional

from lagom import Container
from pytest import raises

from parlant.core.agents import Agent, AgentId
from parlant.core.background_tasks import BackgroundTaskService
from parlant.core.common import ItemNotFoundError, UniqueId
from parlant.core.rules import Rule
from parlant.core.loggers import Logger
from parlant.core.services.indexing.common import ProgressReport
from parlant.core.services.training_service import TrainingService, TrainingStatus


class _FakeRecaller:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.trained_by_agent: dict[AgentId, Sequence[Rule]] = {}

    async def retrain(
        self,
        agent_id: AgentId,
        rules: Sequence[Rule],
        progress_report: Optional[ProgressReport] = None,
    ) -> None:
        self.trained_by_agent[agent_id] = list(rules)
        if progress_report:
            await progress_report.stretch(len(rules) or 1)
        if self.fail:
            raise RuntimeError("training blew up")
        if progress_report:
            await progress_report.increment(len(rules) or 1)


class _FakeAgentStore:
    def __init__(self, agents: Sequence[Agent]) -> None:
        self._agents = agents

    async def list_agents(self, *args: object, **kwargs: object) -> Sequence[Agent]:
        return self._agents

    async def read_agent(self, agent_id: AgentId) -> Agent:
        for agent in self._agents:
            if agent.id == agent_id:
                return agent
        raise ItemNotFoundError(UniqueId(agent_id))


class _FakeEntityQueries:
    def __init__(self, rules_by_agent: dict[AgentId, Sequence[Rule]]) -> None:
        self._rules_by_agent = rules_by_agent

    async def find_rules_for_context(self, agent_id: AgentId, groups: object) -> Sequence[Rule]:
        return self._rules_by_agent.get(agent_id, [])


def _training_service(
    container: Container,
    recaller: _FakeRecaller,
    agents: Sequence[Agent],
    rules_by_agent: dict[AgentId, Sequence[Rule]],
) -> TrainingService:
    return TrainingService(
        recaller=recaller,  # type: ignore[arg-type]
        agent_store=_FakeAgentStore(agents),  # type: ignore[arg-type]
        entity_queries=_FakeEntityQueries(rules_by_agent),  # type: ignore[arg-type]
        background_task_service=container[BackgroundTaskService],
        logger=container[Logger],
    )


async def _wait_until_settled(service: TrainingService, job_id: UniqueId) -> None:
    for _ in range(500):
        job = await service.read_training_job(job_id)
        if job.status in (TrainingStatus.COMPLETED, TrainingStatus.FAILED):
            return
        await asyncio.sleep(0.01)
    raise AssertionError("training job did not settle in time")


async def test_that_a_training_task_trains_every_agent_in_its_own_space(
    container: Container,
) -> None:
    from tests.core.stable.engines.compass.matching.utils import (
        create_agent,
        create_rule,
    )

    agent_a = create_agent()
    agent_b = create_agent()
    rules_a = [create_rule(condition="a refund is wanted", action="refund", groups=[])]
    rules_b = [create_rule(condition="hours are asked", action="give hours", groups=[])]

    recaller = _FakeRecaller()
    service = _training_service(
        container,
        recaller,
        agents=[agent_a, agent_b],
        rules_by_agent={agent_a.id: rules_a, agent_b.id: rules_b},
    )

    job_id = await service.create_training_task()
    await _wait_until_settled(service, job_id)

    job = await service.read_training_job(job_id)
    assert job.status == TrainingStatus.COMPLETED
    assert job.percentage == 100.0
    # Each agent was trained on its OWN rule space.
    assert {g.id for g in recaller.trained_by_agent[agent_a.id]} == {g.id for g in rules_a}
    assert {g.id for g in recaller.trained_by_agent[agent_b.id]} == {g.id for g in rules_b}


async def test_that_a_scoped_training_task_trains_only_the_named_agents(
    container: Container,
) -> None:
    from tests.core.stable.engines.compass.matching.utils import (
        create_agent,
        create_rule,
    )

    agent_a = create_agent()
    agent_b = create_agent()
    rules_a = [create_rule(condition="a refund is wanted", action="refund", groups=[])]
    rules_b = [create_rule(condition="hours are asked", action="give hours", groups=[])]

    recaller = _FakeRecaller()
    service = _training_service(
        container,
        recaller,
        agents=[agent_a, agent_b],
        rules_by_agent={agent_a.id: rules_a, agent_b.id: rules_b},
    )

    job_id = await service.create_training_task(agent_ids=[agent_a.id])
    await _wait_until_settled(service, job_id)

    assert (await service.read_training_job(job_id)).status == TrainingStatus.COMPLETED
    assert agent_a.id in recaller.trained_by_agent
    assert agent_b.id not in recaller.trained_by_agent


async def test_that_train_agent_trains_a_single_agent(container: Container) -> None:
    from tests.core.stable.engines.compass.matching.utils import (
        create_agent,
        create_rule,
    )

    agent = create_agent()
    rules = [create_rule(condition="a refund is wanted", action="refund", groups=[])]
    recaller = _FakeRecaller()
    service = _training_service(
        container, recaller, agents=[agent], rules_by_agent={agent.id: rules}
    )

    await service.train_agent(agent.id)

    assert {g.id for g in recaller.trained_by_agent[agent.id]} == {g.id for g in rules}


async def test_that_creating_a_training_task_for_an_unknown_agent_is_rejected(
    container: Container,
) -> None:
    from tests.core.stable.engines.compass.matching.utils import create_agent

    agent = create_agent()
    recaller = _FakeRecaller()
    service = _training_service(container, recaller, agents=[agent], rules_by_agent={agent.id: []})

    with raises(ItemNotFoundError):
        await service.create_training_task(agent_ids=[AgentId("nonexistent")])

    # Validation happens up front — no training was kicked off.
    assert recaller.trained_by_agent == {}


async def test_that_a_failed_training_task_is_marked_failed(container: Container) -> None:
    from tests.core.stable.engines.compass.matching.utils import (
        create_agent,
        create_rule,
    )

    agent = create_agent()
    rules = [create_rule(condition="x", action="y", groups=[])]
    recaller = _FakeRecaller(fail=True)
    service = _training_service(
        container, recaller, agents=[agent], rules_by_agent={agent.id: rules}
    )

    job_id = await service.create_training_task()
    await _wait_until_settled(service, job_id)

    job = await service.read_training_job(job_id)
    assert job.status == TrainingStatus.FAILED
    assert job.error and "blew up" in job.error


async def test_that_reading_an_unknown_training_job_raises(container: Container) -> None:
    service = _training_service(container, _FakeRecaller(), agents=[], rules_by_agent={})

    with raises(ItemNotFoundError):
        await service.read_training_job(UniqueId("nonexistent"))
