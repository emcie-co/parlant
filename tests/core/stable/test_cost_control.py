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

from parlant.core.container import Container

from parlant.core.agents import AgentId
from parlant.core.common import generate_id
from parlant.core.cost_control import (
    NullCostControlPolicy,
    CompositeCostControlPolicy,
    CostContext,
    CostControlPolicy,
    CostVerdict,
    WorkKind,
)
from parlant.core.customers import CustomerId
from parlant.core.sessions import SessionId


def _context() -> CostContext:
    return CostContext(
        agent_id=AgentId(generate_id()),
        session_id=SessionId(generate_id()),
        customer_id=CustomerId(generate_id()),
        trace_id=generate_id(),
    )


def test_that_the_default_cost_control_policy_is_composite(container: Container) -> None:
    assert isinstance(container[CostControlPolicy], CompositeCostControlPolicy)


async def test_that_the_null_policy_permits_every_work_kind() -> None:
    policy = NullCostControlPolicy()
    for work in WorkKind:
        assert (await policy.check(_context(), work)).allowed


async def test_that_composite_policy_returns_the_first_denial_and_accumulated_warnings() -> None:
    class Policy(CostControlPolicy):
        def __init__(self, verdict: CostVerdict) -> None:
            self.verdict = verdict

        async def check(self, context: CostContext, work: WorkKind) -> CostVerdict:
            return self.verdict

    composite = CompositeCostControlPolicy()
    composite.add(Policy(CostVerdict(allowed=True, warnings=("first",))))
    composite.add(Policy(CostVerdict(allowed=False, reason="stop", scope="session")))

    verdict = await composite.check(_context(), WorkKind.TURN)

    assert verdict.allowed is False
    assert verdict.warnings == ("first",)
    assert verdict.reason == "stop"
