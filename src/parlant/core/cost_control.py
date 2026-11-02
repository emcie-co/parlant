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

"""Admission-control policies for chargeable engine work.

Usage accounting and pricing live in ``parlant.core.pricing``. Policies only
decide whether a TURN, STEP, or BACKGROUND unit of work may proceed.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional, Sequence

from parlant.core.agents import AgentId
from parlant.core.customers import CustomerId
from parlant.core.sessions import SessionId
from parlant.core.engines.session_metadata_workspace import SessionMetadataWorkspace


@dataclass(frozen=True)
class CostContext:
    agent_id: AgentId
    session_id: SessionId
    customer_id: CustomerId
    trace_id: str
    metadata_workspace: SessionMetadataWorkspace | None = field(
        default=None,
        repr=False,
        compare=False,
    )


class WorkKind(Enum):
    TURN = auto()
    STEP = auto()
    BACKGROUND = auto()


@dataclass(frozen=True)
class CostVerdict:
    allowed: bool
    warnings: Sequence[str] = field(default=())
    retry_after_utc: Optional[datetime] = None
    reason: Optional[str] = None
    scope: Optional[str] = None


class CostControlPolicy(ABC):
    @abstractmethod
    async def check(self, context: CostContext, work: WorkKind) -> CostVerdict: ...


class NullCostControlPolicy(CostControlPolicy):
    async def check(self, context: CostContext, work: WorkKind) -> CostVerdict:
        return CostVerdict(allowed=True)


class CompositeCostControlPolicy(CostControlPolicy):
    """Mutable policy collection used by SDK-configured circuit breakers."""

    def __init__(self) -> None:
        self._policies: list[CostControlPolicy] = []

    def add(self, policy: CostControlPolicy) -> None:
        self._policies.append(policy)

    async def check(self, context: CostContext, work: WorkKind) -> CostVerdict:
        warnings: list[str] = []
        for policy in self._policies:
            verdict = await policy.check(context, work)
            warnings.extend(verdict.warnings)
            if not verdict.allowed:
                return CostVerdict(
                    allowed=False,
                    warnings=tuple(warnings),
                    retry_after_utc=verdict.retry_after_utc,
                    reason=verdict.reason,
                    scope=verdict.scope,
                )
        return CostVerdict(allowed=True, warnings=tuple(warnings))
