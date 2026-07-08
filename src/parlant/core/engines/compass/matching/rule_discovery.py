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

"""The session-discovery port: should a rule join the session working set?

A :class:`RuleDiscoverer` answers a different question than a turn evaluator
(see :mod:`rule_evaluation`): not "does this rule match the current turn?" but
"has the conversation given this rule ongoing relevance?". A discovered rule is
admitted into the session working set, where it remains until pruned — so
discovery verdicts are sticky by contract, while turn verdicts are ephemeral.

Discoverers own their own recency semantics: a rule that was evicted from the
session set (``context.state.evicted_session_rules``) may only be readmitted by
conversation that arrived strictly after its eviction, never by the material
that originally admitted it (anti-flapping). That ledger travels in the context
state, so callers just ask about rules — they don't manage floors.
"""

from abc import ABC, abstractmethod
from typing import Sequence

from dataclasses import dataclass

from parlant.core.engines.compass.response_state import EngineContext
from parlant.core.rules import Rule


@dataclass(frozen=True)
class DiscoveredRule:
    """One discovery verdict: does the rule belong in the session working set?"""

    rule: Rule
    is_relevant: bool
    score: float


@dataclass(frozen=True)
class RuleDiscoveryResult:
    discovered_rules: Sequence[DiscoveredRule]
    duration: float


class RuleDiscoverer(ABC):
    """Judges which rules the conversation has made relevant to the session.

    Implementations must honor the eviction ledger in
    ``context.state.evicted_session_rules``: a ledgered rule may only be
    rediscovered by conversation newer than its eviction offset."""

    @abstractmethod
    async def discover(
        self,
        context: EngineContext,
        rules: Sequence[Rule],
    ) -> RuleDiscoveryResult: ...
