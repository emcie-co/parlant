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

"""The turn-evaluation port: is a rule relevant to the current turn?

A :class:`TurnEvaluator` judges rules against the most recent state of the
conversation — one implementation, one judgment style, all speaking the same
result shape. The matcher selects an evaluator per rule and consumes the
verdicts uniformly: ``is_relevant`` determines whether the evaluation proceeds
to relationship resolution, and any ``highlights`` are preserved for the
responder.

Note the contract this port deliberately does NOT cover: session discovery
("should this rule be admitted to the session working set?") is a different
question with different verdict semantics, and has its own port. Evaluators
answer only for the turn at hand.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from parlant.core.engines.compass.context import EngineContext
from parlant.core.engines.rule_evaluation import RuleEvaluation as RuleEvaluation
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.rules import Rule


@dataclass(frozen=True)
class RuleEvaluationResult:
    evaluations: Sequence[RuleEvaluation]
    # Aggregated usage across the per-rule fan-out this call, or None when no
    # requests were sent.
    generation_info: GenerationInfo | None


class TurnEvaluator(ABC):
    """Judges whether rules are relevant to the current turn.

    Implementations typically fan out one model call per rule behind a shared,
    prefix-cached prompt; ``warm_up`` prefills that shared prefix at end of turn
    so the next turn's fan-out sends only live suffixes (best-effort — warming
    failures must not break preparation)."""

    @abstractmethod
    async def evaluate(
        self,
        context: EngineContext,
        rules: Sequence[Rule],
    ) -> RuleEvaluationResult: ...

    async def evaluates_batch_in_one_request(self, context: EngineContext) -> bool:
        """Whether this turn's whole batch would cost a single model request.

        Fanning out one request per rule is what makes a verdict expensive, and
        the matcher's strategy matrix is priced around it: middling rules are
        left to embedding recall because ranking them would mean N requests. An
        evaluator that answers the whole batch at once changes that arithmetic,
        so it says so here and the matcher spends the saving on rules it would
        otherwise only recall.

        False by default: fan-out is the norm, and an evaluator that cannot
        promise a single request must not be read as if it could."""
        return False

    @abstractmethod
    async def warm_up(self, context: EngineContext) -> GenerationInfo | None: ...
