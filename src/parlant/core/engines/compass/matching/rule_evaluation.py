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

"""The turn-evaluation port: does a rule match the current turn?

A :class:`TurnEvaluator` judges rules against the most recent state of the
conversation — one implementation, one judgment style, all speaking the same
result shape. The matcher selects an evaluator per rule and consumes the
verdicts uniformly: ``is_relevant`` gates the match, and any ``highlights`` an
evaluator attaches are surfaced with the match for the responder.

Note the contract this port deliberately does NOT cover: session discovery
("should this rule be admitted to the session working set?") is a different
question with different verdict semantics, and has its own port. Evaluators
answer only for the turn at hand.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

from parlant.core.engines.compass.response_state import EngineContext
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.rules import Rule


@dataclass(frozen=True)
class RuleEvaluation:
    """One evaluator's verdict on one rule, for the current turn."""

    rule: Rule

    reasoning: str
    """Why the rule does or doesn't apply; becomes the match's rationale."""

    is_relevant: bool
    """Whether the rule matches the current turn."""

    score: float | None = None
    """A normalized 0..1 relevance confidence, when the evaluator expresses
    one. Evaluators without a scoring scale return None."""

    highlights: Sequence[str] = field(default=())
    """Optional content the evaluator attaches to surface with the match —
    standalone points the responder should see next to the rule. Most
    evaluators return none."""


@dataclass(frozen=True)
class RuleEvaluationResult:
    evaluations: Sequence[RuleEvaluation]
    # Aggregated usage across the per-rule fan-out this call, or None when no
    # requests were sent.
    generation_info: GenerationInfo | None


class TurnEvaluator(ABC):
    """Judges whether rules match the current turn.

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

    @abstractmethod
    async def warm_up(self, context: EngineContext) -> GenerationInfo | None: ...
