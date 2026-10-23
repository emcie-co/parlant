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

from typing import Any, Mapping, Sequence

from parlant.core.agents import Effort
from parlant.core.nlp.model_routing import ModelRouteQualifiers


class CompassOptimizationPolicy:
    """Engine-specific optimization policy extension point for Compass."""

    def get_process_delay(self, k: int) -> int:
        """Return how many milliseconds to wait before processing a customer message.

        Every customer message can restart processing and cancel the in-flight
        run for the same session. For customers who habitually send several
        messages in succession, delaying later messages in that burst gives
        them time to finish and avoids repeatedly starting costly work.

        ``k`` is the zero-based index of the current customer message since the
        agent's latest message reply. Returning ``0`` starts immediately; the
        caller must not issue an ``asyncio.sleep`` call in that case.
        """
        if k <= 0:
            return 0
        if k == 1:
            return 1_000
        return 3_000

    def get_hedge_timeout(
        self,
        route: str,
        *,
        qualifiers: ModelRouteQualifiers | None = None,
    ) -> float | None:
        """Return the hedge delay for a logical model route, in seconds.

        ``None`` disables hedging. Review calls get a longer window at high
        effort, where implementations may use richer schemas and prompts.
        """
        if route == "matching.rank":
            return 5.0

        if route == "react":
            return 10.0

        if route in {"review.tools", "review.message"}:
            effort = (qualifiers or {}).get("effort")
            if effort in {Effort.HIGH.value, Effort.MAX.value}:
                return 15.0
            return 10.0

        return None

    def use_embedding_cache(self, hints: Mapping[str, Any] = {}) -> bool:
        """Whether recall embeddings are cached on disk between runs."""
        return True

    def get_rule_proposition_retry_temperatures(
        self,
        hints: Mapping[str, Any] = {},
    ) -> Sequence[float]:
        """The generation temperatures (and thereby attempt count) for rule
        propositions during indexing — retries walk this list in order."""
        return [0.0, 0.15, 0.1]

    def get_min_prefill_fan_out(self, hints: Mapping[str, Any] = {}) -> int:
        """The smallest fan-out that justifies prefilling a shared prompt prefix.

        Some evaluations fan out one request per rule, all repeating the same
        prompt prefix. Warming that prefix first turns the repeats into cache
        reads, but the warm-up is itself a request, and providers price a cache
        *write* above an ordinary input token — so warming for a small fan-out
        costs more than not warming at all. Where the line falls depends on the
        provider's write premium; the default errs toward not warming.

        Return ``0`` to never prefill.
        """
        return 5

    def get_max_review_restarts(self, effort: Effort) -> int:
        """Return how many times one turn's step may be restarted because a
        reviewer rejected its output, before the loop gives up on the task and
        produces a final explanatory message instead."""
        match effort:
            case Effort.MIN:
                return 1
            case Effort.LOW:
                return 2
            case Effort.MEDIUM:
                return 3
            case Effort.HIGH:
                return 5
            case Effort.MAX:
                return 10
