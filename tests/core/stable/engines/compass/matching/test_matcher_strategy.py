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

"""How much matching effort a rule gets, by the turn's effort and the rule's weight.

The matrix in `Matcher._get_strategy` trades thoroughness for cost: ranking used
to mean one model request per rule, so middling rules were left to embedding
recall. A decision model collapses the whole batch into one request, which buys
back two cells of that matrix — and only those two, so the tests below pin the
cells that must NOT move as firmly as the ones that must.
"""

from dataclasses import replace
from typing import Any, Mapping, Sequence

from parlant.core.agents import Effort
from parlant.core.common import JSONSerializable, Weight
from parlant.core.container import Container
from parlant.core.engines.compass.matcher import _MatcherStrategy
from parlant.core.nlp.decision import (
    DecisionMaker,
    DecisionResult,
    PropositionAnswer,
    Question,
)
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.nlp.common import UsageInfo
from parlant.core.rules import Rule

from tests.core.stable.engines.compass.matching.harness import make_rule, matching_harness


class _AlwaysRelevantDecisionMaker(DecisionMaker):
    """A decision model that exists. The strategy matrix only asks whether one is
    configured, so the verdicts themselves never matter here."""

    async def decide(
        self,
        state: JSONSerializable,
        questions: Mapping[str, Question],
        hints: Mapping[str, Any] = {},
    ) -> DecisionResult:
        return DecisionResult(
            answers={name: PropositionAnswer(probability=1.0) for name in questions},
            info=GenerationInfo(
                schema_name="Decision",
                model="test-decider",
                duration=0.0,
                usage=UsageInfo(input_tokens=0, output_tokens=0, requests=1),
            ),
        )

    @property
    def id(self) -> str:
        return "test/decider"

    @property
    def max_tokens(self) -> int:
        return 128_000


def _weighted(name: str, weight: Weight) -> Rule:
    return replace(make_rule(name), weight=weight)


# Past `Matcher._SMALL_INVENTORY_THRESHOLD`, below which a whole inventory is
# rendered verbatim and every rule short-circuits to NONE — which would hide the
# matrix these tests are about.
def _filler(count: int = 8) -> list[Rule]:
    return [_weighted(f"FILLER{i}", Weight.LOW) for i in range(count)]


async def _strategy_for(
    container: Container,
    *,
    rule: Rule,
    effort: Effort,
    with_decision_model: bool,
) -> _MatcherStrategy:
    """The strategy the matcher would pick, resolved through the real seam.

    `_load_strategy_choice_signals` is what populates the per-turn signals
    `_get_strategy` reads, so going through it keeps these tests honest about
    where the answer comes from."""
    harness = await matching_harness(
        container,
        [rule, *_filler()],
        rank_relevant=set(),
        effort=effort,
        decision_maker=_AlwaysRelevantDecisionMaker() if with_decision_model else None,
    )

    await harness.matcher._load_strategy_choice_signals(harness.context, [rule])

    return harness.matcher._get_strategy(harness.context, rule)


async def test_that_medium_weight_is_recalled_at_medium_effort_without_a_decision_model(
    container: Container,
) -> None:
    strategy = await _strategy_for(
        container,
        rule=_weighted("PLAIN", Weight.MEDIUM),
        effort=Effort.MEDIUM,
        with_decision_model=False,
    )

    assert strategy == _MatcherStrategy.RECALL


async def test_that_medium_weight_is_ranked_at_medium_effort_with_a_decision_model(
    container: Container,
) -> None:
    strategy = await _strategy_for(
        container,
        rule=_weighted("PLAIN", Weight.MEDIUM),
        effort=Effort.MEDIUM,
        with_decision_model=True,
    )

    assert strategy == _MatcherStrategy.RANK


async def test_that_high_weight_is_recalled_at_low_effort_without_a_decision_model(
    container: Container,
) -> None:
    strategy = await _strategy_for(
        container,
        rule=_weighted("PLAIN", Weight.HIGH),
        effort=Effort.LOW,
        with_decision_model=False,
    )

    assert strategy == _MatcherStrategy.RECALL


async def test_that_high_weight_is_ranked_at_low_effort_with_a_decision_model(
    container: Container,
) -> None:
    strategy = await _strategy_for(
        container,
        rule=_weighted("PLAIN", Weight.HIGH),
        effort=Effort.LOW,
        with_decision_model=True,
    )

    assert strategy == _MatcherStrategy.RANK


async def test_that_low_weight_stays_on_recall_at_medium_effort_with_a_decision_model(
    container: Container,
) -> None:
    # Only the two bought-back cells move. A low-weight rule is still not worth a
    # verdict, however cheap the verdict became.
    strategy = await _strategy_for(
        container,
        rule=_weighted("PLAIN", Weight.LOW),
        effort=Effort.MEDIUM,
        with_decision_model=True,
    )

    assert strategy == _MatcherStrategy.RECALL


async def test_that_min_effort_stays_on_recall_even_with_a_decision_model(
    container: Container,
) -> None:
    # MIN is the deliberately cheapest tier; a decision model does not promote it.
    for weight in (Weight.LOW, Weight.MEDIUM, Weight.HIGH):
        strategy = await _strategy_for(
            container,
            rule=_weighted("PLAIN", weight),
            effort=Effort.MIN,
            with_decision_model=True,
        )

        assert strategy == _MatcherStrategy.RECALL, weight


async def test_that_medium_weight_stays_on_recall_at_low_effort_with_a_decision_model(
    container: Container,
) -> None:
    # LOW buys back only its HIGH-weight cell, not its MEDIUM one.
    strategy = await _strategy_for(
        container,
        rule=_weighted("PLAIN", Weight.MEDIUM),
        effort=Effort.LOW,
        with_decision_model=True,
    )

    assert strategy == _MatcherStrategy.RECALL


async def test_that_the_decision_model_is_resolved_once_per_batch_not_once_per_rule(
    container: Container,
) -> None:
    # The router lookup is ~60us — nothing per turn, but real if it crept into
    # the per-rule path, which `_get_strategy` is.
    rules: Sequence[Rule] = [_weighted(f"RULE{i}", Weight.MEDIUM) for i in range(10)]
    harness = await matching_harness(
        container,
        list(rules),
        rank_relevant=set(),
        effort=Effort.MEDIUM,
        decision_maker=_AlwaysRelevantDecisionMaker(),
    )

    await harness.matcher._load_strategy_choice_signals(harness.context, rules)
    for rule in rules:
        harness.matcher._get_strategy(harness.context, rule)

    assert len(harness.decision_lookups) == 1


# ─────────────────────────── prefill target selection ───────────────────────
#
# Warming the ranker's shared prefix pays for one extra schematic request so the
# per-rule fan-out can hit a cache. A batched decision issues no fan-out and the
# Decisions endpoint does no prompt caching, so warming it buys nothing at all.


async def _prefill_targets(
    container: Container,
    *,
    effort: Effort,
    with_decision_model: bool,
    weight: Weight = Weight.HIGH,
) -> list[Any]:
    # HIGH weight by default so the rules are ranked whether or not a decision
    # model is configured — otherwise "nothing was warmed" could just mean
    # "nothing was going to be ranked anyway".
    harness = await matching_harness(
        container,
        [_weighted(f"RULE{i}", weight) for i in range(8)],
        rank_relevant=set(),
        effort=effort,
        decision_maker=_AlwaysRelevantDecisionMaker() if with_decision_model else None,
    )

    return await harness.matcher._get_prefill_targets(harness.context)


async def test_that_a_decision_capable_turn_warms_no_schematic_prefix(
    container: Container,
) -> None:
    targets = await _prefill_targets(
        container,
        effort=Effort.MEDIUM,
        with_decision_model=True,
    )

    assert targets == []


async def test_that_a_turn_without_a_decision_model_still_warms_the_schematic_prefix(
    container: Container,
) -> None:
    # The same rules at the same effort, differing only in whether a decision
    # model exists — so this is the control for the test above.
    targets = await _prefill_targets(
        container,
        effort=Effort.MEDIUM,
        with_decision_model=False,
    )

    assert targets != []


async def test_that_a_high_effort_turn_still_warms_the_schematic_prefix(
    container: Container,
) -> None:
    # HIGH effort wants per-rule prose, so the ranker fans out even with a
    # decision model configured — and a warm prefix is worth paying for again.
    targets = await _prefill_targets(
        container,
        effort=Effort.HIGH,
        with_decision_model=True,
    )

    assert targets != []
