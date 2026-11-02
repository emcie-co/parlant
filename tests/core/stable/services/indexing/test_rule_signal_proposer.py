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

from contextlib import contextmanager
from typing import Any, Iterator

from pytest import raises

from parlant.core.services.indexing.common import IndexingError
from parlant.core.services.indexing.rule_signal_proposer import (
    RuleSignalProposer,
    RuleSignalPropositionSchema,
)


class _QuietLogger:
    @contextmanager
    def scope(self, _: str) -> Iterator[None]:
        yield

    def warning(self, _: str) -> None:
        pass


class _RetryTemperatures:
    def get_rule_proposition_retry_temperatures(self, hints: Any = {}) -> list[float]:
        return [0.0, 0.15, 0.1]


def _proposer_answering(*propositions: RuleSignalPropositionSchema) -> RuleSignalProposer:
    """A proposer whose model returns the given propositions, one per attempt."""
    proposer = object.__new__(RuleSignalProposer)
    proposer._logger = _QuietLogger()  # type: ignore[assignment]
    proposer._optimization_policy = _RetryTemperatures()  # type: ignore[assignment]
    answers = iter(propositions)

    async def generate(*args: Any, **kwargs: Any) -> RuleSignalPropositionSchema:
        return next(answers)

    proposer._generate_signals = generate  # type: ignore[method-assign]
    return proposer


def _proposition(signals: int, anti_signals: int) -> RuleSignalPropositionSchema:
    return RuleSignalPropositionSchema(
        rationale="",
        signals=[f"signal {i}" for i in range(signals)],
        anti_signals=[f"anti-signal {i}" for i in range(anti_signals)],
    )


async def _propose(proposer: RuleSignalProposer) -> Any:
    return await proposer.propose_signals(
        content="Offer a call.",
        condition=None,
        title="Calls",
        agent=None,
        glossary_terms=[],
    )


async def test_that_extra_signals_are_trimmed_instead_of_failing_indexing() -> None:
    proposer = _proposer_answering(_proposition(signals=11, anti_signals=12))

    result = await _propose(proposer)

    assert list(result.signals) == [f"signal {i}" for i in range(10)]
    assert list(result.anti_signals) == [f"anti-signal {i}" for i in range(10)]


async def test_that_too_few_signals_retry_and_then_fail_indexing() -> None:
    proposer = _proposer_answering(*[_proposition(signals=9, anti_signals=10)] * 3)

    with raises(IndexingError, match="Expected 10 rule signals, but got only 9"):
        await _propose(proposer)


async def test_that_too_few_signals_retry_until_an_attempt_has_enough() -> None:
    proposer = _proposer_answering(
        _proposition(signals=10, anti_signals=7),
        _proposition(signals=10, anti_signals=10),
    )

    result = await _propose(proposer)

    assert len(result.signals) == 10
    assert len(result.anti_signals) == 10


async def test_that_duplicate_signals_do_not_count_toward_the_ten() -> None:
    proposition = _proposition(signals=10, anti_signals=10)
    proposition.signals[9] = "Signal 0 "
    proposer = _proposer_answering(*[proposition] * 3)

    with raises(IndexingError, match="but got only 9"):
        await _propose(proposer)


def test_that_rule_signal_proposer_formats_full_rule() -> None:
    proposer = object.__new__(RuleSignalProposer)

    assert proposer._format_rule(
        "Refunds",
        content="Use the current refund rule.",
        condition="the customer asks for a refund",
    ) == (
        "# Refunds\n\n## Condition: the customer asks for a refund\n\nUse the current refund rule."
    )


def test_that_rule_signal_proposer_formats_partial_rules() -> None:
    proposer = object.__new__(RuleSignalProposer)

    assert (
        proposer._format_rule(
            "Observation",
            content=None,
            condition="the customer reports a lost card",
        )
        == "# Observation\n\n## Condition: the customer reports a lost card"
    )

    assert (
        proposer._format_rule(
            "Description Only",
            content="Speak concisely.",
            condition=None,
        )
        == "# Description Only\n\nSpeak concisely."
    )

    assert (
        proposer._format_rule(
            "Title Only",
            content=None,
            condition=None,
        )
        == "# Title Only"
    )
