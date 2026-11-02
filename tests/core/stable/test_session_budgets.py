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

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, cast

import pytest

from parlant.core.nlp.common import UsageInfo
from parlant.core.common import JSONSerializable
from parlant.core import metadata as metadata_utils
from parlant.core.agents import AgentId
from parlant.core.customers import CustomerId
from parlant.core.pricing import (
    Pricing,
    PricingConfigurationError,
    PricingRegistry,
    SessionBudgetDefinition,
    SessionBudgetLedger,
    SessionBudgetService,
)
from parlant.core.sessions import Session, SessionId, SessionStore
from parlant.sdk import _SessionBudgetEvaluator
from parlant.core.tracer import LocalTracer
from parlant.core.usage_reporter import UsageReporter


def _usage_metadata(
    *costs: tuple[int, str, datetime],
    baselines: dict[str, int] | None = None,
) -> dict[str, object]:
    return {
        "records": {
            f"trace-{sequence}": {
                "sequence": sequence,
                "timestamp_utc": timestamp.isoformat(),
                "models": {},
                "costs": {
                    "total": value,
                    "currency": "USD",
                    "per_model": {},
                },
                "pricing_error": None,
            }
            for sequence, value, timestamp in costs
        },
        "baselines": {key: {"sequence": sequence} for key, sequence in (baselines or {}).items()},
    }


def test_that_a_new_budget_starts_after_existing_usage() -> None:
    now = datetime(2026, 7, 16, tzinfo=timezone.utc)
    budget = SessionBudgetDefinition("handoff", Decimal("10"), "USD")
    ledger = SessionBudgetLedger(_usage_metadata((1, "20", now)))

    assert ledger.initialize(budget) is True
    assert ledger.measure(budget, now=now) == Decimal("0")


def test_that_a_budget_matches_only_strictly_above_its_value() -> None:
    now = datetime(2026, 7, 16, tzinfo=timezone.utc)
    budget = SessionBudgetDefinition("handoff", Decimal("10"), "USD")
    key = budget.key

    equal = SessionBudgetLedger(_usage_metadata((1, "10", now), baselines={key: 0}))
    above = SessionBudgetLedger(_usage_metadata((1, "10.01", now), baselines={key: 0}))

    assert equal.exceeded(budget, now=now) is False
    assert above.exceeded(budget, now=now) is True


def test_that_a_windowed_budget_forgets_older_costs() -> None:
    now = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
    budget = SessionBudgetDefinition(
        "runaway",
        Decimal("5"),
        "USD",
        window=timedelta(minutes=5),
    )
    ledger = SessionBudgetLedger(
        _usage_metadata(
            (1, "100", now - timedelta(minutes=6)),
            (2, "2", now - timedelta(minutes=1)),
            baselines={budget.key: 0},
        )
    )

    assert ledger.measure(budget, now=now) == Decimal("2")


def test_that_reset_only_advances_the_selected_budget_baseline() -> None:
    now = datetime(2026, 7, 16, tzinfo=timezone.utc)
    first = SessionBudgetDefinition("first", Decimal("10"), "USD")
    second = SessionBudgetDefinition("second", Decimal("10"), "USD")
    ledger = SessionBudgetLedger(
        _usage_metadata(
            (1, "7", now),
            baselines={first.key: 0, second.key: 0},
        )
    )

    ledger.reset(first)

    assert ledger.measure(first, now=now) == Decimal("0")
    assert ledger.measure(second, now=now) == Decimal("7")
    records = ledger.serialize()["records"]
    assert isinstance(records, dict) and "trace-1" in records


class _SessionStore:
    def __init__(self) -> None:
        now = datetime.now(timezone.utc)
        self.session = Session(
            id=SessionId("session"),
            creation_utc=now,
            modified_utc=now,
            customer_id=CustomerId("customer"),
            agent_id=AgentId("agent"),
            mode="auto",
            title=None,
            agent_states=[],
            metadata={},
        )

    async def read_session(self, session_id: object) -> Session:
        return self.session

    async def commit_session_metadata(
        self,
        session_id: object,
        expected_revision: int,
        metadata: dict[str, object],
    ) -> Session:
        assert self.session.metadata_revision == expected_revision
        self.session = replace(
            self.session,
            metadata=cast(dict[str, JSONSerializable], metadata),
            metadata_revision=expected_revision + 1,
        )
        return self.session


async def test_that_raw_usage_is_persisted_when_pricing_is_not_set() -> None:
    tracer = LocalTracer()
    reporter = UsageReporter(tracer)
    registry = PricingRegistry(reporter)
    store = _SessionStore()
    service = SessionBudgetService(cast(SessionStore, store), registry, tracer)
    usage = {"model": UsageInfo(input_tokens=10, output_tokens=2)}

    await service.record_trace(SessionId("session"), "trace", usage)

    usage_metadata = cast(
        dict[str, object],
        metadata_utils.internal_value(store.session.metadata, ("usage",), {}),
    )
    record = cast(dict[str, Any], cast(dict[str, object], usage_metadata["records"])["trace"])
    assert record["models"]["model"]["requests"] == 1
    assert record["models"]["model"]["input_tokens"] == 10
    assert record["costs"] is None


async def test_that_an_unpriced_model_is_persisted_and_fails_visibly_for_an_active_budget() -> None:
    tracer = LocalTracer()
    reporter = UsageReporter(tracer)
    registry = PricingRegistry(reporter)
    registry.set_pricing(Pricing("USD", {}))
    registry.register_budget(SessionBudgetDefinition("budget", Decimal("1"), "USD"))
    store = _SessionStore()
    service = SessionBudgetService(cast(SessionStore, store), registry, tracer)
    usage = UsageInfo(input_tokens=10, output_tokens=0)
    registry.report_usage("trace", "missing", usage)

    with pytest.raises(PricingConfigurationError, match="missing"):
        await service.record_trace(
            SessionId("session"),
            "trace",
            {"missing": usage},
        )

    usage_metadata = cast(
        dict[str, object],
        metadata_utils.internal_value(store.session.metadata, ("usage",), {}),
    )
    record = cast(dict[str, Any], cast(dict[str, object], usage_metadata["records"])["trace"])
    assert "missing" in record["pricing_error"]


class _FixedSpendBudget:
    """A budget whose measured spend is whatever the test says it is."""

    def __init__(self, value: str, measured: str) -> None:
        self.value = Decimal(value)
        self.currency = "USD"
        self._measured = Decimal(measured)

    async def _measure(self, session: Any) -> Decimal:
        return self._measured


async def _decide(evaluator: Any) -> bool:
    rule = cast(Any, SimpleNamespace(id="r-1"))
    context = cast(Any, SimpleNamespace(session=None))
    return bool((await evaluator(context, rule)).is_relevant)


async def test_that_the_two_budget_evaluators_split_every_turn_between_them() -> None:
    # Whatever the spend, exactly one of the pair matches — so a rule written
    # either way covers the turn, and the two never both apply.
    for measured in ("0", "9.99", "10", "10.01", "1000"):
        budget = cast(Any, _FixedSpendBudget("10", measured))

        exceeded = await _decide(_SessionBudgetEvaluator(budget, match_when_exceeded=True))
        not_exceeded = await _decide(_SessionBudgetEvaluator(budget, match_when_exceeded=False))

        assert exceeded is not not_exceeded, f"both agreed at a spend of {measured}"


async def test_that_spending_exactly_the_budget_has_not_exceeded_it() -> None:
    budget = cast(Any, _FixedSpendBudget("10", "10"))

    assert await _decide(_SessionBudgetEvaluator(budget, match_when_exceeded=False)) is True
    assert await _decide(_SessionBudgetEvaluator(budget, match_when_exceeded=True)) is False


async def test_that_both_evaluators_explain_the_decision_with_the_numbers() -> None:
    budget = cast(Any, _FixedSpendBudget("10", "12.50"))
    rule = cast(Any, SimpleNamespace(id="r-1"))
    context = cast(Any, SimpleNamespace(session=None))

    for sense in (True, False):
        evaluation = await _SessionBudgetEvaluator(budget, match_when_exceeded=sense)(context, rule)

        assert "12.50" in evaluation.reasoning
        assert "10" in evaluation.reasoning
