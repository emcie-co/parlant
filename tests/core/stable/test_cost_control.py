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

"""The cost-control policy port: a stable interface the engine gates chargeable
work through. The built-in advisory implementation never denies; enforcing
implementations plug into the same port."""

from lagom import Container

from parlant.core.agents import AgentId
from parlant.core.common import generate_id
from parlant.core.cost_control import (
    AdvisoryCostControlPolicy,
    CostContext,
    CostControlPolicy,
    CostVerdict,
    WorkKind,
)
from parlant.core.customers import CustomerId
from parlant.core.nlp.common import UsageInfo
from parlant.core.sessions import SessionId
from parlant.core.tracer import LocalTracer
from parlant.core.usage_reporter import UsageReporter


def _cost_context() -> CostContext:
    return CostContext(
        agent_id=AgentId(generate_id()),
        session_id=SessionId(generate_id()),
        customer_id=CustomerId(generate_id()),
        trace_id=generate_id(),
    )


def test_that_the_container_resolves_the_advisory_policy_for_the_port(
    container: Container,
) -> None:
    policy = container[CostControlPolicy]

    assert isinstance(policy, AdvisoryCostControlPolicy)


async def test_that_the_advisory_policy_allows_every_work_kind() -> None:
    policy = AdvisoryCostControlPolicy(UsageReporter(LocalTracer()))
    context = _cost_context()

    for work in WorkKind:
        verdict = await policy.check(context, work)

        assert verdict.allowed is True
        assert verdict.warnings == ()
        assert verdict.retry_after_utc is None


def test_that_the_advisory_policy_accepts_usage_reports_as_a_no_op() -> None:
    policy = AdvisoryCostControlPolicy(UsageReporter(LocalTracer()))

    # Must never raise and must not block (it is called on the hot path).
    policy.report(
        trace_id=generate_id(),
        model="some-model",
        usage=UsageInfo(input_tokens=100, output_tokens=10),
    )


def test_that_a_cost_verdict_defaults_to_no_warnings_and_no_retry_hint() -> None:
    verdict = CostVerdict(allowed=False)

    assert verdict.warnings == ()
    assert verdict.retry_after_utc is None
    assert verdict.reason is None


def test_that_the_container_wires_usage_reports_to_the_cost_control_policy(
    container: Container,
) -> None:
    # The policy self-subscribes at construction with a late-bound callback, so
    # shadowing `report` on the instance still intercepts.
    received: list[tuple[str, str, UsageInfo]] = []
    policy = container[CostControlPolicy]
    policy.report = lambda trace_id, model, usage: received.append(  # type: ignore[method-assign]
        (trace_id, model, usage)
    )

    usage = UsageInfo(input_tokens=42, output_tokens=7)
    container[UsageReporter].report_usage("some-model", usage)

    assert len(received) == 1
    assert received[0][1] == "some-model"
    assert received[0][2] == usage


# --- Weighted cost units ----------------------------------------------------------
#
# Thresholds are defined over weighted units, not raw tokens: the engine is
# deliberately cache-heavy, and cached input is ~an order of magnitude cheaper
# than uncached. Raw token counts would flag exactly the sessions the
# architecture makes cheap. Note: adapters report input_tokens INCLUSIVE of
# cached_input_tokens.


def test_that_weighted_cost_discounts_cached_input_and_weights_output() -> None:
    from parlant.core.cost_control import CostWeights

    weights = CostWeights(uncached_input=1.0, cached_input=0.1, output=4.0)

    # 10_000 input of which 9_000 cached, plus 500 output:
    # 1_000*1.0 + 9_000*0.1 + 500*4.0 = 3_900
    units = weights.weighted_cost(
        "some-model",
        UsageInfo(input_tokens=10_000, output_tokens=500, cached_input_tokens=9_000),
    )

    assert units == 3_900.0

    # The same tokens uncached are far more expensive: 10_000 + 2_000 = 12_000
    uncached_units = weights.weighted_cost(
        "some-model",
        UsageInfo(input_tokens=10_000, output_tokens=500, cached_input_tokens=0),
    )
    assert uncached_units == 12_000.0


def test_that_model_multipliers_scale_the_weighted_cost() -> None:
    from parlant.core.cost_control import CostWeights

    weights = CostWeights(model_multipliers={"expensive-model": 10.0})
    usage = UsageInfo(input_tokens=1_000, output_tokens=0)

    assert weights.weighted_cost("expensive-model", usage) == 10 * weights.weighted_cost(
        "cheap-model", usage
    )


# --- Sliding cost window ----------------------------------------------------------


def test_that_the_sliding_window_forgets_usage_older_than_its_span() -> None:
    from parlant.core.cost_control import SlidingCostWindow

    window = SlidingCostWindow(span_seconds=60.0)

    window.add(at=0.0, units=100.0)
    window.add(at=30.0, units=50.0)

    assert window.total(now=45.0) == 150.0
    assert window.total(now=61.0) == 50.0  # the first entry aged out
    assert window.total(now=120.0) == 0.0


# --- The advisory policy: real accounting, no teeth --------------------------------


def _advisory(
    threshold: float | None,
    clock_value: list[float],
    window_seconds: float = 60.0,
) -> AdvisoryCostControlPolicy:
    return AdvisoryCostControlPolicy(
        UsageReporter(LocalTracer()),
        advisory_threshold_units=threshold,
        window_seconds=window_seconds,
        clock=lambda: clock_value[0],
    )


async def test_that_the_advisory_policy_attributes_usage_to_the_checked_session() -> None:
    clock = [0.0]
    policy = _advisory(threshold=None, clock_value=clock)
    context = _cost_context()

    await policy.check(context, WorkKind.TURN)  # binds trace -> session
    policy.report(context.trace_id, "some-model", UsageInfo(input_tokens=100, output_tokens=0))

    assert policy.session_units(context.session_id) == 100.0
    assert policy.unattributed_units() == 0.0


async def test_that_usage_on_unknown_traces_lands_in_the_unattributed_bucket() -> None:
    clock = [0.0]
    policy = _advisory(threshold=None, clock_value=clock)

    policy.report("never-checked-trace", "some-model", UsageInfo(input_tokens=70, output_tokens=0))

    assert policy.unattributed_units() == 70.0


async def test_that_crossing_the_advisory_threshold_warns_without_blocking() -> None:
    clock = [0.0]
    policy = _advisory(threshold=1_000.0, clock_value=clock)
    context = _cost_context()

    await policy.check(context, WorkKind.TURN)
    policy.report(context.trace_id, "some-model", UsageInfo(input_tokens=5_000, output_tokens=0))

    verdict = await policy.check(context, WorkKind.TURN)

    assert verdict.allowed is True  # advisory: never blocks
    assert verdict.warnings
    assert verdict.scope == "session"


async def test_that_advisory_warnings_subside_once_the_window_decays() -> None:
    clock = [0.0]
    policy = _advisory(threshold=1_000.0, clock_value=clock, window_seconds=60.0)
    context = _cost_context()

    await policy.check(context, WorkKind.TURN)
    policy.report(context.trace_id, "some-model", UsageInfo(input_tokens=5_000, output_tokens=0))

    clock[0] = 30.0
    assert (await policy.check(context, WorkKind.TURN)).warnings

    clock[0] = 61.0  # the expensive burst aged out of the window
    assert not (await policy.check(context, WorkKind.TURN)).warnings


async def test_that_the_advisory_policy_never_warns_without_a_configured_threshold() -> None:
    clock = [0.0]
    policy = _advisory(threshold=None, clock_value=clock)
    context = _cost_context()

    await policy.check(context, WorkKind.TURN)
    policy.report(
        context.trace_id, "some-model", UsageInfo(input_tokens=10_000_000, output_tokens=0)
    )

    verdict = await policy.check(context, WorkKind.TURN)

    assert verdict.allowed is True
    assert verdict.warnings == ()
