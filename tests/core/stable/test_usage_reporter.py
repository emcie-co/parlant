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

import pytest

from parlant.core import usage_reporter as usage_reporter_module
from parlant.core.nlp.common import UsageInfo
from parlant.core.tracer import LocalTracer
from parlant.core.usage_reporter import UsageReporter


def test_usage_reporter_aggregates_usage_by_model_for_current_trace() -> None:
    tracer = LocalTracer()
    reporter = UsageReporter(tracer)

    with tracer.span("turn-1"):
        reporter.report_usage(
            "model",
            UsageInfo(
                input_tokens=10,
                output_tokens=2,
                cached_input_tokens=3,
                extra={
                    "reasoning_tokens": 1,
                    "reported_cost": "0.0000001",
                    "note": "first",
                },
            ),
            provider_name="test",
        )
        reporter.report_usage(
            "model",
            UsageInfo(
                input_tokens=4,
                output_tokens=1,
                cached_input_tokens=2,
                extra={
                    "reasoning_tokens": 5,
                    "reported_cost": "0.0000002",
                    "note": "second",
                },
            ),
            provider_name="test",
        )
        reporter.report_usage(
            "other",
            UsageInfo(input_tokens=1, output_tokens=0),
            provider_name="test",
        )

        usage = reporter.get_usage()

    assert usage["test:model"] == UsageInfo(
        input_tokens=14,
        output_tokens=3,
        cached_input_tokens=5,
        extra={
            "reasoning_tokens": 6,
            "reported_cost": "0.0000003",
            "note": "second",
        },
        requests=2,
    )
    assert usage["test:other"] == UsageInfo(input_tokens=1, output_tokens=0)

    with tracer.span("turn-2"):
        assert reporter.get_usage() == {}


def test_usage_reporter_evicts_oldest_trace_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(usage_reporter_module, "_USAGE_CACHE_MAX_SIZE", 2)

    tracer = LocalTracer()
    reporter = UsageReporter(tracer)

    trace_ids: list[str] = []
    for i in range(3):
        with tracer.span(f"turn-{i}"):
            trace_ids.append(tracer.trace_id)
            reporter.report_usage(
                "model", UsageInfo(input_tokens=i, output_tokens=0), provider_name="test"
            )

    assert trace_ids[0] not in reporter._cache  # pyright: ignore[reportPrivateUsage]
    assert trace_ids[1] in reporter._cache  # pyright: ignore[reportPrivateUsage]
    assert trace_ids[2] in reporter._cache  # pyright: ignore[reportPrivateUsage]


# --- Usage listeners (the cost-control bridge) ----------------------------------
#
# All chargeable work funnels through report_usage; a listener hook lets the
# cost-control policy observe it without the reporter knowing about policies
# (accounting stays pure) and without touching any provider adapter.


def test_that_usage_reports_notify_registered_listeners() -> None:
    tracer = LocalTracer()
    reporter = UsageReporter(tracer)
    received: list[tuple[str, str, UsageInfo]] = []

    reporter.add_listener(lambda trace_id, model, usage: received.append((trace_id, model, usage)))

    with tracer.span("turn-1"):
        trace_id = tracer.trace_id
        usage = UsageInfo(input_tokens=100, output_tokens=10)
        reporter.report_usage("some-model", usage, provider_name="some-provider")

    assert received == [(trace_id, "some-provider:some-model", usage)]


def test_that_a_failing_listener_does_not_break_usage_accounting() -> None:
    tracer = LocalTracer()
    reporter = UsageReporter(tracer)

    def broken_listener(trace_id: str, model: str, usage: UsageInfo) -> None:
        raise RuntimeError("listener boom")

    reporter.add_listener(broken_listener)

    with tracer.span("turn-1"):
        reporter.report_usage(
            "some-model", UsageInfo(input_tokens=100, output_tokens=10), provider_name="p"
        )

        # Accounting must be unaffected by the listener failure.
        assert reporter.get_usage()["p:some-model"].input_tokens == 100


# --- Provider-qualified usage keys ---------------------------------------------
#
# The same model id can be served by several providers at different rates, so
# usage is keyed by "provider:model" — the same spelling as a model route.


def test_that_usage_is_keyed_by_provider_and_model() -> None:
    tracer = LocalTracer()
    reporter = UsageReporter(tracer)

    with tracer.span("turn-1"):
        reporter.report_usage(
            "claude-sonnet-5-5",
            UsageInfo(input_tokens=10, output_tokens=2),
            provider_name="anthropic",
        )

        assert list(reporter.get_usage()) == ["anthropic:claude-sonnet-5-5"]


def test_that_the_same_model_reported_by_two_providers_is_accounted_separately() -> None:
    tracer = LocalTracer()
    reporter = UsageReporter(tracer)

    with tracer.span("turn-1"):
        reporter.report_usage(
            "gpt-5.5", UsageInfo(input_tokens=10, output_tokens=1), provider_name="openai"
        )
        reporter.report_usage(
            "gpt-5.5", UsageInfo(input_tokens=20, output_tokens=2), provider_name="azure-eu"
        )

        usage = reporter.get_usage()

    assert usage["openai:gpt-5.5"] == UsageInfo(input_tokens=10, output_tokens=1)
    assert usage["azure-eu:gpt-5.5"] == UsageInfo(input_tokens=20, output_tokens=2)
