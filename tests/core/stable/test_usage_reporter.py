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

from parlant.core import usage_reporter as usage_reporter_module
from parlant.core.nlp.common import UsageInfo
from parlant.core.tracer import LocalTracer
from parlant.core.usage_reporter import UsageReporter


def test_usage_reporter_aggregates_usage_by_model_for_current_trace() -> None:
    tracer = LocalTracer()
    reporter = UsageReporter(tracer)

    with tracer.span("turn-1"):
        reporter.report_usage(
            "test/model",
            UsageInfo(
                input_tokens=10,
                output_tokens=2,
                cached_input_tokens=3,
                extra={"reasoning_tokens": 1, "note": "first"},
            ),
        )
        reporter.report_usage(
            "test/model",
            UsageInfo(
                input_tokens=4,
                output_tokens=1,
                cached_input_tokens=2,
                extra={"reasoning_tokens": 5, "note": "second"},
            ),
        )
        reporter.report_usage(
            "test/other",
            UsageInfo(input_tokens=1, output_tokens=0),
        )

        usage = reporter.get_usage()

    assert usage["test/model"] == UsageInfo(
        input_tokens=14,
        output_tokens=3,
        cached_input_tokens=5,
        extra={"reasoning_tokens": 6, "note": "second"},
    )
    assert usage["test/other"] == UsageInfo(input_tokens=1, output_tokens=0)

    with tracer.span("turn-2"):
        assert reporter.get_usage() == {}


def test_usage_reporter_evicts_oldest_trace_usage(monkeypatch) -> None:
    monkeypatch.setattr(usage_reporter_module, "_USAGE_CACHE_MAX_SIZE", 2)

    tracer = LocalTracer()
    reporter = UsageReporter(tracer)

    trace_ids: list[str] = []
    for i in range(3):
        with tracer.span(f"turn-{i}"):
            trace_ids.append(tracer.trace_id)
            reporter.report_usage("test/model", UsageInfo(input_tokens=i, output_tokens=0))

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
        reporter.report_usage("some-model", usage)

    assert received == [(trace_id, "some-model", usage)]


def test_that_a_failing_listener_does_not_break_usage_accounting() -> None:
    tracer = LocalTracer()
    reporter = UsageReporter(tracer)

    def broken_listener(trace_id: str, model: str, usage: UsageInfo) -> None:
        raise RuntimeError("listener boom")

    reporter.add_listener(broken_listener)

    with tracer.span("turn-1"):
        reporter.report_usage("some-model", UsageInfo(input_tokens=100, output_tokens=10))

        # Accounting must be unaffected by the listener failure.
        assert reporter.get_usage()["some-model"].input_tokens == 100
