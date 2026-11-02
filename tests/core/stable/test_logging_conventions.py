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

"""The logging convention: application code logs through the container's
``Logger``, never through Python's builtin ``logging`` module directly.

Two exemptions, both structural:

- ``core/loggers.py`` — the adapter OVER builtin logging; it must touch it.
- ``adapters/vector_db/transient.py`` — monkey-patches ``logging.getLogger``
  around importing ``nano_vectordb``, which hijacks global logging config at
  import time. That guard protects us FROM builtin-logging abuse.
"""

import re

import pytest
from pathlib import Path

_SRC_ROOT = Path(__file__).parent.parent.parent.parent / "src" / "parlant"

_EXEMPT = {
    _SRC_ROOT / "core" / "loggers.py",
    _SRC_ROOT / "adapters" / "vector_db" / "transient.py",
}

_GET_LOGGER = re.compile(r"logging\.getLogger")


def test_that_no_module_uses_the_builtin_logger_directly() -> None:
    offenders = sorted(
        str(path.relative_to(_SRC_ROOT))
        for path in _SRC_ROOT.rglob("*.py")
        if path not in _EXEMPT and _GET_LOGGER.search(path.read_text())
    )

    assert not offenders, (
        "These modules use Python's builtin logging instead of the container's "
        f"Logger: {', '.join(offenders)}. Thread a parlant.core.loggers.Logger "
        "through their constructors instead."
    )


def test_that_a_scoped_logger_prefixes_every_call_with_its_scope() -> None:
    from parlant.core.loggers import ScopedLogger, StdoutLogger
    from parlant.core.tracer import LocalTracer

    class _RecordingLogger(StdoutLogger):
        def __init__(self) -> None:
            super().__init__(LocalTracer())
            self.records: list[str] = []

        def debug(self, message: str) -> None:
            self.records.append(self._add_trace_id_and_scopes(message))

    inner = _RecordingLogger()
    logger = ScopedLogger(inner, "Component")

    logger.debug("plain call")
    assert "[Component] plain call" in inner.records[-1]

    with logger.scope("Inner"):
        logger.debug("nested call")
    assert "[Component][Inner] nested call" in inner.records[-1]

    # The wrapper leaks no scope outside its own calls.
    inner.debug("direct call")
    assert "[Component]" not in inner.records[-1]


def test_that_trace_emits_only_when_the_level_is_trace() -> None:
    from typing import Any
    from parlant.core.loggers import LogLevel, StdoutLogger
    from parlant.core.tracer import LocalTracer

    class _RecordingLogger(StdoutLogger):
        def __init__(self, level: LogLevel) -> None:
            super().__init__(LocalTracer(), log_level=level)
            self.records: list[str] = []
            sink = self

            class _Sink:
                def debug(self, message: str, **kwargs: Any) -> None:
                    sink.records.append(message)

            self._logger = _Sink()  # type: ignore[assignment]

    # TRACE self-filters at any level above TRACE (stdlib has no TRACE severity).
    silent = _RecordingLogger(LogLevel.DEBUG)
    silent.trace("payload")
    assert silent.records == []

    verbose = _RecordingLogger(LogLevel.TRACE)
    verbose.trace("payload")
    assert len(verbose.records) == 1
    assert verbose.records[0].startswith("TRACE ")


async def test_that_the_server_container_tracer_is_the_logger_tracer(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    """The stdout/file LOGGER prefixes lines with its tracer's trace id; the
    engine opens spans on the container tracer. They must be the same instance,
    or terminal and file logs never carry a trace id."""
    import parlant.bin.server as server_module
    from parlant.core.container import Container
    from parlant.core.tracer import Tracer

    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)

    container = Container()
    await server_module._define_tracer(container)

    assert container[Tracer] is server_module.TRACER


def test_that_log_lines_render_the_trace_id_abbreviated_to_its_last_ten_chars() -> None:
    from parlant.core.loggers import StdoutLogger
    from parlant.core.tracer import LocalTracer

    tracer = LocalTracer()

    class _RecordingLogger(StdoutLogger):
        def __init__(self) -> None:
            super().__init__(tracer)
            self.records: list[str] = []

        def debug(self, message: str) -> None:
            self.records.append(self._add_trace_id_and_scopes(message))

    logger = _RecordingLogger()

    with tracer.span("test-span"):
        full_id = tracer.trace_id
        logger.debug("hello")

    assert len(full_id) == 32
    # The rightmost bits are the W3C-guaranteed-random part, so the tail is
    # the safe abbreviation; the full id stays available on the tracer.
    assert logger.records[-1].startswith(f"[{full_id[-10:]}]")

    logger.debug("outside any span")
    assert logger.records[-1].startswith("[<main>]")
