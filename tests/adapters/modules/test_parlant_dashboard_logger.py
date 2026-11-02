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

"""The Dashboard logger must keep its records off the console.

It sits inside the CompositeLogger next to the console FileLogger; if both
wrap the same underlying stdlib logger ("parlant"), every message prints
twice — once console-rendered by the file logger, once as a bare prefix-less
duplicate when the Dashboard logger's structured record hits the console handler.
"""

import logging

from parlant.adapters.modules.parlant_dashboard.logger import ParlantDashboardLogger
from parlant.core.tracer import LocalTracer


def test_that_the_dashboard_logger_does_not_share_the_console_logger_channel() -> None:
    dashboard_logger = ParlantDashboardLogger(tracer=LocalTracer())

    assert dashboard_logger.raw_logger is not logging.getLogger("parlant")
    assert dashboard_logger.raw_logger.propagate is False


def test_that_dashboard_logger_records_never_reach_the_console_channel() -> None:
    class _CapturingHandler(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.records: list[logging.LogRecord] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.records.append(record)

    console_channel = logging.getLogger("parlant")
    capture = _CapturingHandler()
    console_channel.addHandler(capture)
    try:
        dashboard_logger = ParlantDashboardLogger(tracer=LocalTracer())
        dashboard_logger.info("dashboard-only-sentinel")
    finally:
        console_channel.removeHandler(capture)

    assert not any("dashboard-only-sentinel" in record.getMessage() for record in capture.records)
