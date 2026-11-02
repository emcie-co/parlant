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

from parlant.core.agents import Effort
from parlant.core.engines.compass.optimization_policy import CompassOptimizationPolicy


@pytest.mark.parametrize(
    ("k", "expected"),
    [
        (0, 0),
        (1, 1_000),
        (2, 3_000),
        (8, 3_000),
    ],
)
def test_that_process_delays_increase_for_consecutive_customer_messages(
    k: int,
    expected: int,
) -> None:
    assert CompassOptimizationPolicy().get_process_delay(k) == expected


@pytest.mark.parametrize(
    ("route", "qualifiers", "expected"),
    [
        ("matching.rank", None, 5.0),
        ("react", None, 10.0),
        ("review.tools", None, 10.0),
        ("review.message", {"effort": Effort.LOW.value}, 10.0),
        ("review.tools", {"effort": Effort.HIGH.value}, 15.0),
        ("review.message", {"effort": Effort.MAX.value}, 15.0),
        ("compaction", None, None),
    ],
)
def test_that_hedge_timeouts_are_selected_by_route_and_qualifiers(
    route: str,
    qualifiers: dict[str, str] | None,
    expected: float | None,
) -> None:
    assert CompassOptimizationPolicy().get_hedge_timeout(route, qualifiers=qualifiers) == expected
