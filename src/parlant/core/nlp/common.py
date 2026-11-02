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

from enum import IntEnum
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
import json
from typing import Mapping
from typing_extensions import Literal, TypeAlias


class ModelSize(IntEnum):
    SMALL = 0
    MEDIUM = 1
    LARGE = 2
    AUTO = 99


ModelGeneration: TypeAlias = Literal["auto", "stable", "latest"]

ModelType: TypeAlias = Literal["auto", "standard", "reasoning"]

REPORTED_COST_EXTRA_KEY = "reported_cost"


def parse_reported_cost(value: object) -> Decimal | None:
    """Parse a provider-reported monetary amount without float arithmetic."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        return None
    try:
        cost = Decimal(str(value))
    except InvalidOperation:
        return None
    if not cost.is_finite() or cost < 0:
        return None
    return cost


def format_reported_cost(value: object) -> str | None:
    """Return a valid provider cost in fixed-point notation, never exponent form."""
    cost = parse_reported_cost(value)
    return format(cost, "f") if cost is not None else None


def add_usage_extra_value(
    key: str,
    left: int | float | str | None,
    right: int | float | str,
) -> int | float | str:
    """Aggregate one usage meter, preserving costs as exact decimal strings."""
    if key == REPORTED_COST_EXTRA_KEY:
        right_cost = parse_reported_cost(right)
        left_cost = parse_reported_cost(left)
        if right_cost is not None:
            return format((left_cost or Decimal(0)) + right_cost, "f")

    if (
        not isinstance(left, bool)
        and isinstance(left, (int, float))
        and not isinstance(right, bool)
        and isinstance(right, (int, float))
    ):
        return left + right
    return right


@dataclass(frozen=True)
class UsageInfo:
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    extra: Mapping[str, int | float | str] = field(default_factory=dict)
    requests: int = 1

    def __repr__(self) -> str:
        return json.dumps(asdict(self), indent=2)
