# Copyright 2025 Emcie Co Ltd.
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

from typing import Any, Callable, Mapping, NewType, Protocol, Union, cast, get_type_hints
from typing_extensions import Literal, TypedDict

from parlant.core.common import Version


ObjectId = NewType("ObjectId", str)


class MigrationRequired(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class ServerOutdated(Exception):
    def __init__(self, message: str | None = None):
        super().__init__(message)


class VersionedStore(Protocol):
    VERSION: Version


# Metadata Query Grammar
LiteralValue = Union[str, int, float, bool]

FieldName = str

WhereOperator = TypedDict(
    "WhereOperator",
    {
        "$gt": LiteralValue,
        "$gte": LiteralValue,
        "$lt": LiteralValue,
        "$lte": LiteralValue,
        "$ne": LiteralValue,
        "$eq": LiteralValue,
    },
    total=False,
)

InclusionExclusionOperator = TypedDict(
    "InclusionExclusionOperator",
    {
        "$in": list[LiteralValue],
        "$nin": list[LiteralValue],
    },
    total=False,
)

WhereExpression = dict[FieldName, Union[WhereOperator, InclusionExclusionOperator]]

LogicalOperator = TypedDict(
    "LogicalOperator",
    {
        "$and": list[Union[WhereExpression, "LogicalOperator"]],
        "$or": list[Union[WhereExpression, "LogicalOperator"]],
    },
    total=False,
)

Where = Union[WhereExpression, LogicalOperator]


def _evaluate_filter(
    operator: str,
    field_value: LiteralValue,
    filter_value: LiteralValue,
) -> bool:
    tests: dict[str, Callable[[Any, Any], bool]] = {
        "$eq": lambda field_value, filter_value: field_value == filter_value,
        "$ne": lambda field_value, filter_value: field_value != filter_value,
        "$gt": lambda field_value, filter_value: field_value > filter_value,
        "$gte": lambda field_value, filter_value: field_value >= filter_value,
        "$lt": lambda field_value, filter_value: field_value < filter_value,
        "$lte": lambda field_value, filter_value: field_value <= filter_value,
    }

    return tests[operator](field_value, filter_value)


def matches_filters(
    where: Where,
    candidate: Mapping[str, Any],
) -> bool:
    if not where:
        return True

    if next(iter(where.keys())) in ("$and", "$or"):
        op = cast(LogicalOperator, where)
        for operator in op:
            operands: list[Union[WhereExpression, LogicalOperator]] = op[
                cast(Literal["$and", "$or"], operator)
            ]
            if operator == "$and":
                if not all(matches_filters(sub_filter, candidate) for sub_filter in operands):
                    return False
            elif operator == "$or":
                if not any(matches_filters(sub_filter, candidate) for sub_filter in operands):
                    return False

    else:
        field_filters = cast(WhereExpression, where)
        for field_name, field_filter in field_filters.items():
            for operator, filter_value in field_filter.items():
                if operator == "$in":
                    if not any(
                        candidate[field_name] == val
                        for val in cast(list[LiteralValue], filter_value)
                    ):
                        return False
                elif operator == "$nin":
                    if any(
                        candidate[field_name] == val
                        for val in cast(list[LiteralValue], filter_value)
                    ):
                        return False
                else:
                    if not _evaluate_filter(
                        operator, candidate[field_name], cast(LiteralValue, filter_value)
                    ):
                        return False

    return True


def ensure_is_total(document: Mapping[str, Any], schema: type[Mapping[str, Any]]) -> None:
    required_keys = get_type_hints(schema).keys()
    missing_keys = [key for key in required_keys if key not in document]

    if missing_keys:
        raise TypeError(
            f"Provided TypedDict '{schema.__qualname__}' is missing required keys: {missing_keys}. "
            f"Expected at least the keys: {list(required_keys)}."
        )
