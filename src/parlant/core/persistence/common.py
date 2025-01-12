# Copyright 2024 Emcie Co Ltd.
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

from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping, NewType, Union, cast, get_type_hints
from typing_extensions import Literal, TypedDict

from parlant.core.common import SchemaVersion


class VersionedDatabase(ABC):
    @property
    @abstractmethod
    def version(self) -> SchemaVersion:
        """Returns the schema version of this database."""
        ...

    @version.setter
    @abstractmethod
    def version(self, value: SchemaVersion) -> None:
        """Sets the schema version of this database."""
        ...


class VersionedStore(ABC):
    VERSION: SchemaVersion

    @property
    @abstractmethod
    def database_version(self) -> SchemaVersion:
        """Returns the database of this store."""
        ...

    @property
    @abstractmethod
    def database_name(self) -> str:
        """Returns the name of this store's database."""
        ...


ObjectId = NewType("ObjectId", str)

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

WhereExpression = dict[FieldName, WhereOperator]

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
                if not _evaluate_filter(
                    operator, candidate[field_name], cast(LiteralValue, filter_value)
                ):
                    return False

    return True


def ensure_is_total(document: Mapping[str, Any], schema: type[Mapping[str, Any]]) -> None:
    type_hints = get_type_hints(schema)

    if list(document.keys()) != list(type_hints):
        raise TypeError(f"Provided TypedDict {schema.__qualname__} is missing required keys")
