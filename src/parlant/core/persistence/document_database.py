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

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    Mapping,
    NewType,
    Optional,
    Sequence,
    TypeVar,
    TypedDict,
    Union,
    cast,
    get_type_hints,
)

from parlant.core.common import Version


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


class BaseDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String


TDocument = TypeVar("TDocument", bound=BaseDocument)


def ensure_is_total(document: Mapping[str, Any], schema: type[Mapping[str, Any]]) -> None:
    type_hints = get_type_hints(schema)

    if list(document.keys()) != list(type_hints):
        raise TypeError(f"Provided TypedDict {schema.__qualname__} is missing required keys")


class DocumentDatabase(ABC):
    @abstractmethod
    async def create_collection(
        self,
        name: str,
        schema: type[TDocument],
    ) -> DocumentCollection[TDocument]:
        """
        Creates a new collection with the given name and returns the collection.
        """
        ...

    @abstractmethod
    async def get_collection(
        self,
        name: str,
    ) -> DocumentCollection[TDocument]:
        """
        Retrieves an existing collection by its name.
        """
        ...

    @abstractmethod
    async def get_or_create_collection(
        self,
        name: str,
        schema: type[TDocument],
    ) -> DocumentCollection[TDocument]:
        """
        Retrieves an existing collection by its name or creates a new one if it does not exist.
        """
        ...

    @abstractmethod
    async def delete_collection(
        self,
        name: str,
    ) -> None:
        """
        Deletes a collection by its name.
        """
        ...


@dataclass(frozen=True)
class InsertResult:
    acknowledged: bool


@dataclass(frozen=True)
class UpdateResult(Generic[TDocument]):
    acknowledged: bool
    matched_count: int
    modified_count: int
    updated_document: Optional[TDocument]


@dataclass(frozen=True)
class DeleteResult(Generic[TDocument]):
    acknowledged: bool
    deleted_count: int
    deleted_document: Optional[TDocument]


class DocumentCollection(ABC, Generic[TDocument]):
    @abstractmethod
    async def find(
        self,
        filters: Where,
    ) -> Sequence[TDocument]:
        """Finds all documents that match the given filters."""
        ...

    @abstractmethod
    async def find_one(
        self,
        filters: Where,
    ) -> Optional[TDocument]:
        """Returns the first document that matches the query criteria."""
        ...

    @abstractmethod
    async def insert_one(
        self,
        document: TDocument,
    ) -> InsertResult:
        """Inserts a single document into the collection."""
        ...

    @abstractmethod
    async def update_one(
        self,
        filters: Where,
        params: TDocument,
        upsert: bool = False,
    ) -> UpdateResult[TDocument]:
        """Updates the first document that matches the query criteria. If upsert is True,
        inserts the document if it does not exist."""
        ...

    @abstractmethod
    async def delete_one(
        self,
        filters: Where,
    ) -> DeleteResult[TDocument]:
        """Deletes the first document that matches the query criteria."""
        ...
