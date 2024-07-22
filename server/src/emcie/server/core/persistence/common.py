from dataclasses import dataclass
from typing import Literal, NewType, Type, Union

from emcie.server.base_models import DefaultBaseModel


ObjectId = NewType("ObjectId", str)


@dataclass(frozen=True)
class CollectionDescriptor:
    name: str
    schema: Type[DefaultBaseModel]


# Metadata Query Grammar
LiteralValue = Union[str, int, float, bool]
LogicalOperator = Union[Literal["$and"], Literal["$or"]]
WhereOperator = Union[
    Literal["$gt"],
    Literal["$gte"],
    Literal["$lt"],
    Literal["$lte"],
    Literal["$ne"],
    Literal["$eq"],
]
InclusionExclusionOperator = Union[Literal["$in"], Literal["$nin"]]
OperatorExpression = Union[
    dict[Union[WhereOperator, LogicalOperator], LiteralValue],
    dict[InclusionExclusionOperator, list[LiteralValue]],
]

Where = dict[Union[str, LogicalOperator], Union[LiteralValue, OperatorExpression, list["Where"]]]
