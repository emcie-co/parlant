from typing import Any, Callable, Literal, Mapping, NewType, Union


ObjectId = NewType("ObjectId", str)


# Metadata Query Grammar
LiteralValue = Union[str, int, float, bool]

WhereOperator = Union[
    Literal["$gt"],
    Literal["$gte"],
    Literal["$lt"],
    Literal["$lte"],
    Literal["$ne"],
    Literal["$eq"],
]

Where = dict[str, dict[WhereOperator, LiteralValue]]


def matches_filters(
    field_filters: Where,
    candidate: Mapping[str, Any],
) -> bool:
    tests: dict[str, Callable[[Any, Any], bool]] = {
        "$eq": lambda candidate, filter_value: candidate == filter_value,
        "$ne": lambda candidate, filter_value: candidate != filter_value,
        "$gt": lambda candidate, filter_value: candidate > filter_value,
        "$gte": lambda candidate, filter_value: candidate >= filter_value,
        "$lt": lambda candidate, filter_value: candidate < filter_value,
        "$lte": lambda candidate, filter_value: candidate <= filter_value,
    }
    for field_name, field_filter in field_filters.items():
        for filter_name, filter_value in field_filter.items():
            if not tests[filter_name](candidate.get(field_name), filter_value):
                return False
    return True
