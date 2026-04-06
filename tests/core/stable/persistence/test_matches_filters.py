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

import typing
from parlant.core.persistence.common import Where, matches_filters


def test_equal_to() -> None:
    field_filters = typing.cast(Where, {"age": {"$eq": 30}})
    candidate = {"age": 30}
    assert matches_filters(field_filters, candidate)


def test_not_equal_to() -> None:
    field_filters: Where = {"age": {"$ne": 40}}
    candidate = {"age": 30}
    assert matches_filters(field_filters, candidate)


def test_greater_than_true() -> None:
    field_filters: Where = {"age": {"$gt": 25}}
    candidate = {"age": 30}
    assert matches_filters(field_filters, candidate)


def test_greater_than_false() -> None:
    field_filters: Where = {"age": {"$gt": 35}}
    candidate = {"age": 30}
    assert not matches_filters(field_filters, candidate)


def test_greater_than_or_equal_to_true() -> None:
    candidate = {"age": 30}

    field_filters: Where = {"age": {"$gte": 30}}
    assert matches_filters(field_filters, candidate)

    field_filters = {"age": {"$gte": 29}}
    assert matches_filters(field_filters, candidate)


def test_greater_than_or_equal_to_false() -> None:
    candidate = {"age": 30}

    field_filters: Where = {"age": {"$gte": 31}}
    assert not matches_filters(field_filters, candidate)


def test_less_than_true() -> None:
    field_filters: Where = {"age": {"$lt": 35}}
    candidate = {"age": 30}
    assert matches_filters(field_filters, candidate)


def test_less_than_false() -> None:
    field_filters: Where = {"age": {"$lt": 25}}
    candidate = {"age": 30}
    assert not matches_filters(field_filters, candidate)


def test_less_than_or_equal_to_true() -> None:
    candidate = {"age": 30}

    field_filters: Where = {"age": {"$lte": 30}}
    assert matches_filters(field_filters, candidate)

    field_filters = {"age": {"$lte": 31}}
    assert matches_filters(field_filters, candidate)


def test_less_than_or_equal_to_false() -> None:
    field_filters: Where = {"age": {"$lte": 29}}
    candidate = {"age": 30}
    assert not matches_filters(field_filters, candidate)


def test_and_operator_all_true() -> None:
    field_filters: Where = {"$and": [{"age": {"$gte": 25}}, {"age": {"$lt": 35}}]}
    candidate = {"age": 30}
    assert matches_filters(field_filters, candidate)


def test_and_operator_one_false() -> None:
    field_filters: Where = {"$and": [{"age": {"$gte": 25}}, {"age": {"$lt": 30}}]}
    candidate = {"age": 30}
    assert not matches_filters(field_filters, candidate)


def test_and_operator_all_false() -> None:
    field_filters: Where = {"$and": [{"age": {"$gte": 35}}, {"age": {"$lt": 25}}]}
    candidate = {"age": 30}
    assert not matches_filters(field_filters, candidate)


def test_or_operator_one_true() -> None:
    field_filters: Where = {"$or": [{"age": {"$gte": 35}}, {"age": {"$lt": 35}}]}
    candidate = {"age": 30}
    assert matches_filters(field_filters, candidate)


def test_or_operator_all_true() -> None:
    field_filters: Where = {"$or": [{"age": {"$gte": 25}}, {"age": {"$lt": 35}}]}
    candidate = {"age": 30}
    assert matches_filters(field_filters, candidate)


def test_or_operator_all_false() -> None:
    field_filters: Where = {"$or": [{"age": {"$gt": 35}}, {"age": {"$lt": 25}}]}
    candidate = {"age": 30}
    assert not matches_filters(field_filters, candidate)


def test_and_or_combination() -> None:
    field_filters: Where = {
        "$and": [
            {"$or": [{"age": {"$lt": 20}}, {"age": {"$gt": 25}}]},
            {"$or": [{"age": {"$lt": 35}}, {"age": {"$gt": 40}}]},
        ]
    }
    candidate = {"age": 30}
    assert matches_filters(field_filters, candidate)


def test_nested_and_or_combination() -> None:
    field_filters: Where = {
        "$and": [
            {"$or": [{"age": {"$lt": 20}}, {"$and": [{"age": {"$gt": 25}}, {"age": {"$lt": 35}}]}]},
            {"$or": [{"age": {"$lt": 35}}, {"age": {"$gt": 40}}]},
        ]
    }
    candidate = {"age": 30}
    assert matches_filters(field_filters, candidate)


def test_deeply_nested_combination() -> None:
    field_filters: Where = {
        "$or": [
            {"$and": [{"age": {"$gt": 20}}, {"age": {"$lt": 25}}]},
            {"$or": [{"age": {"$lt": 35}}, {"$and": [{"age": {"$gt": 40}}, {"age": {"$lt": 50}}]}]},
        ]
    }
    candidate = {"age": 30}
    assert matches_filters(field_filters, candidate)


def test_in_operator() -> None:
    field_filters: Where = {"id": {"$in": ["a", "b"]}}
    candidate = {"id": "a"}
    assert matches_filters(field_filters, candidate)


def test_nin_operator() -> None:
    field_filters: Where = {"id": {"$nin": ["a", "b"]}}
    candidate = {"id": "c"}
    assert matches_filters(field_filters, candidate)


def test_in_operator_false() -> None:
    field_filters: Where = {"id": {"$in": ["a", "b"]}}
    candidate = {"id": "c"}
    assert not matches_filters(field_filters, candidate)


def test_nin_operator_false() -> None:
    field_filters: Where = {"id": {"$nin": ["a", "b"]}}
    candidate = {"id": "a"}
    assert not matches_filters(field_filters, candidate)
