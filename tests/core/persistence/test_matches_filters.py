import typing
from emcie.server.core.persistence.document_database import Where, matches_filters


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
