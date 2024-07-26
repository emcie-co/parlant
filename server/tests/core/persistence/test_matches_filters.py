from emcie.server.core.persistence.common import Where, matches_filters


def test_equal_to() -> None:
    field_filters: Where = {"age": {"$eq": 30}}
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
