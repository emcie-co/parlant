from emcie.server.core.persistence.document_database import _matches_filters


def test_equal_to() -> None:
    field_filters = {"age": {"$eq": 30}}
    candidate = {"age": 30}
    assert _matches_filters(field_filters, candidate)


def test_not_equal_to() -> None:
    field_filters = {"age": {"$ne": 40}}
    candidate = {"age": 30}
    assert _matches_filters(field_filters, candidate)


def test_greater_than_true() -> None:
    field_filters = {"age": {"$gt": 25}}
    candidate = {"age": 30}
    assert _matches_filters(field_filters, candidate)


def test_greater_than_false() -> None:
    field_filters = {"age": {"$gt": 35}}
    candidate = {"age": 30}
    assert not _matches_filters(field_filters, candidate)


def test_greater_than_or_equal_to_true() -> None:
    candidate = {"age": 30}

    field_filters = {"age": {"$gte": 30}}
    assert _matches_filters(field_filters, candidate)

    field_filters = {"age": {"$gte": 29}}
    assert _matches_filters(field_filters, candidate)


def test_greater_than_or_equal_to_false() -> None:
    candidate = {"age": 30}

    field_filters = {"age": {"$gte": 31}}
    assert not _matches_filters(field_filters, candidate)


def test_less_than_true() -> None:
    field_filters = {"age": {"$lt": 35}}
    candidate = {"age": 30}
    assert _matches_filters(field_filters, candidate)


def test_less_than_false() -> None:
    field_filters = {"age": {"$lt": 25}}
    candidate = {"age": 30}
    assert not _matches_filters(field_filters, candidate)


def test_less_than_or_equal_to_true() -> None:
    candidate = {"age": 30}

    field_filters = {"age": {"$lte": 30}}
    assert _matches_filters(field_filters, candidate)

    field_filters = {"age": {"$lte": 31}}
    assert _matches_filters(field_filters, candidate)


def test_less_than_or_equal_to_false() -> None:
    field_filters = {"age": {"$lte": 29}}
    candidate = {"age": 30}
    assert not _matches_filters(field_filters, candidate)
