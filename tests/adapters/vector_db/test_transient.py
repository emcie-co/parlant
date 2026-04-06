from parlant.adapters.vector_db.transient import TransientVectorCollection


def test_that_negative_cosine_similarity_is_not_treated_as_close_distance() -> None:
    assert TransientVectorCollection._distance_from_similarity(1.0) == 0.0
    assert TransientVectorCollection._distance_from_similarity(0.0) == 1.0
    assert TransientVectorCollection._distance_from_similarity(-1.0) == 2.0
