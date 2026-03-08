from collections.abc import Mapping
from typing import Any

from lagom import Container
import pytest

from parlant.core.capabilities import CapabilityStore
from parlant.core.nlp.embedding import EmbeddingResult


def _stub_embedder(store: CapabilityStore) -> None:
    dimensions = store._vector_collection._embedder.dimensions  # type: ignore[attr-defined]

    async def embed(
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        return EmbeddingResult(
            vectors=[
                [float((len(text) + i) % 13) for i in range(dimensions)]
                for text in texts
            ]
        )

    store._vector_collection._embedder.embed = embed  # type: ignore[attr-defined, method-assign]


@pytest.mark.asyncio
async def test_that_updating_a_capability_replaces_its_vector_documents(
    container: Container,
) -> None:
    store = container[CapabilityStore]
    _stub_embedder(store)

    capability = await store.create_capability(
        title="Phone replacement",
        description="Provide a loaner phone.",
        signals=["broken phone", "replacement device"],
    )

    original_vector_docs = await store._vector_collection.find(  # type: ignore[attr-defined]
        filters={"capability_id": {"$eq": capability.id}}
    )
    assert {doc["content"] for doc in original_vector_docs} == {
        "Phone replacement: Provide a loaner phone.",
        "broken phone",
        "replacement device",
    }

    await store.update_capability(
        capability.id,
        {
            "title": "Tablet replacement",
            "description": "Provide a loaner tablet.",
            "signals": ["broken tablet"],
        },
    )

    updated_vector_docs = await store._vector_collection.find(  # type: ignore[attr-defined]
        filters={"capability_id": {"$eq": capability.id}}
    )
    assert {doc["content"] for doc in updated_vector_docs} == {
        "Tablet replacement: Provide a loaner tablet.",
        "broken tablet",
    }


@pytest.mark.asyncio
async def test_that_deleting_a_capability_removes_its_vector_documents(
    container: Container,
) -> None:
    store = container[CapabilityStore]
    _stub_embedder(store)

    capability = await store.create_capability(
        title="FAQ",
        description="Answer common questions.",
        signals=["faq"],
    )

    await store.delete_capability(capability.id)

    vector_docs = await store._vector_collection.find(  # type: ignore[attr-defined]
        filters={"capability_id": {"$eq": capability.id}}
    )
    assert vector_docs == []
