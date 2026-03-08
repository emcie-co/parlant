from collections.abc import Mapping
from typing import Any

from lagom import Container
import pytest

from parlant.core.journeys import JourneyStore
from parlant.core.nlp.embedding import EmbeddingResult


def _stub_embedder(store: JourneyStore) -> None:
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


async def _read_journey_vector_content(store: JourneyStore, journey_id: str) -> str:
    vector_docs = await store._vector_collection.find(  # type: ignore[attr-defined]
        filters={"journey_id": {"$eq": journey_id}}
    )
    assert len(vector_docs) == 1
    return vector_docs[0]["content"]


@pytest.mark.asyncio
async def test_that_journey_vectors_refresh_when_node_and_edge_content_changes(
    container: Container,
) -> None:
    store = container[JourneyStore]
    _stub_embedder(store)

    journey = await store.create_journey(
        title="Greeting journey",
        description="Help with greetings.",
        conditions=[],
    )

    node = await store.create_node(
        journey_id=journey.id,
        action="Ask for the customer's name",
        tools=[],
    )
    edge = await store.create_edge(
        journey_id=journey.id,
        source=journey.root_id,
        target=node.id,
        condition="customer says hello",
    )

    content_after_create = await _read_journey_vector_content(store, journey.id)
    assert "Ask for the customer's name" in content_after_create
    assert "customer says hello" in content_after_create

    await store.update_node(node.id, {"action": "Ask for the customer's preferred name"})
    await store.update_edge(edge.id, {"condition": "customer greets the agent"})

    content_after_update = await _read_journey_vector_content(store, journey.id)
    assert "Ask for the customer's preferred name" in content_after_update
    assert "Ask for the customer's name" not in content_after_update
    assert "customer greets the agent" in content_after_update
    assert "customer says hello" not in content_after_update


@pytest.mark.asyncio
async def test_that_journey_vectors_refresh_when_nodes_and_edges_are_deleted(
    container: Container,
) -> None:
    store = container[JourneyStore]
    _stub_embedder(store)

    journey = await store.create_journey(
        title="Escalation journey",
        description="Handle escalation.",
        conditions=[],
    )

    node = await store.create_node(
        journey_id=journey.id,
        action="Escalate to a specialist",
        tools=[],
    )
    edge = await store.create_edge(
        journey_id=journey.id,
        source=journey.root_id,
        target=node.id,
        condition="customer requests escalation",
    )

    await store.delete_edge(edge.id)
    content_after_edge_delete = await _read_journey_vector_content(store, journey.id)
    assert "customer requests escalation" not in content_after_edge_delete
    assert "Escalate to a specialist" in content_after_edge_delete

    await store.delete_node(node.id)
    content_after_node_delete = await _read_journey_vector_content(store, journey.id)
    assert "Escalate to a specialist" not in content_after_node_delete
