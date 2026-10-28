from typing import AsyncIterator
from unittest.mock import AsyncMock

import pytest

import parlant.sdk as p
from parlant.adapters.db.transient import TransientDocumentDatabase
from parlant.adapters.vector_db.transient import TransientVectorDatabase
from parlant.core import metadata
from parlant.core.common import IdGenerator
from parlant.core.container import Container
from parlant.core.loggers import Logger
from parlant.core.nlp.embedding import BasicEmbeddingCache
from parlant.core.nlp.tokenization import ZeroEstimatingTokenizer
from parlant.core.rules import CompositeRuleStore, RuleStore, RuleVectorStore
from parlant.core.store_provider import BasicStoreProvider
from parlant.core.tracer import Tracer
from tests.test_utilities import DeterministicEmbedder


class _WordTokenizer(ZeroEstimatingTokenizer):
    async def estimate_token_count(self, prompt: str) -> int:
        return len(prompt.split())


@pytest.fixture
async def indexing_store(logger: Logger, tracer: Tracer) -> AsyncIterator[RuleVectorStore]:
    async def embedder() -> DeterministicEmbedder:
        result = DeterministicEmbedder()
        result._tokenizer = _WordTokenizer()
        return result

    cache = BasicEmbeddingCache(TransientDocumentDatabase())
    async with RuleVectorStore(
        IdGenerator(),
        TransientVectorDatabase(logger, tracer, lambda: cache),
        TransientDocumentDatabase(),
        embedder,
    ) as store:
        yield store


def _server(store: RuleStore) -> p.Server:
    container = Container()
    container[RuleStore] = store
    server = object.__new__(p.Server)
    server._store_provider = BasicStoreProvider(lambda: container)
    return server


@pytest.mark.parametrize("title", [None, "Authored title"])
@pytest.mark.parametrize("references", [True, False, None])
async def test_enrichment_preserves_authored_data_and_merges_generated_fields(
    indexing_store: RuleVectorStore, title: str | None, references: bool | None
) -> None:
    original_metadata = metadata.set_internal_value(
        {"custom": "keep"}, ("evaluation_method",), "custom"
    )
    rule = await indexing_store.create_rule(
        condition="the customer wants a refund",
        title=title,
        signals=[" Money back ", "money BACK", "[__pin__] Refund now"],
        anti_signals=[" Shipping ", "shipping", " "],
        metadata=original_metadata,
    )
    other = await indexing_store.create_rule(content="Unrelated rule")
    other_vectors = await indexing_store._vector_collection.find({"rule_id": {"$eq": other.id}})
    result = p._CachedIndexer.RuleIndexingResult(
        references_other_policies=references,
        signals=["MONEY back", " reimbursement ", ""],
        anti_signals=["SHIPPING", " opening hours ", ""],
        title="Generated title",
    )

    server = _server(indexing_store)
    await server._apply_single_indexing_result(rule.id, result)
    # Reapplying cached enrichment must not accumulate signals or overwrite a title.
    await server._apply_single_indexing_result(rule.id, result)

    updated = await indexing_store.read_rule(rule.id)
    assert updated.title == (title or "Generated title")
    assert updated.condition == rule.condition
    assert updated.signals == ["Money back", "[__pin__] Refund now", "reimbursement"]
    assert updated.anti_signals == ["Shipping", "opening hours"]
    assert updated.metadata["custom"] == "keep"
    assert metadata.internal_value(updated.metadata, ("evaluation_method",)) == "custom"
    assert metadata.internal_value(updated.metadata, ("references_other_policies",)) is references
    if references is None:
        assert updated.metadata == original_metadata

    vectors = await indexing_store._vector_collection.find({"rule_id": {"$eq": rule.id}})
    assert sorted(v["content"] for v in vectors) == sorted(
        [indexing_store._rule_embedding_content(updated), *updated.signals]
    )
    assert await indexing_store.read_rule(other.id) == other
    assert (
        await indexing_store._vector_collection.find({"rule_id": {"$eq": other.id}})
        == other_vectors
    )
    hits = await indexing_store.find_relevant_rules("reimbursement", [updated, other], 1)
    assert [hit.rule.id for hit in hits] == [rule.id]


async def test_empty_enrichment_leaves_existing_rule_and_vectors_untouched(
    indexing_store: RuleVectorStore,
) -> None:
    rule = await indexing_store.create_rule(
        content="Refund purchases",
        title="Authored",
        signals=[" Keep spaces "],
        anti_signals=[" Shipping "],
        metadata={"custom": "keep"},
    )
    vectors = await indexing_store._vector_collection.find({"rule_id": {"$eq": rule.id}})

    await _server(indexing_store)._apply_single_indexing_result(
        rule.id, p._CachedIndexer.RuleIndexingResult(None, [], [], None)
    )

    assert await indexing_store.read_rule(rule.id) == rule
    assert await indexing_store._vector_collection.find({"rule_id": {"$eq": rule.id}}) == vectors


async def test_combined_rule_update_replaces_only_positive_vectors(
    indexing_store: RuleVectorStore,
) -> None:
    rule = await indexing_store.create_rule(content="Refund purchases", signals=["old signal"])
    updated = await indexing_store.update_rule(
        rule.id,
        {"title": "Refunds", "signals": ["money back"], "anti_signals": ["shipping"]},
    )
    vectors = await indexing_store._vector_collection.find({"rule_id": {"$eq": rule.id}})
    assert sorted(v["content"] for v in vectors) == ["# Refunds\n\nRefund purchases", "money back"]
    assert updated.anti_signals == ["shipping"]

    await indexing_store.update_rule(rule.id, {"anti_signals": ["opening hours"]})
    assert await indexing_store._vector_collection.find({"rule_id": {"$eq": rule.id}}) == vectors


async def test_rule_update_reuses_individually_cached_embeddings(
    indexing_store: RuleVectorStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Another rule warms the cache for the final text and signals; cache keys
    # must be independent of rule IDs and of how the update is batched.
    await indexing_store.create_rule(
        content="Refund purchases", title="Refunds", signals=["money back", "reimbursement"]
    )
    rule = await indexing_store.create_rule(content="Refund purchases")
    embed = AsyncMock(side_effect=AssertionError("A warm cache must not call the embedder"))
    monkeypatch.setattr(indexing_store._embedder, "embed", embed)

    await indexing_store.update_rule(
        rule.id, {"title": "Refunds", "signals": ["money back", "reimbursement"]}
    )

    embed.assert_not_awaited()
    vectors = await indexing_store._vector_collection.find({"rule_id": {"$eq": rule.id}})
    assert len(vectors) == 3


async def test_enrichment_updates_once_and_reapplying_cached_results_does_not_write(
    indexing_store: RuleVectorStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rule = await indexing_store.create_rule(content="Refund purchases")
    update = AsyncMock(wraps=indexing_store.update_rule)
    set_metadata = AsyncMock(wraps=indexing_store.set_metadata)
    monkeypatch.setattr(indexing_store, "update_rule", update)
    monkeypatch.setattr(indexing_store, "set_metadata", set_metadata)
    # SDK application must also work through the public composite-store path.
    server = _server(CompositeRuleStore(indexing_store, []))
    result = p._CachedIndexer.RuleIndexingResult(False, ["money back"], ["shipping"], "Refunds")

    await server._apply_single_indexing_result(rule.id, result)

    update.assert_awaited_once_with(
        rule.id,
        {"signals": ["money back"], "anti_signals": ["shipping"], "title": "Refunds"},
    )
    set_metadata.assert_awaited_once()
    updated = await indexing_store.read_rule(rule.id)
    update.reset_mock()
    set_metadata.reset_mock()

    await server._apply_single_indexing_result(rule.id, result)

    update.assert_not_awaited()
    set_metadata.assert_not_awaited()
    assert await indexing_store.read_rule(rule.id) == updated
