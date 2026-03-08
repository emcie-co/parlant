from typing import cast

import pytest
from lagom import Container

from parlant.core.canned_responses import (
    CannedResponseStore,
    CannedResponseVectorDocument,
    CannedResponseVectorStore,
)
from parlant.core.persistence.vector_database import InsertResult


@pytest.mark.asyncio
async def test_that_canned_response_vector_documents_use_unique_ids_for_each_content(
    container: Container,
) -> None:
    store = cast(CannedResponseVectorStore, container[CannedResponseStore])
    inserted_documents: list[CannedResponseVectorDocument] = []

    async def capture_insert(document: CannedResponseVectorDocument) -> InsertResult:
        inserted_documents.append(document)
        return InsertResult(acknowledged=True)

    store._canreps_vector_collection.insert_one = capture_insert  # type: ignore[method-assign]

    await store.create_canned_response(
        value="Payment failed.",
        signals=["card declined", "card declined", "Payment failed."],
    )

    assert len(inserted_documents) == 4
    assert len({document["id"] for document in inserted_documents}) == 4
