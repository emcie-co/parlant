# Copyright (c) 2024 Emcie
# All rights reserved.
#
# This file and its contents are the property of Emcie and are strictly confidential.
# No part of this file may be reproduced, distributed, or transmitted in any form or by any means,
# including photocopying, recording, or other electronic or mechanical methods,
# without the prior written permission of Emcie.
#
# Website: https://emcie.co

from typing import Annotated, Iterable, List, Optional
from fastapi import APIRouter, Query
from pydantic import BaseModel
from emcie.server.rag import RagStore


class DocumentDTO(BaseModel):
    id: str
    metadata: Optional[dict] = None
    content: str


class UpsertDocumentRequest(BaseModel):
    metadata: Optional[dict] = None
    content: str


class UpsertDocumentResponse(BaseModel):
    id: str
    metadata: Optional[dict] = None
    content: str


class ListDocumentsResponse(BaseModel):
    documents: List[DocumentDTO]


def create_router(
    rag_store: RagStore,
) -> APIRouter:
    router = APIRouter()

    @router.put("/{document_id}")
    async def upsert(
        document_id: str,
        request: UpsertDocumentRequest,
    ) -> UpsertDocumentResponse:
        response = await rag_store.upsert(
            document={
                "id": document_id,
                "metadata": request.metadata,
                "document": request.content,
            },
        )

        return UpsertDocumentResponse(
            id=response["id"],
            metadata=response["metadata"],
            content=response["document"],
        )

    @router.get("/")
    async def query(
        rag_query: Annotated[str | None, Query(alias="query")] = None
    ) -> ListDocumentsResponse:
        if rag_query:
            documents = await rag_store.query(rag_query)
        else:
            documents = rag_store.get_all_documents()

        return ListDocumentsResponse(
            documents=[
                DocumentDTO(id=doc["id"], metadata=doc["metadata"], content=doc["document"])
                for doc in documents
            ]
        )

    return router
