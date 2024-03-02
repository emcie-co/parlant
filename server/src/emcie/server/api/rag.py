from typing import Annotated, Iterable, List, Optional
from fastapi import APIRouter, Query
from pydantic import BaseModel
from emcie.server.rag import RagDocument, RagStore


class RagDocumentRequest(BaseModel):
    id: str
    metadata: Optional[dict] = None
    document: str


class RagDocumentResponse(BaseModel):
    id: str
    metadata: Optional[dict] = None
    document: str


def create_router(
    rag_store: RagStore,
) -> APIRouter:
    router = APIRouter()

    @router.post("/")
    async def upsert(
        request: RagDocumentRequest,
    ) -> RagDocumentResponse:
        response = await rag_store.upsert(
            document={
                "id": request.id,
                "metadata": request.metadata,
                "document": request.document,
            },
        )
        return RagDocumentResponse(**response)

    @router.get("/")
    async def query(
        rag_query: Annotated[str | None, Query(alias="query")] = None
    ) -> Iterable[RagDocumentResponse]:
        if rag_query:
            documents = await rag_store.query(rag_query)
        else:
            documents = rag_store.get_all_documents()
        return [RagDocumentResponse(**doc) for doc in documents]

    return router
