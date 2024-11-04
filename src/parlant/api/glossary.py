from fastapi import APIRouter, status
from typing import Optional

from parlant.core.agents import AgentId
from parlant.core.common import DefaultBaseModel
from parlant.core.glossary import TermUpdateParams, GlossaryStore, TermId


class CreateTermRequest(DefaultBaseModel):
    name: str
    description: str
    synonyms: Optional[list[str]] = []


class TermDTO(DefaultBaseModel):
    id: TermId
    name: str
    description: str
    synonyms: list[str] = []


class CreateTermResponse(DefaultBaseModel):
    term: TermDTO


class ListTermsResponse(DefaultBaseModel):
    terms: list[TermDTO]


class PatchTermRequest(DefaultBaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    synonyms: Optional[list[str]] = None


class DeleteTermResponse(DefaultBaseModel):
    term_id: TermId


def create_router(
    glossary_store: GlossaryStore,
) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/{agent_id}/terms",
        status_code=status.HTTP_201_CREATED,
        operation_id="create_term",
    )
    async def create_term(agent_id: AgentId, request: CreateTermRequest) -> CreateTermResponse:
        term = await glossary_store.create_term(
            term_set=agent_id,
            name=request.name,
            description=request.description,
            synonyms=request.synonyms,
        )

        return CreateTermResponse(
            term=TermDTO(
                id=term.id,
                name=term.name,
                description=term.description,
                synonyms=term.synonyms,
            )
        )

    @router.get(
        "/{agent_id}/terms/{term_id}",
        operation_id="read_term",
    )
    async def read_term(agent_id: AgentId, term_id: TermId) -> TermDTO:
        term = await glossary_store.read_term(term_set=agent_id, term_id=term_id)

        return TermDTO(
            id=term.id,
            name=term.name,
            description=term.description,
            synonyms=term.synonyms,
        )

    @router.get(
        "/{agent_id}/terms",
        operation_id="list_terms",
    )
    async def list_terms(agent_id: str) -> ListTermsResponse:
        terms = await glossary_store.list_terms(term_set=agent_id)

        return ListTermsResponse(
            terms=[
                TermDTO(
                    id=term.id,
                    name=term.name,
                    description=term.description,
                    synonyms=term.synonyms,
                )
                for term in terms
            ]
        )

    @router.patch(
        "/{agent_id}/terms/{term_id}",
        operation_id="patch_term",
    )
    async def patch_term(agent_id: AgentId, term_id: TermId, request: PatchTermRequest) -> TermDTO:
        params: TermUpdateParams = {}
        if request.name:
            params["name"] = request.name
        if request.description:
            params["description"] = request.description
        if request.synonyms:
            params["synonyms"] = request.synonyms

        term = await glossary_store.update_term(
            term_set=agent_id,
            term_id=term_id,
            params=params,
        )

        return TermDTO(
            id=term.id,
            name=term.name,
            description=term.description,
            synonyms=term.synonyms,
        )

    @router.delete(
        "/{agent_id}/terms/{term_id}",
        operation_id="delete_term",
    )
    async def delete_term(agent_id: str, term_id: TermId) -> DeleteTermResponse:
        deleted_term_id = await glossary_store.delete_term(term_set=agent_id, term_id=term_id)
        return DeleteTermResponse(term_id=deleted_term_id)

    return router
