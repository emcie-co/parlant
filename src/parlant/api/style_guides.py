# api/style_guides.py
# Copyright 2024 Emcie Co Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Annotated, Sequence, TypeAlias, cast
from fastapi import APIRouter, HTTPException, Path, status

from parlant.api import agents
from parlant.api.common import (
    ExampleJson,
    InvoiceDataDTO,
    PayloadKindDTO,
    StyleGuideContentDTO,
    apigen_config,
    InvoiceDTO,
    example_json_content,
    style_guide_content_dto_to_content,
    style_guide_content_to_dto,
)
from parlant.core.common import DefaultBaseModel
from parlant.core.evaluations import (
    Invoice,
    PayloadKind,
    StyleGuideCoherenceCheck,
    StyleGuideInvoiceData,
    StyleGuidePayload,
)
from parlant.core.style_guides import StyleGuideStore, StyleGuideId

style_guide_dto_example: ExampleJson = {
    "id": "sg_123xyz",
    "principle": "Adopt a whimsical tone and include jokes or playful remarks.",
    "examples": [
        {
            "before": [
                {
                    "source": "customer",
                    "message": "Can you tell me the weather for today?",
                },
                {
                    "source": "ai_agent",
                    "message": "It's sunny and warm, about 75°F.",
                },
            ],
            "after": [
                {
                    "source": "customer",
                    "message": "Can you tell me the weather for today?",
                },
                {
                    "source": "ai_agent",
                    "message": "It's sunny and warm, about 75°F. Perfect weather for pretending you’re on a tropical island... even if you're just in your backyard with a kiddie pool!",
                },
            ],
            "violation": "The 'before' message does not include a joke or playful remark.",
        },
    ],
}

StyleGuideIdPath: TypeAlias = Annotated[
    StyleGuideId,
    Path(
        description="Unique identifier for the style guide",
        examples=["sg_abc123"],
    ),
]


class StyleGuideDTO(
    DefaultBaseModel,
    json_schema_extra={"example": style_guide_dto_example},
):
    """
    Assigns an id to the style-guide content
    """

    id: StyleGuideIdPath
    content: StyleGuideContentDTO


style_guide_creation_params_example: ExampleJson = {
    "invoices": [
        {
            "payload": {
                "kind": "style_guide",
                "style_guide": {
                    "principle": "Adopt a whimsical tone and include jokes or playful remarks.",
                    "examples": [
                        {
                            "before": [
                                {
                                    "source": "customer",
                                    "message": "Can you tell me the weather for today?",
                                },
                                {
                                    "source": "ai_agent",
                                    "message": "It's sunny and warm, about 75°F.",
                                },
                            ],
                            "after": [
                                {
                                    "source": "customer",
                                    "message": "Can you tell me the weather for today?",
                                },
                                {
                                    "source": "ai_agent",
                                    "message": "It's sunny and warm, about 75°F. Perfect weather for pretending you're on a tropical island... even if you're just in your backyard with a kiddie pool!",
                                },
                            ],
                            "violation": "The 'before' message does not include a joke or playful remark.",
                        },
                    ],
                },
            },
            "data": {
                "style_guide": {
                    "coherence_checks": [],
                }
            },
            "approved": True,
            "checksum": "abc123",
            "error": None,
        }
    ]
}


class StyleGuideCreationParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": style_guide_creation_params_example},
):
    """Evaluation invoices to generate StyleGuides from."""

    invoices: Sequence[InvoiceDTO]


style_guide_creation_result_example: ExampleJson = {
    "items": [
        {
            "id": "sg_123xyz",
            "principle": "Be friendly, casual, and helpful",
            "examples": [
                {
                    "before": [
                        {
                            "source": "ai_agent",
                            "message": "Your request is denied.",
                        }
                    ],
                    "after": [
                        {
                            "source": "ai_agent",
                            "message": "We can’t do that now, but let's see what else might help!",
                        }
                    ],
                    "violation": "The 'before' message is abrupt and lacking empathy.",
                }
            ],
        }
    ]
}


class StyleGuideCreationResultDTO(
    DefaultBaseModel,
    json_schema_extra={"example": style_guide_creation_result_example},
):
    """Result wrapper for StyleGuides creation."""

    items: Sequence[StyleGuideDTO]


def _invoice_data_dto_to_invoice_data(dto: InvoiceDataDTO) -> StyleGuideInvoiceData:
    if not dto.style_guide:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Missing guideline invoice data",
        )

    try:
        coherence_checks = [
            StyleGuideCoherenceCheck(
                kind=check.kind.value,
                first=style_guide_content_dto_to_content(check.first),
                second=style_guide_content_dto_to_content(check.second),
                issue=check.issue,
                severity=check.severity,
            )
            for check in dto.style_guide.coherence_checks
        ]

        return StyleGuideInvoiceData(coherence_checks=coherence_checks)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid invoice style guide data",
        )


def _invoice_dto_to_invoice(dto: InvoiceDTO) -> Invoice:
    if dto.payload.kind != PayloadKindDTO.STYLE_GUIDE:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only style guide invoices are supported here",
        )

    if not dto.approved:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unapproved invoice",
        )

    if not dto.payload.style_guide:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Missing style guide payload",
        )

    if not dto.data:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Missing invoice data",
        )

    style_guide_payload = StyleGuidePayload(
        content=style_guide_content_dto_to_content(dto.payload.style_guide.content),
        operation=dto.payload.style_guide.operation.value,
        updated_id=dto.payload.style_guide.updated_id,
        coherence_check=dto.payload.style_guide.coherence_check,
    )

    data = _invoice_data_dto_to_invoice_data(dto.data)

    return Invoice(
        kind=PayloadKind.STYLE_GUIDE,
        payload=style_guide_payload,
        checksum=dto.checksum,
        state_version="",  # FIXME: once state functionality will be implemented this need to be refactored
        approved=dto.approved,
        data=data,
        error=dto.error,
    )


def create_router(
    style_guide_store: StyleGuideStore,
) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/{agent_id}/style_guides",
        status_code=status.HTTP_201_CREATED,
        operation_id="create_style_guides",
        response_model=StyleGuideCreationResultDTO,
        responses={
            status.HTTP_201_CREATED: {
                "description": "Style guides successfully created. Returns the created style guides.",
                "content": example_json_content(style_guide_creation_result_example),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Agent not found"},
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
                "description": "Validation error in request parameters",
            },
        },
        **apigen_config(group_name="style_guides", method_name="create"),
    )
    async def create_style_guides(
        agent_id: agents.AgentIdPath,
        params: StyleGuideCreationParamsDTO,
    ) -> StyleGuideCreationResultDTO:
        """
        Creates new style guides from the provided invoices.

        Invoices are obtained by calling the `create_evaluation` method of the client.
        (Equivalent to making a POST request to `/index/evaluations`)
        See the [documentation](https://parlant.io/docs/concepts/customization/style-guides) for more information.
        """
        invoices = [_invoice_dto_to_invoice(i) for i in params.invoices]

        style_guides = []

        for invoice in invoices:
            payload = cast(StyleGuidePayload, invoice.payload)
            if invoice.payload.operation == "add":
                style_guides.append(
                    await style_guide_store.create_style_guide(
                        style_guide_set=agent_id,
                        principle=payload.content.principle,
                        examples=payload.content.examples,
                    )
                )
            else:
                style_guides.append(
                    await style_guide_store.update_style_guide(
                        style_guide_set=agent_id,
                        style_guide_id=cast(StyleGuideId, payload.updated_id),
                        params={
                            "principle": payload.content.principle,
                            "examples": payload.content.examples,
                        },
                    )
                )

        return StyleGuideCreationResultDTO(
            items=[
                StyleGuideDTO(
                    id=style_guide.id, content=style_guide_content_to_dto(style_guide.content)
                )
                for style_guide in style_guides
            ]
        )

    @router.get(
        "/{agent_id}/style_guides/{style_guide_id}",
        operation_id="read_style_guide",
        response_model=StyleGuideDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Style guide details successfully retrieved.",
                "content": example_json_content(style_guide_dto_example),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Style guide or agent not found"},
        },
        **apigen_config(group_name="style_guides", method_name="retrieve"),
    )
    async def read_style_guide(
        agent_id: agents.AgentIdPath,
        style_guide_id: StyleGuideIdPath,
    ) -> StyleGuideDTO:
        """
        Retrieves a style guide by its ID.
        """
        style_guide = await style_guide_store.read_style_guide(
            style_guide_set=agent_id,
            style_guide_id=style_guide_id,
        )

        return StyleGuideDTO(
            id=style_guide.id,
            content=style_guide_content_to_dto(style_guide.content),
        )

    @router.get(
        "/{agent_id}/style_guides",
        operation_id="list_style_guides",
        response_model=Sequence[StyleGuideDTO],
        responses={
            status.HTTP_200_OK: {
                "description": "List of all style guides for the specified agent.",
                "content": example_json_content([style_guide_dto_example]),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Agent not found"},
        },
        **apigen_config(group_name="style_guides", method_name="list"),
    )
    async def list_style_guides(
        agent_id: agents.AgentIdPath,
    ) -> Sequence[StyleGuideDTO]:
        """
        Lists all style guides for the specified agent (style_guide_set).
        Returns an empty list if none exist.
        """
        style_guides = await style_guide_store.list_style_guides(style_guide_set=agent_id)

        return [
            StyleGuideDTO(
                id=style_guide.id,
                content=style_guide_content_to_dto(style_guide.content),
            )
            for style_guide in style_guides
        ]

    @router.patch(
        "/{agent_id}/style_guides/{style_guide_id}",
        operation_id="update_style_guide",
        response_model=StyleGuideDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Style guide successfully updated. Returns the updated guide.",
                "content": example_json_content(style_guide_dto_example),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Style guide or agent not found"},
            status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        },
        **apigen_config(group_name="style_guides", method_name="update"),
    )
    @router.delete(
        "/{agent_id}/style_guides/{style_guide_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        operation_id="delete_style_guide",
        responses={
            status.HTTP_204_NO_CONTENT: {
                "description": "Style guide successfully deleted. No content returned."
            },
            status.HTTP_404_NOT_FOUND: {"description": "Style guide or agent not found"},
        },
        **apigen_config(group_name="style_guides", method_name="delete"),
    )
    async def delete_style_guide(
        agent_id: agents.AgentIdPath,
        style_guide_id: StyleGuideIdPath,
    ) -> None:
        """Deletes a style guide from the agent.

        Deleting a non-existent style guide will return 404.
        No content will be returned from a successful deletion.
        """
        await style_guide_store.read_style_guide(agent_id, style_guide_id)

        await style_guide_store.delete_style_guide(agent_id, style_guide_id)

    return router
