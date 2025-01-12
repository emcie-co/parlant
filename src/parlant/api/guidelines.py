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

from collections import defaultdict
from dataclasses import dataclass
from fastapi import APIRouter, HTTPException, Path, status
from itertools import chain
from pydantic import Field
from typing import Annotated, Optional, Sequence, TypeAlias

from parlant.api import agents, common
from parlant.api.common import (
    ExampleJson,
    GuidelineActionField,
    GuidelineConditionField,
    InvoiceDataDTO,
    PayloadKindDTO,
    ToolIdDTO,
    apigen_config,
)
from parlant.api.index import InvoiceDTO
from parlant.core.application import Application
from parlant.core.common import (
    DefaultBaseModel,
    GuidelineContent,
    GuidelineId,
)
from parlant.core.evaluations import (
    CoherenceCheck,
    ConnectionProposition,
    GuidelinePayload,
    Invoice,
    InvoiceGuidelineData,
    PayloadKind,
)
from parlant.core.guideline_connections import (
    GuidelineConnectionId,
    GuidelineConnectionStore,
)
from parlant.core.guideline_tool_associations import (
    GuidelineToolAssociationId,
    GuidelineToolAssociationStore,
)
from parlant.core.guidelines import Guideline, GuidelineStore
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.tools import ToolId

API_GROUP = "guidelines"


guideline_dto_example: ExampleJson = {
    "id": "guid_123xz",
    "condition": "when the customer asks about pricing",
    "action": "provide current pricing information and mention any ongoing promotions",
}


GuidelineIdPath: TypeAlias = Annotated[
    GuidelineId,
    Path(
        description="Unique identifier for the guideline",
        examples=["IUCGT-l4pS"],
    ),
]


class GuidelineDTO(
    DefaultBaseModel,
    json_schema_extra={"example": guideline_dto_example},
):
    """Assigns an id to the condition-action pair"""

    id: GuidelineIdPath
    condition: GuidelineConditionField
    action: GuidelineActionField


GuidelineConnectionIdField: TypeAlias = Annotated[
    GuidelineConnectionId,
    Field(
        description="Unique identifier for the `GuildelineConnection`",
    ),
]

GuidelineConnectionIndirectField: TypeAlias = Annotated[
    bool,
    Field(
        description="`True` if there is a path from `source` to `target` but no direct connection",
        examples=[True, False],
    ),
]
guideline_connection_dto_example: ExampleJson = {
    "id": "conn_456xyz",
    "source": {
        "id": "guid_123xz",
        "condition": "when the customer asks about pricing",
        "action": "provide current pricing information",
    },
    "target": {
        "id": "guid_789yz",
        "condition": "when providing pricing information",
        "action": "mention any seasonal discounts",
    },
    "indirect": False,
}


class GuidelineConnectionDTO(
    DefaultBaseModel,
    json_schema_extra={"example": guideline_connection_dto_example},
):
    """
    Represents a connection between two guidelines.

    """

    id: GuidelineConnectionIdField
    source: GuidelineDTO
    target: GuidelineDTO
    indirect: GuidelineConnectionIndirectField


GuidelineToolAssociationIdField: TypeAlias = Annotated[
    GuidelineToolAssociationId,
    Field(
        description="Unique identifier for the association between a tool and a guideline",
        examples=["guid_tool_1"],
    ),
]


guideline_tool_association_example: ExampleJson = {
    "id": "gta_101xyz",
    "guideline_id": "guid_123xz",
    "tool_id": {"service_name": "pricing_service", "tool_name": "get_prices"},
}


class GuidelineToolAssociationDTO(
    DefaultBaseModel,
    json_schema_extra={"example": guideline_tool_association_example},
):
    """
    Represents an association between a Guideline and a Tool, enabling automatic tool invocation
    when the Guideline's conditions are met.
    """

    id: GuidelineToolAssociationIdField
    guideline_id: GuidelineIdPath
    tool_id: ToolIdDTO


guideline_with_connections_example: ExampleJson = {
    "guideline": {
        "id": "guid_123xz",
        "condition": "when the customer asks about pricing",
        "action": "provide current pricing information",
    },
    "connections": [
        {
            "id": "conn_456yz",
            "source": {
                "id": "guid_123xz",
                "condition": "when the customer asks about pricing",
                "action": "provide current pricing information",
            },
            "target": {
                "id": "guid_789yz",
                "condition": "when providing pricing information",
                "action": "mention any seasonal discounts",
            },
            "indirect": False,
        }
    ],
    "tool_associations": [
        {
            "id": "gta_101xyz",
            "guideline_id": "guid_123xz",
            "tool_id": {"service_name": "pricing_service", "tool_name": "get_prices"},
        }
    ],
}


class GuidelineWithConnectionsAndToolAssociationsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": guideline_with_connections_example},
):
    """A Guideline with its connections and tool associations."""

    guideline: GuidelineDTO
    connections: Sequence[GuidelineConnectionDTO]
    tool_associations: Sequence[GuidelineToolAssociationDTO]


guideline_creation_params_example: ExampleJson = {
    "invoices": [
        {
            "payload": {
                "kind": "guideline",
                "guideline": {
                    "content": {
                        "condition": "when the customer asks about pricing",
                        "action": "provide current pricing information",
                    },
                    "operation": "add",
                    "coherence_check": True,
                    "connection_proposition": True,
                },
            },
            "data": {"guideline": {"coherence_checks": [], "connection_propositions": []}},
            "approved": True,
            "checksum": "abc123",
            "error": None,
        }
    ]
}


class GuidelineCreationParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": guideline_creation_params_example},
):
    """Evaluation invoices to generate Guidelines from."""

    invoices: Sequence[InvoiceDTO]


guideline_creation_result_example: ExampleJson = {
    "items": [
        {
            "guideline": {
                "id": "guid_123xz",
                "condition": "when the customer asks about pricing",
                "action": "provide current pricing information",
            },
            "connections": [],
            "tool_associations": [],
        }
    ]
}


class GuidelineCreationResult(
    DefaultBaseModel,
    json_schema_extra={"example": guideline_creation_result_example},
):
    """Result wrapper for Guidelines creation."""

    items: Sequence[GuidelineWithConnectionsAndToolAssociationsDTO]


GuidelineConnectionAdditionSourceField: TypeAlias = Annotated[
    GuidelineId,
    Field(description="`id` of guideline that is source of this connection."),
]

GuidelineConnectionAdditionTargetField: TypeAlias = Annotated[
    GuidelineId,
    Field(description="`id` of guideline that is target of this connection."),
]


guideline_connection_addition_example: ExampleJson = {
    "source": "guid_123xz",
    "target": "guid_789yz",
}


class GuidelineConnectionAdditionDTO(
    DefaultBaseModel,
    json_schema_extra={"example": guideline_connection_addition_example},
):
    """Used to add connections between Guidelines."""

    source: GuidelineConnectionAdditionSourceField
    target: GuidelineConnectionAdditionTargetField


guideline_connection_update_params_example: ExampleJson = {
    "add": [{"source": "guide_123xyz", "target": "guide_789xyz"}],
    "remove": ["guide_456xyz"],
}


class GuidelineConnectionUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": guideline_connection_update_params_example},
):
    """
    Parameters for updaing a guideline connection.

    `add` is expected to be a collection of addition objects.
    `remove` should contain the `id`s of the guidelines to remove.
    """

    add: Optional[Sequence[GuidelineConnectionAdditionDTO]] = None
    remove: Optional[Sequence[GuidelineIdPath]] = None


guideline_tool_association_update_params_example: ExampleJson = {
    "add": [{"service_name": "pricing_service", "tool_name": "get_prices"}],
    "remove": [{"service_name": "old_service", "tool_name": "old_tool"}],
}


class GuidelineToolAssociationUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": guideline_tool_association_update_params_example},
):
    """Parameters for adding/removing tool associations."""

    add: Optional[Sequence[ToolIdDTO]] = None
    remove: Optional[Sequence[ToolIdDTO]] = None


guideline_update_params_example: ExampleJson = {
    "connections": {
        "add": [{"source": "guide_123xyz", "target": "guide_789xyz"}],
        "remove": ["guide_456xyz"],
    },
    "tool_associations": {
        "add": [{"service_name": "pricing_service", "tool_name": "get_prices"}],
        "remove": [{"service_name": "old_service", "tool_name": "old_tool"}],
    },
}


class GuidelineUpdateParamsDTO(
    DefaultBaseModel, json_schema_extra={"example": guideline_update_params_example}
):
    """Parameters for updating Guideline objects."""

    connections: Optional[GuidelineConnectionUpdateParamsDTO] = None
    tool_associations: Optional[GuidelineToolAssociationUpdateParamsDTO] = None


@dataclass
class _GuidelineConnection:
    """Represents one connection between two Guidelines."""

    id: GuidelineConnectionId
    source: Guideline
    target: Guideline


def _invoice_dto_to_invoice(dto: InvoiceDTO) -> Invoice:
    if dto.payload.kind != PayloadKindDTO.GUIDELINE:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only guideline invoices are supported here",
        )

    if not dto.approved:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unapproved invoice",
        )

    if not dto.payload.guideline:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Missing guideline payload",
        )

    payload = GuidelinePayload(
        content=GuidelineContent(
            condition=dto.payload.guideline.content.condition,
            action=dto.payload.guideline.content.action,
        ),
        operation=dto.payload.guideline.operation.value,
        coherence_check=dto.payload.guideline.coherence_check,
        connection_proposition=dto.payload.guideline.connection_proposition,
        updated_id=dto.payload.guideline.updated_id,
    )

    kind = PayloadKind.GUIDELINE

    if not dto.data:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Missing invoice data",
        )

    data = _invoice_data_dto_to_invoice_data(dto.data)

    return Invoice(
        kind=kind,
        payload=payload,
        checksum=dto.checksum,
        state_version="",  # FIXME: once state functionality will be implemented this need to be refactored
        approved=dto.approved,
        data=data,
        error=dto.error,
    )


def _invoice_data_dto_to_invoice_data(dto: InvoiceDataDTO) -> InvoiceGuidelineData:
    if not dto.guideline:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Missing guideline invoice data",
        )

    try:
        coherence_checks = [
            CoherenceCheck(
                kind=check.kind.value,
                first=GuidelineContent(condition=check.first.condition, action=check.first.action),
                second=GuidelineContent(
                    condition=check.second.condition, action=check.second.action
                ),
                issue=check.issue,
                severity=check.severity,
            )
            for check in dto.guideline.coherence_checks
        ]

        if dto.guideline.connection_propositions:
            connection_propositions = [
                ConnectionProposition(
                    check_kind=prop.check_kind.value,
                    source=GuidelineContent(
                        condition=prop.source.condition, action=prop.source.action
                    ),
                    target=GuidelineContent(
                        condition=prop.target.condition, action=prop.target.action
                    ),
                )
                for prop in dto.guideline.connection_propositions
            ]
        else:
            connection_propositions = None

        return InvoiceGuidelineData(
            coherence_checks=coherence_checks, connection_propositions=connection_propositions
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid invoice guideline data",
        )


guideline_example = {
    "id": "guid_123xz",
    "condition": "when the customer asks about pricing",
    "action": "provide current pricing information and mention any ongoing promotions",
}


def create_router(
    application: Application,
    guideline_store: GuidelineStore,
    guideline_connection_store: GuidelineConnectionStore,
    service_registry: ServiceRegistry,
    guideline_tool_association_store: GuidelineToolAssociationStore,
) -> APIRouter:
    router = APIRouter()

    async def get_guideline_connections(
        guideline_set: str,
        guideline_id: GuidelineId,
        include_indirect: bool = True,
    ) -> Sequence[tuple[_GuidelineConnection, bool]]:
        connections = [
            _GuidelineConnection(
                id=c.id,
                source=await guideline_store.read_guideline(
                    guideline_set=guideline_set, guideline_id=c.source
                ),
                target=await guideline_store.read_guideline(
                    guideline_set=guideline_set, guideline_id=c.target
                ),
            )
            for c in chain(
                await guideline_connection_store.list_connections(
                    indirect=include_indirect, source=guideline_id
                ),
                await guideline_connection_store.list_connections(
                    indirect=include_indirect, target=guideline_id
                ),
            )
        ]

        return [(c, guideline_id not in [c.source.id, c.target.id]) for c in connections]

    @router.post(
        "/{agent_id}/guidelines",
        status_code=status.HTTP_201_CREATED,
        operation_id="create_guidelines",
        response_model=GuidelineCreationResult,
        responses={
            status.HTTP_201_CREATED: {
                "description": "Guidelines successfully created. Returns the created guidelines with their connections and tool associations.",
                "content": common.example_json_content(guideline_creation_result_example),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Agent not found"},
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
                "description": "Validation error in request parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="create"),
    )
    async def create_guidelines(
        agent_id: agents.AgentIdPath,
        params: GuidelineCreationParamsDTO,
    ) -> GuidelineCreationResult:
        """
        Creates new guidelines from the provided invoices.

        Invoices are obtained by calling the `create_evaluation` method of the client.
        (Equivalent to making a POST request to `/index/evaluations`)
        See the [documentation](https://parlant.io/docs/concepts/customization/guidelines) for more information.

        The guidelines are created in the specified agent's guideline set.
        Tool associations and connections are automatically handled.
        """
        invoices = [_invoice_dto_to_invoice(i) for i in params.invoices]

        guideline_ids = set(
            await application.create_guidelines(
                guideline_set=agent_id,
                invoices=invoices,
            )
        )

        guidelines = [
            await guideline_store.read_guideline(guideline_set=agent_id, guideline_id=id)
            for id in guideline_ids
        ]

        tool_associations = defaultdict(list)
        for association in await guideline_tool_association_store.list_associations():
            if association.guideline_id in guideline_ids:
                tool_associations[association.guideline_id].append(
                    GuidelineToolAssociationDTO(
                        id=association.id,
                        guideline_id=association.guideline_id,
                        tool_id=ToolIdDTO(
                            service_name=association.tool_id.service_name,
                            tool_name=association.tool_id.tool_name,
                        ),
                    )
                )

        return GuidelineCreationResult(
            items=[
                GuidelineWithConnectionsAndToolAssociationsDTO(
                    guideline=GuidelineDTO(
                        id=guideline.id,
                        condition=guideline.content.condition,
                        action=guideline.content.action,
                    ),
                    connections=[
                        GuidelineConnectionDTO(
                            id=connection.id,
                            source=GuidelineDTO(
                                id=connection.source.id,
                                condition=connection.source.content.condition,
                                action=connection.source.content.action,
                            ),
                            target=GuidelineDTO(
                                id=connection.target.id,
                                condition=connection.target.content.condition,
                                action=connection.target.content.action,
                            ),
                            indirect=indirect,
                        )
                        for connection, indirect in await get_guideline_connections(
                            guideline_set=agent_id,
                            guideline_id=guideline.id,
                            include_indirect=True,
                        )
                    ],
                    tool_associations=tool_associations.get(guideline.id, []),
                )
                for guideline in guidelines
            ]
        )

    @router.get(
        "/{agent_id}/guidelines/{guideline_id}",
        operation_id="read_guideline",
        response_model=GuidelineWithConnectionsAndToolAssociationsDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Guideline details successfully retrieved. Returns the complete guideline with its connections and tool associations.",
                "content": common.example_json_content(guideline_with_connections_example),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Guideline or agent not found"},
        },
        **apigen_config(group_name=API_GROUP, method_name="retrieve"),
    )
    async def read_guideline(
        agent_id: agents.AgentIdPath,
        guideline_id: GuidelineIdPath,
    ) -> GuidelineWithConnectionsAndToolAssociationsDTO:
        """
        Retrieves a specific guideline with all its connections and tool associations.

        Returns both direct and indirect connections between guidelines.
        Tool associations indicate which tools the guideline can use.
        """

        guideline = await guideline_store.read_guideline(
            guideline_set=agent_id, guideline_id=guideline_id
        )

        connections = await get_guideline_connections(
            guideline_set=agent_id,
            guideline_id=guideline_id,
            include_indirect=True,
        )

        return GuidelineWithConnectionsAndToolAssociationsDTO(
            guideline=GuidelineDTO(
                id=guideline.id,
                condition=guideline.content.condition,
                action=guideline.content.action,
            ),
            connections=[
                GuidelineConnectionDTO(
                    id=connection.id,
                    source=GuidelineDTO(
                        id=connection.source.id,
                        condition=connection.source.content.condition,
                        action=connection.source.content.action,
                    ),
                    target=GuidelineDTO(
                        id=connection.target.id,
                        condition=connection.target.content.condition,
                        action=connection.target.content.action,
                    ),
                    indirect=indirect,
                )
                for connection, indirect in connections
            ],
            tool_associations=[
                GuidelineToolAssociationDTO(
                    id=a.id,
                    guideline_id=a.guideline_id,
                    tool_id=ToolIdDTO(
                        service_name=a.tool_id.service_name,
                        tool_name=a.tool_id.tool_name,
                    ),
                )
                for a in await guideline_tool_association_store.list_associations()
                if a.guideline_id == guideline_id
            ],
        )

    @router.get(
        "/{agent_id}/guidelines",
        operation_id="list_guidelines",
        response_model=Sequence[GuidelineDTO],
        responses={
            status.HTTP_200_OK: {
                "description": "List of all guidelines for the specified agent",
                "content": common.example_json_content([guideline_dto_example]),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Agent not found"},
        },
        **apigen_config(group_name=API_GROUP, method_name="list"),
    )
    async def list_guidelines(
        agent_id: agents.AgentIdPath,
    ) -> Sequence[GuidelineDTO]:
        """
        Lists all guidelines for the specified agent.

        Returns an empty list if no guidelines exist.
        Guidelines are returned in no guaranteed order.
        Does not include connections or tool associations.
        """
        guidelines = await guideline_store.list_guidelines(guideline_set=agent_id)

        return [
            GuidelineDTO(
                id=guideline.id,
                condition=guideline.content.condition,
                action=guideline.content.action,
            )
            for guideline in guidelines
        ]

    @router.patch(
        "/{agent_id}/guidelines/{guideline_id}",
        operation_id="update_guideline",
        response_model=GuidelineWithConnectionsAndToolAssociationsDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Guideline successfully updated. Returns the updated guideline with its connections and tool associations.",
                "content": common.example_json_content(guideline_with_connections_example),
            },
            status.HTTP_404_NOT_FOUND: {
                "description": "Guideline, agent, or referenced tool not found"
            },
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
                "description": "Invalid connection rules or validation error in update parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="update"),
    )
    async def update_guideline(
        agent_id: agents.AgentIdPath,
        guideline_id: GuidelineIdPath,
        params: GuidelineUpdateParamsDTO,
    ) -> GuidelineWithConnectionsAndToolAssociationsDTO:
        """Updates a guideline's connections and tool associations.

        Only provided attributes will be updated; others remain unchanged.

        Connection rules:
        - A guideline cannot connect to itself
        - Only direct connections can be removed
        - The connection must specify this guideline as source or target

        Tool Association rules:
        - Tool services and tools must exist before creating associations
        """
        guideline = await guideline_store.read_guideline(
            guideline_set=agent_id,
            guideline_id=guideline_id,
        )

        if params.connections and params.connections.add:
            for req in params.connections.add:
                if req.source == req.target:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail="A guideline cannot be connected to itself",
                    )
                elif req.source == guideline.id:
                    _ = await guideline_store.read_guideline(
                        guideline_set=agent_id,
                        guideline_id=req.target,
                    )
                elif req.target == guideline.id:
                    _ = await guideline_store.read_guideline(
                        guideline_set=agent_id,
                        guideline_id=req.source,
                    )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="The connection must specify the guideline at hand as either source or target",
                    )

                await guideline_connection_store.create_connection(
                    source=req.source,
                    target=req.target,
                )

        connections = await get_guideline_connections(
            agent_id,
            guideline_id,
            include_indirect=False,
        )

        if params.connections and params.connections.remove:
            for id in params.connections.remove:
                if found_connection := next(
                    (c for c, _ in connections if id in [c.source.id, c.target.id]), None
                ):
                    await guideline_connection_store.delete_connection(found_connection.id)
                else:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail="Only direct connections may be removed",
                    )

        if params.tool_associations and params.tool_associations.add:
            for tool_id_dto in params.tool_associations.add:
                service_name = tool_id_dto.service_name
                tool_name = tool_id_dto.tool_name

                service = await service_registry.read_tool_service(service_name)
                _ = await service.read_tool(tool_name)

                await guideline_tool_association_store.create_association(
                    guideline_id=guideline_id,
                    tool_id=ToolId(service_name=service_name, tool_name=tool_name),
                )

        if params.tool_associations and params.tool_associations.remove:
            associations = await guideline_tool_association_store.list_associations()

            for tool_id_dto in params.tool_associations.remove:
                if association := next(
                    (
                        assoc
                        for assoc in associations
                        if assoc.tool_id.service_name == tool_id_dto.service_name
                        and assoc.tool_id.tool_name == tool_id_dto.tool_name
                    ),
                    None,
                ):
                    await guideline_tool_association_store.delete_association(association.id)
                else:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Tool association not found for service '{tool_id_dto.service_name}' and tool '{tool_id_dto.tool_name}'",
                    )

        return GuidelineWithConnectionsAndToolAssociationsDTO(
            guideline=GuidelineDTO(
                id=guideline.id,
                condition=guideline.content.condition,
                action=guideline.content.action,
            ),
            connections=[
                GuidelineConnectionDTO(
                    id=connection.id,
                    source=GuidelineDTO(
                        id=connection.source.id,
                        condition=connection.source.content.condition,
                        action=connection.source.content.action,
                    ),
                    target=GuidelineDTO(
                        id=connection.target.id,
                        condition=connection.target.content.condition,
                        action=connection.target.content.action,
                    ),
                    indirect=indirect,
                )
                for connection, indirect in await get_guideline_connections(
                    agent_id, guideline_id, True
                )
            ],
            tool_associations=[
                GuidelineToolAssociationDTO(
                    id=a.id,
                    guideline_id=a.guideline_id,
                    tool_id=ToolIdDTO(
                        service_name=a.tool_id.service_name,
                        tool_name=a.tool_id.tool_name,
                    ),
                )
                for a in await guideline_tool_association_store.list_associations()
                if a.guideline_id == guideline_id
            ],
        )

    @router.delete(
        "/{agent_id}/guidelines/{guideline_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        operation_id="delete_guideline",
        responses={
            status.HTTP_204_NO_CONTENT: {
                "description": "Guideline successfully deleted. No content returned."
            },
            status.HTTP_404_NOT_FOUND: {"description": "Guideline or agent not found"},
        },
        **apigen_config(group_name=API_GROUP, method_name="delete"),
    )
    async def delete_guideline(
        agent_id: agents.AgentIdPath,
        guideline_id: GuidelineIdPath,
    ) -> None:
        """Deletes a guideline from the agent.

        Also removes all associated connections and tool associations.
        Deleting a non-existent guideline will return 404.
        No content will be returned from a successful deletion.
        """

        await guideline_store.read_guideline(
            guideline_set=agent_id,
            guideline_id=guideline_id,
        )

        await guideline_store.delete_guideline(
            guideline_set=agent_id,
            guideline_id=guideline_id,
        )

        for c in chain(
            await guideline_connection_store.list_connections(indirect=False, source=guideline_id),
            await guideline_connection_store.list_connections(indirect=False, target=guideline_id),
        ):
            await guideline_connection_store.delete_connection(c.id)

        for associastion in await guideline_tool_association_store.list_associations():
            if associastion.guideline_id == guideline_id:
                await guideline_tool_association_store.delete_association(associastion.id)

    return router
